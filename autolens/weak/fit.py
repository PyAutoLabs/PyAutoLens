"""
Weak-lensing fit class.

``FitWeak`` compares a model shear field (derived from a ``Tracer``'s mass profiles via
``LensCalc.shear_yx_2d_via_hessian_from`` — the same primitive ``SimulatorShearYX`` uses) against an observed
``WeakDataset`` and reports per-galaxy residuals, chi-squared and the log-likelihood. It is the weak-lensing
analogue of :class:`autolens.imaging.fit_imaging.FitImaging` and the input to ``AnalysisWeak``.

Each background source galaxy contributes **two** independent measurements (:math:`\\gamma_1` and
:math:`\\gamma_2` carry the same per-galaxy noise but are independent Gaussian draws), so the chi-squared sum
and ``noise_normalization`` count :math:`N \\times 2` elements rather than just :math:`N`.

The model quantity adapts to what the dataset declares:

- ``dataset.is_reduced`` — the model is the *reduced* shear :math:`g = \\gamma / (1 - \\kappa)`, the quantity
  real surveys measure from galaxy ellipticities.
- ``dataset.redshifts`` — the model signal is scaled per galaxy by the lensing-efficiency ratio
  :math:`\\beta_i / \\beta_{\\rm ref}` (``LensingCosmology.scaling_factor_between_redshifts_from``), so a
  catalogue spanning a range of source redshifts is fitted self-consistently. The reference plane is the
  tracer's outermost (source) plane; galaxies at or below the lens redshift carry zero signal. Without
  redshifts the fit assumes the tracer's single effective source plane — exactly the pre-scaling behaviour.
  The scale factors are computed eagerly from concrete plane redshifts (galaxy redshifts are fixed
  constants, not sampled parameters), which keeps them outside any JAX trace.

The class is deliberately standalone — it does not inherit from ``autoarray.fit.fit_dataset.AbstractFit``,
which is shaped for "data + noise_map + mask" pixel-grid fits. ``FitPoint`` (in ``autolens.point``) follows
the same standalone pattern.

JAX support follows the ``LensCalc`` guard pattern: with ``xp=jnp`` the fit statistics are traceable and
``model_shear`` returns a raw ``(N, 2)`` array (``ShearYX2DIrregular`` is not a registered pytree);
``AnalysisWeak`` registers the ``FitWeak`` pytree so ``jax.jit(fit_from)`` round-trips a real fit object.
"""
import math
from functools import cached_property

import numpy as np

from autogalaxy.operate.lens_calc import LensCalc
from autogalaxy.util.shear_field import ShearYX2DIrregular

from autolens.weak.dataset import WeakDataset


class FitWeak:
    def __init__(self, dataset: WeakDataset, tracer, xp=np):
        """
        Fit a ``Tracer`` lens model to a ``WeakDataset`` shear catalogue.

        Parameters
        ----------
        dataset
            The observed weak-lensing shear catalogue.
        tracer
            The PyAutoLens ``Tracer`` whose mass profiles generate the model shear field.
        xp
            The array module (``numpy`` or ``jax.numpy``). With ``jax.numpy`` the fit statistics are
            traceable and ``model_shear`` returns a raw array rather than a ``ShearYX2DIrregular``.
        """
        self.dataset = dataset
        self.tracer = tracer
        self._xp = xp

    @cached_property
    def _redshift_scale_factors(self):
        """
        Per-galaxy lensing-efficiency ratios ``beta_i / beta_ref``, or ``None`` when the dataset carries no
        redshifts.

        Unity for galaxies at the tracer's source-plane redshift, zero at or below the lens redshift (such
        galaxies are not lensed), and ``LensingCosmology.scaling_factor_between_redshifts_from`` in between —
        the same factor multi-plane ray-tracing applies to deflections. Always a concrete NumPy array:
        plane and catalogue redshifts are fixed constants, so this never participates in a JAX trace.
        """
        redshifts = getattr(self.dataset, "redshifts", None)
        if redshifts is None:
            return None

        plane_redshifts = sorted(
            float(galaxy.redshift) for galaxy in self.tracer.galaxies
        )
        redshift_lens = plane_redshifts[0]
        redshift_ref = plane_redshifts[-1]

        cosmology = self.tracer.cosmology

        factors = [
            0.0
            if float(redshift_i) <= redshift_lens
            else float(
                cosmology.scaling_factor_between_redshifts_from(
                    redshift_0=redshift_lens,
                    redshift_1=float(redshift_i),
                    redshift_final=redshift_ref,
                )
            )
            for redshift_i in np.asarray(redshifts)
        ]
        return np.asarray(factors)

    @cached_property
    def model_shear(self):
        """
        The model signal evaluated at the galaxy positions, via ``LensCalc``.

        This is the (optionally per-galaxy-scaled) shear ``gamma``, or the reduced shear
        ``g = gamma / (1 - kappa)`` when the dataset is marked ``is_reduced`` — data and model always live
        in the same space. On the NumPy path the return is a ``ShearYX2DIrregular``; with ``xp=jax.numpy``
        it is a raw ``(N, 2)`` array (the ``LensCalc`` guard pattern).
        """
        xp = self._xp

        lens_calc = LensCalc.from_tracer(self.tracer)

        shear = lens_calc.shear_yx_2d_via_hessian_from(
            grid=self.dataset.positions, xp=xp
        )

        scale = self._redshift_scale_factors
        is_reduced = getattr(self.dataset, "is_reduced", False)

        if scale is None and not is_reduced:
            return shear

        values = xp.asarray(shear)

        if is_reduced:
            convergence = xp.asarray(
                lens_calc.convergence_2d_via_hessian_from(
                    grid=self.dataset.positions, xp=xp
                )
            )
            if scale is not None:
                values = (scale[:, None] * values) / (
                    1.0 - scale * convergence
                )[:, None]
            else:
                values = values / (1.0 - convergence)[:, None]
        else:
            values = scale[:, None] * values

        if xp is np:
            return ShearYX2DIrregular(
                values=np.asarray(values), grid=self.dataset.positions
            )
        return values

    @property
    def residual_map(self):
        """``(N, 2)`` residuals ``data - model`` for each galaxy's ``(gamma_2, gamma_1)`` components."""
        xp = self._xp
        data = xp.asarray(np.asarray(self.dataset.shear_yx))
        return data - xp.asarray(self.model_shear)

    @property
    def normalized_residual_map(self):
        """``(N, 2)`` residuals divided by the per-galaxy noise broadcast across both shear components."""
        xp = self._xp
        noise = xp.asarray(np.asarray(self.dataset.noise_map))[:, None]
        return self.residual_map / noise

    @property
    def chi_squared_map(self):
        """``(N, 2)`` per-component chi-squared contributions."""
        return self.normalized_residual_map**2

    @property
    def chi_squared(self):
        """Scalar chi-squared summed over all ``N x 2`` shear measurements."""
        xp = self._xp
        chi_squared = xp.sum(self.chi_squared_map)
        return float(chi_squared) if xp is np else chi_squared

    @property
    def noise_normalization(self) -> float:
        r"""
        Gaussian likelihood normalisation :math:`\sum \log(2 \pi \sigma^2)` summed over all ``N x 2`` shear
        measurements — the factor of 2 reflects that each galaxy contributes two independent components.
        Always concrete (it depends only on the dataset).
        """
        noise = np.asarray(self.dataset.noise_map)
        return float(2.0 * np.sum(np.log(2.0 * math.pi * noise**2)))

    @property
    def log_likelihood(self):
        r"""Standard Gaussian log-likelihood :math:`-\tfrac{1}{2}(\chi^2 + \text{noise normalization})`."""
        return -0.5 * (self.chi_squared + self.noise_normalization)

    @property
    def figure_of_merit(self):
        """Quantity returned to non-linear searches; same as ``log_likelihood`` (no inversion / evidence)."""
        return self.log_likelihood
