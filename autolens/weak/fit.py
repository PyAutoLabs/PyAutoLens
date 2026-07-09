"""
Weak-lensing fit class.

``FitWeak`` compares a model shear field (derived from a ``Tracer``'s mass profiles via
``LensCalc.shear_yx_2d_via_hessian_from`` — the same primitive ``SimulatorShearYX`` uses) against an observed
``WeakDataset`` and reports per-galaxy residuals, chi-squared and the log-likelihood. It is the weak-lensing
analogue of :class:`autolens.imaging.fit_imaging.FitImaging` and the input to a future ``AnalysisWeak``.

Each background source galaxy contributes **two** independent measurements (:math:`\\gamma_1` and
:math:`\\gamma_2` carry the same per-galaxy noise but are independent Gaussian draws), so the chi-squared sum
and ``noise_normalization`` count :math:`N \\times 2` elements rather than just :math:`N`.

The class is deliberately standalone — it does not inherit from ``autoarray.fit.fit_dataset.AbstractFit``,
which is shaped for "data + noise_map + mask" pixel-grid fits. ``FitPoint`` (in ``autolens.point``) follows
the same standalone pattern.
"""
import math
from functools import cached_property

import numpy as np

from autogalaxy.operate.lens_calc import LensCalc
from autogalaxy.util.shear_field import ShearYX2DIrregular

from autolens.weak.dataset import WeakDataset


class FitWeak:
    def __init__(self, dataset: WeakDataset, tracer):
        """
        Fit a ``Tracer`` lens model to a ``WeakDataset`` shear catalogue.

        Parameters
        ----------
        dataset
            The observed weak-lensing shear catalogue.
        tracer
            The PyAutoLens ``Tracer`` whose mass profiles generate the model shear field.
        """
        self.dataset = dataset
        self.tracer = tracer

    @cached_property
    def model_shear(self) -> ShearYX2DIrregular:
        """
        The model shear field evaluated at the galaxy positions, via ``LensCalc``.

        When the dataset is marked ``is_reduced`` (real catalogues measure galaxy ellipticities,
        i.e. the reduced shear) the model quantity is ``g = gamma / (1 - kappa)``, with the
        convergence from the same Hessian primitive — so data and model always live in the same
        space.
        """
        lens_calc = LensCalc.from_tracer(self.tracer)

        shear = lens_calc.shear_yx_2d_via_hessian_from(grid=self.dataset.positions)

        if not getattr(self.dataset, "is_reduced", False):
            return shear

        convergence = lens_calc.convergence_2d_via_hessian_from(
            grid=self.dataset.positions
        )
        return ShearYX2DIrregular(
            values=np.asarray(shear) / (1.0 - np.asarray(convergence))[:, None],
            grid=self.dataset.positions,
        )

    @property
    def residual_map(self) -> np.ndarray:
        """``(N, 2)`` residuals ``data - model`` for each galaxy's ``(gamma_2, gamma_1)`` components."""
        return np.asarray(self.dataset.shear_yx) - np.asarray(self.model_shear)

    @property
    def normalized_residual_map(self) -> np.ndarray:
        """``(N, 2)`` residuals divided by the per-galaxy noise broadcast across both shear components."""
        noise = np.asarray(self.dataset.noise_map)[:, None]
        return self.residual_map / noise

    @property
    def chi_squared_map(self) -> np.ndarray:
        """``(N, 2)`` per-component chi-squared contributions."""
        return self.normalized_residual_map**2

    @property
    def chi_squared(self) -> float:
        """Scalar chi-squared summed over all ``N x 2`` shear measurements."""
        return float(np.sum(self.chi_squared_map))

    @property
    def noise_normalization(self) -> float:
        r"""
        Gaussian likelihood normalisation :math:`\sum \log(2 \pi \sigma^2)` summed over all ``N x 2`` shear
        measurements — the factor of 2 reflects that each galaxy contributes two independent components.
        """
        noise = np.asarray(self.dataset.noise_map)
        return float(2.0 * np.sum(np.log(2.0 * math.pi * noise**2)))

    @property
    def log_likelihood(self) -> float:
        r"""Standard Gaussian log-likelihood :math:`-\tfrac{1}{2}(\chi^2 + \text{noise normalization})`."""
        return -0.5 * (self.chi_squared + self.noise_normalization)

    @property
    def figure_of_merit(self) -> float:
        """Quantity returned to non-linear searches; same as ``log_likelihood`` (no inversion / evidence)."""
        return self.log_likelihood
