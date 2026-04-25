"""
Simulate weak-lensing shear catalogues from a ``Tracer``.

The shear field of a strong lens system is computed by ``Tracer.shear_yx_2d_via_hessian_from``, which differentiates
the deflection-angle field to derive the lensing Hessian and from there the (gamma_2, gamma_1) shear at any (y, x).
That gives a *noise-free* shear field; this module adds two extras needed for a realistic weak-lensing simulation:

1. **Shape noise.** Each background source galaxy has a random unlensed ellipticity, drawn here as iid Gaussian
   noise per shear component with standard deviation ``noise_sigma``. Realistic values are around 0.2 - 0.4.

2. **Source-galaxy positions.** Either provide an explicit grid of (y, x) source positions, or let the simulator
   draw a uniform-random distribution of ``n_galaxies`` positions inside a square arc-second extent — sufficient
   for development work; future iterations may swap in number-density / luminosity-function-driven distributions.

The output is a :class:`autolens.weak.dataset.WeakDataset`.
"""
from typing import Optional

import numpy as np

import autoarray as aa

from autogalaxy.operate.lens_calc import LensCalc
from autogalaxy.util.shear_field import ShearYX2DIrregular

from autolens.weak.dataset import WeakDataset


class SimulatorShearYX:
    def __init__(self, noise_sigma: float = 0.3, seed: Optional[int] = None):
        """
        Simulator for weak-lensing shear catalogues.

        Parameters
        ----------
        noise_sigma
            Standard deviation of the per-component Gaussian shape noise added to each measured shear vector.
            Values around 0.2 - 0.4 are realistic for ground-based weak-lensing surveys; ``0.0`` produces a
            noise-free shear field, useful for unit tests.
        seed
            Optional seed for the random number generator. When set, both the noise and the random-position
            draws (for ``via_tracer_random_positions_from``) are reproducible.
        """
        self.noise_sigma = float(noise_sigma)
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def via_tracer_from(
        self,
        tracer,
        grid: aa.Grid2DIrregular,
        name: str = "",
    ) -> WeakDataset:
        """
        Simulate a weak-lensing shear catalogue from a ``Tracer`` evaluated at the given (y, x) source positions.

        Computes ``tracer.shear_yx_2d_via_hessian_from(grid=grid)`` — which goes through ``LensCalc`` under the
        hood — and adds iid Gaussian shape noise per component with std ``self.noise_sigma``.

        Parameters
        ----------
        tracer
            A PyAutoLens ``Tracer`` object exposing ``shear_yx_2d_via_hessian_from``.
        grid
            The (y, x) source-galaxy positions where the shear is measured. Must be an ``aa.Grid2DIrregular``
            (or convertible to one).
        name
            Optional label passed through to ``WeakDataset.name``.
        """
        if not isinstance(grid, aa.Grid2DIrregular):
            grid = aa.Grid2DIrregular(values=grid)

        true_shear = self._true_shear_yx_from(tracer=tracer, grid=grid)

        if self.noise_sigma > 0.0:
            noise = self._rng.normal(
                loc=0.0, scale=self.noise_sigma, size=true_shear.shape
            )
            shear_array = np.asarray(true_shear) + noise
        else:
            shear_array = np.asarray(true_shear)

        shear_yx = ShearYX2DIrregular(values=shear_array, grid=grid)

        noise_map = aa.ArrayIrregular(
            values=[self.noise_sigma] * len(grid)
        )

        return WeakDataset(shear_yx=shear_yx, noise_map=noise_map, name=name)

    def via_tracer_random_positions_from(
        self,
        tracer,
        n_galaxies: int,
        grid_extent: float = 3.0,
        name: str = "",
    ) -> WeakDataset:
        """
        Simulate a weak-lensing shear catalogue at ``n_galaxies`` uniform-random source positions.

        Galaxy positions are drawn uniformly inside a square ``[-grid_extent, +grid_extent]`` arc-second box.
        This is the simplest sensible distribution for development purposes; production weak-lensing simulations
        typically use a survey-specific number density / redshift distribution.

        Parameters
        ----------
        tracer
            A PyAutoLens ``Tracer`` object.
        n_galaxies
            Number of background source galaxies to simulate.
        grid_extent
            Half-width of the square (in arc-seconds) inside which positions are drawn.
        name
            Optional label passed through to ``WeakDataset.name``.
        """
        positions = self._rng.uniform(
            low=-grid_extent, high=grid_extent, size=(n_galaxies, 2)
        )
        grid = aa.Grid2DIrregular(values=positions)
        return self.via_tracer_from(tracer=tracer, grid=grid, name=name)

    @staticmethod
    def _true_shear_yx_from(tracer, grid: aa.Grid2DIrregular):
        """
        Evaluate the noise-free shear field of ``tracer`` on ``grid``.

        - If the input exposes ``shear_yx_2d_via_hessian_from`` directly, use it.
        - Else, if it exposes ``deflections_between_planes_from`` (it duck-types as a ``Tracer``), build a
          multi-plane ``LensCalc`` via ``from_tracer``.
        - Otherwise treat the input as a mass-like object (single ``MassProfile``, ``Galaxy``, ``Galaxies``)
          and build a single-plane ``LensCalc`` via ``from_mass_obj``. This fallback keeps the simulator
          usable in unit tests that don't want to spin up a full ``Tracer``.
        """
        method = getattr(tracer, "shear_yx_2d_via_hessian_from", None)
        if method is not None:
            return method(grid=grid)
        if hasattr(tracer, "deflections_between_planes_from"):
            return LensCalc.from_tracer(tracer).shear_yx_2d_via_hessian_from(grid=grid)
        return LensCalc.from_mass_obj(tracer).shear_yx_2d_via_hessian_from(grid=grid)
