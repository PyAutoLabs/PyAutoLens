"""
Diagnostics of what a transformation does to a twin, before any inference (lens-twin
report section 2.4):

- the physical model: convergence and deflection difference maps, enclosed-mass
  profiles M(<R), tangential critical curves and Einstein radii of both twins;
- the predicted observation: ``delta_chi_squared_image``, the noise-free
  sum (I_twin - I_parent)^2 / sigma^2 on the parent's masked grid with its PSF and
  noise map (``compare``);
- the observed data: ``compare_to_data``, chi^2(twin | data) - chi^2(parent | data) of
  two ``al.FitImaging``.

The two chi-squared numbers are different quantities and are named as such.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np

import autoarray as aa
import autogalaxy as ag
from autogalaxy.profiles.light.abstract import LightProfile
from autogalaxy.profiles.mass.abstract.abstract import MassProfile

from autolens.twins.aperture import einstein_radius_from
from autolens.twins.transform.stellar_dark import DEFAULT_APERTURE_SETTINGS

DEFAULT_RADII_FRACTIONS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)


def _difference(twin_values, parent_values) -> np.ndarray:
    """
    twin - parent, with entries that are identical (including both infinite, e.g. at
    an isothermal centre lying on a pixel centre) set to exactly zero rather than NaN.
    """
    twin_values = np.asarray(twin_values)
    parent_values = np.asarray(parent_values)
    with np.errstate(invalid="ignore"):
        return np.where(twin_values == parent_values, 0.0, twin_values - parent_values)


class TwinComparison:
    def __init__(
        self,
        grid: aa.Grid2D,
        convergence_parent: aa.Array2D,
        convergence_twin: aa.Array2D,
        deflections_parent,
        deflections_twin,
        centre: Tuple[float, float],
        radii: np.ndarray,
        mass_enclosed_parent: np.ndarray,
        mass_enclosed_twin: np.ndarray,
        tangential_critical_curves_parent: List[aa.Grid2DIrregular],
        tangential_critical_curves_twin: List[aa.Grid2DIrregular],
        einstein_radius_parent: Optional[float],
        einstein_radius_twin: Optional[float],
        image_parent: aa.Array2D,
        image_twin: aa.Array2D,
        noise_map: aa.Array2D,
    ):
        """
        The result of ``al.twins.compare(parent, twin)``. Differences are twin minus
        parent. ``einstein_radius_*`` is ``None`` when that tracer has no tangential
        critical curve inside the parent's setup grid.

        Attributes
        ----------
        grid
            The uniform grid of the convergence and deflection maps.
        radii, mass_enclosed_parent, mass_enclosed_twin
            M(<R) of all mass in each tracer about ``centre``, in convergence units
            (arcsec^2), by grid integration on ``grid``.
        image_parent, image_twin, noise_map
            The noise-free PSF-convolved images on the parent's masked grid and the
            parent's noise map.
        """
        self.grid = grid
        self.convergence_parent = convergence_parent
        self.convergence_twin = convergence_twin
        self.deflections_parent = deflections_parent
        self.deflections_twin = deflections_twin
        self.centre = centre
        self.radii = np.asarray(radii, dtype="float64")
        self.mass_enclosed_parent = np.asarray(mass_enclosed_parent, dtype="float64")
        self.mass_enclosed_twin = np.asarray(mass_enclosed_twin, dtype="float64")
        self.tangential_critical_curves_parent = tangential_critical_curves_parent
        self.tangential_critical_curves_twin = tangential_critical_curves_twin
        self.einstein_radius_parent = einstein_radius_parent
        self.einstein_radius_twin = einstein_radius_twin
        self.image_parent = image_parent
        self.image_twin = image_twin
        self.noise_map = noise_map

    @property
    def convergence_difference(self) -> aa.Array2D:
        """kappa_twin - kappa_parent on ``grid``."""
        return aa.Array2D(
            values=_difference(
                self.convergence_twin.array, self.convergence_parent.array
            ),
            mask=self.grid.mask,
        )

    @property
    def deflections_difference(self) -> aa.Grid2D:
        """alpha_twin - alpha_parent, (y, x) on ``grid``."""
        return aa.Grid2D(
            values=_difference(
                self.deflections_twin.array, self.deflections_parent.array
            ),
            mask=self.grid.mask,
        )

    @property
    def deflections_y_difference(self) -> aa.Array2D:
        return aa.Array2D(
            values=self.deflections_difference.array[:, 0], mask=self.grid.mask
        )

    @property
    def deflections_x_difference(self) -> aa.Array2D:
        return aa.Array2D(
            values=self.deflections_difference.array[:, 1], mask=self.grid.mask
        )

    @property
    def mass_enclosed_difference(self) -> np.ndarray:
        return _difference(self.mass_enclosed_twin, self.mass_enclosed_parent)

    @property
    def normalized_residual_map(self) -> aa.Array2D:
        """(I_twin - I_parent) / sigma on the parent's masked grid."""
        return aa.Array2D(
            values=_difference(self.image_twin.array, self.image_parent.array)
            / self.noise_map.array,
            mask=self.image_parent.mask,
        )

    @property
    def delta_chi_squared_image(self) -> float:
        """
        Sum over the parent's unmasked pixels of (I_twin - I_parent)^2 / sigma^2, noise
        free: how strongly the parent's observation distinguishes the two models.
        """
        return float(np.sum(self.normalized_residual_map.array**2))

    def __repr__(self):
        return (
            f"TwinComparison(delta_chi_squared_image={self.delta_chi_squared_image:.6g}, "
            f"einstein_radius_parent={self.einstein_radius_parent}, "
            f"einstein_radius_twin={self.einstein_radius_twin})"
        )


def _uniform_grid_from(mask: aa.Mask2D) -> aa.Grid2D:
    return aa.Grid2D.uniform(
        shape_native=mask.shape_native,
        pixel_scales=mask.pixel_scales,
        origin=mask.origin,
        over_sample_size=1,
    )


def _default_centre_from(tracer) -> Tuple[float, float]:
    """The centre of the first non-light, non-shear mass profile of the lowest-redshift galaxy."""
    for galaxy in tracer.galaxies_ascending_redshift:
        for profile in galaxy.profile_dict.values():
            if (
                isinstance(profile, MassProfile)
                and not isinstance(profile, LightProfile)
                and not isinstance(profile, ag.mp.ExternalShear)
            ):
                return tuple(float(value) for value in profile.centre)
    return (0.0, 0.0)


def _einstein_radius_or_none(tracer, grid, pixel_scale) -> Optional[float]:
    try:
        return einstein_radius_from(tracer=tracer, grid=grid, pixel_scale=pixel_scale)
    except ValueError:
        return None


def _on_setup(twin, setup):
    from autolens.twins.twin import Twin

    if twin.setup is setup:
        return twin
    return Twin(reference=twin.reference, tracer=twin.tracer, setup=setup)


def compare(
    parent,
    twin,
    grid: Optional[aa.Grid2D] = None,
    radii: Optional[Sequence[float]] = None,
    centre: Optional[Tuple[float, float]] = None,
    pixel_scale: float = 0.05,
    aperture_settings: Optional[dict] = None,
) -> TwinComparison:
    """
    Compare ``twin`` with ``parent`` (usually ``Twin.baseline(reference)``).

    Parameters
    ----------
    parent, twin
        The two ``Twin`` objects. Both images are computed on the parent's setup (its
        masked grid, PSF and noise map), whatever setup the twin carries.
    grid
        The uniform grid of the convergence and deflection maps and of the M(<R)
        integrals; defaults to the parent's mask geometry, unmasked.
    radii
        The M(<R) radii in arc-seconds; default 0.25, 0.5, ..., 1.5 times the parent's
        Einstein radius (or of 1 arcsec if it has none), dropping radii whose circle
        leaves ``grid``.
    centre
        The M(<R) centre; defaults to the centre of the parent's first total-mass profile
        in its lowest-redshift galaxy.
    pixel_scale
        The resolution of the critical-curve evaluation grid. Critical curves and
        Einstein radii are traced on the parent setup's masked grid, as in
        ``einstein_radius_aperture_from``.
    aperture_settings
        Integrator sub-sizes of the M(<R) integrals (default: ``ag.CircularAperture``'s,
        as in the transformation, so M(<theta_E) of the twin matches the parent's to the
        integrator's accuracy; coarser settings bias a central cusp). A gNFW convergence
        does one ``quad`` per point, so for gNFW twins pass coarse sub-sizes and few
        ``radii``.
    """
    setup = parent.setup
    grid = _uniform_grid_from(setup.mask) if grid is None else grid
    aperture_settings = {
        **DEFAULT_APERTURE_SETTINGS,
        **dict(aperture_settings or {}),
    }

    tracer_parent = parent.tracer
    tracer_twin = twin.tracer

    lens_calc_parent = ag.LensCalc.from_tracer(tracer_parent)
    lens_calc_twin = ag.LensCalc.from_tracer(tracer_twin)

    einstein_radius_parent = _einstein_radius_or_none(
        tracer_parent, setup.grid, pixel_scale
    )
    einstein_radius_twin = _einstein_radius_or_none(
        tracer_twin, setup.grid, pixel_scale
    )

    curves_parent = lens_calc_parent.tangential_critical_curve_list_from(
        grid=setup.grid, pixel_scale=pixel_scale
    )
    curves_twin = lens_calc_twin.tangential_critical_curve_list_from(
        grid=setup.grid, pixel_scale=pixel_scale
    )

    centre = _default_centre_from(tracer_parent) if centre is None else centre
    centre = (float(centre[0]), float(centre[1]))

    if radii is None:
        scale = einstein_radius_parent or 1.0
        half_y = grid.mask.shape_native[0] * grid.mask.pixel_scales[0] / 2.0
        half_x = grid.mask.shape_native[1] * grid.mask.pixel_scales[1] / 2.0
        max_radius = min(
            half_y - abs(centre[0] - grid.mask.origin[0]),
            half_x - abs(centre[1] - grid.mask.origin[1]),
        )
        radii = [
            fraction * scale
            for fraction in DEFAULT_RADII_FRACTIONS
            if fraction * scale <= max_radius
        ]
    radii = np.asarray(radii, dtype="float64")

    def mass_enclosed_from(tracer):
        return np.array(
            [
                ag.CircularAperture(
                    radius=float(radius), centre=centre, **aperture_settings
                ).mass_from(mass_obj=tracer, grid=grid)
                for radius in radii
            ]
        )

    image_parent = parent.noise_free_image_2d()
    image_twin = _on_setup(twin, setup).noise_free_image_2d()

    return TwinComparison(
        grid=grid,
        convergence_parent=tracer_parent.convergence_2d_from(grid=grid),
        convergence_twin=tracer_twin.convergence_2d_from(grid=grid),
        deflections_parent=tracer_parent.deflections_yx_2d_from(grid=grid),
        deflections_twin=tracer_twin.deflections_yx_2d_from(grid=grid),
        centre=centre,
        radii=radii,
        mass_enclosed_parent=mass_enclosed_from(tracer_parent),
        mass_enclosed_twin=mass_enclosed_from(tracer_twin),
        tangential_critical_curves_parent=curves_parent,
        tangential_critical_curves_twin=curves_twin,
        einstein_radius_parent=einstein_radius_parent,
        einstein_radius_twin=einstein_radius_twin,
        image_parent=image_parent,
        image_twin=image_twin,
        noise_map=setup.noise_map,
    )


def _check_parametric(tracer, label: str):
    for galaxy in tracer.galaxies:
        if galaxy.has(cls=aa.Pixelization):
            raise NotImplementedError(
                f"compare_to_data: the {label} tracer has a pixelized source. Delta "
                f"chi^2_data is defined for parametric sources only, because a pixelized "
                f"source is re-solved to the data by each fit."
            )


def compare_to_data(twin, dataset: aa.Imaging, parent=None) -> float:
    """
    Delta chi^2_data = chi^2(twin | data) - chi^2(parent | data): two ``al.FitImaging`` of
    the observed ``dataset``, one with each tracer. Positive means the observed data
    disfavour the twin relative to the parent.

    This is a different number from ``TwinComparison.delta_chi_squared_image`` (which is
    noise free and compares the two models with each other, not with the data).

    Parameters
    ----------
    twin
        The transformed twin.
    dataset
        The masked observed ``al.Imaging`` (e.g. ``ObservationSetup.
        dataset_from_fit_directory(fit_path)``).
    parent
        The twin to compare against; defaults to the reference (``twin.reference.tracer``).

    Raises
    ------
    NotImplementedError
        If either tracer has a pixelized source.
    """
    from autolens.imaging.fit_imaging import FitImaging

    tracer_parent = twin.reference.tracer if parent is None else parent.tracer
    _check_parametric(tracer_parent, "parent")
    _check_parametric(twin.tracer, "twin")

    chi_squared_twin = FitImaging(dataset=dataset, tracer=twin.tracer).chi_squared
    chi_squared_parent = FitImaging(dataset=dataset, tracer=tracer_parent).chi_squared
    return float(chi_squared_twin - chi_squared_parent)
