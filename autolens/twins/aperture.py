"""
The Einstein-radius aperture of the mass normalisation convention (ruling R3,
PyAutoGalaxy ``docs/api/mass_budget.rst``): a circle of radius theta_E,parent, the
Einstein radius of the whole parent lens model, centred on the mass centre of the
galaxy being replaced.

theta_E is computed at the tracer level, ``ag.LensCalc.from_tracer(tracer)
.einstein_radius_from(grid)``: the radius of the circle with the area of the
tangential critical curve of all mass in the tracer (galaxies, external shear and
other fields), so it is defined for multi-galaxy and non-isothermal parents.
"""

from typing import Tuple

import autoarray as aa
import autogalaxy as ag


def einstein_radius_from(
    tracer, grid: aa.type.Grid2DLike, pixel_scale: float = 0.05
) -> float:
    """
    The tracer-level Einstein radius, in arc-seconds.

    Parameters
    ----------
    tracer
        The lens model (all its mass contributes).
    grid
        The grid whose mask sets the region in which the tangential critical curve is
        searched (``LensCalc``'s evaluation grid zooms to it). ``al.twins`` uses the
        observation setup's masked grid, ``setup.grid``.
    pixel_scale
        The resolution of the evaluation grid on which the critical curve is traced.

    Raises
    ------
    ValueError
        If the tracer has no tangential critical curve inside the grid.
    """
    einstein_radius = float(
        ag.LensCalc.from_tracer(tracer).einstein_radius_from(
            grid=grid, pixel_scale=pixel_scale
        )
    )
    if not einstein_radius > 0.0:
        raise ValueError(
            "The tracer has no tangential critical curve inside the grid, so its "
            "Einstein radius (and the Einstein-radius aperture) is undefined."
        )
    return einstein_radius


def einstein_radius_aperture_from(
    tracer,
    grid: aa.type.Grid2DLike,
    centre: Tuple[float, float],
    pixel_scale: float = 0.05,
    **integrator_settings,
) -> ag.CircularAperture:
    """
    The circular aperture of radius theta_E,parent (``einstein_radius_from``) centred on
    ``centre``. The Einstein radius is the returned aperture's ``radius``.

    Parameters
    ----------
    tracer
        The parent lens model.
    grid
        The grid on which theta_E is computed (see ``einstein_radius_from``).
    centre
        The (y,x) aperture centre in arc-seconds: the mass centre of the replaced galaxy.
    pixel_scale
        The resolution of the critical-curve evaluation grid.
    integrator_settings
        ``sub_size_interior``, ``sub_size_edge``, ``sub_size_centre`` and
        ``centre_refine_pixels`` of ``ag.CircularAperture`` (the grid integrator's
        accuracy). A ``gNFW`` convergence does one ``quad`` per point, so pass coarse
        sub-sizes for it.
    """
    return ag.CircularAperture(
        radius=einstein_radius_from(tracer=tracer, grid=grid, pixel_scale=pixel_scale),
        centre=(float(centre[0]), float(centre[1])),
        **integrator_settings,
    )


def check_aperture_in_grid(aperture: ag.CircularAperture, grid: aa.Grid2D):
    """
    Raise ``ValueError`` if the circle extends beyond the extent of the uniform grid the
    integrator uses (``grid.mask`` shape, pixel scales and origin; the mask is ignored).
    """
    mask = grid.mask
    half_y = mask.shape_native[0] * mask.pixel_scales[0] / 2.0
    half_x = mask.shape_native[1] * mask.pixel_scales[1] / 2.0
    dy = abs(aperture.centre[0] - mask.origin[0]) + aperture.radius
    dx = abs(aperture.centre[1] - mask.origin[1]) + aperture.radius
    if dy > half_y or dx > half_x:
        raise ValueError(
            f'The aperture (radius {aperture.radius:.4f}" at {aperture.centre}) extends '
            f'beyond the integration grid (half-extent {half_y:.4f}" x {half_x:.4f}"); '
            f"pass a larger grid."
        )
