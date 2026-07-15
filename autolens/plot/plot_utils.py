import logging

from autogalaxy.util.plot_utils import _critical_curves_from, _caustics_from
from autoarray.plot.utils import numpy_lines as _to_lines

logger = logging.getLogger(__name__)


def _compute_critical_curve_lines(tracer, grid):
    """Compute critical-curve and caustic lines for a tracer on a given grid.

    Returns a 4-tuple ``(image_plane_lines, image_plane_line_colors,
    source_plane_lines, source_plane_line_colors)`` suitable for passing
    directly to :func:`~autoarray.plot.array.plot_array`.  On failure
    (e.g. the mass model has no critical curves) returns
    ``(None, None, None, None)``.

    Parameters
    ----------
    tracer
        The tracer whose mass distribution is used to trace critical curves
        and caustics.
    grid
        Image-plane grid on which the curves are evaluated.
    """
    try:
        tan_cc, rad_cc = _critical_curves_from(tracer, grid)
        tan_ca, rad_ca = _caustics_from(tracer, grid)
        _tan_cc_lines = _to_lines(list(tan_cc) if tan_cc is not None else []) or []
        _rad_cc_lines = _to_lines(list(rad_cc) if rad_cc is not None else []) or []
        _tan_ca_lines = _to_lines(list(tan_ca) if tan_ca is not None else []) or []
        _rad_ca_lines = _to_lines(list(rad_ca) if rad_ca is not None else []) or []
        image_plane_lines = (_tan_cc_lines + _rad_cc_lines) or None
        image_plane_line_colors = (
            ["white"] * len(_tan_cc_lines) + ["yellow"] * len(_rad_cc_lines)
        )
        source_plane_lines = (_tan_ca_lines + _rad_ca_lines) or None
        source_plane_line_colors = (
            ["white"] * len(_tan_ca_lines) + ["yellow"] * len(_rad_ca_lines)
        )
        return image_plane_lines, image_plane_line_colors, source_plane_lines, source_plane_line_colors
    except (ModuleNotFoundError, ValueError):
        # ModuleNotFoundError: jax_zero_contour missing — already warned upstream in
        # plot_utils._critical_curves_method().
        # ValueError: no zero crossings in the eigenvalue grid (e.g. slope >= 2
        # isothermal where lambda_r > 0 everywhere). Curves don't exist for this
        # model, so rendering without overlays is correct.
        return None, None, None, None
    except Exception:
        # Anything else — log loudly with traceback so the next regression of the
        # "ZeroSolver raised inside model-fit, viz fell back to all-zero" failure
        # mode (PyAutoGalaxy abd7b717, PyAutoFit #1280) does not stay silent.
        logger.warning(
            "Critical-curve computation failed unexpectedly; rendering without "
            "overlays. Investigate — this used to be a silent fallback.",
            exc_info=True,
        )
        return None, None, None, None
