"""
Plots of ``al.twins.compare`` diagnostics (ruling R6: the plotter lives in the
library and follows the ``aplt`` conventions of ``autolens/lens/plot/tracer_plots.py``).
"""

from typing import Optional

import numpy as np

from autogalaxy.util.plot_utils import plot_array
from autoarray.plot.utils import (
    conf_subplot_figsize,
    hide_unused_axes,
    numpy_lines as _to_lines,
    save_figure,
    subplots,
    tight_layout,
)

PARENT_COLOR = "white"
TWIN_COLOR = "red"


def _curve_lines(curves):
    return _to_lines(list(curves) if curves else []) or []


def subplot_twin_comparison(
    comparison,
    output_path: Optional[str] = None,
    output_filename: str = "twin_comparison",
    output_format: Optional[str] = None,
    colormap: Optional[str] = None,
    title_prefix: Optional[str] = None,
):
    """
    A 2x3 subplot of a ``TwinComparison`` (``al.twins.compare(parent, twin)``).

    Panels:
      0: parent noise-free image (parent critical curve white, twin red)
      1: twin noise-free image, on the parent's masked grid
      2: (I_twin - I_parent) / sigma, titled with delta chi^2_image
      3: convergence difference kappa_twin - kappa_parent
      4: enclosed mass M(<R) of parent and twin, with both Einstein radii
      5: tangential critical curves of parent and twin

    Parameters
    ----------
    comparison
        The ``TwinComparison`` to plot.
    output_path
        Directory in which the figure is saved; ``None`` shows it instead.
    output_filename
        File name without extension.
    output_format
        e.g. ``"png"`` (or a list of formats); ``None`` reads the config default.
    colormap
        Matplotlib colormap name for the image panels.
    title_prefix
        Optional text prepended to every panel title.
    """
    _pf = (lambda t: f"{title_prefix.rstrip()} {t}") if title_prefix else (lambda t: t)

    parent_lines = _curve_lines(comparison.tangential_critical_curves_parent)
    twin_lines = _curve_lines(comparison.tangential_critical_curves_twin)
    lines = (parent_lines + twin_lines) or None
    line_colors = [PARENT_COLOR] * len(parent_lines) + [TWIN_COLOR] * len(twin_lines)

    fig, axes = subplots(2, 3, figsize=conf_subplot_figsize(2, 3))
    axes_flat = list(np.array(axes).flatten())

    plot_array(
        array=comparison.image_parent,
        ax=axes_flat[0],
        title=_pf("Parent Image (Noise Free)"),
        lines=lines,
        line_colors=line_colors,
        colormap=colormap,
    )
    plot_array(
        array=comparison.image_twin,
        ax=axes_flat[1],
        title=_pf("Twin Image (Noise Free)"),
        lines=lines,
        line_colors=line_colors,
        colormap=colormap,
    )
    plot_array(
        array=comparison.normalized_residual_map,
        ax=axes_flat[2],
        title=_pf(
            f"(Twin - Parent) / Noise, "
            f"$\\Delta\\chi^2_{{image}}$ = {comparison.delta_chi_squared_image:.4g}"
        ),
        colormap=colormap,
        symmetric=True,
        cb_unit="",
    )
    convergence_difference = comparison.convergence_difference
    # An isothermal centre on a pixel centre gives an infinite difference there; blank it.
    convergence_difference = convergence_difference.with_new_array(
        np.where(
            np.isfinite(convergence_difference.array),
            convergence_difference.array,
            np.nan,
        )
    )
    plot_array(
        array=convergence_difference,
        ax=axes_flat[3],
        title=_pf("Convergence Twin - Parent"),
        lines=lines,
        line_colors=line_colors,
        colormap=colormap,
        symmetric=True,
        cb_unit="",
    )

    ax = axes_flat[4]
    ax.plot(comparison.radii, comparison.mass_enclosed_parent, "k-o", label="Parent")
    ax.plot(comparison.radii, comparison.mass_enclosed_twin, "r--s", label="Twin")
    for einstein_radius, style in (
        (comparison.einstein_radius_parent, "k:"),
        (comparison.einstein_radius_twin, "r:"),
    ):
        if einstein_radius is not None:
            ax.axvline(einstein_radius, linestyle=style[1:], color=style[0])
    ax.set_xlabel("R (arcsec)")
    ax.set_ylabel("M(<R) (convergence units, arcsec$^2$)")
    ax.set_title(_pf("Enclosed Mass"))
    ax.legend()

    ax = axes_flat[5]
    for curves, color, label in (
        (parent_lines, "k", "Parent"),
        (twin_lines, "r", "Twin"),
    ):
        for index, curve in enumerate(curves):
            ax.plot(
                curve[:, 1],
                curve[:, 0],
                color=color,
                label=label if index == 0 else None,
            )
    ax.set_aspect("equal")
    ax.set_xlabel("x (arcsec)")
    ax.set_ylabel("y (arcsec)")
    ax.set_title(_pf("Tangential Critical Curves"))
    if parent_lines or twin_lines:
        ax.legend()

    hide_unused_axes(axes_flat)
    tight_layout()
    save_figure(fig, path=output_path, filename=output_filename, format=output_format)
