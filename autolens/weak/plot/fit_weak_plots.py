"""
Module-level matplotlib helpers for visualising a ``FitWeak``.

Same headless-quiver convention as ``weak_dataset_plots`` (`pivot="middle", headwidth=0, headlength=0,
headaxislength=0`): shear is a spin-2 quantity, so the segments are drawn without arrowheads. The data is
plotted in black, the model in red with low alpha — overlaying them on a single axes makes deviations
visible at a glance.
"""
from typing import Optional

import numpy as np

from autoarray.plot.grid import plot_grid
from autoarray.plot.utils import (
    subplots,
    save_figure,
    conf_subplot_figsize,
    tight_layout,
)


def _positions_yx(shear_yx) -> np.ndarray:
    grid = shear_yx.grid
    return np.array(grid.array if hasattr(grid, "array") else grid)


def _quiver_components(shear_yx):
    """Return ``(y, x, u, v, mag)`` for a quiver of a ``ShearYX2DIrregular``."""
    positions = _positions_yx(shear_yx)
    y, x = positions[:, 0], positions[:, 1]
    mag = np.asarray(shear_yx.ellipticities)
    phi_rad = np.deg2rad(np.asarray(shear_yx.phis))
    u = mag * np.cos(phi_rad)
    v = mag * np.sin(phi_rad)
    return y, x, u, v, mag


def plot_data_vs_model(
    fit,
    ax=None,
    title: str = "Data vs Model",
    output_path: Optional[str] = None,
    output_filename: str = "data_vs_model",
    output_format: Optional[str] = None,
):
    """
    Overlay the data and model shear fields as headless quivers on a single axes.

    Data is drawn in black, model in red (alpha=0.6). Deviations are visible where the two segments
    disagree in length or orientation.
    """
    standalone = ax is None
    if standalone:
        fig, ax = subplots(1, 1)

    y_d, x_d, u_d, v_d, _ = _quiver_components(fit.dataset.shear_yx)
    y_m, x_m, u_m, v_m, _ = _quiver_components(fit.model_shear)

    quiver_kwargs = dict(
        pivot="middle", headwidth=0, headlength=0, headaxislength=0
    )
    ax.quiver(x_d, y_d, u_d, v_d, color="black", label="data", **quiver_kwargs)
    ax.quiver(
        x_m, y_m, u_m, v_m, color="red", alpha=0.6, label="model", **quiver_kwargs
    )
    ax.set_xlabel('x (")')
    ax.set_ylabel('y (")')
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize="small")

    if standalone:
        tight_layout()
        save_figure(
            fig,
            path=output_path,
            filename=output_filename,
            format=output_format,
        )


def plot_residuals(
    fit,
    ax=None,
    title: str = "Residuals",
    output_path: Optional[str] = None,
    output_filename: str = "residuals",
    output_format: Optional[str] = None,
):
    """
    Quiver of the per-galaxy residual shear ``data - model``, colour-coded by ``|residual|``.

    Uses the diverging ``RdBu_r`` colormap centred on the median residual magnitude so over- and
    under-shears are visually distinguishable.
    """
    standalone = ax is None
    if standalone:
        fig, ax = subplots(1, 1)

    positions = _positions_yx(fit.dataset.shear_yx)
    y, x = positions[:, 0], positions[:, 1]

    residual = fit.residual_map
    mag = np.sqrt(residual[:, 0] ** 2 + residual[:, 1] ** 2)
    phi_rad = 0.5 * np.arctan2(residual[:, 0], residual[:, 1])
    u = mag * np.cos(phi_rad)
    v = mag * np.sin(phi_rad)

    ax.quiver(
        x,
        y,
        u,
        v,
        mag,
        pivot="middle",
        headwidth=0,
        headlength=0,
        headaxislength=0,
        cmap="RdBu_r",
    )
    ax.set_xlabel('x (")')
    ax.set_ylabel('y (")')
    ax.set_title(title)
    ax.set_aspect("equal")

    if standalone:
        tight_layout()
        save_figure(
            fig,
            path=output_path,
            filename=output_filename,
            format=output_format,
        )


def plot_chi_squared_map(
    fit,
    ax=None,
    title: str = r"$\chi^2$ Map",
    output_path: Optional[str] = None,
    output_filename: str = "chi_squared_map",
    output_format: Optional[str] = None,
):
    """
    Colour-coded scatter of per-galaxy chi-squared (sum over the two shear components).

    Uses ``plot_grid`` from ``autoarray.plot.grid`` with ``color_array`` set to ``chi_squared_map.sum(axis=-1)``.
    """
    per_galaxy_chi_squared = np.asarray(fit.chi_squared_map).sum(axis=-1)

    plot_grid(
        grid=_positions_yx(fit.dataset.shear_yx),
        ax=ax,
        color_array=per_galaxy_chi_squared,
        colormap="magma",
        title=title,
        output_path=output_path if ax is None else None,
        output_filename=output_filename,
        output_format=output_format,
    )


def subplot_fit_quick(
    fit,
    output_path: Optional[str] = None,
    output_format: Optional[str] = None,
    title_prefix: Optional[str] = None,
):
    """
    Placeholder quick-update subplot for a ``FitWeak``.

    Currently delegates to the full ``subplot_fit_weak``. Will be
    replaced with a lighter-weight render in future.
    """
    subplot_fit_weak(
        fit,
        output_path=output_path,
        output_filename="fit_quick",
        output_format=output_format,
        title_prefix=title_prefix,
    )


def subplot_fit_weak(
    fit,
    output_path: Optional[str] = None,
    output_filename: str = "subplot_fit_weak",
    output_format: Optional[str] = None,
    title_prefix: Optional[str] = None,
):
    """
    Produce a 2x2 subplot mosaic visualising a ``FitWeak``.

    Panels: data shear field, model shear field, data-vs-model overlay, chi-squared map.
    """
    from autolens.weak.plot.weak_dataset_plots import plot_shear_yx_2d

    fig, axes = subplots(2, 2, figsize=conf_subplot_figsize(2, 2))
    ax_data, ax_model, ax_overlay, ax_chi = (
        axes[0, 0],
        axes[0, 1],
        axes[1, 0],
        axes[1, 1],
    )

    _prefix = f"{title_prefix.rstrip()} " if title_prefix else ""

    plot_shear_yx_2d(
        shear_yx=fit.dataset.shear_yx, ax=ax_data, title=f"{_prefix}Data"
    )
    plot_shear_yx_2d(
        shear_yx=fit.model_shear, ax=ax_model, title=f"{_prefix}Model"
    )
    plot_data_vs_model(fit=fit, ax=ax_overlay, title=f"{_prefix}Data vs Model")
    plot_chi_squared_map(fit=fit, ax=ax_chi, title=f"{_prefix}$\\chi^2$ Map")

    tight_layout()
    save_figure(
        fig,
        path=output_path,
        filename=output_filename,
        format=output_format,
    )
