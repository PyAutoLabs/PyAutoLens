"""
Module-level matplotlib helpers for visualising a ``WeakDataset``.

A shear catalogue is a set of complex shear measurements ``(gamma_2, gamma_1)``
at the ``(y, x)`` positions of background source galaxies.  The natural way to
draw it is matplotlib's ``quiver`` with **headless line segments**, because
shear is a spin-2 quantity — a 180-degree rotation maps the shear back to
itself, so an arrowhead would suggest a directionality the data does not
have.  This is the same convention used in weak-lensing science papers
(e.g. KiDS, DES).

The plotters access the shear field exclusively through the derived properties
``.ellipticities`` (``|gamma|``) and ``.phis`` (position angle, in **degrees**)
defined on ``AbstractShearField``.  Indexing the underlying ``[:, 0]`` /
``[:, 1]`` storage directly is deliberately avoided so the plotters keep
working if the ``[gamma_2, gamma_1]`` convention pinned by PyAutoGalaxy PR
#366 ever changes.
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
    """Return the ``(N, 2)`` ``[y, x]`` position array for a shear field."""
    grid = shear_yx.grid
    return np.array(grid.array if hasattr(grid, "array") else grid)


def plot_shear_yx_2d(
    shear_yx,
    ax=None,
    title: str = "Shear Field",
    output_path: Optional[str] = None,
    output_filename: str = "shear_yx",
    output_format: Optional[str] = None,
):
    """
    Plot a shear field as a quiver of headless line segments at galaxy positions.

    Each segment is centred on the galaxy position (``pivot="middle"``), has a
    length proportional to the shear magnitude ``|gamma|`` and is oriented at
    the shear position angle ``phi``.  Segments are colour-coded by ``|gamma|``.

    Parameters
    ----------
    shear_yx
        A ``ShearYX2D`` / ``ShearYX2DIrregular`` carrying the shear vectors and
        the ``(y, x)`` galaxy grid.
    ax
        Existing ``Axes`` to draw onto; ``None`` creates a new figure.
    title
        Figure title.
    output_path, output_filename, output_format
        Standard workspace output controls.  When ``ax`` is supplied the saving
        is the caller's responsibility (typically ``subplot_weak_dataset``).
    """
    positions = _positions_yx(shear_yx)
    y, x = positions[:, 0], positions[:, 1]

    mag = np.asarray(shear_yx.ellipticities)
    phi_rad = np.deg2rad(np.asarray(shear_yx.phis))

    u = mag * np.cos(phi_rad)
    v = mag * np.sin(phi_rad)

    standalone = ax is None
    if standalone:
        fig, ax = subplots(1, 1)

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
        cmap="viridis",
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


def plot_ellipticities(
    shear_yx,
    ax=None,
    title: str = r"Shear Magnitude $|\gamma|$",
    output_path: Optional[str] = None,
    output_filename: str = "shear_ellipticities",
    output_format: Optional[str] = None,
):
    """
    Plot a colour-coded scatter of the shear magnitude ``|gamma|`` at each galaxy.

    Delegates to ``autoarray.plot.grid.plot_grid`` with ``color_array`` set to
    the per-galaxy ellipticities.
    """
    plot_grid(
        grid=_positions_yx(shear_yx),
        ax=ax,
        color_array=np.asarray(shear_yx.ellipticities),
        colormap="viridis",
        title=title,
        output_path=output_path if ax is None else None,
        output_filename=output_filename,
        output_format=output_format,
    )


def plot_phis(
    shear_yx,
    ax=None,
    title: str = r"Shear Position Angle $\phi$",
    output_path: Optional[str] = None,
    output_filename: str = "shear_phis",
    output_format: Optional[str] = None,
):
    """
    Plot a colour-coded scatter of the shear position angle ``phi`` at each galaxy.

    Position angles are cyclic, so a cyclic colormap (``twilight``) is used.
    """
    plot_grid(
        grid=_positions_yx(shear_yx),
        ax=ax,
        color_array=np.asarray(shear_yx.phis),
        colormap="twilight",
        title=title,
        output_path=output_path if ax is None else None,
        output_filename=output_filename,
        output_format=output_format,
    )


def plot_noise_map(
    dataset,
    ax=None,
    title: str = "Noise Map",
    output_path: Optional[str] = None,
    output_filename: str = "noise_map",
    output_format: Optional[str] = None,
):
    """
    Plot a colour-coded scatter of the per-galaxy shear noise at each position.

    Takes the full ``WeakDataset`` (not just the shear field) because the noise
    map lives on the dataset.
    """
    plot_grid(
        grid=_positions_yx(dataset.shear_yx),
        ax=ax,
        color_array=np.asarray(dataset.noise_map),
        colormap="magma",
        title=title,
        output_path=output_path if ax is None else None,
        output_filename=output_filename,
        output_format=output_format,
    )


def subplot_weak_dataset(
    dataset,
    output_path: Optional[str] = None,
    output_filename: str = "subplot_weak_dataset",
    output_format: Optional[str] = None,
    title_prefix: Optional[str] = None,
):
    """
    Produce a 2x2 subplot mosaic visualising a ``WeakDataset``.

    Panels: shear field, noise map, shear magnitude, shear position angle.
    """
    fig, axes = subplots(2, 2, figsize=conf_subplot_figsize(2, 2))
    ax_quiver, ax_noise, ax_mag, ax_phi = (
        axes[0, 0],
        axes[0, 1],
        axes[1, 0],
        axes[1, 1],
    )

    _prefix = f"{title_prefix.rstrip()} " if title_prefix else ""
    name_part = f" — {dataset.name}" if dataset.name else ""

    plot_shear_yx_2d(
        shear_yx=dataset.shear_yx,
        ax=ax_quiver,
        title=f"{_prefix}Shear Field{name_part}",
    )
    plot_noise_map(
        dataset=dataset,
        ax=ax_noise,
        title=f"{_prefix}Noise Map{name_part}",
    )
    plot_ellipticities(
        shear_yx=dataset.shear_yx,
        ax=ax_mag,
        title=f"{_prefix}Shear Magnitude{name_part}",
    )
    plot_phis(
        shear_yx=dataset.shear_yx,
        ax=ax_phi,
        title=f"{_prefix}Position Angle{name_part}",
    )

    tight_layout()
    save_figure(
        fig,
        path=output_path,
        filename=output_filename,
        format=output_format,
    )
