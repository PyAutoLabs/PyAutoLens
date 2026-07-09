"""
Module-level matplotlib helpers for visualising cluster-scale strong lenses.

Cluster fields differ from galaxy-scale lenses in ways that break the galaxy-scale
plot defaults: the field of view is arcminutes rather than arcseconds, there are tens
of mass components, several multiply-imaged sources must be told apart at a glance,
and — because the sources sit at *different* redshifts — every source plane has its
own critical curves and caustics. The helpers here encode the cluster-figure
conventions of the observational literature:

- A percentile-driven ``LogNorm`` on a perceptually-ordered colormap, so one image
  spans the BCG core, intra-cluster light and faint arcs without saturating.
- Per-source qualitative colouring from the Wong (2011) colour-blind-safe palette
  (https://www.nature.com/articles/nmeth.1618), applied consistently across every
  panel so "the orange source" means the same thing everywhere.
- Per-source-plane critical curves computed through the multi-plane ``LensCalc``
  (``use_multi_plane=True, plane_j=j``): at cluster scale the critical curves of the
  z=1 and z=2 source planes differ visibly, and plotting only the last plane's curve
  (the galaxy-scale default) under-reports the lensing structure.

Numerical note: the critical-curve solvers walk a uniform grid with a contour finder
and are numpy-path only (they are not ``jax.vmap``-safe). At arcminute scale a grid
coarser than ~1" per pixel starts missing member-galaxy-scale features; the
docstrings below give concrete grid guidance.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np

from autoarray.plot.utils import (
    subplots,
    save_figure,
    tight_layout,
)

# Wong (2011) colour-blind-safe qualitative palette; per-source colours in every helper.
WONG_PALETTE = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#000000",  # black
]

CLUSTER_CMAP = "gnuplot2"


def _image_native_and_pixel_scale(image, pixel_scales) -> Tuple[np.ndarray, float]:
    """Return ``(native 2D array, pixel scale)`` for an ``Array2D`` or raw ndarray."""
    native = np.asarray(image.native if hasattr(image, "native") else image)
    if pixel_scales is None:
        if hasattr(image, "pixel_scales"):
            pixel_scales = image.pixel_scales[0]
        else:
            raise ValueError(
                "pixel_scales must be given when image is a raw ndarray with no "
                ".pixel_scales attribute."
            )
    scale = pixel_scales[0] if isinstance(pixel_scales, (tuple, list)) else pixel_scales
    return native, float(scale)


def _lognorm_from(image_native: np.ndarray):
    """Percentile-driven ``LogNorm`` (5th / 99.5th of positive pixels).

    Cluster imaging has bright cores and faint arcs; fixed ``(vmin, vmax)`` bounds do
    not transfer between datasets. Non-positive pixels are excluded because
    ``LogNorm`` needs strictly positive bounds.
    """
    from matplotlib.colors import LogNorm

    positive = image_native[image_native > 0]
    if positive.size == 0:
        raise ValueError("Image has no positive pixels; cannot build a LogNorm.")
    vmin = float(np.percentile(positive, 5.0))
    vmax = float(np.percentile(positive, 99.5))
    return LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10.0))


def _extent_from(image_native: np.ndarray, pixel_scale: float) -> Tuple:
    """``imshow`` extent ``(left, right, bottom, top)`` for an origin-centred field."""
    half_y = image_native.shape[0] * pixel_scale / 2.0
    half_x = image_native.shape[1] * pixel_scale / 2.0
    return (-half_x, half_x, -half_y, half_y)


def _positions_array(positions) -> np.ndarray:
    """Return an ``(N, 2)`` ``[y, x]`` array from a Grid2DIrregular / list / array."""
    array = positions.array if hasattr(positions, "array") else positions
    return np.atleast_2d(np.asarray(array))


def _draw_image(ax, image, pixel_scales):
    native, scale = _image_native_and_pixel_scale(image, pixel_scales)
    ax.imshow(
        native,
        origin="lower",
        cmap=CLUSTER_CMAP,
        norm=_lognorm_from(native),
        extent=_extent_from(native, scale),
    )


def _draw_centre_markers(ax, centres, halo_centres, marker_size: float):
    """White stars for galaxy centres, a white plus for halo centres.

    Conventional cluster-figure markers, white with a thin black edge so they read
    against both the BCG core and the sky.
    """
    if centres is not None:
        arr = _positions_array(centres)
        ax.scatter(
            arr[:, 1],
            arr[:, 0],
            marker="*",
            s=marker_size,
            c="white",
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
            label="Galaxy centres",
        )
    if halo_centres is not None:
        arr = _positions_array(halo_centres)
        ax.scatter(
            arr[:, 1],
            arr[:, 0],
            marker="P",
            s=marker_size * 0.7,
            c="white",
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
            label="Halo centres",
        )


def _draw_kpc_scale_bar(ax, redshift: float, cosmology, kpc: float):
    """Draw a physical scale bar in the bottom-left corner of *ax*."""
    arcsec_per_kpc = float(cosmology.arcsec_per_kpc_proper(redshift))
    bar_arcsec = kpc * arcsec_per_kpc
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x0 = xlim[0] + 0.05 * (xlim[1] - xlim[0])
    y0 = ylim[0] + 0.05 * (ylim[1] - ylim[0])
    ax.plot([x0, x0 + bar_arcsec], [y0, y0], color="white", lw=3.0, zorder=5)
    ax.text(
        x0 + bar_arcsec / 2.0,
        y0 + 0.02 * (ylim[1] - ylim[0]),
        f"{int(kpc)} kpc",
        color="white",
        ha="center",
        va="bottom",
        fontsize=11,
    )


def _source_plane_indices(tracer, plane_indices) -> List[int]:
    """Every non-image plane index by default (each source plane gets its own curves)."""
    if plane_indices is not None:
        return list(plane_indices)
    return list(range(1, len(tracer.planes)))


def _lens_calc_for_plane(tracer, plane_index: int):
    from autogalaxy.operate.lens_calc import LensCalc

    return LensCalc.from_tracer(tracer, use_multi_plane=True, plane_j=plane_index)


def plot_positions_overlay(
    positions_list,
    image=None,
    pixel_scales=None,
    centres=None,
    halo_centres=None,
    redshift: Optional[float] = None,
    cosmology=None,
    kpc_scale_bar: Optional[float] = 50.0,
    marker_size: float = 220.0,
    ax=None,
    title: str = "Cluster Multiple-Image Positions",
    output_path: Optional[str] = None,
    output_filename: str = "cluster_positions",
    output_format: Optional[str] = None,
):
    """
    Plot every source's observed multiple-image positions, per-source coloured, on the
    full cluster field.

    Parameters
    ----------
    positions_list
        One ``Grid2DIrregular`` (or ``(N, 2)`` array) of image-plane positions per
        source. Source ``i`` is drawn in ``WONG_PALETTE[i]`` — the same colour this
        source gets in every other helper.
    image
        Optional background image (``Array2D`` or raw 2D ndarray) drawn with the
        cluster ``LogNorm``; without it positions are drawn on a blank field.
    pixel_scales
        Arcsec per pixel of *image*; taken from ``image.pixel_scales`` when omitted
        and the image is an ``Array2D``.
    centres, halo_centres
        Optional galaxy / halo centre grids drawn with the conventional white
        star / plus markers.
    redshift, cosmology, kpc_scale_bar
        When all given, a ``kpc_scale_bar``-kpc physical scale bar is drawn (50 kpc at
        z=0.5 is ~8" — visible against a 100" field without dominating it).
    ax
        Existing ``Axes`` to draw onto; ``None`` creates (and saves) a standalone
        figure.
    output_path, output_filename, output_format
        Standard workspace output controls (ignored when *ax* is supplied).
    """
    standalone = ax is None
    if standalone:
        fig, ax = subplots(1, 1)

    if image is not None:
        _draw_image(ax, image, pixel_scales)

    for i, positions in enumerate(positions_list):
        arr = _positions_array(positions)
        colour = WONG_PALETTE[i % len(WONG_PALETTE)]
        ax.scatter(
            arr[:, 1],
            arr[:, 0],
            facecolors="none",
            edgecolors=colour,
            marker="o",
            s=140,
            linewidths=1.8,
            zorder=6,
            label=f"Source {i} images",
        )

    _draw_centre_markers(
        ax, centres=centres, halo_centres=halo_centres, marker_size=marker_size
    )

    if kpc_scale_bar is not None and redshift is not None and cosmology is not None:
        _draw_kpc_scale_bar(
            ax, redshift=redshift, cosmology=cosmology, kpc=kpc_scale_bar
        )

    ax.set_xlabel('x (")')
    ax.set_ylabel('y (")')
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.6)

    if standalone:
        tight_layout()
        save_figure(
            fig, path=output_path, filename=output_filename, format=output_format
        )


def plot_image_group_zooms(
    positions_list,
    image,
    pixel_scales=None,
    zoom_arcsec: float = 6.0,
    max_cols: int = 4,
    title: str = "Multiple-Image Zooms",
    output_path: Optional[str] = None,
    output_filename: str = "cluster_image_zooms",
    output_format: Optional[str] = None,
):
    """
    A panel grid with one zoom per multiple image, rows grouped by source.

    Each panel is a ``zoom_arcsec``-wide cut-out centred on one observed image
    position, framed in that source's palette colour. This is how the eye verifies a
    model at cluster scale: full-field residuals hide a half-arcsecond position
    mismatch that a 6" zoom makes obvious.

    Parameters
    ----------
    positions_list
        One ``Grid2DIrregular`` (or ``(N, 2)`` array) per source.
    image
        Background image (``Array2D`` or raw 2D ndarray).
    pixel_scales
        Arcsec per pixel; taken from ``image.pixel_scales`` when omitted.
    zoom_arcsec
        Full width of each zoom panel in arcsec.
    max_cols
        Maximum panels per row; a source with more images wraps onto extra rows.
    """
    native, scale = _image_native_and_pixel_scale(image, pixel_scales)
    norm = _lognorm_from(native)
    extent = _extent_from(native, scale)
    half = zoom_arcsec / 2.0

    panels = [
        (i, arr_row)
        for i, positions in enumerate(positions_list)
        for arr_row in _positions_array(positions)
    ]
    n_panels = max(len(panels), 1)
    n_cols = min(max_cols, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))

    fig, axes = subplots(n_rows, n_cols)
    axes = np.atleast_1d(axes).ravel()

    for ax in axes[n_panels:]:
        ax.set_axis_off()

    counters = {}
    for ax, (source_index, (y, x)) in zip(axes, panels):
        counters[source_index] = counters.get(source_index, 0) + 1
        colour = WONG_PALETTE[source_index % len(WONG_PALETTE)]
        ax.imshow(native, origin="lower", cmap=CLUSTER_CMAP, norm=norm, extent=extent)
        ax.scatter(
            [x],
            [y],
            facecolors="none",
            edgecolors=colour,
            marker="o",
            s=220,
            linewidths=2.0,
        )
        ax.set_xlim(x - half, x + half)
        ax.set_ylim(y - half, y + half)
        ax.set_title(
            f"Source {source_index} image {counters[source_index]}",
            color=colour,
            fontsize=10,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(colour)
            spine.set_linewidth(2.0)

    fig.suptitle(title)
    tight_layout()
    save_figure(fig, path=output_path, filename=output_filename, format=output_format)


def plot_critical_curves(
    tracer,
    grid,
    image=None,
    pixel_scales=None,
    plane_indices: Optional[Sequence[int]] = None,
    include_radial: bool = False,
    linewidth: float = 2.0,
    ax=None,
    title: str = "Multi-Plane Critical Curves",
    output_path: Optional[str] = None,
    output_filename: str = "cluster_critical_curves",
    output_format: Optional[str] = None,
):
    """
    Plot the critical curves of **every source plane** of a multi-plane tracer.

    At cluster scale each source redshift has its own critical curves (the lensing
    strength D_LS / D_S differs per plane), so a z=2 source's tangential curve sits
    well outside a z=1 source's. Curves are coloured per plane with the same palette
    the positions helpers use for the corresponding source, and labelled with the
    plane redshift.

    Parameters
    ----------
    tracer
        The (multi-plane) tracer whose critical curves are computed via the
        multi-plane ``LensCalc`` (``use_multi_plane=True, plane_j=j``), numpy path.
    grid
        The uniform grid the marching solver walks. At arcminute scale, resolve to
        ~0.5" per pixel or finer over the full field (e.g. ``Grid2D.uniform(
        shape_native=(240, 240), pixel_scales=0.5)`` for a 2' field) — coarser grids
        miss member-galaxy-scale wiggles in the curves.
    image, pixel_scales
        Optional background image for the overlay.
    plane_indices
        Which planes to draw; default every non-image plane.
    include_radial
        Also draw each plane's radial critical curve (dashed).
    ax
        Existing ``Axes`` to draw onto; ``None`` creates (and saves) a standalone
        figure.
    """
    standalone = ax is None
    if standalone:
        fig, ax = subplots(1, 1)

    if image is not None:
        _draw_image(ax, image, pixel_scales)

    for j in _source_plane_indices(tracer, plane_indices):
        colour = WONG_PALETTE[(j - 1) % len(WONG_PALETTE)]
        redshift = float(tracer.planes[j].redshift)
        lens_calc = _lens_calc_for_plane(tracer, plane_index=j)

        tangential_list = lens_calc.tangential_critical_curve_list_from(grid=grid)
        for k, curve in enumerate(tangential_list):
            arr = _positions_array(curve)
            ax.plot(
                arr[:, 1],
                arr[:, 0],
                color=colour,
                lw=linewidth,
                zorder=6,
                label=f"z={redshift:.2f} tangential" if k == 0 else None,
            )

        if include_radial:
            radial_list = lens_calc.radial_critical_curve_list_from(grid=grid)
            for k, curve in enumerate(radial_list):
                arr = _positions_array(curve)
                ax.plot(
                    arr[:, 1],
                    arr[:, 0],
                    color=colour,
                    lw=linewidth * 0.75,
                    ls="--",
                    zorder=6,
                    label=f"z={redshift:.2f} radial" if k == 0 else None,
                )

    ax.set_xlabel('x (")')
    ax.set_ylabel('y (")')
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.6)

    if standalone:
        tight_layout()
        save_figure(
            fig, path=output_path, filename=output_filename, format=output_format
        )


def plot_caustics(
    tracer,
    grid,
    plane_indices: Optional[Sequence[int]] = None,
    include_radial: bool = True,
    linewidth: float = 2.0,
    ax=None,
    title: str = "Multi-Plane Caustics",
    output_path: Optional[str] = None,
    output_filename: str = "cluster_caustics",
    output_format: Optional[str] = None,
):
    """
    Plot the caustics of every source plane of a multi-plane tracer, in source-plane
    arcsec coordinates.

    Each plane's caustics live in that plane's own coordinates — overlaying them on a
    single set of axes (per-plane colours matching ``plot_critical_curves``) shows
    where each source must sit relative to its own caustic network to produce the
    observed image multiplicities.

    Parameters
    ----------
    tracer, grid, plane_indices
        As in ``plot_critical_curves`` (same grid-resolution guidance applies).
    include_radial
        Also draw each plane's radial caustic (dashed); at cluster scale the radial
        caustic bounds the region producing central demagnified images, so it is on
        by default here (unlike the critical-curve helper, where the tangential curve
        is the primary observable).
    """
    standalone = ax is None
    if standalone:
        fig, ax = subplots(1, 1)

    for j in _source_plane_indices(tracer, plane_indices):
        colour = WONG_PALETTE[(j - 1) % len(WONG_PALETTE)]
        redshift = float(tracer.planes[j].redshift)
        lens_calc = _lens_calc_for_plane(tracer, plane_index=j)

        tangential_list = lens_calc.tangential_caustic_list_from(grid=grid)
        for k, curve in enumerate(tangential_list):
            arr = _positions_array(curve)
            ax.plot(
                arr[:, 1],
                arr[:, 0],
                color=colour,
                lw=linewidth,
                zorder=6,
                label=f"z={redshift:.2f} tangential" if k == 0 else None,
            )

        if include_radial:
            radial_list = lens_calc.radial_caustic_list_from(grid=grid)
            for k, curve in enumerate(radial_list):
                arr = _positions_array(curve)
                ax.plot(
                    arr[:, 1],
                    arr[:, 0],
                    color=colour,
                    lw=linewidth * 0.75,
                    ls="--",
                    zorder=6,
                    label=f"z={redshift:.2f} radial" if k == 0 else None,
                )

    ax.set_xlabel('source-plane x (")')
    ax.set_ylabel('source-plane y (")')
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.6)

    if standalone:
        tight_layout()
        save_figure(
            fig, path=output_path, filename=output_filename, format=output_format
        )


def subplot_cluster_dataset(
    positions_list,
    image=None,
    pixel_scales=None,
    tracer=None,
    grid=None,
    centres=None,
    halo_centres=None,
    plane_indices: Optional[Sequence[int]] = None,
    title: str = "Cluster Dataset",
    output_path: Optional[str] = None,
    output_filename: str = "subplot_cluster_dataset",
    output_format: Optional[str] = None,
):
    """
    Combined cluster mosaic: positions overlay | per-plane critical curves (+ image).

    When *tracer* (and *grid*) are omitted only the positions panel is drawn. The
    right panel repeats the positions on top of the critical curves so image
    multiplicities can be read against the curves directly.
    """
    n_cols = 2 if tracer is not None else 1
    fig, axes = subplots(1, n_cols)
    axes = np.atleast_1d(axes).ravel()

    plot_positions_overlay(
        positions_list,
        image=image,
        pixel_scales=pixel_scales,
        centres=centres,
        halo_centres=halo_centres,
        kpc_scale_bar=None,
        ax=axes[0],
        title="Multiple-Image Positions",
    )

    if tracer is not None:
        if grid is None:
            raise ValueError("grid must be given when tracer is supplied.")
        plot_critical_curves(
            tracer,
            grid=grid,
            image=image,
            pixel_scales=pixel_scales,
            plane_indices=plane_indices,
            ax=axes[1],
            title="Per-Plane Critical Curves",
        )
        for i, positions in enumerate(positions_list):
            arr = _positions_array(positions)
            axes[1].scatter(
                arr[:, 1],
                arr[:, 0],
                facecolors="none",
                edgecolors=WONG_PALETTE[i % len(WONG_PALETTE)],
                marker="o",
                s=90,
                linewidths=1.4,
                zorder=7,
            )

    fig.suptitle(title)
    tight_layout()
    save_figure(fig, path=output_path, filename=output_filename, format=output_format)
