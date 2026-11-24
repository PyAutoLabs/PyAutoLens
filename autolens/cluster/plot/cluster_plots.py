"""
Module-level matplotlib helpers for visualising cluster-scale strong lenses.

Cluster fields differ from galaxy-scale lenses in ways that break the galaxy-scale
plot defaults: the field of view is arcminutes rather than arcseconds, there are tens
of mass components, several multiply-imaged sources must be told apart at a glance,
and — because the sources sit at *different* redshifts — every source plane has its
own critical curves and caustics. The helpers here encode the cluster-figure
conventions of the observational literature:

- A grayscale asinh stretch on the cluster image, using its native sky geometry.
- Shared family colours and marker shapes. Dense fields use the redshift palette
  when redshifts are supplied, with source index colours otherwise.
- Per-source-plane critical curves computed through the multi-plane ``LensCalc``
  (``use_multi_plane=True, plane_j=j``): at cluster scale the critical curves of the
  z=1 and z=2 source planes differ visibly, and plotting only the last plane's curve
  (the galaxy-scale default) under-reports the lensing structure.

Critical-curve engine: like every other PyAuto plot, the curve and caustic helpers
honour ``general.critical_curves_method`` in the visualize config, read once per plot
call through PyAutoGalaxy's ``autogalaxy.util.plot_utils._critical_curves_method()``
(which also applies its unknown-value and missing-``jax_zero_contour`` fallbacks to
``marching_squares``):

- ``marching_squares`` (the default) evaluates the eigen values on the supplied uniform
  grid and contours them. At arcminute scale a grid coarser than ~1" per pixel starts
  missing member-galaxy-scale features; the docstrings below give concrete grid
  guidance.
- ``zero_contour`` traces the zero contour of each eigen value with
  ``jax_zero_contour`` via ``LensCalc``'s ``*_via_zero_contour_from`` methods, called
  with their defaults exactly as the other PyAuto plots call them. The grid is then
  unused, and seed points come from ``LensCalc``'s automatic coarse-grid scan, which
  covers only a galaxy-scale (+/-3") box about the origin, so curves outside it are not
  found.

Neither path is claimed to be ``jax.jit`` / ``jax.vmap`` safe here: these are plotting
helpers that run on the numpy path.
"""

from types import SimpleNamespace
from typing import List, Optional, Sequence, Tuple

import numpy as np

import autoarray as aa

from autoarray.plot.utils import (
    subplots,
    save_figure,
    tight_layout,
)

from autolens.point.plot.cluster_sky import (
    angular_axes,
    cluster_field,
    draw_cluster_image,
    family_style,
)
from autolens.point.plot.point_sources import assign_styles
from autolens.point.plot.positions_panel import image_order_from


# Wong (2011) qualitative colours for source-plane curves.
WONG_PALETTE = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
]


def _positions_array(positions) -> np.ndarray:
    """Return an (N, 2) array of (y, x) positions."""
    array = positions.array if hasattr(positions, "array") else positions
    return np.asarray(array).reshape(-1, 2)


def _cluster_image(image, pixel_scales):
    if image is None or (hasattr(image, "geometry") and pixel_scales is None):
        return image
    if pixel_scales is None:
        raise ValueError("pixel_scales must be given for a raw image.")
    if hasattr(image, "geometry") and np.allclose(image.pixel_scales, pixel_scales):
        return image
    return aa.Array2D.no_mask(
        values=np.asarray(getattr(image, "native", image)),
        pixel_scales=pixel_scales,
        origin=getattr(getattr(image, "geometry", None), "origin", (0.0, 0.0)),
    )


def _family_markers(positions_list, datasets, on_image):
    positions_list = list(positions_list)
    if datasets is not None and len(datasets) != len(positions_list):
        raise ValueError("datasets must supply metadata for each position family.")
    records = [
        SimpleNamespace(
            k=index,
            name=dataset.name if dataset is not None else f"Source {index}",
            redshift=dataset.redshift if dataset is not None else None,
        )
        for index, dataset in enumerate(
            datasets if datasets is not None else [None] * len(positions_list)
        )
    ]
    redshift_styles = assign_styles(records)
    for record, positions in zip(records, positions_list):
        arr = _positions_array(positions)
        if not len(arr):
            continue
        if redshift_styles is None:
            bright, dark, marker = family_style(record)
        else:
            bright = dark = record.color
            marker = record.marker
        name, redshift = record.name, record.redshift
        label = f"{name} · {len(arr)} {'image' if len(arr) == 1 else 'images'}"
        if redshift is not None:
            label += f" · z={redshift:.2f}"
        yield arr, bright if on_image else dark, marker, name, label


def _scatter_family(ax, positions, colour, marker, size, label=None):
    from matplotlib import patheffects
    from matplotlib.colors import to_rgb

    artist = ax.scatter(
        positions[:, 1],
        positions[:, 0],
        facecolors="none",
        edgecolors=colour,
        marker=marker,
        s=size,
        linewidths=1.6,
        zorder=7,
        label=label,
    )

    if ax.images and np.dot(to_rgb(colour), (0.2126, 0.7152, 0.0722)) < 0.35:
        artist.set_path_effects(
            [patheffects.withStroke(linewidth=2.6, foreground=(1, 1, 1, 0.65))]
        )


def _set_sky_field(ax, image=None, positions=None):
    if image is not None:
        centre, half = cluster_field(
            image.geometry.extent, np.empty((0, 2)) if positions is None else positions
        )
    else:
        bounds = np.asarray(ax.dataLim.get_points())[:, ::-1]
        if np.isfinite(bounds).all():
            centre = bounds.mean(axis=0)
            half = max(float(np.ptp(bounds, axis=0).max()) * 0.56, 1.0e-4)
        else:
            centre, half = (0.0, 0.0), 1.0
    angular_axes(ax, centre, half)


def _top_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncols=min(
                5 if len(handles) > 12 else 4 if len(handles) > 4 else 3, len(handles)
            ),
            frameon=False,
            fontsize=8,
            handletextpad=0.4,
            columnspacing=1.1,
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


def _draw_kpc_scale_bar(ax, redshift: float, cosmology, kpc: float, colour: str):
    """Draw a physical scale bar in the bottom-left corner of *ax*."""
    arcsec_per_kpc = float(cosmology.arcsec_per_kpc_proper(redshift))
    bar_arcsec = kpc * arcsec_per_kpc
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x0 = xlim[0] + 0.05 * (xlim[1] - xlim[0])
    y0 = ylim[0] + 0.05 * (ylim[1] - ylim[0])
    ax.plot([x0, x0 + bar_arcsec], [y0, y0], color=colour, lw=3.0, zorder=5)
    ax.text(
        x0 + bar_arcsec / 2.0,
        y0 + 0.02 * (ylim[1] - ylim[0]),
        f"{int(kpc)} kpc",
        color=colour,
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


# LensCalc routine names per engine, keyed by (eigen value kind, curve type). The
# marching-squares routines take the uniform ``grid``; the zero-contour routines take
# no grid and are called with their defaults, mirroring
# ``autogalaxy.util.plot_utils._critical_curves_from`` / ``_caustics_from``.
_CURVE_ROUTINES = {
    "marching_squares": {
        ("tangential", "critical_curve"): "tangential_critical_curve_list_from",
        ("radial", "critical_curve"): "radial_critical_curve_list_from",
        ("tangential", "caustic"): "tangential_caustic_list_from",
        ("radial", "caustic"): "radial_caustic_list_from",
    },
    "zero_contour": {
        (
            "tangential",
            "critical_curve",
        ): "tangential_critical_curve_list_via_zero_contour_from",
        (
            "radial",
            "critical_curve",
        ): "radial_critical_curve_list_via_zero_contour_from",
        ("tangential", "caustic"): "tangential_caustic_list_via_zero_contour_from",
        ("radial", "caustic"): "radial_caustic_list_via_zero_contour_from",
    },
}


def _critical_curves_method() -> str:
    """The configured critical-curve engine, via PyAutoGalaxy's shared selector.

    Returns ``"marching_squares"`` or ``"zero_contour"``; unknown values and a
    missing ``jax_zero_contour`` fall back to ``"marching_squares"`` with a warning,
    exactly as for every other PyAuto plot.
    """
    from autogalaxy.util.plot_utils import _critical_curves_method

    return _critical_curves_method()


def _curve_list_from(lens_calc, grid, method: str, kind: str, curve_type: str):
    """Compute one plane's ``kind`` (tangential / radial) critical curves or caustics
    (``curve_type`` = ``"critical_curve"`` / ``"caustic"``) with the engine *method*.
    """
    routine = getattr(lens_calc, _CURVE_ROUTINES[method][(kind, curve_type)])
    if method == "zero_contour":
        return routine()
    return routine(grid=grid)


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
    *,
    datasets: Optional[Sequence] = None,
):
    """
    Plot every source's observed multiple-image positions, per-source coloured, on the
    full cluster field.

    Parameters
    ----------
    positions_list
        One ``Grid2DIrregular`` (or ``(N, 2)`` array) of image-plane positions per
        source. Colours and marker shapes are shared across the cluster dataset figures.
    datasets
        Optional point datasets in the same order, supplying family names and redshifts.
    image
        Optional background image (``Array2D`` or raw 2D ndarray) drawn with the
        shared grayscale asinh stretch; without it positions are drawn on a blank field.
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
    image = _cluster_image(image, pixel_scales)
    families = list(
        _family_markers(positions_list, datasets, on_image=image is not None)
    )
    standalone = ax is None
    if standalone:
        fig, ax = subplots(1, 1, figsize=(10 if len(families) > 5 else 7, 7))
    if image is not None:
        draw_cluster_image(ax, image)

    for arr, colour, marker, _, label in families:
        _scatter_family(ax, arr, colour, marker, size=95, label=label)

    _draw_centre_markers(
        ax, centres=centres, halo_centres=halo_centres, marker_size=marker_size
    )
    positions = [arr for arr, *_ in families]
    if centres is not None:
        positions.append(_positions_array(centres))
    if halo_centres is not None:
        positions.append(_positions_array(halo_centres))
    _set_sky_field(ax, image, np.concatenate(positions) if positions else None)

    if kpc_scale_bar is not None and redshift is not None and cosmology is not None:
        _draw_kpc_scale_bar(
            ax,
            redshift=redshift,
            cosmology=cosmology,
            kpc=kpc_scale_bar,
            colour="white" if image is not None else "#282d32",
        )

    if standalone:
        fig.suptitle(title, y=0.99)
    else:
        ax.set_title(title, y=1.02 + 0.07 * np.ceil(len(families) / 5))
    _top_legend(ax)

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
    *,
    datasets: Optional[Sequence] = None,
):
    """
    A panel grid with one zoom per multiple image, rows grouped by source.

    Each panel is a ``zoom_arcsec``-wide cut-out centred on one observed image
    position, framed in that source's palette colour. This is how the eye verifies a
    model at cluster scale: full-field residuals hide a half-arcsecond position
    mismatch that a 6" zoom makes obvious. Image numbers follow the angular
    ordering used in the point-fit figures.

    Parameters
    ----------
    positions_list
        One ``Grid2DIrregular`` (or ``(N, 2)`` array) per source.
    image
        Background image (``Array2D`` or raw 2D ndarray).
    datasets
        Optional point datasets in the same order, used for family names.
    pixel_scales
        Arcsec per pixel; taken from ``image.pixel_scales`` when omitted.
    zoom_arcsec
        Full width of each zoom panel in arcsec.
    max_cols
        Maximum panels per row; a source with more images wraps onto extra rows.
    """
    if max_cols < 1 or zoom_arcsec <= 0:
        raise ValueError("max_cols and zoom_arcsec must be positive.")
    image = _cluster_image(image, pixel_scales)
    families = list(_family_markers(positions_list, datasets, on_image=True))
    n_cols = min(max_cols, max((len(arr) for arr, *_ in families), default=1))
    n_rows = max(1, sum(int(np.ceil(len(arr) / n_cols)) for arr, *_ in families))
    fig, axes = subplots(
        n_rows, n_cols, squeeze=False, figsize=(2.6 * n_cols, 2.7 * n_rows)
    )
    for ax in axes.flat:
        ax.set_axis_off()

    row = 0
    for positions, colour, marker, name, _ in families:
        for index, position in enumerate(positions[image_order_from(positions)]):
            row_offset, col = divmod(index, n_cols)
            ax = axes[row + row_offset, col]
            ax.set_axis_on()
            draw_cluster_image(ax, image)
            _scatter_family(ax, position[None, :], colour, marker, size=120)
            angular_axes(ax, position, zoom_arcsec / 2, small=True)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title(f"{name} · Image {index + 1}", fontsize=10)
            for spine in ax.spines.values():
                spine.set_edgecolor(colour)
                spine.set_linewidth(1.5)
        row += int(np.ceil(len(positions) / n_cols))

    fig.suptitle(title, y=1 - 0.08 / fig.get_figheight())
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
    well outside a z=1 source's. Curve colours identify the source planes and match the
    caustic colours. Legend labels give each plane's redshift and curve type.

    The curve engine is ``general.critical_curves_method`` from the visualize config,
    read once per call (see the module docstring): ``marching_squares`` (default)
    contours the eigen values on *grid*; ``zero_contour`` traces them with
    ``jax_zero_contour`` through each plane's ``LensCalc``
    ``*_critical_curve_list_via_zero_contour_from`` methods and ignores *grid*. A
    configured ``zero_contour`` without ``jax_zero_contour`` installed falls back to
    ``marching_squares`` with a warning.

    Parameters
    ----------
    tracer
        The (multi-plane) tracer whose critical curves are computed via the
        multi-plane ``LensCalc`` (``use_multi_plane=True, plane_j=j``), numpy path.
    grid
        The uniform grid the marching-squares solver walks (unused by
        ``zero_contour``). At arcminute scale, resolve to ~0.5" per pixel or finer
        over the full field (e.g. ``Grid2D.uniform(shape_native=(240, 240),
        pixel_scales=0.5)`` for a 2' field) — coarser grids miss
        member-galaxy-scale wiggles in the curves.
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
        fig, ax = subplots(1, 1, figsize=(8, 7))

    image = _cluster_image(image, pixel_scales)
    if image is not None:
        draw_cluster_image(ax, image)

    method = _critical_curves_method()

    for j in _source_plane_indices(tracer, plane_indices):
        colour = WONG_PALETTE[(j - 1) % len(WONG_PALETTE)]
        redshift = float(tracer.planes[j].redshift)
        lens_calc = _lens_calc_for_plane(tracer, plane_index=j)

        tangential_list = _curve_list_from(
            lens_calc, grid, method, "tangential", "critical_curve"
        )
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
            radial_list = _curve_list_from(
                lens_calc, grid, method, "radial", "critical_curve"
            )
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

    curves = [
        np.column_stack((line.get_ydata(), line.get_xdata())) for line in ax.lines
    ]
    _set_sky_field(ax, image, np.concatenate(curves) if curves else None)
    _top_legend(ax)
    if standalone:
        fig.suptitle(title, y=0.99)
    else:
        ax.set_title(title, y=1.14)

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

    The engine follows ``general.critical_curves_method``, read once per call, exactly
    as in ``plot_critical_curves``: ``marching_squares`` (default) uses each plane's
    ``LensCalc`` ``*_caustic_list_from(grid=grid)``; ``zero_contour`` uses
    ``*_caustic_list_via_zero_contour_from()``. Either way each caustic is that plane's
    critical curve ray-traced by that plane's multi-plane deflections.

    Parameters
    ----------
    tracer, grid, plane_indices
        As in ``plot_critical_curves`` (same engine and grid-resolution guidance
        applies).
    include_radial
        Also draw each plane's radial caustic (dashed); at cluster scale the radial
        caustic bounds the region producing central demagnified images, so it is on
        by default here (unlike the critical-curve helper, where the tangential curve
        is the primary observable).
    """
    standalone = ax is None
    if standalone:
        fig, ax = subplots(1, 1, figsize=(8, 7))

    method = _critical_curves_method()

    for j in _source_plane_indices(tracer, plane_indices):
        colour = WONG_PALETTE[(j - 1) % len(WONG_PALETTE)]
        redshift = float(tracer.planes[j].redshift)
        lens_calc = _lens_calc_for_plane(tracer, plane_index=j)

        tangential_list = _curve_list_from(
            lens_calc, grid, method, "tangential", "caustic"
        )
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
            radial_list = _curve_list_from(lens_calc, grid, method, "radial", "caustic")
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

    _set_sky_field(ax)
    ax.set_xlabel("Source-plane x")
    ax.set_ylabel("Source-plane y")
    _top_legend(ax)
    if standalone:
        fig.suptitle(title, y=0.99)
    else:
        ax.set_title(title, y=1.14)

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
    *,
    datasets: Optional[Sequence] = None,
):
    """
    Combined cluster mosaic: positions overlay | per-plane critical curves (+ image).

    When *tracer* (and *grid*) are omitted only the positions panel is drawn. The
    right panel repeats the positions on top of the critical curves so image
    multiplicities can be read against the curves directly. Optional ``datasets``
    supply the family names and redshifts in the same order as ``positions_list``.
    """
    positions_list = list(positions_list)
    n_cols = 2 if tracer is not None else 1
    fig, axes = subplots(1, n_cols, figsize=(8 * n_cols, 7))
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
        datasets=datasets,
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
        families = list(
            _family_markers(positions_list, datasets, on_image=image is not None)
        )
        for arr, colour, marker, _, _ in families:
            _scatter_family(axes[1], arr, colour, marker, size=72)
        # Use the same sky field so a family can be located across both panels.
        axis_limits = [
            (min(a.get_xlim()[0] for a in axes), max(a.get_xlim()[1] for a in axes)),
            (min(a.get_ylim()[0] for a in axes), max(a.get_ylim()[1] for a in axes)),
        ]
        centre = [np.mean(limits) for limits in reversed(axis_limits)]
        half = max(high - low for low, high in axis_limits) / 2
        for ax in axes:
            panel_title = ax.get_title()
            angular_axes(ax, centre, half)
            ax.set_title(panel_title, fontsize=11)

    fig.suptitle(title)
    tight_layout()
    save_figure(fig, path=output_path, filename=output_filename, format=output_format)
