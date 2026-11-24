"""
The point-source positions figures for any number of point sources (one for a galaxy-scale lens, many for a
cluster), drawn in one visual language:

- ``dataset_point.png`` (:func:`plot_dataset_positions`): "Dataset Positions": every source's observed positions.
  Cluster maps use an outlined marker for each family and label its images ``k·i`` (family ``k``, image ``i``).
  The top legend gives each family’s image count and redshift when available.
- ``fit.png`` (:func:`plot_fit_positions`): "Fit Positions": the observed positions, the model positions (red ✕) at
  their true position joined to their observed image by a true-length residual line (solid for hard pairing, dashed
  for the soft all-pairs mixture), unmatched model images, and each source's critical curves at its OWN redshift in
  its colour. Cluster overviews frame the observed and model positions in the same square cutout. The legend
  gives the fitted statistic of each source and the fit's
  figure of merit: ``χ² = …``, or ``−2 ln L = …`` for the all-pairs mixture (``FitPositionsImagePairAll``), whose
  value is a mixture log likelihood, not a chi-squared. Beyond 8 sources colour follows redshift (colourbar) and the
  model ✕ is smaller.
- ``fit_per_image.png`` (:func:`subplot_fit_per_image`): one row per source, three square panels: "Residuals" (that
  source's images ranked worst first by Δ/σ on a linear range shared by every source, clipped at a robust maximum
  with off-scale values printed), "IP" (an image-plane zoom on that source's images and its own curves) and "SP"
  (the source-plane panel of :mod:`autolens.point.plot.source_plane_panel`). The images of a row share one colour
  per image across all three panels. Source-plane fits rank by the fit's own per-image chi. Beyond 8 sources each
  source is written to its own file, ``fit_per_image/<name>.png``, with the ranking range still shared.
- ``image_with_points.png`` / ``fit_image_with_points.png`` (:func:`plot_image_with_points`): the observed image with
  every source's positions in the same symbols as the sky map, ``k·i``
  labels and, for the fit, the model ✕ / residual lines and each source's tangential critical curves in its colour.

The per-image figures carry no legends: the symbols are those of the dataset and fit figures.
"""

import logging
from typing import Dict, Optional

import numpy as np

from autoarray.plot.utils import save_figure, subplots

from autolens.point.plot import positions_panel as pp
from autolens.point.plot import point_sources as ps

logger = logging.getLogger(__name__)

# Critical curves and caustics: dark and thick, so they read against the markers.
CURVE_LINEWIDTH = 1.8
RADIAL_CURVE_LINEWIDTH = 1.4
CURVE_COLOR = "0.2"
RADIAL_CURVE_COLOR = "#5c5c00"

# Every observed marker has this area (equalised across shapes by `marker_size_from`).
MARKER_SIZE = 110

# The image-number label style of every positions figure.
LABEL_BBOX = dict(fc="w", ec="none", alpha=0.8, pad=0.5)

# An observed image with no model image of its own: a dashed grey ring around it.
NO_MODEL_RING = dict(
    s=420, facecolors="none", edgecolors="0.3", linewidths=1.3, linestyle=(0, (2.5, 1.5))
)

LEGEND_KWARGS = dict(fontsize=9, framealpha=0.85, markerscale=0.7)


def darken(color, amount: float = 0.3):
    """
    Returns ``color`` darkened by ``amount`` (black stays black), for curves drawn in a source's colour.
    """
    import matplotlib.colors as mcolors

    if isinstance(color, str) and color in ("k", "black"):
        return "k"
    r, g, b = mcolors.to_rgb(color)
    return (r * (1 - amount), g * (1 - amount), b * (1 - amount))


def marker_size_from(marker: str, base: float = MARKER_SIZE) -> float:
    """
    Equalises the visual area of the marker shapes: diamonds read larger and triangles smaller than circles at equal
    ``s``.
    """
    return base * {"D": 0.7, "^": 1.15, "v": 1.15, "<": 1.15, ">": 1.15, "P": 1.1}.get(
        marker, 1.0
    )


def is_many(records) -> bool:
    return len(records) > ps.MAX_QUALITATIVE


def figsize_from(records):
    """
    The library's single-figure size; 12 x 12 inches beyond 8 sources, so ~100 labels fit.
    """
    from autoarray.plot.utils import conf_figsize

    return conf_figsize("figures") if not is_many(records) else (12, 12)


def font_scale_from(records) -> float:
    """
    Beyond 8 sources the figure is 12 inches, so titles and ticks are scaled by ``12 / figsize`` to keep their
    library size relative to the panel; only the image labels shrink.
    """
    from autoarray.plot.utils import conf_figsize

    if not is_many(records):
        return 1.0
    return 12.0 / float(conf_figsize("figures")[0])


def label_fontsize_from(records) -> float:
    return pp.FIT_NUMBER_FONTSIZE if not is_many(records) else 9


def model_size_from(records) -> float:
    return pp.MODEL_MARKER_SIZE if not is_many(records) else 30


def scale_fonts(ax, scale: float):
    if scale == 1.0:
        return
    ax.title.set_fontsize(ax.title.get_fontsize() * scale)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(tick.get_fontsize() * scale)


def finish_panel(ax, title: str, extent, scale: float = 1.0):
    """
    The title, arc-second ticks and extent of a positions panel (library fonts), with an equal aspect ratio.
    """
    from autoarray.plot.utils import apply_extent, apply_labels

    apply_labels(ax, title=title, xlabel="", ylabel="")
    apply_extent(ax, extent)
    ax.set_aspect("equal", adjustable="box")
    scale_fonts(ax, scale)


def _prefix(title_prefix: Optional[str]) -> str:
    return f"{title_prefix.rstrip()} " if title_prefix else ""


def _redshift_text(record) -> str:
    return f", z = {record.redshift:g}" if record.redshift is not None else ""


def image_label_from(record, number, multi: bool) -> str:
    return f"{record.k}·{number}" if multi else f"{number}"


# ---------------------------------------------------------------------------
# Label placement
# ---------------------------------------------------------------------------


def place_labels(
    ax,
    extent,
    points,
    labels,
    obstacles=None,
    fontsize: float = pp.FIT_NUMBER_FONTSIZE,
    marker_radius: float = 7.0,
    label_colors=None,
    halo_color=None,
):
    """
    Labels every point, beside its own marker when a free spot exists, otherwise pushed further out with a thin
    leader line back to its marker (an extension of
    :func:`~autolens.point.plot.positions_panel.adjacent_number_placements_from` for crowded clusters).

    Each label tries 8 directions (outward from the panel centre first) on rings of increasing distance; every
    candidate is scored against all markers, the ``obstacles`` (e.g. model positions), the labels already placed and
    the panel frame. Crowded markers choose first.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(points) == 0:
        return
    width, height, scale = pp._panel_size_points(ax, extent)
    origin = np.array([extent[0], extent[2]])
    markers = (points[:, ::-1] - origin) * scale
    occupied = markers
    if obstacles is not None and len(obstacles):
        occupied = np.vstack(
            [
                markers,
                (np.asarray(obstacles, dtype=float).reshape(-1, 2)[:, ::-1] - origin)
                * scale,
            ]
        )
    centre = np.array([width, height]) / 2.0

    density = np.array([np.sum(np.hypot(*(occupied - p).T) < 40.0) for p in markers])
    order = np.argsort(-density, kind="stable")

    placed, results = [], [None] * len(markers)
    for index in order:
        p = markers[index]
        w = 0.62 * fontsize * len(labels[index]) + 4.0
        h = fontsize + 3.0
        out = p - centre
        base = np.arctan2(out[1], out[0]) if np.hypot(*out) > 1 else np.pi * 0.75
        angles = base + np.radians([0, 45, -45, 90, -90, 135, -135, 180])
        best, best_score = None, np.inf
        for ring, radius in enumerate(
            (
                marker_radius + 0.5,
                marker_radius + 14.0,
                marker_radius + 28.0,
                marker_radius + 44.0,
            )
        ):
            for angle in angles:
                d = np.array([np.cos(angle), np.sin(angle)])
                c = p + d * radius + d * np.array([w / 2, h / 2])
                box = (c[0] - w / 2, c[1] - h / 2, c[0] + w / 2, c[1] + h / 2)
                inside = (
                    box[0] >= 0 and box[1] >= 0 and box[2] <= width and box[3] <= height
                )
                hit_markers = int(
                    pp.box_overlaps_circles(box, occupied, marker_radius).sum()
                )
                hit_labels = sum(pp.boxes_overlap(box, b) for b in placed)
                score = (
                    10 * hit_labels + 6 * hit_markers + (0 if inside else 200) + 2.5 * ring
                )
                if score < best_score:
                    best, best_score = (c, box, ring), score
            if best_score < 2.5 * (ring + 1):
                break
        c, box, ring = best
        placed.append(box)
        results[index] = (c, ring)

    for i, ((c, ring), text) in enumerate(zip(results, labels)):
        y, x = points[i]
        dx, dy = c - markers[i]
        annotation = ax.annotate(
            text,
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=fontsize,
            weight="bold",
            color=label_colors[i] if label_colors is not None else "k",
            zorder=10,
            bbox=LABEL_BBOX if halo_color is None else None,
            arrowprops=(
                dict(arrowstyle="-", color="0.45", lw=0.6, shrinkA=0, shrinkB=3)
                if ring > 0
                else None
            ),
        )
        if halo_color is not None:
            from matplotlib import patheffects

            annotation.set_path_effects([patheffects.withStroke(linewidth=2.5, foreground=halo_color)])


def label_records(ax, extent, records, extra_points=(), extra_labels=(), obstacles=None):
    multi = len(records) > 1
    points, labels = [], []
    for record in records:
        for p, number in zip(record.observed, record.numbers):
            points.append(p)
            labels.append(image_label_from(record, number, multi))
    points += list(extra_points)
    labels += list(extra_labels)
    place_labels(
        ax,
        extent,
        points,
        labels,
        obstacles=obstacles,
        fontsize=label_fontsize_from(records),
    )


def legend_location_from(extent, points) -> str:
    """
    The emptiest corner of a panel for its legend (the source-plane panel's corner ranking), so a legend never lands
    over the images, the model or the mass centres.
    """
    from autolens.point.plot import source_plane_panel as spp

    points = np.asarray(points, dtype=float).reshape(-1, 2)
    (corner, _), *_ = spp.ranked_corners_from(extent, points, size=0.4)
    return spp.CORNER_LOCATIONS[corner]


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------


def scatter_sources(ax, records, edgecolor="k", edgewidth=0.6):
    for record in records:
        if len(record.observed) == 0:
            continue
        ax.scatter(
            record.observed[:, 1],
            record.observed[:, 0],
            s=marker_size_from(record.marker),
            c=[record.color],
            marker=record.marker,
            edgecolors=edgecolor,
            linewidths=edgewidth,
            zorder=5,
            label="Observed",
        )


def draw_noise_circles(ax, records):
    from matplotlib.patches import Circle

    for record in records:
        for (y, x), sigma in zip(record.observed, record.sigma):
            if np.isfinite(sigma) and sigma > 0:
                ax.add_patch(
                    Circle(
                        (x, y),
                        sigma,
                        fill=False,
                        edgecolor=pp.NOISE_COLOR,
                        linewidth=1.0,
                        zorder=5.5,
                    )
                )


def draw_curves(ax, records, curves: Dict, single_color=None):
    """
    Draws each source's critical curves at its own redshift: in the source's (darkened) colour when there are
    several sources, dark grey (radial dark olive, dotted) for one. Beyond 8 sources only tangential curves are
    drawn, thinner.
    """
    multi = len(records) > 1
    many = is_many(records)
    drawn = set()
    for record in records:
        plane_curves = record.curves_from(curves)
        if plane_curves is None:
            continue
        key = (str(record.curve_key), record.plane_index, str(record.color))
        if key in drawn:
            continue
        drawn.add(key)
        color = single_color or (darken(record.color) if multi else CURVE_COLOR)
        for line in plane_curves.tangential:
            ax.plot(
                line[:, 1],
                line[:, 0],
                color=color,
                linewidth=CURVE_LINEWIDTH if not many else 1.3,
                zorder=2,
            )
        if many:
            continue
        for line in plane_curves.radial:
            ax.plot(
                line[:, 1],
                line[:, 0],
                color=color if multi else RADIAL_CURVE_COLOR,
                linewidth=RADIAL_CURVE_LINEWIDTH,
                linestyle=":",
                zorder=2,
            )


def draw_model_and_residuals(
    ax, records, model_size=pp.MODEL_MARKER_SIZE, unmatched_color=pp.UNMATCHED_COLOR
):
    """
    The model positions as red ✕ at their true position, each joined to its observed image by a true-length red
    residual line (solid for hard pairing, dashed for the soft all-pairs mixture), unmatched model images as hollow
    ✕ and a dashed ring round observed images with no model image of their own.
    """
    for record in records:
        if not record.has_model:
            continue
        for obs, index in zip(record.observed, record.indexes):
            if index < 0:
                continue
            m = record.model[index]
            ax.plot(
                [obs[1], m[1]],
                [obs[0], m[0]],
                color=pp.MODEL_COLOR,
                linewidth=1.2,
                linestyle=record.linestyle,
                alpha=0.9,
                zorder=6,
            )
        matched = record.model[~record.unmatched]
        if len(matched):
            ax.scatter(
                matched[:, 1],
                matched[:, 0],
                s=model_size,
                c=pp.MODEL_COLOR,
                marker="x",
                linewidths=1.6 if model_size >= pp.MODEL_MARKER_SIZE else 1.2,
                zorder=8,
                label="Model",
            )
        unmatched = record.model[record.unmatched]
        if len(unmatched):
            ax.scatter(
                unmatched[:, 1],
                unmatched[:, 0],
                s=2 * model_size,
                facecolors="none",
                edgecolors=unmatched_color,
                marker="X",
                linewidths=1.5,
                zorder=7,
                label="Unmatched model image",
            )
        if np.any(record.no_model):
            no_model = record.observed[record.no_model]
            ax.scatter(no_model[:, 1], no_model[:, 0], zorder=6.5, **NO_MODEL_RING)


def merit_label_from(records) -> Optional[str]:
    """
    The fit's figure of merit summed over the sources: ``χ² = …``, or ``−2 ln L = …`` when the fits are the
    all-pairs mixture (``FitPositionsImagePairAll``), whose value is ``−2`` times a mixture log likelihood.
    """
    from autolens.point.plot import source_plane_panel as spp

    values = [r.chi_squared for r in records if r.chi_squared is not None]
    if not values:
        return None
    kinds = {r.mixture for r in records if r.chi_squared is not None}
    if len(kinds) > 1:
        return None
    name = "−2 ln L" if True in kinds else "χ²"
    return f"{name} = {spp._format_chi_squared(float(np.sum(values)))}"


def observed_handles(records, edgecolor="k"):
    """
    One ``Observed (σ = …)`` entry for a single source or more than 8 sources; for 2-8 sources one entry per source,
    report the family's redshift and fitted statistic, or its positional noise for dataset figures.
    """
    from matplotlib.lines import Line2D

    sigma_text = pp.sigma_label_from(np.concatenate([r.sigma for r in records]))
    if 1 < len(records) <= ps.MAX_QUALITATIVE:
        handles = []
        for record in records:
            z = f"z = {record.redshift:g}, " if record.redshift is not None else ""
            value = pp.sigma_label_from(record.sigma)
            if record.chi_squared is not None:
                from autolens.point.plot import source_plane_panel as spp

                statistic = _cluster_statistic(record)
                value = f"{statistic} = {spp._format_chi_squared(record.chi_squared)}"
            handles.append(
                Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker=record.marker,
                    markerfacecolor=record.color,
                    markeredgecolor=edgecolor,
                    markeredgewidth=0.6,
                    markersize=9,
                    label=f"{record.name} ({z}{value})",
                )
            )
        return handles
    face = "k" if len(records) == 1 else "0.6"
    return [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markerfacecolor=face,
            markeredgecolor=edgecolor,
            markersize=9,
            label=f"Observed ({sigma_text})",
        )
    ]


def fit_handles(records, model_size, has_curves: bool, has_mass_centres: bool, curve_label=None):
    from matplotlib.lines import Line2D

    first = records[0]
    multi = len(records) > 1
    handles = []
    if any(r.has_model for r in records):
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker="x",
                color=pp.MODEL_COLOR,
                markeredgewidth=1.6,
                markersize=8 if model_size >= pp.MODEL_MARKER_SIZE else 6,
                label=first.model_label,
            )
        )
        handles.append(
            Line2D(
                [],
                [],
                color=pp.MODEL_COLOR,
                linewidth=1.2,
                linestyle=first.linestyle,
                label=first.residual_label,
            )
        )
    if any(np.any(r.unmatched) for r in records):
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker="X",
                markerfacecolor="none",
                markeredgecolor=pp.UNMATCHED_COLOR,
                markeredgewidth=1.5,
                markersize=9,
                label="Unmatched model image",
            )
        )
    if any(np.any(r.no_model) for r in records):
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markerfacecolor="none",
                markeredgecolor="0.3",
                markeredgewidth=1.3,
                markersize=15,
                label="Observed, no model image",
            )
        )
    if has_curves:
        handles.append(
            Line2D(
                [],
                [],
                color=CURVE_COLOR if not multi else "0.35",
                linewidth=CURVE_LINEWIDTH,
                label=curve_label or ("Critical curve (each source's z)" if multi else "Critical curve"),
            )
        )
    if has_mass_centres:
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker="+",
                color=pp.MASS_CENTRE_COLOR,
                markeredgewidth=1.5,
                markersize=8,
                label="Mass centre",
            )
        )
    merit = merit_label_from(records)
    if merit is not None:
        handles.append(Line2D([], [], linestyle="none", marker="none", label=merit))
    return handles


def add_colourbar(fig, ax, colormap, records, scale: float = 1.0):
    """
    The source-redshift colourbar of the colour-by-redshift scheme (beyond 8 sources), with a tick line at every
    source's redshift.
    """
    if colormap is None:
        return None
    import matplotlib.cm as cm
    from autoarray.plot.utils import conf_mat_plot_fontsize

    cmap, norm = colormap
    position = ax.get_position()
    cax = fig.add_axes([position.x1 + 0.015, position.y0, 0.025, position.height])
    colorbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    fontsize = conf_mat_plot_fontsize("yticks", 22) * 0.8 * scale
    by_redshift = all(r.redshift is not None and np.isfinite(r.redshift) for r in records)
    colorbar.set_label("source redshift" if by_redshift else "family index", fontsize=fontsize)
    colorbar.ax.tick_params(labelsize=fontsize)
    for record in records:
        cax.axhline(record.redshift if by_redshift else record.k, color="k", linewidth=0.5, alpha=0.6)
    return cax


def _curve_points(records, curves) -> np.ndarray:
    points = []
    for record in records:
        plane_curves = record.curves_from(curves)
        if plane_curves is None:
            continue
        for line in plane_curves.tangential + plane_curves.radial:
            points.append(line[:: max(1, len(line) // 40)])
    return np.vstack(points) if points else np.zeros((0, 2))


# ---------------------------------------------------------------------------
# dataset_point.png: Dataset Positions
# ---------------------------------------------------------------------------


def draw_dataset_positions(
    ax, records, extent, title: str, scale: float = 1.0, family_index=None
):
    """
    Draws the "Dataset Positions" panel of ``records`` on ``ax`` (styles must be assigned).
    """
    if family_index is None:
        scatter_sources(ax, records)
        handles = observed_handles(records)
    else:
        from autolens.point.plot.cluster_sky import indexed_family_style

        record = records[0]
        _, record.color, record.marker = indexed_family_style(family_index)
        ax.scatter(
            *record.observed.T[::-1],
            marker=record.marker,
            facecolors="none",
            edgecolors=[record.color],
            linewidths=1.8,
            s=85,
        )
        handles = [_cluster_family_handle(record, record.color, False)]
    draw_noise_circles(ax, records)
    finish_panel(ax, title, extent, scale)
    if family_index is None:
        label_records(ax, extent, records)
    else:
        place_labels(
            ax,
            extent,
            record.observed,
            [f"{family_index}·{n}" for n in record.numbers],
            label_colors=[record.color] * len(record.numbers),
            halo_color="white",
        )
    observed = np.vstack([r.observed for r in records])
    ax.legend(
        handles=handles,
        loc=legend_location_from(extent, observed) if len(records) == 1 else "best",
        fontsize=LEGEND_KWARGS["fontsize"] * scale,
        framealpha=LEGEND_KWARGS["framealpha"],
        markerscale=LEGEND_KWARGS["markerscale"],
    )


def dataset_title_from(records, title_prefix=None) -> str:
    if len(records) == 1:
        return f"{_prefix(title_prefix)}Dataset Positions ({records[0].name})"
    return f"{_prefix(title_prefix)}Dataset Positions"


def plot_dataset_positions(
    datasets,
    solver=None,
    output_path: Optional[str] = None,
    output_format: str = None,
    filename: str = "dataset_point",
    title_prefix: str = None,
    image_extent=None,
    family_index=None,
):
    """
    Plots ``dataset_point.png`` ("Dataset Positions") of one or many ``PointDataset`` objects on the solver extent.

    Cluster families use outlined markers and family·image labels on the same sky field as the image
    overlay when an image extent is supplied. A standalone point dataset uses its own image numbers.
    """

    datasets = list(datasets)
    records = ps.records_from_datasets(datasets)
    colormap = ps.assign_styles(records)
    extent = ps.union_extent_from(records, solver=solver)

    if len(records) > 1 or family_index is not None:
        fig = _cluster_overview(
            records,
            {},
            image_extent if image_extent is not None else extent,
            colormap,
            title_prefix,
            family_index=family_index,
        )
    else:
        fig, ax = subplots(1, 1, figsize=figsize_from(records))
        draw_dataset_positions(
            ax,
            records,
            extent,
            title=dataset_title_from(records, title_prefix),
            scale=font_scale_from(records),
        )
    save_figure(fig, path=output_path, filename=filename, format=output_format)


# ---------------------------------------------------------------------------
# fit.png: Fit Positions
# ---------------------------------------------------------------------------


def draw_fit_positions(
    ax,
    records,
    curves,
    extent,
    tracer,
    title: str,
    scale: float = 1.0,
    legend: str = "inside",
    colormap=None,
    family_index=None,
):
    """
    Draws the "Fit Positions" panel of ``records`` (with fits) on ``ax``; ``legend`` is ``"inside"`` (the emptiest
    corner) or ``"outside"`` (right of the panel, and of the colourbar if there is one).
    """
    multi = len(records) > 1
    many = is_many(records)
    model_size = model_size_from(records)

    draw_curves(ax, records, curves)
    mass_centres = pp.mass_centres_from(tracer) if tracer is not None else np.zeros((0, 2))
    if len(mass_centres):
        pp.plot_mass_centres(
            ax, tracer, size=70 if not many else 50, linewidths=1.4
        )
    if family_index is None:
        scatter_sources(ax, records)
    else:
        from autolens.point.plot.cluster_sky import indexed_family_style

        _, family_color, family_marker = indexed_family_style(family_index)
        (record,) = records
        ax.scatter(
            *record.observed.T[::-1],
            s=marker_size_from(family_marker),
            marker=family_marker,
            facecolors="none",
            edgecolors=family_color,
            linewidths=1.8,
            zorder=5,
        )
    draw_noise_circles(ax, records)
    draw_model_and_residuals(ax, records, model_size)
    finish_panel(ax, title, extent, scale)

    extra_points, extra_labels = [], []
    for record in records:
        for p, letter in zip(record.model[record.unmatched], record.unmatched_labels):
            extra_points.append(p)
            extra_labels.append(f"{record.k}·{letter}" if multi else letter)
    models = [r.model for r in records if r.has_model]
    obstacles = np.vstack(models + [mass_centres]) if models else mass_centres
    if family_index is None:
        label_records(ax, extent, records, extra_points, extra_labels, obstacles=obstacles)
    else:
        labels = [f"{family_index}·{n}" for n in [*record.numbers, *extra_labels]]
        place_labels(
            ax,
            extent,
            [*record.observed, *extra_points],
            labels,
            obstacles=obstacles,
            fontsize=label_fontsize_from(records),
            label_colors=[family_color] * len(labels),
            halo_color="white",
        )

    handles = observed_handles(records) + fit_handles(
        records,
        model_size,
        has_curves=bool(_curve_points(records, curves).size),
        has_mass_centres=bool(len(mass_centres)),
    )
    if family_index is not None:
        handles[0].set_marker(family_marker)
        handles[0].set_markerfacecolor("none")
        handles[0].set_markeredgecolor(family_color)
        handles[0].set_markeredgewidth(1.8)
    kwargs = dict(
        fontsize=LEGEND_KWARGS["fontsize"] * scale,
        framealpha=LEGEND_KWARGS["framealpha"],
        markerscale=LEGEND_KWARGS["markerscale"],
    )
    if legend == "outside":
        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.02 if colormap is None else 1.16, 1.0),
            borderaxespad=0.0,
            **kwargs,
        )
    else:
        drawn = np.vstack(
            [r.observed for r in records]
            + models
            + [mass_centres, _curve_points(records, curves)]
        )
        ax.legend(handles=handles, loc=legend_location_from(extent, drawn), **kwargs)


def _cluster_curves(ax, records, curves, on_image, fallback_lines):
    """Return the curve key for the curves actually drawn on this map."""
    has_curves = False
    for r in records:
        own = r.curves_from(curves)
        if own is None and curves:
            own = curves.get(("source", r.k))
        if own is not None:
            for line in own.tangential:
                ax.plot(
                    line[:, 1], line[:, 0], color=r.color, lw=0.95, alpha=0.8, zorder=2
                )
                has_curves = True
            if not on_image and not is_many(records):
                for line in own.radial:
                    ax.plot(
                        line[:, 1],
                        line[:, 0],
                        color=r.color,
                        lw=0.85,
                        ls=":",
                        alpha=0.8,
                        zorder=2,
                    )
                    has_curves = True
    if has_curves:
        return "Critical curve (each source's z)"
    if fallback_lines is not None and len(fallback_lines):
        for line in fallback_lines:
            line = np.asarray(line)
            ax.plot(line[:, 1], line[:, 0], color="white", lw=0.95, zorder=2)
        return "Critical curve (last source plane)"
    return None


def _cluster_statistic(record):
    from autolens.point.plot import source_plane_panel as spp

    if record.mixture:
        return "−2 ln L"
    plane = spp.presentation_from(record.fit.positions)
    return "Source-plane χ²" if plane.is_source_plane else "Image-plane χ²"


def _cluster_family_handle(record, ink, compact, mixed_statistics=False):
    from matplotlib.lines import Line2D
    from autolens.point.plot import source_plane_panel as spp

    details = []
    equals = "=" if compact else " = "
    if record.redshift is not None:
        details.append(f"z{equals}{record.redshift:g}")
    if record.chi_squared is not None:
        statistic = _cluster_statistic(record)
        if compact and not record.mixture and not mixed_statistics:
            statistic = "χ²"
        details.append(
            f"{statistic}{equals}{spp._format_chi_squared(record.chi_squared)}"
        )
    else:
        count = len(record.observed)
        details.append(f"{count} image" if count == 1 else f"{count} images")
        if not compact and record.fit is None:
            details.append(pp.sigma_label_from(record.sigma))
    label = (
        f"{record.name}  " + "  ".join(details)
        if compact
        else f"{record.name} ({', '.join(details)})"
    )
    return Line2D(
        [],
        [],
        linestyle="none",
        marker=record.marker,
        markersize=8,
        markerfacecolor="none",
        markeredgecolor=ink,
        markeredgewidth=1.8,
        label=label,
    )


def _cluster_overview(
    records,
    curves,
    field_extent,
    colormap,
    title_prefix,
    image=None,
    fallback_lines=None,
    family_index=None,
):
    """Draw a cluster dataset or fit on one shared sky field, with family and symbol keys outside the data."""
    from matplotlib import patheffects
    from autolens.point.plot.cluster_sky import (
        angular_axes,
        cluster_field,
        draw_cluster_image,
        family_style,
        indexed_family_style,
    )

    on_image = image is not None
    is_fit = records[0].fit is not None
    many = is_many(records)
    models = [r.model for r in records if r.has_model]
    centre, half = cluster_field(
        field_extent, np.vstack([r.observed for r in records] + models)
    )
    # Leave one line per family above the map, and space below it for the fit symbols.
    statistics = {_cluster_statistic(r) for r in records if r.chi_squared is not None}
    mixed_statistics = len(statistics) > 1
    family_columns = 1
    if len(records) > 4:
        family_columns = 4 if many and not mixed_statistics else 2
    family_rows = (len(records) + family_columns - 1) // family_columns
    side = 9.0 if many else 6.8
    header = 0.52 + 0.25 * family_rows + (0.24 if many else 0)
    footer = 1.05 if is_fit else 0.58
    height = side + header + footer
    width = side + (1.75 if many else 0.9)
    fig, ax = subplots(figsize=(width, height))
    fig.subplots_adjust(
        left=0.57 / width,
        right=(0.57 + side) / width,
        top=(footer + side) / height,
        bottom=footer / height,
    )
    if on_image:
        draw_cluster_image(ax, image)
    angular_axes(ax, centre, half)
    extent = [*ax.get_xlim(), *ax.get_ylim()]
    title = "Cluster fit on the observed image" if on_image else "Cluster fit overview"
    if not is_fit:
        title = (
            "Cluster positions on the observed image"
            if on_image
            else "Cluster dataset overview"
        )
    ax.set_title(
        f"{_prefix(title_prefix)}{title}",
        x=0,
        ha="left",
        y=1 + (0.23 * family_rows + 0.17 + (0.12 if many else 0)) / side,
        fontsize=19,
        weight="bold",
        color="#172f49",
    )
    points, labels, colors, handles = [], [], [], []
    for r in records:
        if many:
            ink = r.color
        else:
            bright, ink, r.marker = (
                family_style(r)
                if family_index is None
                else indexed_family_style(family_index)
            )
            r.color = bright if on_image else ink
        label_index = r.k if family_index is None else family_index
        handles.append(_cluster_family_handle(r, ink, many, mixed_statistics))
        observed = ax.scatter(
            *r.observed.T[::-1],
            s=marker_size_from(r.marker, base=85),
            marker=r.marker,
            facecolors="none",
            edgecolors=[r.color],
            linewidths=1.8,
            label="Observed",
            zorder=5,
        )
        if on_image:
            observed.set_path_effects(
                [patheffects.withStroke(linewidth=3, foreground="#303038")]
            )
        points.extend(r.observed)
        labels.extend(f"{label_index}·{n}" for n in r.numbers)
        colors.extend([r.color] * len(r.numbers))
        points.extend(r.model[r.unmatched])
        labels.extend(f"{label_index}·{letter}" for letter in r.unmatched_labels)
        colors.extend([r.color] * len(r.unmatched_labels))
    curve_key = (
        _cluster_curves(ax, records, curves, on_image, fallback_lines)
        if is_fit
        else None
    )

    centres = np.empty((0, 2))
    if is_fit and not on_image:
        tracer = records[0].fit.tracer
        centres = pp.mass_centres_from(tracer)
        pp.plot_mass_centres(ax, tracer, size=40, linewidths=1)
    if not on_image:
        draw_noise_circles(ax, records)
    size = model_size_from(records)
    draw_model_and_residuals(
        ax, records, size, unmatched_color=".80" if on_image else pp.UNMATCHED_COLOR
    )
    place_labels(
        ax,
        extent,
        points,
        labels,
        obstacles=np.vstack([centres] + models),
        fontsize=9 if many else 10,
        label_colors=(
            ["white" if on_image else "#172f49"] * len(labels) if many else colors
        ),
        halo_color="#23282d" if on_image else "white",
    )

    legend_title = None
    highlight = None
    if many:
        image_count = sum(len(r.observed) for r in records)
        legend_title = f"{len(records)} families    ·    {image_count} observed images"
        if len(statistics) == 1:
            legend_title = f"{next(iter(statistics))}    ·    {legend_title}"
            finite = [
                i for i, r in enumerate(records)
                if r.chi_squared is not None and np.isfinite(r.chi_squared)
            ]
            if finite:
                highlight = max(finite, key=lambda i: records[i].chi_squared)
    legend = ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0, 1.006, 1, 0),
        mode="expand" if many else None,
        title=legend_title,
        alignment="left",
        frameon=False,
        borderaxespad=0,
        fontsize=9 if many else 10,
        handletextpad=0.4 if many else 0.8,
        handlelength=1.25,
        ncol=family_columns,
    )
    if highlight is not None:
        legend.get_texts()[highlight].set_weight("bold")
    if is_fit:
        symbols = fit_handles(
            records, size, bool(curve_key), bool(len(centres)), curve_label=curve_key
        )
        fig.legend(
            handles=symbols,
            loc="upper center",
            bbox_to_anchor=((0.57 + side / 2) / width, (footer - 0.33) / height),
            ncol=min(3, len(symbols)) or 1,
            fontsize=9,
            frameon=False,
            columnspacing=1.4,
        )
    add_colourbar(fig, ax, colormap, records, 0.65)
    return fig


def plot_fit_positions(
    fits,
    solver=None,
    curves: Optional[Dict] = None,
    output_path: Optional[str] = None,
    output_format: str = None,
    filename: str = "fit",
    title_prefix: str = None,
    image_extent=None,
):
    """
    Plots ``fit.png`` ("Fit Positions") of one or many ``FitPointDataset`` objects of the same lens model on the
    solver extent (see the module docstring). ``curves`` are the per-plane curves of
    :func:`~autolens.point.plot.point_sources.curves_for_planes`, computed if ``None``.
    For a cluster, pass ``image_extent`` to match the image-backed overview's sky field.
    """

    fits = list(fits)
    records = ps.records_from_fits(fits)
    colormap = ps.assign_styles(records)
    solver = solver if solver is not None else getattr(fits[0], "solver", None)
    extent = ps.union_extent_from(records, solver=solver)
    tracer = fits[0].tracer
    if curves is None:
        curves = ps.curves_for_fits(fits, records, extent)

    if len(records) > 1:
        fig = _cluster_overview(
            records,
            curves,
            image_extent if image_extent is not None else extent,
            colormap,
            title_prefix,
        )
    else:
        fig, ax = subplots(1, 1, figsize=figsize_from(records))
        draw_fit_positions(
            ax,
            records,
            curves,
            extent,
            tracer,
            title=f"{_prefix(title_prefix)}Fit Positions",
        )
    save_figure(fig, path=output_path, filename=filename, format=output_format)


# ---------------------------------------------------------------------------
# fit_per_image.png
# ---------------------------------------------------------------------------


def zoom_extent_from(record, min_half: float = 1.0, margin: float = 0.28):
    """
    A square image-plane zoom on one source's observed (and model) images.
    """
    points = np.vstack([record.observed, record.model]) if record.has_model else record.observed
    lo, hi = points.min(axis=0), points.max(axis=0)
    half = max(0.5 * float(np.max(hi - lo)) * (1 + 2 * margin), min_half)
    centre_y, centre_x = 0.5 * (lo + hi)
    return [centre_x - half, centre_x + half, centre_y - half, centre_y + half]


def _format_sigma(value: float) -> str:
    if value >= 100:
        return f"{value:,.0f}σ"
    return f"{value:.1f}σ" if value >= 1 else f"{value:.2f}σ"


def draw_ranking_panel(
    ax, record, scales: ps.RankingScales, title: str, family_index=None
):
    """
    The residual ranking of one source: a dot per image (worst first) at Δ/σ on the shared linear range, its value
    printed to its right; an off-scale image is a ▸ at the edge with its value in bold. Unmatched model images are
    listed last with their magnification. Unlabelled dashed lines mark 1σ and 3σ; there is no x axis.
    """
    from autoarray.plot.utils import apply_labels

    family_color, family_marker = None, "o"
    if family_index is not None:
        from autolens.point.plot.cluster_sky import indexed_family_style

        _, family_color, family_marker = indexed_family_style(family_index)
    xmax, n_slots = scales.xmax, scales.n_slots
    rows = ps.ranking_rows_from(record)
    right = 1.32 * xmax
    for slot, (value, _, kind, i) in enumerate(rows):
        y = n_slots - 1 - slot
        if kind == "unmatched":
            ax.scatter(
                0.0,
                y,
                s=130,
                facecolors="none",
                edgecolors=pp.UNMATCHED_COLOR,
                marker="X",
                linewidths=1.6,
                zorder=4,
            )
            mu = record.mu_model[i] if i < len(record.mu_model) else np.nan
            ax.text(0.04 * right, y, f"μ = {mu:.2g}", va="center", fontsize=14, color="0.3")
            continue
        if not np.isfinite(value):
            continue
        color = family_color or ps.image_color(record.numbers[i])
        x = min(value, xmax)
        ax.hlines(y, 0, x, color="0.85", linewidth=1.4, zorder=1)
        if value > xmax:
            ax.scatter(xmax, y, s=150, c=[color], marker=">", edgecolors="k", linewidths=0.6, zorder=4)
        else:
            ax.scatter(
                x,
                y,
                s=120,
                c=[color],
                marker=family_marker,
                edgecolors="k",
                linewidths=0.6,
                zorder=4,
            )
        if kind == "nomodel":
            ax.scatter(x, y, zorder=3, **{**NO_MODEL_RING, "s": 380})
        ax.text(
            x + 0.035 * right,
            y,
            _format_sigma(value),
            va="center",
            ha="left",
            fontsize=14,
            weight="bold" if value > xmax else "normal",
        )
    ax.axvline(1.0, color="0.35", linewidth=1.3, linestyle="--", zorder=0)
    ax.axvline(3.0, color="#c25400", linewidth=1.5, linestyle="--", zorder=0)
    ax.set_xlim(-0.04 * right, right)
    ax.set_ylim(-0.7, n_slots - 0.3)
    apply_labels(ax, title=title, xlabel="", ylabel="")
    labels = [t[1] for t in rows]
    if family_index is not None:
        labels = [f"{family_index}·{label}" for label in labels]
    labels += [""] * (n_slots - len(rows))
    ax.set_yticks(np.arange(n_slots)[::-1])
    ax.set_yticklabels(labels, fontweight="bold")
    ax.tick_params(axis="y", labelrotation=0, length=0)
    for tick, row in zip(ax.get_yticklabels(), rows):
        if row[2] == "unmatched":
            tick.set_color("0.4")
    ax.xaxis.set_visible(False)
    ax.set_box_aspect(1)


def draw_image_plane_panel(ax, record, curves, tracer, title: str, family_index=None):
    """
    The image-plane zoom of one source: its own critical curves (dark grey, radial dotted olive), the mass centres,
    its observed images coloured by image with their σ circles, the model ✕ and residual lines and its image numbers
    (unmatched model images lettered).
    """
    extent = zoom_extent_from(record)
    plane_curves = record.curves_from(curves)
    if plane_curves is not None:
        for line in plane_curves.tangential:
            ax.plot(line[:, 1], line[:, 0], color=CURVE_COLOR, linewidth=CURVE_LINEWIDTH, zorder=2)
        for line in plane_curves.radial:
            ax.plot(
                line[:, 1],
                line[:, 0],
                color=RADIAL_CURVE_COLOR,
                linewidth=RADIAL_CURVE_LINEWIDTH,
                linestyle=":",
                zorder=2,
            )
    if tracer is not None:
        pp.plot_mass_centres(ax, tracer, size=80, linewidths=1.5)
    if family_index is None:
        marker_style = dict(
            c=[ps.image_color(n) for n in record.numbers],
            edgecolors="k",
            linewidths=0.6,
        )
    else:
        from autolens.point.plot.cluster_sky import indexed_family_style

        _, color, marker = indexed_family_style(family_index)
        marker_style = dict(
            marker=marker, facecolors="none", edgecolors=color, linewidths=1.8
        )
    ax.scatter(*record.observed.T[::-1], s=MARKER_SIZE, zorder=5, **marker_style)
    draw_noise_circles(ax, [record])
    draw_model_and_residuals(ax, [record])
    finish_panel(ax, title, extent)
    points = list(record.observed)
    labels = [f"{n}" for n in record.numbers]
    if np.any(record.unmatched):
        points += list(record.model[record.unmatched])
        labels += record.unmatched_labels
    if family_index is not None:
        labels = [f"{family_index}·{label}" for label in labels]
    place_labels(
        ax,
        extent,
        points,
        labels,
        obstacles=record.model if record.has_model else None,
        fontsize=pp.FIT_NUMBER_FONTSIZE,
        label_colors=None if family_index is None else [color] * len(labels),
        halo_color=None if family_index is None else "white",
    )


def draw_source_plane_panel(ax, record, curves, title: str, family_index=None):
    from autolens.point.plot import source_plane_panel as spp

    numbers = record.numbers
    colors = [ps.image_color(n) for n in numbers]
    marker = "o"
    if family_index is not None:
        from autolens.point.plot.cluster_sky import indexed_family_style

        _, color, marker = indexed_family_style(family_index)
        colors = [color] * len(numbers)
        numbers = [f"{family_index}·{n}" for n in numbers]
    data = spp.source_plane_data_from(record.fit.positions, numbers=numbers)
    plane_curves = record.curves_from(curves)
    spp.plot_source_plane_panel(
        ax,
        data,
        title=title,
        tangential_caustics=plane_curves.tangential_caustics if plane_curves else None,
        radial_caustics=plane_curves.radial_caustics if plane_curves else None,
        point_colors=colors,
        point_marker=marker,
        legend=False,
    )


def _shrink_titles_to_fit(fig, axes):
    """
    Shrinks any title wider than its panel until it fits (it is measured as rendered).
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes:
        width = ax.get_window_extent(renderer).width
        title = ax.title
        while (
            title.get_window_extent(renderer).width > 1.02 * width
            and title.get_fontsize() > 12
        ):
            title.set_fontsize(title.get_fontsize() - 1)


def _fit_per_image_figure(
    records, curves, tracer, scales, title_prefix=None, family_index=None
):
    from autoarray.plot.utils import conf_subplot_figsize

    n = len(records)
    fig, axes = subplots(n, 3, figsize=conf_subplot_figsize(n, 3), squeeze=False)
    prefix = _prefix(title_prefix)
    for row, record in enumerate(records):
        tag = f"({record.name}{_redshift_text(record)})"
        draw_ranking_panel(
            axes[row, 0],
            record,
            scales,
            title=f"{prefix}Residuals {tag}",
            family_index=family_index,
        )
        draw_image_plane_panel(
            axes[row, 1],
            record,
            curves,
            tracer,
            title=f"{prefix}IP {tag}",
            family_index=family_index,
        )
        try:
            draw_source_plane_panel(
                axes[row, 2],
                record,
                curves,
                title=f"{prefix}SP {tag}",
                family_index=family_index,
            )
        except Exception:
            logger.warning(
                "The source-plane panel of fit_per_image.png could not be drawn; it is left empty.",
                exc_info=True,
            )
            axes[row, 2].cla()
            axes[row, 2].set_title(f"{prefix}SP {tag}")
            axes[row, 2].set_axis_off()
        for ax in axes[row]:
            ax.set_box_aspect(1)
    fig.tight_layout()
    _shrink_titles_to_fit(fig, axes.ravel())
    return fig


def subplot_fit_per_image(
    fits,
    curves: Optional[Dict] = None,
    solver=None,
    output_path: Optional[str] = None,
    output_format: str = None,
    filename: str = "fit_per_image",
    title_prefix: str = None,
    scales: Optional[ps.RankingScales] = None,
    family_index=None,
):
    """
    Plots ``fit_per_image.png``: one row per source of "Residuals", "IP" and "SP" panels (see the module docstring).

    The ranking range is shared by every row; it is computed from all ``fits`` (pass ``scales`` to share it with
    other figures). Beyond 8 sources each source is written to its own file, ``<output_path>/<filename>/<name>.png``,
    all on the shared range.
    """

    fits = list(fits)
    records = ps.records_from_fits(fits)
    ps.assign_styles(records)
    tracer = fits[0].tracer
    scales = scales or ps.ranking_scales_from(records)
    if curves is None:
        solver = solver if solver is not None else getattr(fits[0], "solver", None)
        curves = ps.curves_for_fits(
            fits, records, ps.union_extent_from(records, solver=solver)
        )

    if not is_many(records):
        fig = _fit_per_image_figure(
            records, curves, tracer, scales, title_prefix, family_index
        )
        save_figure(fig, path=output_path, filename=filename, format=output_format)
        return

    from pathlib import Path

    folder = str(Path(output_path) / filename) if output_path else None
    for record in records:
        fig = _fit_per_image_figure([record], curves, tracer, scales, title_prefix)
        save_figure(fig, path=folder, filename=record.name, format=output_format)


# ---------------------------------------------------------------------------
# image_with_points.png / fit_image_with_points.png
# ---------------------------------------------------------------------------


# An image wider than this (arc-seconds) is cluster scale: a bright BCG and faint arcs need a log10 colour scale.
CLUSTER_IMAGE_WIDTH = 20.0


def use_log10_from(image, multi: bool) -> bool:
    """
    Clusters (several sources, or a cluster-scale image even when one source is drawn) use a log10 colour scale;
    a galaxy-scale lens keeps the library's linear scale.
    """
    if multi:
        return True
    try:
        width = float(np.max(np.asarray(image.shape_native)) * np.max(image.pixel_scales))
    except Exception:
        return False
    return width > CLUSTER_IMAGE_WIDTH


def plot_image_with_points(
    image,
    datasets=None,
    fits=None,
    curves: Optional[Dict] = None,
    solver=None,
    output_path: Optional[str] = None,
    output_format: str = None,
    title_prefix: str = None,
    image_plane_lines=None,
    image_plane_line_colors=None,
    family_index=None,
):
    """
    Plots the observed image with every source's positions overlaid, in the symbols of the positions figures.

    With ``fits=None`` only the observed positions of ``datasets`` are drawn (``image_with_points.png``); with
    ``fits`` the model positions, residual lines and each source's tangential critical curves at its own redshift are
    added (``fit_image_with_points.png``, so both figures coexist). Cluster overviews use a grayscale asinh
    stretch and an external legend for every family.
    Standalone images use the point-source intensity scale.

    ``image_plane_lines`` (pre-computed last-plane curves) are drawn only when ``curves`` cannot be computed.
    """
    from autoarray.plot.utils import _overlay_yx_for_origin
    from autogalaxy.util.plot_utils import plot_array

    is_fit = fits is not None
    if is_fit:
        fits = list(fits)
        records = ps.records_from_fits(fits)
    else:
        records = ps.records_from_datasets(list(datasets))
    colormap = ps.assign_styles(records)
    multi = len(records) > 1
    many = is_many(records)
    scale = font_scale_from(records)

    if is_fit and curves is None:
        solver = solver if solver is not None else getattr(fits[0], "solver", None)
        curves = ps.curves_for_fits(
            fits, records, ps.union_extent_from(records, solver=solver)
        )

    if multi or family_index is not None:
        fig = _cluster_overview(
            records,
            curves,
            image.geometry.extent,
            colormap,
            title_prefix,
            image=image,
            fallback_lines=image_plane_lines,
            family_index=family_index,
        )
        save_figure(
            fig,
            path=output_path,
            filename="fit_image_with_points" if is_fit else "image_with_points",
            format=output_format,
        )
        return

    name = "Fit Image With Points" if is_fit else "Image With Points"
    title = f"{_prefix(title_prefix)}{name}" + (f" ({records[0].name})" if not multi else "")

    fig, ax = subplots(1, 1, figsize=figsize_from(records))
    plot_array(array=image, ax=ax, use_log10=use_log10_from(image, multi), title=title)
    scale_fonts(ax, scale)
    for other in fig.axes[1:]:
        for tick in other.get_yticklabels():
            tick.set_fontsize(tick.get_fontsize() * scale)

    # Positions are (y,x) arc-seconds: reflect them exactly as `plot_array` does its own overlays, using the
    # extent and origin of the raster it drew, so they stay registered to the image (PyAutoArray#565).
    raster_extent, origin = (
        (ax.images[0].get_extent(), ax.images[0].origin) if ax.images else (None, None)
    )

    def _yx(grid):
        grid = np.asarray(grid, dtype=float).reshape(-1, 2)
        if len(grid) == 0:
            return grid
        return _overlay_yx_for_origin(grid, raster_extent, origin)

    import copy

    shown = []
    for record in records:
        displayed = copy.copy(record)
        displayed.observed = _yx(record.observed)
        displayed.model = _yx(record.model)
        shown.append(displayed)

    from matplotlib import patheffects

    curve_width = CURVE_LINEWIDTH if not many else 1.3
    if is_fit:
        drawn_any = False
        for record in records:
            plane_curves = record.curves_from(curves)
            if plane_curves is None:
                continue
            color = "w" if (not multi or str(record.color) == "k") else record.color
            for line in plane_curves.tangential:
                line = _yx(line)
                ax.plot(
                    line[:, 1],
                    line[:, 0],
                    color=color,
                    linewidth=curve_width,
                    zorder=3,
                    path_effects=[
                        patheffects.withStroke(linewidth=curve_width + 1.6, foreground="k")
                    ],
                )
                drawn_any = True
        if not drawn_any and image_plane_lines:
            for line in image_plane_lines:
                line = _yx(line)
                ax.plot(line[:, 1], line[:, 0], color="w", linewidth=curve_width, zorder=3)

    scatter_sources(ax, shown, edgecolor="w", edgewidth=1.0)
    if is_fit:
        draw_model_and_residuals(ax, shown, model_size_from(records), unmatched_color="0.85")

    extent = [*ax.get_xlim(), *ax.get_ylim()]
    models = [s.model for s in shown if s.has_model]
    label_records(
        ax,
        extent,
        shown,
        obstacles=np.vstack(models) if models else None,
    )
    filename = "fit_image_with_points" if is_fit else "image_with_points"
    save_figure(fig, path=output_path, filename=filename, format=output_format)
