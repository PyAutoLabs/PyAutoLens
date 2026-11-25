"""Cluster position-fit diagnostics, one figure per multiply imaged source."""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

from autoarray.plot.utils import save_figure, subplots
from autolens.point.plot.cluster_sky import (
    angular_axes,
    cluster_field,
    draw_cluster_image,
    family_style,
)
from autolens.point.plot import point_sources as ps
from autolens.point.plot import positions_panel as pp
from autolens.point.plot.family_statistics import contribution_values
from autolens.point.plot.source_plane_panel import (
    SourcePlaneData,
    ellipse_from,
    source_plane_data_from,
)


@dataclass
class Family:
    record: ps.SourceRecord
    source: SourcePlaneData
    terms: np.ndarray
    penalty: float
    statistic: str
    data: np.ndarray
    predictions: np.ndarray
    footprints: np.ndarray  # semi-major, semi-minor, angle in degrees
    sky_color: object
    ink: object
    marker: str

    @classmethod
    def from_record(cls, record, by_redshift=False):
        source = source_plane_data_from(record.fit.positions, numbers=record.numbers)
        terms, penalty, statistic = contribution_values(record)
        if source.presentation.is_source_plane:
            data = source.traced
            predictions = np.broadcast_to(source.centre, data.shape)
            if source.presentation.weighting == "jacobian":
                footprints = [
                    ellipse_from(c) or (np.nan,) * 3 for c in source.covariances
                ]
            else:
                footprints = [(r, r, 0) for r in source.magnification_radii]
        else:
            data = record.observed
            predictions = np.full(data.shape, np.nan)
            paired = record.indexes >= 0
            predictions[paired] = record.model[record.indexes[paired]]
            footprints = [(r, r, 0) for r in record.sigma]
        return cls(
            record,
            source,
            terms,
            penalty,
            statistic,
            data,
            predictions,
            np.asarray(footprints).reshape(-1, 3),
            *(
                (record.color, record.color, record.marker)
                if by_redshift
                else family_style(record)
            ),
        )

    @property
    def labels(self):
        ids = self.record.fit.dataset.image_ids
        return self.record.numbers if ids is None else np.asarray(ids)

    def image_title(self, index):
        if self.record.fit.dataset.image_ids is None:
            return f"Image {self.record.numbers[index]}"
        return self.record.fit.dataset.image_ids[index]

    @property
    def ranked(self):
        return np.lexsort((self.record.numbers, -self.terms))

    @property
    def total(self):
        return self.terms.sum() + self.penalty


def _frame(ax, positions, aspect=1.0):
    positions = np.asarray(positions)
    finite = positions[np.isfinite(positions).all(axis=1)]
    if not len(finite):
        finite = np.zeros((1, 2))
    lo, hi = finite.min(axis=0), finite.max(axis=0)
    height = max(hi[0] - lo[0], (hi[1] - lo[1]) / aspect, 0.001) * 0.67
    angular_axes(ax, (lo + hi) / 2, height * aspect, height)


def _numbers(ax, positions, numbers, colors):
    valid = np.isfinite(positions).all(axis=1)
    positions, numbers = positions[valid], numbers[valid]
    colors = [c for c, keep in zip(colors, valid) if keep]
    extent = [*ax.get_xlim(), *ax.get_ylim()]
    width, height, scale = pp._panel_size_points(ax, extent)
    markers = (positions[:, ::-1] - [extent[0], extent[2]]) * scale
    labels = [str(n) for n in numbers]
    # Longer catalogue IDs may need a leader in a crowded cluster core.
    placements = pp.NUMBER_PLACEMENTS
    if numbers.dtype.kind in ("U", "O"):
        placements += tuple(
            (dx * radius, dy * radius, ha, va)
            for radius in (2, 3, 4)
            for dx, dy, ha, va in pp.NUMBER_PLACEMENTS
        )
    offsets = pp.adjacent_number_placements_from(
        markers, labels, width, height, fontsize=9, placements_to_try=placements
    )
    for (y, x), number, color, (dx, dy, ha, va) in zip(
        positions, labels, colors, offsets
    ):
        label = ax.annotate(
            number,
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            arrowprops=(
                dict(arrowstyle="-", color=color, lw=0.7, shrinkB=4)
                if (dx, dy, ha, va) not in pp.NUMBER_PLACEMENTS
                else None
            ),
            ha=ha,
            va=va,
            color=color,
            size=9,
            weight="bold",
        )
        label.set_path_effects(
            [patheffects.withStroke(linewidth=2, foreground=ax.get_facecolor())]
        )


def _overview(ax, families, family, image, indices):
    if image is not None:
        draw_cluster_image(ax, image)
    positions = np.vstack([f.record.observed for f in families])
    if image is None:
        _frame(ax, positions)
    else:
        centre, half = cluster_field(image.geometry.extent, positions)
        angular_axes(ax, centre, half)
    for other in families:
        record = other.record
        y, x = record.observed.T
        color = other.sky_color if image is not None else other.ink
        z = "" if record.redshift is None else f"    z = {record.redshift:g}"
        label = f"{record.name}{z}    {other.statistic} = {other.total:.2f}"
        handle = ax.scatter(
            x,
            y,
            s=78,
            marker=other.marker,
            facecolors="none",
            edgecolors=color,
            linewidths=1.8,
            label=label if len(families) <= 8 or family is other else "_nolegend_",
        )
        if image is not None:
            handle.set_path_effects(
                [
                    patheffects.Stroke(linewidth=3, foreground="#303038"),
                    patheffects.Normal(),
                ]
            )
        if other is family:
            colors = np.full(len(y), "#9ba5ae", dtype=object)
            colors[indices] = color
            _numbers(ax, record.observed, family.labels, colors)
    ax.set_title("Cluster fit overview", weight="bold")
    ax.legend(
        loc="best",
        fontsize=8,
        framealpha=0.92,
        facecolor="#30383f" if image is not None else "white",
        labelcolor="white" if image is not None else "#172f49",
    )


def _ranking(ax, family, highlight):
    order = family.ranked
    values = family.terms[order].tolist()
    labels = [family.image_title(i) for i in order]
    colors = [family.ink if i in highlight else "#c4cfd6" for i in order]
    if family.penalty:
        values.append(family.penalty)
        colors.append("#718493")
        labels.append("Unmatched models")
    bars = ax.barh(np.arange(len(labels)), values, color=colors, height=0.62)
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=9)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.margins(x=0.2)
    ax.set_xlabel(family.statistic)
    ax.set_title("Per-image contribution", weight="bold")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)


def _detail(ax, family, i):
    y, x = family.data[i]
    my, mx = family.predictions[i]
    major, minor, angle = family.footprints[i]
    if np.isfinite([major, minor, angle]).all():
        ax.add_patch(
            Ellipse(
                (x, y),
                2 * major,
                2 * minor,
                angle=angle,
                facecolor="none",
                edgecolor=family.ink,
                ls=":",
                lw=1.4,
            )
        )
    source = family.source.presentation.is_source_plane
    ax.scatter(
        x,
        y,
        marker=family.marker,
        edgecolors=family.ink,
        facecolors=family.ink if source else "none",
        s=35,
        zorder=3,
    )
    if np.isfinite([my, mx]).all():
        ax.plot([x, mx], [y, my], color="#698090", lw=1)
        ax.scatter(
            mx, my, marker="+" if source else "x", color="#172f49", s=56, zorder=4
        )


def _source_plane(ax, family, highlight):
    rec = family.record
    source = family.source
    bounds = ax.get_window_extent()
    aspect = bounds.width / bounds.height
    _frame(ax, np.vstack([source.traced, source.centre]), aspect)
    colors = np.full(len(rec.numbers), "#718493", dtype=object)
    colors[highlight] = family.ink
    ax.scatter(*source.traced.T[::-1], c=colors, s=28)
    ax.scatter(*source.centre[::-1], marker="+", s=96, color="#172f49")
    _numbers(ax, source.traced, family.labels, colors)
    ax.set_title("All back-traced images", weight="bold")


def _zoom_half_width(family):
    radius = np.where(np.isfinite(family.footprints[:, 0]), family.footprints[:, 0], 0)
    delta = np.abs(family.predictions - family.data)
    return (
        max(
            np.max(np.where(np.isfinite(delta), delta, 0), initial=0),
            np.max(radius, initial=0),
            0.0005,
        )
        * 1.40
    )


def _image_detail(ax, family, i, half):
    if not np.isfinite(family.data[i]).all():
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            f"{family.image_title(i)}\nPosition unavailable",
            transform=ax.transAxes,
            ha="center",
        )
        return
    angular_axes(ax, family.data[i], half, small=True)
    _detail(ax, family, i)
    ax.set_title(family.image_title(i), size=10, weight="bold")


def _detail_key(family):
    is_source = family.source.presentation.is_source_plane
    keys = [
        Line2D(
            [],
            [],
            ls="",
            marker=family.marker,
            color=family.ink,
            mfc=family.ink if is_source else "none",
        ),
        Line2D([], [], ls="", marker="+" if is_source else "x", color="#172f49"),
        Line2D([], [], ls=":", color=family.ink),
    ]
    model_label = "Nearest model (mixture)" if family.record.mixture else "Paired model"
    labels = (
        ["Back-traced image", "Source centre", "1σ source uncertainty"]
        if is_source
        else ["Observed", model_label, "1σ position error"]
    )
    return keys, labels


def _shared_detail(ax, family, indices):
    rec = family.record
    source = family.source
    radius = np.nan_to_num(family.footprints[:, 0], nan=0, posinf=0, neginf=0)
    _frame(
        ax,
        np.vstack(
            [
                family.data - radius[:, None],
                family.data + radius[:, None],
                family.predictions,
                (
                    rec.model
                    if not source.presentation.is_source_plane
                    else source.traced
                ),
            ]
        ),
    )
    for i in indices:
        _detail(ax, family, i)
    if not source.presentation.is_source_plane and np.any(rec.unmatched):
        unmatched = rec.model[rec.unmatched]
        ax.scatter(
            *unmatched.T[::-1],
            marker="x",
            color="#718493",
            s=36,
            label="Unpaired model",
        )
    _numbers(ax, family.data, family.labels, [family.ink] * len(indices))
    ax.set_title(
        (
            "Source-plane detail"
            if source.presentation.is_source_plane
            else "All images in the image plane"
        ),
        weight="bold",
    )


def _page(families, family, image, indices, page, pages, shared, title_prefix):
    import matplotlib.pyplot as plt

    rec = family.record
    source = family.source
    plane = "Source" if source.presentation.is_source_plane else "Image"
    # More ranking rows need extra space, including on subsequent pages.
    height = (8.1 if len(rec.numbers) <= 3 else 10.0) + max(
        0, len(rec.numbers) - 8
    ) * 0.25
    fig = plt.figure(figsize=(14.0, height))
    title = f"{rec.name}  /  {plane}-plane fit"
    fig.text(
        0.08,
        0.967,
        f"{title_prefix} {title}" if title_prefix else title,
        size=22,
        weight="bold",
    )
    subtitle = f"{len(rec.numbers)} images    ·    total {family.statistic} = {family.total:.2f}"
    if rec.redshift is not None:
        subtitle = f"z = {rec.redshift:g}    ·    {subtitle}"
    fig.text(0.08, 0.927, subtitle, size=12)
    if pages > 1:
        fig.text(0.94, 0.969, f"Page {page} of {pages}", ha="right", size=14)
    sky = fig.add_axes([0.085, 0.465, 0.383, 0.415])
    _overview(sky, families, family, image, indices)
    highlight = indices if pages > 1 else family.ranked[:1]
    bar = fig.add_axes([0.085, 0.095, 0.383, 0.23])
    _ranking(bar, family, highlight)
    summary = fig.add_axes([0.55, 0.095, 0.383, 0.23])
    _source_plane(summary, family, highlight)
    fig.text(0.55, 0.025, "+  fitted source centre", fontsize=10)

    half = _zoom_half_width(family)
    legend_y = 0.383
    if shared:
        ax = fig.add_axes([0.55, 0.475, 0.383, 0.36])
        _shared_detail(ax, family, indices)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best", fontsize=8)
    else:
        columns = min(3, len(rec.numbers))
        rows = (min(6, len(rec.numbers)) + columns - 1) // columns
        grid = fig.add_gridspec(
            rows,
            columns,
            left=0.54,
            right=0.96,
            bottom=0.475,
            top=0.837,
            wspace=0.22,
            hspace=0.42,
        )
        tiles = []
        for slot, i in enumerate(indices):
            ax = fig.add_subplot(grid[slot // columns, slot % columns])
            _image_detail(ax, family, i, half)
            tiles.append(ax.get_position())
        # Anchor the heading and key to the square tiles even when the grid has spare height.
        fig.text(
            tiles[0].x0,
            tiles[0].y1 + 0.37 / height,
            f"{plane}-plane detail by image",
            size=14,
            weight="bold",
        )
        legend_y = min(tile.y0 for tile in tiles) - 0.42 / height
    keys, labels = _detail_key(family)
    fig.legend(
        keys,
        labels,
        loc="upper center" if not shared else "lower center",
        bbox_to_anchor=(0.745, legend_y),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    return fig


def plot_family_fits(
    fits,
    image=None,
    single_panel=False,
    output_path=None,
    output_format=None,
    title_prefix=None,
):
    """Write each family's fit.png and, for more than six zooms, fit_page_02.png etc."""
    families = _families_from(fits)
    used = set()
    for family in families:
        rec = family.record
        name = str(rec.name)
        if not re.fullmatch(r"[\w-]+", name) or name.casefold() in used:
            name = f"family_{rec.k}"
            while name.casefold() in used:
                name += "_"
        used.add(name.casefold())
        path = str(Path(output_path) / name) if output_path else None
        _subplot(
            families,
            family,
            image,
            single_panel,
            path,
            "fit",
            output_format,
            title_prefix,
        )


def _families_from(fits):
    records = ps.records_from_fits(fits)
    colormap = ps.assign_styles(records)
    return [
        Family.from_record(record, by_redshift=colormap is not None)
        for record in records
        if len(record.observed)
    ]


def _subplot(
    families,
    family,
    image,
    single_panel,
    output_path,
    output_filename,
    output_format,
    title_prefix,
):
    order = (
        np.argsort(family.record.numbers)
        if family.record.fit.dataset.image_ids is None
        else np.arange(len(family.record.numbers))
    )
    pages = (
        [order]
        if single_panel or len(order) <= 6
        else np.array_split(family.ranked, np.arange(6, len(order), 6))
    )
    for page, indices in enumerate(pages, 1):
        fig = _page(
            families,
            family,
            image,
            indices,
            page,
            len(pages),
            single_panel,
            title_prefix,
        )
        filename = output_filename if page == 1 else f"{output_filename}_page_{page:02}"
        save_figure(fig, path=output_path, filename=filename, format=output_format)


def _select_family(fit, fits, overview=False):
    """Use the cluster ordering for colours, even when drawing only one family."""
    fits = [fit] if fits is None else list(fits)
    index = next((i for i, other in enumerate(fits) if other is fit), None)
    if index is None:
        raise ValueError("fits must include the selected fit.")
    if not len(fit.dataset.positions):
        raise ValueError("The selected family has no observed positions.")
    if overview:
        families = _families_from(fits)
        return next(f for f in families if f.record.fit is fit), families

    # Only the selected fit needs ray tracing for the three single-family panels.
    records = ps.records_from_datasets([other.dataset for other in fits])
    for context, other in zip(records, fits):
        if context.redshift is None:
            plane = ps.plane_index_of(other.positions, other.tracer)
            context.redshift = float(other.tracer.planes[plane].redshift)
    record = ps.source_record_from_fit(index, fit)
    records[index] = record
    colormap = ps.assign_styles(records)
    return Family.from_record(record, by_redshift=colormap is not None), None


@contextmanager
def _panel_axes(ax, output_path, output_filename, output_format):
    if ax is not None:
        yield ax
        return
    fig, panel = subplots(1, 1, figsize=(7, 6))
    yield panel
    fig.tight_layout()
    save_figure(fig, path=output_path, filename=output_filename, format=output_format)


def plot_family_overview(
    fit,
    *,
    fits=None,
    image=None,
    ax=None,
    output_path=None,
    output_filename="family_overview",
    output_format=None,
):
    """Show all families on the sky and number the selected fit's observed images.

    Pass the cluster's sequence of FitPointDataset objects as fits, including fit.
    With fit alone the overview shows one family. image is an optional Array2D.
    Supply ax to compose a figure, or use output_path and output_format to save it.
    """
    family, families = _select_family(fit, fits, overview=True)
    with _panel_axes(ax, output_path, output_filename, output_format) as panel:
        _overview(panel, families, family, image, np.arange(len(family.record.numbers)))


def plot_family_contributions(
    fit,
    *,
    fits=None,
    ax=None,
    output_path=None,
    output_filename="family_contributions",
    output_format=None,
):
    """Rank contributions to the positions fit's own statistic, including any unmatched-model penalty.

    fit is a FitPointDataset. Optional fits preserves the cluster's colours.
    Supply ax to draw into an existing figure or output_path and output_format to save.
    """
    family, _ = _select_family(fit, fits)
    with _panel_axes(ax, output_path, output_filename, output_format) as panel:
        _ranking(panel, family, family.ranked[:1])


def plot_family_image_detail(
    fit,
    image_number=None,
    *,
    image_id=None,
    fits=None,
    ax=None,
    output_path=None,
    output_filename="family_image_detail",
    output_format=None,
):
    """Zoom in on the labelled image_number (one-based) in the plane used by the likelihood.

    For a source-plane fit compare the back-traced position and source centre.
    For an image-plane fit compare the observed and predicted positions.
    Each numbered zoom uses the same angular scale as the others in this family.
    Pass image_id to select by catalogue identifier, or omit both selectors for shared axes.
    Optional fits preserves the cluster's colours. Supply ax to compose a figure.
    """
    family, _ = _select_family(fit, fits)
    numbers = family.record.numbers
    if image_id is not None:
        if image_number is not None:
            raise ValueError("Choose either image_id or image_number.")
        ids = fit.dataset.image_ids
        if ids is None or image_id not in ids:
            raise ValueError(f"Unknown image ID {image_id!r} for {fit.dataset.name}.")
        image_number = numbers[ids.index(image_id)]
    if image_number is not None:
        matches = np.flatnonzero(numbers == image_number)
        if len(matches) != 1:
            raise ValueError(
                f"Unknown image number {image_number!r} for {fit.dataset.name}."
            )
    with _panel_axes(ax, output_path, output_filename, output_format) as panel:
        if image_number is None:
            _shared_detail(panel, family, np.argsort(numbers))
        else:
            _image_detail(panel, family, matches[0], _zoom_half_width(family))
        if panel.axison:
            handles, labels = _detail_key(family)
            extra_handles, extra_labels = panel.get_legend_handles_labels()
            panel.legend(handles + extra_handles, labels + extra_labels, fontsize=8)


def plot_family_source_plane(
    fit,
    *,
    fits=None,
    ax=None,
    output_path=None,
    output_filename="family_source_plane",
    output_format=None,
):
    """Show every observed image of one family back-traced to its source plane.

    Works for both source- and image-plane positions likelihoods. Image labels match
    the overview and contributions. The plus marks the fitted source centre.
    Optional fits preserves the cluster's colours. Supply ax to compose a figure.
    """
    family, _ = _select_family(fit, fits)
    with _panel_axes(ax, output_path, output_filename, output_format) as panel:
        _source_plane(panel, family, family.ranked[:1])
        panel.legend(
            [Line2D([], [], ls="", marker="+", color="#172f49")],
            ["Fitted source centre"],
            fontsize=8,
        )


def subplot_fit_family(
    fit,
    *,
    fits=None,
    image=None,
    single_panel=False,
    output_path=None,
    output_filename="fit_family",
    output_format=None,
    title_prefix=None,
):
    """Draw one selected family's overview, ranking, image details and source plane.

    fit is a FitPointDataset. Pass fits (including fit) for the full cluster sky
    and its colours, and image for an optional Array2D background. Families with
    more than six images have paginated zooms. single_panel=True combines their
    detail in a shared coordinate frame. Saved pages go directly into output_path.
    """
    family, families = _select_family(fit, fits, overview=True)
    _subplot(
        families,
        family,
        image,
        single_panel,
        output_path,
        output_filename,
        output_format,
        title_prefix,
    )
