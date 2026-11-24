"""Cluster position-fit diagnostics, one figure per multiply imaged source."""

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

from autoarray.plot.utils import save_figure
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
            *((record.color, record.color, record.marker) if by_redshift else family_style(record)),
        )

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
    offsets = pp.adjacent_number_placements_from(
        markers, labels, width, height, fontsize=9
    )
    for (y, x), number, color, (dx, dy, ha, va) in zip(
        positions, labels, colors, offsets
    ):
        label = ax.annotate(
            number,
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
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
            _numbers(ax, record.observed, record.numbers, colors)
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
    labels = [f"Image {family.record.numbers[i]}" for i in order]
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


def _page(families, family, image, indices, page, pages, shared, title_prefix):
    import matplotlib.pyplot as plt

    rec = family.record
    source = family.source
    plane = "Source" if source.presentation.is_source_plane else "Image"
    # More ranking rows need extra space, including on subsequent pages.
    height = 11.2 + max(0, len(rec.numbers) - 8) * 0.25
    fig = plt.figure(figsize=(16.2, height))
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
    aspect = 0.383 * 16.2 / (0.23 * height)
    _frame(summary, np.vstack([source.traced, source.centre]), aspect)
    colors = np.full(len(rec.numbers), "#718493", dtype=object)
    colors[highlight] = family.ink
    summary.scatter(*source.traced.T[::-1], c=colors, s=28)
    summary.scatter(*source.centre[::-1], marker="+", s=96, color="#172f49")
    _numbers(summary, source.traced, rec.numbers, colors)
    summary.set_title("All images in the source plane", weight="bold")
    fig.text(0.55, 0.025, "+  fitted source centre", fontsize=10)

    radius = np.where(np.isfinite(family.footprints[:, 0]), family.footprints[:, 0], 0)
    delta = np.abs(family.predictions - family.data)
    half = (
        max(
            np.max(np.where(np.isfinite(delta), delta, 0), initial=0),
            np.max(radius, initial=0),
            0.0005,
        )
        * 1.40
    )
    if shared:
        ax = fig.add_axes([0.55, 0.475, 0.383, 0.36])
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
            ax.legend(loc="best", fontsize=8)
        _numbers(ax, family.data, rec.numbers, [family.ink] * len(indices))
        ax.set_title(
            (
                "Source-plane detail"
                if source.presentation.is_source_plane
                else "All images in the image plane"
            ),
            weight="bold",
        )
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
            wspace=0.5,
            hspace=0.74,
        )
        fig.text(0.54, 0.881, f"{plane}-plane detail by image", size=14, weight="bold")
        for slot, i in enumerate(indices):
            ax = fig.add_subplot(grid[slot // columns, slot % columns])
            if not np.isfinite(family.data[i]).all():
                ax.set_axis_off()
                ax.text(
                    0.5,
                    0.5,
                    f"Image {rec.numbers[i]}\nPosition unavailable",
                    transform=ax.transAxes,
                    ha="center",
                )
                continue
            angular_axes(ax, family.data[i], half, small=True)
            _detail(ax, family, i)
            ax.set_title(f"Image {rec.numbers[i]}", size=10, weight="bold")
    is_source = source.presentation.is_source_plane
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
    model_label = "Nearest model (mixture)" if rec.mixture else "Paired model"
    labels = (
        ["Back-traced image", "Source centre", "1σ source uncertainty"]
        if is_source
        else ["Observed", model_label, "1σ position error"]
    )
    fig.legend(
        keys,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.745, 0.383),
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
    records = ps.records_from_fits(fits)
    colormap = ps.assign_styles(records)
    families = [
        Family.from_record(record, by_redshift=colormap is not None)
        for record in records if len(record.observed)
    ]
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
        order = np.argsort(rec.numbers)
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
            filename = "fit" if page == 1 else f"fit_page_{page:02}"
            save_figure(fig, path=path, filename=filename, format=output_format)
