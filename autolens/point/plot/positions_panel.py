"""
Shared drawing of the point-source "Positions" panels of ``dataset_point.png`` and ``fit.png``.

Both panels are drawn on the same frame:

- **Extent.** The extent of the ``PointSolver`` grid when a solver is available, so the panel shows the region in
  which the model can find images. Without a solver it falls back to the critical-curve region of
  :func:`~autolens.lens.plot.critical_curves.critical_curve_grid_from` (a square of twice the half-span of the
  positions, plus a buffer), which contains every position.
- **Markers.** Outlined amber circles for the observed positions, with a circle of radius equal to each position's
  noise-map value (one isotropic sigma per position) and the sigma always stated in the legend. The circle is drawn
  at its true radius and is never inflated, so at high-resolution noise and a large solver extent it is sub-pixel.
- **No axis labels**, so the arc-second tick labels are not overlapped.

The fit panel additionally draws the model positions, the residual lines of the fit's own pairing, unmatched model
images, critical curves, the centres of the tracer's mass profiles and the image numbers.

**Image numbers.** Images are numbered 1..N in the order of :func:`image_order_from` (descending angle about the
origin, the order the ``PointSolver`` figure numbers its solved images in), and each number is placed immediately
beside its own image by :func:`adjacent_number_placements_from`, which the ``PointSolver`` figure shares. The fit's
image-plane and source-plane panels use the same numbers, so image ``i`` can be followed from one to the other.
"""

import numpy as np
from typing import List, Optional, Tuple

OBSERVED_COLOR = "k"
NOISE_COLOR = "tab:orange"
MODEL_COLOR = "r"
UNMATCHED_COLOR = "0.45"
MASS_CENTRE_COLOR = "tab:green"

# The model cross is thinner and smaller than the observed marker (``s=110``), so under an exact fit the observed
# outline stays visible around it rather than being hidden.
MODEL_MARKER_SIZE = 60
MODEL_MARKER_LINEWIDTH = 1.2

# Critical-curve colours: `_compute_critical_curve_lines` returns white (tangential) and yellow (radial) lines for
# dark image backgrounds, which are invisible on the white positions panel, so they are mapped to dark grey and dark
# olive here, drawn thick so the curves read clearly against the markers.
CRITICAL_CURVE_COLOR_MAP = {"white": "0.2", "yellow": "#5c5c00"}
CRITICAL_CURVE_LINEWIDTH = 1.8


# Candidate placements of an image number relative to its image, as (dx, dy) offsets in points plus the text's
# horizontal and vertical alignment: the number's nearest corner sits a few points off the image marker, upper-left
# first, so a number is always immediately adjacent to its own image.
NUMBER_PLACEMENTS = (
    (-6, 5, "right", "bottom"),
    (6, 5, "left", "bottom"),
    (-6, -5, "right", "top"),
    (6, -5, "left", "top"),
)

# Radius (points) treated as occupied around an image marker when placing labels.
MARKER_RADIUS_POINTS = 6.0

# Font size of the image numbers on the fit panels.
FIT_NUMBER_FONTSIZE = 12


def _np_grid(grid) -> np.ndarray:
    """
    Returns a (N, 2) numpy array of (y,x) coordinates from a grid-like input.
    """
    grid = grid.array if hasattr(grid, "array") else grid
    return np.asarray(grid, dtype=float).reshape(-1, 2)


def solver_extent_from(solver) -> Optional[List[float]]:
    """
    Returns the ``[x_min, x_max, y_min, y_max]`` extent of a ``PointSolver``'s grid, or ``None`` if the solver is
    ``None`` or does not have finite, rectangular bounds (e.g. a custom solver).
    """
    if solver is None:
        return None
    try:
        y_min, y_max = float(solver.y_min), float(solver.y_max)
        x_min, x_max = float(solver.x_min), float(solver.x_max)
    except (AttributeError, TypeError, ValueError):
        return None
    bounds = np.array([x_min, x_max, y_min, y_max])
    if not np.all(np.isfinite(bounds)) or x_max <= x_min or y_max <= y_min:
        return None
    return [x_min, x_max, y_min, y_max]


def fallback_extent_from(
    positions, scale: float = 2.0, buffer: float = 0.1
) -> List[float]:
    """
    Returns the ``[x_min, x_max, y_min, y_max]`` extent used when no solver is available.

    This is the square of :func:`~autolens.lens.plot.critical_curves.critical_curve_grid_from` (centred on the
    bounding box of the positions, with half-width ``scale`` times their half-span plus ``buffer``), so the panel
    covers the region the critical curves are computed on and contains every position.
    """
    positions = _np_grid(positions)
    y_min, y_max = positions[:, 0].min(), positions[:, 0].max()
    x_min, x_max = positions[:, 1].min(), positions[:, 1].max()
    y_centre, x_centre = (y_min + y_max) / 2.0, (x_min + x_max) / 2.0
    half_width = scale * (max(y_max - y_min, x_max - x_min) / 2.0 + buffer)
    return [
        x_centre - half_width,
        x_centre + half_width,
        y_centre - half_width,
        y_centre + half_width,
    ]


def positions_extent_from(positions, solver=None) -> List[float]:
    """
    Returns the ``[x_min, x_max, y_min, y_max]`` extent of a positions panel.

    With a solver this is the solver grid's extent. If an observed position lies outside it (a solver grid too
    small for the data, which ``AnalysisPoint`` already warns about) the extent is enlarged just enough to contain
    that position with a margin, so no observed position is ever clipped. Without a solver it is
    :func:`fallback_extent_from`.
    """
    extent = solver_extent_from(solver)
    if extent is None:
        return fallback_extent_from(positions)

    positions = _np_grid(positions)
    margin = 0.05 * max(extent[1] - extent[0], extent[3] - extent[2])
    x_min, x_max, y_min, y_max = extent
    if np.any(positions[:, 1] < x_min) or np.any(positions[:, 1] > x_max):
        x_min = min(x_min, positions[:, 1].min() - margin)
        x_max = max(x_max, positions[:, 1].max() + margin)
    if np.any(positions[:, 0] < y_min) or np.any(positions[:, 0] > y_max):
        y_min = min(y_min, positions[:, 0].min() - margin)
        y_max = max(y_max, positions[:, 0].max() + margin)
    return [x_min, x_max, y_min, y_max]


def _format_arcsec(value: float) -> str:
    return f"{value:.3g}″"


def sigma_label_from(noise_map) -> str:
    """
    Returns the sigma text of the observed-positions legend entry, e.g. ``σ = 0.005″``, or ``σ = 0.005–0.01″``
    when the positions have different noise-map values.
    """
    noise = np.asarray(
        noise_map.array if hasattr(noise_map, "array") else noise_map, dtype=float
    ).ravel()
    noise = noise[np.isfinite(noise)]
    if noise.size == 0:
        return "σ = n/a"
    if np.allclose(noise, noise[0]):
        return f"σ = {_format_arcsec(noise[0])}"
    return f"σ = {noise.min():.3g}–{_format_arcsec(noise.max())}"


def plot_observed_positions(ax, positions, noise_map):
    """
    Draws the observed positions as outlined amber circles, each with a circle of radius equal to its noise-map
    value (never inflated), labelled ``Observed (σ = ...)`` in the legend.
    """
    from matplotlib.patches import Circle
    from autolens.point.plot.cluster_sky import AMBER_ON_WHITE

    positions = _np_grid(positions)
    noise = np.broadcast_to(
        np.asarray(
            noise_map.array if hasattr(noise_map, "array") else noise_map, dtype=float
        ).ravel(),
        (len(positions),),
    )

    ax.scatter(
        positions[:, 1],
        positions[:, 0],
        s=110,
        facecolors="none",
        edgecolors=AMBER_ON_WHITE,
        linewidths=1.8,
        marker="o",
        zorder=4,
        label=f"Observed ({sigma_label_from(noise_map)})",
    )
    for (y, x), sigma in zip(positions, noise):
        if np.isfinite(sigma) and sigma > 0:
            ax.add_patch(
                Circle(
                    (x, y),
                    radius=sigma,
                    fill=False,
                    edgecolor=NOISE_COLOR,
                    linewidth=1.0,
                    zorder=5,
                )
            )


def plot_critical_curves(ax, lines, line_colors=None):
    """
    Draws critical curves as thick, dark lines labelled ``Critical curve`` in the legend.

    The curves are computed on :func:`~autolens.lens.plot.critical_curves.critical_curve_grid_from` passed the
    solver, whose grid covers the whole panel, so a curve is never broken inside the panel: a curve which runs past
    the panel edge is cut cleanly by the axes frame, which tells the reader it extends beyond the solver grid.
    """
    if not lines:
        return
    labelled = False
    for i, line in enumerate(lines):
        line = np.asarray(line)
        if line.ndim != 2 or len(line) == 0:
            continue
        color = line_colors[i] if line_colors and i < len(line_colors) else "white"
        ax.plot(
            line[:, 1],
            line[:, 0],
            color=CRITICAL_CURVE_COLOR_MAP.get(color, color),
            linewidth=CRITICAL_CURVE_LINEWIDTH,
            zorder=2,
            label=None if labelled else "Critical curve",
        )
        labelled = True


def mass_centres_from(tracer) -> np.ndarray:
    """
    Returns the (y,x) centres of every mass profile of every galaxy in the tracer, as a (N, 2) array with
    duplicates removed.

    External shear is skipped, as its ``centre`` has no physical meaning.
    """
    import autogalaxy as ag

    centres = []
    for galaxy in getattr(tracer, "galaxies", []):
        try:
            mass_profiles = galaxy.cls_list_from(cls=ag.mp.MassProfile)
        except AttributeError:
            continue
        for mass_profile in mass_profiles:
            if isinstance(mass_profile, ag.mp.ExternalShear):
                continue
            centre = getattr(mass_profile, "centre", None)
            if centre is None:
                continue
            try:
                centre = (float(centre[0]), float(centre[1]))
            except (TypeError, ValueError, IndexError):
                continue
            if np.all(np.isfinite(centre)) and centre not in centres:
                centres.append(centre)
    return np.asarray(centres, dtype=float).reshape(-1, 2)


def plot_mass_centres(ax, tracer, size: float = 140, linewidths: float = 2):
    """
    Draws the centre of every mass profile in the tracer as a ``+`` labelled ``Mass centre`` in the legend.
    """
    centres = mass_centres_from(tracer)
    if len(centres) == 0:
        return
    ax.scatter(
        centres[:, 1],
        centres[:, 0],
        s=size,
        c=MASS_CENTRE_COLOR,
        marker="+",
        linewidths=linewidths,
        zorder=4.5,
        label="Mass centre",
    )


def model_pairing_from(
    fit_positions,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Returns the image-plane pairing of observed to model positions which the fit's chi-squared uses, as a tuple
    ``(observed, model, model_index_per_observed)``, where ``model`` holds only the finite model positions and
    ``model_index_per_observed[i]`` is the model position observed position ``i`` is paired with (``-1`` if the
    model has no images).

    - ``FitPositionsImagePair``: the Hungarian (one-to-one) assignment; observed positions the assignment cannot
      pair (fewer model than observed images) are paired with their nearest model position, as in its residuals.
    - ``FitPositionsImagePairRepeat``: each observed position's nearest model position (repeats allowed).
    - ``FitPositionsImagePairAll``: each observed position's nearest model position, the dominant term of its
      all-pairs mixture likelihood (the mixture has no hard pairing).

    Source-plane fits (e.g. ``FitPositionsSource``) have no image-plane pairing, so ``None`` is returned.
    """
    from autolens.point.fit.positions.image.abstract import (
        AbstractFitPositionsImagePair,
    )
    from autolens.point.fit.positions.image.pair import FitPositionsImagePair

    if not isinstance(fit_positions, AbstractFitPositionsImagePair):
        return None

    observed = _np_grid(fit_positions.data)
    model = _np_grid(fit_positions.model_data)
    model = model[np.isfinite(model).all(axis=1)]

    if len(model) == 0:
        return observed, model, np.full(len(observed), -1, dtype=int)

    distances = np.linalg.norm(observed[:, None, :] - model[None, :, :], axis=2)
    indexes = np.argmin(distances, axis=1)

    if isinstance(fit_positions, FitPositionsImagePair):
        from scipy.optimize import linear_sum_assignment

        data_indexes, model_indexes = linear_sum_assignment(distances)
        indexes[data_indexes] = model_indexes

    return observed, model, indexes


def unmatched_model_mask_from(model: np.ndarray, indexes: np.ndarray) -> np.ndarray:
    """
    Returns a boolean mask over the (finite) model positions, true for every model image no observed position is
    paired with: an image the model predicts but which is not observed.
    """
    mask = np.ones(len(model), dtype=bool)
    mask[indexes[indexes >= 0]] = False
    return mask


def residual_style_from(fit_positions) -> Tuple[str, str]:
    """
    Returns the ``(linestyle, legend label)`` of an image-plane fit's residual lines, by its pairing.

    Hard-pairing fits (``FitPositionsImagePair``, ``FitPositionsImagePairRepeat`` and its ``*Solved`` variant) pair
    each observed position with one model position: solid ``"Residual"`` lines. ``FitPositionsImagePairAll`` (and
    its ``*Solved`` variant) uses an all-pairs mixture likelihood with no hard pairing, so the line drawn to the
    nearest model position is only its dominant term: dashed ``"Residual (nearest, soft pairing)"`` lines. The class
    is looked up along its method resolution order, so a subclass is styled as its parent.
    """
    from autolens.point.fit.positions.image.pair_all import FitPositionsImagePairAll

    soft = ("--", "Residual (nearest, soft pairing)")
    table = {FitPositionsImagePairAll: soft}
    for cls in type(fit_positions).__mro__:
        if cls in table:
            return table[cls]
    return "-", "Residual"


def model_label_from(fit_positions) -> str:
    """
    Returns the legend label of an image-plane fit's model positions: ``"Model"``, or
    ``"Model (from solved β*)"`` for a ``*Solved`` fit, whose model images are solved from the analytically solved
    source centre rather than a free ``centre`` parameter (the same ``β*`` the source-plane panel marks with a
    diamond).
    """
    from autolens.point.fit.solved import SolvedCentre

    if isinstance(fit_positions, SolvedCentre):
        return "Model (from solved β*)"
    return "Model"


def plot_fit_model_positions(ax, fit_positions):
    """
    Draws the model positions of a positions fit on top of the observed positions.

    For an image-plane fit the matched model positions are drawn as red crosses, a thin ``Residual`` line joins
    each observed position to the model position the fit pairs it with, and model images no observed position is
    paired with are drawn as hollow grey crosses (``Unmatched model image``).

    For a source-plane fit (e.g. ``FitPositionsSource``) nothing is drawn: its model data are the observed positions
    ray-traced to the source plane, which belong on the fit's source-plane panel, not the image plane.
    """
    pairing = model_pairing_from(fit_positions)

    if pairing is None:
        return

    observed, model, indexes = pairing
    linestyle, label = residual_style_from(fit_positions)

    labelled = False
    for obs, index in zip(observed, indexes):
        if index < 0:
            continue
        ax.plot(
            [obs[1], model[index, 1]],
            [obs[0], model[index, 0]],
            color=MODEL_COLOR,
            linewidth=1.0,
            linestyle=linestyle,
            alpha=0.7,
            zorder=6,
            label=None if labelled else label,
        )
        labelled = True

    unmatched = unmatched_model_mask_from(model, indexes)

    if np.any(~unmatched):
        ax.scatter(
            model[~unmatched, 1],
            model[~unmatched, 0],
            s=MODEL_MARKER_SIZE,
            c=MODEL_COLOR,
            marker="x",
            linewidths=MODEL_MARKER_LINEWIDTH,
            zorder=7,
            label=model_label_from(fit_positions),
        )
    if np.any(unmatched):
        ax.scatter(
            model[unmatched, 1],
            model[unmatched, 0],
            s=110,
            facecolors="none",
            edgecolors=UNMATCHED_COLOR,
            marker="X",
            linewidths=1.5,
            zorder=7,
            label="Unmatched model image",
        )


def finish_positions_panel(ax, title: str, extent: List[float]):
    """
    Applies the title, extent and arc-second ticks of a positions panel, with no axis labels, an equal aspect
    ratio (so noise circles are circles) and a legend.
    """
    from autoarray.plot.utils import apply_labels, apply_extent

    apply_labels(ax, title=title, xlabel="", ylabel="")
    apply_extent(ax, extent)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=9, framealpha=0.85, markerscale=0.7)


def image_order_from(positions) -> np.ndarray:
    """
    Returns the indexes of the positions in numbering order: descending angle about the origin, the order in which
    the ``PointSolver`` figure numbers its solved images. Image ``k`` (1-based) is ``positions[order[k - 1]]``.
    """
    positions = _np_grid(positions)
    return np.argsort(-np.arctan2(positions[:, 0], positions[:, 1]), kind="stable")


def image_numbers_from(positions) -> np.ndarray:
    """
    Returns the 1-based number of every position, in the positions' own (dataset) order, following
    :func:`image_order_from`.
    """
    order = image_order_from(positions)
    numbers = np.empty(len(order), dtype=int)
    numbers[order] = np.arange(1, len(order) + 1)
    return numbers


def box_overlaps_circles(box, centres, radius) -> np.ndarray:
    """
    Returns, for every circle of ``radius`` around ``centres``, whether it overlaps the ``(x0, y0, x1, y1)`` box.
    """
    x0, y0, x1, y1 = box
    if len(centres) == 0:
        return np.zeros(0, dtype=bool)
    gap_x = np.maximum(np.maximum(x0 - centres[:, 0], centres[:, 0] - x1), 0.0)
    gap_y = np.maximum(np.maximum(y0 - centres[:, 1], centres[:, 1] - y1), 0.0)
    return np.hypot(gap_x, gap_y) < radius


def boxes_overlap(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def adjacent_number_placements_from(
    markers: np.ndarray,
    labels: List[str],
    width: float,
    height: float,
    fontsize: float,
    *,
    placements_to_try=NUMBER_PLACEMENTS,
) -> List[Tuple[float, float, str, str]]:
    """
    Returns, for each of the first ``len(labels)`` markers, the ``(dx, dy, ha, va)`` offset (points) of its label.

    Each label sits a few points off its own marker (one of ``NUMBER_PLACEMENTS``): the first placement which stays
    inside the panel and overlaps neither another marker nor an already-placed label, else the one overlapping
    least. Catalogue plots can supply further offsets for crowded labels.

    Parameters
    ----------
    markers
        The (x, y) positions of every marker in points from the panel's lower-left corner; markers after the first
        ``len(labels)`` are obstacles only (e.g. rejected candidates).
    labels
        The text of each numbered marker's label.
    width, height
        The size of the panel in points.
    fontsize
        The label font size in points.
    """
    markers = np.asarray(markers, dtype=float).reshape(-1, 2)
    placed_boxes, placements = [], []
    for index, (x, y) in enumerate(markers[: len(labels)]):
        text_width = 0.65 * fontsize * len(labels[index]) + 2.0
        text_height = fontsize + 2.0
        others = np.delete(markers, index, axis=0)
        best, best_score, best_box = NUMBER_PLACEMENTS[0], np.inf, None
        for dx, dy, ha, va in placements_to_try:
            x0 = x + dx - (text_width if ha == "right" else 0.0)
            y0 = y + dy - (text_height if va == "top" else 0.0)
            box = (x0, y0, x0 + text_width, y0 + text_height)
            inside = (
                box[0] >= 0 and box[1] >= 0 and box[2] <= width and box[3] <= height
            )
            score = (
                int(box_overlaps_circles(box, others, MARKER_RADIUS_POINTS).sum())
                + sum(boxes_overlap(box, other) for other in placed_boxes)
                + (0 if inside else 100)
            )
            if score < best_score:
                best, best_score, best_box = (dx, dy, ha, va), score, box
            if score == 0:
                break
        placements.append(best)
        placed_boxes.append(best_box)
    return placements


def _panel_size_points(ax, extent) -> Tuple[float, float, float]:
    """
    Returns the ``(width, height, points_per_arcsec)`` of an equal-aspect panel showing ``extent``, from the axes'
    position in its figure.
    """
    fig_width, fig_height = ax.figure.get_size_inches()
    position = ax.get_position()
    box_width = position.width * fig_width * 72.0
    box_height = position.height * fig_height * 72.0
    span_x = max(extent[1] - extent[0], 1e-30)
    span_y = max(extent[3] - extent[2], 1e-30)
    scale = min(box_width / span_x, box_height / span_y)
    return span_x * scale, span_y * scale, scale


def plot_image_numbers(
    ax,
    positions,
    numbers,
    extent: List[float],
    fontsize: float = FIT_NUMBER_FONTSIZE,
):
    """
    Labels every position with its image number, placed immediately beside it by
    :func:`adjacent_number_placements_from`.

    Parameters
    ----------
    positions
        The (y,x) positions which are labelled.
    numbers
        The number of each position (see :func:`image_numbers_from`).
    extent
        The ``[x_min, x_max, y_min, y_max]`` extent of the panel, used to convert positions to points.
    """
    positions = _np_grid(positions)
    if len(positions) == 0:
        return
    width, height, scale = _panel_size_points(ax, extent)
    markers = (positions[:, ::-1] - np.array([extent[0], extent[2]])) * scale
    placements = adjacent_number_placements_from(
        markers=markers,
        labels=[str(number) for number in numbers],
        width=width,
        height=height,
        fontsize=fontsize,
    )
    for (y, x), number, (dx, dy, ha, va) in zip(positions, numbers, placements):
        ax.annotate(
            f"{number}",
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=fontsize,
            weight="bold",
            zorder=10,
            bbox=dict(fc="w", ec="none", alpha=0.8, pad=0.5),
        )
