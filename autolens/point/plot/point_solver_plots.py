"""
The ``point_solver.png`` figure: how the ``PointSolver`` refines its triangle tiling until it finds the multiple
images of a point source.

The solver tiles the image plane with triangles, ray-traces every triangle to the source plane and keeps the
triangles whose traced image contains the source-plane coordinate. The kept triangles, plus a rim of neighbours, are
up-sampled (each triangle split into four) and the next step repeats the test on them, until the triangles are
smaller than ``pixel_scale_precision``. The centroids of the final kept triangles are the images, after those with an
absolute magnification below ``magnification_threshold`` (e.g. the triangle at the centre of the lens) are rejected.

The figure shows steps 0, 1, 2 and the final step, every panel on the same fixed extent so the triangles visibly
shrink onto the images:

- **Triangles tested** (grey outlines): every triangle ray-traced at that step.
- **Kept** (blue): the triangles whose traced image contains the source.
- **Kept, then rejected** (hatched, muted blue): kept triangles that converge onto a position the magnification
  threshold rejects.
- **Neighbours** (orange): the rim added around the kept triangles before up-sampling.
- **Up-sampled** (green outlines): the triangles tested at the next step.
- the critical curves of the source's own plane, the observed positions (hollow circles) and the solved images (red
  crosses).

The extent spans the source's own images and rejected candidates (padded, square, with round tick values), not the
solver's full grid, so a cluster source with images tens of arc-seconds apart is drawn as large as a galaxy-scale
one. A galaxy-scale grid only just larger than its images is shown whole instead, so its tiling fills the panel. Panel 0 numbers the images ``1, 2, ...`` and the rejected candidates ``r1, r2, ...``. Kept triangles too small
to draw at the panel's size (below ~3 screen pixels) are drawn as a triangle marker at their centroid instead.

**Display grid.** When the solver's own step-0 triangles are too small to draw at that extent (e.g. a cluster grid
of 0.2" triangles over a 40" span), the source is re-solved for the figure only on a coarse uniform grid covering the
extent, whose pixel scale is the smaller of a tenth of the span and the closest separation between any two images
or rejected candidates divided by 1.6 (so close pairs still start in separate triangles), with the solver's own
``pixel_scale_precision`` and ``magnification_threshold``. The display solve is used only if it recovers the
solver's images and rejected candidates; otherwise the solver's own steps are drawn. The solver passed in, and so
the model-fit, is never changed.

One per-panel draw function, :func:`draw_point_solver_panel`, is shared by the 2x2 still (``point_solver.png``) and
the teaching animation (``point_solver.gif``, one panel per frame), so the two cannot drift apart.

The figure is NumPy-only: the steps are re-solved with ``xp=numpy`` even for a solver built with ``use_jax=True``,
because the JAX path pads every step to ``MAX_CONTAINING_SIZE`` triangles and, inside ``jax.jit``, hides the steps
behind the implicit-differentiation rule.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from autolens.point.plot.cluster_sky import AMBER_ON_WHITE

from autolens.point.plot.positions_panel import (
    CRITICAL_CURVE_COLOR_MAP,
    CRITICAL_CURVE_LINEWIDTH,
    NUMBER_PLACEMENTS,
    UNMATCHED_COLOR,
    adjacent_number_placements_from,
    image_order_from,
)

logger = logging.getLogger(__name__)

TESTED_COLOR = "#8a8f95"
NEIGHBOUR_COLOR = "#f2a33a"
KEPT_COLOR = "#1f6fd1"
REJECTED_COLOR = "#9db7d9"
UP_SAMPLED_COLOR = "#2e9e5b"
CRITICAL_CURVE_COLOR = CRITICAL_CURVE_COLOR_MAP["white"]
RADIAL_CRITICAL_CURVE_COLOR = CRITICAL_CURVE_COLOR_MAP["yellow"]
OBSERVED_COLOR = AMBER_ON_WHITE
SOLVED_COLOR = "r"
REJECTED_LABEL_COLOR = UNMATCHED_COLOR

# The width (inches) of one panel of the 2x2 still, which the display-grid decision is made for.
STILL_PANEL_INCHES = 0.85 * 6.0

# The extent spans the source's images and rejected candidates, padded by this fraction of their range on each side.
EXTENT_PAD = 0.2

# The display grid starts with about this many triangles across the extent...
DISPLAY_TRIANGLES_ACROSS = 10

# ... and with a pixel scale below the closest separation of two images (or rejected candidates) divided by this, so
# a close pair (e.g. an image beside a member galaxy's core) starts in separate triangles.
DISPLAY_SEPARATION_FRACTION = 1.6

# The display solve recovers an image when it lands within this many arc-seconds of the solver's own (or within two
# ``pixel_scale_precision``, if larger).
RECOVERY_TOLERANCE = 0.01

# Tick steps (times a power of ten) the extent is widened to, so its ticks fall on round values.
_NICE_TICK_STEPS = (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)

# A kept triangle whose edge spans fewer typographic points (1/72 inch) than this on the panel is drawn as a
# centroid marker: about 3 screen pixels, independent of the dpi the figure is saved at.
MIN_TRIANGLE_POINTS = 2.5

# Above this many solved images (e.g. a cluster) step 0 is not numbered, so the panel stays readable.
MAX_NUMBERED_IMAGES = 10

# A step-0 tiling with more triangles than this is drawn with hairline outlines.
DENSE_TILING = 5000

# Tested triangles whose edge spans fewer points than this (e.g. a cluster's step-0 tiling of hundreds of thousands
# of triangles) merge into a solid slab when outlined, so they are drawn as a light grey shade instead.
MIN_OUTLINE_POINTS = 1.5

# Seconds each animation frame is shown; the final step is held for ``FINAL_FRAME_REPEATS`` frames.
FRAME_SECONDS = 1.25
FINAL_FRAME_REPEATS = 3


@dataclass
class PointSolverFigureData:
    """
    Everything the point-solver figure draws, collected once with NumPy and shared by every panel and frame.

    Attributes
    ----------
    steps
        The selected refinement steps (steps 0, 1, 2 and the final step, or every step when there are fewer).
    n_steps
        The total number of refinement steps the solver took.
    solved
        The (y,x) solved image positions, ordered by angle about the origin so the numbering is stable.
    rejected
        The (y,x) centroids of the final kept triangles the magnification threshold rejects.
    final_centres
        The (y,x) centroids of every final kept triangle (accepted and rejected).
    magnification_threshold
        The solver's ``magnification_threshold``, quoted in the legend.
    extent
        The ``[x_min, x_max, y_min, y_max]`` extent shared by every panel: the source's own images and rejected
        candidates, padded.
    lines
        The (tangential) critical-curve lines, each a (N, 2) array of (y,x) coordinates.
    positions
        The (y,x) observed positions, or ``None``.
    max_kept
        The largest number of triangles kept at any step of the solver's own solve.
    radial_lines
        The radial critical-curve lines, each a (N, 2) array of (y,x) coordinates.
    display_pixel_scale
        The pixel scale of the coarse display grid the steps were re-solved on, or ``None`` when they are the
        solver's own steps.
    """

    steps: list
    n_steps: int
    solved: np.ndarray
    rejected: np.ndarray
    final_centres: np.ndarray
    magnification_threshold: float
    extent: List[float]
    lines: List[np.ndarray]
    positions: Optional[np.ndarray]
    max_kept: int
    radial_lines: List[np.ndarray] = field(default_factory=list)
    display_pixel_scale: Optional[float] = None


def _yx(grid) -> np.ndarray:
    grid = grid.array if hasattr(grid, "array") else grid
    grid = np.asarray(grid, dtype=float).reshape(-1, 2)
    return grid[np.isfinite(grid).all(axis=1)]


def triangles_xy_from(triangles) -> np.ndarray:
    """
    Returns the finite triangles of a triangles container as a (N, 3, 2) array of (x,y) vertices, ready for a
    matplotlib ``PolyCollection``.

    The solver stores vertices as (y,x), so they are swapped here (the retired ``point/visualise.py`` plotted them
    unswapped, transposing every figure). NaN-padded rows (the JAX containers' fixed-size padding) are dropped.
    """
    array = np.asarray(
        triangles.triangles if hasattr(triangles, "triangles") else triangles,
        dtype=float,
    ).reshape(-1, 3, 2)
    array = array[np.isfinite(array).all(axis=(1, 2))]
    return array[:, :, ::-1]


def _centroid_keys(triangles_xy: np.ndarray) -> set:
    return {tuple(c) for c in np.round(triangles_xy.mean(axis=1), 7)}


def neighbour_rim_from(step) -> np.ndarray:
    """
    Returns the (x,y) neighbourhood triangles of a step which are not themselves kept: the rim of neighbours added
    around the kept triangles before up-sampling.
    """
    kept = _centroid_keys(triangles_xy_from(step.filtered_triangles))
    neighbourhood = triangles_xy_from(step.neighbourhood)
    if len(neighbourhood) == 0:
        return neighbourhood
    keys = np.round(neighbourhood.mean(axis=1), 7)
    return neighbourhood[[tuple(k) not in kept for k in keys]]


def split_kept_from(
    step, final_centres: np.ndarray, rejected: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits a step's kept triangles into those converging onto an accepted image and those converging onto a
    position the magnification threshold rejects, returned as two (N, 3, 2) arrays of (x,y) vertices.

    Each kept triangle is assigned to its nearest final kept centroid; it is "kept, then rejected" when that
    centroid is one the threshold rejects.
    """
    kept = triangles_xy_from(step.filtered_triangles)
    if len(kept) == 0 or len(rejected) == 0:
        return kept, kept[:0]

    centres = kept.mean(axis=1)[:, ::-1]  # (y,x)
    to_final = np.linalg.norm(centres[:, None, :] - final_centres[None], axis=2).min(
        axis=1
    )
    to_rejected = np.linalg.norm(centres[:, None, :] - rejected[None], axis=2).min(
        axis=1
    )
    is_rejected = np.isclose(to_rejected, to_final)
    return kept[~is_rejected], kept[is_rejected]


def select_steps(steps: list) -> list:
    """
    Returns the steps drawn by the figure: steps 0, 1, 2 and the final step, or every step when there are four or
    fewer. On a fine grid steps 1, 2 and final can look alike, as the kept triangles are already sub-pixel.
    """
    indexes = sorted({i for i in (0, 1, 2, len(steps) - 1) if 0 <= i < len(steps)})
    return [steps[i] for i in indexes]


def _extent_from(solver, positions) -> List[float]:
    """
    The solver's full extent (enlarged to contain every observed position), padded: the fallback extent of a source
    with no images, rejected candidates or observed positions to span.
    """
    from autolens.point.plot.positions_panel import (
        positions_extent_from,
        solver_extent_from,
    )

    if positions is not None and len(positions):
        extent = positions_extent_from(positions=positions, solver=solver)
    else:
        extent = solver_extent_from(solver)
    if extent is None:
        raise ValueError(
            "The point-solver figure needs a solver with finite rectangular bounds (y_min, y_max, x_min, x_max)."
        )
    pad = 0.02 * max(extent[1] - extent[0], extent[3] - extent[2])
    return [extent[0] - pad, extent[1] + pad, extent[2] - pad, extent[3] + pad]


def round_extent_from(centre, half_width: float) -> List[float]:
    """
    Returns a square ``[x_min, x_max, y_min, y_max]`` extent containing the square of ``half_width`` about the
    (y,x) ``centre``, widened so the ticks :func:`autoarray.plot.utils.apply_extent` places (the centre and
    ``extent_factor_2d`` of the half-width either side) fall on round values, e.g. ``-20"``, ``0"``, ``20"`` rather
    than ``-15.0"``, ``2.2"``, ``20.0"``.

    The tick step is the smallest round step (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6 or 8 times a power of ten) whose
    extent still contains the square once its centre is snapped to a multiple of half the step's power of ten.
    """
    from autoarray.plot.utils import _conf_ticks

    factor = _conf_ticks("extent_factor_2d", 0.75)
    centre_y, centre_x = (float(v) for v in centre)
    half_width = max(float(half_width), 1e-6)

    target = factor * half_width
    power = 10.0 ** np.floor(np.log10(target))
    for multiple in [m * power for m in _NICE_TICK_STEPS] + [
        m * 10.0 * power for m in _NICE_TICK_STEPS
    ]:
        if multiple < target:
            continue
        snap = 0.5 * 10.0 ** np.floor(np.log10(multiple) + 1e-9)
        y = round(centre_y / snap) * snap
        x = round(centre_x / snap) * snap
        half = multiple / factor
        if half >= half_width + max(abs(y - centre_y), abs(x - centre_x)) - 1e-12:
            return [x - half, x + half, y - half, y + half]
    return [
        centre_x - half_width,
        centre_x + half_width,
        centre_y - half_width,
        centre_y + half_width,
    ]


def source_extent_from(points, min_width: float = 0.0) -> Optional[List[float]]:
    """
    Returns the extent shared by every panel: a square spanning the (y,x) ``points`` (a source's solved images,
    rejected candidates and observed positions), padded by ``EXTENT_PAD`` of their range on each side, at least
    ``min_width`` wide and widened to round ticks by :func:`round_extent_from`. ``None`` if there are no points.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        return None
    lo, hi = points.min(axis=0), points.max(axis=0)
    width = max(float(np.max(hi - lo)) * (1.0 + 2.0 * EXTENT_PAD), min_width)
    return round_extent_from(centre=0.5 * (lo + hi), half_width=0.5 * width)


def _within_solver_extent_from(extent, solver) -> List[float]:
    """
    Returns ``extent``, unless it runs past the area the solver tiles and that area is at most twice as wide (e.g.
    a galaxy-scale grid only just larger than its images), in which case the solver's own extent (with round ticks)
    is returned, so the step-0 tiling fills the panel instead of stopping short of its edges.
    """
    from autolens.point.plot.positions_panel import solver_extent_from

    tiled = solver_extent_from(solver)
    if tiled is None:
        return extent
    inside = (
        extent[0] >= tiled[0]
        and extent[1] <= tiled[1]
        and extent[2] >= tiled[2]
        and extent[3] <= tiled[3]
    )
    width = max(tiled[1] - tiled[0], tiled[3] - tiled[2])
    if inside or width > 2.0 * (extent[1] - extent[0]):
        return extent
    centre = (0.5 * (tiled[2] + tiled[3]), 0.5 * (tiled[0] + tiled[1]))
    return round_extent_from(centre=centre, half_width=0.5 * width)


def _too_small_to_draw(edge: float, extent, panel_inches: float) -> bool:
    width = max(extent[1] - extent[0], extent[3] - extent[2])
    return edge / width * panel_inches * 72.0 < MIN_TRIANGLE_POINTS


def display_pixel_scale_from(extent, points) -> float:
    """
    Returns the starting pixel scale of the display grid: a ``DISPLAY_TRIANGLES_ACROSS``-th of the extent, capped at
    the closest separation of any two (y,x) ``points`` (the solved images and rejected candidates) divided by
    ``DISPLAY_SEPARATION_FRACTION``, so every image starts in its own triangle.
    """
    pixel_scale = (extent[1] - extent[0]) / DISPLAY_TRIANGLES_ACROSS
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(points) > 1:
        separations = np.linalg.norm(points[:, None] - points[None], axis=2)
        closest = float(separations[np.triu_indices(len(points), 1)].min())
        if closest > 0.0:
            pixel_scale = min(pixel_scale, closest / DISPLAY_SEPARATION_FRACTION)
    return pixel_scale


def display_solver_from(solver, extent, pixel_scale: float):
    """
    Returns a solver of the same class as ``solver`` tiling ``extent`` (plus a pixel of margin) with triangles of
    ``pixel_scale``, with the solver's own ``pixel_scale_precision``, ``magnification_threshold``,
    ``neighbor_degree`` and ``residual_threshold``. It is used for the figure only.
    """
    x_min, x_max, y_min, y_max = extent
    return type(solver).for_limits_and_scale(
        y_min=y_min - pixel_scale,
        y_max=y_max + pixel_scale,
        x_min=x_min - pixel_scale,
        x_max=x_max + pixel_scale,
        scale=pixel_scale,
        pixel_scale_precision=solver.pixel_scale_precision,
        magnification_threshold=solver.magnification_threshold,
        neighbor_degree=getattr(solver, "neighbor_degree", 1),
        residual_threshold=getattr(solver, "residual_threshold", 10.0),
    )


def recovers(reference: np.ndarray, found: np.ndarray, tolerance: float) -> bool:
    """
    Returns whether ``found`` holds as many (y,x) points as ``reference`` and every reference point lies within
    ``tolerance`` of one of them.
    """
    reference = np.asarray(reference, dtype=float).reshape(-1, 2)
    found = np.asarray(found, dtype=float).reshape(-1, 2)
    if len(reference) != len(found):
        return False
    if len(reference) == 0:
        return True
    distance = np.linalg.norm(reference[:, None] - found[None], axis=2).min(axis=1)
    return bool(distance.max() <= tolerance)


@dataclass
class _Solve:
    steps: list
    final_centres: np.ndarray
    rejected: np.ndarray
    solved: np.ndarray


def _solve_from(solver, tracer, beta, plane_redshift) -> _Solve:
    """
    Re-solves the point source with NumPy, returning every refinement step, the centroids of the final kept
    triangles, those the magnification threshold rejects and the solved images (in numbering order).
    """
    from autoarray.structures.triangles.shape import Point

    steps = list(
        solver.steps(
            tracer=tracer, shape=Point(*beta), xp=np, plane_redshift=plane_redshift
        )
    )

    final_centres = np.zeros((0, 2))
    rejected = np.zeros((0, 2))
    if len(steps):
        final_centres = triangles_xy_from(steps[-1].filtered_triangles).mean(axis=1)[
            :, ::-1
        ]
    if len(final_centres):
        filtered = np.asarray(
            solver._filter_low_magnification(
                tracer=tracer,
                points=final_centres,
                xp=np,
                plane_redshift=plane_redshift,
            )
        ).reshape(-1, 2)
        rejected = final_centres[~np.isfinite(filtered).all(axis=1)]
    else:
        final_centres = np.zeros((0, 2))

    solved = _yx(
        solver.solve(
            tracer=tracer,
            source_plane_coordinate=beta,
            xp=np,
            plane_redshift=plane_redshift,
        )
    )
    return _Solve(
        steps=steps,
        final_centres=final_centres,
        rejected=rejected,
        solved=solved[image_order_from(solved)],
    )


def _plane_index_from(tracer, plane_redshift: Optional[float]) -> int:
    last = len(tracer.planes) - 1
    if plane_redshift is None:
        return last
    redshifts = np.asarray(tracer.plane_redshifts, dtype=float)
    return int(np.argmin(np.abs(redshifts - float(plane_redshift))))


def point_solver_data_from(
    solver,
    tracer,
    source_plane_coordinate: Tuple[float, float],
    positions=None,
    plane_redshift: Optional[float] = None,
    image_plane_lines=None,
    panel_inches: float = STILL_PANEL_INCHES,
) -> PointSolverFigureData:
    """
    Re-solves the point source with NumPy and collects everything the point-solver figure draws.

    The extent spans the source's solved images, rejected candidates and observed positions. If the solver's
    step-0 triangles are too small to draw on it, the steps are re-solved on a coarse display grid (see the module
    docstring), provided that solve recovers the solver's images and rejected candidates.

    Parameters
    ----------
    solver
        The ``PointSolver`` whose refinement is shown. A solver built with ``use_jax=True`` is re-run with NumPy.
    tracer
        The tracer that ray-traces the triangles to the source plane.
    source_plane_coordinate
        The (y,x) source-plane coordinate of the point source.
    positions
        The observed (y,x) positions, drawn as hollow circles; ``None`` draws none.
    plane_redshift
        The redshift of the plane the source is in, for multi-plane lenses (e.g. clusters). ``None`` is the last
        plane.
    image_plane_lines
        Pre-computed critical-curve lines, drawn as tangential curves. ``None`` computes the tangential and radial
        critical curves of the source's own plane (of the full tracer) on a grid covering the extent; pass ``[]``
        to draw none.
    panel_inches
        The width of a panel in inches, which decides whether the step-0 triangles are too small to draw.
    """
    beta = tuple(float(v) for v in np.asarray(source_plane_coordinate).ravel()[:2])

    real = _solve_from(solver, tracer, beta, plane_redshift)
    if len(real.steps) == 0:
        raise ValueError(
            "The solver took no refinement steps, so there is nothing to plot."
        )

    positions = None if positions is None else _yx(positions)
    candidates = np.vstack([real.solved, real.rejected])
    extent = source_extent_from(
        candidates if positions is None else np.vstack([candidates, positions]),
        min_width=10.0 * float(solver.scale),
    )
    if extent is None:
        extent = _extent_from(solver=solver, positions=positions)
    else:
        extent = _within_solver_extent_from(extent, solver)

    shown, display_pixel_scale = real, None
    if _too_small_to_draw(_edge_of(real.steps[0]), extent, panel_inches):
        pixel_scale = display_pixel_scale_from(extent, candidates)
        if pixel_scale > float(solver.scale):
            display = _solve_from(
                display_solver_from(solver, extent, pixel_scale),
                tracer,
                beta,
                plane_redshift,
            )
            tolerance = max(
                RECOVERY_TOLERANCE, 2.0 * float(solver.pixel_scale_precision)
            )
            if (
                len(display.steps)
                and recovers(real.solved, display.solved, tolerance)
                and recovers(real.rejected, display.rejected, tolerance)
            ):
                shown, display_pixel_scale = display, pixel_scale
            else:
                logger.info(
                    f"PointSolver figure: the {pixel_scale:.3g}\" display grid does not recover the solver's "
                    f"images, so the solver's own steps are drawn."
                )

    radial_lines = []
    if image_plane_lines is None:
        from autolens.point.plot.point_sources import curves_for_planes

        plane_index = _plane_index_from(tracer, plane_redshift)
        curves = curves_for_planes(tracer, [plane_index], extent).get(plane_index)
        lines = [] if curves is None else list(curves.tangential)
        radial_lines = [] if curves is None else list(curves.radial)
    else:
        lines = [np.asarray(line) for line in image_plane_lines]

    return PointSolverFigureData(
        steps=select_steps(shown.steps),
        n_steps=len(shown.steps),
        solved=shown.solved,
        rejected=shown.rejected,
        final_centres=shown.final_centres,
        magnification_threshold=float(solver.magnification_threshold),
        extent=extent,
        lines=lines,
        positions=positions,
        max_kept=max(len(triangles_xy_from(s.filtered_triangles)) for s in real.steps),
        radial_lines=radial_lines,
        display_pixel_scale=display_pixel_scale,
    )


def _edge_of(step) -> float:
    tested = triangles_xy_from(step.initial_triangles)
    if len(tested) == 0:
        return 0.0
    return float(max(np.ptp(tested[0][:, 0]), np.ptp(tested[0][:, 1])))


def is_too_small_to_draw(
    step, data: PointSolverFigureData, panel_inches: float
) -> bool:
    """
    Returns whether a step's triangles span fewer than ``MIN_TRIANGLE_POINTS`` points on a panel
    ``panel_inches`` wide, in which case its kept triangles are drawn as centroid markers.
    """
    return _too_small_to_draw(_edge_of(step), data.extent, panel_inches)


def panel_title_from(step, data: PointSolverFigureData, short: bool = False) -> str:
    """
    Returns a panel title, e.g. ``Step 0 (273 tested, 8 kept + 1 rejected)`` or ``Step 5 — final (...)``; the
    ``short`` form (used by the animation) omits the counts.
    """
    final = step.number == data.n_steps - 1
    title = f"Step {step.number}" + (" — final" if final else "")
    if short:
        return title
    kept, rejected = split_kept_from(step, data.final_centres, data.rejected)
    counts = (
        f"{len(triangles_xy_from(step.initial_triangles)):,} tested, {len(kept)} kept"
    )
    if len(rejected):
        counts += f" + {len(rejected)} rejected"
    return f"{title} ({counts})"


# The placement of image numbers is shared with the point-source fit panels (``positions_panel``), so an image's
# number sits beside it the same way in every point-source figure.
_NUMBER_PLACEMENTS = NUMBER_PLACEMENTS
NUMBER_FONTSIZE = 14

# Rejected-candidate labels (``r1``, ``r2``, ...) are this fraction of the image-number font size.
REJECTED_LABEL_SCALE = 0.9


def _points_per_arcsec(data: PointSolverFigureData, panel_inches: float) -> float:
    width = max(data.extent[1] - data.extent[0], data.extent[3] - data.extent[2])
    return panel_inches * 72.0 / width


def _to_points(data: PointSolverFigureData, xy: np.ndarray, panel_inches: float):
    """(x,y) arc-seconds -> (x,y) points from the panel's lower-left corner."""
    scale = _points_per_arcsec(data, panel_inches)
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    return (xy - np.array([data.extent[0], data.extent[2]])) * scale


def _panel_points(data: PointSolverFigureData, panel_inches: float):
    scale = _points_per_arcsec(data, panel_inches)
    return (
        (data.extent[1] - data.extent[0]) * scale,
        (data.extent[3] - data.extent[2]) * scale,
    )


def image_labels_from(data: PointSolverFigureData) -> List[str]:
    """
    Returns the step-0 labels: ``1, 2, ...`` for the solved images then ``r1, r2, ...`` for the rejected
    candidates. Either set is left unlabelled (empty) beyond ``MAX_NUMBERED_IMAGES``, so a crowded panel stays
    readable.
    """
    labels = []
    if len(data.solved) <= MAX_NUMBERED_IMAGES:
        labels += [str(index + 1) for index in range(len(data.solved))]
    else:
        labels += [""] * len(data.solved)
    if len(data.rejected) <= MAX_NUMBERED_IMAGES:
        labels += [f"r{index + 1}" for index in range(len(data.rejected))]
    else:
        labels += [""] * len(data.rejected)
    return labels


def number_placements_from(
    data: PointSolverFigureData, panel_inches: float
) -> List[Tuple[float, float, str, str]]:
    """
    Returns, for every solved image then every rejected candidate, the ``(dx, dy, ha, va)`` offset (points) of its
    step-0 label.

    Each label sits a few points off its own marker (one of ``_NUMBER_PLACEMENTS``): the first placement which
    stays inside the panel and overlaps neither another marker nor an already-placed label, else the one overlapping
    least. A label is therefore never moved away from its marker to find space.
    """
    markers = _to_points(
        data,
        np.vstack([data.solved[:, ::-1], data.rejected[:, ::-1]]),
        panel_inches,
    )
    width, height = _panel_points(data, panel_inches)
    return adjacent_number_placements_from(
        markers=markers,
        labels=[label or " " for label in image_labels_from(data)],
        width=width,
        height=height,
        fontsize=NUMBER_FONTSIZE,
    )


def _annotate_images(ax, data: PointSolverFigureData, panel_inches: float):
    """
    Labels the solved images ``1, 2, ...`` (bold) and the rejected candidates ``r1, r2, ...`` (grey italic), each
    beside its own marker.
    """
    points = np.vstack([data.solved, data.rejected])
    labels = image_labels_from(data)
    if len(points) == 0:
        return
    placements = number_placements_from(data, panel_inches)
    for (y, x), label, (dx, dy, ha, va) in zip(points, labels, placements):
        if not label:
            continue
        rejected = label.startswith("r")
        ax.annotate(
            label,
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=NUMBER_FONTSIZE * (REJECTED_LABEL_SCALE if rejected else 1.0),
            weight="normal" if rejected else "bold",
            style="italic" if rejected else "normal",
            color=REJECTED_LABEL_COLOR if rejected else "k",
            zorder=10,
            annotation_clip=True,
            bbox=dict(fc="w", ec="none", alpha=0.8, pad=0.5),
        )


def _observed_style(family_index):
    if family_index is None:
        return OBSERVED_COLOR, "o"

    from autolens.point.plot.cluster_sky import indexed_family_style

    _, color, marker = indexed_family_style(family_index)
    return color, marker


def draw_point_solver_panel(
    ax,
    step,
    data: PointSolverFigureData,
    panel_inches: float,
    title: Optional[str] = None,
    tick_fontsize: float = 10,
    title_fontsize: float = 12,
    annotation_fontsize: float = 10,
    family_index=None,
) -> bool:
    """
    Draws one refinement step on ``ax``: the panel shared by the 2x2 still and every animation frame.

    Step 0 is drawn as an overview (no up-sampled triangles, hairline outlines on a dense tiling) with the images
    numbered ``1, 2, ...`` and the rejected candidates ``r1, r2, ...``. Every panel uses the same fixed extent,
    with standard arc-second tick labels and no axis-label text. ``annotation_fontsize`` is no longer used (the
    labels share the image-number font size) and is kept for backwards compatibility.

    Returns ``True`` if the kept triangles were drawn as centroid markers (too small to draw), so the caller can
    add the matching legend entry.
    """
    from matplotlib.collections import PolyCollection

    from autoarray.plot.utils import apply_extent

    overview = step.number == 0
    tested = triangles_xy_from(step.initial_triangles)

    width = max(data.extent[1] - data.extent[0], data.extent[3] - data.extent[2])
    if _edge_of(step) / width * panel_inches * 72.0 < MIN_OUTLINE_POINTS:
        ax.add_collection(
            PolyCollection(
                tested,
                facecolor=TESTED_COLOR,
                alpha=0.18,
                edgecolor="none",
                zorder=1,
            )
        )
    else:
        ax.add_collection(
            PolyCollection(
                tested,
                facecolor="none",
                edgecolor=TESTED_COLOR,
                lw=0.2 if overview and len(tested) > DENSE_TILING else 0.56,
                zorder=1,
            )
        )
    ax.add_collection(
        PolyCollection(
            neighbour_rim_from(step),
            facecolor=NEIGHBOUR_COLOR,
            alpha=0.5,
            edgecolor=NEIGHBOUR_COLOR,
            lw=0.48,
            zorder=2,
        )
    )

    kept, rejected = split_kept_from(step, data.final_centres, data.rejected)
    as_markers = is_too_small_to_draw(step, data, panel_inches)
    if as_markers:
        for triangles, color in ((kept, KEPT_COLOR), (rejected, REJECTED_COLOR)):
            if len(triangles):
                centres = triangles.mean(axis=1)
                ax.scatter(
                    centres[:, 0],
                    centres[:, 1],
                    s=34,
                    marker="^",
                    c=color,
                    edgecolors=KEPT_COLOR,
                    linewidths=0.6,
                    zorder=9,
                )
    else:
        ax.add_collection(
            PolyCollection(
                kept,
                facecolor=KEPT_COLOR,
                alpha=0.85,
                edgecolor=KEPT_COLOR,
                lw=0.64,
                zorder=3,
            )
        )
        ax.add_collection(
            PolyCollection(
                rejected,
                facecolor=REJECTED_COLOR,
                alpha=0.85,
                edgecolor=KEPT_COLOR,
                hatch="////",
                lw=0.64,
                zorder=3,
            )
        )

    if not overview:
        ax.add_collection(
            PolyCollection(
                triangles_xy_from(step.up_sampled),
                facecolor="none",
                edgecolor=UP_SAMPLED_COLOR,
                lw=0.4,
                zorder=4,
            )
        )

    for lines, color in (
        (data.lines, CRITICAL_CURVE_COLOR),
        (data.radial_lines, RADIAL_CRITICAL_CURVE_COLOR),
    ):
        for line in lines:
            line = np.asarray(line)
            if line.ndim == 2 and len(line):
                ax.plot(
                    line[:, 1],
                    line[:, 0],
                    color=color,
                    lw=CRITICAL_CURVE_LINEWIDTH,
                    zorder=5,
                )

    if data.positions is not None and len(data.positions):
        observed_color, observed_marker = _observed_style(family_index)
        ax.scatter(
            data.positions[:, 1],
            data.positions[:, 0],
            s=60,
            facecolors="none",
            edgecolors=observed_color,
            marker=observed_marker,
            linewidths=1.3,
            zorder=7,
        )
    if len(data.solved):
        ax.scatter(
            data.solved[:, 1],
            data.solved[:, 0],
            s=45,
            c=SOLVED_COLOR,
            marker="x",
            linewidths=1.4,
            zorder=8,
        )

    if overview:
        _annotate_images(ax, data, panel_inches=panel_inches)

    apply_extent(ax, data.extent)
    ax.set_aspect("equal")
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelrotation=90)
    ax.set_title(
        panel_title_from(step, data) if title is None else title,
        fontsize=title_fontsize,
    )
    return as_markers


def legend_handles_from(
    data: PointSolverFigureData, marked: bool, family_index=None
) -> list:
    """
    Returns the legend handles shared by the still and the animation; the ``Kept (too small to draw)`` entry is
    included only when a panel drew its kept triangles as centroid markers.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    tested = "Triangles tested"
    if data.display_pixel_scale is not None:
        tested += " (display grid)"
    handles = [
        Patch(fc="none", ec=TESTED_COLOR, label=tested),
        Patch(fc=KEPT_COLOR, alpha=0.85, label="Kept (contains source)"),
        Patch(
            fc=REJECTED_COLOR,
            ec=KEPT_COLOR,
            hatch="////",
            label=f"Kept, then rejected (μ < {data.magnification_threshold:g})",
        ),
        Patch(fc=NEIGHBOUR_COLOR, alpha=0.5, label="Neighbours"),
        Patch(fc="none", ec=UP_SAMPLED_COLOR, label="Up-sampled (next step)"),
    ]
    if marked:
        handles.append(
            Line2D(
                [],
                [],
                marker="^",
                ls="",
                color=KEPT_COLOR,
                label="Kept (too small to draw)",
            )
        )
    if len(data.lines):
        handles.append(
            Line2D(
                [],
                [],
                color=CRITICAL_CURVE_COLOR,
                lw=CRITICAL_CURVE_LINEWIDTH,
                label="Critical curve",
            )
        )
    if len(data.radial_lines):
        handles.append(
            Line2D(
                [],
                [],
                color=RADIAL_CRITICAL_CURVE_COLOR,
                lw=CRITICAL_CURVE_LINEWIDTH,
                label="Radial critical curve",
            )
        )
    if data.positions is not None and len(data.positions):
        observed_color, observed_marker = _observed_style(family_index)
        handles.append(
            Line2D(
                [],
                [],
                marker=observed_marker,
                ls="",
                mfc="none",
                mec=observed_color,
                label="Observed",
            )
        )
    handles.append(
        Line2D([], [], marker="x", ls="", color=SOLVED_COLOR, label="Solved")
    )
    return handles


def warn_if_near_jax_cap(data: PointSolverFigureData, fraction: float = 0.8):
    """
    Logs a warning when a step keeps nearly as many triangles as the JAX solver's static ``MAX_CONTAINING_SIZE``.

    The JAX path truncates every step to that many kept triangles, so a model which keeps more (many images, or a
    source near a caustic) can lose images under JAX that this NumPy re-solve still shows.
    """
    from autoarray.structures.triangles.array import MAX_CONTAINING_SIZE

    if data.max_kept >= fraction * MAX_CONTAINING_SIZE:
        logger.warning(
            f"PointSolver figure: a refinement step kept {data.max_kept} triangles, close to (or above) the JAX "
            f"solver's MAX_CONTAINING_SIZE ({MAX_CONTAINING_SIZE}). The JAX path truncates each step to "
            f"{MAX_CONTAINING_SIZE} kept triangles, so a JAX model-fit may lose images this NumPy figure shows."
        )


def _resolve_format(output_format) -> str:
    from autoarray.plot.utils import _conf_output_format

    if output_format is None:
        output_format = _conf_output_format()
    if isinstance(output_format, (list, tuple)):
        output_format = output_format[0] if output_format else "show"
    return output_format


def _save_animation(
    data, output_path, filename, title_prefix, dpi=100, family_index=None
):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig_width, fig_height = 6.4, 7.3
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.94, bottom=0.2)
    panel_inches = 0.89 * fig_width

    marked = any(is_too_small_to_draw(s, data, panel_inches) for s in data.steps)
    fig.legend(
        handles=legend_handles_from(data, marked, family_index=family_index),
        loc="lower center",
        ncol=3,
        fontsize=8.5,
        frameon=False,
    )

    prefix = f"{title_prefix.rstrip()} " if title_prefix else ""
    last = len(data.steps) - 1
    frames = list(range(last)) + [last] * FINAL_FRAME_REPEATS

    def update(frame):
        ax.clear()
        step = data.steps[frame]
        draw_point_solver_panel(
            ax,
            step,
            data,
            panel_inches=panel_inches,
            title=prefix + panel_title_from(step, data, short=True),
            tick_fontsize=9,
            family_index=family_index,
        )

    animation = FuncAnimation(fig, update, frames=frames)

    if output_path is None:
        return animation

    from autoarray.plot.utils import _FAST_PLOTS

    if not _FAST_PLOTS:
        Path(output_path).mkdir(parents=True, exist_ok=True)
        animation.save(
            str(Path(output_path) / f"{filename}.gif"),
            writer=PillowWriter(fps=1.0 / FRAME_SECONDS),
            dpi=dpi,
        )
    plt.close(fig)
    return None


def plot_point_solver(
    solver,
    tracer,
    source_plane_coordinate: Tuple[float, float],
    positions=None,
    plane_redshift: Optional[float] = None,
    image_plane_lines=None,
    output_path: Optional[str] = None,
    output_format: str = None,
    title_prefix: str = None,
    animate: bool = False,
    filename: str = "point_solver",
    family_index=None,
):
    """
    Plot how the ``PointSolver`` refines its triangle tiling to find the multiple images of a point source.

    The figure (``point_solver.png``) is a 2x2 subplot of steps 0, 1, 2 and the final step on one fixed extent
    spanning the source's images: the triangles tested, kept (blue), kept then rejected by
    ``magnification_threshold`` (hatched), the neighbour rim (orange) and the triangles up-sampled for the next step
    (green), with the critical curves, the observed positions and the solved images. When the solver's triangles
    are too small to draw on that extent the steps are re-solved on a coarse display grid. See the module docstring
    for the full legend.

    With ``animate=True`` the same panels are written as a looping GIF (``point_solver.gif``), one step per frame
    (``FRAME_SECONDS`` each, the final step held longer), via ``matplotlib.animation`` and ``PillowWriter`` (no
    ffmpeg needed). Without an ``output_path`` the ``FuncAnimation`` is returned instead, e.g. for display in a
    notebook.

    The solve is repeated with NumPy whatever the solver's backend, as the JAX path pads and hides the steps.

    Parameters
    ----------
    solver
        The ``PointSolver`` whose refinement is shown.
    tracer
        The tracer that ray-traces the triangles to the source plane.
    source_plane_coordinate
        The (y,x) source-plane coordinate of the point source.
    positions
        The observed (y,x) positions, drawn as hollow circles.
    plane_redshift
        The redshift of the source's plane, for multi-plane lenses (e.g. clusters). ``None`` is the last plane.
    image_plane_lines
        Pre-computed critical-curve lines; ``None`` computes the source plane's own curves, ``[]`` draws none.
    output_path
        Directory in which to save the figure. If ``None`` the still is shown (and the animation returned).
    output_format
        Image format of the still, passed to :func:`~autoarray.plot.utils.save_figure`. The animation is always
        a GIF.
    title_prefix
        Text prepended to every panel title.
    animate
        If ``True`` write the GIF animation instead of the still.
    filename
        The file name (without extension) of the output.
    family_index
        The zero-based cluster family index, if any. It styles the observed positions and their legend key.
    """
    data = point_solver_data_from(
        solver=solver,
        tracer=tracer,
        source_plane_coordinate=source_plane_coordinate,
        positions=positions,
        plane_redshift=plane_redshift,
        image_plane_lines=image_plane_lines,
    )
    return plot_point_solver_from_data(
        data=data,
        output_path=output_path,
        output_format=output_format,
        title_prefix=title_prefix,
        animate=animate,
        filename=filename,
        family_index=family_index,
    )


def plot_point_solver_from_data(
    data: PointSolverFigureData,
    output_path: Optional[str] = None,
    output_format: str = None,
    title_prefix: str = None,
    animate: bool = False,
    filename: str = "point_solver",
    family_index=None,
):
    """
    Draws the point-solver still or animation from data already collected by :func:`point_solver_data_from`, so
    the still and the GIF can be written from one solve.
    """
    from autoarray.plot.utils import hide_unused_axes, save_figure, subplots

    if animate:
        return _save_animation(
            data=data,
            output_path=None if output_path is None else str(output_path),
            filename=filename,
            title_prefix=title_prefix,
            family_index=family_index,
        )

    n = len(data.steps)
    rows, cols = (2, 2) if n > 2 else (1, n)
    fig_width, fig_height = 6.0 * cols, 6.0 * rows + 0.6
    fig, axes = subplots(rows, cols, figsize=(fig_width, fig_height))
    axes_flat = list(np.array(axes).flatten())

    output_format = _resolve_format(output_format)
    panel_inches = 0.85 * fig_width / cols

    prefix = f"{title_prefix.rstrip()} " if title_prefix else ""
    marked = False
    for ax, step in zip(axes_flat, data.steps):
        marked |= draw_point_solver_panel(
            ax,
            step,
            data,
            panel_inches=panel_inches,
            title=prefix + panel_title_from(step, data),
            family_index=family_index,
        )
    hide_unused_axes(axes_flat[n:])

    fig.legend(
        handles=legend_handles_from(data, marked, family_index=family_index),
        loc="lower center",
        ncol=5,
        fontsize=10.5,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.6 / fig_height + 0.03, 1, 1), h_pad=2.0)
    save_figure(
        fig,
        path=None if output_path is None else str(output_path),
        filename=filename,
        format=output_format,
    )
