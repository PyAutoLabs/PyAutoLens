"""
The "Fit Source Plane" panel of the point-source ``fit.png``: the observed positions ray-traced to the source plane,
showing how closely they trace back together.

What the panel draws is paired to the positions fit class, so that it never shows a chi-squared the fit does not
compute (:func:`presentation_from`, one entry per class of ``autolens.point.fit.positions``):

- **Image-plane chi-squared** (``FitPositionsImagePair``, ``FitPositionsImagePairRepeat``,
  ``FitPositionsImagePairAll`` and their ``*Solved`` variants): the panel is context only. It shows the traced
  positions, the source centre and the caustics, with no noise footprints and no chi-squared box, as the fit's
  chi-squared is measured in the image plane.
- **Source-plane chi-squared** (``FitPositionsSource``, ``FitPositionsSourceSolved``): the footprint the
  chi-squared actually uses is drawn on every traced position:

  - ``weighting = "magnification"``: a circle of radius ``σ/|μ|`` (the scalar noise of
    ``χ²ᵢ = |β̂ᵢ − c|² µᵢ² / σᵢ²``), plus the ``A·σ`` ellipse dashed as the propagated noise (reference);
  - ``weighting = "jacobian"``: the ``A·σ`` ellipse, the source-plane covariance ``σ² A Aᵀ = Wᵢ⁻¹`` of the
    precision tensor ``Wᵢ = Aᵢ⁻ᵀΘᵢAᵢ⁻¹`` the chi-squared uses;

  and the legend gives the chi-squared the fit uses (only that one), as a handle-less entry ``χ² = <value>``.

The source centre is a red star for a free-centre fit (the point-source profile's ``centre``) and a red diamond for
a ``*Solved`` fit (the analytically solved ``β*``).

Every fit-type cue lives in the legend, as marker symbols and labels (the centre marker, the weighting-specific
noise label, the chi-squared entry): the panel carries no free-floating annotation text about the fit type.

The panel is zoomed onto the traced positions and the centre with a margin; an inset in the emptiest corner shows the
whole tangential caustic with a box marking the zoom. The caustics are those of the plane the point source is in
(``fit.plane_index``), so a multi-plane (e.g. cluster) point is drawn against its own caustics. The images are
numbered as on the image-plane panel.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from autolens.point.plot import positions_panel as pp

IMAGE_PLANE = "image"
SOURCE_PLANE = "source"

# Caustics are dark and thick (as the critical curves of the image-plane panels), so they read clearly.
CAUSTIC_COLOR = "0.2"
RADIAL_CAUSTIC_COLOR = "#5c5c00"
CAUSTIC_LINEWIDTH = 1.8
INSET_CAUSTIC_LINEWIDTH = 1.4
CENTRE_COLOR = pp.MODEL_COLOR
MAGNIFICATION_NOISE_COLOR = "tab:purple"
JACOBIAN_NOISE_COLOR = pp.NOISE_COLOR

# The zoom extends this fraction of its half-width beyond the traced positions and centre, on every side.
ZOOM_MARGIN = 0.3

# Size of the caustic inset, as a fraction of the panel.
INSET_SIZE = 0.3

# The propagated-noise reference ellipses under magnification weighting: dashed and darker than the primary
# footprint's colour so they read on white, but thinner than the primary σ/|μ| circles.
REFERENCE_NOISE_COLOR = "#c25400"


@dataclass(frozen=True)
class SourcePlanePresentation:
    """
    How the source-plane panel presents a positions fit (see :func:`presentation_from`).

    Parameters
    ----------
    chi_squared_plane
        ``"image"`` if the fit's chi-squared is measured in the image plane (context-only panel), ``"source"`` if
        it is measured in the source plane.
    solved_centre
        Whether the source centre is the analytically solved ``β*`` (a ``*Solved`` fit) rather than a free
        ``centre`` parameter.
    weighting
        The source-plane chi-squared weighting (``"magnification"`` or ``"jacobian"``); ``None`` for image-plane
        fits.
    """

    chi_squared_plane: str
    solved_centre: bool
    weighting: Optional[str] = None

    @property
    def is_source_plane(self) -> bool:
        return self.chi_squared_plane == SOURCE_PLANE

    @property
    def centre_marker(self) -> str:
        return "D" if self.solved_centre else "*"

    @property
    def centre_label(self) -> str:
        return (
            "Solved source centre (β*)" if self.solved_centre else "Model source centre"
        )


def presentation_table():
    """
    Returns the presentation of every positions fit class in ``autolens.point.fit.positions``, as a dict mapping
    the class to ``(chi_squared_plane, solved_centre)``.
    """
    from autolens.point.fit.positions.image.pair import FitPositionsImagePair
    from autolens.point.fit.positions.image.pair_all import (
        FitPositionsImagePairAll,
        FitPositionsImagePairAllSolved,
    )
    from autolens.point.fit.positions.image.pair_repeat import (
        FitPositionsImagePairRepeat,
        FitPositionsImagePairRepeatSolved,
    )
    from autolens.point.fit.positions.source.separations import (
        FitPositionsSource,
        FitPositionsSourceSolved,
    )

    return {
        FitPositionsImagePair: (IMAGE_PLANE, False),
        FitPositionsImagePairRepeat: (IMAGE_PLANE, False),
        FitPositionsImagePairRepeatSolved: (IMAGE_PLANE, True),
        FitPositionsImagePairAll: (IMAGE_PLANE, False),
        FitPositionsImagePairAllSolved: (IMAGE_PLANE, True),
        FitPositionsSource: (SOURCE_PLANE, False),
        FitPositionsSourceSolved: (SOURCE_PLANE, True),
    }


def presentation_from(fit_positions) -> SourcePlanePresentation:
    """
    Returns how the source-plane panel presents a positions fit, paired to its class.

    The class is looked up in :func:`presentation_table` along its method resolution order, so a user subclass of a
    fit class (e.g. ``FitPositionsSource`` with ``weighting = "magnification"``) is presented as its parent. The weighting
    of a source-plane fit is read from the fit itself. A class with no entry is presented as context only, which
    never draws a chi-squared the fit may not compute.
    """
    table = presentation_table()
    for cls in type(fit_positions).__mro__:
        if cls in table:
            plane, solved = table[cls]
            break
    else:
        from autolens.point.fit.solved import SolvedCentre

        plane, solved = IMAGE_PLANE, isinstance(fit_positions, SolvedCentre)

    weighting = (
        getattr(fit_positions, "weighting", None) if plane == SOURCE_PLANE else None
    )
    return SourcePlanePresentation(
        chi_squared_plane=plane, solved_centre=solved, weighting=weighting
    )


def _np(values) -> np.ndarray:
    values = values.array if hasattr(values, "array") else values
    return np.asarray(values, dtype=float)


def traced_positions_from(fit_positions) -> np.ndarray:
    """
    Returns the observed positions ray-traced to the plane of the point source, as a (N, 2) (y,x) array.

    This is the same ray-tracing as ``FitPositionsSource.model_data`` (multi-plane deflections to
    ``fit.plane_index`` when the tracer has more than two planes), applied to any positions fit.
    """
    import autoarray as aa

    tracer = fit_positions.tracer
    # A NumPy copy of the positions: the figure is NumPy-only, even for a fit built with `use_jax=True` (whose data
    # grid is JAX-backed).
    positions = aa.Grid2DIrregular(values=pp._np_grid(fit_positions.data))

    if len(tracer.planes) <= 2:
        deflections = tracer.deflections_yx_2d_from(grid=positions, xp=np)
    else:
        deflections = tracer.deflections_between_planes_from(
            grid=positions, xp=np, plane_i=0, plane_j=fit_positions.plane_index
        )
    return pp._np_grid(
        positions.grid_2d_via_deflection_grid_from(deflection_grid=deflections, xp=np)
    )


def jacobian_covariances_from(fit_positions) -> np.ndarray:
    """
    Returns the (N, 2, 2) source-plane covariance ``Cᵢ = Wᵢ⁻¹ = σᵢ² Aᵢ Aᵢᵀ`` of every traced position, in (y,x)
    order, where ``Wᵢ`` is the jacobian precision tensor of
    :func:`~autolens.point.fit.solved.precision_tensor_components_from`. A position whose tensor is singular (on
    a critical curve) has a ``NaN`` covariance.
    """
    from autolens.point.fit.solved import precision_tensor_components_from

    w11, w12, w21, w22 = (
        _np(w) for w in precision_tensor_components_from(fit_positions, "jacobian")
    )
    precision = np.stack(
        [np.stack([w11, w12], axis=-1), np.stack([w21, w22], axis=-1)], axis=-2
    )
    covariances = np.full_like(precision, np.nan)
    for i, w in enumerate(precision):
        if np.all(np.isfinite(w)) and abs(np.linalg.det(w)) > 0:
            covariances[i] = np.linalg.inv(w)
    return covariances


def ellipse_from(covariance: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """
    Returns the ``(semi_major, semi_minor, angle_deg)`` of the 1-sigma ellipse of a (y,x) covariance, the angle
    measured anticlockwise from the x axis, or ``None`` for a non-finite covariance.
    """
    if not np.all(np.isfinite(covariance)):
        return None
    values, vectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    values = np.clip(values, 0.0, None)
    major = vectors[:, 1]
    angle = float(np.degrees(np.arctan2(major[0], major[1])))
    return float(np.sqrt(values[1])), float(np.sqrt(values[0])), angle


def chi_squared_from(residuals: np.ndarray, covariances: np.ndarray) -> float:
    """
    Returns ``Σᵢ rᵢᵀ Cᵢ⁻¹ rᵢ`` for (y,x) residuals ``rᵢ`` and covariances ``Cᵢ``, skipping non-finite terms.
    """
    total = 0.0
    for r, c in zip(residuals, covariances):
        if np.all(np.isfinite(c)) and np.all(np.isfinite(r)):
            total += float(r @ np.linalg.solve(c, r))
    return total


@dataclass
class SourcePlaneData:
    """
    Everything the source-plane panel draws, read off a positions fit.
    """

    presentation: SourcePlanePresentation
    traced: np.ndarray
    centre: np.ndarray
    numbers: np.ndarray
    sigma: np.ndarray
    magnification_radii: np.ndarray
    covariances: np.ndarray
    chi_squared: Optional[float]


def source_plane_data_from(fit_positions, numbers=None) -> SourcePlaneData:
    """
    Returns the source-plane quantities of a positions fit, computed with the fit's own machinery so the drawn
    noise is exactly that of its chi-squared.

    Parameters
    ----------
    fit_positions
        Any positions fit (``fit.positions`` of a ``FitPointDataset``).
    numbers
        The image number of each observed position; ``None`` uses
        :func:`~autolens.point.plot.positions_panel.image_numbers_from`.
    """
    presentation = presentation_from(fit_positions)

    traced = traced_positions_from(fit_positions)
    centre = np.asarray(fit_positions.source_plane_coordinate, dtype=float).ravel()[:2]
    sigma = np.broadcast_to(_np(fit_positions.noise_map).ravel(), (len(traced),)).copy()

    try:
        mu = np.abs(_np(fit_positions.magnifications_at_positions).ravel())
        with np.errstate(divide="ignore"):
            radii = sigma / mu
    except Exception:
        radii = np.full(len(traced), np.nan)

    try:
        covariances = jacobian_covariances_from(fit_positions)
    except Exception:
        covariances = np.full((len(traced), 2, 2), np.nan)

    chi_squared = None
    if presentation.is_source_plane:
        chi_squared = float(np.sum(_np(fit_positions.chi_squared_map)))

    if numbers is None:
        numbers = pp.image_numbers_from(fit_positions.data)

    return SourcePlaneData(
        presentation=presentation,
        traced=traced,
        centre=centre,
        numbers=np.asarray(numbers),
        sigma=sigma,
        magnification_radii=radii,
        covariances=covariances,
        chi_squared=chi_squared,
    )


def point_plane_caustics_from(
    tracer, plane_index: int, grid
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Returns the ``(tangential, radial)`` caustics of the plane ``plane_index`` the point source is in, as lists of
    (N, 2) (y,x) arrays, computed by marching squares on the image-plane ``grid``.

    For a two-plane tracer this is the tracer's own source plane; for a multi-plane tracer the caustics are those of
    the deflections from the image plane to ``plane_index`` (``LensCalc.from_tracer(..., plane_j=plane_index)``),
    the same deflections the fit traces the positions with. Radial caustics are only computed when a radial critical
    curve is resolved by the grid. Returns empty lists when ``PYAUTO_FAST_PLOTS=1`` or the model has no caustics.
    """
    import os

    if os.environ.get("PYAUTO_FAST_PLOTS") == "1":
        return [], []

    from autogalaxy.operate.lens_calc import LensCalc

    def _lines(curves):
        lines = []
        for curve in curves or []:
            line = pp._np_grid(curve)
            line = line[np.isfinite(line).all(axis=1)]
            if len(line) > 1:
                lines.append(line)
        return lines

    use_multi_plane = len(tracer.planes) > 2
    try:
        lens_calc = LensCalc.from_tracer(
            tracer=tracer,
            use_multi_plane=use_multi_plane,
            plane_i=0,
            plane_j=plane_index if use_multi_plane else -1,
        )
        tangential = _lines(lens_calc.tangential_caustic_list_from(grid=grid))
    except Exception:
        return [], []

    radial = []
    try:
        areas = lens_calc.radial_critical_curve_area_list_from(grid=grid)
        if any(area > grid.pixel_scale for area in areas):
            radial = _lines(lens_calc.radial_caustic_list_from(grid=grid))
    except Exception:
        radial = []

    return tangential, radial


def caustics_from_lines(
    lines, line_colors
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Splits pre-computed source-plane lines into ``(tangential, radial)`` caustics by their colour (``"white"``
    tangential, ``"yellow"`` radial, as returned by the critical-curve helper).
    """
    tangential, radial = [], []
    for i, line in enumerate(lines or []):
        line = np.asarray(line, dtype=float)
        if line.ndim != 2 or len(line) < 2:
            continue
        color = line_colors[i] if line_colors and i < len(line_colors) else "white"
        (radial if color == "yellow" else tangential).append(line)
    return tangential, radial


def zoom_extent_from(data: SourcePlaneData, margin: float = ZOOM_MARGIN) -> List[float]:
    """
    Returns the ``[x_min, x_max, y_min, y_max]`` square zoom of the panel: the traced positions and the centre,
    with ``margin`` of the half-width added on every side.

    The half-width is at least three times the largest ``σ/|μ|`` (so a near-perfect fit is not zoomed to a point)
    and, for a source-plane fit, at least the largest footprint its chi-squared uses, so every footprint is visible.
    """
    points = np.vstack([data.traced, data.centre[None, :]])
    points = points[np.isfinite(points).all(axis=1)]
    lo, hi = points.min(axis=0), points.max(axis=0)
    half = 0.5 * float(np.max(hi - lo))

    radii = data.magnification_radii[np.isfinite(data.magnification_radii)]
    floor = 3.0 * float(radii.max()) if len(radii) else 0.0

    if data.presentation.is_source_plane:
        if data.presentation.weighting == "magnification" and len(radii):
            floor = max(floor, float(radii.max()))
        elif data.presentation.weighting == "jacobian":
            majors = [e[0] for e in map(ellipse_from, data.covariances) if e]
            if majors:
                floor = max(floor, max(majors))

    if floor <= 0.0:
        floor = 0.05 * max(float(np.max(np.abs(points))), 1e-3)

    half = max(half, floor) * (1.0 + 2.0 * margin)
    centre_y, centre_x = 0.5 * (lo + hi)
    return [centre_x - half, centre_x + half, centre_y - half, centre_y + half]


CORNER_LOCATIONS = {
    (0, 0): "lower left",
    (1, 0): "lower right",
    (0, 1): "upper left",
    (1, 1): "upper right",
}


def ranked_corners_from(
    extent: List[float], points: np.ndarray, size: float = INSET_SIZE, pad: float = 0.02
) -> List[Tuple[Tuple[int, int], Tuple[float, float, float, float]]]:
    """
    Returns the panel's corners from emptiest to fullest, as ``((i, j), (x0, y0, width, height))`` with the box in
    axes fraction: ranked by how few of the drawn ``points`` (y,x) a ``size`` box in the corner holds, then by how
    far the nearest point is from its centre.
    """
    x_min, x_max, y_min, y_max = extent
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    fx = (points[:, 1] - x_min) / (x_max - x_min)
    fy = (points[:, 0] - y_min) / (y_max - y_min)
    ranked = []
    for i, x0 in enumerate((pad, 1.0 - pad - size)):
        for j, y0 in enumerate((pad, 1.0 - pad - size)):
            inside = int(
                np.sum(
                    (fx > x0 - 0.03)
                    & (fx < x0 + size + 0.03)
                    & (fy > y0 - 0.03)
                    & (fy < y0 + size + 0.03)
                )
            )
            distance = (
                float(np.min(np.hypot(fx - (x0 + size / 2), fy - (y0 + size / 2))))
                if len(points)
                else 0.0
            )
            ranked.append(((inside, -distance), (i, j), (x0, y0, size, size)))
    ranked.sort(key=lambda r: r[0])
    return [(r[1], r[2]) for r in ranked]


def _footprint_points(data: SourcePlaneData) -> np.ndarray:
    """
    The traced positions, the centre and the ends of every drawn ellipse, used to find empty corners.
    """
    points = [data.traced, data.centre[None, :]]
    covariances = data.covariances if data.presentation.is_source_plane else []
    for (y, x), covariance in zip(data.traced, covariances):
        ellipse = ellipse_from(covariance)
        if ellipse is None:
            continue
        semi_major, _, angle = ellipse
        t = np.radians(angle)
        d = semi_major * np.array([np.sin(t), np.cos(t)])
        points.append(np.array([[y + d[0], x + d[1]], [y - d[0], x - d[1]]]))
    points = np.vstack(points)
    return points[np.isfinite(points).all(axis=1)]


def _plot_caustics(ax, tangential, radial, linewidth=CAUSTIC_LINEWIDTH, label=True):
    labelled = False
    for line, color in [(c, CAUSTIC_COLOR) for c in tangential] + [
        (c, RADIAL_CAUSTIC_COLOR) for c in radial
    ]:
        ax.plot(
            line[:, 1],
            line[:, 0],
            color=color,
            linewidth=linewidth,
            zorder=2,
            label="Caustic" if (label and not labelled) else None,
        )
        labelled = True


def _plot_centre(ax, data: SourcePlaneData, size=180, label=True):
    presentation = data.presentation
    ax.scatter(
        [data.centre[1]],
        [data.centre[0]],
        s=size if presentation.centre_marker == "*" else 0.5 * size,
        c=CENTRE_COLOR,
        marker=presentation.centre_marker,
        linewidths=0.6,
        edgecolors="white",
        zorder=8,
        label=presentation.centre_label if label else None,
    )


def _milliarcsec_range(values: np.ndarray) -> str:
    values = 1e3 * values[np.isfinite(values)]
    if len(values) == 0:
        return "n/a"
    if np.allclose(values, values[0]):
        return f"{values[0]:.2g} mas"
    return f"{values.min():.2g}–{values.max():.2g} mas"


def _format_chi_squared(value: float) -> str:
    return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:.3g}"


def chi_squared_label_from(data: SourcePlaneData) -> Optional[str]:
    """
    Returns the legend entry of the chi-squared the fit uses, e.g. ``"χ² = 10,023"``, or ``None`` for an
    image-plane fit (which has no source-plane chi-squared to show).
    """
    if not data.presentation.is_source_plane or data.chi_squared is None:
        return None
    return f"χ² = {_format_chi_squared(data.chi_squared)}"


def _plot_footprints(ax, data: SourcePlaneData):
    """
    Draws the noise footprint the fit's chi-squared uses on every traced position (source-plane fits only).
    """
    from matplotlib.patches import Circle, Ellipse

    presentation = data.presentation
    if not presentation.is_source_plane:
        return

    def _ellipses(**kwargs):
        drawn = False
        for (y, x), covariance in zip(data.traced, data.covariances):
            ellipse = ellipse_from(covariance)
            if ellipse is None:
                continue
            semi_major, semi_minor, angle = ellipse
            ax.add_patch(
                Ellipse(
                    (x, y),
                    2.0 * semi_major,
                    2.0 * semi_minor,
                    angle=angle,
                    fill=False,
                    **kwargs,
                )
            )
            drawn = True
        return drawn

    if presentation.weighting == "magnification":
        reference = dict(
            edgecolor=REFERENCE_NOISE_COLOR,
            linewidth=1.0,
            linestyle="--",
            alpha=0.9,
            zorder=4.8,
        )
        if _ellipses(**reference):
            ax.plot(
                [],
                [],
                color=REFERENCE_NOISE_COLOR,
                linewidth=1.0,
                linestyle="--",
                alpha=0.9,
                label="Propagated noise A·σ (reference)",
            )
        for (y, x), radius in zip(data.traced, data.magnification_radii):
            if np.isfinite(radius) and radius > 0:
                ax.add_patch(
                    Circle(
                        (x, y),
                        radius,
                        fill=False,
                        edgecolor=MAGNIFICATION_NOISE_COLOR,
                        linewidth=1.2,
                        zorder=5.5,
                    )
                )
        ax.plot(
            [],
            [],
            color=MAGNIFICATION_NOISE_COLOR,
            linewidth=1.2,
            label=f"χ² noise σ/|μ| ({_milliarcsec_range(data.magnification_radii)})",
        )
    else:
        if _ellipses(edgecolor=JACOBIAN_NOISE_COLOR, linewidth=1.2, zorder=5.5):
            ax.plot(
                [],
                [],
                color=JACOBIAN_NOISE_COLOR,
                linewidth=1.2,
                label="χ² noise A·σ (jacobian)",
            )


def _add_inset(ax, data: SourcePlaneData, tangential, radial, zoom, bounds):
    """
    Adds the context inset: the whole tangential caustic (radial caustics are drawn but not framed, as for a
    cluster they can be far larger), the centre and a box marking the main panel's zoom.
    """
    from matplotlib.patches import Rectangle

    framed = list(tangential) or list(radial)
    if not framed:
        return None

    inset = ax.inset_axes(bounds)
    _plot_caustics(
        inset, tangential, radial, linewidth=INSET_CAUSTIC_LINEWIDTH, label=False
    )
    _plot_centre(inset, data, size=50, label=False)

    points = np.vstack(framed + [data.traced, data.centre[None, :]])
    points = points[np.isfinite(points).all(axis=1)]
    lo, hi = points.min(axis=0), points.max(axis=0)
    half = 0.55 * float(np.max(hi - lo))
    centre_y, centre_x = 0.5 * (lo + hi)
    inset.set_xlim(centre_x - half, centre_x + half)
    inset.set_ylim(centre_y - half, centre_y + half)
    inset.set_aspect("equal", adjustable="box")

    x_min, x_max, y_min, y_max = zoom
    # The zoom is often far smaller than an inset pixel: keep its box visible.
    minimum = 0.04 * 2.0 * half
    width, height = max(x_max - x_min, minimum), max(y_max - y_min, minimum)
    x_mid, y_mid = 0.5 * (x_min + x_max), 0.5 * (y_min + y_max)
    inset.add_patch(
        Rectangle(
            (x_mid - width / 2.0, y_mid - height / 2.0),
            width,
            height,
            fill=False,
            edgecolor="k",
            linewidth=1.0,
            zorder=6,
        )
    )
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_edgecolor("0.5")
    inset.patch.set_alpha(0.92)
    return inset


def plot_source_plane_panel(
    ax,
    data: SourcePlaneData,
    title: str,
    tangential_caustics=None,
    radial_caustics=None,
    point_colors=None,
    legend: bool = True,
    point_marker: str = "o",
):
    """
    Draws the "Fit Source Plane" panel of a positions fit on ``ax`` (see the module docstring).

    Parameters
    ----------
    ax
        The matplotlib axes drawn on.
    data
        The panel's quantities, from :func:`source_plane_data_from`.
    title
        The panel title, e.g. ``"Fit Source Plane (point_0)"``.
    tangential_caustics, radial_caustics
        The caustics of the point source's plane, as lists of (N, 2) (y,x) arrays.
    point_colors
        The colour of each traced position (black if ``None``), e.g. the per-image colours of ``fit_per_image.png``.
    legend
        Whether the panel draws its legend (``fit_per_image.png`` draws none).
    point_marker
        The marker shape for the traced positions.
    """
    from autoarray.plot.utils import apply_extent, apply_labels

    tangential = list(tangential_caustics or [])
    radial = list(radial_caustics or [])

    zoom = zoom_extent_from(data)
    corners = ranked_corners_from(zoom, _footprint_points(data))

    has_inset = bool(tangential or radial)
    inset_corner, inset_bounds = corners[0]
    remaining = corners[1:] if has_inset else corners
    legend_corner = remaining[0][0]

    _plot_caustics(ax, tangential, radial)
    _plot_footprints(ax, data)

    traced = data.traced
    ax.scatter(
        traced[:, 1],
        traced[:, 0],
        s=110,
        c=pp.OBSERVED_COLOR if point_colors is None else list(point_colors),
        marker=point_marker,
        edgecolors="k",
        linewidths=0.6,
        zorder=4,
        label=f"Observed, traced ({pp.sigma_label_from(data.sigma)})",
    )
    _plot_centre(ax, data)

    apply_labels(ax, title=title, xlabel="", ylabel="")
    apply_extent(ax, zoom)
    ax.set_aspect("equal", adjustable="box")

    pp.plot_image_numbers(ax, traced, data.numbers, zoom)

    chi_squared_label = chi_squared_label_from(data)
    if chi_squared_label is not None:
        # A handle-less entry: the chi-squared the fit uses, in the legend with the other fit-type cues.
        ax.plot([], [], linestyle="none", marker="none", label=chi_squared_label)

    if legend:
        ax.legend(
            loc=CORNER_LOCATIONS[legend_corner],
            fontsize=8,
            framealpha=0.85,
            markerscale=0.6,
        )

    if has_inset:
        _add_inset(ax, data, tangential, radial, zoom, inset_bounds)

    return zoom
