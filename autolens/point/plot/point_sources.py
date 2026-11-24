"""
The per-source quantities every point-source positions figure draws, for one or many point sources.

A cluster model-fit fits several ``PointDataset`` objects (one per multiply imaged source, each at its own redshift),
so the positions figures (``dataset_point.png``, ``fit.png``, ``fit_per_image.png``, ``image_with_points.png`` and
``fit_image_with_points.png``) are drawn from a list of :class:`SourceRecord` objects, one per source. A single
point-source fit is the one-element case, so every figure uses one function and one visual language whatever the
number of sources.

This module holds:

- :class:`SourceRecord`: the observed positions, noise, image numbers and (for a fit) the model positions and the
  fit's own pairing of a single source, built by :func:`source_record_from_dataset` or
  :func:`source_record_from_fit` from the helpers of :mod:`autolens.point.plot.positions_panel`.
- :func:`assign_styles`: the colour and marker of every source. Source 0 has a black base colour (overridden by the observed-position style); sources 1 to 7
  step through an Okabe-Ito colour AND a marker shape. Beyond 8 sources hue cannot carry identity, so the colour
  follows the source redshift (a perceptually uniform map with a colourbar) and identity is carried by the ``k·i``
  labels.
- :func:`curves_for_planes`: the critical curves and caustics of each source's OWN plane. Each source plane has its
  own curves (a source at higher redshift is lensed more strongly), so drawing the last plane's curves for every
  source is wrong for a multi-plane cluster. The deflection field to a plane is evaluated once on a fine grid,
  differentiated numerically and contoured where the tangential / radial eigenvalue is zero (marching squares). When
  every mass-bearing galaxy is in one lens plane the deflection to any later plane is the last plane's rescaled by a
  distance ratio, so the field is evaluated once for all planes. This takes well under a second for a two-plane
  cluster, where ``LensCalc`` takes tens of seconds per plane. The curves, magnifications and solves always use the
  FULL tracer, as multi-plane deflections are normalised to the last plane.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from autolens.point.plot import positions_panel as pp

# Sources are distinguished by colour (up to this many) and marker shape.
MAX_QUALITATIVE = 8

# Source 0 is black (the single-source look); then Okabe-Ito, re-ordered so source 1 (the common two-source case) is
# orange and source 2 sky blue, which read both on white panels and on the dark image-with-points colormap.
# Vermilion is absent: red is the model colour.
SOURCE_PALETTE = [
    "k",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#CC79A7",
    "#0072B2",
    "#882255",
    "#999933",
]

# Marker shapes cycled per source (with colour).
MARKERS = ["o", "s", "^", "D", "v", "p", "h", "<", ">", "P"]

# Per-image colours inside one source's row of ``fit_per_image.png`` (linking a ranking dot to its image): Okabe-Ito
# in image-number order.
IMAGE_COLORS = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#882255",
    "#117733",
    "#999933",
]

# Number of grid points per side of the deflection-field grid the curves are contoured on.
CURVE_GRID_SHAPE = 400


def image_color(number: int):
    """
    Returns the colour of image ``number`` (1-based) within one source's row of ``fit_per_image.png``.
    """
    return IMAGE_COLORS[(int(number) - 1) % len(IMAGE_COLORS)]


def letters_from(total: int) -> List[str]:
    """
    Returns ``["a", "b", ...]``: the labels of a source's unmatched model images.
    """
    return [chr(ord("a") + i) for i in range(total)]


@dataclass
class SourceRecord:
    """
    Everything the positions figures draw for one point source.

    ``model``, ``indexes``, ``unmatched`` and ``delta`` describe the fit's own image-plane pairing (see
    :func:`~autolens.point.plot.positions_panel.model_pairing_from`); they are empty / ``-1`` / ``NaN`` for a dataset
    (no fit) and for a source-plane fit (whose model data are in the source plane).
    """

    k: int
    name: str
    redshift: Optional[float]
    observed: np.ndarray
    sigma: np.ndarray
    numbers: np.ndarray
    plane_index: Optional[int] = None
    fit: object = None
    model: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    indexes: Optional[np.ndarray] = None
    unmatched: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    no_model: Optional[np.ndarray] = None
    delta: Optional[np.ndarray] = None
    mu_model: np.ndarray = field(default_factory=lambda: np.zeros(0))
    linestyle: str = "-"
    residual_label: str = "Residual"
    model_label: str = "Model"
    chi_squared: Optional[float] = None
    mixture: bool = False
    image_plane_chi: Optional[np.ndarray] = None
    color: object = "k"
    marker: str = "o"
    curve_key: object = None

    def __post_init__(self):
        n = len(self.observed)
        if self.indexes is None:
            self.indexes = np.full(n, -1, dtype=int)
        if self.no_model is None:
            self.no_model = np.zeros(n, dtype=bool)
        if self.delta is None:
            self.delta = np.full(n, np.nan)

    def curves_from(self, curves) -> Optional["PlaneCurves"]:
        """
        This source's curves in a dict of :func:`curves_for_planes` (keyed by plane index, or by ``curve_key`` when
        the sources come from different tracers).
        """
        if not curves:
            return None
        return curves.get(self.curve_key if self.curve_key is not None else self.plane_index)

    @property
    def has_model(self) -> bool:
        return len(self.model) > 0

    @property
    def unmatched_labels(self) -> List[str]:
        return letters_from(int(np.sum(self.unmatched)))

    @property
    def residual_sigmas(self) -> np.ndarray:
        """
        The residual of every image in units of its noise, the fit's OWN per-image chi: ``|observed - paired model|
        / σ`` for an image-plane fit (the nearest model image for the soft all-pairs mixture) and ``sqrt(χ²ᵢ)`` of
        the fit's own source-plane chi-squared map for a source-plane fit.
        """
        if self.image_plane_chi is not None:
            return self.image_plane_chi
        return self.delta / self.sigma


def _sigma_from(noise_map, total: int) -> np.ndarray:
    noise = np.asarray(
        noise_map.array if hasattr(noise_map, "array") else noise_map, dtype=float
    ).ravel()
    return np.broadcast_to(noise, (total,)).copy()


def source_record_from_dataset(k: int, dataset) -> SourceRecord:
    """
    Returns the record of a ``PointDataset`` with no fit: its observed positions, noise and image numbers.
    """
    observed = pp._np_grid(dataset.positions)
    return SourceRecord(
        k=k,
        name=dataset.name,
        redshift=getattr(dataset, "redshift", None),
        observed=observed,
        sigma=_sigma_from(dataset.positions_noise_map, len(observed)),
        numbers=pp.image_numbers_from(dataset.positions),
    )


def plane_index_of(fit_positions, tracer) -> int:
    """
    Returns the (non-negative) index of the plane the point source of a positions fit is in.
    """
    plane_index = int(getattr(fit_positions, "plane_index", -1))
    if plane_index < 0:
        plane_index = len(tracer.planes) + plane_index
    return plane_index


def source_record_from_fit(k: int, fit) -> SourceRecord:
    """
    Returns the record of a ``FitPointDataset``: the dataset's quantities plus the fit's model positions, its own
    pairing, the unmatched model images, the residual of every image and the magnification of every model image.

    Observed images with no model image of their own are flagged in ``no_model``: those with no model image at all,
    and, for the one-to-one ``FitPositionsImagePair``, those its assignment leaves unassigned when the model has
    fewer images than are observed (the fit pairs them with their nearest model image).
    """
    from autolens.point.fit.positions.image.pair import FitPositionsImagePair
    from autolens.point.fit.positions.image.pair_all import FitPositionsImagePairAll
    from autolens.point.plot import source_plane_panel as spp

    fit_positions = fit.positions
    tracer = fit.tracer
    record = source_record_from_dataset(k=k, dataset=fit.dataset)
    record.fit = fit

    observed = record.observed
    plane_index = plane_index_of(fit_positions, tracer)
    record.plane_index = plane_index
    if record.redshift is None:
        record.redshift = float(tracer.planes[plane_index].redshift)

    record.linestyle, record.residual_label = pp.residual_style_from(fit_positions)
    record.model_label = pp.model_label_from(fit_positions)
    record.mixture = isinstance(fit_positions, FitPositionsImagePairAll)
    try:
        record.chi_squared = float(spp._np(fit_positions.chi_squared))
    except Exception:
        record.chi_squared = None

    pairing = pp.model_pairing_from(fit_positions)

    if pairing is None:
        # A source-plane fit: rank by its own per-image chi-squared.
        try:
            chi_squared_map = np.asarray(
                spp._np(fit_positions.chi_squared_map), dtype=float
            ).ravel()
            record.image_plane_chi = np.sqrt(np.clip(chi_squared_map, 0.0, None))
        except Exception:
            record.image_plane_chi = np.full(len(observed), np.nan)
        return record

    _, model, indexes = pairing
    record.model = model
    record.indexes = indexes
    record.unmatched = (
        pp.unmatched_model_mask_from(model, indexes)
        if len(model)
        else np.zeros(0, dtype=bool)
    )

    no_model = indexes < 0
    if (
        isinstance(fit_positions, FitPositionsImagePair)
        and len(model)
        and len(model) < len(observed)
    ):
        from scipy.optimize import linear_sum_assignment

        distances = np.linalg.norm(observed[:, None] - model[None], axis=2)
        rows, _ = linear_sum_assignment(distances)
        assigned = np.zeros(len(observed), dtype=bool)
        assigned[rows] = True
        no_model = no_model | ~assigned
    record.no_model = no_model

    delta = np.full(len(observed), np.nan)
    paired = indexes >= 0
    delta[paired] = np.linalg.norm(observed[paired] - model[indexes[paired]], axis=1)
    record.delta = delta

    if len(model):
        try:
            record.mu_model = magnifications_at(tracer, model, plane_index)
        except Exception:
            record.mu_model = np.full(len(model), np.nan)

    return record


def assign_styles(records: List[SourceRecord]):
    """
    Assigns every record its colour and marker, returning the ``(cmap, norm)`` of the colour-by-redshift scheme
    used beyond ``MAX_QUALITATIVE`` sources (``None`` otherwise).

    Up to ``MAX_QUALITATIVE`` sources the colour is the fixed-order ``SOURCE_PALETTE`` (source 0 black). Beyond that
    the colour follows the source redshift on viridis (truncated so the lightest yellow stays visible on white):
    redshift is the physically meaningful ordering and is what sets each source's critical curve, so the nested
    curves read as a family. Marker shapes cycle through ``MARKERS`` in both cases.
    """
    import matplotlib as mpl

    colormap = None
    if len(records) <= MAX_QUALITATIVE:
        for record, color in zip(records, SOURCE_PALETTE):
            record.color = color
    else:
        redshifts = np.array(
            [r.redshift if r.redshift is not None else np.nan for r in records],
            dtype=float,
        )
        if not np.all(np.isfinite(redshifts)):
            # No redshifts to colour by: colour by source index instead.
            redshifts = np.arange(len(records), dtype=float)
        vmin, vmax = float(redshifts.min()), float(redshifts.max())
        if vmax <= vmin:
            vmax = vmin + 1.0
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "viridis_trunc", mpl.colormaps["viridis"](np.linspace(0.0, 0.85, 256))
        )
        for record, z in zip(records, redshifts):
            record.color = cmap(norm(z))
        colormap = (cmap, norm)
    for k, record in enumerate(records):
        record.marker = MARKERS[k % len(MARKERS)]
    return colormap


def records_from_datasets(datasets) -> List[SourceRecord]:
    records = [source_record_from_dataset(k, d) for k, d in enumerate(datasets)]
    return records


def records_from_fits(fits) -> List[SourceRecord]:
    return [source_record_from_fit(k, fit) for k, fit in enumerate(fits)]


# ---------------------------------------------------------------------------
# Per-plane critical curves and caustics
# ---------------------------------------------------------------------------


@dataclass
class PlaneCurves:
    """
    The critical curves (image plane) and caustics (source plane) of one source plane, as lists of (N, 2) (y,x)
    arrays.
    """

    plane_index: int
    redshift: float
    tangential: List[np.ndarray]
    radial: List[np.ndarray]
    tangential_caustics: List[np.ndarray]
    radial_caustics: List[np.ndarray]

    def image_plane_lines(self):
        """
        The curves as ``(lines, colors)`` in the format of ``_compute_critical_curve_lines`` (``"white"``
        tangential, ``"yellow"`` radial), for the figures which draw the last plane's curves over an image.
        """
        lines = list(self.tangential) + list(self.radial)
        colors = ["white"] * len(self.tangential) + ["yellow"] * len(self.radial)
        return (lines or None), colors

    def source_plane_lines(self):
        lines = list(self.tangential_caustics) + list(self.radial_caustics)
        colors = ["white"] * len(self.tangential_caustics) + ["yellow"] * len(
            self.radial_caustics
        )
        return (lines or None), colors


def deflections_to_plane_from(tracer, points: np.ndarray, plane_index: int) -> np.ndarray:
    """
    Returns the (N, 2) deflections of the image-plane ``points`` to plane ``plane_index`` of the FULL tracer: the
    deflections the positions fit traces its observed positions with.
    """
    import autoarray as aa

    grid = aa.Grid2DIrregular(values=np.asarray(points, dtype=float).reshape(-1, 2))
    if len(tracer.planes) <= 2:
        deflections = tracer.deflections_yx_2d_from(grid=grid, xp=np)
    else:
        deflections = tracer.deflections_between_planes_from(
            grid=grid, xp=np, plane_i=0, plane_j=plane_index
        )
    return pp._np_grid(deflections)


def magnifications_at(tracer, points, plane_index: int, h: float = 1e-3) -> np.ndarray:
    """
    Returns the signed magnification ``1 / det(A)`` at (y,x) ``points`` for plane ``plane_index``, by central finite
    differences of the full tracer's deflections.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(points) == 0:
        return np.zeros(0)
    offsets = np.array([[h, 0.0], [-h, 0.0], [0.0, h], [0.0, -h]])
    stacked = np.concatenate([points + o for o in offsets])
    a = deflections_to_plane_from(tracer, stacked, plane_index).reshape(4, len(points), 2)
    day_dy = (a[0, :, 0] - a[1, :, 0]) / (2 * h)
    dax_dy = (a[0, :, 1] - a[1, :, 1]) / (2 * h)
    day_dx = (a[2, :, 0] - a[3, :, 0]) / (2 * h)
    dax_dx = (a[2, :, 1] - a[3, :, 1]) / (2 * h)
    det = (1 - day_dy) * (1 - dax_dx) - day_dx * dax_dy
    with np.errstate(divide="ignore"):
        return 1.0 / det


def _contour_lines(field_, y_min, x_min, dy, dx, min_points=6):
    from skimage.measure import find_contours

    lines = []
    for contour in find_contours(field_, 0.0):
        if len(contour) < min_points:
            continue
        ys = y_min + contour[:, 0] * dy
        xs = x_min + contour[:, 1] * dx
        # A curve spanning fewer than 8 grid pixels is a marching-squares artefact around a singular mass centre,
        # not a curve the grid resolves.
        if max(np.ptp(ys), np.ptp(xs)) < 8.0 * max(dy, dx):
            continue
        lines.append(np.stack([ys, xs], axis=1))
    return lines


def curves_for_planes(
    tracer,
    plane_indexes,
    extent: List[float],
    shape: int = CURVE_GRID_SHAPE,
) -> Dict[int, PlaneCurves]:
    """
    Returns the critical curves and caustics of every plane in ``plane_indexes``, keyed by plane index.

    The deflection field from the image plane to plane ``j`` of the FULL tracer is evaluated once on a
    ``shape x shape`` grid covering ``extent`` (``[x_min, x_max, y_min, y_max]``), differentiated with
    ``np.gradient`` into the convergence and shear, and contoured with marching squares where
    ``λt = 1 - κ - γ = 0`` (tangential critical curves) and ``λr = 1 - κ + γ = 0`` (radial). Each critical curve is
    ray-traced to plane ``j`` for its caustic.

    When every mass-bearing galaxy sits in one lens plane, the deflection to any later plane is the last plane's
    times a distance ratio. The ratio is measured on probe points and, if the rescaling holds there, the last
    plane's field is reused; otherwise the field is evaluated for that plane.

    Returns an empty dict when ``PYAUTO_FAST_PLOTS=1``.
    """
    if os.environ.get("PYAUTO_FAST_PLOTS") == "1":
        return {}

    last = len(tracer.planes) - 1
    indexes = sorted(
        {last if (j is None or j < 0) else int(j) for j in plane_indexes}
    )

    x_min, x_max, y_min, y_max = extent
    xs = np.linspace(x_min, x_max, shape)
    ys = np.linspace(y_min, y_max, shape)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.stack([grid_y.ravel(), grid_x.ravel()], axis=1)
    dy, dx = ys[1] - ys[0], xs[1] - xs[0]

    rng = np.random.default_rng(0)
    probe = np.stack(
        [rng.uniform(y_min, y_max, 64), rng.uniform(x_min, x_max, 64)], axis=1
    )
    probe_last = deflections_to_plane_from(tracer, probe, last)
    field_last = None

    curves = {}
    for j in indexes:
        if j == last:
            if field_last is None:
                field_last = deflections_to_plane_from(tracer, points, last)
            deflections = field_last
        else:
            probe_j = deflections_to_plane_from(tracer, probe, j)
            usable = np.abs(probe_last) > 1e-6
            ratio = float(np.median(probe_j[usable] / probe_last[usable])) if np.any(usable) else np.nan
            if np.isfinite(ratio) and np.allclose(
                probe_j, ratio * probe_last, rtol=1e-6, atol=1e-9
            ):
                if field_last is None:
                    field_last = deflections_to_plane_from(tracer, points, last)
                deflections = ratio * field_last
            else:
                deflections = deflections_to_plane_from(tracer, points, j)

        alpha_y = deflections[:, 0].reshape(shape, shape)
        alpha_x = deflections[:, 1].reshape(shape, shape)
        day_dy, day_dx = np.gradient(alpha_y, dy, dx)
        dax_dy, dax_dx = np.gradient(alpha_x, dy, dx)
        kappa = 0.5 * (day_dy + dax_dx)
        gamma = np.hypot(0.5 * (dax_dx - day_dy), 0.5 * (day_dx + dax_dy))

        tangential = _contour_lines(1.0 - kappa - gamma, y_min, x_min, dy, dx)
        radial = _contour_lines(1.0 - kappa + gamma, y_min, x_min, dy, dx)

        def _caustics(lines):
            return [line - deflections_to_plane_from(tracer, line, j) for line in lines]

        curves[j] = PlaneCurves(
            plane_index=j,
            redshift=float(tracer.planes[j].redshift),
            tangential=tangential,
            radial=radial,
            tangential_caustics=_caustics(tangential),
            radial_caustics=_caustics(radial),
        )
    return curves


def curves_for_records(tracer, records: List[SourceRecord], extent, shape=CURVE_GRID_SHAPE):
    """
    Returns :func:`curves_for_planes` for the planes of every record (and the last plane), or an empty dict if they
    cannot be computed (a visualization must never stop a model-fit).
    """
    import logging

    plane_indexes = [r.plane_index for r in records if r.plane_index is not None]
    plane_indexes.append(len(tracer.planes) - 1)
    try:
        return curves_for_planes(tracer, plane_indexes, extent, shape=shape)
    except Exception:
        logging.getLogger(__name__).warning(
            "Point-source critical curves could not be computed; figures are drawn without them.",
            exc_info=True,
        )
        return {}


def curves_for_fits(fits, records: List[SourceRecord], extent, shape=CURVE_GRID_SHAPE):
    """
    The curves of every source of several fits. Fits of one lens model (the usual cluster factor graph, where every
    factor fits the same model) share their tracer's planes, so the curves are computed once from the first fit's
    FULL tracer. If the fits' tracers have different planes, each source's curves are computed from its own fit and
    keyed by its index (``record.curve_key``).
    """
    tracers = [fit.tracer for fit in fits]
    redshifts = [tuple(float(p.redshift) for p in t.planes) for t in tracers]
    if all(r == redshifts[0] for r in redshifts):
        return curves_for_records(tracers[0], records, extent, shape=shape)
    curves = {}
    for record, tracer in zip(records, tracers):
        own = curves_for_records(tracer, [record], extent, shape=shape)
        record.curve_key = ("source", record.k)
        if record.plane_index in own:
            curves[record.curve_key] = own[record.plane_index]
    return curves


def union_extent_from(records: List[SourceRecord], solver=None) -> List[float]:
    """
    Returns the ``[x_min, x_max, y_min, y_max]`` extent of a positions figure of several sources: the solver grid's
    extent, enlarged to contain every observed position (:func:`positions_panel.positions_extent_from` of all the
    sources' positions together).
    """
    observed = np.vstack([r.observed for r in records])
    return pp.positions_extent_from(positions=observed, solver=solver)


def ranking_range_from(records: List[SourceRecord]) -> float:
    """
    The linear Δ/σ range of the residual ranking shared by every source: 1.2 times the 90th-percentile residual of
    all images (a robust maximum, so one outlier does not squash every other image), at least 4σ. Larger values are
    drawn off-scale at the edge with their value printed.
    """
    values = np.concatenate([r.residual_sigmas for r in records]) if records else []
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    p90 = float(np.percentile(values, 90)) if len(values) else 3.0
    return max(4.0, 1.2 * p90)


def ranking_rows_from(record: SourceRecord):
    """
    The ranking entries of a source, ``(value, label, kind, index)`` worst first, with unmatched model images (no
    residual) last. ``kind`` is ``"obs"``, ``"nomodel"`` (an observed image with no model image of its own) or
    ``"unmatched"``.
    """
    values = record.residual_sigmas
    rows = []
    for i in range(len(record.observed)):
        kind = "nomodel" if record.no_model[i] else "obs"
        rows.append((float(values[i]), f"{record.numbers[i]}", kind, i))
    rows.sort(key=lambda t: -(t[0] if np.isfinite(t[0]) else -1.0))
    for m, letter in zip(np.where(record.unmatched)[0], record.unmatched_labels):
        rows.append((np.nan, letter, "unmatched", int(m)))
    return rows


@dataclass
class RankingScales:
    """
    The scales shared by every row of ``fit_per_image`` (and every per-source file of a many-source fit), computed
    where every source is visible: the ranking range and the number of ranking slots.
    """

    xmax: float
    n_slots: int


def ranking_scales_from(records: List[SourceRecord]) -> RankingScales:
    return RankingScales(
        xmax=ranking_range_from(records),
        n_slots=max([len(ranking_rows_from(r)) for r in records] + [1]),
    )
