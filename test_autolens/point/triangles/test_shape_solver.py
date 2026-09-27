"""
The `ShapeSolver` validation suite.

`ShapeSolver` shipped in ~2024 with one method (`find_magnification`), its only test
commented out, and no consumer other than an unused workspace demo; the audit of
2026-08-19 called it effectively unmaintained. This module is the validation it never
had, written as part of the `image-source-mappings` epic, which reuses the solver as the
source -> image engine for non-pixelized sources.

Every case here is a statement about the *physics* the solver claims to compute, checked
against something that shares no code with it: the analytic magnification of an
Isothermal sphere, a grid-tracing oracle, the behaviour of shapes of equal area, and the
NumPy path itself. The bugs the suite found are named in the tests that pin them.
"""

import numpy as np
import pytest
from scipy import ndimage

import autoarray as aa
import autolens as al

from autoarray.structures.triangles.array import MAX_CONTAINING_SIZE
from autoarray.structures.triangles.shape import Circle, Polygon, Square, Triangle
from autolens.point.solver.shape_solver import ShapeSolver

EINSTEIN_RADIUS = 1.0


@pytest.fixture
def image_mask():
    return al.Mask2D.circular(shape_native=(100, 100), pixel_scales=0.05, radius=2.4)


@pytest.fixture
def image_grid(image_mask):
    return al.Grid2D.from_mask(mask=image_mask)


@pytest.fixture
def sis_tracer():
    """
    A singular isothermal sphere, whose magnification is analytic.
    """
    return al.Tracer(
        galaxies=[
            al.Galaxy(
                redshift=0.5,
                mass=al.mp.IsothermalSph(
                    centre=(0.0, 0.0), einstein_radius=EINSTEIN_RADIUS
                ),
            ),
            al.Galaxy(redshift=1.0),
        ]
    )


@pytest.fixture
def shape_solver(image_grid):
    return ShapeSolver.for_grid(grid=image_grid, pixel_scale_precision=0.005)


def sis_total_magnification(beta: float) -> float:
    """
    The analytic total magnification of a point source at impact parameter `beta` inside
    the Einstein radius of a singular isothermal sphere.

    The two images sit at ``theta = beta +/- theta_E`` with magnifications
    ``mu_+ = 1 + theta_E / beta`` and ``mu_- = theta_E / beta - 1``, so the sum of their
    absolute magnifications is ``2 * theta_E / beta``.
    """
    return 2.0 * EINSTEIN_RADIUS / beta


# ---------------------------------------------------------------------------------- #
# (a) the Einstein ring, and the analytic magnification of an off-axis source
# ---------------------------------------------------------------------------------- #


def test_on_axis_source_traces_to_an_einstein_ring(shape_solver, sis_tracer):
    """
    A small circle at the centre of the source plane of an isothermal *sphere* maps to the
    Einstein ring: one connected image whose vertices all sit at the Einstein radius.

    A ring is a single connected component, not two: it is the degenerate case where the
    two images of an off-axis source have merged, so the component count is 1 and it is
    the *radius* which carries the physics.
    """
    # `Circle` takes element 0 first, which for every PyAuto `(y, x)` grid is `y`, so a
    # circle at the light profile centre `(y_c, x_c)` is `Circle(y_c, x_c, radius=...)`.
    shape = Circle(0.0, 0.0, radius=0.02)

    triangles = shape_solver.solve_triangles(tracer=sis_tracer, shape=shape)

    vertices = np.asarray(triangles.triangles).reshape(-1, 2)

    radii = np.hypot(vertices[:, 0], vertices[:, 1])

    # The lens centre is itself a (formally infinitely demagnified) image of an on-axis
    # source, so it is excluded before the ring is measured -- it is what
    # `magnification_threshold` removes from `image_regions_from`.
    ring = radii[radii > 0.5 * EINSTEIN_RADIUS]

    assert ring.size > 0.9 * radii.size
    assert np.mean(ring) == pytest.approx(EINSTEIN_RADIUS, abs=0.01)
    assert np.max(ring) - np.min(ring) < 0.1 * EINSTEIN_RADIUS


def test_off_axis_source_has_two_images(shape_solver, sis_tracer):
    """
    A source inside the Einstein radius of an isothermal sphere has exactly two images.
    """
    beta = 0.3

    regions = shape_solver.image_regions_from(
        tracer=sis_tracer, shape=Circle(beta, 0.0, radius=0.05)
    )

    assert len(regions) == 2

    # theta = beta + theta_E on one side and beta - theta_E on the other.
    y_coordinates = sorted(region.centre[0] for region in regions)

    assert y_coordinates[0] == pytest.approx(beta - EINSTEIN_RADIUS, abs=0.05)
    assert y_coordinates[1] == pytest.approx(beta + EINSTEIN_RADIUS, abs=0.05)


def test_per_image_magnification_matches_the_analytic_isothermal_sphere(
    shape_solver, sis_tracer
):
    """
    Each image's magnification, measured as its share of the kept triangle area divided by
    the source area, matches the analytic ``1 + theta_E / beta`` and ``theta_E / beta - 1``.
    """
    beta = 0.3

    magnifications = shape_solver.find_magnification(
        tracer=sis_tracer, shape=Circle(beta, 0.0, radius=0.05), per_image=True
    )

    assert len(magnifications) == 2

    assert magnifications[0] == pytest.approx(1.0 + EINSTEIN_RADIUS / beta, rel=0.01)
    assert magnifications[1] == pytest.approx(EINSTEIN_RADIUS / beta - 1.0, rel=0.01)


def test_total_magnification_converges_as_the_source_shrinks(shape_solver, sis_tracer):
    """
    `find_magnification` measures a *finite* source, which equals the point magnification
    only in the limit of a small source. The error must therefore shrink as the radius
    does -- if it did not, the number would be measuring the tiling rather than the lens.
    """
    beta = 0.3

    analytic = sis_total_magnification(beta=beta)

    errors = [
        abs(
            shape_solver.find_magnification(
                tracer=sis_tracer, shape=Circle(beta, 0.0, radius=radius)
            )
            - analytic
        )
        / analytic
        for radius in (0.2, 0.1, 0.05)
    ]

    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 0.01


def test_magnification_threshold_removes_the_central_image(image_grid, sis_tracer):
    """
    `magnification_threshold` was accepted by the constructor and never used by
    `ShapeSolver`: the filter lived only in `PointSolver.solve`. Without it the singular
    centre of an isothermal sphere contributes a spurious two-triangle "image" at the lens
    centre, which `image_regions_from` would report alongside the two real ones.
    """
    shape = Circle(0.3, 0.0, radius=0.05)

    unfiltered = ShapeSolver.for_grid(
        grid=image_grid, pixel_scale_precision=0.005, magnification_threshold=0.0
    )
    filtered = ShapeSolver.for_grid(
        grid=image_grid, pixel_scale_precision=0.005, magnification_threshold=0.1
    )

    assert (
        len(
            unfiltered.find_magnification(
                tracer=sis_tracer, shape=shape, per_image=True
            )
        )
        == 3
    )
    assert (
        len(filtered.find_magnification(tracer=sis_tracer, shape=shape, per_image=True))
        == 2
    )


# ---------------------------------------------------------------------------------- #
# (b) the regions against a grid-tracing oracle
# ---------------------------------------------------------------------------------- #


def _oracle_regions(tracer, grid, mask, shape):
    """
    The image regions found by tracing the image grid and asking the shape which traced
    pixel centres it contains. This shares no code with the triangle solver, so it is an
    independent check of where the images are -- but it works only at pixel resolution,
    which is why it is a test oracle and not the API.
    """
    traced = tracer.traced_grid_2d_list_from(grid=grid)[-1]

    return aa.inversion.mappings.mapping.image_regions_from_slim_mask(
        mask=mask,
        slim_bool=shape.contains(np.asarray(traced.array)),
        min_pixels=1,
    )


def _native_from_slim(mask, slim_indexes):
    native = np.zeros(shape=mask.shape_native, dtype=bool)

    pixels = np.argwhere(~np.asarray(mask))

    native[pixels[slim_indexes, 0], pixels[slim_indexes, 1]] = True

    return native


def test_regions_agree_with_the_grid_tracing_oracle(
    shape_solver, sis_tracer, image_grid, image_mask
):
    """
    For a source spanning many pixels the solver's regions must contain every pixel the
    oracle finds, and must not extend more than one pixel beyond them.

    The solver keeps a triangle when its traced image *meets* the shape, so it marks every
    pixel containing any point which traces inside, whereas the oracle marks a pixel only
    when its centre does. The solver's region is therefore the oracle's, dilated by up to
    a pixel -- and never less than it.
    """
    shape = Circle(0.3, 0.0, radius=0.1)

    regions = shape_solver.image_regions_from(tracer=sis_tracer, shape=shape)
    oracle = _oracle_regions(
        tracer=sis_tracer, grid=image_grid, mask=image_mask, shape=shape
    )

    assert len(regions) == len(oracle) == 2

    solver_slim = np.unique(np.concatenate([region.slim_indexes for region in regions]))
    oracle_slim = np.unique(np.concatenate([region.slim_indexes for region in oracle]))

    # A source spanning >= 5 pixels, as the agreement statement requires.
    assert oracle_slim.size >= 5

    assert set(oracle_slim.tolist()) <= set(solver_slim.tolist())

    dilated = ndimage.binary_dilation(
        _native_from_slim(mask=image_mask, slim_indexes=oracle_slim),
        structure=np.ones(shape=(3, 3), dtype=bool),
    )

    assert not (
        _native_from_slim(mask=image_mask, slim_indexes=solver_slim) & ~dilated
    ).any()


def test_regions_are_image_region_objects_the_phase_1_api_understands(
    shape_solver, sis_tracer, image_mask
):
    """
    Engine B must return the same object engine A does, so that everything built on
    `ImageRegion` works whichever engine produced the region.
    """
    regions = shape_solver.image_regions_from(
        tracer=sis_tracer, shape=Circle(0.3, 0.0, radius=0.1)
    )

    region = regions[0]

    assert isinstance(region, aa.ImageRegion)

    array = al.Array2D.ones(shape_native=(100, 100), pixel_scales=0.05)

    assert region.flux_from(array=array) == pytest.approx(
        float(len(region.slim_indexes))
    )
    assert region.area() == pytest.approx(len(region.slim_indexes) * 0.05 * 0.05)
    assert len(region.contours) >= 1
    assert region.contours[0].shape[1] == 2


def test_mapping_from_carries_the_shape_boundary_as_its_source_contour(
    shape_solver, sis_tracer
):
    mapping = shape_solver.mapping_from(
        tracer=sis_tracer, shape=Circle(0.3, 0.0, radius=0.1)
    )

    assert isinstance(mapping, aa.Mapping)
    assert len(mapping.image_regions) == 2
    assert mapping.source_centre == pytest.approx((0.3, 0.0))
    assert len(mapping.source_contours) == 1

    contour = mapping.source_contours[0]

    assert contour[0] == pytest.approx(contour[-1])
    assert np.hypot(contour[:, 0] - 0.3, contour[:, 1] - 0.0) == pytest.approx(
        0.1, abs=1.0e-8
    )


def test_image_regions_from_without_a_grid_names_the_missing_input():
    """
    A solver built from limits has an extent but no pixel grid, so it cannot say which
    pixels its regions cover. The error must name the input the caller controls.
    """
    solver = ShapeSolver.for_limits_and_scale(
        y_min=-1.0,
        y_max=1.0,
        x_min=-1.0,
        x_max=1.0,
        scale=0.1,
        pixel_scale_precision=0.01,
    )

    assert solver.grid is None

    with pytest.raises(ValueError) as exc_info:
        solver.image_regions_from(
            tracer=al.Tracer(
                galaxies=[al.Galaxy(redshift=0.5), al.Galaxy(redshift=1.0)]
            ),
            shape=Circle(0.0, 0.0, radius=0.1),
        )

    assert "grid=" in str(exc_info.value)


# ---------------------------------------------------------------------------------- #
# (c) the refinement steps
# ---------------------------------------------------------------------------------- #


def test_refinement_shrinks_the_search_and_converges(shape_solver, sis_tracer):
    """
    The kept area is **not** monotone over the steps, and asserting that it is would pin a
    property the algorithm does not have: step 0 keeps whole coarse triangles, and each
    later step filters the *neighbourhood* of the previous kept set, which is larger than
    that set. Measured over seven steps (``pixel_scale_precision=0.0005``) the kept areas
    run 0.2143413, 0.2154238, 0.2126498, 0.2122439, 0.2123877, 0.2124807, 0.2124794 --
    they oscillate at the fourth decimal while converging.

    Two things are true instead, and are what is pinned here. The search envelope
    (`neighbourhood`, the set the next step filters) shrinks at every step, since the rim
    it adds is one triangle wide and the triangles halve. And the kept area converges: the
    final refinement changes it by less than any earlier one (0.0000013 against a first
    change of 0.0010825 over those seven steps).
    """
    steps = list(
        shape_solver.steps(tracer=sis_tracer, shape=Circle(0.3, 0.0, radius=0.1))
    )

    assert len(steps) == shape_solver.n_steps

    envelopes = [step.neighbourhood.area for step in steps]

    assert envelopes == sorted(envelopes, reverse=True)

    kept = [step.filtered_triangles.area for step in steps]

    changes = [abs(kept[i] - kept[i - 1]) for i in range(1, len(kept))]

    # The final refinement moves the kept area less than any earlier one.
    assert changes[-1] == min(changes)

    # And no step is ever far from the converged answer -- the oscillation is small.
    assert np.max(np.abs(np.asarray(kept) - kept[-1])) < 0.02 * kept[-1]

    # And the converged answer is the one `solve_triangles` returns.
    assert shape_solver.solve_triangles(
        tracer=sis_tracer, shape=Circle(0.3, 0.0, radius=0.1)
    ).area == pytest.approx(kept[-1])


def test_precision_coarser_than_the_triangle_scale_is_rejected(image_grid):
    """
    `n_steps <= 0` used to make `steps` empty and surface as an `IndexError`; it now names
    `pixel_scale_precision`. Pinned here because the validation suite is where the
    solver's input handling is stated.
    """
    with pytest.raises(ValueError) as exc_info:
        ShapeSolver.for_grid(
            grid=image_grid, pixel_scale_precision=1.0
        ).solve_triangles(
            tracer=al.Tracer(
                galaxies=[al.Galaxy(redshift=0.5), al.Galaxy(redshift=1.0)]
            ),
            shape=Circle(0.0, 0.0, radius=0.1),
        )

    assert "pixel_scale_precision" in str(exc_info.value)


# ---------------------------------------------------------------------------------- #
# (d) shapes other than a circle
# ---------------------------------------------------------------------------------- #


def _equal_area_shapes(centre: float, area: float):
    """
    A square, an equilateral triangle and a regular hexagon of the given area, all centred
    on ``(centre, 0.0)`` -- element 0 first, as `Shape` requires.
    """
    side = np.sqrt(area)

    square = Square(
        top=centre - side / 2.0,
        bottom=centre + side / 2.0,
        left=-side / 2.0,
        right=side / 2.0,
    )

    length = np.sqrt(4.0 * area / np.sqrt(3.0))
    height = length * np.sqrt(3.0) / 2.0

    triangle = Triangle(
        (centre + 2.0 * height / 3.0, 0.0),
        (centre - height / 3.0, -length / 2.0),
        (centre - height / 3.0, length / 2.0),
    )

    radius = np.sqrt(2.0 * area / (3.0 * np.sqrt(3.0)))
    angles = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)

    hexagon = Polygon(
        [(centre + radius * np.cos(angle), radius * np.sin(angle)) for angle in angles]
    )

    return {"square": square, "triangle": triangle, "hexagon": hexagon}


def test_shapes_of_equal_area_give_the_same_images(shape_solver, sis_tracer):
    """
    Magnification is an area ratio, so it must depend on the source's area and position
    and not on which `Shape` subclass expresses them. Every shape's `mask` reads the
    triangle vertices in the same element order, so a disagreement here would be a
    coordinate-order bug in one of them (`Triangle.triangle_contains_mask` had exactly
    that bug before phase 2a).
    """
    beta = 0.3

    circle = Circle(beta, 0.0, radius=0.1)

    reference = shape_solver.find_magnification(
        tracer=sis_tracer, shape=circle, per_image=True
    )

    assert len(reference) == 2

    for name, shape in _equal_area_shapes(centre=beta, area=circle.area).items():
        assert shape.area == pytest.approx(circle.area, rel=1.0e-6), name

        magnifications = shape_solver.find_magnification(
            tracer=sis_tracer, shape=shape, per_image=True
        )

        assert len(magnifications) == 2, name
        assert magnifications == pytest.approx(reference, rel=0.02), name

        assert (
            len(shape_solver.image_regions_from(tracer=sis_tracer, shape=shape)) == 2
        ), name


# ---------------------------------------------------------------------------------- #
# (e) a source on an intermediate plane
# ---------------------------------------------------------------------------------- #


@pytest.fixture
def multi_plane_tracer():
    return al.Tracer(
        galaxies=[
            al.Galaxy(
                redshift=0.5,
                mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.2),
            ),
            al.Galaxy(
                redshift=1.0,
                mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=0.2),
            ),
            al.Galaxy(redshift=2.0),
        ]
    )


def test_intermediate_plane_regions_back_trace_into_the_shape(
    shape_solver, multi_plane_tracer
):
    """
    A source on the *middle* plane of a three-plane tracer. Solving to the last plane by
    mistake gives regions which are not images of this source at all, so the check is not
    that regions exist but that they trace back into the shape at the plane asked for.
    """
    shape = Circle(0.2, 0.0, radius=0.1)

    regions = shape_solver.image_regions_from(
        tracer=multi_plane_tracer, shape=shape, plane_redshift=1.0
    )

    assert len(regions) == 2

    plane_index = multi_plane_tracer.plane_index_via_redshift_from(redshift=1.0)

    traced = multi_plane_tracer.traced_grid_2d_list_from(
        grid=al.Grid2DIrregular([region.centre for region in regions])
    )[plane_index]

    assert shape.contains(np.asarray(traced.array)).all()


def test_intermediate_plane_differs_from_the_last_plane(
    shape_solver, multi_plane_tracer
):
    """
    `plane_redshift` must actually change the answer, otherwise the test above would pass
    for a solver which ignored it.
    """
    shape = Circle(0.2, 0.0, radius=0.1)

    at_middle = shape_solver.image_regions_from(
        tracer=multi_plane_tracer, shape=shape, plane_redshift=1.0
    )
    at_last = shape_solver.image_regions_from(tracer=multi_plane_tracer, shape=shape)

    assert len(at_middle) != len(at_last)


# ---------------------------------------------------------------------------------- #
# (f) the JAX path
# ---------------------------------------------------------------------------------- #


def test_use_jax_solver_is_rejected_rather_than_silently_wrong(image_grid, sis_tracer):
    """
    `use_jax=True` used to be silently ignored by `find_magnification`, which hardcoded
    `xp=np`. Consulting `self._xp` instead would have made it silently *wrong*: see
    `test_jax_and_numpy_kept_triangles_agree`. It therefore raises, naming the cap.

    **No `importorskip` here, deliberately.** The rejection must fire on an install with
    no JAX at all -- that is the install most likely to hit it -- so this test runs on the
    `unittest-nojax` CI leg, where it caught the solver resolving `self._xp` (which
    imports `jax.numpy`) before the check, so the user got `ModuleNotFoundError` instead
    of the explanation.
    """
    solver = ShapeSolver.for_grid(
        grid=image_grid, pixel_scale_precision=0.02, use_jax=True
    )

    for call in (
        lambda: solver.find_magnification(
            tracer=sis_tracer, shape=Circle(0.3, 0.0, radius=0.1)
        ),
        lambda: solver.solve_triangles(
            tracer=sis_tracer, shape=Circle(0.3, 0.0, radius=0.1)
        ),
        lambda: list(
            solver.steps(tracer=sis_tracer, shape=Circle(0.3, 0.0, radius=0.1))
        ),
    ):
        with pytest.raises(NotImplementedError) as exc_info:
            call()

        assert f"MAX_CONTAINING_SIZE ({MAX_CONTAINING_SIZE})" in str(exc_info.value)


def test_explicit_jax_module_is_rejected_rather_than_silently_wrong(
    image_grid, sis_tracer
):
    """
    The other route to the JAX path: a NumPy-configured solver handed ``xp=jax.numpy``
    explicitly. It must give the same error as `use_jax=True`, so neither route can reach
    the truncated containers.

    This one needs JAX to have a module to pass, hence the guard -- unlike the test above,
    which is exactly the case that must work *without* it.
    """
    jnp = pytest.importorskip("jax.numpy")

    solver = ShapeSolver.for_grid(grid=image_grid, pixel_scale_precision=0.02)

    with pytest.raises(NotImplementedError) as exc_info:
        solver.find_magnification(
            tracer=sis_tracer, shape=Circle(0.3, 0.0, radius=0.1), xp=jnp
        )

    assert f"MAX_CONTAINING_SIZE ({MAX_CONTAINING_SIZE})" in str(exc_info.value)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFERRED: the JAX triangle containers truncate every refinement step to "
        f"MAX_CONTAINING_SIZE ({MAX_CONTAINING_SIZE}) triangles to keep static shapes. That "
        "is enough for a Point but not for a Shape with area, so the JAX path keeps "
        f"{MAX_CONTAINING_SIZE} of the "
        "~800 triangles the NumPy path keeps and measures a magnification orders of "
        "magnitude too small (0.13 at the original cap of 15, where the truth is 6.86). Lifting the cap is a redesign of the JAX containers (the cap "
        "is what makes their shapes static, and an extended source has no static bound), "
        "not a fix in ShapeSolver, so it is deferred and ShapeSolver raises on use_jax "
        "instead. Remove this xfail when the containers grow a dynamic kept set."
    ),
)
def test_jax_and_numpy_kept_triangles_agree(image_grid, sis_tracer):
    jnp = pytest.importorskip("jax.numpy")

    shape = Circle(0.3, 0.0, radius=0.1)

    numpy_triangles = ShapeSolver.for_grid(
        grid=image_grid, pixel_scale_precision=0.02
    ).solve_triangles(tracer=sis_tracer, shape=shape, xp=np)

    jax_solver = ShapeSolver.for_grid(
        grid=image_grid, pixel_scale_precision=0.02, use_jax=True
    )

    # `_xp_from` is what refuses JAX; the parity this test asserts is of the underlying
    # algorithm, so it is bypassed deliberately here rather than the guard being weakened.
    jax_triangles = super(ShapeSolver, jax_solver).solve_triangles(
        tracer=sis_tracer, shape=shape, xp=jnp
    )

    def vertex_set(triangles):
        vertices = np.asarray(triangles.triangles).reshape(-1, 2)
        vertices = vertices[~np.isnan(vertices).any(axis=1)]
        return set(map(tuple, np.unique(np.round(vertices, 8), axis=0)))

    assert vertex_set(numpy_triangles) == vertex_set(jax_triangles)
