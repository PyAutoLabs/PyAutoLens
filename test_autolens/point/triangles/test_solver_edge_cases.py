"""
Regression tests for @rhayes777's audit finding in PyAutoLens#531.

Two realistic inputs crashed ``PointSolver.solve`` with errors that named nothing the caller
controls:

- a source-plane coordinate outside the region the image-plane grid tiles
  -> ``numpy.exceptions.AxisError: axis 1 is out of bounds for array of dimension 1``
- a ``pixel_scale_precision`` coarser than the initial triangle scale
  -> ``IndexError: list index out of range``

Both are inputs a user reaches on purpose: configurations outside the caustic are routine during
model exploration, and loosening the precision for a quick first pass is an obvious thing to
script.
"""

import numpy as np
import pytest

import autolens as al


@pytest.fixture
def solver_grid():
    return al.Grid2D.uniform(shape_native=(80, 80), pixel_scales=0.05)


@pytest.fixture
def lens_galaxy():
    return al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0), ell_comps=(0.1, 0.0), einstein_radius=1.0
        ),
    )


def _tracer(lens_galaxy, source_plane_coordinate):
    source = al.Galaxy(
        redshift=1.0, point_0=al.ps.Point(centre=source_plane_coordinate)
    )
    return al.Tracer(galaxies=[lens_galaxy, source])


def test__source_outside_tiled_region__returns_empty_grid(solver_grid, lens_galaxy):
    """
    No image is a legitimate answer, so the solver returns a correctly-shaped empty grid rather
    than raising `AxisError` from the `axis=1` reductions.
    """

    source_plane_coordinate = (5.0, 5.0)

    solver = al.PointSolver.for_grid(grid=solver_grid, pixel_scale_precision=0.001)

    result = solver.solve(
        tracer=_tracer(lens_galaxy, source_plane_coordinate),
        source_plane_coordinate=source_plane_coordinate,
    )

    assert len(result) == 0
    assert np.asarray(result.array).shape == (0, 2)


def test__source_outside_tiled_region__warns(solver_grid, lens_galaxy, caplog):
    """The empty result is explained, so it does not read as a silent no-op."""

    source_plane_coordinate = (5.0, 5.0)

    solver = al.PointSolver.for_grid(grid=solver_grid, pixel_scale_precision=0.001)

    with caplog.at_level("WARNING"):
        solver.solve(
            tracer=_tracer(lens_galaxy, source_plane_coordinate),
            source_plane_coordinate=source_plane_coordinate,
        )

    assert "found no images" in caplog.text


def test__all_candidates_rejected_by_magnification__returns_empty_and_warns(
    solver_grid, lens_galaxy, caplog
):
    """
    The second route to an empty answer, which does NOT go through the length-0 branch.

    `_filter_low_magnification` preserves the array length and writes NaN rows, which become
    `inf` and are stripped by `remove_infinities`. An earlier revision warned only on the
    length-0 branch, so this route returned an empty grid silently.
    """

    source_plane_coordinate = (0.05, 0.02)

    solver = al.PointSolver.for_grid(
        grid=solver_grid,
        pixel_scale_precision=0.001,
        magnification_threshold=1.0e100,
    )

    with caplog.at_level("WARNING"):
        result = solver.solve(
            tracer=_tracer(lens_galaxy, source_plane_coordinate),
            source_plane_coordinate=source_plane_coordinate,
        )

    assert len(result) == 0
    assert "found no images" in caplog.text
    assert "magnification_threshold" in caplog.text


def test__successful_solve__does_not_warn(solver_grid, lens_galaxy, caplog):
    """The warning must not fire on the ordinary path."""

    source_plane_coordinate = (0.05, 0.02)

    solver = al.PointSolver.for_grid(grid=solver_grid, pixel_scale_precision=0.001)

    with caplog.at_level("WARNING"):
        result = solver.solve(
            tracer=_tracer(lens_galaxy, source_plane_coordinate),
            source_plane_coordinate=source_plane_coordinate,
        )

    assert len(result) == 4
    assert "found no images" not in caplog.text


@pytest.mark.parametrize("pixel_scale_precision", [0.05, 0.1, 0.2, 0.5])
def test__precision_coarser_than_triangle_scale__raises_naming_the_parameter(
    solver_grid, lens_galaxy, pixel_scale_precision
):
    """
    `n_steps` is `ceil(log2(scale / pixel_scale_precision))`, which is <= 0 once the precision
    reaches the triangle scale. A negative value used to slip past an `== 0` check and produce
    `IndexError: list index out of range`.

    The grid's `pixel_scales` is 0.05, so every value here is at or beyond the boundary.
    """

    source_plane_coordinate = (0.05, 0.02)

    solver = al.PointSolver.for_grid(
        grid=solver_grid, pixel_scale_precision=pixel_scale_precision
    )

    with pytest.raises(ValueError) as error:
        solver.solve(
            tracer=_tracer(lens_galaxy, source_plane_coordinate),
            source_plane_coordinate=source_plane_coordinate,
        )

    # the message must name the parameter the caller controls
    assert "pixel_scale_precision" in str(error.value)


def test__precision_fine_enough__still_solves(solver_grid, lens_galaxy):
    """
    The control from the report: the same source at a workable precision returns its four images.
    Guards the fix against over-reach.
    """

    source_plane_coordinate = (0.05, 0.02)

    solver = al.PointSolver.for_grid(grid=solver_grid, pixel_scale_precision=0.001)

    result = solver.solve(
        tracer=_tracer(lens_galaxy, source_plane_coordinate),
        source_plane_coordinate=source_plane_coordinate,
    )

    assert len(result) == 4


@pytest.mark.parametrize("use_jax", [False, True])
@pytest.mark.parametrize("remove_infinities", [None, False, True])
def test__numpy_override_controls_default_padding(
    solver_grid, lens_galaxy, use_jax, remove_infinities
):
    """An explicit NumPy call strips rejected rows unless padding is requested.

    use_jax=True only sets the constructor preference: this test executes NumPy
    throughout. The opposite override and JIT are covered in the test workspace.
    """
    source = (0.05, 0.02)
    solver = al.PointSolver.for_grid(
        grid=solver_grid,
        pixel_scale_precision=0.001,
        magnification_threshold=1.0e100,
        use_jax=use_jax,
    )
    result = np.asarray(
        solver.solve(
            tracer=_tracer(lens_galaxy, source),
            source_plane_coordinate=source,
            xp=np,
            remove_infinities=remove_infinities,
        ).array
    )
    if remove_infinities is False:
        assert len(result) > 0
        assert np.isinf(result).all()
    else:
        assert result.shape == (0, 2)
