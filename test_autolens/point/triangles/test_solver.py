import numpy as np
import pytest
import time
from typing import Tuple

import autogalaxy as ag
import autofit as af
from autolens import PointSolver, Tracer

from autoarray.structures.triangles.coordinate_array import (
    CoordinateArrayTriangles,
)

from autolens.mock import NullTracer


@pytest.fixture(autouse=True)
def register(tracer):
    af.Model.from_instance(tracer)


@pytest.fixture
def solver(grid):
    return PointSolver.for_grid(
        grid=grid, pixel_scale_precision=0.01, magnification_threshold=1e-8
    )


def test_solver(solver):
    mass_profile = ag.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.0,
    )
    tracer = Tracer(
        galaxies=[ag.Galaxy(redshift=0.5, mass=mass_profile)],
    )
    result = solver.solve(tracer, source_plane_coordinate=(0.0, 0.0))
    assert result


@pytest.mark.parametrize(
    "source_plane_coordinate",
    [
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.5, 0.5),
        (0.1, 0.1),
        (-1.0, -1.0),
    ],
)
def test_trivial(
    source_plane_coordinate: Tuple[float, float],
    grid,
    solver,
):
    coordinates = solver.solve(
        NullTracer(),
        source_plane_coordinate=source_plane_coordinate,
    )
    coordinates = coordinates.array[~np.isnan(coordinates.array).any(axis=1)]
    assert coordinates[0] == pytest.approx(source_plane_coordinate, abs=1.0e-1)


def test_real_example_jax(grid, tracer):

    import jax.numpy as jnp

    jax_solver = PointSolver.for_grid(
        grid=grid, pixel_scale_precision=0.001, xp=jnp, magnification_threshold=1e-8
    )

    result = jax_solver.solve(
        tracer=tracer, source_plane_coordinate=(0.07, 0.07), remove_infinities=True
    )

    assert len(result) == 5

    result = jax_solver.solve(
        tracer=tracer, source_plane_coordinate=(0.07, 0.07), remove_infinities=False
    )

    assert len(result) == 15
