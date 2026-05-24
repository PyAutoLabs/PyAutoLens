"""Unit tests for the ``PointSolver(use_jax=True)`` constructor wiring.

Per the PyAutoArray dependency-graph rule, library unit tests stay NumPy-only —
cross-xp numerical parity for the actual JAX execution path lives in
``autolens_workspace_test/scripts/point_source/solver_use_jax_parity.py``.
These tests cover only the constructor wiring, the default fallbacks
(``xp=None``, ``remove_infinities=None``), and the pytree-tree-flatten roundtrip.
"""
import numpy as np

import autolens as al


def test_use_jax_defaults_false():
    solver = al.PointSolver.for_grid(
        grid=al.Grid2D.uniform(shape_native=(10, 10), pixel_scales=1.0),
        pixel_scale_precision=0.01,
    )
    assert solver.use_jax is False
    assert solver._xp is np


def test_use_jax_flag_threads_through_for_grid():
    solver = al.PointSolver.for_grid(
        grid=al.Grid2D.uniform(shape_native=(10, 10), pixel_scales=1.0),
        pixel_scale_precision=0.01,
        use_jax=True,
    )
    assert solver.use_jax is True


def test_use_jax_flag_threads_through_for_limits_and_scale():
    solver = al.PointSolver.for_limits_and_scale(
        y_min=-1.0,
        y_max=1.0,
        x_min=-1.0,
        x_max=1.0,
        scale=0.1,
        pixel_scale_precision=0.01,
        use_jax=True,
    )
    assert solver.use_jax is True


def test_tree_flatten_roundtrips_use_jax():
    solver = al.PointSolver.for_grid(
        grid=al.Grid2D.uniform(shape_native=(10, 10), pixel_scales=1.0),
        pixel_scale_precision=0.01,
        use_jax=True,
    )
    _children, aux = solver.tree_flatten()
    rebuilt = al.PointSolver.tree_unflatten(aux, _children)
    assert rebuilt.use_jax is True
    assert rebuilt.y_min == solver.y_min
    assert rebuilt.scale == solver.scale


def test_solve_default_remove_infinities_numpy(grid):
    """On the NumPy path the solve() default still removes infinities (back-compat)."""
    solver = al.PointSolver.for_grid(
        grid=grid, pixel_scale_precision=0.5, magnification_threshold=1e-8
    )
    tracer = al.Tracer(
        galaxies=[
            al.Galaxy(
                redshift=0.5,
                mass=al.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.0),
            )
        ]
    )
    result = solver.solve(tracer, source_plane_coordinate=(0.0, 0.0))
    # Removing infinities means no row should contain inf on the default numpy path.
    assert not np.isinf(np.asarray(result.array)).any()
