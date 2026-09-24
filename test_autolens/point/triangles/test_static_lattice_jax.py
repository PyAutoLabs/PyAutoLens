"""
JAX tests of the static step-0 lattice on the `PointSolver` path (point-source CPU phase 3,
PyAutoArray#568).

On the JAX path `AbstractSolver._initial_triangles` builds its tiling with
``static_vertices=True``, so step 0 deflects only the geometrically unique lattice vertices
(11 859 rather than 69 849 rows for the +-9.9" / 0.2" lattice of a 100x100, 0.2" grid) and
gathers them back to the triangles. These tests pin that shape (the guard is red on the flat
table), that a change of geometry changes the table, and that positions, image counts, the
likelihood, `jax.grad` and `jax.vmap` agree with the flat per-triangle table the solver used
before -- including near-caustic sources.

Every test skips on the NumPy-only (no ``[optional]`` extras) CI leg.
"""

import importlib

import numpy as np
import pytest


if importlib.util.find_spec("jax") is None:
    pytestmark = pytest.mark.skip(reason="requires jax (the [optional] extras)")

    def test__placeholder_requires_jax():  # pragma: no cover
        pass

else:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    import autolens as al
    from autolens.jax.registration import register_tracer_classes
    from autolens.point.solver.shape_solver import AbstractSolver
    from autoarray.structures.triangles.coordinate_array import (
        CoordinateArrayTriangles,
        static_vertex_table,
    )

    def _solver(shape_native=(100, 100), pixel_scales=0.2):
        return al.PointSolver.for_grid(
            grid=al.Grid2D.uniform(
                shape_native=shape_native, pixel_scales=pixel_scales
            ),
            pixel_scale_precision=0.001,
            magnification_threshold=0.1,
            use_jax=True,
        )

    def _tracer(einstein_radius=1.6, source=(0.07, 0.07)):
        lens = al.Galaxy(
            redshift=0.5,
            mass=al.mp.Isothermal(
                centre=(0.0, 0.0),
                einstein_radius=einstein_radius,
                ell_comps=(0.05, 0.03),
            ),
        )
        return al.Tracer(
            galaxies=[
                lens,
                al.Galaxy(redshift=1.0, point_0=al.ps.Point(centre=source)),
            ]
        )

    def _flat_initial_triangles(self, xp):
        """The pre-phase-3 JAX step-0 tiling: the flat (3N, 2) per-triangle vertex table."""
        return CoordinateArrayTriangles.for_limits_and_scale(
            y_min=self.y_min,
            y_max=self.y_max,
            x_min=self.x_min,
            x_max=self.x_max,
            scale=self.scale,
        )

    def _traced_grid_rows(solver, tracer, source, monkeypatch):
        """Trace one jitted solve and record the row count of every deflected grid."""
        rows = []
        plane_grid = AbstractSolver._plane_grid

        def recording(self, tracer, grid, xp, plane_redshift=None):
            rows.append(int(np.shape(grid)[0]))
            return plane_grid(self, tracer, grid, xp, plane_redshift)

        monkeypatch.setattr(AbstractSolver, "_plane_grid", recording)
        jax.jit(
            lambda: solver.solve(tracer, source_plane_coordinate=source).array
        ).lower()
        monkeypatch.setattr(AbstractSolver, "_plane_grid", plane_grid)
        return rows

    def _solve(solver, einstein_radius, source):
        return np.asarray(
            jax.jit(
                lambda e, s: solver.solve(
                    _tracer(e, (s[0], s[1])), source_plane_coordinate=(s[0], s[1])
                ).array
            )(einstein_radius, jnp.array(source))
        )

    def _finite_sorted(positions):
        positions = positions[np.all(np.isfinite(positions), axis=1)]
        return positions[np.lexsort((positions[:, 1], positions[:, 0]))]

    # Generic, near-caustic (the SIE's tangential caustic is ~0.1" across) and outer sources.
    SOURCES = [
        (0.07, 0.07),
        (0.0, 0.0),
        (0.012, 0.0),
        (0.0, 0.015),
        (0.03, 0.03),
        (0.3, -0.2),
        (1.2, 0.1),
        (1.58, 0.0),
    ]


def test__step_0_deflects_the_unique_lattice_vertices(monkeypatch):
    """
    The shape guard: the first deflected grid of a jitted solve on the 100x100, 0.2" grid is the
    11 859-row unique-vertex table, not the 69 849-row flat one. Red before phase 3.
    """
    solver = _solver()
    tracer = _tracer()
    register_tracer_classes(tracer)

    rows = _traced_grid_rows(solver, tracer, (0.07, 0.07), monkeypatch)

    assert rows[0] == 11859
    assert rows[0] != 3 * 23283
    assert len(rows) == solver.n_steps


def test__changing_the_geometry_changes_the_step_0_table(monkeypatch):
    solver = _solver()
    tracer = _tracer()
    register_tracer_classes(tracer)

    solver.scale = 0.3
    rows = _traced_grid_rows(solver, tracer, (0.07, 0.07), monkeypatch)

    expected = static_vertex_table(
        float(solver.y_min),
        float(solver.y_max),
        float(solver.x_min),
        float(solver.x_max),
        0.3,
    )[0].shape[0]
    assert rows[0] == expected
    assert rows[0] != 11859


@pytest.mark.parametrize("source", SOURCES)
def test__positions_match_the_flat_table(source, monkeypatch):
    """Identical image counts and positions to <= 1e-10 against the pre-phase-3 flat table."""
    solver = _solver()
    register_tracer_classes(_tracer())

    static = _finite_sorted(_solve(solver, 1.6, source))

    jax.clear_caches()
    monkeypatch.setattr(AbstractSolver, "_initial_triangles", _flat_initial_triangles)
    flat = _finite_sorted(_solve(solver, 1.6, source))

    assert static.shape == flat.shape
    assert static.shape[0] > 0
    np.testing.assert_allclose(static, flat, rtol=0.0, atol=1e-10)


@pytest.mark.parametrize("source", [(0.07, 0.07), (0.0, 0.015), (0.3, -0.2)])
def test__positions_match_numpy(source):
    solver = _solver()
    register_tracer_classes(_tracer())

    positions = _finite_sorted(_solve(solver, 1.6, source))

    solver_np = al.PointSolver.for_grid(
        grid=al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.2),
        pixel_scale_precision=0.001,
        magnification_threshold=0.1,
    )
    positions_np = _finite_sorted(
        np.asarray(
            solver_np.solve(_tracer(1.6, source), source_plane_coordinate=source).array
        )
    )

    assert positions.shape == positions_np.shape
    np.testing.assert_allclose(positions, positions_np, rtol=0.0, atol=1e-10)


def _log_likelihood_and_grad(solver):
    positions = al.Grid2DIrregular([(1.53, 0.35), (-1.24, -0.28)])
    noise_map = al.ArrayIrregular([0.05, 0.05])

    def log_likelihood(einstein_radius):
        fit = al.FitPositionsImagePairAll(
            name="point_0",
            data=positions,
            noise_map=noise_map,
            tracer=_tracer(einstein_radius, (0.07, 0.07)),
            solver=solver,
            xp=jnp,
        )
        return fit.log_likelihood

    value, grad = jax.jit(jax.value_and_grad(log_likelihood))(1.6)
    return float(value), float(grad)


def test__log_likelihood_and_grad_match_the_flat_table(monkeypatch):
    solver = _solver()
    register_tracer_classes(_tracer())

    value, grad = _log_likelihood_and_grad(solver)

    jax.clear_caches()
    monkeypatch.setattr(AbstractSolver, "_initial_triangles", _flat_initial_triangles)
    value_flat, grad_flat = _log_likelihood_and_grad(solver)

    assert np.isfinite(value)
    assert grad != 0.0, "jax.grad through the solver is silently zero"
    assert abs(value - value_flat) <= 1e-12 * abs(value_flat)
    assert grad == pytest.approx(grad_flat, rel=1e-10)


def test__vmap_matches_scalar_solves():
    solver = _solver()
    register_tracer_classes(_tracer())

    sources = jnp.array([(0.07, 0.07), (0.0, 0.015), (0.3, -0.2), (0.03, 0.03)])

    batched = np.asarray(
        jax.jit(
            jax.vmap(
                lambda s: solver.solve(
                    _tracer(1.6, (s[0], s[1])), source_plane_coordinate=(s[0], s[1])
                ).array
            )
        )(sources)
    )

    for i, source in enumerate(np.asarray(sources)):
        scalar = _finite_sorted(_solve(solver, 1.6, tuple(source)))
        np.testing.assert_allclose(
            _finite_sorted(batched[i]), scalar, rtol=0.0, atol=1e-10
        )
