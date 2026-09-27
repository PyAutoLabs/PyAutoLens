"""
``AnalysisPoint`` declares ``gradient_mode = "forward"`` (PyAutoFit#1648).

The source-plane point-source likelihood carries an inner forward-mode lensing Hessian, so reverse
mode runs reverse-over-forward through every mass profile; ``jax.jacfwd`` over the flat parameter
vector was 2-4.5x faster and compiled up to 8x faster in autolens_profiling #327/#331. These tests
pin that the declaration exists, that the forward gradient IS the reverse gradient on this
likelihood, and that a real ``MultiStartGradient`` fit runs end-to-end in forward mode.

The two JAX checks run in a subprocess, deliberately. Building a JAX ``Fitness`` registers the
model's classes (``Galaxy`` included) through ``autofit.jax.register_model``, and JAX has no way to
unregister a pytree node. ``autolens.jax.registration.register_tracer_classes`` -- used by the
``PointSolver`` lattice tests later in this suite -- registers ``Galaxy`` through
``autoarray.abstract_ndarray.register_instance_pytree``, which raises on a class another route
already registered, so running these checks in-process would break every test after them. A fresh
interpreter keeps them independent of test order. The same entry point (``python <this file>
parity <output_dir>`` / ``python <this file> multi_start <output_dir> declared reverse``) is what a GPU run invokes.
"""

import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest

import autofit as af
import autolens as al

jax = pytest.importorskip("jax")

from autofit.jax.gradient import resolve_gradient_mode  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

TEST_DIR = Path(__file__).resolve().parents[2]

# The autolens_profiling "simple" point-source dataset (Isothermal, einstein_radius 1.6, source
# at (0.07, 0.07)), inlined so the test needs no workspace.
POSITIONS = [
    (-1.0285552366239088, -1.092385071774359),
    (0.3509304746580634, 1.6278794213275605),
    (1.5748094742113987, 0.4342241069464528),
    (1.2452712382900812, 1.2238422712179249),
]


def _dataset():
    return al.PointDataset(
        name="point_0",
        positions=al.Grid2DIrregular(POSITIONS),
        positions_noise_map=al.ArrayIrregular([0.05] * len(POSITIONS)),
    )


def _model():
    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.005)
    mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.005)
    mass.einstein_radius = af.GaussianPrior(mean=1.6, sigma=0.05)
    mass.ell_comps.ell_comps_0 = af.GaussianPrior(mean=0.05263158, sigma=0.01)
    mass.ell_comps.ell_comps_1 = af.GaussianPrior(mean=0.0, sigma=0.01)
    lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)
    source = af.Model(al.Galaxy, redshift=1.0, point_0=af.Model(al.ps.PointSolved))
    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def _analysis():
    return al.AnalysisPoint(
        dataset=_dataset(),
        solver=None,
        fit_positions_cls=al.FitPositionsSourceSolved,
        use_jax=True,
    )


def test__analysis_point_declares_forward_mode():
    assert al.AnalysisPoint.gradient_mode == "forward"
    assert resolve_gradient_mode(_analysis()) == "forward"
    assert resolve_gradient_mode(_analysis(), override="reverse") == "reverse"


# --------------------------------------------------------------------------
# Subprocess checks
# --------------------------------------------------------------------------


def _check_parity():
    """Forward == reverse ``(value, grad)`` of the ``FitPositionsSourceSolved`` likelihood through
    ``Fitness`` on the flat vector: prior medians + one draw per ``PRNGKey(0..15)``."""
    from autofit.jax import register_model
    from autofit.jax.gradient import value_and_grad_from
    from autofit.non_linear.fitness import Fitness

    model = _model()
    # Load-bearing: without registration jax.grad of an AnalysisPoint likelihood is silently
    # all-zero (Fitness registers it too on a JAX analysis; explicit so the check does not
    # depend on that).
    register_model(model)

    fitness = Fitness(
        model=model,
        analysis=_analysis(),
        fom_is_log_likelihood=False,
        convert_to_chi_squared=True,
    )

    reverse = jax.jit(jax.value_and_grad(fitness.call))
    forward = jax.jit(value_and_grad_from(fitness.call, "forward"))
    # `Fitness.grad` builds the analysis's declared mode (forward); jitted here only for speed.
    fitness_grad = jax.jit(fitness.grad)

    vectors = [np.asarray(model.physical_values_from_prior_medians, dtype=float)]
    for key in range(16):
        unit = np.asarray(
            jax.random.uniform(
                jax.random.PRNGKey(key), (model.prior_count,), minval=0.25, maxval=0.75
            ),
            dtype=float,
        )
        vectors.append(np.asarray(model.vector_from_unit_vector(list(unit)), dtype=float))

    worst = 0.0
    for vector in vectors:
        value_r, grad_r = reverse(vector)
        value_f, grad_f = forward(vector)
        grad_r = np.asarray(grad_r)

        assert np.isfinite(value_r)
        assert np.all(np.isfinite(grad_r))
        assert np.any(grad_r != 0.0)

        np.testing.assert_allclose(value_f, value_r, rtol=1e-10)
        scale = np.max(np.abs(grad_r))
        np.testing.assert_allclose(grad_f, grad_r, rtol=1e-8, atol=1e-12 * scale)
        np.testing.assert_allclose(
            fitness_grad(vector), grad_r, rtol=1e-8, atol=1e-12 * scale
        )
        worst = max(worst, float(np.max(np.abs(np.asarray(grad_f) - grad_r)) / scale))

    print(f"PARITY_OK n_vectors={len(vectors)} max_rel_grad_diff={worst:.3e}")


def _check_multi_start(gradient_mode=None):
    """A short real ``MultiStartAdam`` point-source fit; ``gradient_mode=None`` uses the
    ``AnalysisPoint`` declaration. Prints the resolved mode and the best-fit vector."""
    search = af.MultiStartAdam(
        name=f"point_gradient_mode_{gradient_mode}_{uuid.uuid4().hex}",
        n_starts=4,
        n_steps=5,
        seed=3,
        convergence=af.MultiStartGradientConvergence(check_for_convergence=False),
        gradient_mode=gradient_mode,
    )

    try:
        result = search.fit(model=_model(), analysis=_analysis())
    finally:
        shutil.rmtree(search.paths.output_path, ignore_errors=True)

    assert np.isfinite(result.log_likelihood)
    tag = gradient_mode or "declared"
    best = ",".join(repr(float(v)) for v in result.samples.max_log_likelihood(as_instance=False))
    print(f"{tag}.MODE={result.samples.samples_info['gradient_mode']}")
    print(f"{tag}.BEST={best}")
    print(f"{tag}.LOG_LIKELIHOOD={float(result.log_likelihood)!r}")


def _run(*args, tmp_path):
    result = subprocess.run(
        [sys.executable, __file__, *args],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**os.environ, "PYAUTO_SKIP_WORKSPACE_VERSION_CHECK": "1"},
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    parsed = dict(
        line.split("=", 1)
        for line in result.stdout.splitlines()
        if line.split("=", 1)[0].endswith((".MODE", ".BEST", ".LOG_LIKELIHOOD"))
    )
    return parsed, result.stdout


def test__forward_gradient_matches_reverse_through_fitness(tmp_path):
    _, stdout = _run("parity", str(tmp_path), tmp_path=tmp_path)

    assert "PARITY_OK n_vectors=17" in stdout


def test__multi_start_gradient_point_source_fit_runs_in_forward_mode(tmp_path):
    # One interpreter, both fits: the AnalysisPoint declaration, then the search override.
    parsed, _ = _run("multi_start", str(tmp_path), "declared", "reverse", tmp_path=tmp_path)

    assert parsed["declared.MODE"] == "forward"
    assert parsed["reverse.MODE"] == "reverse"
    np.testing.assert_allclose(
        [float(v) for v in parsed["declared.BEST"].split(",")],
        [float(v) for v in parsed["reverse.BEST"].split(",")],
        rtol=1e-6,
    )


if __name__ == "__main__":
    from autonerves import conf

    command = sys.argv[1]
    # The suite's config, as `test_autolens/conftest.py` pushes it for in-process tests.
    conf.instance.push(new_path=TEST_DIR / "config", output_path=Path(sys.argv[2]))

    if command == "parity":
        _check_parity()
    elif command == "multi_start":
        # Each further argument is a mode to fit with; "declared" = no override.
        for mode in sys.argv[3:] or ["declared"]:
            _check_multi_start(None if mode == "declared" else mode)
    else:
        raise SystemExit(f"unknown command {command!r}")
