from pathlib import Path
import importlib.util

import pytest

import autofit as af
import autolens as al

from autolens.point.model.result import ResultPoint

directory = Path(__file__).resolve().parent


def _jax_installed() -> bool:
    return importlib.util.find_spec("jax") is not None


def test__pyauto_disable_jax_env_downgrades_use_jax__point(
    monkeypatch, point_dataset
):
    # THE BUG TEST. Before the one-reader fix `AnalysisPoint` had no local
    # env read and `AnalysisLens.__init__` overwrote the base-resolved
    # `self._use_jax` with the raw `use_jax` parameter, so the disable-jax
    # env var was silently a no-op (base set False, AnalysisLens set True).
    # It must now downgrade to False.
    monkeypatch.setenv("PYAUTO_DISABLE_JAX", "1")

    solver = al.m.MockPointSolver(model_positions=point_dataset.positions)

    analysis = al.AnalysisPoint(
        dataset=point_dataset, solver=solver, use_jax=True
    )

    assert analysis._use_jax is False


@pytest.mark.skipif(not _jax_installed(), reason="jax not installed")
def test__use_jax_true_env_unset__not_downgraded__point(
    monkeypatch, point_dataset
):
    # No over-downgrade: with the env var unset and jax installed,
    # `use_jax=True` must survive as `self._use_jax is True`.
    monkeypatch.delenv("PYAUTO_DISABLE_JAX", raising=False)

    solver = al.m.MockPointSolver(model_positions=point_dataset.positions)

    analysis = al.AnalysisPoint(
        dataset=point_dataset, solver=solver, use_jax=True
    )

    assert analysis._use_jax is True


def _test__make_result__result_imaging_is_returned(point_dataset):
    model = af.Collection(
        galaxies=af.Collection(
            lens=al.Galaxy(redshift=0.5, point_0=al.ps.Point(centre=(0.0, 0.0)))
        )
    )

    search = al.m.MockSearch(name="test_search")

    solver = al.m.MockPointSolver(model_positions=point_dataset.positions)

    analysis = al.AnalysisPoint(dataset=point_dataset, solver=solver, use_jax=False)

    result = search.fit(model=model, analysis=analysis)

    assert isinstance(result, ResultPoint)


def test__figure_of_merit__matches_correct_fit_given_galaxy_profiles(
    positions_x2, positions_x2_noise_map
):
    point_dataset = al.PointDataset(
        name="point_0",
        positions=positions_x2,
        positions_noise_map=positions_x2_noise_map,
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=al.Galaxy(redshift=0.5, point_0=al.ps.Point(centre=(0.0, 0.0)))
        )
    )

    solver = al.m.MockPointSolver(model_positions=positions_x2)

    analysis = al.AnalysisPoint(dataset=point_dataset, solver=solver, use_jax=False)

    instance = model.instance_from_unit_vector([])
    analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

    tracer = analysis.tracer_via_instance_from(instance=instance)

    fit_positions = al.FitPositionsImagePairRepeat(
        name="point_0",
        data=positions_x2,
        noise_map=positions_x2_noise_map,
        tracer=tracer,
        solver=solver,
    )

    assert fit_positions.chi_squared == 0.0
    assert fit_positions.log_likelihood == analysis_log_likelihood

    model_positions = al.Grid2DIrregular([(0.0, 1.0), (1.0, 2.0)])
    solver = al.m.MockPointSolver(model_positions=model_positions)

    analysis = al.AnalysisPoint(dataset=point_dataset, solver=solver, use_jax=False)

    analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

    fit_positions = al.FitPositionsImagePairRepeat(
        name="point_0",
        data=positions_x2,
        noise_map=positions_x2_noise_map,
        tracer=tracer,
        solver=solver,
    )

    assert fit_positions.residual_map.in_list == [1.0, 1.0]
    assert fit_positions.chi_squared == 2.0
    assert fit_positions.log_likelihood == analysis_log_likelihood


def test__figure_of_merit__includes_fit_fluxes(
    positions_x2, positions_x2_noise_map, fluxes_x2, fluxes_x2_noise_map
):
    point_dataset = al.PointDataset(
        name="point_0",
        positions=positions_x2,
        positions_noise_map=positions_x2_noise_map,
        fluxes=fluxes_x2,
        fluxes_noise_map=fluxes_x2_noise_map,
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=al.Galaxy(
                redshift=0.5,
                sis=al.mp.IsothermalSph(einstein_radius=1.0),
                point_0=al.ps.PointFlux(flux=1.0),
            )
        )
    )

    solver = al.m.MockPointSolver(model_positions=positions_x2)

    analysis = al.AnalysisPoint(dataset=point_dataset, solver=solver, use_jax=False)

    instance = model.instance_from_unit_vector([])

    analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

    tracer = analysis.tracer_via_instance_from(instance=instance)

    fit_positions = al.FitPositionsImagePairRepeat(
        name="point_0",
        data=positions_x2,
        noise_map=positions_x2_noise_map,
        tracer=tracer,
        solver=solver,
    )

    fit_fluxes = al.FitFluxes(
        name="point_0",
        data=fluxes_x2,
        noise_map=fluxes_x2_noise_map,
        positions=positions_x2,
        tracer=tracer,
    )

    assert (
        fit_positions.log_likelihood + fit_fluxes.log_likelihood
        == analysis_log_likelihood
    )

    model_positions = al.Grid2DIrregular([(0.0, 1.0), (1.0, 2.0)])
    solver = al.m.MockPointSolver(model_positions=model_positions)

    analysis = al.AnalysisPoint(dataset=point_dataset, solver=solver, use_jax=False)

    instance = model.instance_from_unit_vector([])
    analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

    fit_positions = al.FitPositionsImagePairRepeat(
        name="point_0",
        data=positions_x2,
        noise_map=positions_x2_noise_map,
        tracer=tracer,
        solver=solver,
    )

    fit_fluxes = al.FitFluxes(
        name="point_0",
        data=fluxes_x2,
        noise_map=fluxes_x2_noise_map,
        positions=positions_x2,
        tracer=tracer,
    )

    assert fit_positions.residual_map.in_list == [1.0, 1.0]
    assert fit_positions.chi_squared == 2.0
    assert (
        fit_positions.log_likelihood + fit_fluxes.log_likelihood
        == analysis_log_likelihood
    )
