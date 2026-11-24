import os

import numpy as np

import autoarray as aa
import autofit as af
import autolens as al

from autolens.weak.fit import FitWeak
from autolens.weak.model.result import ResultWeak


def _lens_model(einstein_radius=1.6, ell_comps=(0.0, 0.05)):
    return af.Collection(
        galaxies=af.Collection(
            lens=al.Galaxy(
                redshift=0.5,
                mass=al.mp.Isothermal(
                    centre=(0.0, 0.0),
                    ell_comps=ell_comps,
                    einstein_radius=einstein_radius,
                ),
            ),
            source=al.Galaxy(redshift=1.0),
        )
    )


def _make_dataset(noise_sigma=0.3, seed=1, einstein_radius=1.6):
    grid = aa.Grid2DIrregular(
        values=[(0.7, 0.5), (1.0, 1.0), (-0.3, 0.6), (-1.1, -0.8)]
    )
    instance = _lens_model(einstein_radius=einstein_radius).instance_from_unit_vector(
        []
    )
    tracer = al.Tracer(galaxies=[instance.galaxies.lens, instance.galaxies.source])
    simulator = al.SimulatorShearYX(noise_sigma=noise_sigma, seed=seed)
    return simulator.via_tracer_from(tracer=tracer, grid=grid, name="test")


def test__log_likelihood__matches_fit_weak():
    dataset = _make_dataset()

    model = _lens_model()
    instance = model.instance_from_unit_vector([])

    analysis = al.AnalysisWeak(dataset=dataset)

    analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

    tracer = analysis.tracer_via_instance_from(instance=instance)

    fit = FitWeak(dataset=dataset, tracer=tracer)

    assert fit.log_likelihood == analysis_log_likelihood


def test__log_likelihood__noise_free_round_trip_gives_zero_chi_squared():
    """
    `FitWeak.model_shear` uses the same `LensCalc.shear_yx_2d_via_hessian_from` primitive as
    `SimulatorShearYX`, so fitting a noise-free dataset with its own truth tracer must give
    chi_squared == 0 through the analysis path too.
    """
    dataset = _make_dataset(noise_sigma=0.0)
    dataset.noise_map = aa.ArrayIrregular(values=[0.3, 0.3, 0.3, 0.3])

    analysis = al.AnalysisWeak(dataset=dataset)

    instance = _lens_model().instance_from_unit_vector([])

    fit = analysis.fit_from(instance=instance)

    assert fit.chi_squared == 0.0
    np.testing.assert_allclose(fit.residual_map, 0.0, atol=1e-12)


def test__log_likelihood__changes_when_model_wrong():
    dataset = _make_dataset(noise_sigma=0.0)
    dataset.noise_map = aa.ArrayIrregular(values=[0.3, 0.3, 0.3, 0.3])

    analysis = al.AnalysisWeak(dataset=dataset)

    truth = _lens_model(einstein_radius=1.6).instance_from_unit_vector([])
    wrong = _lens_model(einstein_radius=1.0).instance_from_unit_vector([])

    assert analysis.log_likelihood_function(
        instance=wrong
    ) < analysis.log_likelihood_function(instance=truth)


def test__result_class_is_result_weak():
    assert al.AnalysisWeak.Result is ResultWeak


def test__save_attributes__dataset_json_round_trips():
    dataset = _make_dataset()

    analysis = al.AnalysisWeak(dataset=dataset)

    paths = af.DirectoryPaths()

    analysis.save_attributes(paths=paths)

    loaded = al.from_json(file_path=paths._files_path / "dataset.json")

    assert isinstance(loaded, al.WeakDataset)
    assert loaded.name == "test"
    np.testing.assert_allclose(
        np.asarray(loaded.shear_yx), np.asarray(dataset.shear_yx)
    )

    os.remove(paths._files_path / "dataset.json")


# --- setup (likelihood lifecycle 5C-2) ---------------------------------------------------

import json
import logging

import pytest

from autonerves.environment import Environment

from autolens import exc


def _setup_environment(gpu=False, gpu_name=None, cores=64):
    return Environment(
        gpu=gpu,
        gpu_name=gpu_name,
        gpu_memory_gb=None,
        cores=cores,
        ram_gb=None,
        numba=True,
        nufftax=True,
        site="local",
    )


@pytest.fixture(name="detect_many_core")
def make_detect_many_core(monkeypatch):
    monkeypatch.delenv("PYAUTO_SETUP", raising=False)
    monkeypatch.delenv("PYAUTO_DISABLE_JAX", raising=False)
    monkeypatch.setattr(
        Environment, "detect", classmethod(lambda cls: _setup_environment(cores=64))
    )


def test__setup_key_from__weak_isothermal():
    analysis = al.AnalysisWeak(dataset=_make_dataset())

    assert analysis.setup_key_from(model=_lens_model()) == ("weak", "isothermal")


def test__setup_auto__weak_isothermal__jax_dense__banner_and_setup_json(
    detect_many_core, caplog, tmp_path
):
    pytest.importorskip("jax")
    import jax.numpy as jnp

    analysis = al.AnalysisWeak(dataset=_make_dataset(), setup="auto")

    assert analysis._use_jax is False

    with caplog.at_level(logging.INFO):
        analysis.modify_before_fit(paths=af.DirectoryPaths(), model=_lens_model())

    assert analysis.setup.setup_id == "jax_cpu_dense_fp64"
    assert analysis.setup.rule_id == "weak-isothermal-standard-cpu-many-core"
    assert analysis._use_jax is True
    assert analysis._xp is jnp

    banners = [r.message for r in caplog.records if r.message.startswith("Setup: ")]
    assert len(banners) == 1
    assert banners[0].startswith("Setup: jax/cpu/dense/fp64 (64 CPU cores) · rule weak-isothermal")

    paths = af.DirectoryPaths(name="weak_setup", path_prefix=str(tmp_path))
    analysis.save_attributes(paths=paths)

    with open(paths._files_path / "setup.json") as f:
        setup_dict = json.load(f)

    assert setup_dict["setup_id"] == "jax_cpu_dense_fp64"
    assert setup_dict["decision_id"] == "weak-isothermal-standard-cpu-many-core"
    assert setup_dict["chosen_by"] == "auto"


def test__setup_auto__pyauto_setup_env_wins__weak(detect_many_core, monkeypatch):
    monkeypatch.setenv("PYAUTO_SETUP", "numba_cpu_dense_fp64")

    analysis = al.AnalysisWeak(dataset=_make_dataset(), setup="auto", use_jax=True)
    analysis.modify_before_fit(paths=af.DirectoryPaths(), model=_lens_model())

    assert analysis.setup.setup_id == "numba_cpu_dense_fp64"
    assert analysis.setup_chosen_by == "PYAUTO_SETUP"
    assert analysis._use_jax is False
    assert analysis._xp is np


def test__no_setup__weak_unchanged(detect_many_core, caplog, tmp_path):
    dataset = _make_dataset()

    analysis = al.AnalysisWeak(dataset=dataset)

    with caplog.at_level(logging.INFO):
        analysis.modify_before_fit(paths=af.DirectoryPaths(), model=_lens_model())

    assert analysis.setup is None
    assert analysis._use_jax is False
    assert analysis.dataset is dataset

    lines = [r.message for r in caplog.records if r.message.startswith("Setup: ")]
    assert lines == [
        "Setup: auto would choose jax_cpu_dense_fp64 (rule "
        'weak-isothermal-standard-cpu-many-core); pass setup="auto" to use it'
    ]

    paths = af.DirectoryPaths(name="weak_no_setup", path_prefix=str(tmp_path))
    analysis.save_attributes(paths=paths)

    assert not (paths._files_path / "setup.json").exists()


def test__factor_graph__setup_raises_setup_exception__weak():
    factor_graph = af.FactorGraphModel(
        af.AnalysisFactor(
            prior_model=_lens_model(),
            analysis=al.AnalysisWeak(dataset=_make_dataset(), setup="auto"),
        )
    )
    instance = factor_graph.global_prior_model.instance_from_unit_vector([])

    with pytest.raises(exc.SetupException, match="factor graph"):
        factor_graph.log_likelihood_function(instance)
