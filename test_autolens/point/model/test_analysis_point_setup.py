"""
The compute `setup` of `al.AnalysisPoint` (likelihood lifecycle 5C-2): the rule key follows the positions
fit (image plane or source plane), a resolved setup sets the backend, logs the banner and writes
`files/setup.json`, and a likelihood evaluated before the setup resolves raises instead of ignoring it.
"""

import json
import logging

import numpy as np
import pytest

import autofit as af
import autoarray as aa
import autolens as al

from autonerves.environment import Environment

from autolens import exc


def _environment(gpu=False, gpu_name=None, cores=64):
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


MANY_CORE = _environment(cores=64)
A100 = _environment(gpu=True, gpu_name="NVIDIA A100-SXM4-80GB")


@pytest.fixture(autouse=True)
def unset_setup_env(monkeypatch):
    monkeypatch.delenv("PYAUTO_SETUP", raising=False)
    monkeypatch.delenv("PYAUTO_DISABLE_JAX", raising=False)
    monkeypatch.delenv("PYAUTO_RAISE_ANALYSIS_EXCEPTIONS", raising=False)


def _detect(monkeypatch, environment):
    monkeypatch.setattr(Environment, "detect", classmethod(lambda cls: environment))


def _model():
    return af.Collection(
        galaxies=af.Collection(
            lens=al.Galaxy(redshift=0.5, point_0=al.ps.Point(centre=(0.0, 0.0)))
        )
    )


def _analysis(point_dataset, **kwargs):
    solver = al.m.MockPointSolver(model_positions=point_dataset.positions)
    return al.AnalysisPoint(dataset=point_dataset, solver=solver, **kwargs)


@pytest.mark.parametrize(
    "fit_positions_cls, key",
    [
        (al.FitPositionsImagePairAllSolved, ("point_source_image", "image_plane")),
        (al.FitPositionsImagePairAll, ("point_source_image", "image_plane")),
        (al.FitPositionsImagePairRepeat, ("point_source_image", "image_plane")),
        (al.FitPositionsImagePair, ("point_source_image", "image_plane")),
        (al.FitPositionsSource, ("point_source_source", "source_plane")),
        (al.FitPositionsSourceSolved, ("point_source_source", "source_plane")),
        (None, ("point_source_image", "image_plane")),
    ],
)
def test__setup_key_from__follows_fit_positions_cls(
    point_dataset, fit_positions_cls, key
):
    analysis = _analysis(point_dataset, fit_positions_cls=fit_positions_cls)

    assert analysis.setup_key_from(model=_model()) == key


def test__setup_kwarg__stored(point_dataset):
    assert _analysis(point_dataset).setup_input == "auto"
    assert _analysis(point_dataset).setup_defaulted is True
    assert _analysis(point_dataset, use_jax=True).setup_input is None
    assert _analysis(point_dataset, setup=None).setup_input is None
    assert _analysis(point_dataset, setup="auto").setup_input == "auto"

    analysis = _analysis(point_dataset, setup="jax_cpu_na_fp64")

    assert isinstance(analysis.setup_input, aa.Setup)
    assert analysis.setup is None


def test__setup_auto__image_plane__jax_na__banner_and_setup_json(
    point_dataset, monkeypatch, caplog, tmp_path
):
    pytest.importorskip("jax")
    import jax.numpy as jnp

    _detect(monkeypatch, MANY_CORE)

    analysis = _analysis(point_dataset, setup="auto", use_jax=False)

    with caplog.at_level(logging.INFO):
        result = analysis.modify_before_fit(paths=af.DirectoryPaths(), model=_model())

    assert result is analysis
    assert analysis.setup.setup_id == "jax_cpu_na_fp64"
    assert analysis.setup.rule_id == "point-source-image-image-plane-standard-cpu-many-core"
    assert analysis.setup_chosen_by == "auto"
    assert analysis._use_jax is True
    assert analysis._xp is jnp

    banners = [r.message for r in caplog.records if r.message.startswith("Setup:")]
    assert banners == [
        "Setup:    JAX · CPU · no operator · fp64   (rule point_source_image/image_plane)"
    ]

    paths = af.DirectoryPaths(name="point_setup", path_prefix=str(tmp_path))
    analysis.save_attributes(paths=paths)

    with open(paths._files_path / "setup.json") as f:
        setup_dict = json.load(f)

    assert setup_dict["setup_id"] == "jax_cpu_na_fp64"
    assert setup_dict["operator"] == "na"
    assert setup_dict["rule_id"] == "point-source-image-image-plane-standard-cpu-many-core"
    assert setup_dict["chosen_by"] == "auto"


def test__setup_auto__source_plane__a100__jax_gpu_na(point_dataset, monkeypatch):
    _detect(monkeypatch, A100)

    analysis = _analysis(
        point_dataset, setup="auto", fit_positions_cls=al.FitPositionsSource
    )
    analysis.modify_before_fit(paths=af.DirectoryPaths(), model=_model())

    assert analysis.setup.setup_id == "jax_gpu_na_fp64"
    assert analysis.setup.rule_id == "point-source-source-source-plane-standard-gpu-a100"
    assert analysis.setup.extrapolated is False


def test__setup_auto__disable_jax__numpy(point_dataset, monkeypatch):
    _detect(monkeypatch, MANY_CORE)
    monkeypatch.setenv("PYAUTO_DISABLE_JAX", "1")

    analysis = _analysis(point_dataset, setup="auto")
    analysis.modify_before_fit(paths=af.DirectoryPaths(), model=_model())

    assert analysis.setup.setup_id == "jax_cpu_na_fp64"
    assert analysis._use_jax is False
    assert analysis._xp is np


def test__setup_auto__pyauto_setup_env_wins(point_dataset, monkeypatch, tmp_path):
    _detect(monkeypatch, A100)
    monkeypatch.setenv("PYAUTO_SETUP", "numba_cpu_dense_fp64")

    analysis = _analysis(point_dataset, setup="auto", use_jax=True)
    analysis.modify_before_fit(paths=af.DirectoryPaths(), model=_model())

    assert analysis.setup.setup_id == "numba_cpu_dense_fp64"
    assert analysis.setup_chosen_by == "PYAUTO_SETUP"
    assert analysis._use_jax is False

    paths = af.DirectoryPaths(name="point_env", path_prefix=str(tmp_path))
    analysis.save_attributes(paths=paths)

    with open(paths._files_path / "setup.json") as f:
        assert json.load(f)["chosen_by"] == "PYAUTO_SETUP"


def test__no_setup__unchanged__would_choose_logged(
    point_dataset, monkeypatch, caplog, tmp_path
):
    _detect(monkeypatch, MANY_CORE)

    analysis = _analysis(
        point_dataset, fit_positions_cls=al.FitPositionsImagePairRepeat, use_jax=False
    )

    with caplog.at_level(logging.INFO):
        analysis.modify_before_fit(paths=af.DirectoryPaths(), model=_model())

    assert analysis.setup is None
    assert analysis._use_jax is False

    lines = [r.message for r in caplog.records if "ompute setup" in r.message]
    assert lines == [
        "Compute setup: not set (auto would choose JAX · CPU · no operator · fp64; "
        'pass setup="auto")'
    ]

    paths = af.DirectoryPaths(name="point_no_setup", path_prefix=str(tmp_path))
    analysis.save_attributes(paths=paths)

    assert (paths._files_path / "dataset.json").exists()
    assert not (paths._files_path / "setup.json").exists()

    instance = _model().instance_from_unit_vector([])
    assert analysis.log_likelihood_function(instance=instance) == pytest.approx(
        analysis.fit_from(instance=instance).log_likelihood
    )


def test__factor_graph__setup_raises_setup_exception(point_dataset):
    factors = [
        af.AnalysisFactor(
            prior_model=_model(),
            analysis=_analysis(point_dataset, setup="auto", use_jax=False),
        )
    ]

    factor_graph = af.FactorGraphModel(*factors)
    instance = factor_graph.global_prior_model.instance_from_unit_vector([])

    with pytest.raises(exc.SetupException, match="evaluated before the setup") as e:
        factor_graph.log_likelihood_function(instance)

    assert not isinstance(e.value, af.exc.FitException)
    assert "AnalysisPoint" in str(e.value)
