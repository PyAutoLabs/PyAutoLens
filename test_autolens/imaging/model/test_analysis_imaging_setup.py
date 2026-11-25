"""
The compute `setup` of `al.AnalysisImaging` (likelihood lifecycle 5C-2): threading of the `setup=`
kwarg to the PyAutoGalaxy base class, `setup="auto"` resolution in `modify_before_fit`, the backend
(`_use_jax` / `_xp`) staying in step with the resolved setup, the factor-graph guard and the
operator / backend mismatch surfacing before sampling.
"""

import copy
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


def _rectangular_model():
    pixelization = af.Model(
        al.Pixelization,
        mesh=al.mesh.RectangularUniform(shape=(3, 3)),
        regularization=al.reg.Constant,
    )
    return af.Collection(
        galaxies=af.Collection(
            lens=al.Galaxy(redshift=0.5, mass=al.mp.IsothermalSph(einstein_radius=1.0)),
            source=af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization),
        )
    )


def _light_model():
    return af.Collection(
        galaxies=af.Collection(
            lens=al.Galaxy(redshift=0.5, light=al.lp.Sersic(intensity=0.1))
        )
    )


def _analysis(dataset, **kwargs):
    return al.AnalysisImaging(
        dataset=dataset,
        raise_inversion_positions_likelihood_exception=False,
        **kwargs,
    )


def test__setup_kwarg__threaded_to_galaxy_base(masked_imaging_7x7):
    assert _analysis(masked_imaging_7x7).setup_input is None
    assert _analysis(masked_imaging_7x7, setup="AUTO").setup_input == "auto"

    analysis = _analysis(masked_imaging_7x7, setup="numba_cpu_dense_fp64")

    assert isinstance(analysis.setup_input, aa.Setup)
    assert analysis.setup_input.setup_id == "numba_cpu_dense_fp64"
    assert analysis.setup is None

    with pytest.raises(ValueError):
        _analysis(masked_imaging_7x7, setup="not_a_setup")


def test__setup_auto__cpu_many_core_rectangular__numba_sparse_operator__numpy_xp(
    masked_imaging_7x7, monkeypatch, caplog
):
    pytest.importorskip("numba")
    from autoarray.inversion.inversion.imaging_numba.inversion_imaging_numba_util import (
        SparseLinAlgImagingNumba,
    )

    _detect(monkeypatch, MANY_CORE)

    analysis = _analysis(masked_imaging_7x7, setup="auto", use_jax=True)

    with caplog.at_level(logging.INFO):
        result = analysis.modify_before_fit(
            paths=af.DirectoryPaths(), model=_rectangular_model()
        )

    assert result is analysis
    assert analysis.setup.setup_id == "numba_cpu_sparse_fp64"
    assert analysis.setup_chosen_by == "auto"
    assert analysis._use_jax is False
    assert analysis._xp is np
    assert analysis.dataset.setup == analysis.setup
    assert isinstance(analysis.dataset.sparse_operator, SparseLinAlgImagingNumba)

    banners = [r.message for r in caplog.records if r.message.startswith("Setup:")]
    assert banners == [
        "Setup:    numba · CPU · sparse operator · fp64   (rule imaging/rectangular)"
    ]


def test__setup_auto__a100_rectangular__jax_dense__jax_xp(
    masked_imaging_7x7, monkeypatch
):
    pytest.importorskip("jax")
    import jax.numpy as jnp

    _detect(monkeypatch, A100)

    analysis = _analysis(masked_imaging_7x7, setup="auto", use_jax=False)

    assert analysis._xp is np

    analysis.modify_before_fit(paths=af.DirectoryPaths(), model=_rectangular_model())

    assert analysis.setup.setup_id == "jax_gpu_dense_fp64"
    assert analysis._use_jax is True
    assert analysis._xp is jnp
    assert analysis.dataset.sparse_operator is None


def test__modify_before_fit__resolves_setup_once(masked_imaging_7x7, monkeypatch):
    _detect(monkeypatch, MANY_CORE)

    calls = {"n": 0}

    original = al.AnalysisImaging.resolve_setup

    def counting(self, model, **kwargs):
        calls["n"] += 1
        return original(self, model=model, **kwargs)

    monkeypatch.setattr(al.AnalysisImaging, "resolve_setup", counting)

    analysis = _analysis(masked_imaging_7x7, setup="numba_cpu_dense_fp64")
    analysis.modify_before_fit(paths=af.DirectoryPaths(), model=_light_model())

    assert calls["n"] == 1
    assert analysis.setup.setup_id == "numba_cpu_dense_fp64"


def test__setup_auto__pyauto_setup_env_wins(masked_imaging_7x7, monkeypatch, tmp_path):
    _detect(monkeypatch, A100)
    monkeypatch.setenv("PYAUTO_SETUP", "numba_cpu_dense_fp64")

    analysis = _analysis(masked_imaging_7x7, setup="auto", use_jax=True)
    analysis.modify_before_fit(paths=af.DirectoryPaths(), model=_rectangular_model())

    assert analysis.setup.setup_id == "numba_cpu_dense_fp64"
    assert analysis.setup_chosen_by == "PYAUTO_SETUP"
    assert analysis._use_jax is False
    assert analysis._xp is np

    paths = af.DirectoryPaths(name="pyauto_setup", path_prefix=str(tmp_path))
    analysis.save_attributes(paths=paths)

    with open(paths._files_path / "setup.json") as f:
        setup_dict = json.load(f)

    assert setup_dict["setup_id"] == "numba_cpu_dense_fp64"
    assert setup_dict["chosen_by"] == "PYAUTO_SETUP"


def test__no_setup__dataset_and_backend_unchanged(
    masked_imaging_7x7, monkeypatch, tmp_path, caplog
):
    _detect(monkeypatch, MANY_CORE)

    analysis = _analysis(masked_imaging_7x7, use_jax=False)

    with caplog.at_level(logging.INFO):
        analysis.modify_before_fit(
            paths=af.DirectoryPaths(), model=_rectangular_model()
        )

    assert analysis.dataset is masked_imaging_7x7
    assert analysis._use_jax is False
    assert analysis.setup is None

    lines = [r.message for r in caplog.records if "ompute setup" in r.message]
    assert lines == [
        "Compute setup: not set (auto would choose numba · CPU · sparse operator · fp64; "
        'pass setup="auto")'
    ]

    paths = af.DirectoryPaths(name="no_setup", path_prefix=str(tmp_path))
    analysis.save_attributes(paths=paths)

    assert not (paths._files_path / "setup.json").exists()

    instance = _light_model().instance_from_unit_vector([])
    assert analysis.log_likelihood_function(instance=instance) == pytest.approx(
        al.FitImaging(
            dataset=masked_imaging_7x7,
            tracer=analysis.tracer_via_instance_from(instance=instance),
        ).figure_of_merit
    )


def test__explicit_setup__search_fit__setup_json_written(masked_imaging_7x7, monkeypatch):
    _detect(monkeypatch, MANY_CORE)

    analysis = _analysis(masked_imaging_7x7, setup="numba_cpu_dense_fp64")

    search = al.m.MockSearch(name="test_search_lens_setup")
    search.fit(model=_light_model(), analysis=analysis)

    with open(search.paths._files_path / "setup.json") as f:
        setup_dict = json.load(f)

    assert setup_dict["setup_id"] == "numba_cpu_dense_fp64"
    assert setup_dict["chosen_by"] == "explicit"


def test__factor_graph__setup_raises_setup_exception(masked_imaging_7x7):
    factors = [
        af.AnalysisFactor(
            prior_model=_light_model(),
            analysis=_analysis(masked_imaging_7x7, setup="auto", use_jax=False),
        )
        for _ in range(2)
    ]

    factor_graph = af.FactorGraphModel(*factors)
    instance = factor_graph.global_prior_model.instance_from_unit_vector([])

    with pytest.raises(exc.SetupException, match="factor graph") as e:
        factor_graph.log_likelihood_function(instance)

    assert not isinstance(e.value, af.exc.FitException)
    assert "apply_setup" in str(e.value)
    assert "use_jax" in str(e.value)


def test__factor_graph_guard__fit_from_unaffected(masked_imaging_7x7):
    analysis = _analysis(masked_imaging_7x7, setup="auto", use_jax=False)

    instance = _light_model().instance_from_unit_vector([])

    fit = analysis.fit_from(instance=instance)

    assert fit.figure_of_merit == pytest.approx(
        al.FitImaging(
            dataset=masked_imaging_7x7,
            tracer=analysis.tracer_via_instance_from(instance=instance),
        ).figure_of_merit
    )


def test__factor_graph__no_setup__unchanged(masked_imaging_7x7):
    factors = [
        af.AnalysisFactor(
            prior_model=_light_model(),
            analysis=_analysis(masked_imaging_7x7, use_jax=False),
        )
        for _ in range(2)
    ]

    factor_graph = af.FactorGraphModel(*factors)
    instance = factor_graph.global_prior_model.instance_from_unit_vector([])

    assert np.isfinite(factor_graph.log_likelihood_function(instance))


def test__mismatched_operator__numpy_path__raises_at_modify_before_fit(
    masked_imaging_7x7, monkeypatch
):
    """
    A jax sparse setup with JAX disabled runs the numpy path. A stale numba operator left on the dataset
    (simulated by an `apply_setup` that records the setup without rebuilding the operator) is a mismatch,
    which must raise an `InversionException` at `modify_before_fit` rather than be turned into a
    `FitException` (a resample) on every likelihood call.
    """
    pytest.importorskip("numba")

    monkeypatch.setenv("PYAUTO_DISABLE_JAX", "1")
    _detect(monkeypatch, MANY_CORE)

    dataset = masked_imaging_7x7.apply_setup("numba_cpu_sparse_fp64")

    def apply_setup_keeping_operator(self, setup, **kwargs):
        dataset_copy = copy.copy(self)
        dataset_copy.setup = aa.Setup.from_setup_id(setup) if isinstance(setup, str) else setup
        return dataset_copy

    monkeypatch.setattr(aa.Imaging, "apply_setup", apply_setup_keeping_operator)

    analysis = _analysis(dataset, setup="jax_cpu_sparse_fp64")

    with pytest.raises(aa.exc.InversionException, match="mismatch") as e:
        analysis.modify_before_fit(paths=af.DirectoryPaths(), model=_rectangular_model())

    assert not isinstance(e.value, af.exc.FitException)
    assert analysis._use_jax is False
