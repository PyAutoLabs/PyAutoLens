"""
`FactorGraphModel`s mixing PyAutoLens dataset types with default-constructed analyses (R6: every
analysis defaults to `setup="auto"`): each factor resolves its own setup, and factors left on the
default adopt one shared backend so the whole graph fits without `use_jax=` pins.
"""
import importlib.util

import numpy as np
import pytest

import autofit as af
import autofit.graphical as g
import autolens as al

from autofit import exc
from autonerves.environment import Environment

MANY_CORE = Environment(
    gpu=False,
    gpu_name=None,
    gpu_memory_gb=None,
    cores=64,
    ram_gb=None,
    numba=True,
    nufftax=True,
    site="local",
)

HAS_JAX = importlib.util.find_spec("jax") is not None


@pytest.fixture(autouse=True)
def many_core(monkeypatch):
    monkeypatch.delenv("PYAUTO_SETUP", raising=False)
    monkeypatch.delenv("PYAUTO_DISABLE_JAX", raising=False)
    monkeypatch.setattr(Environment, "detect", classmethod(lambda cls: MANY_CORE))


def _imaging_model():
    return af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy, redshift=0.5, mass=al.mp.Isothermal, light=al.lp.Sersic
            ),
            source=af.Model(al.Galaxy, redshift=1.0, light=al.lp.Sersic),
        )
    )


def _point_model():
    return af.Collection(
        galaxies=af.Collection(
            lens=al.Galaxy(redshift=0.5, point_0=al.ps.Point(centre=(0.0, 0.0)))
        )
    )


def _graph(*factors, **kwargs):
    return g.FactorGraphModel(
        *[
            g.AnalysisFactor(model, analysis=analysis, name=f"factor_{i}")
            for i, (analysis, model) in enumerate(factors)
        ],
        **kwargs,
    )


def _modify(graph, name):
    return graph.modify_before_fit(
        paths=af.DirectoryPaths(name=name), model=graph.global_prior_model
    )


def test__default_imaging_and_interferometer__pass_modify_before_fit(
    masked_imaging_7x7, interferometer_7
):
    graph = _graph(
        (al.AnalysisImaging(dataset=masked_imaging_7x7), _imaging_model()),
        (al.AnalysisInterferometer(dataset=interferometer_7), _imaging_model()),
    )

    _modify(graph, "lens_graph_imaging_interferometer")

    for factor in graph.model_factors:
        assert factor.analysis.setup is None
        assert factor.analysis.is_jax is HAS_JAX

    graph.check_setup_agreement()
    graph.check_backend_agreement()

    instance = graph.global_prior_model.instance_from_prior_medians()
    assert np.isfinite(float(graph.log_likelihood_function(instance)))


def test__default_imaging_and_point__pass_modify_before_fit(
    masked_imaging_7x7, point_dataset
):
    solver = al.m.MockPointSolver(model_positions=point_dataset.positions)

    graph = _graph(
        (al.AnalysisImaging(dataset=masked_imaging_7x7), _imaging_model()),
        (
            al.AnalysisPoint(
                dataset=point_dataset,
                solver=solver,
                fit_positions_cls=al.FitPositionsImagePairAll,
            ),
            _point_model(),
        ),
    )

    _modify(graph, "lens_graph_imaging_point")

    for factor in graph.model_factors:
        assert factor.analysis.setup is None
        assert factor.analysis.is_jax is HAS_JAX

    graph.check_setup_agreement()
    graph.check_backend_agreement()

    instance = graph.global_prior_model.instance_from_prior_medians()
    assert np.isfinite(float(graph.log_likelihood_function(instance)))


def test__explicit_disagreeing_setups__still_raise(masked_imaging_7x7, point_dataset):
    solver = al.m.MockPointSolver(model_positions=point_dataset.positions)

    graph = _graph(
        (
            al.AnalysisImaging(dataset=masked_imaging_7x7, setup="numba_cpu_dense_fp64"),
            _imaging_model(),
        ),
        (
            al.AnalysisPoint(dataset=point_dataset, solver=solver, setup="auto"),
            _point_model(),
        ),
    )

    if not HAS_JAX:
        pytest.skip("the point auto setup is JAX; without JAX it may still disagree on numpy")

    with pytest.raises(exc.SearchException, match="setups passed explicitly"):
        _modify(graph, "lens_graph_explicit_bad")
