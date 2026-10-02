"""Construction-time extent diagnostics use only observed NumPy data."""

import logging
from types import SimpleNamespace

import numpy as np
import pytest

import autolens as al

LOGGER = "autolens.point.model.analysis"


@pytest.fixture(autouse=True)
def production_mode(monkeypatch, caplog):
    monkeypatch.setenv("PYAUTO_TEST_MODE", "0")
    monkeypatch.delenv("PYAUTO_SMALL_DATASETS", raising=False)
    caplog.set_level(logging.INFO, logger=LOGGER)


def make_analysis(positions=((0, 0),), noise=0.1, bounds=(-2, 2, -2, 2), scale=0.2):
    dataset = al.PointDataset(
        name="point_0", positions=positions, positions_noise_map=noise
    )
    solver = al.PointSolver(
        y_min=bounds[0],
        y_max=bounds[1],
        x_min=bounds[2],
        x_max=bounds[3],
        scale=scale,
        pixel_scale_precision=0.001,
    )
    return al.AnalysisPoint(dataset=dataset, solver=solver, use_jax=False)


def records(caplog):
    return [r for r in caplog.records if r.name == LOGGER]


def test_safe_extent(caplog):
    make_analysis()
    assert records(caplog) == []


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("coordinate", [1.4, 2.1])
def test_each_edge_near_and_outside(caplog, axis, sign, coordinate):
    position = [0.0, 0.0]
    position[axis] = sign * coordinate
    make_analysis(positions=[position])
    assert len(records(caplog)) == 1
    assert records(caplog)[0].levelno == logging.WARNING
    assert "does not guarantee image completeness" in records(caplog)[0].message


def test_offset_asymmetric_bounds(caplog):
    make_analysis(positions=[(10, -4)], bounds=(8, 13, -7, -2))
    assert records(caplog) == []
    make_analysis(positions=[(12.5, -4)], bounds=(8, 13, -7, -2))
    assert len(records(caplog)) == 1


@pytest.mark.parametrize("scale,noise", [(0.5, 0.1), (0.2, 0.3)])
def test_margin_accounts_for_scale_and_noise(caplog, scale, noise):
    make_analysis(positions=[(0.8, 0)], scale=scale, noise=noise)
    assert records(caplog)[0].levelno == logging.WARNING


def test_noise_is_paired_with_each_position(caplog):
    make_analysis(positions=[(1.5, 0), (0, 0)], noise=[0.01, 0.4])
    assert records(caplog) == []


@pytest.mark.parametrize(
    "bounds,expected",
    [
        ((-2.1, 2.1, -2.1, 2.1), False),
        ((-2.2, 2.2, -2.2, 2.2), True),
        ((-10, 10, -2, 2), False),
        ((8, 12, -20, 20), False),
    ],
)
def test_oversize_requires_both_axes_and_margin(caplog, bounds, expected):
    positions = [(10, 0)] if bounds[0] > 0 else [(0, 0)]
    make_analysis(positions=positions, bounds=bounds)
    assert bool(records(caplog)) == expected
    if expected:
        assert records(caplog)[0].levelno == logging.INFO


@pytest.mark.parametrize(
    "variable,value",
    [
        ("PYAUTO_TEST_MODE", "1"),
        ("PYAUTO_TEST_MODE", "2"),
        ("PYAUTO_SMALL_DATASETS", "1"),
    ],
)
def test_modes_suppress_hint_but_keep_warning(monkeypatch, caplog, variable, value):
    monkeypatch.setenv(variable, value)
    make_analysis(bounds=(-10, 10, -10, 10))
    assert records(caplog) == []
    make_analysis(positions=[(2.1, 0)])
    assert records(caplog)[0].levelno == logging.WARNING


@pytest.mark.parametrize(
    "positions,noise",
    [(None, None), ([], []), ([[np.nan, 0]], [0.1]), ([[0, 0]], None)],
)
def test_missing_or_unusable_positions(caplog, positions, noise):
    solver = make_analysis().solver
    dataset = SimpleNamespace(positions=positions, positions_noise_map=noise)
    al.AnalysisPoint(dataset=dataset, solver=solver, use_jax=False)
    assert records(caplog) == []


@pytest.mark.parametrize("solver", [None, SimpleNamespace()])
def test_missing_or_custom_solver(caplog, solver):
    dataset = make_analysis().dataset
    al.AnalysisPoint(dataset=dataset, solver=solver, use_jax=False)
    assert records(caplog) == []


def test_each_analysis_checks_shared_solver_once_not_per_likelihood(
    caplog, monkeypatch
):
    analysis = make_analysis(positions=[(2.1, 0)])
    other = al.AnalysisPoint(
        dataset=analysis.dataset, solver=analysis.solver, use_jax=False
    )
    for item in (analysis, other):
        monkeypatch.setattr(
            item, "fit_from", lambda instance: SimpleNamespace(log_likelihood=1.0)
        )
        for _ in range(3):
            assert item.log_likelihood_function(None) == 1.0
    assert len(records(caplog)) == 2
