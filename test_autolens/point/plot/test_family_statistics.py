from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import norm

import autolens as al
from autolens.point.plot.family_statistics import contribution_values


def family_record(fit_positions_cls, model):
    dataset = al.PointDataset(
        name="point_0",
        positions=al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0)]),
        positions_noise_map=al.ArrayIrregular([0.5, 1.0]),
    )
    tracer = al.Tracer(
        galaxies=[
            al.Galaxy(redshift=0.5),
            al.Galaxy(redshift=1.0, point_0=al.ps.Point(centre=(0.1, 0.1))),
        ]
    )
    solver = al.m.MockPointSolver(model_positions=al.Grid2DIrregular(model))
    fit = al.FitPointDataset(
        dataset=dataset, tracer=tracer, solver=solver, fit_positions_cls=fit_positions_cls
    )
    model = np.asarray(model)
    return SimpleNamespace(
        fit=fit,
        model=model[np.isfinite(model).all(axis=1)],
        mixture=isinstance(fit.positions, al.FitPositionsImagePairAll),
    )


@pytest.mark.parametrize(
    "fit_positions_cls, expected_terms, label",
    [
        (al.FitPositionsImagePair, [40.0, 2.0], "Image-plane χ²"),
        (al.FitPositionsSource, [0.08, 23.62], "Source-plane χ²"),
    ],
)
def test_contributions_of_position_residuals(fit_positions_cls, expected_terms, label):
    record = family_record(fit_positions_cls, [(3.0, 1.0), (2.0, 3.0)])
    terms, penalty, actual_label = contribution_values(record)

    assert terms == pytest.approx(expected_terms)
    assert penalty == 0.0
    assert terms.sum() == pytest.approx(record.fit.positions.chi_squared)
    assert actual_label == label


@pytest.mark.parametrize("policy, expected_penalty", [("ignore", 0.0), ("penalize", 16.0)])
def test_unmatched_model_penalty(policy, expected_penalty):
    record = family_record(
        al.FitPositionsImagePairRepeat,
        [(0.5, 0.0), (3.0, 3.5), (3.0, 7.0), (np.inf, np.inf)],
    )
    record.fit.positions.unmatched_model_policy = policy
    terms, penalty, label = contribution_values(record)

    assert terms == pytest.approx([1.0, 0.25])
    assert penalty == pytest.approx(expected_penalty)
    assert terms.sum() + penalty == pytest.approx(record.fit.positions.chi_squared)
    assert label == "Image-plane χ²"


def test_mixture_contributions_include_component_weights_and_noise():
    record = family_record(
        al.FitPositionsImagePairAll, [(0.0, 0.0), (3.0, 4.0), (np.inf, np.inf)]
    )
    terms, penalty, label = contribution_values(record)

    expected_terms = -2 * np.log(
        [norm(scale=sigma).pdf([0.0, 5.0]).mean() for sigma in [0.5, 1.0]]
    )
    assert terms == pytest.approx(expected_terms)
    assert terms.sum() == pytest.approx(record.fit.positions.chi_squared)
    assert penalty == 0.0
    assert label == "-2 log mixture"


def test_mixture_contributions_use_the_no_image_floor():
    record = family_record(al.FitPositionsImagePairAll, [(np.inf, np.inf)])
    terms, penalty, label = contribution_values(record)

    assert terms == pytest.approx([4.0e8, 1.0e8])
    assert terms.sum() == record.fit.positions.chi_squared
    assert penalty == 0.0
    assert label == "-2 log mixture"
