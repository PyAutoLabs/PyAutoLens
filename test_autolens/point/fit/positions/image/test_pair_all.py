import numpy as np
import pytest

import autolens as al


point = al.ps.Point(centre=(0.1, 0.1))
galaxy = al.Galaxy(redshift=1.0, point_0=point)
tracer = al.Tracer(galaxies=[al.Galaxy(redshift=0.5), galaxy])


@pytest.fixture
def data():
    return np.array([(0.0, 0.0), (1.0, 0.0)])


@pytest.fixture
def noise_map():
    return np.array([1.0, 1.0])


@pytest.fixture
def fit(data, noise_map):
    model_positions = al.Grid2DIrregular(
        [
            (-1.0749, -1.1),
            (1.19117, 1.175),
        ]
    )

    return al.FitPositionsImagePairAll(
        name="point_0",
        data=data,
        noise_map=noise_map,
        tracer=tracer,
        solver=al.mock.MockPointSolver(model_positions),
    )


def test__fit_positions_image_pair_all__two_model_positions__per_permutation_likelihoods_and_chi_squared_correct(
    fit,
):
    assert np.allclose(
        fit.all_permutations_log_likelihoods(),
        [
            -1.51114426,
            -1.50631469,
        ],
    )
    assert fit.chi_squared == -2.0 * -4.40375330990644


def test__fit_positions_image_pair_all__model_has_inf_position__inf_excluded_from_permutations(
    data,
    noise_map,
):
    model_positions = al.Grid2DIrregular(
        [
            (-1.0749, -1.1),
            (1.19117, 1.175),
            (np.inf, np.inf),
        ]
    )
    fit = al.FitPositionsImagePairAll(
        name="point_0",
        data=data,
        noise_map=noise_map,
        tracer=tracer,
        solver=al.mock.MockPointSolver(model_positions),
    )

    assert np.allclose(
        fit.all_permutations_log_likelihoods(),
        [
            -1.51114426,
            -1.50631469,
        ],
    )
    assert fit.chi_squared == -2.0 * -4.40375330990644


def test__fit_positions_image_pair_all__model_has_duplicate_position__duplicate_permutations_handled(
    data,
    noise_map,
):
    model_positions = al.Grid2DIrregular(
        [
            (-1.0749, -1.1),
            (1.19117, 1.175),
            (1.19117, 1.175),
        ]
    )
    fit = al.FitPositionsImagePairAll(
        name="point_0",
        data=data,
        noise_map=noise_map,
        tracer=tracer,
        solver=al.mock.MockPointSolver(model_positions),
    )

    assert np.allclose(
        fit.all_permutations_log_likelihoods(),
        [-1.14237812, -0.87193683],
    )
    assert fit.chi_squared == -2.0 * -4.211539531047171
