import numpy as np
import pytest

import autolens as al


def test__fit_positions_image_pair__three_observed_positions__model_not_repeated__residuals_correct():
    point = al.ps.Point(centre=(0.1, 0.1))
    galaxy = al.Galaxy(redshift=1.0, point_0=point)
    tracer = al.Tracer(galaxies=[al.Galaxy(redshift=0.5), galaxy])

    data = al.Grid2DIrregular([(2.0, 0.0), (1.0, 0.0), (0.0, 0.0)])
    noise_map = al.ArrayIrregular([0.5, 1.0])
    model_data = al.Grid2DIrregular([(4.0, 0.0), (3.0, 0.0), (0.0, 0.0)])

    solver = al.m.MockPointSolver(model_positions=model_data)

    fit = al.FitPositionsImagePair(
        name="point_0",
        data=data,
        noise_map=noise_map,
        tracer=tracer,
        solver=solver,
    )

    assert fit.model_data.in_list == [(4.0, 0.0), (3.0, 0.0), (0.0, 0.0)]
    assert fit.residual_map.in_list == [1.0, 3.0, 0.0]


def test__under_prediction__unmatched_observed_positions_are_penalized():
    # 3 observed, 1 model: the Hungarian assignment pairs one observed position; the other
    # two historically dropped out of the chi-squared entirely (rewarding under-prediction).
    # They now contribute their distance to the nearest model position, in data order.
    point = al.ps.Point(centre=(0.1, 0.1))
    galaxy = al.Galaxy(redshift=1.0, point_0=point)
    tracer = al.Tracer(galaxies=[al.Galaxy(redshift=0.5), galaxy])

    data = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0), (6.0, 8.0)])
    noise_map = al.ArrayIrregular([1.0, 1.0, 1.0])
    model_data = al.Grid2DIrregular([(0.0, 0.0)])

    solver = al.m.MockPointSolver(model_positions=model_data)

    fit = al.FitPositionsImagePair(
        name="point_0",
        data=data,
        noise_map=noise_map,
        tracer=tracer,
        solver=solver,
    )

    assert fit.residual_map.in_list == [0.0, 5.0, 10.0]
    assert fit.chi_squared == pytest.approx(125.0, 1.0e-4)


def test__under_prediction__no_model_images_hits_finite_floor():
    point = al.ps.Point(centre=(0.1, 0.1))
    galaxy = al.Galaxy(redshift=1.0, point_0=point)
    tracer = al.Tracer(galaxies=[al.Galaxy(redshift=0.5), galaxy])

    data = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0)])
    noise_map = al.ArrayIrregular([0.5, 1.0])
    model_data = al.Grid2DIrregular(np.zeros(shape=(0, 2)))

    solver = al.m.MockPointSolver(model_positions=model_data)

    fit = al.FitPositionsImagePair(
        name="point_0",
        data=data,
        noise_map=noise_map,
        tracer=tracer,
        solver=solver,
    )

    assert fit.residual_map.in_list == [1.0e4, 1.0e4]
    assert np.isfinite(float(fit.log_likelihood))
