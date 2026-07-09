import numpy as np
import pytest

import autolens as al


def test__fit_positions_image_pair_repeat__two_residuals_and_all_fit_quantities_correct():
    point = al.ps.Point(centre=(0.1, 0.1))
    galaxy = al.Galaxy(redshift=1.0, point_0=point)
    tracer = al.Tracer(galaxies=[al.Galaxy(redshift=0.5), galaxy])

    data = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0)])
    noise_map = al.ArrayIrregular([0.5, 1.0])
    model_data = al.Grid2DIrregular([(3.0, 1.0), (2.0, 3.0)])

    solver = al.m.MockPointSolver(model_positions=model_data)

    fit = al.FitPositionsImagePairRepeat(
        name="point_0",
        data=data,
        noise_map=noise_map,
        tracer=tracer,
        solver=solver,
    )

    assert fit.model_data.in_list == [(3.0, 1.0), (2.0, 3.0)]
    assert fit.noise_map.in_list == [0.5, 1.0]
    assert fit.residual_map.in_list == [np.sqrt(10.0), np.sqrt(2.0)]
    assert fit.normalized_residual_map.in_list == [
        np.sqrt(10.0) / 0.5,
        np.sqrt(2.0) / 1.0,
    ]
    assert fit.chi_squared_map.in_list == [
        (np.sqrt(10.0) / 0.5) ** 2,
        np.sqrt(2.0) ** 2.0,
    ]
    assert fit.chi_squared == pytest.approx(42.0, 1.0e-4)
    assert fit.noise_normalization == pytest.approx(2.28945, 1.0e-4)
    assert fit.log_likelihood == pytest.approx(-22.14472, 1.0e-4)


def test__fit_positions_image_pair_repeat__three_observed_positions__model_allocated_by_repeat():
    point = al.ps.Point(centre=(0.1, 0.1))
    galaxy = al.Galaxy(redshift=1.0, point_0=point)
    tracer = al.Tracer(galaxies=[al.Galaxy(redshift=0.5), galaxy])

    data = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0), (3.0, 4.0)])
    noise_map = al.ArrayIrregular([0.5, 1.0])
    model_data = al.Grid2DIrregular([(3.0, 1.0), (3.0, 4.0)])

    solver = al.m.MockPointSolver(model_positions=model_data)

    fit = al.FitPositionsImagePairRepeat(
        name="point_0",
        data=data,
        noise_map=noise_map,
        tracer=tracer,
        solver=solver,
    )

    assert fit.model_data.in_list == [(3.0, 1.0), (3.0, 4.0)]
    assert fit.residual_map.in_list == [np.sqrt(10.0), 0.0, 0.0]


def test__over_prediction__unmatched_bright_model_image_is_penalized():
    # 2 observed, 3 model: the third model image is far from every observed position and
    # (with a no-mass tracer, magnification 1 everywhere) counts as a detectable extra.
    point = al.ps.Point(centre=(0.1, 0.1))
    galaxy = al.Galaxy(redshift=1.0, point_0=point)
    tracer = al.Tracer(galaxies=[al.Galaxy(redshift=0.5), galaxy])

    data = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0)])
    noise_map = al.ArrayIrregular([0.5, 1.0])
    model_data = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0), (10.0, 10.0)])

    solver = al.m.MockPointSolver(model_positions=model_data)

    fit = al.FitPositionsImagePairRepeat(
        name="point_0",
        data=data,
        noise_map=noise_map,
        tracer=tracer,
        solver=solver,
    )

    # Matched residuals are zero; the whole chi-squared is the extra-image penalty:
    # distance from (10, 10) to nearest observed (3, 4) over the mean noise (0.75).
    assert int(fit.n_unmatched_model_positions) == 1
    penalty_distance = np.sqrt(7.0**2 + 6.0**2)
    assert fit.chi_squared == pytest.approx((penalty_distance / 0.75) ** 2, 1.0e-4)

    class FitIgnore(al.FitPositionsImagePairRepeat):
        unmatched_model_policy = "ignore"

    fit_ignore = FitIgnore(
        name="point_0",
        data=data,
        noise_map=noise_map,
        tracer=tracer,
        solver=solver,
    )

    assert fit_ignore.chi_squared == pytest.approx(0.0, abs=1.0e-8)


def test__over_prediction__demagnified_central_image_exempt_under_magnification_filter():
    # An isothermal lens demagnifies positions near its centre far below the 0.1 threshold:
    # under the default magnification_filter policy the extra central image is exempt, while
    # the explicit "penalize" policy charges for it.
    lens = al.Galaxy(
        redshift=0.5,
        mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.0),
    )
    point = al.ps.Point(centre=(0.0, 0.0))
    source = al.Galaxy(redshift=1.0, point_0=point)
    tracer = al.Tracer(galaxies=[lens, source])

    data = al.Grid2DIrregular([(0.0, 1.05), (0.0, -0.95)])
    noise_map = al.ArrayIrregular([0.5, 0.5])
    model_data = al.Grid2DIrregular([(0.0, 1.05), (0.0, -0.95), (0.0, 0.01)])

    solver = al.m.MockPointSolver(model_positions=model_data)

    fit = al.FitPositionsImagePairRepeat(
        name="point_0",
        data=data,
        noise_map=noise_map,
        tracer=tracer,
        solver=solver,
    )

    assert int(fit.n_unmatched_model_positions) == 0
    assert fit.chi_squared == pytest.approx(0.0, abs=1.0e-8)

    class FitPenalize(al.FitPositionsImagePairRepeat):
        unmatched_model_policy = "penalize"

    fit_penalize = FitPenalize(
        name="point_0",
        data=data,
        noise_map=noise_map,
        tracer=tracer,
        solver=solver,
    )

    assert int(fit_penalize.n_unmatched_model_positions) == 1
    assert fit_penalize.chi_squared > 1.0


def test__under_prediction__no_model_images_hits_finite_floor():
    point = al.ps.Point(centre=(0.1, 0.1))
    galaxy = al.Galaxy(redshift=1.0, point_0=point)
    tracer = al.Tracer(galaxies=[al.Galaxy(redshift=0.5), galaxy])

    data = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0)])
    noise_map = al.ArrayIrregular([0.5, 1.0])
    model_data = al.Grid2DIrregular(np.zeros(shape=(0, 2)))

    solver = al.m.MockPointSolver(model_positions=model_data)

    fit = al.FitPositionsImagePairRepeat(
        name="point_0",
        data=data,
        noise_map=noise_map,
        tracer=tracer,
        solver=solver,
    )

    assert fit.residual_map.in_list == [1.0e4, 1.0e4]
    assert np.isfinite(float(fit.log_likelihood))
