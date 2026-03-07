import pytest

import autolens as al


def test__one_set_of_fluxes__residuals_likelihood_correct():
    tracer = al.m.MockTracerPoint(profile=al.ps.PointFlux(flux=2.0))

    data = al.ArrayIrregular([1.0, 2.0])
    noise_map = al.ArrayIrregular([3.0, 1.0])
    positions = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0)])

    fit = al.FitFluxes(
        name="point_0",
        data=data,
        noise_map=noise_map,
        positions=positions,
        tracer=tracer,
    )

    assert fit.data.in_list == [1.0, 2.0]
    assert fit.noise_map.in_list == [3.0, 1.0]
    assert fit.model_fluxes.in_list == [2.0, 2.0]
    assert fit.residual_map.in_list == [-1.0, -0.0]
    assert fit.normalized_residual_map.in_list == [-1.0 / 3.0, -0.0]
    assert fit.chi_squared_map.in_list == [1.0 / 9.0, 0.0]
    assert fit.chi_squared == pytest.approx(1.0 / 9.0, 1.0e-4)
    assert fit.noise_normalization == pytest.approx(5.87297, 1.0e-4)
    assert fit.log_likelihood == pytest.approx(-2.992044910633, 1.0e-4)


def test__use_real_tracer(gal_x1_mp):
    point_source = al.ps.PointFlux(centre=(0.1, 0.1), flux=2.0)
    galaxy_point_source = al.Galaxy(redshift=1.0, point_0=point_source)
    tracer = al.Tracer(galaxies=[gal_x1_mp, galaxy_point_source])

    data = al.ArrayIrregular([1.0, 2.0])
    noise_map = al.ArrayIrregular([3.0, 1.0])
    positions = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0)])

    fit = al.FitFluxes(
        name="point_0",
        data=data,
        noise_map=noise_map,
        positions=positions,
        tracer=tracer,
    )

    assert fit.model_fluxes.in_list[1] == pytest.approx(2.5, 1.0e-4)
    assert fit.log_likelihood == pytest.approx(-3.11702, 1.0e-4)
