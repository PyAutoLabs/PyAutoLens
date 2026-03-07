import pytest

import autogalaxy as ag
import autolens as al


def test__magnifications_at_positions__multi_plane_calculation(gal_x1_mp):
    g0 = al.Galaxy(redshift=0.5, mass=al.mp.IsothermalSph(einstein_radius=1.0))
    g1 = al.Galaxy(redshift=1.0, point_0=al.ps.PointFlux(flux=1.0))
    g2 = al.Galaxy(redshift=2.0, point_1=al.ps.PointFlux(flux=2.0))

    tracer = al.Tracer(galaxies=[g0, g1, g2])

    data = al.ArrayIrregular([1.0])
    noise_map = al.ArrayIrregular([3.0])
    positions = al.Grid2DIrregular([(2.0, 0.0)])

    fit_0 = al.FitFluxes(
        name="point_0",
        data=data,
        noise_map=noise_map,
        positions=positions,
        tracer=tracer,
    )

    od_0 = ag.LensCalc.from_tracer(tracer, use_multi_plane=True, plane_j=1)
    magnification_0 = abs(od_0.magnification_2d_via_hessian_from(grid=positions))

    assert fit_0.magnifications_at_positions[0] == magnification_0

    fit_1 = al.FitFluxes(
        name="point_1",
        data=data,
        noise_map=noise_map,
        positions=positions,
        tracer=tracer,
    )

    od_1 = ag.LensCalc.from_tracer(tracer, use_multi_plane=True, plane_j=2)
    magnification_1 = abs(od_1.magnification_2d_via_hessian_from(grid=positions))

    assert fit_1.magnifications_at_positions[0] == magnification_1

    assert fit_0.magnifications_at_positions[0] != pytest.approx(
        fit_1.magnifications_at_positions.array[0], 1.0e-1
    )
