import numpy as np

import autoarray as aa
import autolens as al

from autolens.weak.plot.shear_profile_plots import (
    shear_profile_from,
    shear_tangential_cross_from,
)


def _sis_dataset(n_positions=400, noise_sigma=0.0, einstein_radius=1.6, seed=2):
    lens = al.Galaxy(
        redshift=0.5,
        mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=einstein_radius),
    )
    source = al.Galaxy(redshift=1.0)
    tracer = al.Tracer(galaxies=[lens, source])
    simulator = al.SimulatorShearYX(noise_sigma=noise_sigma, seed=seed)
    return simulator.via_tracer_random_positions_from(
        tracer=tracer, n_galaxies=n_positions, grid_extent=3.0
    )


def test__tangential_shear__sis_is_pure_tangential():
    """
    An SIS shear field is purely tangential about the lens centre, so the tangential
    component must equal the shear magnitude and the cross component must vanish —
    this validates the sign and angle conventions end to end against the simulator.
    """
    dataset = _sis_dataset()

    gamma_t, gamma_x, radii = shear_tangential_cross_from(
        shear_yx=dataset.shear_yx, centre=(0.0, 0.0)
    )

    np.testing.assert_allclose(
        gamma_t, np.asarray(dataset.shear_yx.ellipticities), atol=1e-6
    )
    np.testing.assert_allclose(gamma_x, 0.0, atol=1e-6)
    assert (radii > 0.0).all()


def test__tangential_profile__matches_sis_analytic():
    """Binned gamma_t must track the SIS analytic profile theta_E / 2r, noise-free."""
    einstein_radius = 1.6
    dataset = _sis_dataset(einstein_radius=einstein_radius)

    bin_radii, gamma_t, _, gamma_x, _ = shear_profile_from(
        shear_yx=dataset.shear_yx, centre=(0.0, 0.0), bins=8
    )

    valid = ~np.isnan(gamma_t)
    assert valid.sum() >= 6

    analytic = einstein_radius / (2.0 * bin_radii[valid])

    # Binned means of 1/r inside finite bins deviate from the bin-centre value;
    # 15% tolerance comfortably covers that discretisation while catching sign or
    # convention errors (which are order-unity).
    np.testing.assert_allclose(gamma_t[valid], analytic, rtol=0.15)
    np.testing.assert_allclose(gamma_x[valid], 0.0, atol=1e-6)


def test__profile__off_centre_recentres():
    """Recomputing about the true (offset) lens centre must stay pure-tangential."""
    lens = al.Galaxy(
        redshift=0.5,
        mass=al.mp.IsothermalSph(centre=(0.5, -0.3), einstein_radius=1.0),
    )
    tracer = al.Tracer(galaxies=[lens, al.Galaxy(redshift=1.0)])
    dataset = al.SimulatorShearYX(noise_sigma=0.0, seed=3).via_tracer_random_positions_from(
        tracer=tracer, n_galaxies=200, grid_extent=3.0
    )

    _, gamma_x_true_centre, _ = shear_tangential_cross_from(
        shear_yx=dataset.shear_yx, centre=(0.5, -0.3)
    )
    _, gamma_x_wrong_centre, _ = shear_tangential_cross_from(
        shear_yx=dataset.shear_yx, centre=(0.0, 0.0)
    )

    # One random galaxy lands ~on the centre, where the hessian finite-difference
    # error grows like 1/r — hence the looser tolerance than the on-centre test.
    np.testing.assert_allclose(gamma_x_true_centre, 0.0, atol=1e-5)
    assert np.abs(gamma_x_wrong_centre).max() > 1e-3
