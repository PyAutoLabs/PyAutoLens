import numpy as np

import autolens as al

from autolens.weak.plot.convergence_plots import convergence_via_kaiser_squires_from


def _sis_dataset(n_galaxies=2000, einstein_radius=1.6, seed=4):
    lens = al.Galaxy(
        redshift=0.5,
        mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=einstein_radius),
    )
    tracer = al.Tracer(galaxies=[lens, al.Galaxy(redshift=1.0)])
    return al.SimulatorShearYX(
        noise_sigma=0.0, seed=seed
    ).via_tracer_random_positions_from(
        tracer=tracer, n_galaxies=n_galaxies, grid_extent=3.0
    )


def test__kaiser_squires__peak_at_lens_centre_and_weak_b_mode():
    dataset = _sis_dataset()

    e_mode, b_mode = convergence_via_kaiser_squires_from(
        shear_yx=dataset.shear_yx,
        shape_native=(32, 32),
        smoothing_sigma_pixels=1.0,
    )

    e = np.asarray(e_mode.native)
    b = np.asarray(b_mode.native)

    # The reconstruction must peak at the lens centre — the central pixels of the map.
    peak_row, peak_col = np.unravel_index(np.argmax(e), e.shape)
    centre = (e.shape[0] - 1) / 2.0
    assert abs(peak_row - centre) <= 2.0
    assert abs(peak_col - centre) <= 2.0

    # E-mode signal dominates the B-mode systematics map for a pure lensing field.
    inner = np.s_[8:24, 8:24]  # interior region, away from FFT edge artefacts
    assert np.abs(e[inner]).max() > 5.0 * np.abs(b[inner]).max()


def test__kaiser_squires__map_shapes_and_zero_mean():
    dataset = _sis_dataset(n_galaxies=500)

    e_mode, b_mode = convergence_via_kaiser_squires_from(
        shear_yx=dataset.shear_yx, shape_native=(20, 24), smoothing_sigma_pixels=0.0
    )

    assert e_mode.shape_native == (20, 24)
    assert b_mode.shape_native == (20, 24)

    # The k = 0 mode is zeroed (mass-sheet degeneracy), so maps are zero-mean.
    assert abs(np.asarray(e_mode.native).mean()) < 1e-12


def test__kaiser_squires__elliptical_lens_orientation_parity():
    """
    The SIS tests are rotation-invariant, so they cannot catch an axis-parity (mirror)
    error in the inversion. An elliptical lens breaks that degeneracy: the KS map's
    major axis must align with the lens mass's major axis (45 deg), not its mirror.
    """
    lens = al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0),
            einstein_radius=1.6,
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.5, angle=45.0),
        ),
    )
    tracer = al.Tracer(galaxies=[lens, al.Galaxy(redshift=1.0)])
    dataset = al.SimulatorShearYX(
        noise_sigma=0.0, seed=1
    ).via_tracer_random_positions_from(tracer=tracer, n_galaxies=5000, grid_extent=3.0)

    e_mode, _ = convergence_via_kaiser_squires_from(
        shear_yx=dataset.shear_yx, shape_native=(64, 64), smoothing_sigma_pixels=1.0
    )
    e = np.asarray(e_mode.native)

    ys, xs = np.mgrid[0:64, 0:64]
    y_coord = 3.0 - (ys + 0.5) * (6.0 / 64)
    x_coord = -3.0 + (xs + 0.5) * (6.0 / 64)
    weights = np.clip(np.where(e > np.percentile(e, 95), e, 0.0), 0.0, None)
    x_bar = (weights * x_coord).sum() / weights.sum()
    y_bar = (weights * y_coord).sum() / weights.sum()
    q_xx = (weights * (x_coord - x_bar) ** 2.0).sum() / weights.sum()
    q_yy = (weights * (y_coord - y_bar) ** 2.0).sum() / weights.sum()
    q_xy = (weights * (x_coord - x_bar) * (y_coord - y_bar)).sum() / weights.sum()

    angle = 0.5 * np.degrees(np.arctan2(2.0 * q_xy, q_xx - q_yy))

    assert abs(angle - 45.0) < 10.0
