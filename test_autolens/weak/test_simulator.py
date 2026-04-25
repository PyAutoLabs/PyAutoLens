import numpy as np
import pytest

import autoarray as aa
import autogalaxy as ag
import autolens as al

from autogalaxy.operate.lens_calc import LensCalc
from autogalaxy.util.shear_field import ShearYX2DIrregular


def _isothermal_tracer(einstein_radius=1.6, ell_comps=(0.0, 0.05)):
    lens = al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0),
            ell_comps=ell_comps,
            einstein_radius=einstein_radius,
        ),
    )
    source = al.Galaxy(redshift=1.0)
    return al.Tracer(galaxies=[lens, source])


def test__simulator_shear_yx__zero_noise_matches_tracer_shear_via_hessian():
    """
    With ``noise_sigma=0`` the simulator must reproduce the tracer's Hessian-derived shear exactly — the
    simulator only adds noise on top of that, never modifies the underlying shear computation. The simulator
    uses ``LensCalc.from_tracer`` when the input duck-types as a ``Tracer``, so that's what we compare to.
    """
    tracer = _isothermal_tracer()
    grid = aa.Grid2DIrregular(values=[(0.7, 0.5), (1.0, 1.0), (-0.3, 0.6)])

    simulator = al.SimulatorShearYX(noise_sigma=0.0, seed=0)
    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid, name="sim")

    expected = LensCalc.from_tracer(tracer).shear_yx_2d_via_hessian_from(grid=grid)

    assert isinstance(dataset, al.WeakDataset)
    assert dataset.name == "sim"
    assert dataset.n_galaxies == 3
    np.testing.assert_allclose(
        np.asarray(dataset.shear_yx), np.asarray(expected), rtol=1e-6, atol=1e-9
    )
    assert list(dataset.noise_map) == pytest.approx([0.0, 0.0, 0.0])


def test__simulator_shear_yx__noise_changes_values_but_preserves_shape_and_grid():
    tracer = _isothermal_tracer()
    grid = aa.Grid2DIrregular(values=[(0.7, 0.5), (1.0, 1.0)])

    truth = LensCalc.from_tracer(tracer).shear_yx_2d_via_hessian_from(grid=grid)
    simulator = al.SimulatorShearYX(noise_sigma=0.3, seed=42)
    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

    assert isinstance(dataset.shear_yx, ShearYX2DIrregular)
    assert np.asarray(dataset.shear_yx).shape == (2, 2)
    assert dataset.positions is grid
    assert np.any(np.asarray(dataset.shear_yx) != np.asarray(truth))
    assert list(dataset.noise_map) == pytest.approx([0.3, 0.3])


def test__simulator_shear_yx__seed_makes_runs_reproducible():
    tracer = _isothermal_tracer()
    grid = aa.Grid2DIrregular(values=[(0.7, 0.5), (1.0, 1.0), (-0.3, 0.6)])

    a = al.SimulatorShearYX(noise_sigma=0.3, seed=123).via_tracer_from(
        tracer=tracer, grid=grid
    )
    b = al.SimulatorShearYX(noise_sigma=0.3, seed=123).via_tracer_from(
        tracer=tracer, grid=grid
    )

    np.testing.assert_allclose(
        np.asarray(a.shear_yx), np.asarray(b.shear_yx), rtol=1e-12, atol=1e-12
    )


def test__simulator_shear_yx__random_positions_within_extent_and_correct_count():
    tracer = _isothermal_tracer()
    simulator = al.SimulatorShearYX(noise_sigma=0.0, seed=7)

    dataset = simulator.via_tracer_random_positions_from(
        tracer=tracer, n_galaxies=20, grid_extent=2.0
    )

    assert dataset.n_galaxies == 20
    positions = np.asarray(dataset.positions)
    assert positions.shape == (20, 2)
    assert positions.min() >= -2.0
    assert positions.max() <= 2.0


def test__simulator_shear_yx__integration_isothermal_shear_magnitude_is_nontrivial():
    """
    Integration check: an isothermal lens at the origin produces a shear field whose magnitude near the
    Einstein radius is of order ~kappa(theta_E) — i.e. clearly non-zero. This catches gross regressions
    where the simulator silently returns zero or constant shear.
    """
    einstein_radius = 1.6
    tracer = _isothermal_tracer(einstein_radius=einstein_radius, ell_comps=(0.0, 0.0))

    # Galaxies sitting on a ring at the Einstein radius
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    ring = np.stack(
        [einstein_radius * np.sin(angles), einstein_radius * np.cos(angles)], axis=1
    )
    grid = aa.Grid2DIrregular(values=ring)

    simulator = al.SimulatorShearYX(noise_sigma=0.0, seed=0)
    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

    magnitudes = np.sqrt(np.sum(np.asarray(dataset.shear_yx) ** 2, axis=1))

    assert np.all(np.isfinite(magnitudes))
    # For a singular isothermal sphere, |gamma| ~ kappa = 0.5 * theta_E / theta = 0.5 at theta_E.
    assert magnitudes.mean() == pytest.approx(0.5, rel=0.2)
