import numpy as np

import autoarray as aa
import autolens as al

from autolens.weak.fit import FitWeak


def _tracer(z_lens=0.5, z_source=1.0):
    lens = al.Galaxy(
        redshift=z_lens,
        mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.6),
    )
    return al.Tracer(galaxies=[lens, al.Galaxy(redshift=z_source)])


def _dataset(redshifts, is_reduced=False, positions=None):
    positions = positions or [(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)]
    grid = aa.Grid2DIrregular(values=positions)
    tracer = _tracer()
    dataset = al.SimulatorShearYX(noise_sigma=0.0, seed=1).via_tracer_from(
        tracer=tracer, grid=grid
    )
    dataset.redshifts = aa.ArrayIrregular(values=list(redshifts))
    dataset.is_reduced = is_reduced
    return dataset


def test__scale_factors__unity_at_ref_zero_at_lens_and_cosmology_between():
    dataset = _dataset(redshifts=[1.0, 0.5, 0.75])
    tracer = _tracer()

    fit = FitWeak(dataset=dataset, tracer=tracer)
    factors = fit._redshift_scale_factors

    assert factors[0] == 1.0  # at the source plane
    assert factors[1] == 0.0  # at the lens plane: unlensed
    expected_mid = tracer.cosmology.scaling_factor_between_redshifts_from(
        redshift_0=0.5, redshift_1=0.75, redshift_final=1.0
    )
    np.testing.assert_allclose(factors[2], expected_mid)
    assert 0.0 < factors[2] < 1.0


def test__model_shear__scales_per_galaxy():
    """Three galaxies at the SAME position, different redshifts: the model shear must be the
    single-plane shear multiplied by each galaxy's beta ratio."""
    dataset = _dataset(redshifts=[1.0, 0.5, 0.75])
    tracer = _tracer()

    fit_scaled = FitWeak(dataset=dataset, tracer=tracer)

    dataset_plain = _dataset(redshifts=[1.0, 0.5, 0.75])
    dataset_plain.redshifts = None
    fit_plain = FitWeak(dataset=dataset_plain, tracer=tracer)

    plain = np.asarray(fit_plain.model_shear)
    scaled = np.asarray(fit_scaled.model_shear)
    factors = fit_scaled._redshift_scale_factors

    np.testing.assert_allclose(scaled, factors[:, None] * plain, atol=1e-12)
    np.testing.assert_allclose(scaled[1], 0.0, atol=1e-12)  # z = z_lens galaxy


def test__no_redshifts__behaviour_unchanged():
    """A dataset without redshifts must reproduce the pre-scaling likelihood exactly."""
    grid = aa.Grid2DIrregular(values=[(0.7, 0.5), (1.0, 1.0), (-0.3, 0.6)])
    tracer = _tracer()
    dataset = al.SimulatorShearYX(noise_sigma=0.3, seed=2).via_tracer_from(
        tracer=tracer, grid=grid
    )

    fit = FitWeak(dataset=dataset, tracer=tracer)
    assert fit._redshift_scale_factors is None

    # All galaxies AT the reference plane must also match the no-redshift fit exactly.
    dataset.redshifts = aa.ArrayIrregular(values=[1.0, 1.0, 1.0])
    fit_ref = FitWeak(dataset=dataset, tracer=tracer)
    np.testing.assert_allclose(fit_ref.log_likelihood, fit.log_likelihood, rtol=1e-12)


def test__reduced_and_scaled__g_uses_scaled_kappa():
    """For a reduced dataset with redshifts, g_i = s_i*gamma / (1 - s_i*kappa) — both the shear
    and the convergence carry the efficiency factor."""
    from autogalaxy.operate.lens_calc import LensCalc

    dataset = _dataset(redshifts=[0.75, 1.0], is_reduced=True, positions=[(1.0, 1.0), (1.0, 1.0)])
    tracer = _tracer()

    fit = FitWeak(dataset=dataset, tracer=tracer)

    lens_calc = LensCalc.from_tracer(tracer)
    gamma = np.asarray(lens_calc.shear_yx_2d_via_hessian_from(grid=dataset.positions))
    kappa = np.asarray(lens_calc.convergence_2d_via_hessian_from(grid=dataset.positions))
    s = fit._redshift_scale_factors

    expected = (s[:, None] * gamma) / (1.0 - s * kappa)[:, None]
    np.testing.assert_allclose(np.asarray(fit.model_shear), expected, atol=1e-12)
