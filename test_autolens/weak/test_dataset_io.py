import numpy as np
import pytest

import autolens as al

from autolens.weak.fit import FitWeak


def _sis_tracer(einstein_radius=1.6, centre=(0.0, 0.0)):
    lens = al.Galaxy(
        redshift=0.5,
        mass=al.mp.IsothermalSph(centre=centre, einstein_radius=einstein_radius),
    )
    return al.Tracer(galaxies=[lens, al.Galaxy(redshift=1.0)])


def _dataset(**kwargs):
    return al.SimulatorShearYX(
        noise_sigma=kwargs.pop("noise_sigma", 0.3), seed=1, **kwargs
    ).via_tracer_random_positions_from(
        tracer=_sis_tracer(), n_galaxies=20, grid_extent=3.0, name="io_test"
    )


def test__from_arrays__noise_or_weights_exclusively():
    positions = [(1.0, 0.0), (0.0, 1.0)]

    dataset = al.WeakDataset.from_arrays(
        positions=positions,
        gamma_1=[0.1, -0.1],
        gamma_2=[0.05, 0.02],
        weights=[25.0, 100.0],
    )

    # sigma = weights**-0.5
    np.testing.assert_allclose(np.asarray(dataset.noise_map), [0.2, 0.1])
    assert dataset.is_reduced is True  # loader default: real catalogues are reduced

    # [gamma_2, gamma_1] storage convention round-trips.
    np.testing.assert_allclose(np.asarray(dataset.shear_yx)[:, 1], [0.1, -0.1])
    np.testing.assert_allclose(np.asarray(dataset.shear_yx)[:, 0], [0.05, 0.02])

    with pytest.raises(ValueError):
        al.WeakDataset.from_arrays(
            positions=positions, gamma_1=[0.1, -0.1], gamma_2=[0.05, 0.02]
        )
    with pytest.raises(ValueError):
        al.WeakDataset.from_arrays(
            positions=positions,
            gamma_1=[0.1, -0.1],
            gamma_2=[0.05, 0.02],
            noise_map=[0.3, 0.3],
            weights=[25.0, 100.0],
        )


def test__csv_round_trip(tmp_path):
    dataset = _dataset()
    dataset.redshifts = None

    file_path = tmp_path / "catalogue.csv"
    dataset.to_csv(file_path)

    loaded = al.WeakDataset.from_csv(file_path, is_reduced=False)

    assert loaded.name == "io_test"
    assert loaded.is_reduced is False
    np.testing.assert_allclose(np.asarray(loaded.shear_yx), np.asarray(dataset.shear_yx))
    np.testing.assert_allclose(np.asarray(loaded.positions), np.asarray(dataset.positions))
    np.testing.assert_allclose(np.asarray(loaded.noise_map), np.asarray(dataset.noise_map))
    assert loaded.redshifts is None


def test__csv_round_trip__with_redshifts(tmp_path):
    dataset = al.WeakDataset.from_arrays(
        positions=[(1.0, 0.0), (0.0, 1.0)],
        gamma_1=[0.1, -0.1],
        gamma_2=[0.05, 0.02],
        noise_map=[0.3, 0.25],
        redshifts=[1.2, 0.9],
        name="with_z",
    )

    file_path = tmp_path / "catalogue.csv"
    dataset.to_csv(file_path)
    loaded = al.WeakDataset.from_csv(file_path)

    np.testing.assert_allclose(np.asarray(loaded.redshifts), [1.2, 0.9])
    assert loaded.is_reduced is True


def test__from_fits__column_mapping_and_weights(tmp_path):
    from astropy.io import fits as astropy_fits
    from astropy.table import Table

    table = Table(
        {
            "y_arcsec": [1.0, -1.0, 0.5],
            "x_arcsec": [0.0, 1.0, -0.5],
            "e1": [0.1, -0.05, 0.02],
            "e2": [0.03, 0.07, -0.01],
            "weight": [25.0, 100.0, 4.0],
            "z_source": [1.1, 0.8, 1.4],
        }
    )
    file_path = tmp_path / "catalogue.fits"
    astropy_fits.BinTableHDU(table).writeto(file_path)

    dataset = al.WeakDataset.from_fits(
        file_path=file_path,
        y_col="y_arcsec",
        x_col="x_arcsec",
        gamma_1_col="e1",
        gamma_2_col="e2",
        weight_col="weight",
        redshift_col="z_source",
        name="fits_test",
    )

    assert dataset.n_galaxies == 3
    assert dataset.is_reduced is True
    np.testing.assert_allclose(np.asarray(dataset.noise_map), [0.2, 0.1, 0.5])
    np.testing.assert_allclose(np.asarray(dataset.shear_yx)[:, 1], [0.1, -0.05, 0.02])
    np.testing.assert_allclose(np.asarray(dataset.redshifts), [1.1, 0.8, 1.4])


def test__json_round_trip__preserves_new_fields(tmp_path):
    dataset = al.WeakDataset.from_arrays(
        positions=[(1.0, 0.0), (0.0, 1.0)],
        gamma_1=[0.1, -0.1],
        gamma_2=[0.05, 0.02],
        noise_map=[0.3, 0.25],
        redshifts=[1.2, 0.9],
        is_reduced=True,
        name="json_test",
    )

    file_path = tmp_path / "dataset.json"
    al.output_to_json(obj=dataset, file_path=file_path)
    loaded = al.from_json(file_path=file_path)

    assert loaded.is_reduced is True
    np.testing.assert_allclose(np.asarray(loaded.redshifts), [1.2, 0.9])


def test__reduced_shear__round_trip_and_differs_from_shear():
    """
    A reduced-shear simulation fitted by its truth tracer must round-trip to zero residuals
    (FitWeak computes the model reduced shear when the dataset declares is_reduced), and the
    reduced/plain quantities must differ where the convergence is non-negligible.
    """
    tracer = _sis_tracer()

    dataset_reduced = al.SimulatorShearYX(
        noise_sigma=0.0, seed=2, reduced=True
    ).via_tracer_random_positions_from(tracer=tracer, n_galaxies=50, grid_extent=3.0)

    assert dataset_reduced.is_reduced is True

    fit = FitWeak(dataset=dataset_reduced, tracer=tracer)
    np.testing.assert_allclose(fit.residual_map, 0.0, atol=1e-10)

    dataset_plain = al.SimulatorShearYX(
        noise_sigma=0.0, seed=2, reduced=False
    ).via_tracer_random_positions_from(tracer=tracer, n_galaxies=50, grid_extent=3.0)

    # For an SIS, |g| = |gamma| / (1 - kappa) with kappa = theta_E / 2r > 0, so the
    # reduced values must be strictly larger in magnitude at every galaxy.
    reduced_mag = np.linalg.norm(np.asarray(dataset_reduced.shear_yx), axis=1)
    plain_mag = np.linalg.norm(np.asarray(dataset_plain.shear_yx), axis=1)
    assert (reduced_mag > plain_mag).all()


def test__reduced_mismatch__truth_tracer_no_longer_perfect():
    """A plain-shear dataset fitted as if reduced (or vice versa) must NOT round-trip — guards
    against the flag being ignored somewhere in the chain."""
    tracer = _sis_tracer()

    dataset_plain = al.SimulatorShearYX(
        noise_sigma=0.0, seed=3, reduced=False
    ).via_tracer_random_positions_from(tracer=tracer, n_galaxies=30, grid_extent=3.0)

    dataset_plain.is_reduced = True  # deliberately mislabel

    fit = FitWeak(dataset=dataset_plain, tracer=tracer)
    assert np.abs(fit.residual_map).max() > 1e-3
