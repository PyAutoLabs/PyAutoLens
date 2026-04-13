import copy
import warnings
import numpy as np
import pytest

import autolens as al


def test__model_image__without_psf_blurring__slim_values_and_log_likelihood_correct(
    masked_imaging_7x7_no_blur,
):
    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.m.MockLightProfile(image_2d_value=1.0, image_2d_first_value=2.0),
    )
    tracer = al.Tracer(galaxies=[g0])

    fit = al.FitImaging(dataset=masked_imaging_7x7_no_blur, tracer=tracer)

    assert fit.model_data.slim == pytest.approx(
        np.array([2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 1.0e-4
    )
    assert fit.log_likelihood == pytest.approx(-14.6337, 1.0e-4)


def test__model_image__with_psf_blurring__blurred_slim_values_and_log_likelihood_correct(
    masked_imaging_7x7,
):
    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.m.MockLightProfile(image_2d_value=1.0, image_2d_first_value=2.0),
    )
    tracer = al.Tracer(galaxies=[g0])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.model_data.slim == pytest.approx(
        np.array([1.33, 1.16, 1.0, 1.16, 1.0, 1.0, 1.0, 1.0, 1.0]), 1.0e-1
    )
    assert fit.log_likelihood == pytest.approx(-14.52960, 1.0e-4)


def test__figure_of_merit__two_sersic_light_profiles__no_inversion(masked_imaging_7x7):
    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0),
        disk=al.lp.Sersic(centre=(0.05, 0.05), intensity=2.0),
        mass_profile=al.mp.IsothermalSph(centre=(0.05, 0.05), einstein_radius=1.0),
    )
    g1 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0))

    tracer = al.Tracer(galaxies=[g0, g1])
    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is False
    assert fit.figure_of_merit == pytest.approx(-648.4814555620, 1.0e-4)


def test__figure_of_merit__basis_profiles_no_inversion__same_result_as_standard_profiles(
    masked_imaging_7x7,
):
    basis = al.lp_basis.Basis(
        profile_list=[
            al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0),
            al.lp.Sersic(centre=(0.05, 0.05), intensity=2.0),
        ]
    )
    g0 = al.Galaxy(
        redshift=0.5,
        bulge=basis,
        mass_profile=al.mp.IsothermalSph(centre=(0.05, 0.05), einstein_radius=1.0),
    )
    g1 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0))

    tracer = al.Tracer(galaxies=[g0, g1])
    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is False
    assert fit.figure_of_merit == pytest.approx(-648.4814555620, 1.0e-4)


def test__figure_of_merit__pixelization_inversion__expected_log_evidence(
    masked_imaging_7x7,
):
    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )
    g0 = al.Galaxy(redshift=0.5, pixelization=pixelization)

    tracer = al.Tracer(galaxies=[al.Galaxy(redshift=0.5), g0])
    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-22.9122020414322, 1.0e-4)


def test__figure_of_merit__light_profile_with_pixelization_inversion__expected_log_evidence(
    masked_imaging_7x7,
):
    galaxy_light = al.Galaxy(
        redshift=0.5, bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0)
    )
    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )
    galaxy_pix = al.Galaxy(redshift=1.0, pixelization=pixelization)

    tracer = al.Tracer(galaxies=[galaxy_light, galaxy_pix])
    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-29.2344033729098, 1.0e-4)


def test__figure_of_merit__linear_light_profiles__inversion__expected_log_evidence(
    masked_imaging_7x7,
):
    g0_linear = al.Galaxy(
        redshift=0.5,
        bulge=al.lp_linear.Sersic(centre=(0.05, 0.05), sersic_index=1.0),
        disk=al.lp_linear.Sersic(centre=(0.05, 0.05), sersic_index=4.0),
        mass_profile=al.mp.IsothermalSph(centre=(0.05, 0.05), einstein_radius=1.0),
    )
    g1 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0))

    tracer = al.Tracer(galaxies=[g0_linear, g1])
    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-16.9731347648, 1.0e-4)


def test__figure_of_merit__linear_basis_profiles__inversion__expected_log_evidence(
    masked_imaging_7x7,
):
    basis = al.lp_basis.Basis(
        profile_list=[
            al.lp_linear.Sersic(centre=(0.05, 0.05), sersic_index=1.0),
            al.lp_linear.Sersic(centre=(0.05, 0.05), sersic_index=4.0),
        ]
    )
    g0_linear = al.Galaxy(
        redshift=0.5,
        bulge=basis,
        mass_profile=al.mp.IsothermalSph(centre=(0.05, 0.05), einstein_radius=1.0),
    )
    g1 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0))

    tracer = al.Tracer(galaxies=[g0_linear, g1])
    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-16.97313476, 1.0e-4)


def test__figure_of_merit__linear_basis_with_regularization__inversion__expected_log_evidence(
    masked_imaging_7x7,
):
    basis = al.lp_basis.Basis(
        profile_list=[
            al.lp_linear.Sersic(centre=(0.05, 0.05), sersic_index=1.0),
            al.lp_linear.Sersic(centre=(0.05, 0.05), sersic_index=4.0),
        ],
        regularization=al.reg.Constant(coefficient=1.0),
    )
    g0_basis = al.Galaxy(
        redshift=0.5,
        bulge=basis,
        mass_profile=al.mp.IsothermalSph(einstein_radius=1.0),
    )
    g1 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0))

    tracer = al.Tracer(galaxies=[g0_basis, g1])
    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-51.0835080747, 1.0e-4)


def test__figure_of_merit__linear_profiles_with_pixelization__inversion__expected_log_evidence(
    masked_imaging_7x7,
):
    basis = al.lp_basis.Basis(
        profile_list=[
            al.lp_linear.Sersic(centre=(0.05, 0.05), sersic_index=1.0),
            al.lp_linear.Sersic(centre=(0.05, 0.05), sersic_index=4.0),
        ]
    )
    g0_linear = al.Galaxy(
        redshift=0.5,
        bulge=basis,
        mass_profile=al.mp.IsothermalSph(centre=(0.05, 0.05), einstein_radius=1.0),
    )
    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )
    galaxy_pix = al.Galaxy(redshift=1.0, pixelization=pixelization)

    tracer = al.Tracer(galaxies=[g0_linear, galaxy_pix])
    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-22.801816971, 1.0e-4)


def test__figure_of_merit__operated_light_profiles__no_inversion__expected_log_likelihood(
    masked_imaging_7x7,
):
    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0),
        mass_profile=al.mp.IsothermalSph(centre=(0.05, 0.05), einstein_radius=1.0),
    )
    g1 = al.Galaxy(
        redshift=1.0,
        bulge=al.lp_operated.Sersic(centre=(0.05, 0.05), intensity=1.0),
    )

    tracer = al.Tracer(galaxies=[g0, g1])
    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is False
    assert fit.figure_of_merit == pytest.approx(-745.25961066, 1.0e-4)


def test__figure_of_merit__linear_operated_light_profiles__inversion__expected_log_evidence(
    masked_imaging_7x7,
):
    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.lp_linear_operated.Sersic(centre=(0.05, 0.05), sersic_index=1.0),
        mass_profile=al.mp.IsothermalSph(centre=(0.05, 0.05), einstein_radius=1.0),
    )
    g1 = al.Galaxy(
        redshift=1.0,
        bulge=al.lp_linear_operated.Sersic(centre=(0.05, 0.05), sersic_index=4.0),
    )

    tracer = al.Tracer(galaxies=[g0, g1])
    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-14.933306470, 1.0e-4)


def test__figure_of_merit__covariance_noise_matrix__expected_log_likelihood(
    masked_imaging_covariance_7x7,
):
    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0),
        disk=al.lp.Sersic(centre=(0.05, 0.05), intensity=2.0),
        mass_profile=al.mp.IsothermalSph(centre=(0.05, 0.05), einstein_radius=1.0),
    )
    g1 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0))

    tracer = al.Tracer(galaxies=[g0, g1])
    fit = al.FitImaging(dataset=masked_imaging_covariance_7x7, tracer=tracer)

    assert fit.perform_inversion is False
    assert fit.figure_of_merit == pytest.approx(-775.003133428, 1.0e-4)


# --- sub_2 oversampling variants ---


def test__figure_of_merit__sub_2_oversampling__two_sersic_profiles__no_inversion(
    image_7x7, psf_3x3, noise_map_7x7, mask_2d_7x7
):
    dataset = al.Imaging(
        data=image_7x7,
        psf=psf_3x3,
        noise_map=noise_map_7x7,
        over_sample_size_lp=2,
    )
    masked_imaging_7x7 = dataset.apply_mask(mask=mask_2d_7x7)

    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(intensity=1.0),
        disk=al.lp.Sersic(intensity=2.0),
        mass_profile=al.mp.IsothermalSph(einstein_radius=1.0),
    )
    g1 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(intensity=1.0))

    tracer = al.Tracer(galaxies=[g0, g1])
    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is False
    assert fit.figure_of_merit == pytest.approx(-41.60614104506, 1.0e-4)


def test__figure_of_merit__sub_2_oversampling__basis_profiles__no_inversion(
    image_7x7, psf_3x3, noise_map_7x7, mask_2d_7x7
):
    dataset = al.Imaging(
        data=image_7x7,
        psf=psf_3x3,
        noise_map=noise_map_7x7,
        over_sample_size_lp=2,
    )
    masked_imaging_7x7 = dataset.apply_mask(mask=mask_2d_7x7)

    basis = al.lp_basis.Basis(
        profile_list=[
            al.lp.Sersic(intensity=1.0),
            al.lp.Sersic(intensity=2.0),
        ]
    )
    g0 = al.Galaxy(
        redshift=0.5,
        bulge=basis,
        mass_profile=al.mp.IsothermalSph(einstein_radius=1.0),
    )
    g1 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(intensity=1.0))

    tracer = al.Tracer(galaxies=[g0, g1])
    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is False
    assert fit.figure_of_merit == pytest.approx(-41.60614104506277, 1.0e-4)


def test__figure_of_merit__sub_2_oversampling__pixelization_inversion__expected_log_evidence(
    image_7x7, psf_3x3, noise_map_7x7, mask_2d_7x7
):
    dataset = al.Imaging(
        data=image_7x7,
        psf=psf_3x3,
        noise_map=noise_map_7x7,
        over_sample_size_lp=2,
    )
    masked_imaging_7x7 = dataset.apply_mask(mask=mask_2d_7x7)

    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )
    g0 = al.Galaxy(redshift=0.5, pixelization=pixelization)

    tracer = al.Tracer(galaxies=[al.Galaxy(redshift=0.5), g0])
    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-22.912202041432, 1.0e-4)


def test__figure_of_merit__sub_2_oversampling__light_with_pixelization__expected_log_evidence(
    image_7x7, psf_3x3, noise_map_7x7, mask_2d_7x7
):
    dataset = al.Imaging(
        data=image_7x7,
        psf=psf_3x3,
        noise_map=noise_map_7x7,
        over_sample_size_lp=2,
    )
    masked_imaging_7x7 = dataset.apply_mask(mask=mask_2d_7x7)

    galaxy_light = al.Galaxy(redshift=0.5, bulge=al.lp.Sersic(intensity=1.0))
    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )
    galaxy_pix = al.Galaxy(redshift=1.0, pixelization=pixelization)

    tracer = al.Tracer(galaxies=[galaxy_light, galaxy_pix])
    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-23.040374194334518, 1.0e-4)


def test__figure_of_merit__sub_2_oversampling__linear_light_profiles__inversion__expected_log_evidence(
    image_7x7, psf_3x3, noise_map_7x7, mask_2d_7x7
):
    dataset = al.Imaging(
        data=image_7x7,
        psf=psf_3x3,
        noise_map=noise_map_7x7,
        over_sample_size_lp=2,
    )
    masked_imaging_7x7 = dataset.apply_mask(mask=mask_2d_7x7)

    g0_linear = al.Galaxy(
        redshift=0.5,
        bulge=al.lp_linear.Sersic(sersic_index=1.0),
        disk=al.lp_linear.Sersic(sersic_index=4.0),
        mass_profile=al.mp.IsothermalSph(einstein_radius=1.0),
    )
    g1 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(intensity=1.0))

    tracer = al.Tracer(galaxies=[g0_linear, g1])
    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-18.47282483, 1.0e-4)


def test__figure_of_merit__sub_2_oversampling__linear_basis_profiles__inversion__expected_log_evidence(
    image_7x7, psf_3x3, noise_map_7x7, mask_2d_7x7
):
    dataset = al.Imaging(
        data=image_7x7,
        psf=psf_3x3,
        noise_map=noise_map_7x7,
        over_sample_size_lp=2,
    )
    masked_imaging_7x7 = dataset.apply_mask(mask=mask_2d_7x7)

    basis = al.lp_basis.Basis(
        profile_list=[
            al.lp_linear.Sersic(sersic_index=1.0),
            al.lp_linear.Sersic(sersic_index=4.0),
        ]
    )
    g0_linear = al.Galaxy(
        redshift=0.5,
        bulge=basis,
        mass_profile=al.mp.IsothermalSph(einstein_radius=1.0),
    )
    g1 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(intensity=1.0))

    tracer = al.Tracer(galaxies=[g0_linear, g1])
    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-18.4728248395, 1.0e-4)


def test__figure_of_merit__sub_2_oversampling__linear_basis_with_regularization__inversion(
    image_7x7, psf_3x3, noise_map_7x7, mask_2d_7x7
):
    dataset = al.Imaging(
        data=image_7x7,
        psf=psf_3x3,
        noise_map=noise_map_7x7,
        over_sample_size_lp=2,
    )
    masked_imaging_7x7 = dataset.apply_mask(mask=mask_2d_7x7)

    basis = al.lp_basis.Basis(
        profile_list=[
            al.lp_linear.Sersic(sersic_index=1.0),
            al.lp_linear.Sersic(sersic_index=4.0),
        ],
        regularization=al.reg.Constant(coefficient=1.0),
    )
    g0_basis = al.Galaxy(
        redshift=0.5,
        bulge=basis,
        mass_profile=al.mp.IsothermalSph(einstein_radius=1.0),
    )
    g1 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(intensity=1.0))

    tracer = al.Tracer(galaxies=[g0_basis, g1])
    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-27.8563999226980, 1.0e-4)


def test__figure_of_merit__sub_2_oversampling__linear_profiles_with_pixelization__inversion(
    image_7x7, psf_3x3, noise_map_7x7, mask_2d_7x7
):
    dataset = al.Imaging(
        data=image_7x7,
        psf=psf_3x3,
        noise_map=noise_map_7x7,
        over_sample_size_lp=2,
    )
    masked_imaging_7x7 = dataset.apply_mask(mask=mask_2d_7x7)

    basis = al.lp_basis.Basis(
        profile_list=[
            al.lp_linear.Sersic(sersic_index=1.0),
            al.lp_linear.Sersic(sersic_index=4.0),
        ]
    )
    g0_linear = al.Galaxy(
        redshift=0.5,
        bulge=basis,
        mass_profile=al.mp.IsothermalSph(einstein_radius=1.0),
    )
    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )
    galaxy_pix = al.Galaxy(redshift=1.0, pixelization=pixelization)

    tracer = al.Tracer(galaxies=[g0_linear, galaxy_pix])
    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-22.8051006673, 1.0e-4)


def test__figure_of_merit__sub_2_oversampling__operated_light_profiles__no_inversion(
    image_7x7, psf_3x3, noise_map_7x7, mask_2d_7x7
):
    dataset = al.Imaging(
        data=image_7x7,
        psf=psf_3x3,
        noise_map=noise_map_7x7,
        over_sample_size_lp=2,
    )
    masked_imaging_7x7 = dataset.apply_mask(mask=mask_2d_7x7)

    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(intensity=1.0),
        mass_profile=al.mp.IsothermalSph(einstein_radius=1.0),
    )
    g1 = al.Galaxy(redshift=1.0, bulge=al.lp_operated.Sersic(intensity=1.0))

    tracer = al.Tracer(galaxies=[g0, g1])
    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is False
    assert fit.figure_of_merit == pytest.approx(-51.32884812, 1.0e-4)


def test__figure_of_merit__sub_2_oversampling__linear_operated_profiles__inversion(
    image_7x7, psf_3x3, noise_map_7x7, mask_2d_7x7
):
    dataset = al.Imaging(
        data=image_7x7,
        psf=psf_3x3,
        noise_map=noise_map_7x7,
        over_sample_size_lp=2,
    )
    masked_imaging_7x7 = dataset.apply_mask(mask=mask_2d_7x7)

    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.lp_linear_operated.Sersic(sersic_index=1.0),
        mass_profile=al.mp.IsothermalSph(einstein_radius=1.0),
    )
    g1 = al.Galaxy(
        redshift=1.0, bulge=al.lp_linear_operated.Sersic(sersic_index=4.0)
    )

    tracer = al.Tracer(galaxies=[g0, g1])
    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-14.532352248, 1.0e-4)


def test__figure_of_merit__sub_2_oversampling__covariance_noise_matrix__expected_log_likelihood(
    masked_imaging_covariance_7x7,
):
    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0),
        disk=al.lp.Sersic(centre=(0.05, 0.05), intensity=2.0),
        mass_profile=al.mp.IsothermalSph(centre=(0.05, 0.05), einstein_radius=1.0),
    )
    g1 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0))

    tracer = al.Tracer(galaxies=[g0, g1])
    fit = al.FitImaging(dataset=masked_imaging_covariance_7x7, tracer=tracer)

    assert fit.perform_inversion is False
    assert fit.figure_of_merit == pytest.approx(-775.0031334280, 1.0e-4)


def test__fit__sky__background_sky_level_subtracted_from_data__expected_log_likelihood(
    masked_imaging_7x7,
):
    masked_imaging_7x7 = copy.copy(masked_imaging_7x7)
    masked_imaging_7x7.data -= 100.0

    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0),
        disk=al.lp.Sersic(centre=(0.05, 0.05), intensity=2.0),
        mass_profile=al.mp.IsothermalSph(centre=(0.05, 0.05), einstein_radius=1.0),
    )
    g1 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0))

    tracer = al.Tracer(galaxies=[g0, g1])
    fit = al.FitImaging(
        dataset=masked_imaging_7x7,
        tracer=tracer,
        dataset_model=al.DatasetModel(background_sky_level=5.0),
    )

    assert fit.figure_of_merit == pytest.approx(-18050.8847818, 1.0e-4)


def test__fit__grid_offset__shifts_model_to_match_offset_galaxy_positions__expected_log_likelihood(
    masked_imaging_7x7,
):
    # Numerical value changes slightly due to precision in grid values
    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(centre=(-1.05, -2.05), intensity=1.0),
        disk=al.lp.Sersic(centre=(-1.05, -2.05), intensity=2.0),
        mass_profile=al.mp.IsothermalSph(centre=(-1.05, -2.05), einstein_radius=1.0),
    )
    g1 = al.Galaxy(
        redshift=1.0, bulge=al.lp.Sersic(centre=(-1.05, -2.05), intensity=1.0)
    )

    tracer = al.Tracer(galaxies=[g0, g1])
    fit = al.FitImaging(
        dataset=masked_imaging_7x7,
        tracer=tracer,
        dataset_model=al.DatasetModel(grid_offset=(1.0, 2.0)),
    )

    assert fit.figure_of_merit == pytest.approx(-648.48145556, 1.0e-4)


def test__fit__grid_offset__with_linear_profiles_and_pixelization__expected_log_evidence(
    masked_imaging_7x7,
):
    g0_linear = al.Galaxy(
        redshift=0.5,
        bulge=al.lp_linear.Sersic(centre=(-1.0, -2.0), sersic_index=1.0),
        disk=al.lp_linear.Sersic(centre=(-1.0, -2.0), sersic_index=4.0),
        mass_profile=al.mp.IsothermalSph(centre=(-1.0, -2.0), einstein_radius=1.0),
    )
    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )
    galaxy_pix = al.Galaxy(redshift=1.0, pixelization=pixelization)

    tracer = al.Tracer(galaxies=[g0_linear, galaxy_pix])
    fit = al.FitImaging(
        dataset=masked_imaging_7x7,
        tracer=tracer,
        dataset_model=al.DatasetModel(grid_offset=(1.0, 2.0)),
    )

    assert fit.figure_of_merit == pytest.approx(-22.8051006, 1.0e-4)


def test__galaxy_image_dict__normal_light_profiles__images_match_individual_galaxy_images(
    masked_imaging_7x7,
):
    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0),
        mass_profile=al.mp.IsothermalSph(centre=(0.05, 0.05), einstein_radius=1.0),
    )
    g1 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0))
    g2 = al.Galaxy(redshift=1.0)

    tracer = al.Tracer(galaxies=[g0, g1, g2])
    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    image_2d_list = tracer.image_2d_list_from(grid=masked_imaging_7x7.grids.lp)

    assert fit.galaxy_image_dict[g0] == pytest.approx(image_2d_list[0].array, 1.0e-4)
    assert fit.galaxy_image_dict[g1] == pytest.approx(image_2d_list[1].array, 1.0e-4)
    assert (fit.galaxy_image_dict[g2] == np.zeros(9)).all()


def test__galaxy_image_dict__mixed_linear_and_pixelization__image_values_correct(
    masked_imaging_7x7,
):
    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0),
        mass_profile=al.mp.IsothermalSph(centre=(0.05, 0.05), einstein_radius=1.0),
    )
    g0_linear = al.Galaxy(
        redshift=0.5,
        bulge=al.lp_linear.Sersic(),
        mass_profile=al.mp.IsothermalSph(einstein_radius=1.0),
    )
    g2 = al.Galaxy(redshift=1.0)
    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )
    galaxy_pix_0 = al.Galaxy(redshift=1.0, pixelization=pixelization)
    galaxy_pix_1 = al.Galaxy(redshift=1.0, pixelization=pixelization)

    tracer = al.Tracer(galaxies=[g0, g0_linear, g2, galaxy_pix_0, galaxy_pix_1])
    masked_imaging_7x7.data[0] = 3.0
    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    image_2d_list = al.Tracer(
        galaxies=[
            al.Galaxy(
                redshift=0.5,
                bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0),
                mass_profile=al.mp.IsothermalSph(centre=(0.05, 0.05), einstein_radius=1.0),
            ),
            al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0)),
            al.Galaxy(redshift=1.0),
        ]
    ).image_2d_list_from(grid=masked_imaging_7x7.grids.lp)

    assert fit.galaxy_image_dict[g0] == pytest.approx(image_2d_list[0].array, 1.0e-4)
    assert fit.galaxy_image_dict[g0_linear][4] == pytest.approx(-27.926926, 1.0e-4)
    assert fit.galaxy_image_dict[galaxy_pix_0][4] == pytest.approx(0.94954435552, 1.0e-4)
    assert fit.galaxy_image_dict[galaxy_pix_1][4] == pytest.approx(0.94954436798, 1.0e-4)
    assert (fit.galaxy_image_dict[g2] == np.zeros(9)).all()


def test__galaxy_model_image_dict__normal_light_profiles__model_images_match_blurred_images(
    masked_imaging_7x7,
):
    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0),
        mass_profile=al.mp.IsothermalSph(centre=(0.05, 0.05), einstein_radius=1.0),
    )
    g1 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0))
    g2 = al.Galaxy(redshift=1.0)

    tracer = al.Tracer(galaxies=[g0, g1, g2])
    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    blurred_image_2d_list = tracer.blurred_image_2d_list_from(
        grid=masked_imaging_7x7.grids.lp,
        psf=masked_imaging_7x7.psf,
        blurring_grid=masked_imaging_7x7.grids.blurring,
    )

    assert fit.galaxy_model_image_dict[g0] == pytest.approx(
        blurred_image_2d_list[0].array, 1.0e-4
    )
    assert fit.galaxy_model_image_dict[g1] == pytest.approx(
        blurred_image_2d_list[1].array, 1.0e-4
    )
    assert (fit.galaxy_model_image_dict[g2] == np.zeros(9)).all()
    assert fit.model_data.native == pytest.approx(
        fit.galaxy_model_image_dict[g0].native.array
        + fit.galaxy_model_image_dict[g1].native.array,
        1.0e-4,
    )


def test__galaxy_model_image_dict__linear_light_profiles__model_images_sum_to_model_data(
    masked_imaging_7x7,
):
    g0_linear = al.Galaxy(
        redshift=0.5,
        bulge=al.lp_linear.Sersic(),
        mass_profile=al.mp.IsothermalSph(einstein_radius=1.0),
    )
    g1_linear = al.Galaxy(redshift=1.0, bulge=al.lp_linear.Sersic())
    g2 = al.Galaxy(redshift=1.0)

    tracer = al.Tracer(galaxies=[g0_linear, g1_linear, g2])
    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.galaxy_model_image_dict[g0_linear][4] == pytest.approx(
        -4.99645959e-01, 1.0e-2
    )
    assert fit.galaxy_model_image_dict[g1_linear][4] == pytest.approx(
        1.9986499980, 1.0e-2
    )
    assert (fit.galaxy_model_image_dict[g2] == np.zeros(9)).all()
    assert fit.model_data == pytest.approx(
        fit.galaxy_model_image_dict[g0_linear].array
        + fit.galaxy_model_image_dict[g1_linear].array,
        1.0e-4,
    )


def test__galaxy_model_image_dict__pixelization_only__reconstruction_image_sums_to_model_data(
    masked_imaging_7x7,
):
    g0_no_light = al.Galaxy(
        redshift=0.5, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0)
    )
    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )
    galaxy_pix_0 = al.Galaxy(redshift=1.0, pixelization=pixelization)

    tracer = al.Tracer(galaxies=[g0_no_light, galaxy_pix_0])
    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert (fit.galaxy_model_image_dict[g0_no_light].array == np.zeros(9)).all()
    assert fit.galaxy_model_image_dict[galaxy_pix_0].array[4] == pytest.approx(
        1.259965886, 1.0e-4
    )
    assert fit.model_data == pytest.approx(
        fit.galaxy_model_image_dict[galaxy_pix_0].array, 1.0e-4
    )


def test__galaxy_model_image_dict__mixed_linear_and_pixelization__all_contributions_correct(
    masked_imaging_7x7,
):
    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0),
        mass_profile=al.mp.IsothermalSph(centre=(0.05, 0.05), einstein_radius=1.0),
    )
    g0_linear = al.Galaxy(
        redshift=0.5,
        bulge=al.lp_linear.Sersic(),
        mass_profile=al.mp.IsothermalSph(einstein_radius=1.0),
    )
    g2 = al.Galaxy(redshift=1.0)
    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )
    galaxy_pix_0 = al.Galaxy(redshift=1.0, pixelization=pixelization)
    galaxy_pix_1 = al.Galaxy(redshift=1.0, pixelization=pixelization)

    tracer = al.Tracer(galaxies=[g0, g0_linear, g2, galaxy_pix_0, galaxy_pix_1])
    masked_imaging_7x7.data[0] = 3.0
    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    blurred_image_2d_list = al.Tracer(
        galaxies=[
            al.Galaxy(
                redshift=0.5,
                bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0),
                mass_profile=al.mp.IsothermalSph(centre=(0.05, 0.05), einstein_radius=1.0),
            ),
            al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0)),
            al.Galaxy(redshift=1.0),
        ]
    ).blurred_image_2d_list_from(
        grid=masked_imaging_7x7.grids.lp,
        psf=masked_imaging_7x7.psf,
        blurring_grid=masked_imaging_7x7.grids.blurring,
    )

    assert fit.galaxy_model_image_dict[g0] == pytest.approx(
        blurred_image_2d_list[0].array, 1.0e-4
    )
    assert fit.galaxy_model_image_dict[g0_linear][4] == pytest.approx(-9.31143037, 1.0e-4)
    assert fit.galaxy_model_image_dict[galaxy_pix_0][4] == pytest.approx(0.94918443, 1.0e-4)
    assert fit.galaxy_model_image_dict[galaxy_pix_1][4] == pytest.approx(0.94918442, 1.0e-4)
    assert (fit.galaxy_model_image_dict[g2] == np.zeros(9)).all()


def test__subtracted_image_of_galaxies_dict__single_plane_with_multiple_galaxies__each_subtraction_correct(
    masked_imaging_7x7,
):
    g0 = al.Galaxy(redshift=0.5, bulge=al.lp.Sersic(intensity=1.0))
    g1 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(intensity=2.0))
    g2 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(intensity=3.0))

    tracer = al.Tracer(galaxies=[g0, g1, g2])
    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    g0_image = g0.blurred_image_2d_from(
        grid=masked_imaging_7x7.grids.lp,
        blurring_grid=masked_imaging_7x7.grids.blurring,
        psf=masked_imaging_7x7.psf,
    )
    g1_image = g1.blurred_image_2d_from(
        grid=masked_imaging_7x7.grids.lp,
        blurring_grid=masked_imaging_7x7.grids.blurring,
        psf=masked_imaging_7x7.psf,
    )
    g2_image = g2.blurred_image_2d_from(
        grid=masked_imaging_7x7.grids.lp,
        blurring_grid=masked_imaging_7x7.grids.blurring,
        psf=masked_imaging_7x7.psf,
    )

    assert fit.subtracted_images_of_galaxies_dict[g0] == pytest.approx(
        masked_imaging_7x7.data.array - g1_image.array - g2_image.array, 1.0e-4
    )
    assert fit.subtracted_images_of_galaxies_dict[g1] == pytest.approx(
        masked_imaging_7x7.data.array - g0_image.array - g2_image.array, 1.0e-4
    )
    assert fit.subtracted_images_of_galaxies_dict[g2] == pytest.approx(
        masked_imaging_7x7.data.array - g0_image.array - g1_image.array, 1.0e-4
    )


def test__subtracted_image_of_galaxies_dict__three_planes_with_lensing__each_subtraction_correct(
    masked_imaging_7x7,
):
    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(intensity=1.0),
        mass_profile=al.mp.IsothermalSph(einstein_radius=1.0),
    )
    g1 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(intensity=2.0))
    g2 = al.Galaxy(redshift=2.0, bulge=al.lp.Sersic(intensity=3.0))

    tracer = al.Tracer(galaxies=[g0, g1, g2])
    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    blurred_image_2d_list = tracer.blurred_image_2d_list_from(
        grid=masked_imaging_7x7.grids.lp,
        psf=masked_imaging_7x7.psf,
        blurring_grid=masked_imaging_7x7.grids.blurring,
    )

    assert fit.subtracted_images_of_galaxies_dict[g0] == pytest.approx(
        masked_imaging_7x7.data.array
        - blurred_image_2d_list[1].array
        - blurred_image_2d_list[2].array,
        1.0e-4,
    )
    assert fit.subtracted_images_of_galaxies_dict[g1] == pytest.approx(
        masked_imaging_7x7.data.array
        - blurred_image_2d_list[0].array
        - blurred_image_2d_list[2].array,
        1.0e-4,
    )
    assert fit.subtracted_images_of_galaxies_dict[g2] == pytest.approx(
        masked_imaging_7x7.data.array
        - blurred_image_2d_list[0].array
        - blurred_image_2d_list[1].array,
        1.0e-4,
    )


def test__model_images_of_planes_list__mixed_profiles__each_plane_image_matches_galaxy_contributions(
    masked_imaging_7x7_sub_2,
):
    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(intensity=1.0),
        mass_profile=al.mp.IsothermalSph(einstein_radius=1.0),
    )
    g1_linear = al.Galaxy(redshift=0.75, bulge=al.lp_linear.Sersic())
    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )
    galaxy_pix_0 = al.Galaxy(redshift=1.0, pixelization=pixelization)
    galaxy_pix_1 = al.Galaxy(redshift=1.0, pixelization=pixelization)

    tracer = al.Tracer(galaxies=[g0, g1_linear, galaxy_pix_0, galaxy_pix_1])
    fit = al.FitImaging(dataset=masked_imaging_7x7_sub_2, tracer=tracer)

    assert fit.model_images_of_planes_list[0] == pytest.approx(
        fit.galaxy_model_image_dict[g0].array, 1.0e-4
    )
    assert fit.model_images_of_planes_list[1] == pytest.approx(
        fit.galaxy_model_image_dict[g1_linear].array, 1.0e-4
    )
    assert fit.model_images_of_planes_list[2] == pytest.approx(
        fit.galaxy_model_image_dict[galaxy_pix_0].array
        + fit.galaxy_model_image_dict[galaxy_pix_1].array,
        1.0e-4,
    )
    assert fit.model_images_of_planes_list[2][0] == pytest.approx(1.5614163282, 1.0e-4)


def test__subtracted_images_of_planes_list__three_distinct_redshift_planes__each_plane_subtraction_correct(
    masked_imaging_7x7_no_blur,
):
    g0 = al.Galaxy(redshift=0.5, bulge=al.lp.Sersic(intensity=1.0))
    g1 = al.Galaxy(redshift=0.75, bulge=al.lp.Sersic(intensity=2.0))
    g2 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(intensity=3.0))

    tracer = al.Tracer(galaxies=[g0, g1, g2])
    fit = al.FitImaging(dataset=masked_imaging_7x7_no_blur, tracer=tracer)

    assert fit.subtracted_images_of_planes_list[0].slim[0] == pytest.approx(0.200638, 1.0e-4)
    assert fit.subtracted_images_of_planes_list[1].slim[0] == pytest.approx(0.360511, 1.0e-4)
    assert fit.subtracted_images_of_planes_list[2].slim[0] == pytest.approx(0.520383, 1.0e-4)


def test__subtracted_images_of_planes_list__three_planes__sub_2_oversampling__subtraction_correct(
    masked_imaging_7x7_no_blur_sub_2,
):
    g0 = al.Galaxy(redshift=0.5, bulge=al.lp.Sersic(intensity=1.0))
    g1 = al.Galaxy(redshift=0.75, bulge=al.lp.Sersic(intensity=2.0))
    g2 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(intensity=3.0))

    tracer = al.Tracer(galaxies=[g0, g1, g2])
    fit = al.FitImaging(dataset=masked_imaging_7x7_no_blur_sub_2, tracer=tracer)

    assert fit.subtracted_images_of_planes_list[2].slim[0] == pytest.approx(
        0.475542485138, 1.0e-4
    )


def test__subtracted_images_of_planes_list__two_galaxies_at_same_redshift__subtraction_correct(
    masked_imaging_7x7_no_blur,
):
    g0 = al.Galaxy(redshift=0.5, bulge=al.lp.Sersic(intensity=1.0))
    g1 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(intensity=2.0))
    g2 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(intensity=3.0))

    tracer = al.Tracer(galaxies=[g0, g1, g2])
    fit = al.FitImaging(dataset=masked_imaging_7x7_no_blur, tracer=tracer)

    assert fit.subtracted_images_of_planes_list[0].slim[0] == pytest.approx(0.200638, 1.0e-4)
    assert fit.subtracted_images_of_planes_list[1].slim[0] == pytest.approx(0.840127, 1.0e-4)


def test__unmasked_blurred_images__two_plane_tracer__unmasked_images_match_tracer_output(
    masked_imaging_7x7,
):
    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(intensity=1.0),
        mass_profile=al.mp.IsothermalSph(einstein_radius=1.0),
    )
    g1 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(intensity=1.0))

    tracer = al.Tracer(galaxies=[g0, g1])
    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    unmasked_blurred_image = tracer.unmasked_blurred_image_2d_from(
        grid=masked_imaging_7x7.grids.lp, psf=masked_imaging_7x7.psf
    )
    assert (fit.unmasked_blurred_image == unmasked_blurred_image).all()

    unmasked_blurred_image_of_planes_list = tracer.unmasked_blurred_image_2d_list_from(
        grid=masked_imaging_7x7.grids.lp, psf=masked_imaging_7x7.psf
    )
    assert (
        unmasked_blurred_image_of_planes_list[0]
        == fit.unmasked_blurred_image_of_planes_list[0]
    ).all()
    assert (
        unmasked_blurred_image_of_planes_list[1]
        == fit.unmasked_blurred_image_of_planes_list[1]
    ).all()


def test__tracer_linear_light_profiles_to_light_profiles__solved_intensities_replace_linear_profiles(
    masked_imaging_7x7,
):
    g0 = al.Galaxy(
        redshift=0.5, bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0)
    )
    g0_linear = al.Galaxy(
        redshift=0.5,
        bulge=al.lp_linear.Sersic(centre=(0.05, 0.05), sersic_index=1.0),
        mass_profile=al.mp.IsothermalSph(centre=(0.05, 0.05), einstein_radius=1.0),
    )
    g1_linear = al.Galaxy(
        redshift=1.0,
        bulge=al.lp_linear.Sersic(centre=(0.05, 0.05), sersic_index=4.0),
    )

    tracer = al.Tracer(galaxies=[g0, g0_linear, g1_linear])
    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.tracer.galaxies[0].bulge.intensity == pytest.approx(1.0, 1.0e-4)

    tracer_converted = fit.tracer_linear_light_profiles_to_light_profiles

    assert tracer_converted.galaxies[0].bulge.intensity == pytest.approx(1.0, 1.0e-4)
    assert tracer_converted.galaxies[1].bulge.intensity == pytest.approx(
        -5.830442986, 1.0e-4
    )
    assert tracer_converted.galaxies[2].bulge.intensity == pytest.approx(
        0.135755913, 1.0e-4
    )


def test__total_mappers__no_pixelization__returns_zero(masked_imaging_7x7):
    tracer = al.Tracer(
        galaxies=[
            al.Galaxy(redshift=0.5),
            al.Galaxy(redshift=1.0),
            al.Galaxy(redshift=2.0),
        ]
    )
    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.total_mappers == 0


def test__total_mappers__one_pixelization__returns_one(masked_imaging_7x7):
    pixelization = al.Pixelization(mesh=al.mesh.RectangularUniform())
    tracer = al.Tracer(
        galaxies=[
            al.Galaxy(redshift=0.5),
            al.Galaxy(redshift=1.0),
            al.Galaxy(redshift=2.0, pixelization=pixelization),
        ]
    )
    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.total_mappers == 1


def test__total_mappers__three_pixelizations_across_planes__returns_three(
    masked_imaging_7x7,
):
    pixelization = al.Pixelization(mesh=al.mesh.RectangularUniform())
    tracer = al.Tracer(
        galaxies=[
            al.Galaxy(redshift=0.5, pixelization=pixelization),
            al.Galaxy(redshift=1.0, pixelization=pixelization),
            al.Galaxy(redshift=2.0, pixelization=pixelization),
        ]
    )
    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.total_mappers == 3


def test__mask_padding__off_centre_mask__same_likelihood_as_centred():
    """
    When the mask is close to the image edge and the PSF kernel footprint
    extends beyond the image boundary, the blurring mask is automatically
    padded. This test verifies that the padded fit produces the same
    log-likelihood as an equivalent centred fit that requires no padding.
    """
    pixel_scales = 0.2

    psf = al.Convolver.from_gaussian(
        shape_native=(11, 11), pixel_scales=pixel_scales, sigma=0.75, normalize=True
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.Sersic(
            centre=(0.0, 0.0), intensity=0.1,
            effective_radius=0.3, sersic_index=2.0,
        ),
        mass=al.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.0),
    )
    source_galaxy = al.Galaxy(
        redshift=1.0,
        light=al.lp.Exponential(
            centre=(0.0, 0.0), intensity=0.3, effective_radius=0.2,
        ),
    )
    tracer_centred = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    sim = al.SimulatorImaging(
        exposure_time=300.0, psf=psf, add_poisson_noise_to_data=False,
    )

    # --- Centred fit on a large grid: no padding needed ---
    grid_large = al.Grid2D.uniform(
        shape_native=(51, 51), pixel_scales=pixel_scales, over_sample_size=1,
    )
    dataset_centred = sim.via_tracer_from(tracer=tracer_centred, grid=grid_large)
    dataset_centred.noise_map = al.Array2D.ones(
        shape_native=(51, 51), pixel_scales=pixel_scales,
    )
    mask_centred = al.Mask2D.circular(
        shape_native=(51, 51), pixel_scales=pixel_scales,
        radius=0.6, centre=(0.0, 0.0),
    )
    masked_centred = dataset_centred.apply_mask(mask=mask_centred)
    fit_centred = al.FitImaging(dataset=masked_centred, tracer=tracer_centred)

    # --- Off-centre fit on a small grid: triggers padding ---
    offset = (0.0, 1.2)
    lens_off = al.Galaxy(
        redshift=0.5,
        light=al.lp.Sersic(
            centre=offset, intensity=0.1,
            effective_radius=0.3, sersic_index=2.0,
        ),
        mass=al.mp.Isothermal(centre=offset, einstein_radius=1.0),
    )
    source_off = al.Galaxy(
        redshift=1.0,
        light=al.lp.Exponential(
            centre=offset, intensity=0.3, effective_radius=0.2,
        ),
    )
    tracer_off = al.Tracer(galaxies=[lens_off, source_off])

    grid_small = al.Grid2D.uniform(
        shape_native=(21, 21), pixel_scales=pixel_scales, over_sample_size=1,
    )
    dataset_off = sim.via_tracer_from(tracer=tracer_off, grid=grid_small)
    dataset_off.noise_map = al.Array2D.ones(
        shape_native=(21, 21), pixel_scales=pixel_scales,
    )
    mask_off = al.Mask2D.circular(
        shape_native=(21, 21), pixel_scales=pixel_scales,
        radius=0.6, centre=offset,
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        masked_off = dataset_off.apply_mask(mask=mask_off)
        fit_off = al.FitImaging(dataset=masked_off, tracer=tracer_off)
        padding_occurred = any("Mask padded" in str(x.message) for x in w)

    assert padding_occurred, "Expected mask padding to be triggered for the off-centre mask"

    assert fit_centred.chi_squared == pytest.approx(0.0, abs=1e-4)
    assert fit_off.chi_squared == pytest.approx(0.0, abs=1e-4)
    assert fit_off.log_likelihood == pytest.approx(
        fit_centred.log_likelihood, abs=1e-4,
    )
