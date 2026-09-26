import numpy as np
import pytest

import autoarray as aa
import autolens as al


def test__model_visibilities(interferometer_7):
    g0 = al.Galaxy(redshift=0.5, bulge=al.m.MockLightProfile(image_2d=np.ones(9)))
    tracer = al.Tracer(galaxies=[g0])

    fit = al.FitInterferometer(dataset=interferometer_7, tracer=tracer)

    assert fit.model_data.slim[0].real == pytest.approx(1.48496, abs=1.0e-4)
    assert fit.model_data.slim[0].imag == pytest.approx(0.0, abs=1.0e-4)
    assert fit.log_likelihood == pytest.approx(-34.1685958, abs=1.0e-4)


def test__fit_figure_of_merit(interferometer_7):
    # TODO : Use pytest.parameterize

    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0),
        disk=al.lp.Sersic(centre=(0.05, 0.05), intensity=2.0),
        mass_profile=al.mp.IsothermalSph(centre=(0.05, 0.05), einstein_radius=1.0),
    )

    g1 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(intensity=1.0))

    tracer = al.Tracer(galaxies=[g0, g1])

    fit = al.FitInterferometer(dataset=interferometer_7, tracer=tracer)

    assert fit.perform_inversion is False
    assert fit.figure_of_merit == pytest.approx(-12758.714175708, 1.0e-4)

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

    fit = al.FitInterferometer(dataset=interferometer_7, tracer=tracer)

    assert fit.perform_inversion is False
    assert fit.figure_of_merit == pytest.approx(-12779.937568696, 1.0e-4)

    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=0.01),
    )

    g0 = al.Galaxy(redshift=0.5, pixelization=pixelization)

    tracer = al.Tracer(galaxies=[al.Galaxy(redshift=0.5), g0])

    fit = al.FitInterferometer(
        dataset=interferometer_7,
        tracer=tracer,
    )

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-71.7704487241, 1.0e-4)

    galaxy_light = al.Galaxy(
        redshift=0.5, bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0)
    )

    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )

    galaxy_pix = al.Galaxy(redshift=1.0, pixelization=pixelization)

    tracer = al.Tracer(galaxies=[galaxy_light, galaxy_pix])

    fit = al.FitInterferometer(
        dataset=interferometer_7,
        tracer=tracer,
    )

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-196.15073725528, 1.0e-4)

    g0_linear = al.Galaxy(
        redshift=0.5,
        bulge=al.lp_linear.Sersic(centre=(0.05, 0.05), sersic_index=1.0),
        disk=al.lp_linear.Sersic(centre=(0.05, 0.05), sersic_index=4.0),
        mass_profile=al.mp.IsothermalSph(centre=(0.05, 0.05), einstein_radius=1.0),
    )

    tracer = al.Tracer(galaxies=[g0_linear, g1])

    fit = al.FitInterferometer(dataset=interferometer_7, tracer=tracer)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-197.670468767, 1.0e-4)

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

    tracer = al.Tracer(galaxies=[g0_linear, g1])

    fit = al.FitInterferometer(dataset=interferometer_7, tracer=tracer)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-197.6704687, 1.0e-4)

    tracer = al.Tracer(galaxies=[g0_linear, galaxy_pix])

    fit = al.FitInterferometer(dataset=interferometer_7, tracer=tracer)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-35.07914066930113, 1.0e-4)


def test___galaxy_image_dict(interferometer_7, interferometer_7_grid):
    # Normal Light Profiles Only

    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0),
        mass_profile=al.mp.IsothermalSph(centre=(0.05, 0.05), einstein_radius=1.0),
    )
    g1 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0))
    g2 = al.Galaxy(redshift=1.0)

    tracer = al.Tracer(galaxies=[g0, g1, g2])

    fit = al.FitInterferometer(
        dataset=interferometer_7_grid,
        tracer=tracer,
    )

    traced_grid_2d_list_from = tracer.traced_grid_2d_list_from(
        grid=interferometer_7.grids.lp
    )

    g0_image = g0.image_2d_from(grid=traced_grid_2d_list_from[0])
    g1_image = g1.image_2d_from(grid=traced_grid_2d_list_from[1])

    assert fit.galaxy_image_dict[g0] == pytest.approx(g0_image.array, 1.0e-4)
    assert fit.galaxy_image_dict[g1] == pytest.approx(g1_image.array, 1.0e-4)

    # Linear Light Profiles Only

    g0_linear = al.Galaxy(
        redshift=0.5,
        bulge=al.lp_linear.Sersic(centre=(0.05, 0.05)),
        mass_profile=al.mp.IsothermalSph(centre=(0.05, 0.05), einstein_radius=1.0),
    )
    g1_linear = al.Galaxy(redshift=1.0, bulge=al.lp_linear.Sersic())

    tracer = al.Tracer(galaxies=[g0_linear, g1_linear, g2])

    fit = al.FitInterferometer(
        dataset=interferometer_7_grid,
        tracer=tracer,
    )

    assert fit.galaxy_image_dict[g0_linear][4] == pytest.approx(1.00018622848, 1.0e-2)
    assert fit.galaxy_image_dict[g1_linear][3] == pytest.approx(-0.017435532289, 1.0e-2)

    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )

    g0_no_light = al.Galaxy(
        redshift=0.5,
        mass_profile=al.mp.IsothermalSph(centre=(0.05, 0.05), einstein_radius=1.0),
    )
    galaxy_pix_0 = al.Galaxy(redshift=1.0, pixelization=pixelization)

    tracer = al.Tracer(galaxies=[g0_no_light, galaxy_pix_0])

    fit = al.FitInterferometer(
        dataset=interferometer_7,
        tracer=tracer,
    )

    assert (fit.galaxy_image_dict[g0_no_light].native == np.zeros((7, 7))).all()

    assert fit.galaxy_image_dict[galaxy_pix_0][0] == pytest.approx(
        -0.14416215690290285, 1.0e-4
    )

    # Normal light + Linear Light PRofiles + Pixelization + Regularization

    galaxy_pix_1 = al.Galaxy(redshift=1.0, pixelization=pixelization)
    tracer = al.Tracer(galaxies=[g0, g0_linear, g2, galaxy_pix_0, galaxy_pix_1])

    fit = al.FitInterferometer(
        dataset=interferometer_7_grid,
        tracer=tracer,
    )

    assert fit.galaxy_image_dict[g0] == pytest.approx(g0_image.array, 1.0e-4)

    assert fit.galaxy_image_dict[g0_linear][4] == pytest.approx(
        -22.896762784783178, 1.0e-4
    )

    assert fit.galaxy_image_dict[galaxy_pix_0][4] == pytest.approx(
        -0.027972920686385083, 1.0e-3
    )
    assert fit.galaxy_image_dict[galaxy_pix_1][4] == pytest.approx(
        -0.02797291718230969, 1.0e-3
    )
    assert (fit.galaxy_image_dict[g2] == np.zeros(9)).all()


def test__galaxy_model_visibilities_dict(interferometer_7, interferometer_7_grid):
    # Normal Light Profiles Only

    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0),
        mass_profile=al.mp.IsothermalSph(centre=(0.05, 0.05), einstein_radius=1.0),
    )
    g1 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0))
    g2 = al.Galaxy(redshift=1.0)

    tracer = al.Tracer(galaxies=[g0, g1, g2])

    fit = al.FitInterferometer(dataset=interferometer_7, tracer=tracer)

    traced_grid_2d_list_from = tracer.traced_grid_2d_list_from(
        grid=interferometer_7.grids.lp
    )

    g0_profile_visibilities = g0.visibilities_from(
        grid=traced_grid_2d_list_from[0], transformer=interferometer_7_grid.transformer
    )

    g1_profile_visibilities = g1.visibilities_from(
        grid=traced_grid_2d_list_from[1], transformer=interferometer_7_grid.transformer
    )

    assert fit.galaxy_model_visibilities_dict[g0].slim.array == pytest.approx(
        g0_profile_visibilities.array, 1.0e-4
    )
    assert fit.galaxy_model_visibilities_dict[g1].slim.array == pytest.approx(
        g1_profile_visibilities.array, 1.0e-4
    )
    assert (
        fit.galaxy_model_visibilities_dict[g2].slim.array
        == (0.0 + 0.0j) * np.zeros((7,))
    ).all()

    assert fit.model_data.slim.array == pytest.approx(
        fit.galaxy_model_visibilities_dict[g0].slim.array
        + fit.galaxy_model_visibilities_dict[g1].slim.array,
        1.0e-4,
    )

    # Linear Light Profiles Only

    g0_linear = al.Galaxy(
        redshift=0.5,
        bulge=al.lp_linear.Sersic(centre=(0.05, 0.05)),
        mass_profile=al.mp.IsothermalSph(centre=(0.05, 0.05), einstein_radius=1.0),
    )
    g1_linear = al.Galaxy(redshift=1.0, bulge=al.lp_linear.Sersic(centre=(0.05, 0.05)))

    tracer = al.Tracer(galaxies=[g0_linear, g1_linear, g2])

    fit = al.FitInterferometer(dataset=interferometer_7, tracer=tracer)

    assert fit.galaxy_model_visibilities_dict[g0_linear][0] == pytest.approx(
        1.0138228768598911 + 0.006599377953512708j, 1.0e-2
    )
    assert fit.galaxy_model_visibilities_dict[g1_linear][0] == pytest.approx(
        -0.012892097547972572 - 0.0019719184145301906j, 1.0e-2
    )
    assert (fit.galaxy_model_visibilities_dict[g2] == np.zeros((7,))).all()

    assert fit.model_data.array == pytest.approx(
        fit.galaxy_model_visibilities_dict[g0_linear].array
        + fit.galaxy_model_visibilities_dict[g1_linear].array,
        1.0e-4,
    )

    # Pixelization + Regularizaiton only

    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )

    g0_no_light = al.Galaxy(
        redshift=0.5,
        mass_profile=al.mp.IsothermalSph(centre=(0.05, 0.05), einstein_radius=1.0),
    )
    galaxy_pix_0 = al.Galaxy(redshift=1.0, pixelization=pixelization)

    tracer = al.Tracer(galaxies=[g0_no_light, galaxy_pix_0])

    fit = al.FitInterferometer(dataset=interferometer_7, tracer=tracer)

    assert (fit.galaxy_model_visibilities_dict[g0_no_light] == np.zeros((7,))).all()
    assert fit.galaxy_model_visibilities_dict[galaxy_pix_0][0] == pytest.approx(
        0.9372276032882099 + 0.32701101190914555j, 1.0e-4
    )

    assert fit.model_data.array == pytest.approx(
        fit.galaxy_model_visibilities_dict[galaxy_pix_0].array, 1.0e-4
    )

    # Normal light + Linear Light PRofiles + Pixelization + Regularizaiton

    galaxy_pix_1 = al.Galaxy(redshift=1.0, pixelization=pixelization)

    tracer = al.Tracer(galaxies=[g0, g0_linear, g2, galaxy_pix_0, galaxy_pix_1])

    fit = al.FitInterferometer(dataset=interferometer_7, tracer=tracer)

    assert fit.galaxy_model_visibilities_dict[g0].array == pytest.approx(
        g0_profile_visibilities.array, 1.0e-4
    )

    assert fit.galaxy_model_visibilities_dict[g0_linear][0] == pytest.approx(
        -23.101974527607315 - 0.15037997746935183j, 1.0e-4
    )

    assert fit.galaxy_model_visibilities_dict[galaxy_pix_0][0] == pytest.approx(
        -0.06797397090116747 + 0.0892532970401005j, 1.0e-4
    )
    assert fit.galaxy_model_visibilities_dict[galaxy_pix_1][0] == pytest.approx(
        -0.06797396569775271 + 0.08925329704010045j, 1.0e-4
    )
    assert (fit.galaxy_model_visibilities_dict[g2] == np.zeros((7,))).all()


def test__model_visibilities_of_planes_list(interferometer_7):
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

    fit = al.FitInterferometer(dataset=interferometer_7, tracer=tracer)

    assert fit.model_visibilities_of_planes_list[0].array == pytest.approx(
        fit.galaxy_model_visibilities_dict[g0].array, 1.0e-4
    )
    assert fit.model_visibilities_of_planes_list[1].array == pytest.approx(
        fit.galaxy_model_visibilities_dict[g1_linear].array, 1.0e-4
    )
    assert fit.model_visibilities_of_planes_list[2].array == pytest.approx(
        fit.galaxy_model_visibilities_dict[galaxy_pix_0].array
        + fit.galaxy_model_visibilities_dict[galaxy_pix_1].array,
        1.0e-4,
    )


def test__fit_figure_of_merit__sparse_operator__lens_light_profile_and_source_mge__matches_dense(
    interferometer_7,
):
    """
    With the sparse operator applied, a lens ordinary light profile plus a source MGE (linear Gaussians)
    must reproduce the dense fit: the lens light's visibilities are subtracted before the inversion, so the
    sparse data vector must use the dirty image of the profile-subtracted visibilities.
    """
    dataset_sparse = interferometer_7.apply_sparse_operator(use_jax=False)

    lens = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(intensity=0.1, centre=(0.05, 0.05)),
        mass=al.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.0),
    )
    source = al.Galaxy(
        redshift=1.0,
        bulge=al.lp_basis.Basis(
            profile_list=[
                al.lp_linear.Gaussian(sigma=sigma, centre=(0.1, 0.1))
                for sigma in (0.3, 1.0, 3.0)
            ]
        ),
    )

    # The second case, with no lens light, is the all-linear control which uses the cached dirty image.
    for galaxies, has_lens_light in (
        ([lens, source], True),
        ([al.Galaxy(redshift=0.5, mass=lens.mass), source], False),
    ):
        tracer = al.Tracer(galaxies=galaxies)

        fit = al.FitInterferometer(dataset=interferometer_7, tracer=tracer)
        fit_sparse = al.FitInterferometer(dataset=dataset_sparse, tracer=tracer)

        assert (
            fit_sparse.inversion.dataset.sparse_dirty_image is not None
        ) is has_lens_light

        assert isinstance(fit_sparse.inversion, aa.InversionInterferometerSparse)
        assert isinstance(fit.inversion, aa.InversionInterferometerMapping)

        assert fit_sparse.log_likelihood == pytest.approx(
            fit.log_likelihood, rel=1.0e-8
        )
        assert fit_sparse.log_evidence == pytest.approx(fit.log_evidence, rel=1.0e-8)
