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


        assert isinstance(fit_sparse.inversion, aa.InversionInterferometerSparse)
        assert isinstance(fit.inversion, aa.InversionInterferometerMapping)

        assert fit_sparse.log_likelihood == pytest.approx(
            fit.log_likelihood, rel=1.0e-8
        )
        assert fit_sparse.log_evidence == pytest.approx(fit.log_evidence, rel=1.0e-8)

        assert (
            fit_sparse.inversion.dataset.sparse_dirty_image is not None
        ) is has_lens_light

        # The image `i_p` the sparse dirty image is corrected with (`d~ - W~ i_p`) must be exactly the image
        # the fit's `profile_visibilities` are the Fourier transform of.
        profile_visibilities = tracer.visibilities_from(
            grid=dataset_sparse.grids.lp, transformer=dataset_sparse.transformer
        )

        np.testing.assert_allclose(
            dataset_sparse.transformer.visibilities_from(
                image=fit_sparse.profile_image
            ).array,
            profile_visibilities.array,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            fit_sparse.profile_visibilities.array,
            profile_visibilities.array,
            rtol=1.0e-12,
            atol=1.0e-12,
        )


def test__profile_visibilities__linear_light_only__zeros_without_fourier_transform(
    interferometer_7, monkeypatch
):
    """
    A tracer whose light is entirely linear (a lens with only mass and a source MGE `Basis` of linear
    Gaussians) has an all-zero ordinary light image, so `profile_visibilities` must be zeros without
    performing a Fourier transform.
    """
    calls = []

    visibilities_from = interferometer_7.transformer.visibilities_from

    def spy(*args, **kwargs):
        calls.append(1)
        return visibilities_from(*args, **kwargs)

    monkeypatch.setattr(interferometer_7.transformer, "visibilities_from", spy)

    source = al.Galaxy(
        redshift=1.0,
        bulge=al.lp_basis.Basis(
            profile_list=[
                al.lp_linear.Gaussian(sigma=sigma, centre=(0.1, 0.1))
                for sigma in (0.3, 1.0, 3.0)
            ]
        ),
    )

    tracer = al.Tracer(
        galaxies=[
            al.Galaxy(
                redshift=0.5,
                mass=al.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.0),
            ),
            source,
        ]
    )

    fit = al.FitInterferometer(dataset=interferometer_7, tracer=tracer)

    profile_visibilities = fit.profile_visibilities

    assert calls == []
    assert profile_visibilities.shape == interferometer_7.data.shape
    assert np.all(profile_visibilities.array == 0.0)

    # Sparse: nothing is subtracted from the visibilities, so the likelihood must not even build the (zero)
    # profile visibilities: the inversion is passed `data=None` and reads its data term from the sparse
    # operator's cached scalar.
    dataset_sparse = interferometer_7.apply_sparse_operator(use_jax=False)

    # The sparse dataset reuses the dense dataset's transformer, so one spy covers both.
    assert dataset_sparse.transformer is interferometer_7.transformer

    zeros_calls = []

    zeros = aa.Visibilities.zeros

    def zeros_spy(*args, **kwargs):
        zeros_calls.append(1)
        return zeros(*args, **kwargs)

    monkeypatch.setattr(aa.Visibilities, "zeros", zeros_spy)

    fit_sparse = al.FitInterferometer(dataset=dataset_sparse, tracer=tracer)

    figure_of_merit = fit_sparse.figure_of_merit

    assert fit_sparse._uses_precomputed_data_term
    assert fit_sparse.inversion.dataset.data is None
    assert calls == []
    assert zeros_calls == []
    assert "profile_visibilities" not in fit_sparse.__dict__
    assert "profile_subtracted_visibilities" not in fit_sparse.__dict__

    # The value is the one given by passing the visibilities explicitly (the array path).
    with monkeypatch.context() as m:
        m.setattr(
            al.FitInterferometer,
            "_uses_precomputed_data_term",
            property(lambda self: False),
        )

        fit_array = al.FitInterferometer(dataset=dataset_sparse, tracer=tracer)

        assert fit_array.inversion.dataset.data is not None
        assert figure_of_merit == pytest.approx(fit_array.figure_of_merit, rel=1.0e-12)

    # Output paths still see real (zero) profile visibilities.
    assert np.all(fit_sparse.profile_visibilities.array == 0.0)


def _pixelized_source_tracer(coefficient=1.0, lens_light=False):
    lens = al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.0),
    )

    if lens_light:
        lens.bulge = al.lp.Sersic(intensity=0.1, centre=(0.05, 0.05))

    source = al.Galaxy(
        redshift=1.0,
        pixelization=al.Pixelization(
            mesh=al.mesh.RectangularUniform(shape=(3, 3)),
            regularization=al.reg.Constant(coefficient=coefficient),
        ),
    )

    return al.Tracer(galaxies=[lens, source])


def test__fit_figure_of_merit__sparse_operator__pixelization_only__data_term_scalar_matches_dense(
    interferometer_7, monkeypatch
):
    """
    A lens with only mass and a pixelized source on the sparse path passes `data=None` to its inversion, whose
    `fast_chi_squared` then reads the data term cached on the sparse operator; the log evidence must match the
    dense fit, and be exactly the value obtained when the visibilities are passed explicitly.
    """
    dataset_sparse = interferometer_7.apply_sparse_operator(use_jax=False)

    tracer = _pixelized_source_tracer()

    fit = al.FitInterferometer(dataset=interferometer_7, tracer=tracer)
    fit_sparse = al.FitInterferometer(dataset=dataset_sparse, tracer=tracer)

    assert isinstance(fit.inversion, aa.InversionInterferometerMapping)
    assert isinstance(fit_sparse.inversion, aa.InversionInterferometerSparse)

    assert fit_sparse._uses_precomputed_data_term
    assert fit_sparse.inversion.dataset.data is None

    assert fit_sparse.log_likelihood == pytest.approx(fit.log_likelihood, rel=1.0e-8)
    assert fit_sparse.log_evidence == pytest.approx(fit.log_evidence, rel=1.0e-8)
    assert fit_sparse.noise_normalization == fit.noise_normalization

    # Output quantities that read the visibilities get them via `inversion_with_data`.
    inversion_with_data = fit_sparse.inversion_with_data

    assert inversion_with_data.dataset.data is fit_sparse.data
    assert inversion_with_data.reconstruction is fit_sparse.inversion.reconstruction
    assert inversion_with_data.fast_chi_squared == pytest.approx(
        fit_sparse.inversion.fast_chi_squared, rel=1.0e-12
    )

    mapper = inversion_with_data.cls_list_from(cls=aa.Mapper)[0]

    np.testing.assert_array_equal(
        inversion_with_data.data_subtracted_dict[mapper].array,
        interferometer_7.data.array,
    )

    # The dense fit's inversion already carries its data, so it is returned unchanged.
    assert fit.inversion_with_data is fit.inversion

    # Control: with the gate forced off the visibilities are passed explicitly, and the figure of merit is
    # bit-identical.
    figure_of_merit = fit_sparse.figure_of_merit

    monkeypatch.setattr(
        "autolens.interferometer.fit_interferometer.uses_precomputed_data_term_from",
        lambda **kwargs: False,
    )

    fit_forced = al.FitInterferometer(dataset=dataset_sparse, tracer=tracer)

    assert not fit_forced._uses_precomputed_data_term
    assert fit_forced.inversion.dataset.data is not None
    assert fit_forced.figure_of_merit == figure_of_merit


def test__fit_figure_of_merit__sparse_operator__pixelization_only__no_visibility_arrays(
    interferometer_7, monkeypatch
):
    """
    On the gated sparse path the likelihood must perform no Fourier transform and build no visibility-sized
    zeros, and must never evaluate `profile_visibilities` or `profile_subtracted_visibilities`.
    """
    dataset_sparse = interferometer_7.apply_sparse_operator(use_jax=False)

    calls = []

    visibilities_from = dataset_sparse.transformer.visibilities_from

    def spy(*args, **kwargs):
        calls.append(1)
        return visibilities_from(*args, **kwargs)

    monkeypatch.setattr(dataset_sparse.transformer, "visibilities_from", spy)

    zeros_calls = []

    zeros = aa.Visibilities.zeros

    def zeros_spy(*args, **kwargs):
        zeros_calls.append(1)
        return zeros(*args, **kwargs)

    monkeypatch.setattr(aa.Visibilities, "zeros", zeros_spy)

    fit = al.FitInterferometer(
        dataset=dataset_sparse, tracer=_pixelized_source_tracer()
    )

    fit.figure_of_merit

    assert calls == []
    assert zeros_calls == []
    assert "profile_visibilities" not in fit.__dict__
    assert "profile_subtracted_visibilities" not in fit.__dict__

    # Output paths still see real (zero) profile visibilities.
    assert np.all(fit.profile_visibilities.array == 0.0)


def test__fit_figure_of_merit__sparse_operator__light_profile__unchanged_vs_data_passed(
    interferometer_7, monkeypatch
):
    """
    With a lens ordinary light profile the sparse fit must keep passing the profile-subtracted visibilities, so
    its figure of merit is exactly the one computed with the data passed explicitly.
    """
    dataset_sparse = interferometer_7.apply_sparse_operator(use_jax=False)

    tracer = _pixelized_source_tracer(lens_light=True)

    fit_sparse = al.FitInterferometer(dataset=dataset_sparse, tracer=tracer)

    assert not fit_sparse._uses_precomputed_data_term

    figure_of_merit = fit_sparse.figure_of_merit

    monkeypatch.setattr(
        al.FitInterferometer,
        "_uses_precomputed_data_term",
        property(lambda self: False),
    )

    fit_forced = al.FitInterferometer(dataset=dataset_sparse, tracer=tracer)

    assert fit_forced.figure_of_merit == figure_of_merit
    np.testing.assert_array_equal(
        fit_sparse.inversion.dataset.data.array,
        (interferometer_7.data - fit_sparse.profile_visibilities).array,
    )


def test__fit_figure_of_merit__sparse_operator__pixelization_only__jax_jit_matches_numpy(
    interferometer_7,
):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    dataset_sparse = interferometer_7.apply_sparse_operator(use_jax=False)

    def figure_of_merit_from(coefficient, xp):
        fit = al.FitInterferometer(
            dataset=dataset_sparse,
            tracer=_pixelized_source_tracer(coefficient=coefficient),
            xp=xp,
        )

        assert fit.inversion.dataset.data is None

        return fit.figure_of_merit

    figure_of_merit_numpy = figure_of_merit_from(coefficient=1.0, xp=np)

    figure_of_merit_jax = jax.jit(lambda c: figure_of_merit_from(c, xp=jnp))(1.0)

    assert float(figure_of_merit_jax) == pytest.approx(
        figure_of_merit_numpy, rel=1.0e-8
    )


def _array_free_dataset_from(dataset):
    """
    The array-free counterpart of `dataset` (no visibilities, uv-wavelengths or transformer), built by
    streaming its visibilities through `Interferometer.from_stream` with the same transformer class.
    """
    return aa.Interferometer.from_stream(
        [(dataset.uv_wavelengths, dataset.data, dataset.noise_map)],
        real_space_mask=dataset.real_space_mask,
        transformer_class=type(dataset.transformer),
    )


def test__fit_figure_of_merit__array_free_dataset__pixelization_only__matches_in_memory_sparse(
    interferometer_7,
):
    dataset_sparse = interferometer_7.apply_sparse_operator(use_jax=False)
    dataset_array_free = _array_free_dataset_from(interferometer_7)

    assert dataset_array_free.is_array_free

    tracer = _pixelized_source_tracer()

    fit_sparse = al.FitInterferometer(dataset=dataset_sparse, tracer=tracer)
    fit_array_free = al.FitInterferometer(dataset=dataset_array_free, tracer=tracer)

    assert fit_array_free._uses_precomputed_data_term
    assert fit_array_free.inversion.dataset.data is None
    assert isinstance(fit_array_free.inversion, aa.InversionInterferometerSparse)

    assert fit_array_free.figure_of_merit == pytest.approx(
        fit_sparse.figure_of_merit, rel=1.0e-8
    )
    assert fit_array_free.log_evidence == pytest.approx(
        fit_sparse.log_evidence, rel=1.0e-8
    )

    assert fit_array_free.profile_visibilities is None
    assert fit_array_free.profile_subtracted_visibilities is None
    assert fit_array_free.inversion_with_data is fit_array_free.inversion


def test__fit_figure_of_merit__array_free_dataset__pixelization_only__jax_jit_matches_numpy(
    interferometer_7,
):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    dataset_sparse = interferometer_7.apply_sparse_operator(use_jax=False)
    dataset_array_free = _array_free_dataset_from(interferometer_7)

    def figure_of_merit_from(coefficient, dataset, xp):
        fit = al.FitInterferometer(
            dataset=dataset,
            tracer=_pixelized_source_tracer(coefficient=coefficient),
            xp=xp,
        )

        assert fit.inversion.dataset.data is None

        return fit.figure_of_merit

    figure_of_merit_sparse = figure_of_merit_from(
        coefficient=1.0, dataset=dataset_sparse, xp=np
    )
    figure_of_merit_numpy = figure_of_merit_from(
        coefficient=1.0, dataset=dataset_array_free, xp=np
    )

    figure_of_merit_jax = jax.jit(
        lambda c: figure_of_merit_from(c, dataset=dataset_array_free, xp=jnp)
    )(1.0)

    assert figure_of_merit_numpy == pytest.approx(figure_of_merit_sparse, rel=1.0e-8)
    assert float(figure_of_merit_jax) == pytest.approx(
        figure_of_merit_numpy, rel=1.0e-8
    )


def test__fit_figure_of_merit__array_free_dataset__lens_light_profile__raises(
    interferometer_7,
):
    dataset_array_free = _array_free_dataset_from(interferometer_7)

    tracer = _pixelized_source_tracer(lens_light=True)

    fit = al.FitInterferometer(dataset=dataset_array_free, tracer=tracer)

    assert not fit._uses_precomputed_data_term

    with pytest.raises(aa.exc.DatasetException):
        fit.profile_visibilities

    with pytest.raises(aa.exc.DatasetException):
        al.FitInterferometer(dataset=dataset_array_free, tracer=tracer).figure_of_merit
