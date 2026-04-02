import os
from os import path
import shutil

import autolens as al
import numpy as np
import pytest


def test__perfect_fit__chi_squared_0():
    grid = al.Grid2D.uniform(
        shape_native=(51, 51), pixel_scales=0.1, over_sample_size=1
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.Sersic(centre=(0.1, 0.1), intensity=0.1),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=1.0),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0, light=al.lp.Exponential(centre=(0.1, 0.1), intensity=0.5)
    )

    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    simulator = al.SimulatorInterferometer(
        uv_wavelengths=np.ones(shape=(7, 2)),
        transformer_class=al.TransformerDFT,
        exposure_time=300.0,
        noise_if_add_noise_false=1.0,
        noise_sigma=None,
    )

    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

    file_path = path.join(
        "{}".format(path.dirname(path.realpath(__file__))),
        "data_temp",
        "simulate_and_fit",
    )

    try:
        shutil.rmtree(file_path)
    except FileNotFoundError:
        pass

    if path.exists(file_path) is False:
        os.makedirs(file_path)

    from autoarray.dataset.plot.interferometer_plots import fits_interferometer

    fits_interferometer(
        dataset=dataset,
        data_path=path.join(file_path, "data.fits"),
        noise_map_path=path.join(file_path, "noise_map.fits"),
        uv_wavelengths_path=path.join(file_path, "uv_wavelengths.fits"),
        overwrite=True,
    )

    real_space_mask = al.Mask2D.all_false(
        shape_native=(51, 51),
        pixel_scales=0.1,
    )

    dataset = al.Interferometer.from_fits(
        data_path=path.join(file_path, "data.fits"),
        noise_map_path=path.join(file_path, "noise_map.fits"),
        uv_wavelengths_path=path.join(file_path, "uv_wavelengths.fits"),
        real_space_mask=real_space_mask,
        transformer_class=al.TransformerDFT,
    )

    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    fit = al.FitInterferometer(
        dataset=dataset,
        tracer=tracer,
    )

    assert fit.chi_squared == pytest.approx(0.0)

    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(7, 7)),
        regularization=al.reg.Constant(coefficient=0.0001),
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.Sersic(centre=(0.1, 0.1), intensity=0.1),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=1.0),
    )

    source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    fit = al.FitInterferometer(
        dataset=dataset,
        tracer=tracer,
    )
    assert abs(fit.chi_squared) < 1.0e-4

    file_path = path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "data_temp"
    )

    if path.exists(file_path) is True:
        shutil.rmtree(file_path)


def test__simulate_interferometer_data_and_fit__known_likelihood():
    mask = al.Mask2D.circular(radius=3.0, shape_native=(31, 31), pixel_scales=0.2)

    grid = al.Grid2D.from_mask(mask=mask, over_sample_size=1)

    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(16, 16)),
        regularization=al.reg.Constant(coefficient=1.0),
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.Sersic(centre=(0.1, 0.1), intensity=0.1),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=1.8),
    )
    source_galaxy_0 = al.Galaxy(redshift=1.0, pixelization=pixelization)
    source_galaxy_1 = al.Galaxy(redshift=2.0, pixelization=pixelization)
    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy_0, source_galaxy_1])

    simulator = al.SimulatorInterferometer(
        uv_wavelengths=np.ones(shape=(7, 2)),
        transformer_class=al.TransformerDFT,
        exposure_time=300.0,
        noise_seed=1,
    )

    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

    fit = al.FitInterferometer(
        dataset=dataset,
        tracer=tracer,
    )

    assert fit.figure_of_merit == pytest.approx(-5.433894158056919, 1.0e-2)


def test__simulate_interferometer_data_and_fit__linear_light_profiles_agree_with_standard_light_profiles():
    grid = al.Grid2D.uniform(
        shape_native=(51, 51), pixel_scales=0.1, over_sample_size=1
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.Sersic(centre=(0.1, 0.1), intensity=0.1),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=1.0),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        bulge=al.lp.Sersic(intensity=0.1, sersic_index=1.0),
        disk=al.lp.Sersic(intensity=0.2, sersic_index=4.0),
    )

    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    simulator = al.SimulatorInterferometer(
        uv_wavelengths=np.array(
            [
                [0.04, 200.0, 0.3, 400000.0, 60000000.0],
                [0.00003, 500.0, 600000.0, 0.1, 75555555],
            ]
        ),
        transformer_class=al.TransformerDFT,
        exposure_time=300.0,
        noise_if_add_noise_false=1.0,
        noise_sigma=None,
    )

    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

    fit = al.FitInterferometer(
        dataset=dataset,
        tracer=tracer,
    )

    lens_galaxy_linear = al.Galaxy(
        redshift=0.5,
        light=al.lp_linear.Sersic(centre=(0.1, 0.1)),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=1.0),
    )

    source_galaxy_linear = al.Galaxy(
        redshift=1.0,
        bulge=al.lp_linear.Sersic(sersic_index=1.0),
        disk=al.lp_linear.Sersic(sersic_index=4.0),
    )

    tracer_linear = al.Tracer(galaxies=[lens_galaxy_linear, source_galaxy_linear])

    fit_linear = al.FitInterferometer(
        dataset=dataset,
        tracer=tracer_linear,
    )

    assert fit_linear.inversion.reconstruction == pytest.approx(
        np.array([0.1, 0.1, 0.2]), 1.0e-4
    )
    assert fit_linear.linear_light_profile_intensity_dict[
        lens_galaxy_linear.light
    ] == pytest.approx(0.1, 1.0e-2)
    assert fit_linear.linear_light_profile_intensity_dict[
        source_galaxy_linear.bulge
    ] == pytest.approx(0.1, 1.0e-2)
    assert fit_linear.linear_light_profile_intensity_dict[
        source_galaxy_linear.disk
    ] == pytest.approx(0.2, 1.0e-2)
    assert fit.log_likelihood == pytest.approx(fit_linear.log_likelihood)

    lens_galaxy_image = lens_galaxy.image_2d_from(grid=dataset.grids.lp)

    assert fit_linear.galaxy_image_dict[lens_galaxy_linear] == pytest.approx(
        lens_galaxy_image.array, 1.0e-4
    )

    traced_grid_2d_list = tracer.traced_grid_2d_list_from(grid=dataset.grids.lp)

    source_galaxy_image = source_galaxy.image_2d_from(grid=traced_grid_2d_list[1])

    assert fit_linear.galaxy_image_dict[source_galaxy_linear] == pytest.approx(
        source_galaxy_image.array, 1.0e-4
    )

    lens_galaxy_visibilities = lens_galaxy.visibilities_from(
        grid=dataset.grids.lp, transformer=dataset.transformer
    )

    assert fit_linear.galaxy_model_visibilities_dict[
        lens_galaxy_linear
    ].array == pytest.approx(lens_galaxy_visibilities.array, 1.0e-4)

    source_galaxy_visibilities = source_galaxy.visibilities_from(
        grid=traced_grid_2d_list[1], transformer=dataset.transformer
    )

    assert fit_linear.galaxy_model_visibilities_dict[
        source_galaxy_linear
    ].array == pytest.approx(source_galaxy_visibilities.array, 1.0e-4)


def test__simulate_interferometer_data_and_fit__linear_light_profiles_and_pixelization():
    grid = al.Grid2D.uniform(
        shape_native=(51, 51), pixel_scales=0.1, over_sample_size=1
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.Sersic(centre=(0.1, 0.1), intensity=100.0),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=1.0),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        bulge=al.lp.Sersic(centre=(0.1, 0.1), intensity=0.1, sersic_index=1.0),
        disk=al.lp.Sersic(centre=(0.1, 0.1), intensity=0.2, sersic_index=4.0),
    )

    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    simulator = al.SimulatorInterferometer(
        uv_wavelengths=np.array(
            [
                [0.04, 200.0, 0.3, 400000.0, 60000000.0],
                [0.00003, 500.0, 600000.0, 0.1, 75555555],
            ]
        ),
        transformer_class=al.TransformerDFT,
        exposure_time=300.0,
        noise_if_add_noise_false=1.0,
        noise_sigma=None,
    )

    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

    lens_galaxy_linear = al.Galaxy(
        redshift=0.5,
        light=al.lp_linear.Sersic(centre=(0.1, 0.1)),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=1.0),
    )

    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=0.01),
    )

    source_galaxy_pix = al.Galaxy(redshift=1.0, pixelization=pixelization)

    tracer_linear = al.Tracer(galaxies=[lens_galaxy_linear, source_galaxy_pix])

    fit_linear = al.FitInterferometer(
        dataset=dataset,
        tracer=tracer_linear,
    )

    assert fit_linear.inversion.reconstruction == pytest.approx(
        np.array(
            [
                101.76664331,
                0.49639672,
                0.49531196,
                0.49854243,
                0.44661417,
                0.44782337,
                0.44844437,
                0.39942579,
                0.40320996,
                0.40104302,
            ]
        ),
        1.0e-2,
    )
    assert fit_linear.figure_of_merit == pytest.approx(-29.223696823166, 1.0e-4)
