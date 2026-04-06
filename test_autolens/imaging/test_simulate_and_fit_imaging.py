import os
from os import path
import shutil

import autolens as al
import numpy as np
import pytest


def test__perfect_fit__chi_squared_0():

    grid = al.Grid2D.uniform(shape_native=(11, 11), pixel_scales=0.2, over_sample_size=1)

    psf = al.Convolver.from_gaussian(
        shape_native=(3, 3), pixel_scales=0.2, sigma=0.75, normalize=True
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.Sersic(centre=(0.1, 0.1), intensity=0.1),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=1.8),
    )
    source_galaxy = al.Galaxy(
        redshift=1.0, light=al.lp.Exponential(centre=(0.1, 0.1), intensity=0.5)
    )
    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    dataset = al.SimulatorImaging(exposure_time=300.0, psf=psf, add_poisson_noise_to_data=False)

    dataset = dataset.via_tracer_from(tracer=tracer, grid=grid)
    dataset.noise_map = al.Array2D.ones(
        shape_native=dataset.data.shape_native, pixel_scales=0.2
    )

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

    from autoarray.dataset.plot.imaging_plots import fits_imaging

    fits_imaging(
        dataset=dataset,
        data_path=path.join(file_path, "data.fits"),
        noise_map_path=path.join(file_path, "noise_map.fits"),
        psf_path=path.join(file_path, "psf.fits"),
        overwrite=True,
    )

    dataset = al.Imaging.from_fits(
        data_path=path.join(file_path, "data.fits"),
        noise_map_path=path.join(file_path, "noise_map.fits"),
        psf_path=path.join(file_path, "psf.fits"),
        pixel_scales=0.2,
        over_sample_size_lp=1
    )

    mask = al.Mask2D.circular(
        shape_native=dataset.data.shape_native, pixel_scales=0.2, radius=0.8
    )

    masked_dataset = dataset.apply_mask(mask=mask)

    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    fit = al.FitImaging(dataset=masked_dataset, tracer=tracer)

    assert fit.chi_squared == pytest.approx(0.0, 1e-4)

    file_path = path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "data_temp"
    )

    if path.exists(file_path) is True:
        shutil.rmtree(file_path)



def test__simulate_imaging_data_and_fit__known_likelihood():

    grid = al.Grid2D.uniform(shape_native=(31, 31), pixel_scales=0.2)

    psf = al.Convolver.from_gaussian(
        shape_native=(3, 3), pixel_scales=0.2, sigma=0.75, normalize=True
    )

    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(16, 16)),
        regularization=al.reg.Constant(coefficient=(1.0)),
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(centre=(0.1, 0.1), intensity=0.1),
        disk=al.lp.Sersic(centre=(0.2, 0.2), intensity=0.2),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=1.8),
    )
    source_galaxy_0 = al.Galaxy(redshift=1.0, pixelization=pixelization)
    source_galaxy_1 = al.Galaxy(redshift=2.0, pixelization=pixelization)
    tracer = al.Tracer(
        galaxies=[lens_galaxy, source_galaxy_0, source_galaxy_1]
    )

    simulator = al.SimulatorImaging(exposure_time=300.0, psf=psf, noise_seed=1)

    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

    mask = al.Mask2D.circular(
        shape_native=dataset.data.shape_native, pixel_scales=0.2, radius=2.005
    )

    masked_dataset = dataset.apply_mask(mask=mask)

    fit = al.FitImaging(dataset=masked_dataset, tracer=tracer,
                        settings=al.Settings(use_border_relocator=True)
                        )

    assert fit.figure_of_merit == pytest.approx(565.6348654, 1.0e-2)


def test__simulate_imaging_data_and_fit__linear_light_profiles_agree_with_standard_light_profiles():

    grid = al.Grid2D.uniform(shape_native=(11, 11), pixel_scales=0.2, over_sample_size=1)

    psf = al.Convolver.from_gaussian(
        shape_native=(3, 3), pixel_scales=0.2, sigma=0.75, normalize=True
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.Sersic(centre=(0.1, 0.1), intensity=0.1),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=1.8),
    )
    source_galaxy = al.Galaxy(
        redshift=1.0,
        bulge=al.lp.Sersic(intensity=0.1, sersic_index=1.0),
        disk=al.lp.Sersic(intensity=0.2, sersic_index=4.0),
    )
    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    dataset = al.SimulatorImaging(exposure_time=300.0, psf=psf, add_poisson_noise_to_data=False)

    dataset = dataset.via_tracer_from(tracer=tracer, grid=grid)
    dataset.noise_map = al.Array2D.ones(
        shape_native=dataset.data.shape_native, pixel_scales=0.2
    )

    mask = al.Mask2D.circular(
        shape_native=dataset.data.shape_native, pixel_scales=0.2, radius=0.805
    )

    masked_dataset = dataset.apply_mask(mask=mask)
    masked_dataset = masked_dataset.apply_over_sampling(
        over_sample_size_lp=1
    )

    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    fit = al.FitImaging(dataset=masked_dataset, tracer=tracer)

    lens_galaxy_linear = al.Galaxy(
        redshift=0.5,
        light=al.lp_linear.Sersic(centre=(0.1, 0.1)),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=1.8),
    )
    source_galaxy_linear = al.Galaxy(
        redshift=1.0,
        bulge=al.lp_linear.Sersic(sersic_index=1.0),
        disk=al.lp_linear.Sersic(sersic_index=4.0),
    )

    tracer_linear = al.Tracer(
        galaxies=[lens_galaxy_linear, source_galaxy_linear]
    )

    fit_linear = al.FitImaging(
        dataset=masked_dataset,
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
    assert fit.log_likelihood == fit_linear.figure_of_merit
    assert fit_linear.figure_of_merit == pytest.approx(-45.02798, 1.0e-4)

    lens_galaxy_image = lens_galaxy.blurred_image_2d_from(
        grid=masked_dataset.grids.lp,
        psf=masked_dataset.psf,
        blurring_grid=masked_dataset.grids.blurring,
    )

    assert fit_linear.galaxy_model_image_dict[lens_galaxy_linear] == pytest.approx(
        lens_galaxy_image.array, 1.0e-4
    )
    assert fit_linear.model_images_of_planes_list[0] == pytest.approx(
        lens_galaxy_image.array, 1.0e-4
    )

    traced_grid_2d_list = tracer.traced_grid_2d_list_from(grid=masked_dataset.grids.lp)
    traced_blurring_grid_2d_list = tracer.traced_grid_2d_list_from(
        grid=masked_dataset.grids.blurring
    )

    source_galaxy_image = source_galaxy.blurred_image_2d_from(
        grid=traced_grid_2d_list[1],
        psf=masked_dataset.psf,
        blurring_grid=traced_blurring_grid_2d_list[1],
    )

    assert fit_linear.galaxy_model_image_dict[source_galaxy_linear] == pytest.approx(
        source_galaxy_image.array, 1.0e-4
    )

    assert fit_linear.model_images_of_planes_list[1] == pytest.approx(
        source_galaxy_image.array, 1.0e-4
    )


def test__simulate_imaging_data_and_fit__linear_light_profiles_and_pixelization():

    grid = al.Grid2D.uniform(shape_native=(11, 11), pixel_scales=0.2, over_sample_size=1)

    psf = al.Convolver.from_gaussian(
        shape_native=(3, 3), pixel_scales=0.2, sigma=0.75, normalize=True
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.Sersic(centre=(0.1, 0.1), intensity=100.0),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=1.8),
    )
    source_galaxy = al.Galaxy(
        redshift=1.0,
        bulge=al.lp.Sersic(intensity=0.1, sersic_index=1.0),
        disk=al.lp.Sersic(intensity=0.2, sersic_index=4.0),
    )
    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    dataset = al.SimulatorImaging(exposure_time=300.0, psf=psf, add_poisson_noise_to_data=False)

    dataset = dataset.via_tracer_from(tracer=tracer, grid=grid)
    dataset.noise_map = al.Array2D.ones(
        shape_native=dataset.data.shape_native, pixel_scales=0.2
    )

    mask = al.Mask2D.circular(
        shape_native=dataset.data.shape_native, pixel_scales=0.2, radius=0.805
    )

    masked_dataset = dataset.apply_mask(mask=mask)
    masked_dataset = masked_dataset.apply_over_sampling(
        over_sample_size_lp=1
    )

    lens_galaxy_linear = al.Galaxy(
        redshift=0.5,
        light=al.lp_linear.Sersic(centre=(0.1, 0.1)),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=1.8),
    )

    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=0.01),
    )

    source_galaxy_pix = al.Galaxy(redshift=1.0, pixelization=pixelization)

    tracer_linear = al.Tracer(
        galaxies=[lens_galaxy_linear, source_galaxy_pix]
    )

    fit_linear = al.FitImaging(
        dataset=masked_dataset,
        tracer=tracer_linear,
        settings=al.Settings(use_border_relocator=True)
    )

    assert fit_linear.inversion.reconstruction[0:2] == pytest.approx(
        np.array(
            [
                99.993449641, 0.114213814,
            ]
        ),
        1.0e-4,
    )
    assert fit_linear.figure_of_merit == pytest.approx(-87.6933733814, 1.0e-4)

    lens_galaxy_image = lens_galaxy.blurred_image_2d_from(
        grid=masked_dataset.grids.lp,
        psf=masked_dataset.psf,
        blurring_grid=masked_dataset.grids.blurring,
    )

    assert fit_linear.galaxy_model_image_dict[lens_galaxy_linear] == pytest.approx(
        lens_galaxy_image.array, 1.0e-2
    )
    assert fit_linear.model_images_of_planes_list[0] == pytest.approx(
        lens_galaxy_image.array, 1.0e-2
    )

    assert fit_linear.galaxy_model_image_dict[source_galaxy_pix][0] == pytest.approx(
        0.0524952137, 1.0e-4
    )

    assert fit_linear.model_images_of_planes_list[1][0] == pytest.approx(
        0.052495213, 1.0e-4
    )

    fit_linear = al.FitImaging(
        dataset=masked_dataset,
        tracer=tracer_linear,
        settings=al.Settings(
            use_positive_only_solver=True,
        ),
    )

    assert fit_linear.inversion.reconstruction == pytest.approx(
        np.array(
            [
                100.01548,
                0.0,
                0.0,
                0.0,
                0.0,
                1.13328604,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ),
        abs=1.0e-1,
    )
    assert fit_linear.figure_of_merit == pytest.approx(-85.7890126577, 1.0e-4)


def test__simulate_imaging_data_and_fit__linear_light_profiles_and_pixelization__sub_2():

    grid = al.Grid2D.uniform(shape_native=(11, 11), pixel_scales=0.2, over_sample_size=2)

    psf = al.Convolver.from_gaussian(
        shape_native=(3, 3), pixel_scales=0.2, sigma=0.75, normalize=True
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.Sersic(centre=(0.1, 0.1), intensity=100.0),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=1.8),
    )
    source_galaxy = al.Galaxy(
        redshift=1.0,
        bulge=al.lp.Sersic(intensity=0.1, sersic_index=1.0),
        disk=al.lp.Sersic(intensity=0.2, sersic_index=4.0),
    )
    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    dataset = al.SimulatorImaging(exposure_time=300.0, psf=psf, add_poisson_noise_to_data=False)

    dataset = dataset.via_tracer_from(tracer=tracer, grid=grid)
    dataset.noise_map = al.Array2D.ones(
        shape_native=dataset.data.shape_native, pixel_scales=0.2
    )

    mask = al.Mask2D.circular(
        shape_native=dataset.data.shape_native, pixel_scales=0.2, radius=0.805
    )

    dataset = al.Imaging(
        data=dataset.data,
        psf=dataset.psf,
        noise_map=dataset.noise_map,
        over_sample_size_lp=2,
        over_sample_size_pixelization=2
    )

    masked_dataset = dataset.apply_mask(mask=mask)

    lens_galaxy_linear = al.Galaxy(
        redshift=0.5,
        light=al.lp_linear.Sersic(centre=(0.1, 0.1)),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=1.8),
    )

    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=0.01),
    )

    source_galaxy_pix = al.Galaxy(redshift=1.0, pixelization=pixelization)

    tracer_linear = al.Tracer(
        galaxies=[lens_galaxy_linear, source_galaxy_pix]
    )

    fit_linear = al.FitImaging(
        dataset=masked_dataset,
        tracer=tracer_linear,
        settings=al.Settings(use_border_relocator=True)
    )

    assert fit_linear.inversion.reconstruction[0:2] == pytest.approx(
        np.array(
            [
                99.974078996,  0.251635768,
            ]
        ),
        1.0e-4,
    )

    assert fit_linear.figure_of_merit == pytest.approx(-87.9411586204183, 1.0e-4)

    lens_galaxy_image = lens_galaxy.blurred_image_2d_from(
        grid=masked_dataset.grids.lp,
        psf=masked_dataset.psf,
        blurring_grid=masked_dataset.grids.blurring,
    )

    assert fit_linear.galaxy_model_image_dict[lens_galaxy_linear] == pytest.approx(
        lens_galaxy_image.array, 1.0e-2
    )
    assert fit_linear.model_images_of_planes_list[0] == pytest.approx(
        lens_galaxy_image.array, 1.0e-2
    )

    assert fit_linear.galaxy_model_image_dict[source_galaxy_pix][0] == pytest.approx(
        0.134961296372, 1.0e-4
    )

    assert fit_linear.model_images_of_planes_list[1][0] == pytest.approx(
        0.134961296, 1.0e-4
    )

    assert fit_linear.subtracted_images_of_planes_list[1][0] == pytest.approx(
        0.34355239059, 1.0e-4
    )

    fit_linear = al.FitImaging(
        dataset=masked_dataset,
        tracer=tracer_linear,
        settings=al.Settings(
            use_positive_only_solver=True,
        ),
    )

    assert fit_linear.inversion.reconstruction == pytest.approx(
        np.array(
            [
                100.01548,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ),
        abs=1.0e-1,
    )
    assert fit_linear.figure_of_merit == pytest.approx(-86.95872610, abs=1.0e-4)


def test__simulate_imaging_data_and_fit__linear_light_profiles_and_pixelization__delaunay_split():

    grid = al.Grid2D.uniform(shape_native=(11, 11), pixel_scales=0.2, over_sample_size=2)

    psf = al.Convolver.from_gaussian(
        shape_native=(3, 3), pixel_scales=0.2, sigma=0.75, normalize=True
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.Sersic(centre=(0.1, 0.1), intensity=100.0),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=1.8),
    )
    source_galaxy = al.Galaxy(
        redshift=1.0,
        bulge=al.lp.Sersic(intensity=0.1, sersic_index=1.0),
        disk=al.lp.Sersic(intensity=0.2, sersic_index=4.0),
    )
    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    dataset = al.SimulatorImaging(exposure_time=300.0, psf=psf, add_poisson_noise_to_data=False)

    dataset = dataset.via_tracer_from(tracer=tracer, grid=grid)
    dataset.noise_map = al.Array2D.ones(
        shape_native=dataset.data.shape_native, pixel_scales=0.2
    )

    mask = al.Mask2D.circular(
        shape_native=dataset.data.shape_native, pixel_scales=0.2, radius=0.81
    )

    dataset = al.Imaging(
        data=dataset.data,
        psf=dataset.psf,
        noise_map=dataset.noise_map,
        over_sample_size_lp=2,
        over_sample_size_pixelization=2
    )

    masked_dataset = dataset.apply_mask(mask=mask)

    lens_galaxy_linear = al.Galaxy(
        redshift=0.5,
        light=al.lp_linear.Sersic(centre=(0.1, 0.1)),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=1.8),
    )

    pixelization = al.Pixelization(
        mesh=al.mesh.Delaunay(pixels=25, zeroed_pixels=5),
        regularization=al.reg.AdaptSplit(inner_coefficient=0.01, outer_coefficient=0.1, signal_scale=0.1),
    )

    source_galaxy_pix = al.Galaxy(redshift=1.0, pixelization=pixelization)

    image_mesh = al.image_mesh.Overlay(shape=(7, 7))

    image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
        mask=masked_dataset.mask,
    )

    adapt_images = al.AdaptImages(
        galaxy_image_dict={source_galaxy_pix: masked_dataset.data},
        galaxy_image_plane_mesh_grid_dict={source_galaxy_pix: image_plane_mesh_grid},
    )

    tracer_linear = al.Tracer(
        galaxies=[lens_galaxy_linear, source_galaxy_pix]
    )

    fit_linear = al.FitImaging(
        dataset=masked_dataset,
        tracer=tracer_linear,
        adapt_images=adapt_images,
        settings=al.Settings(use_positive_only_solver=False),
    )

    assert fit_linear.inversion.reconstruction[0:3] == pytest.approx(
        np.array(
            [
                1.00179579e+02,  5.35321466e-01,  8.55754143e-01
            ]
        ),
        1.0e-4,
    )

    assert fit_linear.figure_of_merit == pytest.approx(-190.564548990939, 1.0e-4)

    lens_galaxy_image = lens_galaxy.blurred_image_2d_from(
        grid=masked_dataset.grids.lp,
        blurring_grid=masked_dataset.grids.blurring,
        psf=masked_dataset.psf
    )

    assert fit_linear.galaxy_model_image_dict[lens_galaxy_linear] == pytest.approx(
        lens_galaxy_image, 1.0e-2
    )
    assert fit_linear.model_images_of_planes_list[0] == pytest.approx(
        lens_galaxy_image, 1.0e-2
    )

    assert fit_linear.galaxy_model_image_dict[source_galaxy_pix][0] == pytest.approx(
            0.1667703826, 1.0e-4
    )

    assert fit_linear.model_images_of_planes_list[1][0] == pytest.approx(
        0.166757208736973, 1.0e-4
    )

    assert fit_linear.subtracted_images_of_planes_list[1][0] == pytest.approx(
        0.180018267146, 1.0e-4
    )


    fit_linear = al.FitImaging(
        dataset=masked_dataset,
        tracer=tracer_linear,
        adapt_images=adapt_images,
        settings=al.Settings(
            use_positive_only_solver=True,
            use_edge_zeroed_pixels=True
        ),
    )

    assert fit_linear.inversion.reconstruction[0:2] == pytest.approx(
        np.array(
            [
                99.9785287998059,
                0.8958653625423
            ]
        ),
        1.0e-4,
    )
    assert fit_linear.figure_of_merit == pytest.approx(-190.6935526756, 1.0e-4)


def test__simulate_imaging_data_and_fit__complex_fit_compare_mapping_matrix_sparse_operator():

    grid = al.Grid2D.uniform(shape_native=(15, 15), pixel_scales=0.1)

    psf = al.Convolver.from_gaussian(
        shape_native=(3, 3), pixel_scales=0.2, sigma=0.75, normalize=True
    )

    lens_0 = al.Galaxy(
        redshift=0.1,
        light=al.lp.Sersic(centre=(0.1, 0.1)),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=0.2),
    )

    lens_1 = al.Galaxy(
        redshift=0.2,
        light=al.lp.Sersic(centre=(0.2, 0.2)),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=0.2),
    )

    lens_2 = al.Galaxy(
        redshift=0.3,
        light=al.lp.Sersic(centre=(0.3, 0.3)),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=0.2),
    )

    source_0 = al.Galaxy(redshift=0.4, bulge=al.lp.Sersic(centre=(0.3, 0.3)))
    source_1 = al.Galaxy(redshift=0.5, bulge=al.lp.Sersic(centre=(0.3, 0.3)))
    tracer = al.Tracer(galaxies=[lens_0, lens_1, lens_2, source_0, source_1])

    dataset = al.SimulatorImaging(exposure_time=300.0, psf=psf, add_poisson_noise_to_data=False)

    dataset = dataset.via_tracer_from(tracer=tracer, grid=grid)
    dataset.sub_size = 2
    dataset.noise_map = al.Array2D.ones(
        shape_native=dataset.data.shape_native, pixel_scales=0.2
    )
    mask = al.Mask2D.circular(
        shape_native=dataset.data.shape_native, pixel_scales=0.2, radius=0.8
    )

    masked_dataset = dataset.apply_mask(mask=mask)

    lens_0 = al.Galaxy(
        redshift=0.1,
        light=al.lp_linear.Sersic(centre=(0.1, 0.1)),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=0.2),
    )

    lens_1 = al.Galaxy(
        redshift=0.2,
        light=al.lp_linear.Sersic(centre=(0.2, 0.2)),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=0.2),
    )

    lens_2 = al.Galaxy(
        redshift=0.3,
        light=al.lp_linear.Sersic(centre=(0.3, 0.3)),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=0.2),
    )

    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )

    source_0 = al.Galaxy(redshift=0.4, pixelization=pixelization)

    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )

    source_1 = al.Galaxy(redshift=0.5, pixelization=pixelization)

    tracer = al.Tracer(
        galaxies=[lens_0, lens_1, lens_2, source_0, source_1]
    )

    fit_mapping = al.FitImaging(
        dataset=masked_dataset,
        tracer=tracer,
    )

    masked_dataset_sparse_operator = masked_dataset.apply_sparse_operator()

    fit_sparse_operator = al.FitImaging(
        dataset=masked_dataset_sparse_operator,
        tracer=tracer,
    )

    assert fit_mapping.inversion.curvature_matrix == pytest.approx(
            fit_sparse_operator.inversion.curvature_matrix,
        1.0e-4,
    )

    assert fit_mapping.inversion.regularization_matrix == pytest.approx(
            fit_sparse_operator.inversion.regularization_matrix,
        1.0e-4,
    )


def test__fit_figure_of_merit__mge_mass_model(masked_imaging_7x7, masked_imaging_covariance_7x7):

    grid = al.Grid2D.uniform(shape_native=(11, 11), pixel_scales=0.2,
                             over_sample_size=8)

    psf = al.Convolver.from_gaussian(
        shape_native=(3, 3), pixel_scales=0.2, sigma=0.75, normalize=True
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light_0=al.lp.Gaussian(intensity=1.0),
        light_1=al.lp.Gaussian(intensity=2.0),
        mass_0=al.mp.Gaussian(intensity=1.0, mass_to_light_ratio=3.0),
        mass_1=al.mp.Gaussian(intensity=2.0, mass_to_light_ratio=4.0),
    )
    source_galaxy = al.Galaxy(
        redshift=1.0, light=al.lp.Exponential(centre=(0.1, 0.1), intensity=0.5)
    )
    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    dataset = al.SimulatorImaging(exposure_time=300.0, psf=psf, add_poisson_noise_to_data=False)

    dataset = dataset.via_tracer_from(tracer=tracer, grid=grid)
    dataset.noise_map = al.Array2D.ones(
        shape_native=dataset.data.shape_native, pixel_scales=0.2
    )

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

    from autoarray.dataset.plot.imaging_plots import fits_imaging

    fits_imaging(
        dataset=dataset,
        data_path=path.join(file_path, "data.fits"),
        noise_map_path=path.join(file_path, "noise_map.fits"),
        psf_path=path.join(file_path, "psf.fits"),
        overwrite=True,
    )

    dataset = al.Imaging.from_fits(
        data_path=path.join(file_path, "data.fits"),
        noise_map_path=path.join(file_path, "noise_map.fits"),
        psf_path=path.join(file_path, "psf.fits"),
        pixel_scales=0.2,
        over_sample_size_lp=8
    )

    mask = al.Mask2D.circular(
        shape_native=dataset.data.shape_native, pixel_scales=0.2, radius=0.805
    )

    masked_dataset = dataset.apply_mask(mask=mask)

    basis = al.lp_basis.Basis(
        profile_list=[
            al.lmp.Gaussian(intensity=1.0, mass_to_light_ratio=3.0),
            al.lmp.Gaussian(intensity=2.0, mass_to_light_ratio=4.0),
        ]
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        bulge=basis,
    )
    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    fit = al.FitImaging(dataset=masked_dataset, tracer=tracer)

    # The value is actually not zero before the blurring grid assumes a sub_size=1
    # and does not use the iterative grid, which has a small impact on the chi-squared

    assert fit.chi_squared == pytest.approx(5.706423629698664e-05, 1e-4)

    masked_dataset = masked_dataset.apply_over_sampling(
        over_sample_size_lp=8
    )

    basis = al.lp_basis.Basis(
        profile_list=[
            al.lmp_linear.Gaussian(intensity=1.0, mass_to_light_ratio=3.0),
            al.lmp_linear.Gaussian(intensity=2.0, mass_to_light_ratio=4.0),
        ]
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        bulge=basis,
    )
    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    fit = al.FitImaging(dataset=masked_dataset, tracer=tracer)

    # The value is actually not zero before the blurring grid assumes a sub_size=1
    # and does not use the iterative grid, which has a small impact on the chi-squared

    assert fit.chi_squared == pytest.approx(3.295535243634485e-05, 1e-4)

    file_path = path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "data_temp"
    )

    if path.exists(file_path) is True:
        shutil.rmtree(file_path)