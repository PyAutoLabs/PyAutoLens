import os
from pathlib import Path
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

    file_path = Path(__file__).resolve().parent / "data_temp" / "simulate_and_fit"

    try:
        shutil.rmtree(file_path)
    except FileNotFoundError:
        pass

    if not file_path.exists():
        os.makedirs(file_path)

    from autoarray.dataset.plot.imaging_plots import fits_imaging

    fits_imaging(
        dataset=dataset,
        data_path=file_path / "data.fits",
        noise_map_path=file_path / "noise_map.fits",
        psf_path=file_path / "psf.fits",
        overwrite=True,
    )

    dataset = al.Imaging.from_fits(
        data_path=file_path / "data.fits",
        noise_map_path=file_path / "noise_map.fits",
        psf_path=file_path / "psf.fits",
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

    file_path = Path(__file__).resolve().parent / "data_temp"

    if file_path.exists():
        shutil.rmtree(file_path)


def _perfect_lens_fit_dataset(tracer, grid):
    """Helper: simulate noiseless imaging through a tracer and unit noise map."""
    psf = al.Convolver.from_gaussian(
        shape_native=(3, 3), pixel_scales=grid.pixel_scales[0], sigma=0.05, normalize=True
    )
    simulator = al.SimulatorImaging(
        exposure_time=300.0, psf=psf, add_poisson_noise_to_data=False
    )
    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)
    dataset.noise_map = al.Array2D.ones(
        shape_native=dataset.data.shape_native, pixel_scales=grid.pixel_scales
    )
    return dataset


def test__perfect_fit__sim_offset_lens_and_source__fit_with_dataset_model_grid_offset__chi_squared_zero():
    """Sim a lens system shifted away from origin; fit with origin-centred profiles +
    DatasetModel.grid_offset."""
    grid = al.Grid2D.uniform(shape_native=(31, 31), pixel_scales=0.2, over_sample_size=1)
    offset = (0.3, 0.2)

    lens_sim = al.Galaxy(
        redshift=0.5,
        light=al.lp.Sersic(centre=offset, intensity=0.1, effective_radius=0.3),
        mass=al.mp.Isothermal(centre=offset, einstein_radius=1.0),
    )
    src_sim = al.Galaxy(
        redshift=1.0,
        light=al.lp.Sersic(
            centre=(offset[0] + 0.05, offset[1] + 0.05),
            intensity=0.5,
            effective_radius=0.3,
        ),
    )
    tracer_sim = al.Tracer(galaxies=[lens_sim, src_sim])
    dataset = _perfect_lens_fit_dataset(tracer_sim, grid)
    mask = al.Mask2D.circular(
        shape_native=dataset.data.shape_native, pixel_scales=0.2, radius=2.5
    )
    masked = dataset.apply_mask(mask=mask)

    lens_fit = al.Galaxy(
        redshift=0.5,
        light=al.lp.Sersic(centre=(0.0, 0.0), intensity=0.1, effective_radius=0.3),
        mass=al.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.0),
    )
    src_fit = al.Galaxy(
        redshift=1.0,
        light=al.lp.Sersic(
            centre=(0.05, 0.05), intensity=0.5, effective_radius=0.3
        ),
    )
    tracer_fit = al.Tracer(galaxies=[lens_fit, src_fit])
    dataset_model = al.DatasetModel(grid_offset=offset)
    fit = al.FitImaging(
        dataset=masked, tracer=tracer_fit, dataset_model=dataset_model
    )

    assert fit.chi_squared == pytest.approx(0.0, abs=1e-4)


def test__perfect_fit__sim_rotated_lens_mass__fit_with_dataset_model_grid_rotation__chi_squared_zero():
    """Sim a strong lens with a rotated mass ellipse; fit with axis-aligned mass +
    DatasetModel.grid_rotation_angle. The source centre is pre-rotated by -theta about
    the origin to compensate for the grid rotation."""
    import numpy as np

    grid = al.Grid2D.uniform(shape_native=(51, 51), pixel_scales=0.1, over_sample_size=1)
    theta = 10.0
    src_centre = (0.05, 0.05)

    lens_sim = al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0),
            einstein_radius=1.2,
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=theta),
        ),
    )
    src_sim = al.Galaxy(
        redshift=1.0,
        light=al.lp.SersicCore(
            centre=src_centre,
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=theta + 30.0),
            intensity=0.5,
            effective_radius=0.1,
        ),
    )
    tracer_sim = al.Tracer(galaxies=[lens_sim, src_sim])
    dataset = _perfect_lens_fit_dataset(tracer_sim, grid)
    mask = al.Mask2D.circular(
        shape_native=dataset.data.shape_native, pixel_scales=0.1, radius=2.0
    )
    masked = dataset.apply_mask(mask=mask)

    # Rotate the source centre by -theta about the origin (compensating for the
    # grid rotation by +theta).
    cos_t = np.cos(-np.deg2rad(theta))
    sin_t = np.sin(-np.deg2rad(theta))
    src_centre_rotated = (
        src_centre[1] * sin_t + src_centre[0] * cos_t,
        src_centre[1] * cos_t - src_centre[0] * sin_t,
    )

    lens_fit = al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0),
            einstein_radius=1.2,
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=0.0),
        ),
    )
    src_fit = al.Galaxy(
        redshift=1.0,
        light=al.lp.SersicCore(
            centre=src_centre_rotated,
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=30.0),
            intensity=0.5,
            effective_radius=0.1,
        ),
    )
    tracer_fit = al.Tracer(galaxies=[lens_fit, src_fit])
    dataset_model = al.DatasetModel(grid_rotation_angle=-theta)
    fit = al.FitImaging(
        dataset=masked, tracer=tracer_fit, dataset_model=dataset_model
    )

    assert fit.chi_squared == pytest.approx(0.0, abs=1e-4)


def test__perfect_fit__sim_offset_and_rotated_lens__fit_with_dataset_model_offset_and_rotation__chi_squared_zero():
    """Combined offset + rotation for the strong-lens use case relevant to Hannah's
    multi-band JWST fits: sim with shifted-and-rotated lens, fit at identity profiles +
    DatasetModel carrying both transforms."""
    import numpy as np

    grid = al.Grid2D.uniform(shape_native=(51, 51), pixel_scales=0.1, over_sample_size=1)
    offset = (0.2, 0.1)
    theta = 8.0

    # In the simulated data frame: profiles have centres at (cy+oy, cx+ox) where
    # the (cy, cx) is the "model frame" centre rotated by +theta about the offset.
    cos_t = np.cos(np.deg2rad(theta))
    sin_t = np.sin(np.deg2rad(theta))

    def to_sim_frame(model_centre):
        # Rotate by +theta about origin, then add offset.
        y_rot = model_centre[1] * sin_t + model_centre[0] * cos_t
        x_rot = model_centre[1] * cos_t - model_centre[0] * sin_t
        return (y_rot + offset[0], x_rot + offset[1])

    model_lens_centre = (0.0, 0.0)
    model_src_centre = (0.05, 0.05)

    lens_sim = al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(
            centre=to_sim_frame(model_lens_centre),
            einstein_radius=1.0,
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=theta),
        ),
    )
    src_sim = al.Galaxy(
        redshift=1.0,
        light=al.lp.SersicCore(
            centre=to_sim_frame(model_src_centre),
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=theta + 20.0),
            intensity=0.4,
            effective_radius=0.1,
        ),
    )
    tracer_sim = al.Tracer(galaxies=[lens_sim, src_sim])
    dataset = _perfect_lens_fit_dataset(tracer_sim, grid)
    mask = al.Mask2D.circular(
        shape_native=dataset.data.shape_native, pixel_scales=0.1, radius=2.0
    )
    masked = dataset.apply_mask(mask=mask)

    lens_fit = al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(
            centre=model_lens_centre,
            einstein_radius=1.0,
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=0.0),
        ),
    )
    src_fit = al.Galaxy(
        redshift=1.0,
        light=al.lp.SersicCore(
            centre=model_src_centre,
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=20.0),
            intensity=0.4,
            effective_radius=0.1,
        ),
    )
    tracer_fit = al.Tracer(galaxies=[lens_fit, src_fit])
    dataset_model = al.DatasetModel(grid_offset=offset, grid_rotation_angle=-theta)
    fit = al.FitImaging(
        dataset=masked, tracer=tracer_fit, dataset_model=dataset_model
    )

    assert fit.chi_squared == pytest.approx(0.0, abs=1e-4)


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

    file_path = Path(__file__).resolve().parent / "data_temp" / "simulate_and_fit"

    try:
        shutil.rmtree(file_path)
    except FileNotFoundError:
        pass

    if not file_path.exists():
        os.makedirs(file_path)

    from autoarray.dataset.plot.imaging_plots import fits_imaging

    fits_imaging(
        dataset=dataset,
        data_path=file_path / "data.fits",
        noise_map_path=file_path / "noise_map.fits",
        psf_path=file_path / "psf.fits",
        overwrite=True,
    )

    dataset = al.Imaging.from_fits(
        data_path=file_path / "data.fits",
        noise_map_path=file_path / "noise_map.fits",
        psf_path=file_path / "psf.fits",
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

    file_path = Path(__file__).resolve().parent / "data_temp"

    if file_path.exists():
        shutil.rmtree(file_path)