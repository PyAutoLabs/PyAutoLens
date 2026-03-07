import autolens as al

from autogalaxy.fixtures import *


def make_grid_2d_7x7():
    return aa.Grid2D.from_mask(mask=make_mask_2d_7x7(), over_sample_size=1)


def make_positions_x2():
    return al.Grid2DIrregular(values=[(1.0, 1.0), (2.0, 2.0)])


def make_positions_noise_map_x2():
    return al.ArrayIrregular(values=[1.0, 1.0])


def make_fluxes_x2():
    return al.ArrayIrregular(values=[1.0, 2.0])


def make_fluxes_noise_map_x2():
    return al.ArrayIrregular(values=[1.0, 1.0])


def make_point_dataset():
    return al.PointDataset(
        name="point_0",
        positions=make_positions_x2(),
        positions_noise_map=make_positions_noise_map_x2(),
        fluxes=make_fluxes_x2(),
        fluxes_noise_map=make_fluxes_noise_map_x2(),
    )


def make_solver():
    grid = al.Grid2D.uniform(shape_native=(10, 10), pixel_scales=0.5)
    return al.PointSolver.for_grid(
        grid=grid, pixel_scale_precision=0.25, magnification_threshold=1e-8
    )


def make_tracer_x1_plane_7x7():
    return al.Tracer(galaxies=[make_gal_x1_lp()])


def make_tracer_x2_plane_7x7():
    source_gal_x1_lp = al.Galaxy(redshift=1.0, light_profile_0=make_lp_0())

    return al.Tracer(galaxies=[make_gal_x1_mp(), make_gal_x1_lp(), source_gal_x1_lp])


def make_tracer_x2_plane_inversion_7x7():
    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(), regularization=al.reg.Constant()
    )

    source_gal_inversion = al.Galaxy(redshift=1.0, pixelization=pixelization)

    return al.Tracer(
        galaxies=[make_gal_x1_mp(), make_gal_x1_lp(), source_gal_inversion]
    )


def make_tracer_x2_plane_point():
    source_gal_x1_lp = al.Galaxy(redshift=1.0, point_0=al.ps.PointFlux())

    return al.Tracer(galaxies=[make_gal_x1_mp(), make_gal_x1_lp(), source_gal_x1_lp])


def make_fit_imaging_x1_plane_7x7():
    return al.FitImaging(
        dataset=make_masked_imaging_7x7(), tracer=make_tracer_x1_plane_7x7()
    )


def make_fit_imaging_x2_plane_7x7():
    return al.FitImaging(
        dataset=make_masked_imaging_7x7(), tracer=make_tracer_x2_plane_7x7()
    )


def make_fit_imaging_x2_plane_inversion_7x7():
    return al.FitImaging(
        dataset=make_masked_imaging_7x7(), tracer=make_tracer_x2_plane_inversion_7x7()
    )


def make_fit_interferometer_x1_plane_7x7():
    return al.FitInterferometer(
        dataset=make_interferometer_7(),
        tracer=make_tracer_x1_plane_7x7(),
    )


def make_fit_interferometer_x2_plane_7x7():
    return al.FitInterferometer(
        dataset=make_interferometer_7(),
        tracer=make_tracer_x2_plane_7x7(),
    )


def make_fit_interferometer_x2_plane_inversion_7x7():
    return al.FitInterferometer(
        dataset=make_interferometer_7(),
        tracer=make_tracer_x2_plane_inversion_7x7(),
    )


def make_fit_point_dataset_x2_plane():
    return al.FitPointDataset(
        dataset=make_point_dataset(),
        tracer=make_tracer_x2_plane_point(),
        solver=make_solver(),
    )


def make_adapt_galaxy_name_image_dict_7x7():
    image_0 = ag.Array2D(
        np.full(fill_value=2.0, shape=make_mask_2d_7x7().pixels_in_mask),
        mask=make_mask_2d_7x7(),
    )

    image_1 = ag.Array2D(
        np.full(fill_value=3.0, shape=make_mask_2d_7x7().pixels_in_mask),
        mask=make_mask_2d_7x7(),
    )

    adapt_galaxy_name_image_dict = {
        "('galaxies', 'lens')": image_0,
        "('galaxies', 'source')": image_1,
    }

    return adapt_galaxy_name_image_dict


def make_adapt_galaxy_name_image_plane_mesh_grid_dict_7x7():
    image_plane_mesh_grid_0 = ag.Grid2DIrregular(
        values=[(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
    )

    image_plane_mesh_grid_1 = ag.Grid2DIrregular(
        values=[(3.0, 3.0), (4.0, 4.0), (5.0, 5.0)]
    )

    adapt_galaxy_name_image_plane_mesh_grid_dict = {
        str(("galaxies", "lens")): image_plane_mesh_grid_0,
        str(("galaxies", "source")): image_plane_mesh_grid_1,
    }

    return adapt_galaxy_name_image_plane_mesh_grid_dict


def make_adapt_images_7x7():
    return ag.AdaptImages(
        galaxy_name_image_dict=make_adapt_galaxy_name_image_dict_7x7(),
        galaxy_name_image_plane_mesh_grid_dict=make_adapt_galaxy_name_image_plane_mesh_grid_dict_7x7(),
    )


def make_analysis_imaging_7x7():
    analysis = al.AnalysisImaging(
        dataset=make_masked_imaging_7x7(),
        use_jax=False,
        adapt_images=make_adapt_images_7x7(),
    )
    return analysis


def make_analysis_interferometer_7():
    analysis = al.AnalysisInterferometer(
        dataset=make_interferometer_7(),
        adapt_images=make_adapt_images_7x7(),
        use_jax=False,
    )
    return analysis


def make_analysis_point_x2():
    return al.AnalysisPoint(
        point_dict=make_point_dict(),
        solver=al.m.MockPointSolver(model_positions=make_positions_x2()),
        use_jax=False,
    )
