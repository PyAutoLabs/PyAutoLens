from pathlib import Path

import pytest

import autolens.plot as aplt
from autolens.lens.plot.tracer_plots import (
    subplot_tracer,
    subplot_lensed_images,
    subplot_galaxies_images,
    fits_tracer,
    fits_source_plane_images,
)

directory = Path(__file__).resolve().parent


@pytest.fixture(name="plot_path")
def make_tracer_plotter_setup():
    return directory / "files" / "plots" / "tracer"


def test__subplot_tracer__two_plane_tracer__output_file_created(
    tracer_x2_plane_7x7, grid_2d_7x7, plot_path, plot_patch
):
    subplot_tracer(
        tracer=tracer_x2_plane_7x7,
        grid=grid_2d_7x7,
        output_path=plot_path,
        output_format="png",
    )
    assert str(plot_path / "tracer.png") in plot_patch.paths


def test__subplot_galaxies_images__two_plane_tracer__output_file_created(
    tracer_x2_plane_7x7, grid_2d_7x7, plot_path, plot_patch
):
    subplot_galaxies_images(
        tracer=tracer_x2_plane_7x7,
        grid=grid_2d_7x7,
        output_path=plot_path,
        output_format="png",
    )
    assert str(plot_path / "galaxies_images.png") in plot_patch.paths


def test__subplot_lensed_images__two_plane_tracer__output_file_created(
    tracer_x2_plane_7x7, grid_2d_7x7, plot_path, plot_patch
):
    subplot_lensed_images(
        tracer=tracer_x2_plane_7x7,
        grid=grid_2d_7x7,
        output_path=plot_path,
        output_format="png",
    )
    assert str(plot_path / "lensed_images.png") in plot_patch.paths


def test__fits_tracer__two_plane_tracer__tracer_fits_file_written(
    tracer_x2_plane_7x7, grid_2d_7x7, tmp_path
):
    fits_tracer(
        tracer=tracer_x2_plane_7x7,
        grid=grid_2d_7x7,
        output_path=tmp_path,
    )
    assert (tmp_path / "tracer.fits").exists()


def test__fits_source_plane_images__two_plane_tracer__source_plane_fits_file_written(
    tracer_x2_plane_7x7, grid_2d_7x7, tmp_path
):
    fits_source_plane_images(
        tracer=tracer_x2_plane_7x7,
        grid=grid_2d_7x7,
        output_path=tmp_path,
    )
    assert (tmp_path / "source_plane_images.fits").exists()


def test__plot_magnification_2d__two_plane_tracer__output_file_created(
    tracer_x2_plane_7x7, grid_2d_7x7, plot_path, plot_patch
):
    aplt.plot_magnification_2d(
        mass_obj=tracer_x2_plane_7x7,
        grid=grid_2d_7x7,
        output_path=plot_path,
        output_format="png",
    )
    assert str(plot_path / "magnification_2d.png") in plot_patch.paths


def test__plot_magnification_2d__three_plane_tracer__contours_drawn():
    import matplotlib.pyplot as plt

    import autolens as al

    grid = al.Grid2D.uniform(shape_native=(40, 40), pixel_scales=0.1)
    tracer = al.Tracer(
        galaxies=[
            al.Galaxy(redshift=0.5, mass=al.mp.Isothermal(einstein_radius=1.0)),
            al.Galaxy(
                redshift=1.0,
                mass=al.mp.Isothermal(centre=(0.3, 0.0), einstein_radius=0.2),
            ),
            al.Galaxy(redshift=2.0, light=al.lp.SersicCore(intensity=1.0)),
        ]
    )

    fig, ax = plt.subplots()
    aplt.plot_magnification_2d(
        mass_obj=tracer, grid=grid, include_critical_curves=False, ax=ax
    )

    assert any(hasattr(collection, "levels") for collection in ax.collections)

    plt.close("all")


def test__plot_magnification_source_plane__mesh_of_a_fit__output_file_created(
    fit_imaging_x2_plane_inversion_7x7, plot_path, plot_patch
):
    import autoarray as aa

    fit = fit_imaging_x2_plane_inversion_7x7

    aplt.plot_magnification_source_plane(
        mapper=fit.inversion.cls_list_from(cls=aa.Mapper)[0],
        mass_obj=fit.tracer,
        grid=fit.dataset.grid,
        output_path=plot_path,
        output_format="png",
    )
    assert str(plot_path / "magnification_source_plane.png") in plot_patch.paths


def test__plot_magnification_source_plane_grid__two_plane_tracer__output_file_created(
    tracer_x2_plane_7x7, grid_2d_7x7, plot_path, plot_patch
):
    aplt.plot_magnification_source_plane_grid(
        mass_obj=tracer_x2_plane_7x7,
        grid=grid_2d_7x7,
        source_shape_native=(10, 10),
        sub_size=2,
        output_path=plot_path,
        output_format="png",
    )
    assert str(plot_path / "magnification_source_plane_grid.png") in plot_patch.paths
