from pathlib import Path

import numpy as np
import pytest

import autoarray as aa
import autolens as al

from autolens.cluster.plot.cluster_plots import (
    plot_positions_overlay,
    plot_image_group_zooms,
    plot_critical_curves,
    plot_caustics,
    subplot_cluster_dataset,
)

directory = Path(__file__).resolve().parent


@pytest.fixture(name="multi_plane_tracer")
def make_multi_plane_tracer():
    """A minimal cluster-like multi-plane system: one lens, sources at z=1 and z=2."""
    lens = al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.6),
    )
    source_0 = al.Galaxy(redshift=1.0)
    source_1 = al.Galaxy(redshift=2.0)
    return al.Tracer(galaxies=[lens, source_0, source_1])


@pytest.fixture(name="grid")
def make_grid():
    return aa.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.1)


@pytest.fixture(name="image")
def make_image():
    values = np.random.default_rng(seed=1).uniform(low=0.01, high=1.0, size=(50, 50))
    return aa.Array2D.no_mask(values=values, pixel_scales=0.2)


@pytest.fixture(name="positions_list")
def make_positions_list():
    return [
        aa.Grid2DIrregular(values=[(1.5, 0.1), (-1.4, -0.2), (0.2, 1.6)]),
        aa.Grid2DIrregular(values=[(2.1, 0.4), (-1.9, -0.5)]),
    ]


@pytest.fixture(name="plot_path")
def make_plot_path():
    return directory / "files" / "plots" / "cluster"


def test__plot_positions_overlay(positions_list, image, plot_path, plot_patch):
    plot_positions_overlay(
        positions_list,
        image=image,
        centres=aa.Grid2DIrregular(values=[(0.0, 0.0)]),
        halo_centres=aa.Grid2DIrregular(values=[(0.1, 0.1)]),
        output_path=plot_path,
        output_format="png",
    )

    assert str(plot_path / "cluster_positions.png") in plot_patch.paths


def test__plot_image_group_zooms(positions_list, image, plot_path, plot_patch):
    plot_image_group_zooms(
        positions_list,
        image=image,
        zoom_arcsec=3.0,
        output_path=plot_path,
        output_format="png",
    )

    assert str(plot_path / "cluster_image_zooms.png") in plot_patch.paths


def test__plot_critical_curves__per_plane(
    multi_plane_tracer, grid, plot_path, plot_patch
):
    plot_critical_curves(
        multi_plane_tracer,
        grid=grid,
        include_radial=True,
        output_path=plot_path,
        output_format="png",
    )

    assert str(plot_path / "cluster_critical_curves.png") in plot_patch.paths


def test__plot_caustics__per_plane(multi_plane_tracer, grid, plot_path, plot_patch):
    plot_caustics(
        multi_plane_tracer,
        grid=grid,
        output_path=plot_path,
        output_format="png",
    )

    assert str(plot_path / "cluster_caustics.png") in plot_patch.paths


def test__per_plane_critical_curves_differ(multi_plane_tracer, grid):
    # The z=2 plane sees a stronger lens (larger D_LS / D_S) than the z=1 plane, so
    # its tangential critical curve must sit at a larger radius — the physical reason
    # per-plane curves are drawn at all.
    from autolens.cluster.plot.cluster_plots import _lens_calc_for_plane

    curve_z1 = _lens_calc_for_plane(
        multi_plane_tracer, plane_index=1
    ).tangential_critical_curve_list_from(grid=grid)[0]
    curve_z2 = _lens_calc_for_plane(
        multi_plane_tracer, plane_index=2
    ).tangential_critical_curve_list_from(grid=grid)[0]

    radius_z1 = np.median(np.linalg.norm(np.asarray(curve_z1.array), axis=1))
    radius_z2 = np.median(np.linalg.norm(np.asarray(curve_z2.array), axis=1))

    assert radius_z2 > radius_z1 * 1.05


def test__subplot_cluster_dataset(
    multi_plane_tracer, grid, positions_list, image, plot_path, plot_patch
):
    subplot_cluster_dataset(
        positions_list,
        image=image,
        tracer=multi_plane_tracer,
        grid=grid,
        output_path=plot_path,
        output_format="png",
    )

    assert str(plot_path / "subplot_cluster_dataset.png") in plot_patch.paths
