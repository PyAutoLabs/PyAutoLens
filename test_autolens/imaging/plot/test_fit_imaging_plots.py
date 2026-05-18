from pathlib import Path

import pytest

from autolens.imaging.plot.fit_imaging_plots import (
    subplot_fit,
    subplot_fit_log10,
    subplot_fit_x1_plane,
    subplot_fit_log10_x1_plane,
    subplot_of_planes,
    subplot_tracer_from_fit,
    subplot_fit_combined,
    subplot_fit_combined_log10,
)

directory = Path(__file__).resolve().parent


@pytest.fixture(name="plot_path")
def make_fit_imaging_plotter_setup():
    return directory / "files" / "plots" / "fit"


def test__subplot_fit__two_plane_tracer__output_file_created(
    fit_imaging_x2_plane_7x7, plot_path, plot_patch
):
    subplot_fit(
        fit=fit_imaging_x2_plane_7x7,
        output_path=plot_path,
        output_format="png",
    )
    assert str(plot_path / "fit.png") in plot_patch.paths


def test__subplot_fit__inversion_source__mid_zoom_panel_renders(
    fit_imaging_x2_plane_inversion_7x7, plot_path, plot_patch
):
    subplot_fit(
        fit=fit_imaging_x2_plane_inversion_7x7,
        output_path=plot_path,
        output_format="png",
    )
    assert str(plot_path / "fit.png") in plot_patch.paths


def test__subplot_fit__single_plane_tracer__delegates_to_x1_plane_and_creates_file(
    fit_imaging_x1_plane_7x7, plot_path, plot_patch
):
    subplot_fit(
        fit=fit_imaging_x1_plane_7x7,
        output_path=plot_path,
        output_format="png",
    )
    assert str(plot_path / "fit_x1_plane.png") in plot_patch.paths


def test__subplot_fit_x1_plane__single_plane_tracer__output_file_created(
    fit_imaging_x1_plane_7x7, plot_path, plot_patch
):
    subplot_fit_x1_plane(
        fit=fit_imaging_x1_plane_7x7,
        output_path=plot_path,
        output_format="png",
    )
    assert str(plot_path / "fit_x1_plane.png") in plot_patch.paths


def test__subplot_fit_log10__two_plane_tracer__output_file_created(
    fit_imaging_x2_plane_7x7, plot_path, plot_patch
):
    subplot_fit_log10(
        fit=fit_imaging_x2_plane_7x7,
        output_path=plot_path,
        output_format="png",
    )
    assert str(plot_path / "fit_log10.png") in plot_patch.paths


def test__subplot_fit_log10__single_plane_tracer__delegates_to_x1_plane_and_creates_file(
    fit_imaging_x1_plane_7x7, plot_path, plot_patch
):
    subplot_fit_log10(
        fit=fit_imaging_x1_plane_7x7,
        output_path=plot_path,
        output_format="png",
    )
    assert str(plot_path / "fit_log10.png") in plot_patch.paths


def test__subplot_fit_log10_x1_plane__single_plane_tracer__output_file_created(
    fit_imaging_x1_plane_7x7, plot_path, plot_patch
):
    subplot_fit_log10_x1_plane(
        fit=fit_imaging_x1_plane_7x7,
        output_path=plot_path,
        output_format="png",
    )
    assert str(plot_path / "fit_log10.png") in plot_patch.paths


def test__subplot_of_planes__no_plane_index_specified__all_plane_files_created(
    fit_imaging_x2_plane_7x7, plot_path, plot_patch
):
    subplot_of_planes(
        fit=fit_imaging_x2_plane_7x7,
        output_path=plot_path,
        output_format="png",
    )

    assert str(plot_path / "fit_of_plane_0.png") in plot_patch.paths
    assert str(plot_path / "fit_of_plane_1.png") in plot_patch.paths


def test__subplot_of_planes__plane_index_0_specified__only_plane_0_file_created(
    fit_imaging_x2_plane_7x7, plot_path, plot_patch
):
    subplot_of_planes(
        fit=fit_imaging_x2_plane_7x7,
        output_path=plot_path,
        output_format="png",
        plane_index=0,
    )

    assert str(plot_path / "fit_of_plane_0.png") in plot_patch.paths
    assert str(plot_path / "fit_of_plane_1.png") not in plot_patch.paths


def test__subplot_tracer_from_fit__two_plane_tracer__output_file_created(
    fit_imaging_x2_plane_7x7, plot_path, plot_patch
):
    subplot_tracer_from_fit(
        fit=fit_imaging_x2_plane_7x7,
        output_path=plot_path,
        output_format="png",
    )
    assert str(plot_path / "tracer.png") in plot_patch.paths


def test__subplot_fit_combined__list_of_two_fits__output_file_created(
    fit_imaging_x1_plane_7x7, fit_imaging_x2_plane_7x7, plot_path, plot_patch
):
    subplot_fit_combined(
        fit_list=[fit_imaging_x1_plane_7x7, fit_imaging_x2_plane_7x7],
        output_path=plot_path,
        output_format="png",
    )
    assert str(plot_path / "fit_combined.png") in plot_patch.paths


def test__subplot_fit_combined_log10__list_of_two_fits__output_file_created(
    fit_imaging_x1_plane_7x7, fit_imaging_x2_plane_7x7, plot_path, plot_patch
):
    subplot_fit_combined_log10(
        fit_list=[fit_imaging_x1_plane_7x7, fit_imaging_x2_plane_7x7],
        output_path=plot_path,
        output_format="png",
    )
    assert str(plot_path / "fit_combined_log10.png") in plot_patch.paths
