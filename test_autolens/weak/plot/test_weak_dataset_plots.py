from pathlib import Path

import autoarray as aa
import autolens as al

import pytest

from autolens.weak.plot.weak_dataset_plots import (
    plot_shear_yx_2d,
    plot_ellipticities,
    plot_phis,
    plot_noise_map,
    subplot_weak_dataset,
)

directory = Path(__file__).resolve().parent


def _isothermal_tracer():
    lens = al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.6),
    )
    source = al.Galaxy(redshift=1.0)
    return al.Tracer(galaxies=[lens, source])


@pytest.fixture(name="weak_dataset")
def make_weak_dataset():
    """Deterministic 4-galaxy WeakDataset built from an Isothermal lens."""
    grid = aa.Grid2DIrregular(
        values=[(0.7, 0.5), (1.0, 1.0), (-0.3, 0.6), (-1.1, -0.8)]
    )
    simulator = al.SimulatorShearYX(noise_sigma=0.0, seed=0)
    return simulator.via_tracer_from(
        tracer=_isothermal_tracer(), grid=grid, name="test"
    )


@pytest.fixture(name="plot_path")
def make_plot_path():
    return directory / "files" / "plots" / "weak_dataset"


def test__plot_shear_yx_2d(weak_dataset, plot_path, plot_patch):
    plot_shear_yx_2d(
        shear_yx=weak_dataset.shear_yx,
        output_path=plot_path,
        output_filename="shear_yx",
        output_format="png",
    )
    assert str(plot_path / "shear_yx.png") in plot_patch.paths


def test__plot_ellipticities(weak_dataset, plot_path, plot_patch):
    plot_ellipticities(
        shear_yx=weak_dataset.shear_yx,
        output_path=plot_path,
        output_filename="shear_ellipticities",
        output_format="png",
    )
    assert str(plot_path / "shear_ellipticities.png") in plot_patch.paths


def test__plot_phis(weak_dataset, plot_path, plot_patch):
    plot_phis(
        shear_yx=weak_dataset.shear_yx,
        output_path=plot_path,
        output_filename="shear_phis",
        output_format="png",
    )
    assert str(plot_path / "shear_phis.png") in plot_patch.paths


def test__plot_noise_map(weak_dataset, plot_path, plot_patch):
    plot_noise_map(
        dataset=weak_dataset,
        output_path=plot_path,
        output_filename="noise_map",
        output_format="png",
    )
    assert str(plot_path / "noise_map.png") in plot_patch.paths


def test__subplot_weak_dataset(weak_dataset, plot_path, plot_patch):
    subplot_weak_dataset(
        dataset=weak_dataset,
        output_path=plot_path,
        output_filename="subplot_weak_dataset",
        output_format="png",
    )
    assert str(plot_path / "subplot_weak_dataset.png") in plot_patch.paths
