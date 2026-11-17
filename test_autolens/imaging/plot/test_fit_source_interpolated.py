"""
`fit.png` source panels with the likelihood's interpolant bottom right
(`plots.yaml` `fit: subplot_fit_source_interpolated`, default on, PyAutoArray#700).
"""

import matplotlib.pyplot as plt
import pytest

import autolens as al
from autolens import fixtures
from autolens.imaging.plot import fit_imaging_plots
from autolens.interferometer.plot import fit_interferometer_plots


def _delaunay_tracer_and_adapt(mask):
    lens = al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=1.0),
    )
    source = al.Galaxy(
        redshift=1.0,
        pixelization=al.Pixelization(
            mesh=al.mesh.Delaunay(pixels=9, zeroed_pixels=0),
            regularization=al.reg.Constant(coefficient=0.01),
        ),
    )
    image_plane_mesh_grid = al.image_mesh.Overlay(shape=(3, 3)).image_plane_mesh_grid_from(
        mask=mask,
    )
    adapt_images = al.AdaptImages(
        galaxy_image_plane_mesh_grid_dict={source: image_plane_mesh_grid},
    )
    return al.Tracer(galaxies=[lens, source]), adapt_images


@pytest.fixture(name="fit_imaging_delaunay")
def make_fit_imaging_delaunay():
    dataset = fixtures.make_masked_imaging_7x7()
    tracer, adapt_images = _delaunay_tracer_and_adapt(dataset.mask)
    return al.FitImaging(dataset=dataset, tracer=tracer, adapt_images=adapt_images)


@pytest.fixture(name="fit_interferometer_delaunay")
def make_fit_interferometer_delaunay():
    dataset = fixtures.make_interferometer_7()
    tracer, adapt_images = _delaunay_tracer_and_adapt(dataset.real_space_mask)
    return al.FitInterferometer(dataset=dataset, tracer=tracer, adapt_images=adapt_images)


def _titles(module, monkeypatch, subplot_fit, fit, **kwargs):
    captured = {}

    def _capture(fig, *args, **kw):
        captured["axes"] = {
            ax.get_title(): ax for ax in fig.axes if ax.get_title()
        }
        plt.close(fig)

    monkeypatch.setattr(module, "save_figure", _capture)
    subplot_fit(fit, **kwargs)
    return captured["axes"]


def test__imaging__default__max_zoom__no_zoom__interpolated_bottom_right(
    fit_imaging_delaunay, monkeypatch
):
    captured = {}

    def _capture(fig, *args, **kw):
        captured["fig"] = fig

    monkeypatch.setattr(fit_imaging_plots, "save_figure", _capture)
    fit_imaging_plots.subplot_fit(fit_imaging_delaunay)
    fig = captured["fig"]
    titles = {ax.get_title(): ax for ax in fig.axes if ax.get_title()}

    assert "Source Plane (Mid Zoom)" not in titles
    max_zoom = titles["Source Plane (Max Zoom)"]
    no_zoom = titles["Source Plane (No Zoom)"]
    interpolated = titles["Source Plane (Interpolated)"]

    # Right-hand column, top to bottom: max zoom, no zoom, interpolated.
    def _y(ax):
        return ax.get_position().y0

    assert _y(max_zoom) > _y(no_zoom) > _y(interpolated)
    assert max_zoom.get_position().x0 == pytest.approx(interpolated.get_position().x0)

    assert len(max_zoom.get_images()) == 0
    assert len(no_zoom.get_images()) == 0
    assert len(interpolated.get_images()) == 1
    # The interpolated panel covers the max-zoom extent, for a direct comparison.
    assert interpolated.get_xlim() == pytest.approx(max_zoom.get_xlim())
    plt.close(fig)


def test__imaging__switch_off__max_mid_no_zoom(fit_imaging_delaunay, monkeypatch):
    axes = _titles(
        fit_imaging_plots, monkeypatch, fit_imaging_plots.subplot_fit,
        fit_imaging_delaunay, source_interpolated=False,
    )
    assert "Source Plane (Max Zoom)" in axes
    assert "Source Plane (Mid Zoom)" in axes
    assert "Source Plane (No Zoom)" in axes
    assert "Source Plane (Interpolated)" not in axes


def test__imaging__rectangular_source__keeps_max_mid_no_zoom(monkeypatch):
    axes = _titles(
        fit_imaging_plots, monkeypatch, fit_imaging_plots.subplot_fit,
        fixtures.make_fit_imaging_x2_plane_inversion_7x7(),
    )
    assert "Source Plane (Max Zoom)" in axes
    assert "Source Plane (Mid Zoom)" in axes
    assert "Source Plane (No Zoom)" in axes
    assert "Source Plane (Interpolated)" not in axes


def test__interferometer__default__zoomed_then_interpolated(
    fit_interferometer_delaunay, monkeypatch
):
    on = _titles(
        fit_interferometer_plots, monkeypatch, fit_interferometer_plots.subplot_fit,
        fit_interferometer_delaunay,
    )
    assert len(on["Source Plane (Zoomed)"].get_images()) == 0
    assert len(on["Source Plane (Interpolated)"].get_images()) == 1
    assert "Source Plane (No Zoom)" not in on

    off = _titles(
        fit_interferometer_plots, monkeypatch, fit_interferometer_plots.subplot_fit,
        fit_interferometer_delaunay, source_interpolated=False,
    )
    assert "Source Plane (Zoomed)" in off
    assert "Source Plane (No Zoom)" in off
    assert "Source Plane (Interpolated)" not in off


def test__plots_yaml__switch_defaults_on():
    from autolens.analysis.plotter import plot_setting

    assert plot_setting(section=["fit", "fit_imaging"], name="subplot_fit_source_interpolated") is True
    assert plot_setting(section=["fit", "fit_interferometer"], name="subplot_fit_source_interpolated") is True
