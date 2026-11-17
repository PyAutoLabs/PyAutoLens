"""
`fit.png` source panel drawn as the likelihood's interpolant
(`plots.yaml` `fit: subplot_fit_source_interpolated`, PyAutoArray#700).
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


def test__imaging__switch_off__mid_zoom_triangles(fit_imaging_delaunay, monkeypatch):
    axes = _titles(fit_imaging_plots, monkeypatch, fit_imaging_plots.subplot_fit, fit_imaging_delaunay)
    assert "Source Plane (Max Zoom)" in axes
    assert "Source Plane (Mid Zoom)" in axes
    assert "Source Plane (Interpolated)" not in axes


def test__imaging__switch_on__mid_zoom_is_interpolated_raster(fit_imaging_delaunay, monkeypatch):
    axes = _titles(
        fit_imaging_plots, monkeypatch, fit_imaging_plots.subplot_fit,
        fit_imaging_delaunay, source_interpolated=True,
    )
    assert "Source Plane (Mid Zoom)" not in axes
    assert "Source Plane (Max Zoom)" not in axes
    interpolated = axes["Source Plane (Interpolated)"]
    zoom = axes["Source Plane (Zoom)"]
    assert len(interpolated.get_images()) == 1
    # Triangles at a reasonable zoom and unzoomed; the interpolated panel shares
    # the zoomed panel's extent so the two compare side by side.
    assert len(zoom.get_images()) == 0
    assert len(axes["Source Plane (No Zoom)"].get_images()) == 0
    assert zoom.get_xlim() == pytest.approx(interpolated.get_xlim())
    assert zoom.get_ylim() == pytest.approx(interpolated.get_ylim())


def test__imaging__switch_on__rectangular_source_is_unchanged(monkeypatch):
    axes = _titles(
        fit_imaging_plots, monkeypatch, fit_imaging_plots.subplot_fit,
        fixtures.make_fit_imaging_x2_plane_inversion_7x7(), source_interpolated=True,
    )
    assert "Source Plane (Max Zoom)" in axes
    assert "Source Plane (Mid Zoom)" in axes
    assert "Source Plane (Interpolated)" not in axes


def test__interferometer__switch_on__zoomed_panel_is_interpolated(
    fit_interferometer_delaunay, monkeypatch
):
    off = _titles(
        fit_interferometer_plots, monkeypatch, fit_interferometer_plots.subplot_fit,
        fit_interferometer_delaunay,
    )
    assert "Source Plane (Zoomed)" in off

    on = _titles(
        fit_interferometer_plots, monkeypatch, fit_interferometer_plots.subplot_fit,
        fit_interferometer_delaunay, source_interpolated=True,
    )
    assert "Source Plane (Zoomed)" not in on
    assert len(on["Source Plane (Interpolated)"].get_images()) == 1
    assert "Source Plane (No Zoom)" in on


def test__plots_yaml__switch_defaults_off():
    from autolens.analysis.plotter import plot_setting

    assert plot_setting(section=["fit", "fit_imaging"], name="subplot_fit_source_interpolated") is False
    assert plot_setting(section=["fit", "fit_interferometer"], name="subplot_fit_source_interpolated") is False
