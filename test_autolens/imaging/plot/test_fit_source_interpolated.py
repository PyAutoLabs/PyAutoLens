"""
`fit.png` / `fit_log10.png` source panels with the likelihood's interpolant in the
middle of the right-hand column, between max zoom and no zoom (`plots.yaml`
`fit: subplot_fit_source_interpolated`, default on, PyAutoArray#700).
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


def test__imaging__default__max_zoom__interpolated__no_zoom(
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

    # Right-hand column, top to bottom: max zoom, interpolated, no zoom.
    def _y(ax):
        return ax.get_position().y0

    assert _y(max_zoom) > _y(interpolated) > _y(no_zoom)
    assert max_zoom.get_position().x0 == pytest.approx(interpolated.get_position().x0)
    assert max_zoom.get_position().x0 == pytest.approx(no_zoom.get_position().x0)

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


def _panel_titles(module, monkeypatch, subplot, fit, **kwargs):
    """The 3 x 4 panel axes of a subplot in grid order (colorbars excluded)."""
    captured = {}

    def _capture(fig, *args, **kw):
        captured["axes"] = fig.axes[:12]

    monkeypatch.setattr(module, "save_figure", _capture)
    subplot(fit, **kwargs)
    axes = captured["axes"]
    titles = [ax.get_title() for ax in axes]
    plt.close(axes[0].get_figure())
    return titles, axes


def test__imaging_log10__default__max_zoom__interpolated__no_zoom(
    fit_imaging_delaunay, monkeypatch
):
    titles, axes = _panel_titles(
        fit_imaging_plots, monkeypatch, fit_imaging_plots.subplot_fit_log10,
        fit_imaging_delaunay,
    )
    assert titles[3] == "Source Plane (Max Zoom)"
    assert titles[7] == "Source Plane (Interpolated)"
    assert titles[11] == "Source Plane (No Zoom)"
    assert "Source Plane (Mid Zoom)" not in titles

    interpolated = axes[7]
    assert len(interpolated.get_images()) == 1
    assert interpolated.get_images()[0].norm.__class__.__name__ == "LogNorm"
    assert interpolated.get_xlim() == pytest.approx(axes[3].get_xlim())


def test__imaging_log10__switch_off__max_mid_no_zoom(fit_imaging_delaunay, monkeypatch):
    titles, _ = _panel_titles(
        fit_imaging_plots, monkeypatch, fit_imaging_plots.subplot_fit_log10,
        fit_imaging_delaunay, source_interpolated=False,
    )
    assert titles[3] == "Source Plane (Max Zoom)"
    assert titles[7] == "Source Plane (Mid Zoom)"
    assert titles[11] == "Source Plane (No Zoom)"
    assert "Source Plane (Interpolated)" not in titles


def test__imaging_log10__rectangular_source__keeps_max_mid_no_zoom(monkeypatch):
    titles, _ = _panel_titles(
        fit_imaging_plots, monkeypatch, fit_imaging_plots.subplot_fit_log10,
        fixtures.make_fit_imaging_x2_plane_inversion_7x7(),
    )
    assert titles[3] == "Source Plane (Max Zoom)"
    assert titles[7] == "Source Plane (Mid Zoom)"
    assert titles[11] == "Source Plane (No Zoom)"


_INTERFEROMETER_FIXED_PANELS = {
    0: "Amplitudes vs UV-Distance",
    1: "Dirty Image",
    2: "Dirty Model Image",
    4: "Dirty Residual Map",
    5: "Normalized Residual Map (Real)",
    6: "Normalized Residual Map (Imag)",
    8: "Dirty Normalized Residual Map",
    9: r"Normalized Residual Map $1\sigma$",
    10: "Dirty Chi-Squared Map",
}


def _assert_interferometer_fixed_panels(titles):
    assert "Dirty Signal-To-Noise Map" not in titles
    for index, title in _INTERFEROMETER_FIXED_PANELS.items():
        assert titles[index] == title


def _colorbar_vmax(ax):
    mappables = ax.get_images() or ax.collections
    return mappables[0].norm.vmax


def test__interferometer__default__max_zoom__interpolated__no_zoom(
    fit_interferometer_delaunay, monkeypatch
):
    titles, axes = _panel_titles(
        fit_interferometer_plots, monkeypatch, fit_interferometer_plots.subplot_fit,
        fit_interferometer_delaunay,
    )
    _assert_interferometer_fixed_panels(titles)
    assert titles[3] == "Source Plane (Max Zoom)"
    assert titles[7] == "Source Plane (Interpolated)"
    assert titles[11] == "Source Plane (No Zoom)"

    assert len(axes[3].get_images()) == 0
    assert len(axes[7].get_images()) == 1
    assert len(axes[11].get_images()) == 0
    assert axes[7].get_xlim() == pytest.approx(axes[3].get_xlim())

    # Every source-plane panel shares one colour scale.
    vmax = fit_interferometer_plots._get_source_vmax(fit_interferometer_delaunay)
    assert vmax is not None
    for index in (3, 7, 11):
        assert _colorbar_vmax(axes[index]) == pytest.approx(vmax)


def test__interferometer__switch_off__max_mid_no_zoom(
    fit_interferometer_delaunay, monkeypatch
):
    titles, _ = _panel_titles(
        fit_interferometer_plots, monkeypatch, fit_interferometer_plots.subplot_fit,
        fit_interferometer_delaunay, source_interpolated=False,
    )
    _assert_interferometer_fixed_panels(titles)
    assert titles[3] == "Source Plane (Max Zoom)"
    assert titles[7] == "Source Plane (Mid Zoom)"
    assert titles[11] == "Source Plane (No Zoom)"
    assert "Source Plane (Interpolated)" not in titles


@pytest.mark.parametrize(
    "make_fit",
    [
        fixtures.make_fit_interferometer_x2_plane_inversion_7x7,
        fixtures.make_fit_interferometer_x2_plane_7x7,
    ],
    ids=["rectangular", "parametric"],
)
def test__interferometer__no_interpolated_view__max_mid_no_zoom(make_fit, monkeypatch):
    titles, _ = _panel_titles(
        fit_interferometer_plots, monkeypatch, fit_interferometer_plots.subplot_fit,
        make_fit(),
    )
    _assert_interferometer_fixed_panels(titles)
    assert titles[3] == "Source Plane (Max Zoom)"
    assert titles[7] == "Source Plane (Mid Zoom)"
    assert titles[11] == "Source Plane (No Zoom)"


def test__plots_yaml__switch_defaults_on():
    from autolens.analysis.plotter import plot_setting

    assert plot_setting(section=["fit", "fit_imaging"], name="subplot_fit_source_interpolated") is True
    assert plot_setting(section=["fit", "fit_interferometer"], name="subplot_fit_source_interpolated") is True
