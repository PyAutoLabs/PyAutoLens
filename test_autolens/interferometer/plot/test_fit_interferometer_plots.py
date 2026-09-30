from pathlib import Path
import pytest

from autolens.interferometer.plot.fit_interferometer_plots import (
    subplot_fit,
    subplot_fit_real_space,
)


@pytest.fixture(name="plot_path")
def make_fit_interferometer_plotter_setup():
    return Path(__file__).resolve().parent / "files" / "plots" / "fit"


def test__subplot_fit(
    fit_interferometer_x2_plane_7x7, plot_path, plot_patch
):
    subplot_fit(
        fit=fit_interferometer_x2_plane_7x7,
        output_path=plot_path,
        output_format="png",
    )
    assert str(plot_path / "fit.png") in plot_patch.paths


def test__subplot_fit_real_space(
    fit_interferometer_x2_plane_7x7,
    fit_interferometer_x2_plane_inversion_7x7,
    plot_path,
    plot_patch,
):
    subplot_fit_real_space(
        fit=fit_interferometer_x2_plane_7x7,
        output_path=plot_path,
        output_format="png",
    )
    assert str(plot_path / "fit_real_space.png") in plot_patch.paths


def test__subplots__array_free_dataset(
    interferometer_7, plot_path, plot_patch, monkeypatch
):
    import autoarray as aa
    import autolens as al
    from autolens.interferometer.plot import fit_interferometer_plots
    from autolens.interferometer.plot.fit_interferometer_plots import (
        subplot_fit_dirty_images,
        subplot_fit_interferometer_combined,
        subplot_tracer_from_fit,
    )

    dataset = aa.Interferometer.from_stream(
        [(interferometer_7.uv_wavelengths, interferometer_7.data, interferometer_7.noise_map)],
        real_space_mask=interferometer_7.real_space_mask,
        transformer_class=type(interferometer_7.transformer),
    )

    lens = al.Galaxy(
        redshift=0.5, mass=al.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.0)
    )
    source = al.Galaxy(
        redshift=1.0,
        pixelization=al.Pixelization(
            mesh=al.mesh.RectangularUniform(shape=(3, 3)),
            regularization=al.reg.Constant(coefficient=1.0),
        ),
    )

    fit = al.FitInterferometer(
        dataset=dataset, tracer=al.Tracer(galaxies=[lens, source])
    )

    titles = []
    plot_array = fit_interferometer_plots.plot_array

    def _plot_array(*args, title=None, **kwargs):
        titles.append(title)
        return plot_array(*args, title=title, **kwargs)

    monkeypatch.setattr(fit_interferometer_plots, "plot_array", _plot_array)

    natural_titles = [
        "Dirty Image (Natural)",
        "Dirty Model Image (Natural)",
        "Dirty Residual Map (Natural)",
    ]

    subplot_fit(fit=fit, output_path=plot_path, output_format="png")
    assert str(plot_path / "fit.png") in plot_patch.paths
    assert titles[:3] == natural_titles

    titles.clear()
    subplot_fit_dirty_images(fit=fit, output_path=plot_path, output_format="png")
    assert str(plot_path / "fit_dirty_images.png") in plot_patch.paths
    assert titles == natural_titles

    titles.clear()
    subplot_fit_interferometer_combined(
        fit_list=[fit], output_path=plot_path, output_format="png"
    )
    assert str(plot_path / "fit_combined.png") in plot_patch.paths
    assert titles[0] == "Dirty Image (Natural) (ch 0)"
    assert "Dirty Model Image (Natural)" in titles
    assert "Dirty Residual Map (Natural)" in titles

    titles.clear()
    subplot_fit_real_space(fit=fit, output_path=plot_path, output_format="png")
    assert str(plot_path / "fit_real_space.png") in plot_patch.paths
    assert titles[0] == "Reconstructed Image"

    titles.clear()
    subplot_tracer_from_fit(fit=fit, output_path=plot_path, output_format="png")
    assert titles[0] == "Dirty Model Image (Natural)"
