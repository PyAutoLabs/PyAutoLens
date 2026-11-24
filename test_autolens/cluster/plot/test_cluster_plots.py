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


#####################################################################
# Critical-curve engine dispatch (general.critical_curves_method). #
#####################################################################

import sys

from autogalaxy.operate.lens_calc import LensCalc

from autolens.cluster.plot import cluster_plots

_ROUTINES = {
    "marching_squares": [
        "tangential_critical_curve_list_from",
        "radial_critical_curve_list_from",
        "tangential_caustic_list_from",
        "radial_caustic_list_from",
    ],
    "zero_contour": [
        "tangential_critical_curve_list_via_zero_contour_from",
        "radial_critical_curve_list_via_zero_contour_from",
        "tangential_caustic_list_via_zero_contour_from",
        "radial_caustic_list_via_zero_contour_from",
    ],
}


@pytest.fixture(name="power_law_multi_plane_tracer")
def make_power_law_multi_plane_tracer():
    """Two-source-plane cluster-like system whose shallower-than-isothermal lens
    (slope 1.7, analytic deflections so the test stays fast) has tangential *and*
    radial critical curves in both source planes (z=1 and z=2)."""
    lens = al.Galaxy(
        redshift=0.5,
        mass=al.mp.PowerLaw(
            centre=(0.0, 0.0), ell_comps=(0.05, 0.0), einstein_radius=1.6, slope=1.7
        ),
    )
    return al.Tracer(galaxies=[lens, al.Galaxy(redshift=1.0), al.Galaxy(redshift=2.0)])


@pytest.fixture(name="engine_spy")
def make_engine_spy(monkeypatch, grid):
    """Record every LensCalc curve routine called, keyed by the source plane whose
    ``LensCalc`` it was called on.

    Marching-squares routines run for real. Zero-contour routines are stubbed (a real
    ``jax_zero_contour`` trace costs several seconds per curve) with the matching
    marching-squares result *of the same per-plane LensCalc*, so the plotted
    geometry still comes from that plane's multi-plane ray-tracing.
    """
    calls = []
    depth = [0]
    real_lens_calc_for_plane = cluster_plots._lens_calc_for_plane
    real_ms = {name: getattr(LensCalc, name) for name in _ROUTINES["marching_squares"]}

    def lens_calc_for_plane(tracer, plane_index):
        lens_calc = real_lens_calc_for_plane(tracer, plane_index=plane_index)
        lens_calc._spy_plane = plane_index
        return lens_calc

    monkeypatch.setattr(cluster_plots, "_lens_calc_for_plane", lens_calc_for_plane)

    def ms_spy(name):
        def spy(self, *args, **kwargs):
            # Only record calls made by the plot helpers, not the critical-curve
            # call a marching-squares caustic routine makes internally.
            if depth[0] == 0:
                calls.append((self._spy_plane, name, args, kwargs))
            depth[0] += 1
            try:
                return real_ms[name](self, *args, **kwargs)
            finally:
                depth[0] -= 1

        return spy

    def zc_spy(name, ms_name):
        def spy(self, *args, **kwargs):
            calls.append((self._spy_plane, name, args, kwargs))
            depth[0] += 1
            try:
                return real_ms[ms_name](self, grid=grid)
            finally:
                depth[0] -= 1

        return spy

    for name in _ROUTINES["marching_squares"]:
        monkeypatch.setattr(LensCalc, name, ms_spy(name))
    for name, ms_name in zip(_ROUTINES["zero_contour"], _ROUTINES["marching_squares"]):
        monkeypatch.setattr(LensCalc, name, zc_spy(name, ms_name))

    return calls


def _set_engine(monkeypatch, method):
    """Configure ``general.critical_curves_method`` in the visualize config."""
    from autonerves import conf

    monkeypatch.setitem(
        conf.instance["visualize"]["general"]["general"],
        "critical_curves_method",
        method,
    )


def _median_radius_by_label(ax):
    """``{legend label: median |(y, x)|}`` of the first line drawn per label."""
    return {
        line.get_label(): float(np.median(np.hypot(line.get_xdata(), line.get_ydata())))
        for line in ax.get_lines()
        if not line.get_label().startswith("_")
    }


@pytest.mark.parametrize("method", ["marching_squares", "zero_contour"])
def test__plot_critical_curves__routes_each_plane_to_configured_engine(
    method, power_law_multi_plane_tracer, grid, engine_spy, monkeypatch
):
    import matplotlib.pyplot as plt

    selector_calls = []

    def selector():
        selector_calls.append(1)
        return method

    monkeypatch.setattr(cluster_plots, "_critical_curves_method", selector)

    fig, ax = plt.subplots()
    plot_critical_curves(
        power_law_multi_plane_tracer, grid=grid, include_radial=True, ax=ax
    )

    tangential, radial = _ROUTINES[method][0], _ROUTINES[method][1]
    assert [(plane, name) for plane, name, _, _ in engine_spy] == [
        (1, tangential),
        (1, radial),
        (2, tangential),
        (2, radial),
    ]
    for _, _, args, kwargs in engine_spy:
        if method == "zero_contour":
            assert args == () and kwargs == {}
        else:
            assert kwargs == {"grid": grid}
    assert len(selector_calls) == 1

    radii = _median_radius_by_label(ax)
    assert set(radii) == {
        "z=1.00 tangential",
        "z=1.00 radial",
        "z=2.00 tangential",
        "z=2.00 radial",
    }
    # The z=2 plane sees a stronger lens, so its curves differ from the z=1 plane's.
    assert radii["z=2.00 tangential"] > radii["z=1.00 tangential"] * 1.05
    assert not np.isclose(radii["z=2.00 radial"], radii["z=1.00 radial"], rtol=0.02)
    plt.close(fig)


@pytest.mark.parametrize("method", ["marching_squares", "zero_contour"])
def test__plot_caustics__routes_each_plane_to_configured_engine(
    method, power_law_multi_plane_tracer, grid, engine_spy, monkeypatch
):
    import matplotlib.pyplot as plt

    _set_engine(monkeypatch, method)
    monkeypatch.setitem(sys.modules, "jax_zero_contour", object())

    fig, ax = plt.subplots()
    plot_caustics(power_law_multi_plane_tracer, grid=grid, ax=ax)

    tangential, radial = _ROUTINES[method][2], _ROUTINES[method][3]
    assert [(plane, name) for plane, name, _, _ in engine_spy] == [
        (1, tangential),
        (1, radial),
        (2, tangential),
        (2, radial),
    ]

    radii = _median_radius_by_label(ax)
    assert set(radii) == {
        "z=1.00 tangential",
        "z=1.00 radial",
        "z=2.00 tangential",
        "z=2.00 radial",
    }
    # Source-plane mapping: each plane's caustics are its own critical curves traced
    # through its own deflections, so the two planes' caustics differ too.
    assert not np.isclose(
        radii["z=2.00 tangential"], radii["z=1.00 tangential"], rtol=0.02
    )
    assert not np.isclose(radii["z=2.00 radial"], radii["z=1.00 radial"], rtol=0.02)
    plt.close(fig)


def test__plot_critical_curves__tangential_only_by_default(
    power_law_multi_plane_tracer, grid, engine_spy, monkeypatch, plot_patch
):
    _set_engine(monkeypatch, "zero_contour")
    monkeypatch.setitem(sys.modules, "jax_zero_contour", object())

    plot_critical_curves(power_law_multi_plane_tracer, grid=grid, plane_indices=[2])

    assert [(plane, name) for plane, name, _, _ in engine_spy] == [
        (2, "tangential_critical_curve_list_via_zero_contour_from")
    ]


def test__zero_contour_without_dependency__falls_back_to_marching_squares(
    power_law_multi_plane_tracer, grid, engine_spy, monkeypatch, caplog
):
    import logging
    import matplotlib.pyplot as plt

    _set_engine(monkeypatch, "zero_contour")
    # A ``None`` entry makes ``import jax_zero_contour`` raise ImportError.
    monkeypatch.setitem(sys.modules, "jax_zero_contour", None)

    fig, ax = plt.subplots()
    with caplog.at_level(logging.WARNING, logger="autogalaxy.util.plot_utils"):
        plot_critical_curves(
            power_law_multi_plane_tracer, grid=grid, include_radial=True, ax=ax
        )
        plot_caustics(power_law_multi_plane_tracer, grid=grid, ax=ax)
    plt.close(fig)

    assert {name for _, name, _, _ in engine_spy} == set(_ROUTINES["marching_squares"])
    fallback_warnings = [
        record
        for record in caplog.records
        if "jax_zero_contour is not installed" in record.getMessage()
    ]
    # Read once per plot call, so one warning each.
    assert len(fallback_warnings) == 2


def test_overlay_preserves_image_origin_and_uses_family_metadata():
    import matplotlib.pyplot as plt
    from matplotlib.colors import AsinhNorm, to_hex

    fig, ax = plt.subplots()
    image = aa.Array2D.no_mask(
        values=np.arange(24).reshape(4, 6),
        pixel_scales=(0.3, 0.2),
        origin=(12.0, -7.0),
    )
    datasets = [
        al.PointDataset(
            name="Arc A",
            positions=[(12.1, -7.2)],
            positions_noise_map=0.2,
            redshift=1.1,
        ),
        al.PointDataset(
            name="Arc B",
            positions=[(11.8, -6.9)],
            positions_noise_map=0.2,
            redshift=2.3,
        ),
    ]
    plot_positions_overlay(
        [dataset.positions for dataset in datasets],
        image=image,
        ax=ax,
        datasets=datasets,
    )

    raster = ax.images[0]
    assert raster.get_extent() == pytest.approx(image.geometry.extent)
    assert raster.origin == "upper"
    assert isinstance(raster.norm, AsinhNorm)
    assert np.asarray(
        [artist.get_offsets()[0] for artist in ax.collections]
    ) == pytest.approx(np.asarray([(-7.2, 12.1), (-6.9, 11.8)]))
    assert [to_hex(artist.get_edgecolors()[0]) for artist in ax.collections] == [
        "#ffbc53",
        "#5fe3ee",
    ]
    legend = ax.get_legend()
    assert legend.get_bbox_to_anchor().y0 > ax.bbox.y1
    assert [label.get_text() for label in legend.get_texts()] == [
        "Arc A · 1 image · z=1.10",
        "Arc B · 1 image · z=2.30",
    ]
    plt.close(fig)


def test_zooms_start_each_family_on_its_own_row(positions_list, image, monkeypatch):
    import matplotlib.pyplot as plt

    figures = []
    monkeypatch.setattr(
        cluster_plots, "save_figure", lambda fig, **kwargs: figures.append(fig)
    )
    plot_image_group_zooms(positions_list, image=image, max_cols=2, zoom_arcsec=0.4)

    fig = figures[0]
    assert [ax.get_title() for ax in fig.axes] == [
        "Source 0 · Image 1",
        "Source 0 · Image 2",
        "Source 0 · Image 3",
        "",
        "Source 1 · Image 1",
        "Source 1 · Image 2",
    ]
    assert not fig.axes[3].axison
    # Geometric image 2 is the third position in the dataset input.
    assert fig.axes[1].get_xlim() == pytest.approx((1.4, 1.8))
    assert fig.axes[1].get_ylim() == pytest.approx((0.0, 0.4))
    assert fig.axes[4].get_xlim() == pytest.approx((0.2, 0.6))
    assert fig.axes[4].get_ylim() == pytest.approx((1.9, 2.3))
    plt.close(fig)


def test_dense_overlay_groups_legend_in_five_columns():
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    fig, ax = plt.subplots(figsize=(10, 7))
    legend_positions = [[(0.0, float(index))] for index in range(20)]
    plot_positions_overlay(legend_positions, ax=ax)

    colours = [to_hex(artist.get_edgecolors()[0]) for artist in ax.collections]
    assert len(set(colours)) == len(legend_positions)
    assert colours[0] != "#a15b0c"
    assert colours[1] != "#00778a"

    fig.canvas.draw()
    labels = ax.get_legend().get_texts()
    y_positions = {round(label.get_window_extent().y0, 3) for label in labels}
    assert len(labels) == len(legend_positions)
    assert len(y_positions) == 4
    plt.close(fig)
