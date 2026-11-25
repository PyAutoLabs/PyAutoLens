from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.colors import AsinhNorm, to_rgba

import autolens as al
from autolens.point.plot import multi_source_plots as msp
from autolens.point.plot import point_sources as ps
from autolens.point.plot import source_plane_panel as spp

directory = Path(__file__).resolve().parent


def _tracer(n_sources=2, first_centre=(0.05, 0.03)):
    redshifts = [1.0, 2.0, 1.5, 2.5, 3.0, 1.2, 1.8, 2.2, 2.8, 3.5][:n_sources]
    centres = [first_centre, (-0.04, 0.06), (0.02, -0.05), (0.0, 0.08), (-0.06, -0.02)]
    sources = [
        al.Galaxy(
            redshift=z,
            **{f"point_{k}": al.ps.Point(centre=centres[k % len(centres)])},
        )
        for k, z in enumerate(redshifts)
    ]
    return al.Tracer(
        galaxies=[
            al.Galaxy(
                redshift=0.5,
                mass=al.mp.Isothermal(
                    centre=(0.0, 0.0), ell_comps=(0.05, 0.0), einstein_radius=1.0
                ),
            )
        ]
        + sources
    )


def _solver():
    return al.PointSolver.for_grid(
        grid=al.Grid2D.uniform(shape_native=(60, 60), pixel_scales=0.05),
        pixel_scale_precision=0.001,
        magnification_threshold=0.1,
    )


def _datasets(tracer, solver, n_sources=2, offset=0.003):
    datasets = []
    for k in range(n_sources):
        plane_index = tracer.extract_plane_index_of_profile(profile_name=f"point_{k}")
        centre = tracer.planes[plane_index][0].cls_list_from(cls=al.ps.Point)[0].centre
        positions = solver.solve(
            tracer=tracer, source_plane_coordinate=centre, plane_redshift=tracer.planes[plane_index].redshift
        )
        positions = np.asarray(positions.array)
        positions = positions[np.isfinite(positions).all(axis=1)] + offset
        datasets.append(
            al.PointDataset(
                name=f"point_{k}",
                positions=al.Grid2DIrregular(values=positions),
                positions_noise_map=0.005,
                redshift=float(tracer.planes[plane_index].redshift),
            )
        )
    return datasets


@pytest.fixture(name="cluster", scope="module")
def make_cluster():
    tracer = _tracer()
    solver = _solver()
    datasets = _datasets(tracer, solver)
    fits = {
        cls.__name__: [
            al.FitPointDataset(
                dataset=d, tracer=tracer, solver=solver, fit_positions_cls=cls
            )
            for d in datasets
        ]
        for cls in (
            al.FitPositionsImagePair,
            al.FitPositionsImagePairAll,
            al.FitPositionsSource,
        )
    }
    return SimpleNamespace(tracer=tracer, solver=solver, datasets=datasets, fits=fits)


@pytest.fixture(name="figures")
def make_figures(monkeypatch):
    from matplotlib import pyplot as plt

    figures = []

    def capture(fig, path=None, filename=None, **kwargs):
        figures.append(SimpleNamespace(fig=fig, path=path, filename=filename))

    monkeypatch.setattr(msp, "save_figure", capture)
    yield figures
    for saved in figures:
        plt.close(saved.fig)


def _legend_labels(ax):
    legend = ax.get_legend()
    return [] if legend is None else [t.get_text() for t in legend.get_texts()]


def test__curves_for_planes__each_source_plane_matches_lens_calc(cluster):
    from autogalaxy.operate.lens_calc import LensCalc

    tracer = cluster.tracer
    extent = [-2.0, 2.0, -2.0, 2.0]
    curves = ps.curves_for_planes(tracer, [1, 2], extent, shape=300)

    grid = al.Grid2D.uniform(shape_native=(200, 200), pixel_scales=0.02)
    radii = {}
    for j in (1, 2):
        lens_calc = LensCalc.from_tracer(
            tracer=tracer, use_multi_plane=True, plane_i=0, plane_j=j
        )
        reference = np.asarray(
            lens_calc.tangential_critical_curve_list_from(grid=grid)[0].array
        )
        fast = curves[j].tangential[0]
        assert np.median(np.hypot(*fast.T)) == pytest.approx(
            np.median(np.hypot(*reference.T)), abs=0.03
        )
        radii[j] = np.median(np.hypot(*fast.T))
        # Caustics are the critical curves traced to the plane: far smaller than the curves.
        assert np.max(np.abs(curves[j].tangential_caustics[0])) < 0.5

    # The z=2 source is lensed more strongly than the z=1 source: a larger critical curve.
    assert radii[2] > radii[1] + 0.05


def test__curves_for_planes__fast_plots_returns_none(cluster, monkeypatch):
    monkeypatch.setenv("PYAUTO_FAST_PLOTS", "1")
    assert ps.curves_for_planes(cluster.tracer, [1], [-2, 2, -2, 2]) == {}


def test__assign_styles__black_first_then_colour_and_shape_then_redshift():
    records = [
        ps.SourceRecord(
            k=k, name=f"point_{k}", redshift=1.0 + 0.1 * k,
            observed=np.zeros((1, 2)), sigma=np.ones(1), numbers=np.ones(1, dtype=int),
        )
        for k in range(10)
    ]
    assert ps.assign_styles(records[:3]) is None
    assert [r.color for r in records[:3]] == ["k", "#E69F00", "#56B4E9"]
    assert [r.marker for r in records[:3]] == ["o", "s", "^"]

    colormap = ps.assign_styles(records)
    assert colormap is not None
    assert len({tuple(r.color) for r in records}) == 10


def test__plot_dataset_positions__all_sources_labelled_no_curves(cluster, figures):
    msp.plot_dataset_positions(cluster.datasets, solver=cluster.solver)

    ax = figures[0].fig.axes[0]
    assert figures[0].filename == "dataset_point"
    assert ax.get_title() == "Cluster dataset overview"
    labels = _legend_labels(ax)
    for dataset, label in zip(cluster.datasets, labels):
        assert label == f"{dataset.name} (z = {dataset.redshift:g}, {len(dataset.positions)} images, σ = 0.005″)"
    texts = {t.get_text() for t in ax.texts}
    assert "0·1" in texts and "1·1" in texts
    assert len(ax.lines) == 0


def test__twenty_family_dataset__all_entries_above_the_sky_map(figures):
    datasets = [
        al.PointDataset(
            name=f"point_{k}",
            redshift=1 + k / 10,
            positions_noise_map=0.04,
            positions=al.Grid2DIrregular(values=[(k - 10, 3), (10 - k, -3)]),
        )
        for k in range(20)
    ]
    image = al.Array2D.ones(shape_native=(80, 80), pixel_scales=1.0)
    msp.plot_dataset_positions(datasets, image_extent=image.geometry.extent)
    msp.plot_image_with_points(image, datasets=datasets)
    for saved in figures:
        fig = saved.fig
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ax = fig.axes[0]
        assert ax.get_xlim() == ax.get_ylim() == (-30.0, 30.0)
        legend = ax.get_legend()
        assert legend.get_title().get_text() == "20 families    ·    40 observed images"
        assert len(legend.get_texts()) == 20
        assert {text.get_text() for text in ax.texts} == {
            f"{k}·{n}" for k in range(20) for n in (1, 2)
        }
        assert (
            legend.get_window_extent(renderer).y0 >= ax.get_window_extent(renderer).y1
        )
        boxes = [text.get_window_extent(renderer) for text in legend.get_texts()]
        assert all(
            not a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1 :]
        )


def _assert_family_legend_labels(labels, fits, statistic):
    for fit in fits:
        label = next(label for label in labels if label.startswith(fit.dataset.name))
        value = spp._format_chi_squared(float(fit.positions.chi_squared))
        assert f"z = {fit.dataset.redshift:g}" in label
        assert f"{statistic} = {value}" in label
        assert "σ" not in label


def test__plot_fit_positions__global_figure(cluster, figures):
    msp.plot_fit_positions(cluster.fits["FitPositionsImagePair"], solver=cluster.solver)

    fig = figures[0].fig
    (ax,) = fig.axes
    assert figures[0].filename == "fit"
    assert ax.get_title() == "Cluster fit overview"
    _assert_family_legend_labels(
        _legend_labels(ax), cluster.fits["FitPositionsImagePair"], "Image-plane χ²"
    )
    keys = [text.get_text() for legend in fig.legends for text in legend.get_texts()]
    assert "Critical curve (each source's z)" in keys
    assert any(label.startswith("χ² = ") for label in keys)

    record = ps.source_record_from_fit(0, cluster.fits["FitPositionsImagePair"][0])
    expected_segment = np.stack([record.observed[0], record.model[record.indexes[0]]])
    assert any(
        np.allclose(line.get_xydata(), expected_segment[:, ::-1])
        for line in ax.lines
        if len(line.get_xydata()) == 2
    )


def test__plot_fit_positions__mixture_shows_minus_two_ln_l(cluster, figures):
    msp.plot_fit_positions(
        cluster.fits["FitPositionsImagePairAll"], solver=cluster.solver
    )

    fig = figures[0].fig
    _assert_family_legend_labels(
        _legend_labels(fig.axes[0]), cluster.fits["FitPositionsImagePairAll"], "−2 ln L"
    )
    keys = [text.get_text() for legend in fig.legends for text in legend.get_texts()]
    assert any(label.startswith("−2 ln L = ") for label in keys)
    assert not any(label.startswith("χ²") for label in keys)
    assert "Residual (nearest, soft pairing)" in keys


def test__subplot_fit_per_image__rows_no_legends_sigma_values(cluster, figures):
    msp.subplot_fit_per_image(
        cluster.fits["FitPositionsImagePair"], solver=cluster.solver
    )

    fig = figures[0].fig
    assert figures[0].filename == "fit_per_image"
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert titles == [
        "Residuals (point_0, z = 1)",
        "IP (point_0, z = 1)",
        "SP (point_0, z = 1)",
        "Residuals (point_1, z = 2)",
        "IP (point_1, z = 2)",
        "SP (point_1, z = 2)",
    ]
    assert all(ax.get_legend() is None for ax in fig.axes)
    ranking = fig.axes[0]
    assert not ranking.xaxis.get_visible()
    assert any(t.get_text().endswith("σ") for t in ranking.texts)


def test__subplot_fit_per_image__source_plane_fit_ranks_by_own_chi(cluster):
    fit = cluster.fits["FitPositionsSource"][0]
    record = ps.source_record_from_fit(0, fit)
    chi_squared_map = np.asarray(fit.positions.chi_squared_map.array).ravel()
    assert record.residual_sigmas == pytest.approx(np.sqrt(chi_squared_map))
    assert not record.has_model


def test__ranking_scales__shared_where_all_sources_visible_lone_source_own_range():
    def record(values, k):
        n = len(values)
        r = ps.SourceRecord(
            k=k, name=f"point_{k}", redshift=1.0, observed=np.zeros((n, 2)),
            sigma=np.ones(n), numbers=np.arange(1, n + 1),
        )
        r.delta = np.asarray(values, dtype=float)
        return r

    small = record([0.5, 1.0], 0)
    large = record([10.0, 20.0, 30.0], 1)

    shared = ps.ranking_scales_from([small, large])
    lone = ps.ranking_scales_from([small])

    assert shared.xmax > 20.0
    assert lone.xmax == pytest.approx(4.0)
    assert shared.n_slots == 3


def test__subplot_fit_per_image__many_sources_one_file_each(figures):
    tracer = _tracer(n_sources=10)
    solver = _solver()
    fits = [
        al.FitPointDataset(
            dataset=d, tracer=tracer, solver=solver,
            fit_positions_cls=al.FitPositionsImagePair,
        )
        for d in _datasets(tracer, solver, n_sources=10)
    ]

    msp.subplot_fit_per_image(fits, curves={}, output_path="out")

    assert len(figures) == 10
    assert {f.filename for f in figures} == {f"point_{k}" for k in range(10)}
    assert all(str(f.path).endswith("fit_per_image") for f in figures)


@pytest.mark.parametrize(
    ("fit_name", "statistic"),
    [
        (None, None),
        ("FitPositionsImagePair", "Image-plane χ²"),
        ("FitPositionsSource", "Source-plane χ²"),
    ],
)
def test__plot_image_with_points__fitted_sources_show_fit_statistic(
    cluster, figures, fit_name, statistic
):
    image = al.Array2D.ones(shape_native=(30, 30), pixel_scales=0.1)

    fits = cluster.fits.get(fit_name)
    msp.plot_image_with_points(
        image,
        datasets=cluster.datasets,
        fits=fits,
        solver=cluster.solver,
    )

    ax = figures[0].fig.axes[0]
    if fits is None:
        assert len(_legend_labels(ax)) == len(cluster.datasets)
        assert all("χ²" not in label for label in _legend_labels(ax))
        assert figures[0].filename == "image_with_points"
    else:
        _assert_family_legend_labels(_legend_labels(ax), fits, statistic)
        assert figures[0].filename == "fit_image_with_points"
        assert ax.get_title() == "Cluster fit on the observed image"
    texts = {t.get_text() for t in ax.texts}
    assert "0·1" in texts and "1·1" in texts


@pytest.mark.parametrize("far_positions", [False, True])
def test__cluster_fit_overviews__same_physical_square(
    cluster, figures, monkeypatch, far_positions
):
    fits = cluster.fits["FitPositionsImagePair"]
    image = al.Array2D.ones(shape_native=(80, 80), pixel_scales=1.0)
    records = ps.records_from_fits(fits)
    if far_positions:
        records[0].observed = records[0].observed.copy()
        records[0].observed[0] = (37.0, 34.0)
        records[1].model = records[1].model.copy()
        records[1].model[0] = (-38.0, -36.0)
        monkeypatch.setattr(ps, "records_from_fits", lambda fits: records)

    msp.plot_fit_positions(fits, curves={}, image_extent=image.geometry.extent)
    msp.plot_image_with_points(image, fits=fits, curves={})

    (plain,) = figures[0].fig.axes
    (sky,) = figures[1].fig.axes
    assert plain.get_xlim() == pytest.approx(sky.get_xlim())
    assert plain.get_ylim() == pytest.approx(sky.get_ylim())
    assert np.ptp(sky.get_xlim()) == pytest.approx(np.ptp(sky.get_ylim()))
    if far_positions:
        x_min, x_max = sky.get_xlim()
        y_min, y_max = sky.get_ylim()
        for record in records:
            for y, x in np.vstack([record.observed, record.model]):
                assert x_min <= x <= x_max
                assert y_min <= y <= y_max
    else:
        assert sky.get_xlim() == pytest.approx((-30, 30))
        assert sky.get_ylim() == pytest.approx((-30, 30))

    assert not plain.images
    (raster,) = sky.images
    assert isinstance(raster.norm, AsinhNorm)
    assert raster.origin == "upper"
    assert raster.get_extent() == pytest.approx(image.geometry.extent)
    gray = raster.cmap(0.4)
    assert gray[:3] == pytest.approx([gray[0]] * 3)
    for ax, colors in (
        (plain, ("#a15b0c", "#00778a")),
        (sky, ("#ffbc53", "#5fe3ee")),
    ):
        labels = {text.get_text() for text in ax.texts}
        assert {f"{r.k}·{n}" for r in records for n in r.numbers} <= labels
        for record, color in zip(records, colors):
            assert any(
                np.allclose(ring.get_edgecolors(), to_rgba(color))
                and not ring.get_facecolors().size
                and np.allclose(ring.get_offsets(), record.observed[:, ::-1])
                for ring in ax.collections
                if ring.get_edgecolors().size
                and ring.get_offsets().shape == record.observed.shape
            )


def test__cluster_fit_image__fallback_curve_labels_last_source_plane(cluster, figures):
    image = al.Array2D.ones(shape_native=(80, 80), pixel_scales=1.0)
    curve = np.array([[-2.0, 1.0], [-1.0, 2.0], [0.0, 3.0]])

    msp.plot_image_with_points(
        image,
        fits=cluster.fits["FitPositionsImagePair"],
        curves={},
        image_plane_lines=[curve],
    )

    fig = figures[0].fig
    keys = [text.get_text() for legend in fig.legends for text in legend.get_texts()]
    assert "Critical curve (last source plane)" in keys
    assert "Critical curve (each source's z)" not in keys
    (ax,) = fig.axes
    assert any(
        np.allclose(line.get_xydata(), curve[:, ::-1])
        for line in ax.lines
        if len(line.get_xydata()) == len(curve)
    )


def test__visualizer__combined_figures_written_for_several_point_analyses(
    cluster, plot_patch, monkeypatch, tmp_path
):
    from autolens.point.model.visualizer import VisualizerPoint

    plot_path = tmp_path / "combined"

    image = al.Array2D.ones(shape_native=(30, 30), pixel_scales=0.1)
    analyses = [
        al.AnalysisPoint(
            dataset=d,
            solver=cluster.solver,
            image=image,
            fit_positions_cls=al.FitPositionsImagePair,
            use_jax=False,
        )
        for d in cluster.datasets
    ]
    instance = SimpleNamespace(galaxies=cluster.tracer.galaxies)
    paths = SimpleNamespace(image_path=plot_path, output_path=plot_path)
    solver_calls = []
    solve = cluster.solver.solve

    def track_solution(**kwargs):
        solution = solve(**kwargs)
        solver_calls.append(
            (tuple(kwargs["source_plane_coordinate"]), kwargs["plane_redshift"])
        )
        return solution

    monkeypatch.setattr(cluster.solver, "solve", track_solution)

    VisualizerPoint.visualize_combined(
        analyses=analyses,
        paths=paths,
        instance=[instance, instance],
        during_analysis=False,
    )

    for name in (
        "fit.png",
        "point_0/fit.png",
        "point_1/fit.png",
        "fit_image_with_points.png",
        "dataset_point.png",
        "image_with_points.png",
    ):
        assert str(plot_path / name) in plot_patch.paths

    assert str(plot_path / "fit_per_image.png") not in plot_patch.paths
    assert solver_calls == [((0.05, 0.03), 1.0), ((-0.04, 0.06), 2.0)]

    changed_centre = (0.07, -0.02)
    changed = SimpleNamespace(galaxies=_tracer(first_centre=changed_centre).galaxies)
    solver_calls.clear()
    VisualizerPoint.visualize_combined(
        analyses=analyses,
        paths=paths,
        instance=[changed, changed],
        during_analysis=False,
    )
    assert solver_calls == [(changed_centre, 1.0), ((-0.04, 0.06), 2.0)]

    # A single point analysis has no combined figures (its own figures are output per analysis).
    plot_patch.paths.clear()
    VisualizerPoint.visualize_combined(
        analyses=analyses[:1], paths=paths, instance=[instance], during_analysis=False
    )
    assert plot_patch.paths == []


def test__plotter__fit_per_image_toggle(cluster, plot_patch, monkeypatch):
    import autolens.point.model.plotter as plotter_module
    from autolens.point.model.plotter import PlotterPoint

    plot_path = directory / "files" / "plots" / "toggle"
    plotter = PlotterPoint(image_path=plot_path)
    fits = cluster.fits["FitPositionsImagePair"][:1]

    plotter.fit_per_image(fit_list=fits, curves={})
    assert str(plot_path / "fit_per_image.png") in plot_patch.paths

    plot_patch.paths.clear()
    monkeypatch.setattr(plotter_module, "plot_setting", lambda section, name: False)
    plotter.fit_per_image(fit_list=fits, curves={})
    assert plot_patch.paths == []


@pytest.fixture(name="family_figures")
def make_family_figures(monkeypatch):
    from matplotlib.figure import Figure

    figures = {}

    def capture(fig, filename, **kwargs):
        figures[Path(filename)] = fig

    monkeypatch.setattr(Figure, "savefig", capture)
    return figures


@pytest.mark.parametrize("family_index", [None, 1])
def test__visualizer__single_analysis_legacy_figures_use_explicit_family_context(
    cluster, family_figures, tmp_path, monkeypatch, family_index
):
    from autolens.point.model.plotter import PlotterPoint
    from autolens.point.model.visualizer import VisualizerPoint

    monkeypatch.setenv("PYAUTO_FAST_PLOTS", "1")
    monkeypatch.setattr(PlotterPoint, "tracer", lambda *args, **kwargs: None)
    monkeypatch.setattr(PlotterPoint, "galaxies", lambda *args, **kwargs: None)
    dataset = cluster.datasets[1]
    analysis = al.AnalysisPoint(
        dataset=dataset,
        solver=cluster.solver,
        image=al.Array2D.ones(shape_native=(30, 30), pixel_scales=0.1),
        fit_positions_cls=al.FitPositionsImagePair,
        use_jax=False,
    )
    paths = SimpleNamespace(image_path=tmp_path, output_path=tmp_path)
    instance = SimpleNamespace(galaxies=cluster.tracer.galaxies)

    VisualizerPoint.visualize_before_fit(
        analysis=analysis, paths=paths, model=None, family_index=family_index
    )
    VisualizerPoint.visualize(
        analysis=analysis,
        paths=paths,
        instance=instance,
        during_analysis=False,
        family_index=family_index,
    )

    image_id = "1" if family_index is None else "1·1"
    for filename in ("dataset_point", "fit", "image_with_points", "fit_image_with_points"):
        ax = family_figures[tmp_path / f"{filename}.png"].axes[0]
        assert image_id in {text.get_text() for text in ax.texts}
        on_image = "image_with_points" in filename
        if on_image:
            assert ax.images[0].get_cmap().name == "gray"
        if family_index is None:
            color = "#ffbc53" if on_image else "#a15b0c"
        else:
            color = "#5fe3ee" if on_image else "#00778a"
        assert any(
            ring.get_edgecolors().size
            and np.allclose(ring.get_edgecolors(), to_rgba(color))
            and not ring.get_facecolors().size
            for ring in ax.collections
        )

    ranking, image_plane, source_plane = family_figures[tmp_path / "fit_per_image.png"].axes
    assert image_id in {tick.get_text() for tick in ranking.get_yticklabels()}
    for ax in (image_plane, source_plane):
        assert image_id in {text.get_text() for text in ax.texts}


@pytest.mark.parametrize("with_image", [False, True])
def test__plot_family_fits__one_shared_detail_panel(
    cluster, family_figures, tmp_path, monkeypatch, with_image
):
    from autolens.point.plot.family_fit_plots import plot_family_fits

    monkeypatch.setenv("PYAUTO_FAST_PLOTS", "1")
    fits = cluster.fits["FitPositionsImagePair"]
    image = (
        al.Array2D.ones(shape_native=(30, 30), pixel_scales=0.1)
        if with_image
        else None
    )
    plot_family_fits(
        fits, image=image, single_panel=True, output_path=tmp_path, output_format="png"
    )

    assert set(family_figures) == {
        tmp_path / "point_0" / "fit.png",
        tmp_path / "point_1" / "fit.png",
    }
    for fig in family_figures.values():
        fig.canvas.draw()
        axes = {ax.get_title(): ax for ax in fig.axes}
        assert "Cluster fit overview" in axes
        ranking = axes["Per-image contribution"].get_window_extent()
        source_plane = axes["All back-traced images"].get_window_extent()
        assert (ranking.width, ranking.height) == pytest.approx(
            (source_plane.width, source_plane.height)
        )
        assert "All images in the image plane" in axes
        assert not any(title.startswith("Image ") for title in axes)
        assert any(ax.images for ax in fig.axes) == with_image


@pytest.mark.parametrize(
    ("n_images", "fit_positions_cls"),
    [
        (6, al.FitPositionsImagePair),
        (7, al.FitPositionsImagePair),
        (7, al.FitPositionsSource),
    ],
)
def test__plot_family_fits__image_tiles_keep_absolute_centres_and_full_rankings(
    cluster, family_figures, tmp_path, monkeypatch, n_images, fit_positions_cls
):
    from autolens.point.plot.family_fit_plots import plot_family_fits
    from autolens.point.plot.positions_panel import image_numbers_from
    from autolens.point.plot.source_plane_panel import source_plane_data_from

    monkeypatch.setenv("PYAUTO_FAST_PLOTS", "1")
    observed = [
        (0.9, 0.7),
        (-0.8, 0.6),
        (0.7, -0.4),
        (-0.5, -0.9),
        (0.3, 0.4),
        (-0.1, -0.3),
        (0.2, -0.7),
    ][:n_images]
    dataset = al.PointDataset(
        name="point_0",
        positions=al.Grid2DIrregular(values=observed),
        positions_noise_map=0.005,
        redshift=1.0,
    )
    fit = al.FitPointDataset(
        dataset=dataset,
        tracer=cluster.tracer,
        solver=cluster.solver,
        fit_positions_cls=fit_positions_cls,
    )
    image = al.Array2D.ones(shape_native=(100, 100), pixel_scales=0.2)

    plot_family_fits([fit], image=image, output_path=tmp_path, output_format="png")

    pages = tmp_path / dataset.name
    expected_paths = {pages / "fit.png"}
    if n_images > 6:
        expected_paths.add(pages / "fit_page_02.png")
    assert set(family_figures) == expected_paths
    numbers = image_numbers_from(dataset.positions)
    centres = (
        source_plane_data_from(fit.positions, numbers=numbers).traced
        if fit_positions_cls is al.FitPositionsSource
        else observed
    )
    centres_by_title = {
        f"Image {number}": (y, x) for number, (y, x) in zip(numbers, centres)
    }
    terms = np.asarray(fit.positions.chi_squared_map.array)
    ranked_numbers = [
        number for number, _ in sorted(zip(numbers, terms), key=lambda row: (-row[1], row[0]))
    ]
    tile_order = sorted(numbers) if n_images == 6 else ranked_numbers
    sizes = []
    rankings = []
    spans = []
    for path, fig in family_figures.items():
        fig.canvas.draw()
        ranking = next(
            ax for ax in fig.axes if ax.get_title() == "Per-image contribution"
        )
        ranked_ids = [tick.get_text() for tick in ranking.get_yticklabels()]
        assert set(centres_by_title) <= set(ranked_ids)
        rankings.append(ranked_ids)

        page_tiles = []
        for ax in fig.axes:
            title = ax.get_title()
            if title not in centres_by_title:
                continue
            page_tiles.append(title)
            y, x = centres_by_title[title]
            assert np.mean(ax.get_xlim()) == pytest.approx(x)
            assert np.mean(ax.get_ylim()) == pytest.approx(y)
            spans.append((np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim())))
            bounds = ax.get_window_extent()
            sizes.append((bounds.width, bounds.height))

        page_numbers = tile_order[:6] if path.name == "fit.png" else tile_order[6:]
        assert page_tiles == [f"Image {number}" for number in page_numbers]

    assert all(ranking == rankings[0] for ranking in rankings)
    assert np.asarray(spans) == pytest.approx(np.broadcast_to(spans[0], (n_images, 2)))
    assert np.asarray(sizes) == pytest.approx(np.broadcast_to(sizes[0], (n_images, 2)))


@pytest.mark.parametrize(
    "likelihood",
    ["FitPositionsSource", "FitPositionsImagePair", "FitPositionsImagePairAll"],
)
def test__public_family_panels_match_selected_subplot(
    cluster, family_figures, tmp_path, likelihood
):
    import matplotlib.pyplot as plt
    import autolens.plot as aplt

    fits = cluster.fits[likelihood]
    fit = fits[1]
    output = dict(output_path=tmp_path, output_format="png")
    aplt.subplot_fit_family(fit, fits=fits, **output)
    combined = {
        ax.get_title(): ax for ax in family_figures[tmp_path / "fit_family.png"].axes
    }

    fig, axes = plt.subplots(2, 2)
    try:
        for ax, plot, options in zip(
            axes.flat,
            [
                aplt.plot_family_overview,
                aplt.plot_family_contributions,
                aplt.plot_family_image_detail,
                aplt.plot_family_source_plane,
            ],
            [{}, {}, {"image_number": 2}, {}],
        ):
            plot(fit, fits=fits, ax=ax, **options, **output)
            reference = combined[ax.get_title()]
            assert [t.get_text() for t in ax.texts] == [
                t.get_text() for t in reference.texts
            ]
            assert ax.get_xlabel() == reference.get_xlabel()
            assert len(ax.collections) == len(reference.collections)
            for actual, expected in zip(ax.collections, reference.collections):
                np.testing.assert_allclose(actual.get_offsets(), expected.get_offsets())
                np.testing.assert_allclose(
                    actual.get_edgecolors(), expected.get_edgecolors()
                )
        assert _legend_labels(axes[0, 0]) == _legend_labels(
            combined["Cluster fit overview"]
        )
        assert axes[1, 0].get_xlim() == pytest.approx(combined["Image 2"].get_xlim())
        assert plt.fignum_exists(fig.number)
        assert len(family_figures) == 1
    finally:
        plt.close(fig)

    aplt.plot_family_image_detail(fit, **output)
    shared = family_figures[tmp_path / "family_image_detail.png"].axes[0]
    assert shared.get_title() == (
        "Source-plane detail"
        if likelihood == "FitPositionsSource"
        else "All images in the image plane"
    )
    with pytest.raises(ValueError, match="Unknown image number 0"):
        aplt.plot_family_image_detail(fit, image_number=0)
    with pytest.raises(ValueError, match="include the selected fit"):
        aplt.plot_family_overview(fit, fits=[fits[0]])


def test__family_panels_use_tracer_redshifts_for_older_datasets(cluster, monkeypatch):
    from autolens.point.plot.family_fit_plots import _select_family

    low_z, high_z = cluster.fits["FitPositionsSource"]
    monkeypatch.setattr(low_z.dataset, "redshift", None)
    monkeypatch.setattr(high_z.dataset, "redshift", None)
    fits = [low_z, high_z] + [low_z] * (ps.MAX_QUALITATIVE - 1)
    selected, _ = _select_family(high_z, fits)
    combined, _ = _select_family(high_z, fits, overview=True)
    assert selected.ink == combined.ink


@pytest.mark.parametrize("likelihood", ["FitPositionsSource", "FitPositionsImagePair"])
def test__catalogue_ids_follow_positions_across_all_panels(
    cluster, family_figures, tmp_path, monkeypatch, likelihood
):
    import autolens.plot as aplt

    fit = cluster.fits[likelihood][0]
    ids = [f"101{chr(97+i)}" for i in range(len(fit.dataset.positions))]
    monkeypatch.setattr(fit.dataset, "image_ids", ids)
    output = dict(output_path=tmp_path, output_format="png")
    aplt.subplot_fit_family(fit, **output)
    axes = {ax.get_title(): ax for ax in family_figures[tmp_path / "fit_family.png"].axes}
    for name in ["Cluster fit overview", "All back-traced images"]:
        assert [t.get_text() for t in axes[name].texts] == ids
    assert {t.get_text() for t in axes["Per-image contribution"].get_yticklabels()} >= set(ids)
    assert set(ids) <= axes.keys()

    aplt.plot_family_image_detail(fit, image_id=ids[-1], **output)
    zoom = family_figures[tmp_path / "family_image_detail.png"].axes[0]
    assert zoom.get_title() == ids[-1]
    np.testing.assert_allclose(zoom.get_xlim(), axes[ids[-1]].get_xlim())
    with pytest.raises(ValueError, match="Unknown image ID"):
        aplt.plot_family_image_detail(fit, image_id="missing")
