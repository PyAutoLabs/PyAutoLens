"""
The run-time estimator (likelihood lifecycle 5E-3, PyAutoLens#887): the `runtime_estimates.yaml` loader,
row matching by decision / setup / instrument size, the Nautilus heuristic and parallel efficiency, the
one-line `Estimate:` format and the two PyAutoFit hooks on the lens analyses (`runtime_estimate`,
`resources_per_eval`).
"""

import logging
from types import SimpleNamespace

import pytest
import yaml

import autofit as af
import autolens as al

from autolens.analysis import runtime_estimate as re_mod
from autolens.analysis import wall_model


@pytest.fixture(autouse=True)
def unset_setup_env(monkeypatch):
    monkeypatch.delenv("PYAUTO_SETUP", raising=False)
    monkeypatch.delenv("PYAUTO_DISABLE_JAX", raising=False)


class Nautilus:
    """A Nautilus-like search: only its name and `n_live` matter to the estimator."""

    def __init__(self, n_live=150, number_of_cores=1):
        self.n_live = n_live
        self.number_of_cores = number_of_cores
        self.paths = None
        self.logger = logging.getLogger("test_runtime_estimate")


class Emcee:
    pass


def _rectangular_model():
    pixelization = af.Model(
        al.Pixelization,
        mesh=al.mesh.RectangularUniform(shape=(3, 3)),
        regularization=al.reg.Constant,
    )
    return af.Collection(
        galaxies=af.Collection(
            lens=af.Model(al.Galaxy, redshift=0.5, mass=al.mp.IsothermalSph),
            source=af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization),
        )
    )


def _sized_imaging(n_pixels):
    """An imaging-like dataset sized by its masked pixel count (what the matcher reads)."""
    return SimpleNamespace(mask=SimpleNamespace(pixels_in_mask=n_pixels))


def _sized_interferometer(n_vis):
    return SimpleNamespace(uv_wavelengths=SimpleNamespace(shape=(n_vis, 2)))


def _row(**kwargs):
    rows = re_mod.load_runtime_estimates()["rows"]
    return next(r for r in rows if all(r.get(k) == v for k, v in kwargs.items()))


# Loader


def test__loader__version_2_rows_and_curve():
    table = re_mod.load_runtime_estimates()

    assert len(table["rows"]) > 100
    assert table["curve"][8] == pytest.approx(0.55)
    assert table["curve"][1] == pytest.approx(1.0)
    assert {"setup_id", "decision_id", "ms_per_eval", "n_pixels_masked", "n_vis"} <= set(
        table["rows"][0]
    )


def test__loader__version_mismatch__clear_error(tmp_path):
    path = tmp_path / "runtime_estimates.yaml"
    path.write_text(yaml.safe_dump({"schema": "runtime-estimates", "version": 1}))

    with pytest.raises(ValueError, match="version 1.*version 2"):
        re_mod.load_runtime_estimates(path=path)


def test__loader__missing_curve__falls_back_to_ported_constant(tmp_path):
    path = tmp_path / "runtime_estimates.yaml"
    path.write_text(
        yaml.safe_dump({"schema": "runtime-estimates", "version": 2, "estimates": []})
    )

    assert re_mod.load_runtime_estimates(path=path)["curve"] == wall_model.PARALLEL_EFFICIENCY


# Matching


@pytest.mark.parametrize(
    "n_pixels, instrument",
    [(3800, "euclid"), (15000, "hst"), (40000, "jwst"), (300000, "ao")],
)
def test__imaging__matched_by_masked_pixels(n_pixels, instrument):
    estimate = al.estimate_runtime(
        dataset=_sized_imaging(n_pixels),
        model=_rectangular_model(),
        search=Nautilus(),
        setup="numba_cpu_sparse_fp64",
        likelihood="imaging/rectangular",
    )

    assert estimate.matched_by == "size"
    assert estimate.instrument == instrument
    row = _row(
        likelihood="imaging/rectangular",
        setup_id="numba_cpu_sparse_fp64",
        instrument=instrument,
    )
    assert estimate.ms_per_eval == row["ms_per_eval"]
    assert estimate.record_id == row["record_id"]
    assert estimate.calibrated is True


def test__imaging__real_dataset_and_model__likelihood_from_model(masked_imaging_7x7):
    estimate = al.estimate_runtime(
        dataset=masked_imaging_7x7,
        model=_rectangular_model(),
        search=Nautilus(),
        setup=al.Setup.from_setup_id("numba_cpu_sparse_fp64"),
    )

    assert estimate.likelihood == "imaging/rectangular"
    assert estimate.instrument == "euclid"  # 9 masked pixels: the smallest cell
    assert estimate.matched_by == "size"


def test__decision_id__matches_rows_of_that_decision():
    setup = al.Setup.from_setup_id(
        "jax_gpu_dense_fp64",
        decision_id="imaging-delaunay-standard-gpu-a100",
        hardware_class="gpu_a100",
    )

    estimate = al.estimate_runtime(
        dataset=_sized_imaging(15000),
        model=_rectangular_model(),
        search=Nautilus(),
        setup=setup,
    )

    assert estimate.likelihood == "imaging/delaunay"
    assert estimate.instrument == "hst"


@pytest.mark.parametrize("n_vis, instrument", [(190, "sma"), (1_000_000, "alma")])
def test__interferometer__matched_by_visibilities__uncalibrated(n_vis, instrument):
    estimate = al.estimate_runtime(
        dataset=_sized_interferometer(n_vis),
        model=_rectangular_model(),
        search=Nautilus(),
        setup="numba_cpu_sparse_fp64_wtilde",
        likelihood="interferometer/rectangular",
    )

    assert estimate.matched_by == "size"
    assert estimate.instrument == instrument
    assert estimate.calibrated is False
    assert estimate.line().endswith(" · uncalibrated (held-out 0.2–0.5×)")


def test__interferometer__real_dataset__size_from_visibilities(interferometer_7):
    assert re_mod.dataset_size_from(interferometer_7, "interferometer") == 7


def test__no_size__several_instruments__band():
    estimate = al.estimate_runtime(
        dataset=None,
        model=_rectangular_model(),
        search=Nautilus(),
        setup="numba_cpu_sparse_fp64",
        likelihood="imaging/rectangular",
    )

    assert estimate.matched_by == "band"
    assert estimate.instrument is None
    assert estimate.wall_seconds is None
    low, high = estimate.wall_band
    assert low < high
    assert "(no size match; range across instruments)" in estimate.line()


def test__point__unknown_positions_fit__band_across_likelihoods(point_dataset):
    estimate = al.estimate_runtime(
        dataset=point_dataset,
        model=_rectangular_model(),
        search=Nautilus(),
        setup="jax_cpu_na_fp64",
    )

    assert estimate.matched_by == "band"
    assert estimate.ms_per_eval_band == (
        _row(likelihood="point_source_source/source_plane", setup_id="jax_cpu_na_fp64")[
            "ms_per_eval"
        ],
        _row(likelihood="point_source_image/image_plane", setup_id="jax_cpu_na_fp64")[
            "ms_per_eval"
        ],
    )
    assert "range across likelihoods" in estimate.line()
    assert estimate.calibrated is True


def test__point__known_likelihood__single(point_dataset):
    estimate = al.estimate_runtime(
        dataset=point_dataset,
        model=_rectangular_model(),
        search=Nautilus(),
        setup="jax_cpu_na_fp64",
        likelihood="point_source_image/image_plane",
    )

    assert estimate.matched_by == "single"
    assert estimate.instrument == "simple"
    assert estimate.calibrated is True


def test__weak__single__uncalibrated():
    dataset = al.WeakDataset.__new__(al.WeakDataset)

    estimate = al.estimate_runtime(
        dataset=dataset,
        model=_rectangular_model(),
        search=Nautilus(),
        setup="jax_cpu_dense_fp64",
    )

    assert estimate.likelihood == "weak/isothermal"
    assert estimate.matched_by == "single"
    assert estimate.calibrated is False
    assert estimate.note == "uncalibrated (held-out 0.2–0.5×)"


def test__no_setup__or_no_row__none():
    assert (
        al.estimate_runtime(
            dataset=None, model=None, search=Nautilus(), setup=None
        )
        is None
    )
    assert (
        al.estimate_runtime(
            dataset=None,
            model=None,
            search=Nautilus(),
            setup="numba_cpu_dense_fp32",
            likelihood="imaging/rectangular",
        )
        is None
    )


# Heuristic, efficiency and line


def test__non_nautilus__no_estimate_line():
    estimate = al.estimate_runtime(
        dataset=_sized_imaging(15000),
        model=_rectangular_model(),
        search=Emcee(),
        setup="numba_cpu_sparse_fp64",
        likelihood="imaging/rectangular",
    )

    assert estimate.wall_seconds is None
    assert estimate.expected_evals is None
    assert estimate.line() == (
        "Estimate: ~697 ms/eval (hst cell) · no estimate for Emcee "
        "(run-time model covers Nautilus only)"
    )


def test__real_nautilus__recognised():
    estimate = al.estimate_runtime(
        dataset=_sized_imaging(15000),
        model=_rectangular_model(),
        search=af.Nautilus(name="runtime_estimate_test", n_live=150),
        setup="numba_cpu_sparse_fp64",
        likelihood="imaging/rectangular",
    )

    assert estimate.n_live == 150
    assert estimate.wall_seconds is not None


def test__numba__efficiency_applied__wall_model():
    model = _rectangular_model()

    estimate = al.estimate_runtime(
        dataset=_sized_imaging(15000),
        model=model,
        search=Nautilus(n_live=150),
        setup="numba_cpu_sparse_fp64",
        number_of_cores=8,
        likelihood="imaging/rectangular",
    )

    evals = round(4.30 * 150 * model.prior_count**1.637 * 1.13)
    per_call = 0.696736 / (8 * 0.55)

    assert estimate.expected_evals == evals
    assert estimate.evals_model == "nautilus-heuristic"
    assert estimate.parallel_efficiency == pytest.approx(0.55)
    assert estimate.number_of_cores == 8
    assert estimate.wall_seconds == pytest.approx(30 + evals * (0.014 + per_call))
    assert "η=0.55 on 8 cores" in estimate.line()


def test__numba__efficiency_between_sweep_points__next_lower():
    estimate = al.estimate_runtime(
        dataset=_sized_imaging(15000),
        model=_rectangular_model(),
        search=Nautilus(),
        setup="numba_cpu_sparse_fp64",
        number_of_cores=12,
        likelihood="imaging/rectangular",
    )

    assert estimate.parallel_efficiency == pytest.approx(0.55)


def test__jax__per_call_as_is__one_worker():
    model = _rectangular_model()

    estimate = al.estimate_runtime(
        dataset=_sized_imaging(15000),
        model=model,
        search=Nautilus(n_live=150),
        setup="jax_gpu_dense_fp64",
        number_of_cores=8,
        likelihood="imaging/rectangular",
    )

    evals = estimate.expected_evals
    assert estimate.parallel_efficiency is None
    assert estimate.number_of_cores == 1
    assert estimate.wall_seconds == pytest.approx(30 + evals * (0.014 + 0.0246826))
    assert "1 worker (JAX)" in estimate.line()
    assert "η=" not in estimate.line()


def test__line__one_line_format():
    line = al.estimate_runtime(
        dataset=_sized_imaging(15000),
        model=_rectangular_model(),
        search=Nautilus(n_live=150),
        setup="numba_cpu_sparse_fp64",
        number_of_cores=8,
        likelihood="imaging/rectangular",
    ).line()

    assert "\n" not in line
    assert line.startswith("Estimate: ~697 ms/eval (hst cell) · Nautilus n_live=150 ≈ ")
    assert "evals (heuristic) · η=0.55 on 8 cores · ~" in line
    assert line.endswith(" on this hardware")


# Analysis hooks


def _imaging_analysis(dataset, setup):
    analysis = al.AnalysisImaging(
        dataset=dataset, raise_inversion_positions_likelihood_exception=False
    )
    analysis.setup = al.Setup.from_setup_id(setup) if setup else None
    analysis._runtime_estimate_model = _rectangular_model()
    return analysis


def test__analysis__no_setup__no_estimate_no_resources(masked_imaging_7x7):
    analysis = al.AnalysisImaging(dataset=masked_imaging_7x7)

    assert analysis.runtime_estimate(search=Nautilus(), model=_rectangular_model()) is None
    assert analysis.resources_per_eval() is None


def test__analysis__resources_per_eval__matched_row(masked_imaging_7x7):
    analysis = _imaging_analysis(masked_imaging_7x7, "numba_cpu_sparse_fp64_wtilde")
    assert analysis.resources_per_eval() is None  # no row for that setup

    analysis = _imaging_analysis(masked_imaging_7x7, "jax_cpu_sparse_fp64")
    row = _row(
        likelihood="imaging/rectangular",
        setup_id="jax_cpu_sparse_fp64",
        instrument="euclid",
    )

    assert analysis.resources_per_eval() == {
        "cpus_per_eval": row["cpus_per_eval"],
        "peak_rss_gb": row["peak_rss_gb"],
    }


def test__analysis__resources_per_eval__unmatched__largest_peak_rss(masked_imaging_7x7):
    analysis = _imaging_analysis(masked_imaging_7x7, "jax_cpu_sparse_fp64")
    analysis.dataset = None  # no size to match

    rows = [
        r
        for r in re_mod.load_runtime_estimates()["rows"]
        if r["likelihood"] == "imaging/rectangular"
        and r["setup_id"] == "jax_cpu_sparse_fp64"
    ]

    assert analysis.resources_per_eval()["peak_rss_gb"] == max(
        r["peak_rss_gb"] for r in rows
    )


def test__analysis__resources_per_eval__never_raises(masked_imaging_7x7, monkeypatch):
    analysis = _imaging_analysis(masked_imaging_7x7, "jax_cpu_sparse_fp64")

    def boom(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(re_mod, "matched_rows_for", boom)

    assert analysis.resources_per_eval() is None
    assert analysis.runtime_estimate(search=Nautilus(), model=None) is None


def test__analysis__interferometer__estimate_line(interferometer_7):
    analysis = al.AnalysisInterferometer(dataset=interferometer_7)
    analysis.setup = al.Setup.from_setup_id("numba_cpu_sparse_fp64_wtilde")

    line = analysis.runtime_estimate(
        search=Nautilus(), model=_rectangular_model(), number_of_cores=4
    )

    assert line.startswith("Estimate: ~231 ms/eval (sma cell)")
    assert "η=0.62 on 4 cores" in line
    assert line.endswith("uncalibrated (held-out 0.2–0.5×)")


def test__analysis__point__estimate_line_from_positions_fit(point_dataset):
    analysis = al.AnalysisPoint(
        dataset=point_dataset,
        solver=al.m.MockPointSolver(model_positions=point_dataset.positions),
        fit_positions_cls=al.FitPositionsSource,
    )
    analysis.setup = al.Setup.from_setup_id("jax_cpu_na_fp64")

    line = analysis.runtime_estimate(
        search=Nautilus(), model=_rectangular_model(), number_of_cores=1
    )

    assert line.startswith("Estimate: ~0.065 ms/eval (simple cell)")
    assert "1 worker (JAX)" in line
    assert analysis.resources_per_eval()["peak_rss_gb"] == pytest.approx(0.5467)


def test__hook__resolve_fit_parallel_logs_estimate(masked_imaging_7x7, caplog):
    pytest.importorskip("numba")
    from autofit.non_linear.search.runtime import resolve_fit_parallel

    model = _rectangular_model()
    analysis = al.AnalysisImaging(
        dataset=masked_imaging_7x7,
        setup="numba_cpu_sparse_fp64",
        raise_inversion_positions_likelihood_exception=False,
    )
    analysis.modify_before_fit(paths=af.DirectoryPaths(), model=model)

    search = Nautilus(n_live=150, number_of_cores=8)

    with caplog.at_level(logging.INFO):
        resolution = resolve_fit_parallel(search=search, analysis=analysis, model=model)

    assert resolution.number_of_cores == 8
    estimates = [r.message for r in caplog.records if r.message.startswith("Estimate:")]
    assert len(estimates) == 1
    assert estimates[0].startswith("Estimate: ~283 ms/eval (euclid cell)")
    assert "η=0.55 on 8 cores" in estimates[0]


def test__hook__auto_cores_capped_by_peak_rss(masked_imaging_7x7, monkeypatch):
    pytest.importorskip("numba")
    from autofit.non_linear.parallel import pool

    model = _rectangular_model()
    analysis = al.AnalysisImaging(
        dataset=masked_imaging_7x7,
        setup="numba_cpu_sparse_fp64",
        raise_inversion_positions_likelihood_exception=False,
    )
    analysis.modify_before_fit(paths=af.DirectoryPaths(), model=model)
    # An hst-sized dataset (15380 masked pixels): the hst cell, 1.024 GB peak RSS.
    mask = al.Mask2D.circular(shape_native=(150, 150), pixel_scales=0.05, radius=3.5)
    analysis.dataset = al.Imaging(
        data=al.Array2D.ones(shape_native=(150, 150), pixel_scales=0.05),
        noise_map=al.Array2D.ones(shape_native=(150, 150), pixel_scales=0.05),
    ).apply_mask(mask)

    monkeypatch.setattr(pool, "detected_cores", lambda: 64)
    monkeypatch.setattr(pool, "available_memory_gb", lambda: 10.0)

    resolution = pool.resolve_number_of_cores("auto", analysis=analysis)

    assert resolution.number_of_cores == 9  # floor(10 / 1.024)
    assert resolution.reason == "memory"
