from pathlib import Path
import numpy as np
import pytest

import autofit as af
import autolens as al
from autoarray import Array2D

from autolens.analysis import result as res
from autolens.imaging.model.result import ResultImaging

directory = Path(__file__).resolve().parent


def test__max_log_likelihood_tracer(
    analysis_imaging_7x7,
    tracer_x2_plane_7x7,
):
    model = af.Collection(
        galaxies=af.Collection(
            lens=al.Galaxy(redshift=0.5, mass=al.mp.Isothermal),
            source=al.Galaxy(redshift=1.0, light=al.lp.Sersic),
        )
    )

    search = al.m.MockSearch(name="test_search_2")

    result = search.fit(model=model, analysis=analysis_imaging_7x7)

    assert isinstance(result.max_log_likelihood_tracer, al.Tracer)
    assert isinstance(result.max_log_likelihood_tracer.galaxies[0], al.Galaxy)


def test__source_plane_light_profile_centre(analysis_imaging_7x7):
    lens = al.Galaxy(redshift=0.5, light=al.lp.SersicSph(intensity=1.0))

    source = al.Galaxy(
        redshift=1.0, light=al.lp.SersicSph(centre=(1.0, 2.0), intensity=2.0)
    )

    tracer = al.Tracer(galaxies=[lens, source])

    samples_summary = al.m.MockSamplesSummary(max_log_likelihood_instance=tracer)

    result = res.Result(samples_summary=samples_summary, analysis=analysis_imaging_7x7)

    assert result.source_plane_light_profile_centre_from().in_list == [(1.0, 2.0)]

    source_0 = al.Galaxy(
        redshift=1.0,
        light=al.lp.SersicSph(centre=(1.0, 2.0), intensity=2.0),
        light1=al.lp.SersicSph(centre=(3.0, 4.0), intensity=2.0),
    )

    source_1 = al.Galaxy(
        redshift=1.0, light=al.lp.SersicSph(centre=(5.0, 6.0), intensity=2.0)
    )

    tracer = al.Tracer(galaxies=[lens, source_0, source_1])

    samples_summary = al.m.MockSamplesSummary(max_log_likelihood_instance=tracer)

    result = res.Result(samples_summary=samples_summary, analysis=analysis_imaging_7x7)

    assert result.source_plane_light_profile_centre_from().in_list == [(1.0, 2.0)]

    source_0 = al.Galaxy(
        redshift=1.0, light=al.lp.SersicSph(centre=(1.0, 2.0), intensity=2.0)
    )

    source_1 = al.Galaxy(
        redshift=2.0, light=al.lp.SersicSph(centre=(5.0, 6.0), intensity=2.0)
    )

    tracer = al.Tracer(galaxies=[lens, source_0, source_1])

    samples_summary = al.m.MockSamplesSummary(max_log_likelihood_instance=tracer)

    result = res.Result(samples_summary=samples_summary, analysis=analysis_imaging_7x7)

    assert result.source_plane_light_profile_centre_from().in_list == [(5.0, 6.0)]

    tracer = al.Tracer(galaxies=[al.Galaxy(redshift=0.5)])

    samples_summary = al.m.MockSamplesSummary(max_log_likelihood_instance=tracer)

    result = res.Result(samples_summary=samples_summary, analysis=analysis_imaging_7x7)

    assert result.source_plane_light_profile_centre_from() == None


def test__source_plane_inversion_centre(analysis_imaging_7x7):
    lens = al.Galaxy(redshift=0.5, light=al.lp.SersicSph(intensity=1.0))

    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform((3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )

    source = al.Galaxy(redshift=1.0, pixelization=pixelization)

    tracer = al.Tracer(galaxies=[lens, source])

    samples_summary = al.m.MockSamplesSummary(max_log_likelihood_instance=tracer)

    result = ResultImaging(
        samples_summary=samples_summary, analysis=analysis_imaging_7x7
    )

    inversion = result.max_log_likelihood_fit.inversion

    assert (
        result.source_plane_inversion_centre_from().in_list[0]
        == inversion.max_pixel_centre().in_list[0]
    )

    lens = al.Galaxy(redshift=0.5, light=al.lp.SersicSph(intensity=1.0))
    source = al.Galaxy(redshift=1.0)

    tracer = al.Tracer(galaxies=[lens, source])

    samples_summary = al.m.MockSamplesSummary(max_log_likelihood_instance=tracer)

    result = ResultImaging(
        samples_summary=samples_summary, analysis=analysis_imaging_7x7
    )

    assert result.source_plane_inversion_centre_from() == None

    lens = al.Galaxy(redshift=0.5, light=al.lp_linear.Sersic())
    source = al.Galaxy(redshift=1.0)

    tracer = al.Tracer(galaxies=[lens, source])

    samples_summary = al.m.MockSamplesSummary(max_log_likelihood_instance=tracer)

    result = ResultImaging(
        samples_summary=samples_summary, analysis=analysis_imaging_7x7
    )

    assert result.source_plane_inversion_centre_from() == None


def test__source_plane_centre(analysis_imaging_7x7):
    lens = al.Galaxy(redshift=0.5, light=al.lp.SersicSph(intensity=1.0))

    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform((3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )

    source = al.Galaxy(
        redshift=1.0,
        light=al.lp.SersicSph(centre=(9.0, 8.0), intensity=2.0),
        pixelization=pixelization,
    )

    tracer = al.Tracer(galaxies=[lens, source])

    samples_summary = al.m.MockSamplesSummary(max_log_likelihood_instance=tracer)

    result = ResultImaging(
        samples_summary=samples_summary, analysis=analysis_imaging_7x7
    )

    assert result.source_plane_centre_from().in_list[0] == pytest.approx(
        (-0.916666673333, -0.916666), 1.0e-4
    )


def test__image_plane_multiple_image_positions(analysis_imaging_7x7):
    lens = al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(
            centre=(0.001, 0.001), einstein_radius=1.0, ell_comps=(0.0, 0.111111)
        ),
    )

    source = al.Galaxy(
        redshift=1.0,
        light1=al.lp.SersicSph(centre=(0.0, 0.05), intensity=2.0),
    )

    tracer = al.Tracer(galaxies=[lens, source])

    samples_summary = al.m.MockSamplesSummary(max_log_likelihood_instance=tracer)

    result = ResultImaging(
        samples_summary=samples_summary, analysis=analysis_imaging_7x7
    )

    multiple_images = result.image_plane_multiple_image_positions()

    assert pytest.approx((0.968719, 0.366210), 1.0e-2) in multiple_images.in_list


def _result_with_stubbed_solver(monkeypatch, analysis_imaging_7x7, solver_output):
    """
    A `Result` whose point solver returns `solver_output` verbatim, so the
    handling of the solver's output can be tested independent of the lens model.
    """
    tracer = al.Tracer(
        galaxies=[
            al.Galaxy(
                redshift=0.5,
                mass=al.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.0),
            ),
            al.Galaxy(redshift=1.0, bulge=al.lp.SersicSph(centre=(0.1, 0.1))),
        ]
    )

    samples_summary = al.m.MockSamplesSummary(max_log_likelihood_instance=tracer)

    result = res.Result(samples_summary=samples_summary, analysis=analysis_imaging_7x7)

    monkeypatch.setattr(
        res.PointSolver, "solve", lambda self, *args, **kwargs: solver_output
    )

    return result


def test__image_plane_multiple_image_positions__inf_padded_single_image_triggers_recovery(
    monkeypatch, analysis_imaging_7x7
):
    """
    Under JAX the point solver returns a static-shape array padded with (inf, inf)
    rows, so a single-image solve has more than one row. The single-image recovery
    must count finite rows, not array rows.
    """
    padded = np.full((20, 2), np.inf)
    padded[0] = (1.0, 0.5)

    result = _result_with_stubbed_solver(monkeypatch, analysis_imaging_7x7, padded)

    recovered = al.Grid2DIrregular(values=[(1.0, 0.5), (-0.9, -0.4)])
    calls = []

    def _recovery(*args, **kwargs):
        calls.append(True)
        return recovered

    monkeypatch.setattr(
        result, "image_plane_multiple_image_positions_for_single_image_from", _recovery
    )

    positions = result.image_plane_multiple_image_positions()

    assert calls == [True]
    assert positions.in_list == recovered.in_list


def test__image_plane_multiple_image_positions__inf_padded_double_returns_finite_rows(
    monkeypatch, analysis_imaging_7x7
):
    padded = np.full((20, 2), np.inf)
    padded[0] = (1.0, 0.5)
    padded[1] = (-0.9, -0.4)

    result = _result_with_stubbed_solver(monkeypatch, analysis_imaging_7x7, padded)

    positions = result.image_plane_multiple_image_positions()

    assert positions.in_list == [(1.0, 0.5), (-0.9, -0.4)]
    assert np.isfinite(positions.array).all()


def test__image_plane_multiple_image_positions__unpadded_double_unchanged(
    monkeypatch, analysis_imaging_7x7
):
    solved = np.array([[1.0, 0.5], [-0.9, -0.4]])

    result = _result_with_stubbed_solver(monkeypatch, analysis_imaging_7x7, solved)

    positions = result.image_plane_multiple_image_positions()

    assert positions.in_list == [(1.0, 0.5), (-0.9, -0.4)]


def test__image_plane_multiple_image_positions_for_single_image_from__counts_finite_rows(
    monkeypatch, analysis_imaging_7x7
):
    """
    The inward walk must also count finite rows: an inf-padded single image is not
    a success, and a padded double returns only its finite rows.
    """
    single = np.full((20, 2), np.inf)
    single[0] = (1.0, 0.5)
    double = np.full((20, 2), np.inf)
    double[0] = (0.3, 0.2)
    double[1] = (-0.3, -0.2)

    outputs = iter([single, single, double])

    result = _result_with_stubbed_solver(monkeypatch, analysis_imaging_7x7, None)
    monkeypatch.setattr(
        res.PointSolver, "solve", lambda self, *args, **kwargs: next(outputs)
    )

    positions = result.image_plane_multiple_image_positions_for_single_image_from()

    assert positions.in_list == [(0.3, 0.2), (-0.3, -0.2)]


def test__positions_threshold_from(analysis_imaging_7x7):
    tracer = al.Tracer(
        galaxies=[
            al.Galaxy(
                redshift=0.5,
                mass=al.mp.Isothermal(
                    centre=(0.1, 0.0), einstein_radius=1.0, ell_comps=(0.0, 0.0)
                ),
            ),
            al.Galaxy(redshift=1.0, bulge=al.lp.SersicSph(centre=(0.0, 0.0))),
        ]
    )

    samples_summary = al.m.MockSamplesSummary(max_log_likelihood_instance=tracer)

    result = res.Result(samples_summary=samples_summary, analysis=analysis_imaging_7x7)

    # Knife-edge pin: the fixture is exactly mirror-symmetric (lens centre and source on
    # x = 0), so the point solver's branch is decided by sub-ULP tie-breaking; PyAutoArray#519's
    # exact identity transform at angle 0 selected the other branch (positions_threshold
    # 0.0019501455 -> 0.0019534291, positions[1].x sign flip, magnitude bit-identical). Values
    # are the exact-transform branch; a 1e-15 centre nudge flips them. The value moved again
    # (0.0019534291 -> 0.00098983199) when the Isothermal q <= 0.99999 clamp was removed
    # (PyAutoGalaxy#631), as the fixture is now exactly circular. See PyAutoLens#721.
    assert result.positions_threshold_from() == pytest.approx(0.00098983199, 1.0e-4)
    assert result.positions_threshold_from(factor=5.0) == pytest.approx(
        0.0049491599483, 1.0e-4
    )
    assert result.positions_threshold_from(minimum_threshold=10.0) == pytest.approx(
        10.0, 1.0e-4
    )
    assert result.positions_threshold_from(
        positions=al.Grid2DIrregular([(0.0, 0.0)])
    ) == pytest.approx(0.0, 1.0e-4)


def test__positions_likelihood_from(analysis_imaging_7x7):
    tracer = al.Tracer(
        galaxies=[
            al.Galaxy(
                redshift=0.5,
                mass=al.mp.Isothermal(
                    centre=(0.1, 0.0), einstein_radius=1.0, ell_comps=(0.0, 0.0)
                ),
            ),
            al.Galaxy(redshift=1.0, bulge=al.lp.SersicSph(centre=(0.0, 0.0))),
        ]
    )

    samples_summary = al.m.MockSamplesSummary(max_log_likelihood_instance=tracer)

    result = res.Result(samples_summary=samples_summary, analysis=analysis_imaging_7x7)

    positions_likelihood = result.positions_likelihood_from(
        factor=0.1, minimum_threshold=0.2
    )

    assert isinstance(positions_likelihood, al.PositionsLH)
    assert positions_likelihood.threshold == pytest.approx(0.2, 1.0e-4)


def test__positions_likelihood_from__skip_checks_returns_none_outside_test_mode(
    monkeypatch, analysis_imaging_7x7,
):
    monkeypatch.setenv("PYAUTO_SKIP_CHECKS", "1")
    monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)

    samples_summary = al.m.MockSamplesSummary(max_log_likelihood_instance=al.Tracer(galaxies=[]))
    result = res.Result(samples_summary=samples_summary, analysis=analysis_imaging_7x7)

    assert result.positions_likelihood_from(factor=0.1, minimum_threshold=0.2) is None


def test__positions_likelihood_from__skip_checks_returns_synthetic_in_test_mode(
    monkeypatch, analysis_imaging_7x7,
):
    monkeypatch.setenv("PYAUTO_SKIP_CHECKS", "1")
    monkeypatch.setenv("PYAUTO_TEST_MODE", "2")

    samples_summary = al.m.MockSamplesSummary(max_log_likelihood_instance=al.Tracer(galaxies=[]))
    result = res.Result(samples_summary=samples_summary, analysis=analysis_imaging_7x7)

    positions_likelihood = result.positions_likelihood_from(
        factor=0.1, minimum_threshold=0.2
    )

    assert isinstance(positions_likelihood, al.PositionsLH)
    assert positions_likelihood.threshold == pytest.approx(0.2, 1.0e-4)
    assert len(positions_likelihood.positions) == 2
    assert positions_likelihood.positions[0] == pytest.approx((1.0, 0.0))
    assert positions_likelihood.positions[1] == pytest.approx((-1.0, 0.0))


def test__positions_likelihood_from__test_mode_fallback(
    monkeypatch, analysis_imaging_7x7,
):
    monkeypatch.setenv("PYAUTO_TEST_MODE", "2")

    tracer = al.Tracer(
        galaxies=[
            al.Galaxy(
                redshift=0.5,
                mass=al.mp.Isothermal(
                    centre=(0.1, 0.0), einstein_radius=1.0, ell_comps=(0.0, 0.0)
                ),
            ),
            al.Galaxy(redshift=1.0, bulge=al.lp.SersicSph(centre=(0.0, 0.0))),
        ]
    )

    samples_summary = al.m.MockSamplesSummary(max_log_likelihood_instance=tracer)
    result = res.Result(samples_summary=samples_summary, analysis=analysis_imaging_7x7)

    empty_positions = al.Grid2DIrregular(np.empty((0, 2)))

    positions_likelihood = result.positions_likelihood_from(
        factor=0.1, minimum_threshold=0.2, positions=empty_positions
    )

    assert isinstance(positions_likelihood, al.PositionsLH)
    assert len(positions_likelihood.positions) == 2
    assert positions_likelihood.positions[0] == pytest.approx((1.0, 0.0))
    assert positions_likelihood.positions[1] == pytest.approx((-1.0, 0.0))


def test__positions_likelihood_from__mass_centre_radial_distance_min(
    analysis_imaging_7x7,
):
    tracer = al.Tracer(
        galaxies=[
            al.Galaxy(
                redshift=0.5,
                mass=al.mp.Isothermal(
                    centre=(0.1, 0.0), einstein_radius=1.0, ell_comps=(0.0, 0.0)
                ),
            ),
            al.Galaxy(redshift=1.0, bulge=al.lp.SersicSph(centre=(0.0, 0.0))),
        ]
    )

    samples_summary = al.m.MockSamplesSummary(max_log_likelihood_instance=tracer)

    result = res.Result(samples_summary=samples_summary, analysis=analysis_imaging_7x7)

    positions_likelihood = result.positions_likelihood_from(
        factor=0.1, minimum_threshold=0.2, mass_centre_radial_distance_min=0.1
    )

    assert isinstance(positions_likelihood, al.PositionsLH)
    assert len(positions_likelihood.positions) == 2
    # Knife-edge pin: the fixture is exactly mirror-symmetric (lens centre and source on
    # x = 0), so the point solver's branch is decided by sub-ULP tie-breaking; PyAutoArray#519's
    # exact identity transform at angle 0 selected the other branch (positions_threshold
    # 0.0019501455 -> 0.0019534291, positions[1].x sign flip, magnitude bit-identical). Values
    # are the exact-transform branch; a 1e-15 centre nudge flips them. The positions moved
    # again (branch and order) when the Isothermal q <= 0.99999 clamp was removed
    # (PyAutoGalaxy#631), as the fixture is now exactly circular. See PyAutoLens#721.
    assert positions_likelihood.positions[0] == pytest.approx(
        (9.99023438e-01, 5.63818622e-04), 1.0e-4
    )
    assert positions_likelihood.positions[1] == pytest.approx(
        (-1.00000000e00, 1.12763724e-03), 1.0e-4
    )


def test__results_include_mask__available_as_property(
    analysis_imaging_7x7, masked_imaging_7x7, samples_summary_with_result
):
    result = res.ResultDataset(
        samples_summary=samples_summary_with_result,
        analysis=analysis_imaging_7x7,
    )

    assert (result.mask == masked_imaging_7x7.mask).all()


def test___image_dict(analysis_imaging_7x7):
    galaxies = af.ModelInstance()
    galaxies.lens = al.Galaxy(redshift=0.5)
    galaxies.source = al.Galaxy(redshift=1.0)

    instance = af.ModelInstance()
    instance.galaxies = galaxies

    result = ResultImaging(
        samples_summary=al.m.MockSamplesSummary(max_log_likelihood_instance=instance),
        analysis=analysis_imaging_7x7,
    )

    image_dict = result.model_image_galaxy_dict

    assert isinstance(image_dict[str(("galaxies", "lens"))], Array2D)
    assert isinstance(image_dict[str(("galaxies", "source"))], Array2D)

    result.instance.galaxies.lens = al.Galaxy(redshift=0.5)

    image_dict = result.model_image_galaxy_dict

    assert (image_dict[str(("galaxies", "lens"))].native == np.zeros((7, 7))).all()
    assert isinstance(image_dict[str(("galaxies", "source"))], Array2D)


def test__positions_likelihood_from__loads_cached_positions_on_second_call(
    tmp_path, monkeypatch, analysis_imaging_7x7
):
    class _StubPaths:
        def __init__(self, files_path):
            self._files_path = files_path

        def preserve_in_zip(self, file_path):
            """No zip in the stub — mirrors AbstractPaths' no-zip no-op."""

    tracer = al.Tracer(
        galaxies=[
            al.Galaxy(
                redshift=0.5,
                mass=al.mp.Isothermal(
                    centre=(0.1, 0.0), einstein_radius=1.0, ell_comps=(0.0, 0.0)
                ),
            ),
            al.Galaxy(redshift=1.0, bulge=al.lp.SersicSph(centre=(0.0, 0.0))),
        ]
    )

    samples_summary = al.m.MockSamplesSummary(max_log_likelihood_instance=tracer)

    result = res.Result(samples_summary=samples_summary, analysis=analysis_imaging_7x7)
    result.paths = _StubPaths(files_path=tmp_path)

    first = result.positions_likelihood_from(factor=0.1, minimum_threshold=0.2)

    assert (tmp_path / "multiple_image_positions.json").exists()

    # The second call must load the cached positions — solving again raises.
    def _poison(*args, **kwargs):
        raise AssertionError("point solver re-ran — cached positions not used")

    result_cached = res.Result(
        samples_summary=samples_summary, analysis=analysis_imaging_7x7
    )
    result_cached.paths = _StubPaths(files_path=tmp_path)
    monkeypatch.setattr(
        result_cached, "image_plane_multiple_image_positions", _poison
    )

    second = result_cached.positions_likelihood_from(factor=0.1, minimum_threshold=0.2)

    assert isinstance(second, al.PositionsLH)
    assert second.positions.array == pytest.approx(first.positions.array, 1.0e-8)
    assert second.threshold == pytest.approx(first.threshold, 1.0e-8)


@pytest.mark.parametrize("form", ["flat", "collection", "absent", "none", "empty"])
def test__max_log_likelihood_fields__normalizes_model_slot(form, analysis_imaging_7x7):
    field = al.MassField(redshift=0.5, shear=al.mp.ExternalShear(gamma_1=0.05))
    instance = af.ModelInstance()
    if form == "flat":
        instance.fields = field
    elif form == "collection":
        instance.fields = af.ModelInstance({"field": field})
    elif form == "none":
        instance.fields = None
    elif form == "empty":
        instance.fields = af.ModelInstance()
    result = res.Result(
        samples_summary=al.m.MockSamplesSummary(max_log_likelihood_instance=instance),
        analysis=analysis_imaging_7x7,
    )
    expected = [field] if form in ("flat", "collection") else []
    assert result.max_log_likelihood_fields == expected
