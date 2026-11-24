from pathlib import Path

import numpy as np
import pytest

import autolens as al
import autolens.plot as aplt
from autolens.point.plot import point_solver_plots as psp


@pytest.fixture(name="tracer")
def make_tracer():
    return al.Tracer(
        galaxies=[
            al.Galaxy(
                redshift=0.5,
                mass=al.mp.Isothermal(
                    centre=(0.0, 0.0),
                    einstein_radius=1.6,
                    ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
                ),
            ),
            al.Galaxy(redshift=1.0, point_0=al.ps.Point(centre=(0.07, 0.07))),
        ]
    )


@pytest.fixture(name="solver")
def make_solver():
    """A coarse teaching grid: 6 refinement steps, quick to solve."""
    return al.PointSolver.for_grid(
        grid=al.Grid2D.uniform(shape_native=(11, 11), pixel_scales=0.5),
        pixel_scale_precision=0.01,
        magnification_threshold=0.1,
    )


@pytest.fixture(name="data")
def make_data(solver, tracer):
    return psp.point_solver_data_from(
        solver=solver,
        tracer=tracer,
        source_plane_coordinate=(0.07, 0.07),
        image_plane_lines=[],
    )


class _Step:
    def __init__(self, number):
        self.number = number


def test__select_steps__first_three_and_final():
    steps = [_Step(i) for i in range(6)]
    assert [s.number for s in psp.select_steps(steps)] == [0, 1, 2, 5]

    steps = [_Step(i) for i in range(3)]
    assert [s.number for s in psp.select_steps(steps)] == [0, 1, 2]

    steps = [_Step(0)]
    assert [s.number for s in psp.select_steps(steps)] == [0]


def test__triangles_xy_from__swaps_yx_and_drops_nan_rows():
    triangles = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]],
        ]
    )
    xy = psp.triangles_xy_from(triangles)
    assert xy.shape == (1, 3, 2)
    assert xy[0, 0].tolist() == [2.0, 1.0]


def test__point_solver_data_from(data, solver):
    assert data.n_steps == solver.n_steps
    assert [s.number for s in data.steps] == [0, 1, 2, solver.n_steps - 1]

    # Four images, plus the lens-centre candidate the magnification threshold rejects.
    assert len(data.solved) == 4
    assert len(data.rejected) == 1
    assert np.linalg.norm(data.rejected[0]) < 0.1

    # The coarse teaching grid's triangles are drawable, so its own steps are shown.
    assert data.display_pixel_scale is None


def test__extent__spans_the_sources_images_and_rejected_candidates(data):
    """
    Every panel shares one square extent covering the source's images and rejected candidates (not the solver's
    full grid), padded so none sits on the edge.
    """
    x_min, x_max, y_min, y_max = data.extent
    assert x_max - x_min == pytest.approx(y_max - y_min)

    points = np.vstack([data.solved, data.rejected])
    margin = 0.1 * (x_max - x_min)
    assert np.all(points[:, 1] > x_min + margin) and np.all(
        points[:, 1] < x_max - margin
    )
    assert np.all(points[:, 0] > y_min + margin) and np.all(
        points[:, 0] < y_max - margin
    )


def test__extent__grid_only_just_larger_than_the_images_is_shown_whole(data, solver):
    x_min, x_max, y_min, y_max = data.extent
    assert x_min <= solver.x_min and x_max >= solver.x_max
    assert y_min <= solver.y_min and y_max >= solver.y_max
    assert x_max - x_min < 1.2 * (solver.x_max - solver.x_min)


def test__source_extent_from__cluster_scale_span_is_the_images_not_the_grid():
    points = np.array([[14.6, -14.8], [-16.4, 19.0], [0.0, 0.0]])
    x_min, x_max, y_min, y_max = psp.source_extent_from(points)

    assert x_min < -14.8 and x_max > 19.0 and y_min < -16.4 and y_max > 14.6
    assert x_max - x_min < 60.0
    assert psp.source_extent_from(np.zeros((0, 2))) is None


@pytest.mark.parametrize(
    "centre, half_width",
    [((2.2, 2.5), 20.5), ((0.31, -0.07), 1.9), ((-3.1, 7.4), 0.42)],
)
def test__round_extent_from__ticks_are_round_and_contain_the_square(centre, half_width):
    from autoarray.plot.utils import _inward_ticks

    x_min, x_max, y_min, y_max = psp.round_extent_from(centre, half_width)

    assert x_max - x_min == pytest.approx(y_max - y_min)
    assert x_min <= centre[1] - half_width and x_max >= centre[1] + half_width
    assert y_min <= centre[0] - half_width and y_max >= centre[0] + half_width
    # No more than twice as wide as asked for.
    assert x_max - x_min <= 4.0 * half_width

    for lo, hi in ((x_min, x_max), (y_min, y_max)):
        ticks = _inward_ticks(lo, hi, 0.75, 3)
        step = ticks[1] - ticks[0]
        power = 10.0 ** np.floor(np.log10(step))
        # The tick values are round: two significant figures at most.
        assert np.allclose(np.round(ticks / power * 10.0), ticks / power * 10.0)


def test__split_kept__rejected_triangles_converge_on_rejected_candidate(data):
    kept, rejected = psp.split_kept_from(
        data.steps[0], data.final_centres, data.rejected
    )
    assert len(kept) > 0
    assert len(rejected) > 0
    centre = rejected.mean(axis=(0, 1))
    assert np.linalg.norm(centre) < 0.5


def test__too_small_to_draw__final_step_only(data):
    assert not psp.is_too_small_to_draw(data.steps[0], data, panel_inches=5.0)
    assert psp.is_too_small_to_draw(data.steps[-1], data, panel_inches=5.0)


def test__draw_panel__final_step_draws_centroid_markers(data):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    marked = psp.draw_point_solver_panel(ax, data.steps[-1], data, panel_inches=5.0)
    assert marked
    assert any(
        getattr(c, "get_paths", None) and c.get_offsets().shape[0] > 0
        for c in ax.collections
    )
    assert ax.get_xlabel() == "" and ax.get_ylabel() == ""
    assert all('"' in label.get_text() for label in ax.get_xticklabels())
    plt.close(fig)


def test__legend__too_small_entry_only_when_marked(data):
    labels = [h.get_label() for h in psp.legend_handles_from(data, marked=False)]
    assert "Kept (too small to draw)" not in labels
    assert "Kept, then rejected (μ < 0.1)" in labels

    labels = [h.get_label() for h in psp.legend_handles_from(data, marked=True)]
    assert "Kept (too small to draw)" in labels


def test__plot_point_solver__still(solver, tracer, plot_patch, tmp_path):
    aplt.plot_point_solver(
        solver=solver,
        tracer=tracer,
        source_plane_coordinate=(0.07, 0.07),
        positions=al.Grid2DIrregular([(1.0, 1.0)]),
        image_plane_lines=[],
        output_path=tmp_path,
        output_format="png",
    )
    assert str(tmp_path / "point_solver.png") in plot_patch.paths


def test__plot_point_solver__animate_writes_looping_gif(solver, tracer, tmp_path):
    from PIL import Image

    aplt.plot_point_solver(
        solver=solver,
        tracer=tracer,
        source_plane_coordinate=(0.07, 0.07),
        image_plane_lines=[],
        output_path=tmp_path,
        animate=True,
    )
    gif = tmp_path / "point_solver.gif"
    assert gif.is_file()

    with Image.open(gif) as image:
        durations = []
        for i in range(image.n_frames):
            image.seek(i)
            durations.append(image.info["duration"])
        assert image.info.get("loop") == 0

    # One frame per step (0, 1, 2, final), the identical held final frames merged into one longer frame.
    assert len(durations) == 4
    assert durations[-1] == pytest.approx(
        psp.FINAL_FRAME_REPEATS * durations[0], rel=0.05
    )


def test__plot_point_solver__animate_without_path_returns_animation(solver, tracer):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    animation = aplt.plot_point_solver(
        solver=solver,
        tracer=tracer,
        source_plane_coordinate=(0.07, 0.07),
        image_plane_lines=[],
        animate=True,
    )
    assert isinstance(animation, FuncAnimation)
    animation._draw_was_started = (
        True  # never rendered here, which matplotlib would otherwise warn about
    )
    plt.close("all")


def test__warn_if_near_jax_cap(data, caplog):
    import logging

    data.max_kept = 3
    with caplog.at_level(logging.WARNING):
        psp.warn_if_near_jax_cap(data)
    assert "MAX_CONTAINING_SIZE" not in caplog.text

    data.max_kept = 19
    with caplog.at_level(logging.WARNING):
        psp.warn_if_near_jax_cap(data)
    assert "MAX_CONTAINING_SIZE" in caplog.text


def _number_texts(ax):
    return [t for t in ax.texts if t.get_text().isdigit()]


def _gap_in_points(fig, ax, text, x, y):
    """Distance (points) from an image's display position to the nearest point of its number's box."""
    bbox = text.get_window_extent(renderer=fig.canvas.get_renderer())
    px, py = ax.transData.transform((x, y))
    gap_x = max(bbox.x0 - px, 0.0, px - bbox.x1)
    gap_y = max(bbox.y0 - py, 0.0, py - bbox.y1)
    return np.hypot(gap_x, gap_y) * 72.0 / fig.dpi


@pytest.mark.parametrize("crowded", [False, True])
def test__image_numbers_sit_next_to_their_own_images(data, crowded):
    """
    Every step-0 image number is drawn a few points from its own image, never moved away to find space, even when
    images crowd together (the crowded case packs the images into a small part of the panel).
    """
    import matplotlib.pyplot as plt

    if crowded:
        data.solved = np.array(
            [[0.2, 0.0], [0.25, 0.12], [0.1, 0.22], [-0.05, 0.15], [0.3, -0.1]]
        )

    fig, ax = plt.subplots(figsize=(6, 6))
    psp.draw_point_solver_panel(ax, data.steps[0], data, panel_inches=5.0)
    fig.canvas.draw()

    texts = _number_texts(ax)
    assert [t.get_text() for t in texts] == [
        str(i) for i in range(1, len(data.solved) + 1)
    ]
    for text, (y, x) in zip(texts, data.solved):
        assert text.xy == pytest.approx((x, y))
        assert _gap_in_points(fig, ax, text, x, y) < 10.0
    plt.close(fig)


def test__rejected_candidates__r_labels_and_no_annotation_box(data):
    """
    Rejected candidates are labelled ``r1, r2, ...`` in grey italic next to each candidate, with no annotation box
    or arrows.
    """
    import matplotlib.pyplot as plt

    data.rejected = np.array([[0.0, 0.0], [1.0, -1.0], [-1.0, 0.5]])
    fig, ax = plt.subplots(figsize=(6, 6))
    psp.draw_point_solver_panel(ax, data.steps[0], data, panel_inches=5.0)
    fig.canvas.draw()

    labels = [t for t in ax.texts if t.get_text().startswith("r")]
    assert [t.get_text() for t in labels] == ["r1", "r2", "r3"]
    for text, (y, x) in zip(labels, data.rejected):
        assert text.xy == pytest.approx((x, y))
        assert text.get_style() == "italic"
        assert text.get_color() == psp.REJECTED_LABEL_COLOR
        assert _gap_in_points(fig, ax, text, x, y) < 10.0

    assert not any("magnification_threshold" in t.get_text() for t in ax.texts)
    assert not any(t.get_text() == "" for t in ax.texts)
    assert all(t.get_transform() != ax.transAxes for t in ax.texts)
    plt.close(fig)


def test__display_pixel_scale_from__tenth_of_span_capped_by_closest_separation():
    extent = [-5.0, 5.0, -5.0, 5.0]
    far = np.array([[3.0, 3.0], [-3.0, -3.0], [3.0, -3.0]])
    assert psp.display_pixel_scale_from(extent, far) == pytest.approx(1.0)

    close = np.vstack([far, [[3.0, 3.8]]])
    assert psp.display_pixel_scale_from(extent, close) == pytest.approx(
        0.8 / psp.DISPLAY_SEPARATION_FRACTION
    )


def test__recovers():
    reference = np.array([[1.0, 1.0], [-1.0, 0.5]])
    assert psp.recovers(reference, reference[::-1] + 0.001, tolerance=0.01)
    assert not psp.recovers(reference, reference[:1], tolerance=0.01)
    assert not psp.recovers(reference, reference + 0.1, tolerance=0.01)
    assert psp.recovers(np.zeros((0, 2)), np.zeros((0, 2)), tolerance=0.01)


@pytest.fixture(name="fine_solver")
def make_fine_solver():
    """A grid whose step-0 triangles are too small to draw on the source's span."""
    return al.PointSolver.for_grid(
        grid=al.Grid2D.uniform(shape_native=(200, 200), pixel_scales=0.02),
        pixel_scale_precision=0.01,
        magnification_threshold=0.1,
    )


def test__display_grid__fine_solver_is_re_solved_on_a_coarse_grid(fine_solver, tracer):
    real = psp.point_solver_data_from(
        solver=fine_solver,
        tracer=tracer,
        source_plane_coordinate=(0.07, 0.07),
        image_plane_lines=[],
        panel_inches=1e6,  # large enough that the solver's own triangles are drawable
    )
    assert real.display_pixel_scale is None
    assert real.n_steps == fine_solver.n_steps

    data = psp.point_solver_data_from(
        solver=fine_solver,
        tracer=tracer,
        source_plane_coordinate=(0.07, 0.07),
        image_plane_lines=[],
    )
    assert psp.is_too_small_to_draw(real.steps[0], data, psp.STILL_PANEL_INCHES)

    expected = psp.display_pixel_scale_from(
        data.extent, np.vstack([real.solved, real.rejected])
    )
    assert data.display_pixel_scale == pytest.approx(expected)
    assert psp._edge_of(data.steps[0]) == pytest.approx(expected)
    assert not psp.is_too_small_to_draw(data.steps[0], data, psp.STILL_PANEL_INCHES)

    # The display solve finds the solver's own images and rejected candidate.
    assert psp.recovers(real.solved, data.solved, tolerance=0.02)
    assert psp.recovers(real.rejected, data.rejected, tolerance=0.02)
    assert data.extent == real.extent

    labels = [h.get_label() for h in psp.legend_handles_from(data, marked=False)]
    assert "Triangles tested (display grid)" in labels


def test__display_grid__falls_back_to_the_solvers_steps_when_it_misses_an_image(
    fine_solver, tracer, monkeypatch
):
    def missing_solver(solver, extent, pixel_scale):
        # Tiles only a corner of the extent, so the display solve misses images.
        return al.PointSolver.for_limits_and_scale(
            y_min=extent[3] - pixel_scale,
            y_max=extent[3] + pixel_scale,
            x_min=extent[1] - pixel_scale,
            x_max=extent[1] + pixel_scale,
            scale=pixel_scale,
            pixel_scale_precision=solver.pixel_scale_precision,
            magnification_threshold=solver.magnification_threshold,
        )

    monkeypatch.setattr(psp, "display_solver_from", missing_solver)

    data = psp.point_solver_data_from(
        solver=fine_solver,
        tracer=tracer,
        source_plane_coordinate=(0.07, 0.07),
        image_plane_lines=[],
    )
    assert data.display_pixel_scale is None
    assert data.n_steps == fine_solver.n_steps
    assert psp._edge_of(data.steps[0]) == pytest.approx(fine_solver.scale, rel=1e-6)
    assert len(data.solved) == 4

    labels = [h.get_label() for h in psp.legend_handles_from(data, marked=False)]
    assert "Triangles tested" in labels


def test__critical_curves__source_plane_own_curves_when_not_given(solver, tracer):
    data = psp.point_solver_data_from(
        solver=solver,
        tracer=tracer,
        source_plane_coordinate=(0.07, 0.07),
    )
    assert len(data.lines) >= 1

    labels = [h.get_label() for h in psp.legend_handles_from(data, marked=False)]
    assert "Critical curve" in labels



def test__solver_panel__family_style_alters_only_observed_markers(data):
    from dataclasses import replace
    from matplotlib import pyplot as plt
    from matplotlib.colors import to_rgba

    data = replace(data, positions=data.solved)
    fig, axes = plt.subplots(1, 2)
    try:
        for ax, family_index, expected_color, expected_marker in (
            (axes[0], None, psp.OBSERVED_COLOR, "o"),
            (axes[1], 1, "#00778a", "D"),
        ):
            marked = psp.draw_point_solver_panel(
                ax, data.steps[0], data, panel_inches=4.0, family_index=family_index
            )
            handles = {
                h.get_label(): h
                for h in psp.legend_handles_from(data, marked, family_index=family_index)
            }
            observed = handles["Observed"]
            assert observed.get_marker() == expected_marker
            assert to_rgba(observed.get_markeredgecolor()) == to_rgba(expected_color)
            assert to_rgba(handles["Solved"].get_color()) == to_rgba(psp.SOLVED_COLOR)
            assert {text.get_text() for text in ax.texts} >= {"1", "r1"}
            observed_artist, solved_artist = ax.collections[-2:]
            assert observed_artist.get_edgecolors()[0] == pytest.approx(to_rgba(expected_color))
            assert solved_artist.get_facecolors()[0] == pytest.approx(to_rgba(psp.SOLVED_COLOR))
    finally:
        plt.close(fig)
