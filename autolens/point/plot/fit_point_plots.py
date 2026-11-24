import logging

import numpy as np
from typing import Optional

from autoarray.plot.utils import subplots, save_figure, conf_subplot_figsize, tight_layout

logger = logging.getLogger(__name__)


def subplot_fit(
    fit,
    output_path: Optional[str] = None,
    output_format: str = None,
    image_plane_lines=None,
    image_plane_line_colors=None,
    source_plane_lines=None,
    source_plane_line_colors=None,
    title_prefix: str = None,
    solver=None,
    curves=None,
    family_index=None,
):
    """
    Produce ``fit.png`` of a `FitPointDataset`: the "Fit Positions" panel of
    :func:`~autolens.point.plot.multi_source_plots.plot_fit_positions` for this one source, plus a "Fit Fluxes"
    panel when the dataset has fluxes.

    The "Fit Positions" panel is drawn on the ``PointSolver`` grid extent with no axis labels. It shows the observed
    positions (outlined amber circles, σ circles, σ in the legend) numbered 1..N as on every point-source figure; for an
    image-plane fit the model positions (red ✕) at their true position, each joined to its observed image by a
    true-length residual line of the fit's own pairing (solid for hard pairing, dashed for the soft all-pairs
    mixture), and unmatched model images (hollow ✕); the critical curves of the plane the point source is in (dark
    and thick); the mass centres; and the fit's figure of merit (``χ²``, or ``−2 ln L`` for the all-pairs mixture).
    The legend sits in the panel's emptiest corner.

    The fit's source-plane panel and per-image residual ranking are drawn by
    :func:`~autolens.point.plot.multi_source_plots.subplot_fit_per_image` (``fit_per_image.png``).

    Parameters
    ----------
    fit : FitPointDataset
        The point-source dataset fit to visualise.
    output_path : str, optional
        Directory in which to save the figure.  If ``None`` the figure is
        not saved to disk.
    output_format : str, optional
        Image format passed to :func:`~autoarray.plot.utils.save_figure`.
    image_plane_lines, image_plane_line_colors, source_plane_lines, source_plane_line_colors
        Kept for API compatibility. Pass ``image_plane_lines=[]`` (or ``curves={}``) to draw no curves; otherwise
        the curves of the point source's own plane are drawn (``curves``).
    title_prefix : str, optional
        Text prepended to every panel title.
    solver : PointSolver, optional
        The solver whose grid sets the positions panel extent. Defaults to
        ``fit.solver``.
    curves : dict, optional
        The per-plane curves of :func:`~autolens.point.plot.point_sources.curves_for_planes`; computed on the
        panel extent if ``None``.
    """
    from autoarray.plot.yx import plot_yx
    from autolens.point.plot import multi_source_plots as msp
    from autolens.point.plot import point_sources as ps

    if solver is None:
        solver = getattr(fit, "solver", None)

    records = ps.records_from_fits([fit])
    colormap = ps.assign_styles(records)
    extent = ps.union_extent_from(records, solver=solver)

    no_curves_requested = (
        image_plane_lines is not None
        and len(image_plane_lines) == 0
        and not source_plane_lines
    )
    if curves is None:
        if no_curves_requested:
            curves = {}
        elif image_plane_lines:
            # Pre-computed lines given by the caller: draw them as this source's curves.
            curves = {
                records[0].plane_index: _plane_curves_from_lines(
                    fit,
                    records[0].plane_index,
                    image_plane_lines,
                    image_plane_line_colors,
                    source_plane_lines,
                    source_plane_line_colors,
                )
            }
        else:
            curves = ps.curves_for_records(fit.tracer, records, extent)

    has_fluxes = fit.dataset.fluxes is not None
    n = 2 if has_fluxes else 1

    if n == 1:
        from autoarray.plot.utils import conf_figsize

        fig, ax = subplots(1, 1, figsize=conf_figsize("figures"))
        axes_flat = [ax]
    else:
        fig, axes = subplots(1, n, figsize=conf_subplot_figsize(1, n))
        axes_flat = list(np.array(axes).flatten())

    _prefix = f"{title_prefix.rstrip()} " if title_prefix else ""

    msp.draw_fit_positions(
        axes_flat[0],
        records,
        curves,
        extent,
        fit.tracer,
        title=f"{_prefix}Fit Positions ({fit.dataset.name})",
        colormap=colormap,
        family_index=family_index,
    )

    if has_fluxes:
        y = np.array(fit.dataset.fluxes)
        x = np.arange(len(y))
        plot_yx(
            y=y,
            x=x,
            ax=axes_flat[1],
            title=f"{_prefix}Fit Fluxes ({fit.dataset.name})",
            output_path=None,
            output_filename="fit_point_fluxes",
            output_format=output_format,
        )
        tight_layout()

    save_figure(fig, path=output_path, filename="fit", format=output_format)


def _plane_curves_from_lines(
    fit,
    plane_index,
    image_plane_lines,
    image_plane_line_colors,
    source_plane_lines,
    source_plane_line_colors,
):
    """
    Wraps pre-computed critical-curve lines (``"white"`` tangential, ``"yellow"`` radial) as the ``PlaneCurves`` of
    the point source's plane.
    """
    from autolens.point.plot import point_sources as ps
    from autolens.point.plot import source_plane_panel

    tangential, radial = source_plane_panel.caustics_from_lines(
        image_plane_lines, image_plane_line_colors
    )
    tangential_caustics, radial_caustics = source_plane_panel.caustics_from_lines(
        source_plane_lines, source_plane_line_colors
    )
    return ps.PlaneCurves(
        plane_index=plane_index,
        redshift=float(fit.tracer.planes[plane_index].redshift),
        tangential=tangential,
        radial=radial,
        tangential_caustics=tangential_caustics,
        radial_caustics=radial_caustics,
    )


def source_plane_caustics_from(
    fit,
    solver=None,
    lines_given: bool = False,
    source_plane_lines=None,
    source_plane_line_colors=None,
):
    """
    Returns the ``(tangential, radial)`` caustics drawn on the source-plane panel of ``fit``: those of the plane
    the point source is in.

    The pre-computed ``source_plane_lines`` are the caustics of the tracer's last plane, so they are reused when
    the point source is in that plane. Otherwise (a point in an earlier plane of a multi-plane tracer, or no lines
    passed) the caustics of the point's own plane are computed on ``critical_curve_grid_from``. If lines were given
    and both are empty (``[]``), no caustics are drawn.
    """
    from autolens.lens.plot.critical_curves import critical_curve_grid_from
    from autolens.point.plot import source_plane_panel

    plane_index = fit.positions.plane_index
    last_plane = len(fit.tracer.planes) - 1
    in_last_plane = plane_index in (-1, last_plane)

    if lines_given and source_plane_lines is not None and in_last_plane:
        return source_plane_panel.caustics_from_lines(
            source_plane_lines, source_plane_line_colors
        )
    if lines_given and source_plane_lines is not None and len(source_plane_lines) == 0:
        return [], []

    return source_plane_panel.point_plane_caustics_from(
        tracer=fit.tracer,
        plane_index=plane_index,
        grid=critical_curve_grid_from(dataset=fit.dataset, solver=solver),
    )


def plot_image_with_points(
    image,
    dataset,
    fit=None,
    output_path: Optional[str] = None,
    output_format: str = None,
    image_plane_lines=None,
    image_plane_line_colors=None,
    title_prefix: str = None,
    curves=None,
    solver=None,
    family_index=None,
):
    """
    Plot an observed image of a lensed point source (e.g. a quasar) with its point-source positions overlaid.

    This is :func:`~autolens.point.plot.multi_source_plots.plot_image_with_points` for one source: the observed
    positions as outlined amber circles on the grayscale observed image, numbered; with a fit, the model
    positions (red ✕, image-plane fits only, as a source-plane fit's model data lie in the source plane) joined to
    their observed image by residual lines and the critical curves of the point source's own plane. There is no
    legend.

    - ``fit=None`` (dataset-only): saved as ``image_with_points``.
    - ``fit`` given: saved as ``fit_image_with_points``, so it does not overwrite the dataset-only figure.

    The image is used only for visualization: it does not enter the point-source likelihood.

    Parameters
    ----------
    image : Array2D
        The observed image the positions are plotted over.
    dataset : PointDataset
        The point-source dataset whose observed positions are overlaid.
    fit : FitPointDataset, optional
        If given, its model positions and curves are overlaid and the figure is saved as ``fit_image_with_points``.
    output_path : str, optional
        Directory in which to save the figure.  If ``None`` the figure is
        not saved to disk.
    output_format : str, optional
        Image format passed to :func:`~autoarray.plot.utils.save_figure`.
    image_plane_lines, image_plane_line_colors
        Pre-computed critical-curve lines, drawn only if the per-plane ``curves`` cannot be computed.
    curves : dict, optional
        The per-plane curves of :func:`~autolens.point.plot.point_sources.curves_for_planes`.
    """
    from autolens.point.plot import multi_source_plots as msp

    msp.plot_image_with_points(
        image=image,
        datasets=[dataset],
        fits=None if fit is None else [fit],
        curves=curves,
        solver=solver,
        output_path=output_path,
        output_format=output_format,
        title_prefix=title_prefix,
        image_plane_lines=image_plane_lines,
        image_plane_line_colors=image_plane_line_colors,
        family_index=family_index,
    )
