import logging
import numpy as np
from typing import Optional, List

import autoarray as aa
import autogalaxy as ag

from autogalaxy.util.plot_utils import plot_array
from autoarray.plot.array import _zoom_array_2d
from autoarray.plot.utils import subplots, save_figure, hide_unused_axes, conf_subplot_figsize, tight_layout, plot_regions
from autoarray.inversion.mappers.abstract import Mapper
from autoarray.inversion.plot.mapper_plots import plot_mapper
from autogalaxy.util.plot_utils import _critical_curves_from

# Canonical, single-source definition. Re-exported here so existing
# ``from autolens.imaging.plot.fit_imaging_plots import _compute_critical_curve_lines``
# imports keep working. It lives under autolens.lens.plot (empty package
# __init__) rather than autolens.plot to avoid a circular import through the
# autolens.plot aggregator __init__.
from autolens.lens.plot.critical_curves import _compute_critical_curve_lines

logger = logging.getLogger(__name__)


def _compute_critical_curves_from_fit(fit):
    """Compute critical-curve and caustic lines from a FitImaging object.

    Convenience wrapper around :func:`_compute_critical_curve_lines` that
    derives the tracer and grid from *fit* directly, using the fully unmasked
    image-plane grid so the curves cover the whole image extent.

    Returns the same 4-tuple as :func:`_compute_critical_curve_lines`.
    """
    tracer = fit.tracer_linear_light_profiles_to_light_profiles
    return _compute_critical_curve_lines(tracer, fit.mask.derive_grid.all_false)


def _get_source_vmax(fit):
    """
    Return the colour-scale maximum for source-plane panels.

    Computes the global maximum pixel value across all source-plane model
    images (planes with index >= 1), so that source and subtracted panels
    share a common colour scale.  Returns ``None`` when no source-plane
    model images exist (e.g. a lens-only fit) so callers can fall back to
    automatic scaling.

    Parameters
    ----------
    fit : FitImaging
        The imaging fit whose ``model_images_of_planes_list`` is inspected.

    Returns
    -------
    float or None
        Global maximum pixel value of all source-plane model images, or
        ``None`` if none are available.
    """
    try:
        return float(np.max([mi.array for mi in fit.model_images_of_planes_list[1:]]))
    except (ValueError, IndexError):
        return None


from autolens.lens.plot.tracer_plots import plane_image_from


def _plot_source_plane(fit, ax, plane_index, zoom_to_brightest=True,
                       colormap=None, use_log10=False, title=None,
                       lines=None, line_colors=None, vmax=None,
                       zoom_extent_scale: float = 1.0,
                       regions=None, region_alpha: float = 0.25,
                       region_labels=None):
    """
    Plot the source-plane image (or a blank inversion placeholder) into an axes.

    When the plane at ``plane_index`` does not contain a
    `~autoarray.Pixelization` (i.e. it is a parametric source), the source
    galaxy light profiles are evaluated on a plain uniform grid
    (``fit.mask.derive_grid.all_false``) — **not** a ray-traced grid.  This
    shows the source as it appears in its own plane, without lensing
    distortion.  :func:`~autolens.lens.plot.tracer_plots.plane_image_from`
    handles the optional zoom to the brightest region.  When the plane *does*
    contain a pixelization (an inversion source), the source reconstruction
    is rendered via :func:`~autoarray.inversion.plot.mapper_plots.plot_mapper`
    using ``zoom_to_brightest`` to control whether the view is zoomed in on
    the brightest pixels or shown at full extent.

    Parameters
    ----------
    fit : FitImaging
        The imaging fit providing the tracer, mask, and inversion state.
    ax : matplotlib.axes.Axes or None
        The axes into which the source-plane image is drawn.  Passing
        ``None`` is a no-op.
    plane_index : int
        Index of the plane in ``fit.tracer.planes`` to visualise.
    zoom_to_brightest : bool, optional
        For parametric sources, zooms the evaluation grid in on the brightest
        region of the source plane via :func:`plane_image_from`.  For inversion
        sources, zooms the colormap extent to the brightest reconstructed pixels.
    colormap : str, optional
        Matplotlib colormap name.
    use_log10 : bool, optional
        If ``True`` the colour scale is applied on a log10 stretch.
    regions : list, optional
        Source-plane regions to overlay as filled polygons, each a list of ``(N, 2)``
        ``(y, x)`` arrays -- e.g. the ``source_contours`` of a
        `~autoarray.inversion.mappings.mapping.Mapping`.  Drawn identically on the
        parametric and the inversion path, so a mapping keeps its colour across both.
    region_alpha : float, optional
        The alpha of each region's fill.
    region_labels : list of str, optional
        A label drawn at the centre of each region, matching its image-plane counterpart.
    """
    tracer = fit.tracer_linear_light_profiles_to_light_profiles
    if not tracer.planes[plane_index].has(cls=aa.Pixelization):
        if zoom_to_brightest:
            grid = fit.mask.derive_grid.all_false
        else:
            zoom = aa.Zoom2D(mask=fit.mask)
            grid = aa.Grid2D.from_extent(
                extent=zoom.extent_from(buffer=0),
                shape_native=zoom.shape_native,
            )
        zoom_extent_bounds = None
        if zoom_extent_scale != 1.0:
            zoom = aa.Zoom2D(mask=fit.mask)
            no_zoom_grid = aa.Grid2D.from_extent(
                extent=zoom.extent_from(buffer=0),
                shape_native=zoom.shape_native,
            )
            zoom_extent_bounds = no_zoom_grid.geometry.extent
        image = plane_image_from(
            galaxies=tracer.planes[plane_index],
            grid=grid,
            zoom_to_brightest=zoom_to_brightest,
            zoom_extent_scale=zoom_extent_scale,
            zoom_extent_bounds=zoom_extent_bounds,
        )
        plot_array(
            array=image, ax=ax,
            title=title if title is not None else f"Source Plane {plane_index}",
            colormap=colormap, use_log10=use_log10, vmax=vmax, lines=lines,
            line_colors=line_colors,
        )
        # `autogalaxy.util.plot_utils.plot_array` is a thin wrapper which does not forward
        # `regions=`; the overlay is drawn onto the same axes afterwards instead, using the
        # same `plot_regions` the pixelized path below reaches through `plot_mapper`. The
        # polygons are arcsec `(y, x)`, so they land correctly whatever zoom the wrapper
        # applied.
        plot_regions(ax=ax, regions=regions, region_alpha=region_alpha,
                     region_labels=region_labels)
    else:
        # Inversion path: plot the source reconstruction via the mapper.
        try:
            inversion = fit.inversion
            mapper_list = inversion.cls_list_from(cls=Mapper)
            mapper = mapper_list[plane_index - 1] if plane_index > 0 else mapper_list[0]
            pixel_values = inversion.reconstruction_dict[mapper]
            plot_mapper(
                mapper,
                solution_vector=pixel_values,
                ax=ax,
                title=title if title is not None else f"Source Reconstruction (plane {plane_index})",
                colormap=colormap,
                use_log10=use_log10,
                vmax=vmax,
                zoom_to_brightest=zoom_to_brightest,
                zoom_extent_scale=zoom_extent_scale,
                lines=lines,
                line_colors=line_colors,
                regions=regions,
                region_alpha=region_alpha,
                region_labels=region_labels,
            )
        except Exception as exc:
            logger.warning(f"Could not plot source reconstruction for plane {plane_index}: {exc}")
            if ax is not None:
                ax.axis("off")
                ax.set_title(f"Source Reconstruction (plane {plane_index})")


def subplot_fit(
    fit,
    output_path: Optional[str] = None,
    output_format: str = None,
    colormap: Optional[str] = None,
    plane_index: Optional[int] = None,
    image_plane_lines=None,
    image_plane_line_colors=None,
    source_plane_lines=None,
    source_plane_line_colors=None,
    title_prefix: str = None,
):
    """
    Produce a 12-panel subplot summarising an imaging fit.

    Arranges the following panels in a 3 × 4 grid:

    * Data
    * Model image
    * Signal-to-noise map
    * Source plane image (max zoom)
    * Lens-light model image
    * Lens-light-subtracted image (source scale)
    * Source model image (source scale)
    * Source plane image (mid zoom — 2× wider than max zoom, square, shrunk
      uniformly so all edges stay inside the no-zoom extent)
    * Normalised residual map (symmetric scale)
    * Normalised residual map clipped to ± 1 σ
    * Chi-squared map
    * Source plane image (full extent)

    For single-plane tracers the function delegates to
    :func:`subplot_fit_x1_plane`, which uses a simpler 2 × 3 layout.

    Parameters
    ----------
    fit : FitImaging
        The imaging fit to visualise.
    output_path : str, optional
        Directory in which to save the figure.  If ``None`` the figure is
        not saved to disk.
    output_format : str, optional
        Image format passed to :func:`~autoarray.plot.utils.save_figure`
        (e.g. ``"png"``, ``"pdf"``).
    colormap : str, optional
        Matplotlib colormap name applied to all image panels.
    plane_index : int, optional
        Index of the source plane to use for the source-scale panels.
        Defaults to the final plane in the tracer.
    """
    if len(fit.tracer.planes) == 1:
        return subplot_fit_x1_plane(fit, output_path=output_path,
                                    output_format=output_format, colormap=colormap,
                                    title_prefix=title_prefix)

    plane_index_tag = "" if plane_index is None else f"_{plane_index}"
    final_plane_index = (
        len(fit.tracer.planes) - 1 if plane_index is None else plane_index
    )

    source_vmax = _get_source_vmax(fit)

    if image_plane_lines is None and source_plane_lines is None:
        image_plane_lines, image_plane_line_colors, source_plane_lines, source_plane_line_colors = (
            _compute_critical_curves_from_fit(fit)
        )

    _pf = (lambda t: f"{title_prefix.rstrip()} {t}") if title_prefix else (lambda t: t)
    fig, axes = subplots(3, 4, figsize=conf_subplot_figsize(3, 4))
    axes_flat = list(axes.flatten())

    plot_array(array=fit.data, ax=axes_flat[0], title=_pf("Data"), colormap=colormap)

    plot_array(array=fit.model_data, ax=axes_flat[1], title=_pf("Model Image"),
               colormap=colormap, lines=image_plane_lines,
               line_colors=image_plane_line_colors)

    plot_array(array=fit.signal_to_noise_map, ax=axes_flat[2],
               title=_pf("Signal-To-Noise Map"), colormap=colormap)

    # Source plane (max zoom)
    _plot_source_plane(fit, axes_flat[3], final_plane_index, zoom_to_brightest=True,
                       colormap=colormap, title=_pf("Source Plane (Max Zoom)"),
                       lines=source_plane_lines, line_colors=source_plane_line_colors,
                       vmax=source_vmax)

    # Lens model image
    try:
        lens_model_img = fit.model_images_of_planes_list[0]
    except (IndexError, AttributeError):
        lens_model_img = None
    if lens_model_img is not None:
        plot_array(array=lens_model_img, ax=axes_flat[4],
                   title=_pf("Lens Light Model Image"), colormap=colormap)
    else:
        axes_flat[4].axis("off")

    # Subtracted image at source scale
    try:
        subtracted_img = fit.subtracted_images_of_planes_list[final_plane_index]
    except (IndexError, AttributeError):
        subtracted_img = None
    if subtracted_img is not None:
        plot_array(array=subtracted_img, ax=axes_flat[5], title=_pf("Lens Light Subtracted"),
                   colormap=colormap, vmin=0.0 if source_vmax is not None else None,
                   vmax=source_vmax)
    else:
        axes_flat[5].axis("off")

    # Source model image at source scale
    try:
        source_model_img = fit.model_images_of_planes_list[final_plane_index]
    except (IndexError, AttributeError):
        source_model_img = None
    if source_model_img is not None:
        plot_array(array=source_model_img, ax=axes_flat[6], title=_pf("Source Model Image"),
                   colormap=colormap, vmax=source_vmax, lines=image_plane_lines,
                   line_colors=image_plane_line_colors)
    else:
        axes_flat[6].axis("off")

    # Source plane (mid zoom) — same centre as Max Zoom, 2.5x wider extent
    _plot_source_plane(fit, axes_flat[7], final_plane_index, zoom_to_brightest=True,
                       colormap=colormap, title=_pf("Source Plane (Mid Zoom)"),
                       lines=source_plane_lines, line_colors=source_plane_line_colors,
                       vmax=source_vmax, zoom_extent_scale=2.0)

    # Normalized residual map (symmetric)
    norm_resid = fit.normalized_residual_map
    _abs_max = _symmetric_vmax(norm_resid)
    plot_array(array=norm_resid, ax=axes_flat[8], title=_pf("Normalized Residual Map"),
               colormap=colormap, vmin=-_abs_max, vmax=_abs_max)

    # Normalized residual map clipped to [-1, 1]
    plot_array(array=norm_resid, ax=axes_flat[9],
               title=_pf(r"Normalized Residual Map $1\sigma$"),
               colormap=colormap, vmin=-1.0, vmax=1.0)

    plot_array(array=fit.chi_squared_map, ax=axes_flat[10],
               title=_pf("Chi-Squared Map"), colormap=colormap, cb_unit=r"$\chi^2$")

    # Source plane not zoomed
    _plot_source_plane(fit, axes_flat[11], final_plane_index, zoom_to_brightest=False,
                       colormap=colormap, title=_pf("Source Plane (No Zoom)"),
                       lines=source_plane_lines, line_colors=source_plane_line_colors,
                       vmax=source_vmax)

    hide_unused_axes(axes_flat)
    tight_layout()
    save_figure(fig, path=output_path, filename=f"fit{plane_index_tag}", format=output_format)


def subplot_fit_x1_plane(
    fit,
    output_path: Optional[str] = None,
    output_format: str = None,
    colormap: Optional[str] = None,
    title_prefix: str = None,
):
    """
    Produce a 6-panel subplot for a single-plane tracer imaging fit.

    Arranges the following panels in a 2 × 3 grid:

    * Data
    * Signal-to-noise map
    * Model image
    * Normalised residual map (lens-light subtracted proxy)
    * Normalised residual map with zero minimum
    * Normalised residual map (symmetric scale)

    This simplified layout is used automatically by :func:`subplot_fit`
    when the tracer has only one plane (no source plane).

    Parameters
    ----------
    fit : FitImaging
        The single-plane imaging fit to visualise.
    output_path : str, optional
        Directory in which to save the figure.  If ``None`` the figure is
        not saved to disk.
    output_format : str, optional
        Image format passed to :func:`~autoarray.plot.utils.save_figure`.
    colormap : str, optional
        Matplotlib colormap name applied to all image panels.
    """
    _pf = (lambda t: f"{title_prefix.rstrip()} {t}") if title_prefix else (lambda t: t)
    fig, axes = subplots(2, 3, figsize=conf_subplot_figsize(2, 3))
    axes_flat = list(axes.flatten())

    try:
        vmax = float(np.max(fit.model_images_of_planes_list[0].array))
    except (IndexError, AttributeError, ValueError):
        vmax = None

    plot_array(array=fit.data, ax=axes_flat[0], title=_pf("Data"), colormap=colormap, vmax=vmax)

    plot_array(array=fit.signal_to_noise_map, ax=axes_flat[1],
               title=_pf("Signal-To-Noise Map"), colormap=colormap)

    plot_array(array=fit.model_data, ax=axes_flat[2], title=_pf("Model Image"),
               colormap=colormap, vmax=vmax)

    norm_resid = fit.normalized_residual_map
    plot_array(array=norm_resid, ax=axes_flat[3], title=_pf("Lens Light Subtracted"),
               colormap=colormap, cb_unit=r"$\sigma$")

    plot_array(array=norm_resid, ax=axes_flat[4], title=_pf("Subtracted Image Zero Minimum"),
               colormap=colormap, vmin=0.0, cb_unit=r"$\sigma$")

    _abs_max = _symmetric_vmax(norm_resid)
    plot_array(array=norm_resid, ax=axes_flat[5], title=_pf("Normalized Residual Map"),
               colormap=colormap, vmin=-_abs_max, vmax=_abs_max, cb_unit=r"$\sigma$")

    tight_layout()
    save_figure(fig, path=output_path, filename="fit_x1_plane", format=output_format)


def subplot_fit_log10(
    fit,
    output_path: Optional[str] = None,
    output_format: str = None,
    colormap: Optional[str] = None,
    plane_index: Optional[int] = None,
    image_plane_lines=None,
    image_plane_line_colors=None,
    source_plane_lines=None,
    source_plane_line_colors=None,
    title_prefix: str = None,
):
    """
    Produce a 12-panel subplot summarising an imaging fit with log10 colour scaling.

    Equivalent to :func:`subplot_fit` but applies a log10 stretch to all
    positive-valued panels (data, model image, lens-light model, subtracted
    image, source model image, chi-squared map, source plane images).
    Residual panels are left on a linear scale because they contain negative
    values.  Includes Source Plane (Max Zoom) and Source Plane (Mid Zoom)
    panels — Mid Zoom shares the Max Zoom centre with extents 2x larger,
    kept square and shrunk uniformly so all edges stay inside the No Zoom
    extent.

    For single-plane tracers the function delegates to
    :func:`subplot_fit_log10_x1_plane`.

    Parameters
    ----------
    fit : FitImaging
        The imaging fit to visualise.
    output_path : str, optional
        Directory in which to save the figure.  If ``None`` the figure is
        not saved to disk.
    output_format : str, optional
        Image format passed to :func:`~autoarray.plot.utils.save_figure`.
    colormap : str, optional
        Matplotlib colormap name applied to all image panels.
    plane_index : int, optional
        Index of the source plane to use for the source-scale panels.
        Defaults to the final plane in the tracer.
    """
    if len(fit.tracer.planes) == 1:
        return subplot_fit_log10_x1_plane(fit, output_path=output_path,
                                          output_format=output_format, colormap=colormap,
                                          title_prefix=title_prefix)

    plane_index_tag = "" if plane_index is None else f"_{plane_index}"
    final_plane_index = (
        len(fit.tracer.planes) - 1 if plane_index is None else plane_index
    )

    source_vmax = _get_source_vmax(fit)

    if image_plane_lines is None and source_plane_lines is None:
        image_plane_lines, image_plane_line_colors, source_plane_lines, source_plane_line_colors = (
            _compute_critical_curves_from_fit(fit)
        )

    _pf = (lambda t: f"{title_prefix.rstrip()} {t}") if title_prefix else (lambda t: t)
    fig, axes = subplots(3, 4, figsize=conf_subplot_figsize(3, 4))
    axes_flat = list(axes.flatten())

    plot_array(array=fit.data, ax=axes_flat[0], title=_pf("Data"), colormap=colormap,
               use_log10=True)

    plot_array(array=fit.model_data, ax=axes_flat[1], title=_pf("Model Image"),
               colormap=colormap, use_log10=True, lines=image_plane_lines,
               line_colors=image_plane_line_colors)

    try:
        plot_array(array=fit.signal_to_noise_map, ax=axes_flat[2],
                   title=_pf("Signal-To-Noise Map"), colormap=colormap, use_log10=True)
    except ValueError:
        axes_flat[2].axis("off")

    # Source plane (max zoom)
    _plot_source_plane(fit, axes_flat[3], final_plane_index, zoom_to_brightest=True,
                       colormap=colormap, use_log10=True,
                       title=_pf("Source Plane (Max Zoom)"),
                       lines=source_plane_lines, line_colors=source_plane_line_colors,
                       vmax=source_vmax)

    try:
        lens_model_img = fit.model_images_of_planes_list[0]
        plot_array(array=lens_model_img, ax=axes_flat[4],
                   title=_pf("Lens Light Model Image"), colormap=colormap, use_log10=True)
    except (IndexError, AttributeError):
        axes_flat[4].axis("off")

    try:
        subtracted_img = fit.subtracted_images_of_planes_list[final_plane_index]
        plot_array(array=subtracted_img, ax=axes_flat[5],
                   title=_pf("Lens Light Subtracted"), colormap=colormap, use_log10=True)
    except (IndexError, AttributeError):
        axes_flat[5].axis("off")

    try:
        source_model_img = fit.model_images_of_planes_list[final_plane_index]
        plot_array(array=source_model_img, ax=axes_flat[6],
                   title=_pf("Source Model Image"), colormap=colormap, use_log10=True,
                   lines=image_plane_lines, line_colors=image_plane_line_colors)
    except (IndexError, AttributeError):
        axes_flat[6].axis("off")

    # Source plane (mid zoom) — same centre as Max Zoom, 2.5x wider extent
    _plot_source_plane(fit, axes_flat[7], final_plane_index, zoom_to_brightest=True,
                       colormap=colormap, use_log10=True,
                       title=_pf("Source Plane (Mid Zoom)"),
                       lines=source_plane_lines, line_colors=source_plane_line_colors,
                       vmax=source_vmax, zoom_extent_scale=2.0)

    norm_resid = fit.normalized_residual_map
    _abs_max = _symmetric_vmax(norm_resid)
    plot_array(array=norm_resid, ax=axes_flat[8], title=_pf("Normalized Residual Map"),
               colormap=colormap, vmin=-_abs_max, vmax=_abs_max, cb_unit=r"$\sigma$")

    plot_array(array=norm_resid, ax=axes_flat[9],
               title=_pf(r"Normalized Residual Map $1\sigma$"),
               colormap=colormap, vmin=-1.0, vmax=1.0, cb_unit=r"$\sigma$")

    plot_array(array=fit.chi_squared_map, ax=axes_flat[10], title=_pf("Chi-Squared Map"),
               colormap=colormap, use_log10=True, cb_unit=r"$\chi^2$")

    _plot_source_plane(fit, axes_flat[11], final_plane_index, zoom_to_brightest=False,
                       colormap=colormap, use_log10=True,
                       title=_pf("Source Plane (No Zoom)"),
                       lines=source_plane_lines, line_colors=source_plane_line_colors,
                       vmax=source_vmax)

    tight_layout()
    save_figure(fig, path=output_path, filename=f"fit_log10{plane_index_tag}", format=output_format)


def subplot_fit_log10_x1_plane(
    fit,
    output_path: Optional[str] = None,
    output_format: str = None,
    colormap: Optional[str] = None,
    title_prefix: str = None,
):
    """
    Produce a 6-panel log10 subplot for a single-plane tracer imaging fit.

    Equivalent to :func:`subplot_fit_x1_plane` but applies a log10 colour
    stretch to the data, model image, and chi-squared panels.  Residual
    panels remain on a linear scale.

    This simplified layout is used automatically by
    :func:`subplot_fit_log10` when the tracer has only one plane.

    Parameters
    ----------
    fit : FitImaging
        The single-plane imaging fit to visualise.
    output_path : str, optional
        Directory in which to save the figure.  If ``None`` the figure is
        not saved to disk.
    output_format : str, optional
        Image format passed to :func:`~autoarray.plot.utils.save_figure`.
    colormap : str, optional
        Matplotlib colormap name applied to all image panels.
    """
    _pf = (lambda t: f"{title_prefix.rstrip()} {t}") if title_prefix else (lambda t: t)
    fig, axes = subplots(2, 3, figsize=conf_subplot_figsize(2, 3))
    axes_flat = list(axes.flatten())

    try:
        vmax = float(np.max(fit.model_images_of_planes_list[0].array))
    except (IndexError, AttributeError, ValueError):
        vmax = None

    plot_array(array=fit.data, ax=axes_flat[0], title=_pf("Data"), colormap=colormap,
               vmax=vmax, use_log10=True)

    try:
        plot_array(array=fit.signal_to_noise_map, ax=axes_flat[1],
                   title=_pf("Signal-To-Noise Map"), colormap=colormap, use_log10=True)
    except ValueError:
        axes_flat[1].axis("off")

    plot_array(array=fit.model_data, ax=axes_flat[2], title=_pf("Model Image"),
               colormap=colormap, vmax=vmax, use_log10=True)

    norm_resid = fit.normalized_residual_map
    plot_array(array=norm_resid, ax=axes_flat[3], title=_pf("Lens Light Subtracted"),
               colormap=colormap, cb_unit=r"$\sigma$")
    _abs_max = _symmetric_vmax(norm_resid)
    plot_array(array=norm_resid, ax=axes_flat[4], title=_pf("Normalized Residual Map"),
               colormap=colormap, vmin=-_abs_max, vmax=_abs_max, cb_unit=r"$\sigma$")
    plot_array(array=fit.chi_squared_map, ax=axes_flat[5], title=_pf("Chi-Squared Map"),
               colormap=colormap, use_log10=True, cb_unit=r"$\chi^2$")

    tight_layout()
    save_figure(fig, path=output_path, filename="fit_log10", format=output_format)


def subplot_of_planes(
    fit,
    output_path: Optional[str] = None,
    output_format: str = None,
    colormap: Optional[str] = None,
    plane_index: Optional[int] = None,
    title_prefix: str = None,
):
    """
    Produce a 4-panel subplot for each plane in the tracer.

    For every plane (or the single plane specified by ``plane_index``), a
    1 × 4 row is saved to its own figure containing:

    * Data
    * Lens-light-subtracted image for that plane
    * Model image contributed by that plane
    * Source-plane image evaluated at that plane

    Each figure is saved with the filename
    ``subplot_of_plane_<plane_index>``.

    Parameters
    ----------
    fit : FitImaging
        The imaging fit whose planes are visualised.
    output_path : str, optional
        Directory in which to save the figures.  If ``None`` the figures
        are not saved to disk.
    output_format : str, optional
        Image format passed to :func:`~autoarray.plot.utils.save_figure`.
    colormap : str, optional
        Matplotlib colormap name applied to all image panels.
    plane_index : int, optional
        If provided, only the subplot for that specific plane is produced.
        If ``None`` (default) a subplot is produced for every plane in the
        tracer.
    """
    if plane_index is None:
        plane_indexes = range(len(fit.tracer.planes))
    else:
        plane_indexes = [plane_index]

    _pf = (lambda t: f"{title_prefix.rstrip()} {t}") if title_prefix else (lambda t: t)
    for pidx in plane_indexes:
        fig, axes = subplots(1, 4, figsize=conf_subplot_figsize(1, 4))
        axes_flat = list(axes.flatten())

        plot_array(array=fit.data, ax=axes_flat[0], title=_pf("Data"), colormap=colormap)

        try:
            subtracted = fit.subtracted_images_of_planes_list[pidx]
            plot_array(array=subtracted, ax=axes_flat[1],
                       title=_pf(f"Subtracted Image Plane {pidx}"), colormap=colormap)
        except (IndexError, AttributeError):
            axes_flat[1].axis("off")

        try:
            model_img = fit.model_images_of_planes_list[pidx]
            plot_array(array=model_img, ax=axes_flat[2],
                       title=_pf(f"Model Image Plane {pidx}"), colormap=colormap)
        except (IndexError, AttributeError):
            axes_flat[2].axis("off")

        _plot_source_plane(fit, axes_flat[3], pidx, colormap=colormap,
                           title=_pf(f"Source Plane {pidx}"))

        tight_layout()
        save_figure(fig, path=output_path, filename=f"fit_of_plane_{pidx}", format=output_format)


def subplot_tracer_from_fit(
    fit,
    output_path: Optional[str] = None,
    output_format: str = None,
    colormap: Optional[str] = None,
    image_plane_lines=None,
    image_plane_line_colors=None,
    source_plane_lines=None,
    source_plane_line_colors=None,
    title_prefix: str = None,
):
    """
    Produce a 9-panel tracer subplot derived from a `FitImaging` object.

    Panels (3x3 = 9 axes):
      0: Model image with critical curves
      1: Source model image (image-plane projection) with critical curves
      2: Source plane (no zoom) with caustics
      3: Lens image (log10) with critical curves
      4: Convergence (log10)
      5: Potential (log10)
      6: Deflections Y with critical curves
      7: Deflections X with critical curves
      8: Magnification with critical curves

    Parameters
    ----------
    fit : FitImaging
        The imaging fit whose best-fit tracer is visualised.
    output_path : str, optional
        Directory in which to save the figure.  If ``None`` the figure is
        not saved to disk.
    output_format : str, optional
        Image format passed to :func:`~autoarray.plot.utils.save_figure`.
    colormap : str, optional
        Matplotlib colormap name applied to all image panels.
    """
    from autogalaxy.galaxy.plot.galaxies_plots import plot_magnification_2d

    final_plane_index = len(fit.tracer.planes) - 1
    tracer = fit.tracer_linear_light_profiles_to_light_profiles

    # --- grid ---
    grid = fit.mask.derive_grid.all_false

    if image_plane_lines is None and source_plane_lines is None:
        image_plane_lines, image_plane_line_colors, source_plane_lines, source_plane_line_colors = (
            _compute_critical_curves_from_fit(fit)
        )

    source_vmax = _get_source_vmax(fit)

    traced_grids = tracer.traced_grid_2d_list_from(grid=grid)
    lens_galaxies = ag.Galaxies(galaxies=tracer.planes[0])
    lens_image = lens_galaxies.image_2d_from(grid=traced_grids[0])

    deflections = tracer.deflections_yx_2d_from(grid=grid)
    deflections_y = aa.Array2D(values=deflections.slim[:, 0], mask=grid.mask)
    deflections_x = aa.Array2D(values=deflections.slim[:, 1], mask=grid.mask)

    _pf = (lambda t: f"{title_prefix.rstrip()} {t}") if title_prefix else (lambda t: t)
    fig, axes = subplots(3, 3, figsize=conf_subplot_figsize(3, 3))
    axes_flat = list(axes.flatten())

    # Panel 0: Model Image
    plot_array(array=fit.model_data, ax=axes_flat[0], title=_pf("Model Image"),
               lines=image_plane_lines, line_colors=image_plane_line_colors,
               colormap=colormap)

    # Panel 1: Source Model Image (same as subplot_fit panel 7)
    try:
        source_model_img = fit.model_images_of_planes_list[final_plane_index]
    except (IndexError, AttributeError):
        source_model_img = None
    if source_model_img is not None:
        plot_array(array=source_model_img, ax=axes_flat[1], title=_pf("Source Model Image"),
                   colormap=colormap, vmax=source_vmax,
                   lines=image_plane_lines, line_colors=image_plane_line_colors)
    else:
        axes_flat[1].axis("off")

    # Panel 2: Source Plane (No Zoom) (same as subplot_fit panel 12)
    _plot_source_plane(fit, axes_flat[2], final_plane_index, zoom_to_brightest=False,
                       colormap=colormap, title=_pf("Source Plane (No Zoom)"),
                       lines=source_plane_lines, line_colors=source_plane_line_colors,
                       vmax=source_vmax)

    # Panel 3: Lens Image (log10)
    plot_array(array=lens_image, ax=axes_flat[3], title=_pf("Lens Image"),
               lines=image_plane_lines, line_colors=image_plane_line_colors,
               colormap=colormap, use_log10=True)

    # Panel 4: Convergence (log10)
    try:
        convergence = tracer.convergence_2d_from(grid=grid)
        plot_array(array=convergence, ax=axes_flat[4], title=_pf("Convergence"),
                   colormap=colormap, use_log10=True)
    except Exception:
        axes_flat[4].axis("off")

    # Panel 5: Potential (log10)
    try:
        potential = tracer.potential_2d_from(grid=grid)
        plot_array(array=potential, ax=axes_flat[5], title=_pf("Potential"),
                   colormap=colormap, use_log10=True)
    except Exception:
        axes_flat[5].axis("off")

    # Panel 6: Deflections Y
    plot_array(array=deflections_y, ax=axes_flat[6], title=_pf("Deflections Y"),
               colormap=colormap)

    # Panel 7: Deflections X
    plot_array(array=deflections_x, ax=axes_flat[7], title=_pf("Deflections X"),
               colormap=colormap)

    # Panel 8: Magnification
    plot_magnification_2d(mass_obj=tracer, grid=grid, ax=axes_flat[8], title=_pf("Magnification"),
                          include_critical_curves=False, colormap=colormap)

    tight_layout()
    save_figure(fig, path=output_path, filename="tracer", format=output_format)


def subplot_fit_combined(
    fit_list: List,
    output_path: Optional[str] = None,
    output_format: str = None,
    colormap: Optional[str] = None,
    title_prefix: str = None,
):
    """
    Produce a combined multi-row subplot for a list of `FitImaging` objects.

    Each row corresponds to one fit and contains six panels:

    * Data
    * Lens-light-subtracted image (plane 1)
    * Lens model image (plane 0)
    * Source model image (final plane)
    * Source plane image (final plane)
    * Normalised residual map

    This layout is useful for visually comparing fits from multiple
    datasets or epochs side by side.

    Parameters
    ----------
    fit_list : list of FitImaging
        The imaging fits to display.  Each fit occupies one row of the
        figure.
    output_path : str, optional
        Directory in which to save the figure.  If ``None`` the figure is
        not saved to disk.
    output_format : str, optional
        Image format passed to :func:`~autoarray.plot.utils.save_figure`.
    colormap : str, optional
        Matplotlib colormap name applied to all image panels.
    """
    n_fits = len(fit_list)
    n_cols = 6
    fig, axes = subplots(n_fits, n_cols, figsize=conf_subplot_figsize(n_fits, n_cols))
    if n_fits == 1:
        all_axes = [list(axes)]
    else:
        all_axes = [list(axes[i]) for i in range(n_fits)]

    final_plane_index = len(fit_list[0].tracer.planes) - 1

    _pf = (lambda t: f"{title_prefix.rstrip()} {t}") if title_prefix else (lambda t: t)
    for row, fit in enumerate(fit_list):
        row_axes = all_axes[row]

        plot_array(array=fit.data, ax=row_axes[0], title=_pf("Data"), colormap=colormap)

        try:
            subtracted = fit.subtracted_images_of_planes_list[1]
            plot_array(array=subtracted, ax=row_axes[1], title=_pf("Subtracted Image"),
                       colormap=colormap)
        except (IndexError, AttributeError):
            row_axes[1].axis("off")

        try:
            lens_model = fit.model_images_of_planes_list[0]
            plot_array(array=lens_model, ax=row_axes[2], title=_pf("Lens Model Image"),
                       colormap=colormap)
        except (IndexError, AttributeError):
            row_axes[2].axis("off")

        try:
            source_model = fit.model_images_of_planes_list[final_plane_index]
            plot_array(array=source_model, ax=row_axes[3], title=_pf("Source Model Image"),
                       colormap=colormap)
        except (IndexError, AttributeError):
            row_axes[3].axis("off")

        try:
            _plot_source_plane(fit, row_axes[4], final_plane_index, colormap=colormap,
                               title=_pf(f"Source Plane {final_plane_index}"))
        except Exception:
            row_axes[4].axis("off")

        plot_array(array=fit.normalized_residual_map, ax=row_axes[5],
                   title=_pf("Normalized Residual Map"), colormap=colormap, cb_unit=r"$\sigma$")

    tight_layout()
    save_figure(fig, path=output_path, filename="fit_combined", format=output_format)


def subplot_fit_combined_log10(
    fit_list: List,
    output_path: Optional[str] = None,
    output_format: str = None,
    colormap: Optional[str] = None,
    title_prefix: str = None,
):
    """
    Produce a combined log10 multi-row subplot for a list of `FitImaging` objects.

    Equivalent to :func:`subplot_fit_combined` but applies a log10 colour
    stretch to the data, lens model, and source model panels.  The
    normalised residual panel remains on a linear scale.

    Parameters
    ----------
    fit_list : list of FitImaging
        The imaging fits to display.  Each fit occupies one row of the
        figure.
    output_path : str, optional
        Directory in which to save the figure.  If ``None`` the figure is
        not saved to disk.
    output_format : str, optional
        Image format passed to :func:`~autoarray.plot.utils.save_figure`.
    colormap : str, optional
        Matplotlib colormap name applied to all image panels.
    """
    n_fits = len(fit_list)
    n_cols = 6
    fig, axes = subplots(n_fits, n_cols, figsize=conf_subplot_figsize(n_fits, n_cols))
    if n_fits == 1:
        all_axes = [list(axes)]
    else:
        all_axes = [list(axes[i]) for i in range(n_fits)]

    final_plane_index = len(fit_list[0].tracer.planes) - 1

    _pf = (lambda t: f"{title_prefix.rstrip()} {t}") if title_prefix else (lambda t: t)
    for row, fit in enumerate(fit_list):
        row_axes = all_axes[row]

        plot_array(array=fit.data, ax=row_axes[0], title=_pf("Data"), colormap=colormap,
                   use_log10=True)

        try:
            subtracted = fit.subtracted_images_of_planes_list[1]
            plot_array(array=subtracted, ax=row_axes[1], title=_pf("Subtracted Image"),
                       colormap=colormap, use_log10=True)
        except (IndexError, AttributeError):
            row_axes[1].axis("off")

        try:
            lens_model = fit.model_images_of_planes_list[0]
            plot_array(array=lens_model, ax=row_axes[2], title=_pf("Lens Model Image"),
                       colormap=colormap, use_log10=True)
        except (IndexError, AttributeError):
            row_axes[2].axis("off")

        try:
            source_model = fit.model_images_of_planes_list[final_plane_index]
            plot_array(array=source_model, ax=row_axes[3], title=_pf("Source Model Image"),
                       colormap=colormap, use_log10=True)
        except (IndexError, AttributeError):
            row_axes[3].axis("off")

        try:
            _plot_source_plane(fit, row_axes[4], final_plane_index, colormap=colormap,
                               use_log10=True,
                               title=_pf(f"Source Plane {final_plane_index}"))
        except Exception:
            row_axes[4].axis("off")

        plot_array(array=fit.normalized_residual_map, ax=row_axes[5],
                   title=_pf("Normalized Residual Map"), colormap=colormap, cb_unit=r"$\sigma$")

    tight_layout()
    save_figure(fig, path=output_path, filename="fit_combined_log10", format=output_format)


def _symmetric_vmax(array) -> float:
    """
    Return the absolute-maximum finite value for symmetric colormap scaling.

    Zooms into the unmasked region of ``array``, extracts all finite pixel
    values, and returns their absolute maximum.  Used to set ``vmin`` and
    ``vmax`` symmetrically around zero for residual-map panels so that the
    zero-residual colour is centred in the colormap.

    Parameters
    ----------
    array : Array2D or array-like
        The array from which the symmetric colour limit is computed.

    Returns
    -------
    float
        The absolute maximum of all finite pixel values in the (zoomed)
        array.  Returns ``1.0`` if the array contains no finite values.
    """
    try:
        vals = _zoom_array_2d(array).native.array
    except AttributeError:
        vals = np.asarray(array)
    finite = vals[np.isfinite(vals)]
    return float(np.max(np.abs(finite))) if finite.size else 1.0


def subplot_mappings(
    fit,
    plane_index: int = -1,
    shape=None,
    output_path: Optional[str] = None,
    output_filename: str = "mappings",
    output_format: str = None,
    colormap: Optional[str] = None,
    threshold: Optional[float] = None,
    min_pixels: Optional[int] = None,
    total_clumps: Optional[int] = None,
    pix_indexes: Optional[List] = None,
    weight_threshold: float = 0.0,
    pixel_scale_precision: Optional[float] = None,
    magnification_threshold: float = 0.1,
    region_alpha: float = 0.25,
    image_plane_lines=None,
    image_plane_line_colors=None,
    source_plane_lines=None,
    source_plane_line_colors=None,
    title_prefix: str = None,
):
    """
    2x2 subplot showing, in one look, how the brightest regions of the source map to the
    image plane and back.

    The panels are:

    * (0, 0) the lens-light-subtracted data, with the image-plane regions and the critical
      curves overlaid;
    * (0, 1) the model image of the source plane, with the same regions;
    * (1, 0) the source plane zoomed to its brightest region, with the source regions and
      the caustics;
    * (1, 1) the source plane at full extent, with the same.

    Every source region is filled in its own colour and numbered, and the multiple images
    it maps to carry the same colour and the same number in the image-plane panels -- so a
    reader can read a merging pair of source galaxies off the figure by colour alone.

    This is the fit-level counterpart of
    :func:`~autoarray.inversion.plot.inversion_plots.subplot_mappings`, which works only
    for a pixelized source. It works for **either** kind of source: a plane which
    ``has(cls=aa.Pixelization)`` is mapped via the inversion's clumps, and a parametric
    plane via `ShapeSolver`'s ray-traced triangles.  See
    :mod:`autolens.lens.mappings` for the two engines.

    Parameters
    ----------
    fit : FitImaging
        The imaging fit to visualise.
    plane_index : int, optional
        The index of the source plane whose mappings are drawn.  Defaults to the last plane.
    shape : Shape, optional
        The source-plane region for a parametric source.  ``None`` uses a `Circle` at the
        first source light profile's centre, with its half-light radius.  Ignored for a
        pixelized source, which finds its own clumps.
    output_path : str, optional
        Directory in which to save the figure.  ``None`` does not save it.
    output_filename : str, optional
        The filename stem; ``_{plane_index}`` is appended, as the sibling subplots do.
    output_format : str, optional
        Image format passed to :func:`~autoarray.plot.utils.save_figure`.
    colormap : str, optional
        Matplotlib colormap name applied to all image panels.
    threshold, min_pixels, total_clumps, pix_indexes, weight_threshold
        The clump-finding controls of the pixelized (engine A) path, passed through to
        `Inversion.mappings_from`.  ``threshold`` is the knob which decides what counts as
        a distinct source structure: ~0.5 isolates one smooth source, ~0.2 merges two
        nearby galaxies, ~0.8 splits a source into its star-forming knots.
    pixel_scale_precision, magnification_threshold
        The solver controls of the parametric (engine B) path.
    region_alpha : float, optional
        The alpha of each drawn region's fill.
    image_plane_lines, image_plane_line_colors, source_plane_lines, source_plane_line_colors
        Pre-computed critical curves and caustics; computed from the fit when not given.
    title_prefix : str, optional
        A string prefixed to every panel title.
    """
    from autolens.lens import mappings as mappings_module

    plane_index_tag = len(fit.tracer.planes) - 1 if plane_index == -1 else plane_index

    clump_kwargs = {}

    if threshold is not None:
        clump_kwargs["threshold"] = threshold
    if min_pixels is not None:
        clump_kwargs["min_pixels"] = min_pixels
    if total_clumps is not None:
        clump_kwargs["total_clumps"] = total_clumps
    if pix_indexes is not None:
        clump_kwargs["pix_indexes"] = pix_indexes

    # No guard around this call. A source plane which cannot be mapped -- a plane with
    # neither a pixelization nor a light profile, a profile with no half-light radius, a
    # singular inversion -- raises, naming what the caller has to change. Swallowing that
    # would draw the four panels with no regions on them, which reads as "this source has
    # no multiple images" and is a worse outcome than a traceback. The one case which
    # legitimately produces nothing to draw is an empty `mappings` list (below), which is
    # a real answer rather than a failure.
    mappings = mappings_module.mappings_from_fit(
        fit=fit,
        plane_index=plane_index,
        shape=shape,
        pixel_scale_precision=pixel_scale_precision,
        magnification_threshold=magnification_threshold,
        weight_threshold=weight_threshold,
        **clump_kwargs,
    )

    image_regions = [mapping.image_contours for mapping in mappings]
    source_regions = [mapping.source_contours for mapping in mappings]
    region_labels = [str(i + 1) for i in range(len(mappings))]

    if image_plane_lines is None and source_plane_lines is None:
        (
            image_plane_lines,
            image_plane_line_colors,
            source_plane_lines,
            source_plane_line_colors,
        ) = _compute_critical_curves_from_fit(fit)

    source_vmax = _get_source_vmax(fit)

    _pf = (lambda t: f"{title_prefix.rstrip()} {t}") if title_prefix else (lambda t: t)

    fig, axes = subplots(2, 2, figsize=conf_subplot_figsize(2, 2))
    axes_flat = list(axes.flatten())

    # panel (0, 0): the data with everything but this plane subtracted.
    try:
        subtracted_image = fit.subtracted_images_of_planes_list[plane_index]
    except (IndexError, AttributeError):
        subtracted_image = None

    if subtracted_image is not None:
        plot_array(
            array=subtracted_image,
            ax=axes_flat[0],
            title=_pf("Lens Light Subtracted"),
            colormap=colormap,
            vmin=0.0 if source_vmax is not None else None,
            vmax=source_vmax,
            lines=image_plane_lines,
            line_colors=image_plane_line_colors,
        )
        plot_regions(
            ax=axes_flat[0],
            regions=image_regions,
            region_alpha=region_alpha,
            region_labels=region_labels,
        )
    else:
        axes_flat[0].axis("off")

    # panel (0, 1): the model image of the plane, with the same regions.
    try:
        model_image = fit.model_images_of_planes_list[plane_index]
    except (IndexError, AttributeError):
        model_image = None

    if model_image is not None:
        plot_array(
            array=model_image,
            ax=axes_flat[1],
            title=_pf("Source Model Image"),
            colormap=colormap,
            vmax=source_vmax,
            lines=image_plane_lines,
            line_colors=image_plane_line_colors,
        )
        plot_regions(
            ax=axes_flat[1],
            regions=image_regions,
            region_alpha=region_alpha,
            region_labels=region_labels,
        )
    else:
        axes_flat[1].axis("off")

    # panels (1, 0) and (1, 1): the source plane, zoomed and unzoomed.
    for ax, zoom, title in (
        (axes_flat[2], True, "Source Plane (Zoom)"),
        (axes_flat[3], False, "Source Plane (No Zoom)"),
    ):
        _plot_source_plane(
            fit,
            ax,
            plane_index,
            zoom_to_brightest=zoom,
            colormap=colormap,
            title=_pf(title),
            lines=source_plane_lines,
            line_colors=source_plane_line_colors,
            vmax=source_vmax,
            regions=source_regions,
            region_alpha=region_alpha,
            region_labels=region_labels,
        )

    hide_unused_axes(axes_flat)
    tight_layout()
    save_figure(
        fig,
        path=output_path,
        filename=f"{output_filename}_{plane_index_tag}",
        format=output_format,
    )
