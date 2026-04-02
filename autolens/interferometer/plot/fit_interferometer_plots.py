import logging
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

import autoarray as aa
import autogalaxy as ag

from autogalaxy.plot.plot_utils import plot_array
from autoarray.plot.yx import plot_yx
from autoarray.plot.utils import save_figure, conf_subplot_figsize
from autoarray.plot.utils import numpy_lines as _to_lines
from autoarray.inversion.mappers.abstract import Mapper
from autoarray.inversion.plot.mapper_plots import plot_mapper
from autogalaxy.plot.plot_utils import _critical_curves_from, _caustics_from
from autolens.lens.plot.tracer_plots import plane_image_from

logger = logging.getLogger(__name__)


def _compute_critical_curve_lines(tracer, grid):
    """Compute critical-curve and caustic lines for a tracer on a given grid.

    Returns a 4-tuple ``(image_plane_lines, image_plane_line_colors,
    source_plane_lines, source_plane_line_colors)`` suitable for passing
    directly to :func:`~autoarray.plot.array.plot_array`.  On failure
    returns ``(None, None, None, None)``.
    """
    try:
        tan_cc, rad_cc = _critical_curves_from(tracer, grid)
        tan_ca, rad_ca = _caustics_from(tracer, grid)
        _tan_cc_lines = _to_lines(list(tan_cc) if tan_cc is not None else []) or []
        _rad_cc_lines = _to_lines(list(rad_cc) if rad_cc is not None else []) or []
        _tan_ca_lines = _to_lines(list(tan_ca) if tan_ca is not None else []) or []
        _rad_ca_lines = _to_lines(list(rad_ca) if rad_ca is not None else []) or []
        image_plane_lines = (_tan_cc_lines + _rad_cc_lines) or None
        image_plane_line_colors = (
            ["white"] * len(_tan_cc_lines) + ["yellow"] * len(_rad_cc_lines)
        )
        source_plane_lines = (_tan_ca_lines + _rad_ca_lines) or None
        source_plane_line_colors = (
            ["white"] * len(_tan_ca_lines) + ["yellow"] * len(_rad_ca_lines)
        )
        return image_plane_lines, image_plane_line_colors, source_plane_lines, source_plane_line_colors
    except Exception:
        return None, None, None, None


def _plot_source_plane(fit, ax, plane_index, zoom_to_brightest=True,
                       colormap=None, use_log10=False, title=None,
                       lines=None, line_colors=None, vmax=None):
    """
    Plot the source-plane image into an axes, matching the imaging subplot_fit behaviour.

    For parametric sources, evaluates the source light profiles directly on
    the unmasked real-space grid (``fit.dataset.real_space_mask.derive_grid.all_false``)
    via :func:`~autolens.lens.plot.tracer_plots.plane_image_from` — identical
    to the imaging path.  For pixelized sources, renders the inversion
    reconstruction via :func:`~autoarray.inversion.plot.mapper_plots.plot_mapper`.

    Parameters
    ----------
    fit : FitInterferometer
        The interferometer fit providing the tracer, real-space mask, and inversion.
    ax : matplotlib.axes.Axes or None
        The axes into which the source-plane image is drawn.  ``None`` is a no-op.
    plane_index : int
        Index of the plane in ``fit.tracer.planes`` to visualise.
    zoom_to_brightest : bool, optional
        For parametric sources: zoom the evaluation grid in on the brightest
        region.  For inversion sources: zoom the colourmap to brightest pixels.
    colormap : str, optional
        Matplotlib colormap name.
    use_log10 : bool, optional
        Apply a log10 colour stretch.
    title : str, optional
        Axes title.  Defaults to ``"Source Plane (Zoomed)"`` /
        ``"Source Plane (No Zoom)"`` according to ``zoom_to_brightest``.
    lines : list, optional
        Caustic lines to overlay (passed to :func:`plot_array` / :func:`plot_mapper`).
    line_colors : list, optional
        Colours for each entry in *lines*.
    vmax : float, optional
        Shared colour-scale maximum.
    """
    if ax is None:
        return

    if title is None:
        title = "Source Plane (Zoomed)" if zoom_to_brightest else "Source Plane (No Zoom)"

    tracer = fit.tracer_linear_light_profiles_to_light_profiles
    if not tracer.planes[plane_index].has(cls=aa.Pixelization):
        if zoom_to_brightest:
            grid = fit.dataset.real_space_mask.derive_grid.all_false
        else:
            zoom = aa.Zoom2D(mask=fit.dataset.real_space_mask)
            grid = aa.Grid2D.from_extent(
                extent=zoom.extent_from(buffer=0),
                shape_native=zoom.shape_native,
            )
        image = plane_image_from(
            galaxies=tracer.planes[plane_index],
            grid=grid,
            zoom_to_brightest=zoom_to_brightest,
        )
        plot_array(
            array=image, ax=ax,
            title=title,
            colormap=colormap, use_log10=use_log10, vmax=vmax,
            lines=lines, line_colors=line_colors,
        )
    else:
        try:
            inversion = fit.inversion
            mapper_list = inversion.cls_list_from(cls=Mapper)
            mapper = mapper_list[plane_index - 1] if plane_index > 0 else mapper_list[0]
            pixel_values = inversion.reconstruction_dict[mapper]
            plot_mapper(
                mapper,
                solution_vector=pixel_values,
                ax=ax,
                title=title,
                colormap=colormap,
                use_log10=use_log10,
                vmax=vmax,
                zoom_to_brightest=zoom_to_brightest,
                lines=lines,
                line_colors=line_colors,
            )
        except Exception as exc:
            logger.warning(f"Could not plot source reconstruction for plane {plane_index}: {exc}")
            ax.axis("off")
            ax.set_title(title)


def subplot_fit(
    fit,
    output_path: Optional[str] = None,
    output_format: str = "png",
    colormap: Optional[str] = None,
    image_plane_lines=None,
    image_plane_line_colors=None,
    source_plane_lines=None,
    source_plane_line_colors=None,
):
    """
    Produce a 12-panel subplot summarising an interferometer fit.

    Arranges the following panels in a 3 × 4 grid:

    * Amplitudes vs UV-distance (scatter)
    * Dirty image
    * Dirty signal-to-noise map
    * Dirty model image
    * Source plane image (final plane)
    * Normalised residual (real part) vs UV-distance (scatter)
    * Normalised residual (imaginary part) vs UV-distance (scatter)
    * Source plane image zoomed (final plane)
    * Dirty normalised residual map
    * Dirty normalised residual map clipped to ± 1 σ
    * (panel 9 re-used for 1σ clip — see implementation note)
    * Dirty chi-squared map
    * Source plane image (full extent)

    Parameters
    ----------
    fit : FitInterferometer
        The interferometer fit to visualise.
    output_path : str, optional
        Directory in which to save the figure.  If ``None`` the figure is
        not saved to disk.
    output_format : str, optional
        Image format passed to :func:`~autoarray.plot.utils.save_figure`.
    colormap : str, optional
        Matplotlib colormap name applied to all image panels.
    """
    final_plane_index = len(fit.tracer.planes) - 1

    if image_plane_lines is None and source_plane_lines is None:
        tracer = fit.tracer_linear_light_profiles_to_light_profiles
        _cc_grid = fit.dataset.real_space_mask.derive_grid.all_false
        image_plane_lines, image_plane_line_colors, source_plane_lines, source_plane_line_colors = (
            _compute_critical_curve_lines(tracer, _cc_grid)
        )

    fig, axes = plt.subplots(3, 4, figsize=conf_subplot_figsize(3, 4))
    axes_flat = list(axes.flatten())

    # Panel 0: amplitudes vs UV-distances
    plot_yx(
        y=np.real(fit.residual_map),
        x=fit.dataset.uv_distances / 10 ** 3.0,
        ax=axes_flat[0],
        title="Amplitudes vs UV-Distance",
        xtick_suffix='"',
        ytick_suffix="Jy",
        plot_axis_type="scatter",
    )

    plot_array(array=fit.dirty_image, ax=axes_flat[1], title="Dirty Image",
               colormap=colormap)
    plot_array(array=fit.dirty_signal_to_noise_map, ax=axes_flat[2],
               title="Dirty Signal-To-Noise Map", colormap=colormap)

    # Panel 3 (4th): dirty model image with critical curves
    plot_array(array=fit.dirty_model_image, ax=axes_flat[3], title="Dirty Model Image",
               colormap=colormap, lines=image_plane_lines,
               line_colors=image_plane_line_colors)

    # Panel 4: dirty residual map
    plot_array(array=fit.dirty_residual_map, ax=axes_flat[4],
               title="Dirty Residual Map", colormap=colormap)

    # Panel 5: normalized residual vs UV-distances (real)
    plot_yx(
        y=np.real(fit.normalized_residual_map),
        x=fit.dataset.uv_distances / 10 ** 3.0,
        ax=axes_flat[5],
        title="Normalized Residual Map (Real)",
        xtick_suffix='"',
        ytick_suffix=r"$\sigma$",
        plot_axis_type="scatter",
    )

    # Panel 6: normalized residual vs UV-distances (imag)
    plot_yx(
        y=np.imag(fit.normalized_residual_map),
        x=fit.dataset.uv_distances / 10 ** 3.0,
        ax=axes_flat[6],
        title="Normalized Residual Map (Imag)",
        xtick_suffix='"',
        ytick_suffix=r"$\sigma$",
        plot_axis_type="scatter",
    )

    # Panel 7 (8th): source plane zoomed with caustics
    _plot_source_plane(fit, axes_flat[7], final_plane_index,
                       zoom_to_brightest=True, colormap=colormap,
                       title="Source Plane (Zoomed)",
                       lines=source_plane_lines,
                       line_colors=source_plane_line_colors)

    plot_array(array=fit.dirty_normalized_residual_map, ax=axes_flat[8],
               title="Dirty Normalized Residual Map", colormap=colormap, cb_unit=r"$\sigma$")

    # Panel 9: clipped to ±1σ
    plot_array(
        fit.dirty_normalized_residual_map,
        ax=axes_flat[9],
        title=r"Normalized Residual Map $1\sigma$",
        colormap=colormap,
        vmin=-1.0, vmax=1.0,
        cb_unit=r"$\sigma$",
    )

    plot_array(array=fit.dirty_chi_squared_map, ax=axes_flat[10],
               title="Dirty Chi-Squared Map", colormap=colormap, cb_unit=r"$\chi^2$")

    # Panel 11 (12th): source plane not zoomed with caustics
    _plot_source_plane(fit, axes_flat[11], final_plane_index,
                       zoom_to_brightest=False, colormap=colormap,
                       title="Source Plane (No Zoom)",
                       lines=source_plane_lines,
                       line_colors=source_plane_line_colors)

    plt.tight_layout()
    save_figure(fig, path=output_path, filename="fit", format=output_format)


def subplot_fit_dirty_images(
    fit,
    output_path: Optional[str] = None,
    output_format: str = "png",
    colormap: Optional[str] = None,
    use_log10: bool = False,
    image_plane_lines=None,
    image_plane_line_colors=None,
):
    """
    Produce a 2×3 subplot of dirty-image diagnostics for an interferometer fit.

    Panels (row-major order):
      Dirty Image | Dirty Signal-To-Noise Map | Dirty Model Image (critical curves)
      Dirty Residual Map | Dirty Norm Residual Map | Dirty Chi-Squared Map

    Parameters
    ----------
    fit : FitInterferometer
        The interferometer fit to visualise.
    output_path : str, optional
        Directory in which to save the figure.
    output_format : str, optional
        Image format.
    colormap : str, optional
        Matplotlib colormap name.
    use_log10 : bool, optional
        Apply a log10 colour stretch.
    """
    if image_plane_lines is None:
        tracer = fit.tracer_linear_light_profiles_to_light_profiles
        _cc_grid = fit.dataset.real_space_mask.derive_grid.all_false
        image_plane_lines, image_plane_line_colors, _, _ = (
            _compute_critical_curve_lines(tracer, _cc_grid)
        )

    fig, axes = plt.subplots(2, 3, figsize=conf_subplot_figsize(2, 3))
    axes_flat = list(axes.flatten())

    plot_array(array=fit.dirty_image, ax=axes_flat[0], title="Dirty Image",
               colormap=colormap, use_log10=use_log10)
    plot_array(array=fit.dirty_signal_to_noise_map, ax=axes_flat[1],
               title="Dirty Signal-To-Noise Map", colormap=colormap)
    plot_array(array=fit.dirty_model_image, ax=axes_flat[2],
               title="Dirty Model Image", colormap=colormap, use_log10=use_log10,
               lines=image_plane_lines, line_colors=image_plane_line_colors)
    plot_array(array=fit.dirty_residual_map, ax=axes_flat[3],
               title="Dirty Residual Map", colormap=colormap)
    plot_array(array=fit.dirty_normalized_residual_map, ax=axes_flat[4],
               title="Dirty Normalized Residual Map", colormap=colormap, cb_unit=r"$\sigma$")
    plot_array(array=fit.dirty_chi_squared_map, ax=axes_flat[5],
               title="Dirty Chi-Squared Map", colormap=colormap, cb_unit=r"$\chi^2$")

    plt.tight_layout()
    save_figure(fig, path=output_path, filename="fit_dirty_images", format=output_format)


def subplot_fit_real_space(
    fit,
    output_path: Optional[str] = None,
    output_format: str = "png",
    colormap: Optional[str] = None,
    source_plane_lines=None,
    source_plane_line_colors=None,
):
    """
    Produce a real-space subplot for an interferometer fit.

    Renders the model in image space rather than the visibility (UV)
    domain.  The layout depends on whether the fit uses an inversion:

    * **No inversion** — two panels: the full lensed model image and the
      source-plane image of the final plane evaluated on the zoomed
      real-space grid.
    * **With inversion** — two placeholder panels are shown (axes turned
      off), because the inversion reconstruction is rendered by the
      inversion plotter.

    Parameters
    ----------
    fit : FitInterferometer
        The interferometer fit to visualise.
    output_path : str, optional
        Directory in which to save the figure.  If ``None`` the figure is
        not saved to disk.
    output_format : str, optional
        Image format passed to :func:`~autoarray.plot.utils.save_figure`.
    colormap : str, optional
        Matplotlib colormap name applied to all image panels.
    """
    tracer = fit.tracer_linear_light_profiles_to_light_profiles
    final_plane_index = len(fit.tracer.planes) - 1

    fig, axes = plt.subplots(1, 2, figsize=conf_subplot_figsize(1, 2))
    axes_flat = list(axes.flatten())

    if fit.inversion is None:
        # Parametric source: image-plane model image + source-plane image
        grid = fit.dataset.real_space_mask.derive_grid.all_false
        image = tracer.image_2d_from(grid=grid)
        plot_array(array=image, ax=axes_flat[0], title="Image", colormap=colormap)

        _plot_source_plane(fit, axes_flat[1], final_plane_index,
                           zoom_to_brightest=True, colormap=colormap,
                           title="Source Plane (Zoomed)",
                           lines=source_plane_lines, line_colors=source_plane_line_colors)
    else:
        # Pixelized source: dirty model image + source reconstruction
        plot_array(array=fit.dirty_model_image, ax=axes_flat[0],
                   title="Reconstructed Image", colormap=colormap)
        _plot_source_plane(fit, axes_flat[1], final_plane_index,
                           zoom_to_brightest=True, colormap=colormap,
                           title="Source Reconstruction",
                           lines=source_plane_lines, line_colors=source_plane_line_colors)

    plt.tight_layout()
    save_figure(fig, path=output_path, filename="fit_real_space", format=output_format)


def subplot_tracer_from_fit(
    fit,
    output_path: Optional[str] = None,
    output_format: str = "png",
    colormap: Optional[str] = None,
    image_plane_lines=None,
    image_plane_line_colors=None,
    source_plane_lines=None,
    source_plane_line_colors=None,
):
    """
    Produce a 9-panel tracer subplot derived from a `FitInterferometer` object.

    Panels (3x3 = 9 axes):
      0: Dirty Model Image with critical curves
      1: Source Model Image (dirty, image-plane projection) with critical curves
      2: Source plane (no zoom) with caustics
      3: Lens image (log10) with critical curves
      4: Convergence (log10)
      5: Potential (log10)
      6: Deflections Y with critical curves
      7: Deflections X with critical curves
      8: Magnification with critical curves

    Parameters
    ----------
    fit : FitInterferometer
        The interferometer fit whose best-fit tracer is visualised.
    output_path : str, optional
        Directory in which to save the figure.  If ``None`` the figure is
        not saved to disk.
    output_format : str, optional
        Image format passed to :func:`~autoarray.plot.utils.save_figure`.
    colormap : str, optional
        Matplotlib colormap name applied to all image panels.
    """
    from autogalaxy.operate.lens_calc import LensCalc

    final_plane_index = len(fit.tracer.planes) - 1
    tracer = fit.tracer_linear_light_profiles_to_light_profiles

    # --- grid from real-space mask (matches imaging behaviour) ---
    grid = fit.dataset.real_space_mask.derive_grid.all_false

    if image_plane_lines is None and source_plane_lines is None:
        image_plane_lines, image_plane_line_colors, source_plane_lines, source_plane_line_colors = (
            _compute_critical_curve_lines(tracer, grid)
        )

    traced_grids = tracer.traced_grid_2d_list_from(grid=grid)
    lens_galaxies = ag.Galaxies(galaxies=tracer.planes[0])
    lens_image = lens_galaxies.image_2d_from(grid=traced_grids[0])

    deflections = tracer.deflections_yx_2d_from(grid=grid)
    deflections_y = aa.Array2D(values=deflections.slim[:, 0], mask=grid.mask)
    deflections_x = aa.Array2D(values=deflections.slim[:, 1], mask=grid.mask)

    magnification = LensCalc.from_mass_obj(tracer).magnification_2d_from(grid=grid)

    fig, axes = plt.subplots(3, 3, figsize=conf_subplot_figsize(3, 3))
    axes_flat = list(axes.flatten())

    # Panel 0: Dirty Model Image
    plot_array(array=fit.dirty_model_image, ax=axes_flat[0], title="Dirty Model Image",
               lines=image_plane_lines, line_colors=image_plane_line_colors,
               colormap=colormap)

    # Panel 1: Lensed source image (image-plane projection).
    # Use galaxy_image_dict so that pixelized (inversion) sources are included.
    try:
        galaxy_image_dict = fit.galaxy_image_dict
        source_galaxies_list = tracer.planes[final_plane_index]
        source_model_img = sum(
            galaxy_image_dict[galaxy]
            for galaxy in source_galaxies_list
            if galaxy in galaxy_image_dict
        )
        if np.all(source_model_img == 0):
            source_model_img = None
    except Exception:
        source_model_img = None
    if source_model_img is not None:
        plot_array(array=source_model_img, ax=axes_flat[1], title="Source Model Image",
                   colormap=colormap,
                   lines=image_plane_lines, line_colors=image_plane_line_colors)
    else:
        axes_flat[1].axis("off")

    # Panel 2: Source Plane (No Zoom) with caustics
    _plot_source_plane(fit, axes_flat[2], final_plane_index, zoom_to_brightest=False,
                       colormap=colormap, title="Source Plane (No Zoom)",
                       lines=source_plane_lines, line_colors=source_plane_line_colors)

    # Panel 3: Lens Image (log10)
    plot_array(array=lens_image, ax=axes_flat[3], title="Lens Image",
               lines=image_plane_lines, line_colors=image_plane_line_colors,
               colormap=colormap, use_log10=True)

    # Panel 4: Convergence (log10)
    try:
        convergence = tracer.convergence_2d_from(grid=grid)
        plot_array(array=convergence, ax=axes_flat[4], title="Convergence",
                   colormap=colormap, use_log10=True)
    except Exception:
        axes_flat[4].axis("off")

    # Panel 5: Potential (log10)
    try:
        potential = tracer.potential_2d_from(grid=grid)
        plot_array(array=potential, ax=axes_flat[5], title="Potential",
                   colormap=colormap, use_log10=True)
    except Exception:
        axes_flat[5].axis("off")

    # Panel 6: Deflections Y
    plot_array(array=deflections_y, ax=axes_flat[6], title="Deflections Y",
               colormap=colormap)

    # Panel 7: Deflections X
    plot_array(array=deflections_x, ax=axes_flat[7], title="Deflections X",
               colormap=colormap)

    # Panel 8: Magnification
    plot_array(array=magnification, ax=axes_flat[8], title="Magnification",
               colormap=colormap)

    plt.tight_layout()
    save_figure(fig, path=output_path, filename="tracer", format=output_format)
