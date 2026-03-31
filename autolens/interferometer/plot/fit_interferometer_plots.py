import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

import autoarray as aa
import autogalaxy as ag

from autoarray.plot.array import plot_array
from autoarray.plot.utils import save_figure
from autoarray.plot.utils import numpy_lines as _to_lines
from autogalaxy.plot.plot_utils import _critical_curves_from


def _plot_yx(y, x, ax, title, xlabel="", ylabel=""):
    """
    Render a scatter plot of *y* versus *x* into an existing axes object.

    Parameters
    ----------
    y : array-like
        Dependent-variable values (y-axis).
    x : array-like
        Independent-variable values (x-axis).
    ax : matplotlib.axes.Axes
        The axes into which the scatter plot is drawn.
    title : str
        Axes title.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    """
    ax.scatter(x, y, s=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _plot_source_plane(fit, ax, plane_index, zoom_to_brightest=True,
                       colormap=None, use_log10=False):
    """
    Plot the source-plane image (or a blank inversion placeholder) into an axes.

    For parametric sources the function ray-traces a zoomed real-space grid
    to the source plane and renders the resulting image via
    :func:`~autoarray.plot.array.plot_array`.  When the plane contains a
    pixelization (inversion source), the axes are turned off and labelled
    as a placeholder, because the reconstruction is rendered separately.

    Parameters
    ----------
    fit : FitInterferometer
        The interferometer fit providing the tracer and real-space mask.
    ax : matplotlib.axes.Axes or None
        The axes into which the source-plane image is drawn.  Passing
        ``None`` is a no-op.
    plane_index : int
        Index of the plane in ``fit.tracer.planes`` to visualise.
    zoom_to_brightest : bool, optional
        Reserved for future zoomed rendering; currently unused in the
        rendering call.
    colormap : str, optional
        Matplotlib colormap name.
    use_log10 : bool, optional
        If ``True`` the colour scale is applied on a log10 stretch.
    """
    tracer = fit.tracer_linear_light_profiles_to_light_profiles
    if not tracer.planes[plane_index].has(cls=aa.Pixelization):
        zoom = aa.Zoom2D(mask=fit.dataset.real_space_mask)
        grid = aa.Grid2D.from_extent(
            extent=zoom.extent_from(buffer=0), shape_native=zoom.shape_native
        )
        traced_grids = tracer.traced_grid_2d_list_from(grid=grid)
        plane_galaxies = ag.Galaxies(galaxies=tracer.planes[plane_index])
        image = plane_galaxies.image_2d_from(grid=traced_grids[plane_index])
        plot_array(
            array=image, ax=ax,
            title=f"Source Plane {plane_index}",
            colormap=colormap, use_log10=use_log10,
        )
    else:
        if ax is not None:
            ax.axis("off")
            ax.set_title(f"Source Reconstruction (plane {plane_index})")


def subplot_fit(
    fit,
    output_path: Optional[str] = None,
    output_format: str = "png",
    colormap: Optional[str] = None,
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
    tracer = fit.tracer_linear_light_profiles_to_light_profiles
    final_plane_index = len(fit.tracer.planes) - 1

    fig, axes = plt.subplots(3, 4, figsize=(28, 21))
    axes_flat = list(axes.flatten())

    # Panel 0: amplitudes vs UV distances
    _plot_yx(
        y=np.real(fit.residual_map),
        x=fit.dataset.uv_distances / 10 ** 3.0,
        ax=axes_flat[0],
        title="Amplitudes vs UV-Distance",
        xlabel=r"k$\lambda$",
    )

    plot_array(array=fit.dirty_image, ax=axes_flat[1], title="Dirty Image",
               colormap=colormap)
    plot_array(array=fit.dirty_signal_to_noise_map, ax=axes_flat[2],
               title="Dirty Signal-To-Noise Map", colormap=colormap)
    plot_array(array=fit.dirty_model_image, ax=axes_flat[3], title="Dirty Model Image",
               colormap=colormap)

    # Panel 4: source image
    _plot_source_plane(fit, axes_flat[4], final_plane_index, colormap=colormap)

    # Normalized residual vs UV distances (real)
    _plot_yx(
        y=np.real(fit.normalized_residual_map),
        x=fit.dataset.uv_distances / 10 ** 3.0,
        ax=axes_flat[5],
        title="Norm Residual vs UV-Distance (real)",
        ylabel=r"$\sigma$",
        xlabel=r"k$\lambda$",
    )

    # Normalized residual vs UV distances (imag)
    _plot_yx(
        y=np.imag(fit.normalized_residual_map),
        x=fit.dataset.uv_distances / 10 ** 3.0,
        ax=axes_flat[6],
        title="Norm Residual vs UV-Distance (imag)",
        ylabel=r"$\sigma$",
        xlabel=r"k$\lambda$",
    )

    # Panel 7: source plane zoomed
    _plot_source_plane(fit, axes_flat[7], final_plane_index, colormap=colormap)

    plot_array(array=fit.dirty_normalized_residual_map, ax=axes_flat[8],
               title="Dirty Normalized Residual Map", colormap=colormap, cb_unit=r"$\sigma$")

    # Panel 9: clipped to [-1, 1]
    plot_array(
        fit.dirty_normalized_residual_map,
        ax=axes_flat[8],
        title=r"Normalized Residual Map $1\sigma$",
        colormap=colormap,
        use_log10=False,
        vmin=-1.0, vmax=1.0,
        cb_unit=r"$\sigma$",
    )

    plot_array(array=fit.dirty_chi_squared_map, ax=axes_flat[10],
               title="Dirty Chi-Squared Map", colormap=colormap, cb_unit=r"$\chi^2$")

    # Panel 11: source plane not zoomed
    _plot_source_plane(fit, axes_flat[11], final_plane_index,
                       zoom_to_brightest=False, colormap=colormap)

    plt.tight_layout()
    save_figure(fig, path=output_path, filename="fit", format=output_format)


def subplot_fit_real_space(
    fit,
    output_path: Optional[str] = None,
    output_format: str = "png",
    colormap: Optional[str] = None,
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

    if fit.inversion is None:
        # No inversion: image + source plane image
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        axes_flat = list(axes.flatten())

        zoom = aa.Zoom2D(mask=fit.dataset.real_space_mask)
        grid = aa.Grid2D.from_extent(
            extent=zoom.extent_from(buffer=0), shape_native=zoom.shape_native
        )
        traced_grids = tracer.traced_grid_2d_list_from(grid=grid)

        image = tracer.image_2d_from(grid=grid)
        plot_array(array=image, ax=axes_flat[0], title="Image", colormap=colormap)

        source_galaxies = ag.Galaxies(galaxies=tracer.planes[final_plane_index])
        source_image = source_galaxies.image_2d_from(
            grid=traced_grids[final_plane_index]
        )
        plot_array(array=source_image, ax=axes_flat[1], title="Source Plane Image",
                   colormap=colormap)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        axes_flat = list(axes.flatten())
        # Inversion: show blank placeholder panels
        for _ax in axes_flat:
            _ax.axis("off")
        axes_flat[0].set_title("Reconstructed Data")
        axes_flat[1].set_title("Source Plane (Zoom)")

    plt.tight_layout()
    save_figure(fig, path=output_path, filename="fit_real_space", format=output_format)
