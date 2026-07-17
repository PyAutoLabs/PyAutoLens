"""
Matplotlib figures of the gravitational-imaging (potential correction)
technique: masked-data panels, irregular source-reconstruction views and the
multi-panel summaries of the dpsi-only and joint source+dpsi fits.

Ported from the ``potential_correction`` package of Cao et al. 2025
(https://github.com/caoxiaoyue/lensing_potential_correction). If you use this
functionality in your research, please cite Cao et al. 2025; citation
materials are provided at
https://github.com/caoxiaoyue/potential_correction_paper.
"""

import copy
import os

import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from scipy.spatial import Voronoi, voronoi_plot_2d


def _plot_anchor_points(ax, anchor_points):
    if anchor_points is not None and np.shape(anchor_points) == (3, 2):
        anchor_points = np.asarray(anchor_points)
        ax.plot(anchor_points[:, 1], anchor_points[:, 0], "rx")


def imshow_masked_data(
    data_1d,
    mask_2d,
    dpix=None,
    ax=None,
    n_contours=None,
    show_scale=None,
    n_cbar_ticks=None,
    centralize=False,
    **kwargs,
):
    """
    Shows a 1D (slim) data vector on its 2D mask, with masked pixels blanked,
    a colorbar and optional contours.
    """
    data_1d = np.asarray(data_1d)
    mask_2d = np.asarray(mask_2d)
    if centralize:
        data_1d = data_1d - np.median(data_1d)
    data_2d = np.zeros_like(mask_2d, dtype="float")
    data_2d[~mask_2d] = data_1d
    data_2d_masked = np.ma.masked_array(data_2d, mask=mask_2d)

    if "extent" in kwargs.keys():
        extent = kwargs.pop("extent")
    else:
        hw = mask_2d.shape[0] * dpix * 0.5
        extent = [-hw, hw, -hw, hw]
    im = ax.imshow(data_2d_masked, extent=extent, **kwargs)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    if show_scale is None:
        cbar = plt.colorbar(im, cax=cax)
    else:
        fmt_cbar = ticker.ScalarFormatter(useMathText=True)
        fmt_cbar.set_powerlimits((show_scale, show_scale))
        cbar = plt.colorbar(im, cax=cax, format=fmt_cbar)
        cbar.ax.yaxis.set_offset_position("left")
        cbar.update_ticks()
    if n_cbar_ticks is not None:
        cbar.locator = MaxNLocator(nbins=n_cbar_ticks)
        cbar.update_ticks()

    if dpix is None:
        if n_contours is not None:
            raise ValueError("dpix is None, cannot show contours")
    else:
        coord_1d = np.arange(len(mask_2d)) * dpix
        coord_1d = coord_1d - np.mean(coord_1d)
        xgrid, ygrid = np.meshgrid(coord_1d, coord_1d)
        rgrid = np.sqrt(xgrid**2 + ygrid**2)
        limit = np.max(rgrid[~mask_2d])
        ax.set_xlim(-1.0 * limit, limit)
        ax.set_ylim(-1.0 * limit, limit)
        if (n_contours is not None) and isinstance(n_contours, int):
            # invert the grids so contours match autolens imaging orientation
            xgrid = np.flipud(xgrid)
            ygrid = np.flipud(ygrid)
            CS = ax.contour(
                xgrid, ygrid, data_2d_masked, levels=n_contours, colors="k",
                corner_mask=True,
            )
            if show_scale is not None:
                ax.clabel(
                    CS, inline=True,
                    fmt=lambda x: f"{x * 1.0 / 10 ** show_scale:.1f}",
                )
            else:
                ax.clabel(CS, inline=True)

    return ax


def show_image_irregular_interpolate(
    points, values, ax=None, enlarge_factor=1.1, npixels=100, cmap="jet", **kwargs
):
    """
    Shows values on irregular (y, x) points by linear interpolation onto a
    regular grid.
    """
    points = np.asarray(points)
    points = points[:, ::-1]  # to scipy (x, y) order

    half_width = max(np.abs(points.min()), np.abs(points.max()))
    half_width *= enlarge_factor

    coordinate_1d, dpix = np.linspace(
        -1.0 * half_width, half_width, npixels, endpoint=True, retstep=True
    )
    xgrid, ygrid = np.meshgrid(coordinate_1d, coordinate_1d)
    extent = [
        -1.0 * half_width - 0.5 * dpix,
        half_width + 0.5 * dpix,
        -1.0 * half_width - 0.5 * dpix,
        half_width + 0.5 * dpix,
    ]

    source_image = griddata(points, values, (xgrid, ygrid), method="linear", fill_value=0.0)

    im = ax.imshow(source_image, origin="lower", extent=extent, cmap=cmap, **kwargs)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def show_image_irregular(
    points, values, ax=None, enlarge_factor=1.1, title="Source", cmap="jet",
    minima=None, maxima=None,
):
    """
    Shows values on irregular (y, x) points as coloured Voronoi cells.
    """
    points = np.asarray(points)
    points = points[:, ::-1]  # to scipy (x, y) order

    half_width = max(np.abs(points.min()), np.abs(points.max()))
    half_width *= enlarge_factor

    # far-away sentinel points close the outer Voronoi cells
    points = np.append(points, [[999, 999], [-999, 999], [999, -999], [-999, -999]], axis=0)
    vor = Voronoi(points)
    if minima is None:
        minima = min(values)
    if maxima is None:
        maxima = max(values)
    norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    mapper.set_array([])
    voronoi_plot_2d(
        vor, ax=ax, show_points=False, show_vertices=False, line_width=0.05,
        point_size=1, line_colors="k", line_alpha=0.2,
    )
    for r in range(len(vor.point_region) - 4):
        region = vor.regions[vor.point_region[r]]
        if -1 not in region:
            polygon = [vor.vertices[i] for i in region]
            ax.fill(*zip(*polygon), color=mapper.to_rgba(values[r]))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(mapper, cax=cax)
    ax.set_xlim(-half_width, half_width)
    ax.set_ylim(-half_width, half_width)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)


def show_fit_dpsi(fit, output="result.png"):
    """
    The six-panel summary of a ``FitDpsiImaging``: residual data, model,
    residual-of-residual, normalized residual, and the dpsi and dkappa maps.
    """
    plt.figure(figsize=(15, 10))
    cmap = copy.copy(plt.get_cmap("jet"))
    cmap.set_bad(color="white")
    myargs_data = {"origin": "upper", "cmap": cmap}
    myargs_data["extent"] = copy.copy(fit.pair_dpsi_data_obj.data_bound)
    myargs_dpsi = copy.deepcopy(myargs_data)
    xlimit = [
        fit.pair_dpsi_data_obj.xgrid_data_1d.min(),
        fit.pair_dpsi_data_obj.xgrid_data_1d.max(),
    ]
    ylimit = [
        fit.pair_dpsi_data_obj.ygrid_data_1d.min(),
        fit.pair_dpsi_data_obj.ygrid_data_1d.max(),
    ]

    plt.subplot(231)
    ax = plt.gca()
    imshow_masked_data(fit.input_image_residual, fit.masked_imaging.mask, ax=ax, **myargs_data)
    ax.set_title("Data")
    _plot_anchor_points(ax, fit.anchor_points)
    ax.set_xlim(*xlimit)
    ax.set_ylim(*ylimit)

    plt.subplot(232)
    ax = plt.gca()
    imshow_masked_data(fit.model_image_residual_slim, fit.masked_imaging.mask, ax=ax, **myargs_data)
    ax.set_title("Model")
    ax.set_xlim(*xlimit)
    ax.set_ylim(*ylimit)

    residual_of_image_residual = fit.input_image_residual - fit.model_image_residual_slim
    plt.subplot(233)
    ax = plt.gca()
    imshow_masked_data(residual_of_image_residual, fit.masked_imaging.mask, ax=ax, **myargs_data)
    ax.set_title("Residual")
    ax.set_xlim(*xlimit)
    ax.set_ylim(*ylimit)

    norm_residual = residual_of_image_residual / fit.masked_imaging.noise_map.slim
    plt.subplot(234)
    ax = plt.gca()
    imshow_masked_data(norm_residual, fit.masked_imaging.mask, ax=ax, **myargs_data)
    ax.set_title("Normalized Residual")
    ax.set_xlim(*xlimit)
    ax.set_ylim(*ylimit)

    plt.subplot(235)
    ax = plt.gca()
    imshow_masked_data(fit.dpsi_slim, fit.pair_dpsi_data_obj.mask_dpsi, ax=ax, **myargs_dpsi)
    _plot_anchor_points(ax, fit.anchor_points)
    ax.set_title("Dpsi Map")
    ax.set_xlim(*xlimit)
    ax.set_ylim(*ylimit)

    dkappa_slim = fit.pair_dpsi_data_obj.hamiltonian_dpsi @ fit.dpsi_slim
    plt.subplot(236)
    ax = plt.gca()
    imshow_masked_data(dkappa_slim, fit.pair_dpsi_data_obj.mask_dpsi, ax=ax, **myargs_dpsi)
    ax.set_title("Dkappa Map")
    ax.set_xlim(*xlimit)
    ax.set_ylim(*ylimit)

    this_path = os.path.dirname(output)
    if this_path:
        os.makedirs(this_path, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()


def show_fit_dpsi_src(fit, output="result.png", show_src_grid=True, interpolate=True):
    """
    The nine-panel summary of a ``FitDpsiSrcImaging``: data, noise, SNR,
    model, residual, normalized residual, dpsi map, dkappa map and the
    source reconstruction.
    """
    fig = plt.figure(figsize=(15, 10))
    cmap = copy.copy(plt.get_cmap("jet"))
    cmap.set_bad(color="white")
    myargs_data = {"origin": "upper", "cmap": cmap}
    myargs_data["extent"] = copy.copy(fit.pair_dpsi_data_obj.data_bound)
    xlimit = [
        fit.pair_dpsi_data_obj.xgrid_data_1d.min(),
        fit.pair_dpsi_data_obj.xgrid_data_1d.max(),
    ]
    ylimit = [
        fit.pair_dpsi_data_obj.ygrid_data_1d.min(),
        fit.pair_dpsi_data_obj.ygrid_data_1d.max(),
    ]
    myargs_dpsi = copy.deepcopy(myargs_data)
    image_plane_mesh_grid = fit.src_mapper.image_plane_mesh_grid
    source_plane_mesh_grid = fit.src_mapper.source_plane_mesh_grid

    plt.subplot(331)
    ax = plt.gca()
    imshow_masked_data(fit.masked_imaging.data, fit.masked_imaging.mask, ax=ax, **myargs_data)
    ax.set_title("Data")
    _plot_anchor_points(ax, fit.anchor_points)
    ax.set_xlim(*xlimit)
    ax.set_ylim(*ylimit)

    plt.subplot(332)
    ax = plt.gca()
    imshow_masked_data(fit.masked_imaging.noise_map, fit.masked_imaging.mask, ax=ax, **myargs_data)
    ax.set_title("Noise")
    ax.set_xlim(*xlimit)
    ax.set_ylim(*ylimit)

    plt.subplot(333)
    ax = plt.gca()
    imshow_masked_data(
        fit.masked_imaging.data / fit.masked_imaging.noise_map,
        fit.masked_imaging.mask, ax=ax, **myargs_data,
    )
    ax.set_title("SNR")
    ax.set_xlim(*xlimit)
    ax.set_ylim(*ylimit)

    plt.subplot(334)
    ax = plt.gca()
    imshow_masked_data(fit.model_image_slim, fit.masked_imaging.mask, ax=ax, **myargs_data)
    if show_src_grid:
        ax.scatter(image_plane_mesh_grid[:, 1], image_plane_mesh_grid[:, 0], c="black", s=0.5, alpha=0.5)
    ax.set_title("Model")
    ax.set_xlim(*xlimit)
    ax.set_ylim(*ylimit)

    residual = fit.masked_imaging.data - fit.model_image_slim
    plt.subplot(335)
    ax = plt.gca()
    imshow_masked_data(residual, fit.masked_imaging.mask, ax=ax, **myargs_data)
    ax.set_title("Residual")
    ax.set_xlim(*xlimit)
    ax.set_ylim(*ylimit)

    norm_residual = residual / fit.masked_imaging.noise_map
    plt.subplot(336)
    ax = plt.gca()
    imshow_masked_data(norm_residual, fit.masked_imaging.mask, ax=ax, **myargs_data)
    ax.set_title("Norm Residual")
    ax.set_xlim(*xlimit)
    ax.set_ylim(*ylimit)

    n_src_pixels = fit.src_regularization_matrix.shape[0]
    dpsi_slim = fit.src_dpsi_slim[n_src_pixels:]
    plt.subplot(337)
    ax = plt.gca()
    imshow_masked_data(dpsi_slim, fit.pair_dpsi_data_obj.mask_dpsi, ax=ax, **myargs_dpsi)
    _plot_anchor_points(ax, fit.anchor_points)
    ax.set_title("Dpsi Map")
    ax.set_xlim(*xlimit)
    ax.set_ylim(*ylimit)

    dkappa_slim = fit.pair_dpsi_data_obj.hamiltonian_dpsi @ dpsi_slim
    plt.subplot(338)
    ax = plt.gca()
    imshow_masked_data(dkappa_slim, fit.pair_dpsi_data_obj.mask_dpsi, ax=ax, **myargs_dpsi)
    ax.set_title("Dkappa Map")
    ax.set_xlim(*xlimit)
    ax.set_ylim(*ylimit)

    src_slim = fit.src_dpsi_slim[0:n_src_pixels]
    plt.subplot(339)
    ax = plt.gca()
    if interpolate:
        show_image_irregular_interpolate(
            source_plane_mesh_grid, src_slim, ax=ax, enlarge_factor=1.1, npixels=100, cmap="jet"
        )
    else:
        show_image_irregular(
            source_plane_mesh_grid, src_slim, enlarge_factor=1.1, cmap="jet", ax=ax, title="Source"
        )
    if show_src_grid:
        ax.scatter(source_plane_mesh_grid[:, 1], source_plane_mesh_grid[:, 0], c="black", s=0.1, alpha=0.5)
    ax.set_title("Source")

    plt.tight_layout()

    if output == "show":
        return fig
    this_path = os.path.dirname(output)
    if this_path:
        os.makedirs(this_path, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
