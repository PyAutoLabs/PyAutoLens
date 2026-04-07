import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List

import autoarray as aa
import autogalaxy as ag

from autogalaxy.plot.plot_utils import plot_array
from autoarray.plot.utils import save_figure, hide_unused_axes, conf_subplot_figsize, tight_layout
from autoarray.plot.utils import numpy_positions as _to_positions


def plane_image_from(
    galaxies,
    grid: aa.Grid2D,
    buffer: float = 1.0e-2,
    zoom_to_brightest: bool = True,
) -> aa.Array2D:
    """
    Return the unlensed source-plane image of a list of galaxies.

    The galaxies' light profiles are evaluated directly on *grid* — a plain
    uniform grid, **not** a ray-traced grid.  This shows what the source
    looks like in its own plane, without any lensing distortion.  A typical
    caller passes ``fit.mask.derive_grid.all_false`` (the full unmasked
    image-plane grid) so that the source is rendered at a natural scale and
    position before any optional zoom is applied.

    When ``zoom_to_brightest`` is ``True`` the function first evaluates the
    galaxy images on *grid* to locate the bright region, then builds a
    smaller uniform grid centred on that region and re-evaluates the images
    at full resolution.  The zoom threshold is read from
    ``visualize / general / zoom / plane_percent`` in the config.

    Parameters
    ----------
    galaxies
        The galaxies whose images are summed to form the plane image.
    grid
        Uniform grid on which the source light profiles are evaluated.
        Should be a plain spatial grid (e.g. ``fit.mask.derive_grid.all_false``),
        not a ray-traced source-plane grid.
    buffer
        Arc-second padding added around the bright region when constructing
        the zoomed grid.
    zoom_to_brightest
        If ``True``, zoom the grid in on the brightest pixels before
        evaluating the final image.

    Returns
    -------
    aa.Array2D
        Plane image on the (possibly zoomed) grid.
    """
    from autoconf import conf

    shape = grid.shape_native

    if zoom_to_brightest:
        try:
            image = sum(g.image_2d_from(grid=grid) for g in galaxies)
            image_native = image.native

            zoom_percent = conf.instance["visualize"]["general"]["zoom"]["plane_percent"]
            fractional_value = float(np.max(image_native)) * zoom_percent

            fractional_bool = image_native > fractional_value
            true_indices = np.argwhere(fractional_bool)

            y_max_pix = np.min(true_indices[:, 0])
            y_min_pix = np.max(true_indices[:, 0])
            x_min_pix = np.min(true_indices[:, 1])
            x_max_pix = np.max(true_indices[:, 1])

            grid_native = grid.native
            extent = (
                grid_native[0, x_min_pix][1] - buffer,
                grid_native[0, x_max_pix][1] + buffer,
                grid_native[y_min_pix, 0][0] - buffer,
                grid_native[y_max_pix, 0][0] + buffer,
            )
            extent = aa.util.geometry.extent_symmetric_from(extent=extent)

            pixel_scales = (
                float((extent[3] - extent[2]) / shape[0]),
                float((extent[1] - extent[0]) / shape[1]),
            )
            origin = ((extent[3] + extent[2]) / 2.0, (extent[1] + extent[0]) / 2.0)

            grid = aa.Grid2D.uniform(
                shape_native=shape,
                pixel_scales=pixel_scales,
                origin=origin,
            )
        except (ValueError, IndexError):
            pass

    image = sum(g.image_2d_from(grid=grid) for g in galaxies)
    return aa.Array2D.no_mask(
        values=image.native, pixel_scales=grid.pixel_scales, origin=grid.origin
    )


def subplot_tracer(
    tracer,
    grid: aa.type.Grid2DLike,
    output_path: Optional[str] = None,
    output_format: str = None,
    colormap: Optional[str] = None,
    use_log10: bool = False,
    positions=None,
    image_plane_lines=None,
    image_plane_line_colors=None,
    source_plane_lines=None,
    source_plane_line_colors=None,
):
    """Multi-panel subplot of the tracer: image, source images, and mass quantities.

    Panels (3x3 = 9 axes):
      0: full lensed image with critical curves
      1: source galaxy image (no caustics)
      2: source plane image (with caustics)
      3: lens galaxy image (log10)
      4: convergence (log10, with critical curves)
      5: potential (log10, with critical curves)
      6: deflections y (with critical curves)
      7: deflections x (with critical curves)
      8: magnification (with critical curves)
    """
    from autogalaxy.operate.lens_calc import LensCalc

    final_plane_index = len(tracer.planes) - 1
    traced_grids = tracer.traced_grid_2d_list_from(grid=grid)

    if image_plane_lines is None and source_plane_lines is None:
        from autolens.imaging.plot.fit_imaging_plots import _compute_critical_curve_lines
        image_plane_lines, image_plane_line_colors, source_plane_lines, source_plane_line_colors = (
            _compute_critical_curve_lines(tracer, grid)
        )
    pos_list = _to_positions(positions)

    # --- compute arrays ---
    image = tracer.image_2d_from(grid=grid)

    source_galaxies = ag.Galaxies(galaxies=tracer.planes[final_plane_index])
    source_image = source_galaxies.image_2d_from(grid=traced_grids[final_plane_index])
    try:
        source_vmax = float(np.max(source_image.array))
    except (AttributeError, ValueError):
        source_vmax = None

    lens_galaxies = ag.Galaxies(galaxies=tracer.planes[0])
    lens_image = lens_galaxies.image_2d_from(grid=traced_grids[0])

    convergence = tracer.convergence_2d_from(grid=grid)
    potential = tracer.potential_2d_from(grid=grid)

    deflections = tracer.deflections_yx_2d_from(grid=grid)
    deflections_y = aa.Array2D(values=deflections.slim[:, 0], mask=grid.mask)
    deflections_x = aa.Array2D(values=deflections.slim[:, 1], mask=grid.mask)

    magnification = LensCalc.from_mass_obj(tracer).magnification_2d_from(grid=grid)

    fig, axes = plt.subplots(3, 3, figsize=conf_subplot_figsize(3, 3))
    axes_flat = list(axes.flatten())

    plot_array(array=image, ax=axes_flat[0], title="Model Image",
               lines=image_plane_lines, line_colors=image_plane_line_colors,
               positions=pos_list, colormap=colormap, use_log10=use_log10)
    plot_array(array=source_image, ax=axes_flat[1], title="Source Model Image",
               colormap=colormap, use_log10=use_log10, vmax=source_vmax)
    plot_array(array=source_image, ax=axes_flat[2], title="Source Plane (No Zoom)",
               lines=source_plane_lines, line_colors=source_plane_line_colors,
               colormap=colormap, use_log10=use_log10)
    plot_array(array=lens_image, ax=axes_flat[3], title="Lens Image",
               lines=image_plane_lines, line_colors=image_plane_line_colors,
               colormap=colormap, use_log10=True)
    plot_array(array=convergence, ax=axes_flat[4], title="Convergence",
               colormap=colormap, use_log10=True)
    plot_array(array=potential, ax=axes_flat[5], title="Potential",
               colormap=colormap, use_log10=True)
    plot_array(array=deflections_y, ax=axes_flat[6], title="Deflections Y",
               lines=image_plane_lines, line_colors=image_plane_line_colors, colormap=colormap)
    plot_array(array=deflections_x, ax=axes_flat[7], title="Deflections X",
               lines=image_plane_lines, line_colors=image_plane_line_colors, colormap=colormap)
    plot_array(array=magnification, ax=axes_flat[8], title="Magnification",
               lines=image_plane_lines, line_colors=image_plane_line_colors, colormap=colormap)

    hide_unused_axes(axes_flat)
    tight_layout()
    save_figure(fig, path=output_path, filename="tracer", format=output_format)


def subplot_lensed_images(
    tracer,
    grid: aa.type.Grid2DLike,
    output_path: Optional[str] = None,
    output_format: str = None,
    colormap: Optional[str] = None,
    use_log10: bool = False,
):
    """
    Produce a subplot with one panel per tracer plane showing each plane's image.

    For each plane in the tracer the galaxies in that plane are evaluated on
    the ray-traced grid for that plane, producing the lensed image
    contribution from those galaxies.  Each panel is titled
    ``"Image Of Plane <index>"``.

    Parameters
    ----------
    tracer : Tracer
        The tracer whose planes are ray-traced and imaged.
    grid : aa.type.Grid2DLike
        The 2-D (y, x) arc-second grid on which the lensed images are
        evaluated.
    output_path : str, optional
        Directory in which to save the figure.  If ``None`` the figure is
        not saved to disk.
    output_format : str, optional
        Image format passed to :func:`~autoarray.plot.utils.save_figure`.
    colormap : str, optional
        Matplotlib colormap name.
    use_log10 : bool, optional
        If ``True`` the colour scale is applied on a log10 stretch.
    """
    traced_grids = tracer.traced_grid_2d_list_from(grid=grid)
    n = tracer.total_planes

    fig, axes = plt.subplots(1, n, figsize=conf_subplot_figsize(1, n))
    axes_flat = [axes] if n == 1 else list(np.array(axes).flatten())

    for plane_index in range(n):
        galaxies = ag.Galaxies(galaxies=tracer.planes[plane_index])
        image = galaxies.image_2d_from(grid=traced_grids[plane_index])
        plot_array(
            array=image,
            ax=axes_flat[plane_index],
            title=f"Image Of Plane {plane_index}",
            colormap=colormap,
            use_log10=use_log10,
        )

    tight_layout()
    save_figure(fig, path=output_path, filename="lensed_images", format=output_format)


def subplot_galaxies_images(
    tracer,
    grid: aa.type.Grid2DLike,
    output_path: Optional[str] = None,
    output_format: str = None,
    colormap: Optional[str] = None,
    use_log10: bool = False,
):
    """
    Produce a subplot showing per-galaxy images for every plane in the tracer.

    Renders the following panels in a single row:

    1. Lens-plane (plane 0) image.
    2. For each subsequent plane *i* (i ≥ 1):

       a. The lensed image of galaxies in plane *i* evaluated on the
          ray-traced grid (titled ``"Image Of Plane <i>"``).
       b. The source-plane image of galaxies in plane *i* (titled
          ``"Plane Image Of Plane <i>"``).

    The total number of panels is ``2 * total_planes - 1``.

    Parameters
    ----------
    tracer : Tracer
        The tracer whose planes are ray-traced and imaged.
    grid : aa.type.Grid2DLike
        The 2-D (y, x) arc-second grid on which the images are evaluated.
    output_path : str, optional
        Directory in which to save the figure.  If ``None`` the figure is
        not saved to disk.
    output_format : str, optional
        Image format passed to :func:`~autoarray.plot.utils.save_figure`.
    colormap : str, optional
        Matplotlib colormap name.
    use_log10 : bool, optional
        If ``True`` the colour scale is applied on a log10 stretch.
    """
    traced_grids = tracer.traced_grid_2d_list_from(grid=grid)
    n = 2 * tracer.total_planes - 1

    fig, axes = plt.subplots(1, n, figsize=conf_subplot_figsize(1, n))
    axes_flat = [axes] if n == 1 else list(np.array(axes).flatten())

    idx = 0

    lens_galaxies = ag.Galaxies(galaxies=tracer.planes[0])
    lens_image = lens_galaxies.image_2d_from(grid=traced_grids[0])
    plot_array(
        array=lens_image,
        ax=axes_flat[idx],
        title="Image Of Plane 0",
        colormap=colormap,
        use_log10=use_log10,
    )
    idx += 1

    for plane_index in range(1, tracer.total_planes):
        plane_galaxies = ag.Galaxies(galaxies=tracer.planes[plane_index])
        plane_grid = traced_grids[plane_index]

        image = plane_galaxies.image_2d_from(grid=plane_grid)
        if idx < n:
            plot_array(
                array=image,
                ax=axes_flat[idx],
                title=f"Image Of Plane {plane_index}",
                colormap=colormap,
                use_log10=use_log10,
            )
            idx += 1

        if idx < n:
            plot_array(
                array=image,
                ax=axes_flat[idx],
                title=f"Plane Image Of Plane {plane_index}",
                colormap=colormap,
                use_log10=use_log10,
            )
            idx += 1

    tight_layout()
    save_figure(fig, path=output_path, filename="galaxies_images", format=output_format)


def fits_tracer(
    tracer,
    grid: aa.type.Grid2DLike,
    output_path,
) -> None:
    """Write a FITS file containing lensing maps for the tracer.

    Produces ``tracer.fits`` in *output_path*.  The file contains extensions:
    ``mask``, ``convergence``, ``potential``, ``deflections_y``,
    ``deflections_x``, all evaluated on a zoomed grid derived from
    *grid*'s mask.

    Parameters
    ----------
    tracer : Tracer
        The tracer whose lensing maps are evaluated.
    grid : aa.type.Grid2DLike
        Image-plane grid; a zoomed version is derived internally.
    output_path : str or Path
        Directory in which to write ``tracer.fits``.
    """
    from pathlib import Path
    from autoconf.fitsable import hdu_list_for_output_from

    output_path = Path(output_path)
    zoom = aa.Zoom2D(mask=grid.mask)
    grid_zoom = aa.Grid2D.from_mask(mask=zoom.mask_2d_from(buffer=1))

    deflections = tracer.deflections_yx_2d_from(grid=grid_zoom).native
    image_list = [
        tracer.convergence_2d_from(grid=grid_zoom).native,
        tracer.potential_2d_from(grid=grid_zoom).native,
        deflections[:, :, 0],
        deflections[:, :, 1],
    ]
    hdu_list = hdu_list_for_output_from(
        values_list=[image_list[0].mask.astype("float")] + image_list,
        ext_name_list=["mask", "convergence", "potential", "deflections_y", "deflections_x"],
        header_dict=grid_zoom.mask.header_dict,
    )
    hdu_list.writeto(output_path / "tracer.fits", overwrite=True)


def fits_source_plane_images(
    tracer,
    grid: aa.type.Grid2DLike,
    output_path,
) -> None:
    """Write a FITS file containing source-plane images for each source plane.

    Produces ``source_plane_images.fits`` in *output_path*.  One HDU is
    written per source plane (``tracer.planes[1:]``), named
    ``source_plane_image_1``, ``source_plane_image_2``, …, plus a ``mask``
    extension.  Planes without a
    :class:`~autogalaxy.profiles.light.abstract.LightProfile` produce a
    zero-valued array.

    The shape of the source-plane grid is read from config key
    ``visualize / plots / tracer / fits_source_plane_shape``.

    Parameters
    ----------
    tracer : Tracer
        The tracer whose source-plane images are evaluated.
    grid : aa.type.Grid2DLike
        Image-plane grid; used to derive the zoomed extent for the
        source-plane grid.
    output_path : str or Path
        Directory in which to write ``source_plane_images.fits``.
    """
    import ast
    from pathlib import Path
    from autoconf import conf
    from autoconf.fitsable import hdu_list_for_output_from

    output_path = Path(output_path)
    shape_native = tuple(ast.literal_eval(
        conf.instance["visualize"]["plots"]["tracer"]["fits_source_plane_shape"]
    ))

    zoom = aa.Zoom2D(mask=grid.mask)
    grid_source = aa.Grid2D.from_extent(
        extent=zoom.mask_2d_from(buffer=1).geometry.extent,
        shape_native=shape_native,
    )

    image_list = [grid_source.mask.astype("float")]
    ext_name_list = ["mask"]
    for i, plane in enumerate(tracer.planes[1:]):
        if plane.has(cls=ag.LightProfile):
            image = plane.image_2d_from(grid=grid_source).native
        else:
            image = np.zeros(grid_source.shape_native)
        image_list.append(image)
        ext_name_list.append(f"source_plane_image_{i + 1}")

    hdu_list = hdu_list_for_output_from(
        values_list=image_list,
        ext_name_list=ext_name_list,
        header_dict=grid_source.mask.header_dict,
    )
    hdu_list.writeto(output_path / "source_plane_images.fits", overwrite=True)
