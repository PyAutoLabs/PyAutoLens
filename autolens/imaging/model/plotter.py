import matplotlib.pyplot as plt
import numpy as np
from typing import List

import autoarray as aa
from autogalaxy.imaging.model.plotter import PlotterImaging as AgPlotterImaging
from autogalaxy.imaging.plot.fit_imaging_plots import (
    fits_fit,
    fits_galaxy_images,
    fits_model_galaxy_images,
)

from autolens.analysis.plotter import Plotter
from autolens.imaging.fit_imaging import FitImaging
from autolens.imaging.plot.fit_imaging_plots import (
    subplot_fit,
    subplot_fit_log10,
    subplot_of_planes,
    subplot_tracer_from_fit,
    subplot_fit_combined,
    subplot_fit_combined_log10,
    _compute_critical_curve_lines,
)

from autolens.analysis.plotter import plot_setting


class PlotterImaging(Plotter):

    imaging = AgPlotterImaging.imaging
    imaging_combined = AgPlotterImaging.imaging_combined

    def fit_imaging(
        self, fit: FitImaging, quick_update: bool = False
    ):
        """
        Visualizes a `FitImaging` object, which fits an imaging dataset.

        Parameters
        ----------
        fit
            The maximum log likelihood `FitImaging` of the non-linear search.
        quick_update
            If True only the essential subplot_fit is output.
        """

        def should_plot(name):
            return plot_setting(section=["fit", "fit_imaging"], name=name)

        output_path = str(self.image_path)
        fmt = self.fmt

        plane_indexes_to_plot = [i for i in fit.tracer.plane_indexes_with_images if i != 0]

        # Compute critical curves and caustics once for all subplot functions.
        tracer = fit.tracer_linear_light_profiles_to_light_profiles
        _zoom = aa.Zoom2D(mask=fit.mask)
        _cc_grid = aa.Grid2D.from_extent(
            extent=_zoom.extent_from(buffer=0), shape_native=_zoom.shape_native
        )
        ip_lines, ip_colors, sp_lines, sp_colors = _compute_critical_curve_lines(tracer, _cc_grid)

        if should_plot("subplot_fit") or quick_update:

            if len(fit.tracer.planes) > 2:
                for plane_index in plane_indexes_to_plot:
                    subplot_fit(
                        fit, output_path=output_path, output_format=fmt,
                        plane_index=plane_index,
                        image_plane_lines=ip_lines, image_plane_line_colors=ip_colors,
                        source_plane_lines=sp_lines, source_plane_line_colors=sp_colors,
                    )
            else:
                subplot_fit(
                    fit, output_path=output_path, output_format=fmt,
                    image_plane_lines=ip_lines, image_plane_line_colors=ip_colors,
                    source_plane_lines=sp_lines, source_plane_line_colors=sp_colors,
                )

        if quick_update:
            return

        if plot_setting(section="tracer", name="subplot_tracer"):
            subplot_tracer_from_fit(
                fit, output_path=output_path, output_format=fmt,
                image_plane_lines=ip_lines, image_plane_line_colors=ip_colors,
                source_plane_lines=sp_lines, source_plane_line_colors=sp_colors,
            )

        if should_plot("subplot_fit_log10"):
            try:
                if len(fit.tracer.planes) > 2:
                    for plane_index in plane_indexes_to_plot:
                        subplot_fit_log10(
                            fit, output_path=output_path, output_format=fmt,
                            plane_index=plane_index,
                            image_plane_lines=ip_lines, image_plane_line_colors=ip_colors,
                            source_plane_lines=sp_lines, source_plane_line_colors=sp_colors,
                        )
                else:
                    subplot_fit_log10(
                        fit, output_path=output_path, output_format=fmt,
                        image_plane_lines=ip_lines, image_plane_line_colors=ip_colors,
                        source_plane_lines=sp_lines, source_plane_line_colors=sp_colors,
                    )
            except ValueError:
                pass

        if should_plot("subplot_of_planes"):
            subplot_of_planes(fit, output_path=output_path, output_format=fmt)

        if should_plot("fits_fit"):
            fits_fit(fit=fit, output_path=self.image_path)

        if should_plot("fits_galaxy_images"):
            fits_galaxy_images(fit=fit, output_path=self.image_path)

        if should_plot("fits_model_galaxy_images"):
            fits_model_galaxy_images(fit=fit, output_path=self.image_path)

    def fit_imaging_combined(
            self,
            fit_list: List[FitImaging],
            quick_update: bool = False,
    ):
        """
        Output visualization of all `FitImaging` objects in a summed combined analysis.

        Parameters
        ----------
        fit_list
            The list of imaging fits which are visualized.
        """

        def should_plot(name):
            return plot_setting(section=["fit", "fit_imaging"], name=name)

        output_path = str(self.image_path)
        fmt = self.fmt

        if should_plot("subplot_fit") or quick_update:
            subplot_fit_combined(fit_list, output_path=output_path, output_format=fmt)

            if quick_update:
                return

            subplot_fit_combined_log10(fit_list, output_path=output_path, output_format=fmt)
