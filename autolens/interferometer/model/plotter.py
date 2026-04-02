import autoarray as aa

from autogalaxy.interferometer.model.plotter import (
    PlotterInterferometer as AgPlotterInterferometer,
)

from autogalaxy.interferometer.plot.fit_interferometer_plots import (
    fits_galaxy_images,
    fits_dirty_images,
)

from autolens.interferometer.fit_interferometer import FitInterferometer
from autolens.interferometer.plot.fit_interferometer_plots import (
    subplot_fit,
    subplot_fit_dirty_images,
    subplot_fit_real_space,
    _compute_critical_curve_lines,
)
from autolens.analysis.plotter import Plotter

from autolens.analysis.plotter import plot_setting


class PlotterInterferometer(Plotter):
    interferometer = AgPlotterInterferometer.interferometer

    def fit_interferometer(
        self,
        fit: FitInterferometer,
        quick_update: bool = False,
    ):
        """
        Visualizes a `FitInterferometer` object.

        Parameters
        ----------
        fit
            The maximum log likelihood `FitInterferometer` of the non-linear search.
        """

        def should_plot(name):
            return plot_setting(section=["fit", "fit_interferometer"], name=name)

        output_path = str(self.image_path)
        fmt = self.fmt

        # Compute critical curves and caustics once for all subplot functions.
        tracer = fit.tracer_linear_light_profiles_to_light_profiles
        _zoom = aa.Zoom2D(mask=fit.dataset.real_space_mask)
        _cc_grid = aa.Grid2D.from_extent(
            extent=_zoom.extent_from(buffer=0), shape_native=_zoom.shape_native
        )
        ip_lines, ip_colors, sp_lines, sp_colors = _compute_critical_curve_lines(tracer, _cc_grid)

        if should_plot("subplot_fit"):
            subplot_fit(
                fit, output_path=output_path, output_format=fmt,
                image_plane_lines=ip_lines, image_plane_line_colors=ip_colors,
                source_plane_lines=sp_lines, source_plane_line_colors=sp_colors,
            )

        if should_plot("subplot_fit_dirty_images") or quick_update:
            subplot_fit_dirty_images(
                fit, output_path=output_path, output_format=fmt,
                image_plane_lines=ip_lines, image_plane_line_colors=ip_colors,
            )

        if quick_update:
            return

        if should_plot("subplot_fit_real_space"):
            subplot_fit_real_space(
                fit, output_path=output_path, output_format=fmt,
                source_plane_lines=sp_lines, source_plane_line_colors=sp_colors,
            )

        if should_plot("fits_galaxy_images"):
            fits_galaxy_images(fit=fit, output_path=self.image_path)

        if should_plot("fits_dirty_images"):
            fits_dirty_images(fit=fit, output_path=self.image_path)
