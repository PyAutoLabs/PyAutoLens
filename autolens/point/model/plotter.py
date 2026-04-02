import autogalaxy as ag

from autolens.analysis.plotter import Plotter
from autolens.imaging.plot.fit_imaging_plots import _compute_critical_curve_lines

from autolens.point.fit.dataset import FitPointDataset
from autolens.point.plot.fit_point_plots import subplot_fit as subplot_fit_point
from autolens.point.dataset import PointDataset
from autolens.point.plot.point_dataset_plots import subplot_dataset

from autolens.analysis.plotter import plot_setting


class PlotterPoint(Plotter):
    def dataset_point(self, dataset: PointDataset):
        """
        Output visualization of a `PointDataset` dataset.

        Parameters
        ----------
        dataset
            The point dataset which is visualized.
        """

        def should_plot(name):
            return plot_setting(section=["point_dataset"], name=name)

        output_path = str(self.image_path)
        fmt = self.fmt

        if should_plot("subplot_dataset"):
            subplot_dataset(dataset, output_path=output_path, output_format=fmt)

    def fit_point(
        self,
        fit: FitPointDataset,
        quick_update: bool = False,
        image_plane_lines=None,
        image_plane_line_colors=None,
        source_plane_lines=None,
        source_plane_line_colors=None,
    ):
        """
        Visualizes a `FitPointDataset` object.

        Parameters
        ----------
        fit
            The maximum log likelihood `FitPointDataset` of the non-linear search.
        image_plane_lines
            Pre-computed critical-curve lines to overlay on image-plane panels.
        image_plane_line_colors
            Colours for each image-plane line.
        source_plane_lines
            Pre-computed caustic lines to overlay on source-plane panels.
        source_plane_line_colors
            Colours for each source-plane line.
        """

        def should_plot(name):
            return plot_setting(section=["fit", "fit_point_dataset"], name=name)

        output_path = str(self.image_path)
        fmt = self.fmt

        # Use pre-computed critical curves if provided, otherwise compute once here.
        if image_plane_lines is None and source_plane_lines is None:
            grid = ag.Grid2D.from_extent(
                extent=fit.dataset.extent_from(), shape_native=(100, 100)
            )
            ip_lines, ip_colors, sp_lines, sp_colors = _compute_critical_curve_lines(
                fit.tracer, grid
            )
        else:
            ip_lines, ip_colors, sp_lines, sp_colors = (
                image_plane_lines, image_plane_line_colors,
                source_plane_lines, source_plane_line_colors,
            )

        if should_plot("subplot_fit") or quick_update:
            subplot_fit_point(
                fit, output_path=output_path, output_format=fmt,
                image_plane_lines=ip_lines,
                image_plane_line_colors=ip_colors,
                source_plane_lines=sp_lines,
                source_plane_line_colors=sp_colors,
            )

        if quick_update:
            return
