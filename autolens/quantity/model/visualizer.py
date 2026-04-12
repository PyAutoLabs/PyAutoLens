import os

import autofit as af
from autoconf.test_mode import skip_visualization

from autogalaxy.quantity.model.plotter import PlotterQuantity

from autolens.analysis.plotter import Plotter


class VisualizerQuantity(af.Visualizer):
    @staticmethod
    def visualize(
        analysis,
        paths: af.DirectoryPaths,
        instance: af.ModelInstance,
        during_analysis: bool,
    ):
        """
        Output images of the maximum log likelihood model inferred by the model-fit. This function is called throughout
        the non-linear search at regular intervals, and therefore provides on-the-fly visualization of how well the
        model-fit is going.

        The visualization performed by this function includes:

        - Images of the best-fit `Plane`, including the images of each of its galaxies.

        - Images of the best-fit `FitQuantity`, including the model-image, residuals and chi-squared of its fit to
        the imaging data.

        The images output by this function are customized using the file `config/visualize/plots.yaml`.

        Parameters
        ----------
        paths
            The paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization, and the pickled objects used by the aggregator output by this function.
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).
        """

        if skip_visualization():
            return

        fit = analysis.fit_quantity_for_instance(instance=instance)

        plotter = PlotterQuantity(image_path=paths.image_path)
        plotter.fit_quantity(fit=fit)

        plotter = Plotter(image_path=paths.image_path)
        plotter.tracer(
            tracer=fit.tracer,
            grid=analysis.dataset.grids.lp,
        )
