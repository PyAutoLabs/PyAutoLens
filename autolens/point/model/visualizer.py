from copy import copy

import autofit as af

from autolens.point.model.plotter import PlotterPoint
from autolens.point.fit.positions.image.abstract import AbstractFitPositionsImagePair
from autolens.lens.plot.critical_curves import critical_curve_grid_from
from autolens.point.plot import point_sources as ps


class VisualizerPoint(af.Visualizer):
    @staticmethod
    def visualize_before_fit(
        analysis,
        paths: af.AbstractPaths,
        model: af.AbstractPriorModel,
    ):
        """
        PyAutoFit calls this function immediately before the non-linear search begins.

        It visualizes objects which do not change throughout the model fit like the dataset.

        Parameters
        ----------
        paths
            The paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.
        model
            The model object, which includes model components representing the galaxies that are fitted to
            the imaging data.
        """

        plotter = PlotterPoint(
            image_path=paths.image_path, title_prefix=analysis.title_prefix
        )

        plotter.dataset_point(
            dataset=analysis.dataset, solver=getattr(analysis, "solver", None)
        )

        if getattr(analysis, "image", None) is not None:
            # Dataset-only figure (observed positions, no model): `image_with_points.png`.
            plotter.image_with_points(image=analysis.image, dataset=analysis.dataset)

    @staticmethod
    def visualize(
        analysis,
        paths: af.DirectoryPaths,
        instance: af.ModelInstance,
        during_analysis: bool,
        quick_update: bool = False,
    ):
        """
        Output images of the maximum log likelihood model inferred by the model-fit. This function is called throughout
        the non-linear search at regular intervals, and therefore provides on-the-fly visualization of how well the
        model-fit is going.

        The visualization performed by this function includes:

        - Images of the best-fit `Tracer`, including the images of each of its galaxies.

        - Images of the best-fit `FitPointDataset`, including the model-image, residuals and chi-squared of its fit to
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
        fit = analysis.fit_for_visualization(instance=instance)

        plotter = PlotterPoint(
            image_path=paths.image_path, title_prefix=analysis.title_prefix
        )

        tracer = fit.tracer
        solver = getattr(fit, "solver", None)

        grid = critical_curve_grid_from(dataset=fit.dataset)

        # The critical curves and caustics of the point source's own plane (and of the last plane, for the tracer
        # figures), computed once from the FULL tracer on the solver extent for every figure (the fast
        # deflection-field contouring of `point_sources.curves_for_planes`).
        curves = curves_for_point_fits(fits=[fit], solver=solver)
        ip_lines, ip_colors, sp_lines, sp_colors = last_plane_lines_from(tracer, curves)

        plotter.fit_point(fit=fit, quick_update=quick_update, curves=curves)

        # Output even on `quick_update`, so the observed image with model positions updates on-the-fly. Saved as
        # `fit_image_with_points.png`, so the dataset-only `image_with_points.png` from before the fit is kept.
        if getattr(analysis, "image", None) is not None:
            plotter.image_with_points(
                image=analysis.image,
                dataset=fit.dataset,
                fit=fit,
                image_plane_lines=ip_lines,
                image_plane_line_colors=ip_colors,
                curves=curves,
            )

        if quick_update:
            return

        plotter.fit_per_image(fit_list=[fit], curves=curves, solver=solver)

        plotter.tracer(
            tracer=tracer,
            grid=grid,
            image_plane_lines=ip_lines,
            image_plane_line_colors=ip_colors,
            source_plane_lines=sp_lines,
            source_plane_line_colors=sp_colors,
        )
        plotter.galaxies(
            galaxies=tracer.galaxies,
            grid=grid,
        )

        # Off by default (`fit_point_dataset: subplot_point_solver`): re-solves the best-fit model with NumPy to show
        # the solver's refinement steps, so it is output only for the full (not quick-update) visualization. The
        # figure computes the critical curves of the source's own plane on its own (per-source) extent.
        plotter.point_solver(fit=fit)

    @staticmethod
    def visualize_before_fit_combined(
        analyses,
        paths: af.AbstractPaths,
        model: af.AbstractPriorModel,
    ):
        """
        Before the fit of several point datasets (e.g. a cluster), output the figures of every source in one panel
        to the top-level ``image`` folder: ``dataset_point.png`` and, if an observed image was passed to an
        ``AnalysisPoint``, ``image_with_points.png``. Each dataset's own figures are output by
        ``visualize_before_fit`` to its ``analyses/analysis_<i>`` folder.
        """
        analyses = _point_analyses_from(analyses)
        if len(analyses) < 2:
            return

        plotter = PlotterPoint(
            image_path=paths.image_path, title_prefix=analyses[0].title_prefix
        )
        _visualize_datasets_combined(plotter, analyses)

    @staticmethod
    def visualize_combined(
        analyses,
        paths: af.AbstractPaths,
        instance: af.ModelInstance,
        during_analysis: bool,
        quick_update: bool = False,
    ):
        """
        Output the figures of every point dataset of a multi-dataset (e.g. cluster) fit in one panel, to the
        top-level ``image`` folder of the search (each dataset's own figures are output by ``visualize`` to its
        ``analyses/analysis_<i>`` folder):

        - ``fit.png``: every source's observed and model positions and its critical curves at its own redshift
          (also output on a quick update);
        - ``<family>/fit.png``: the cluster overview, per-image contributions, fitted-plane details and the
          complete source-plane view. Additional groups of six zooms use ``fit_page_02.png``, etc.;
        - ``fit_image_with_points.png`` if an observed image was passed to an ``AnalysisPoint``;
        - the dataset figures of ``visualize_before_fit_combined`` (``dataset_point.png``,
          ``image_with_points.png``), as a factor-graph fit does not call that function.

        The curves are computed once, from the FULL tracer, for every source plane.
        """
        pairs = [
            (analysis, single_instance)
            for analysis, single_instance in zip(analyses, instance)
            if _is_point_analysis(analysis)
        ]
        if len(pairs) < 2:
            return

        analyses = [analysis for analysis, _ in pairs]
        plotter = PlotterPoint(
            image_path=paths.image_path, title_prefix=analyses[0].title_prefix
        )

        fit_list = [
            _combined_plot_fit(analysis.fit_for_visualization(instance=single_instance))
            for analysis, single_instance in pairs
        ]
        solver = getattr(analyses[0], "solver", None) or getattr(
            fit_list[0], "solver", None
        )
        curves = curves_for_point_fits(fits=fit_list, solver=solver)

        plotter.fit_point_combined(
            fit_list=fit_list, curves=curves, solver=solver, quick_update=quick_update
        )

        image = _image_from(analyses)
        if image is not None:
            plotter.image_with_points_combined(
                image=image,
                dataset_list=[fit.dataset for fit in fit_list],
                fit_list=fit_list,
                curves=curves,
                solver=solver,
            )

        if quick_update:
            return

        plotter.fit_per_image(
            fit_list=fit_list, curves=curves, solver=solver, combined=True, image=image
        )
        _visualize_datasets_combined(plotter, analyses)


def _combined_plot_fit(fit):
    """Share a fixed image-plane solution across all panels of this visualization call."""
    positions = fit.positions
    if not isinstance(positions, AbstractFitPositionsImagePair):
        return fit
    if type(positions).model_data is not AbstractFitPositionsImagePair.model_data:
        return fit
    result = copy(fit)
    result.positions = copy(positions)
    result.positions._plot_model_data = positions.model_data
    return result


def _is_point_analysis(analysis) -> bool:
    from autolens.point.model.analysis import AnalysisPoint

    return isinstance(analysis, AnalysisPoint)


def _point_analyses_from(analyses):
    if analyses is None:
        return []
    return [getattr(a, "analysis", a) for a in analyses if _is_point_analysis(getattr(a, "analysis", a))]


def _image_from(analyses):
    for analysis in analyses:
        image = getattr(analysis, "image", None)
        if image is not None:
            return image
    return None


def _visualize_datasets_combined(plotter, analyses):
    solver = getattr(analyses[0], "solver", None)
    dataset_list = [analysis.dataset for analysis in analyses]
    plotter.dataset_point_combined(dataset_list=dataset_list, solver=solver)
    image = _image_from(analyses)
    if image is not None:
        plotter.image_with_points_combined(
            image=image, dataset_list=dataset_list, solver=solver
        )


def curves_for_point_fits(fits, solver=None):
    """
    The per-plane critical curves and caustics of point-source fits: every source's own plane and the last plane,
    from the FULL tracer, on the extent of the positions figures (the solver grid, enlarged to contain every observed
    position).
    """
    records = ps.records_from_fits(fits)
    extent = ps.union_extent_from(records, solver=solver)
    return ps.curves_for_fits(fits, records, extent)


def last_plane_lines_from(tracer, curves):
    """
    The last plane's curves as the ``(image_plane_lines, colors, source_plane_lines, colors)`` the tracer and galaxy
    figures draw (``None`` lines when there are none).
    """
    last = curves.get(len(tracer.planes) - 1) if curves else None
    if last is None:
        return None, None, None, None
    ip_lines, ip_colors = last.image_plane_lines()
    sp_lines, sp_colors = last.source_plane_lines()
    return ip_lines, ip_colors, sp_lines, sp_colors
