
from autolens.analysis.plotter import Plotter
from autolens.point.fit.dataset import FitPointDataset
from autolens.point.plot.fit_point_plots import subplot_fit as subplot_fit_point
from autolens.point.plot.fit_point_plots import plot_image_with_points
from autolens.point.dataset import PointDataset
from autolens.point.plot.point_dataset_plots import subplot_dataset
from autolens.point.plot import point_solver_plots
from autolens.point.plot import multi_source_plots

from autolens.analysis.plotter import plot_setting


def _setting(section, name: str, default: bool = True) -> bool:
    """
    A ``plots.yaml`` toggle which falls back to ``default`` when a (e.g. older workspace) config does not have the
    key.
    """
    try:
        return plot_setting(section=section, name=name)
    except KeyError:
        return default


class PlotterPoint(Plotter):
    def dataset_point(self, dataset: PointDataset, solver=None, family_index=None):
        """
        Output visualization of a `PointDataset` dataset.

        Parameters
        ----------
        dataset
            The point dataset which is visualized.
        solver
            The `PointSolver` of the model-fit, whose grid sets the extent of the positions panel. If `None`, the
            panel extent falls back to the critical-curve region around the positions.
        """

        def should_plot(name):
            return plot_setting(section=["point_dataset"], name=name)

        output_path = str(self.image_path)
        fmt = self.fmt

        if should_plot("subplot_dataset"):
            subplot_dataset(
                dataset,
                output_path=output_path,
                output_format=fmt,
                title_prefix=self.title_prefix,
                solver=solver,
                family_index=family_index,
            )

    def image_with_points(
        self,
        image,
        dataset: PointDataset,
        fit: FitPointDataset = None,
        image_plane_lines=None,
        image_plane_line_colors=None,
        curves=None,
        family_index=None,
    ):
        """
        Output an observed image of the lensed point source with the point-source positions overlaid.

        Before the fit (``fit=None``) only the observed positions are drawn, output as `image_with_points.png`
        and controlled by the `point_dataset: image_with_points` setting. After the fit the model positions (and
        critical curves, if given) are also drawn, output as `fit_image_with_points.png` (so both figures coexist)
        and controlled by the `fit_point_dataset: fit_image_with_points` setting.

        Parameters
        ----------
        image
            The observed image of the lensed point source (e.g. a quasar), used only for visualization.
        dataset
            The point dataset whose observed positions are overlaid.
        fit
            The maximum log likelihood `FitPointDataset`, whose model positions are overlaid if given.
        image_plane_lines
            Pre-computed critical-curve lines to overlay on the image.
        image_plane_line_colors
            Colours for each image-plane line.
        curves
            The per-plane curves of :func:`~autolens.point.plot.point_sources.curves_for_planes`, so each source's
            curves are drawn at its own redshift.
        """
        if fit is None:
            section, name = ["point_dataset"], "image_with_points"
        else:
            section, name = ["fit", "fit_point_dataset"], "fit_image_with_points"

        if not _setting(section, name):
            return

        plot_image_with_points(
            image=image,
            dataset=dataset,
            fit=fit,
            output_path=str(self.image_path),
            output_format=self.fmt,
            image_plane_lines=image_plane_lines,
            image_plane_line_colors=image_plane_line_colors,
            title_prefix=self.title_prefix,
            curves=curves,
            family_index=family_index,
        )

    def fit_point(
        self,
        fit: FitPointDataset,
        quick_update: bool = False,
        image_plane_lines=None,
        image_plane_line_colors=None,
        source_plane_lines=None,
        source_plane_line_colors=None,
        curves=None,
        family_index=None,
    ):
        """
        Visualizes a `FitPointDataset` object: ``fit.png`` (controlled by ``fit: subplot_fit``, also output on a
        quick update).

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
        curves
            The per-plane curves of :func:`~autolens.point.plot.point_sources.curves_for_planes`; computed if
            ``None``.
        """
        if not quick_update and not _setting(["fit", "fit_point_dataset"], "subplot_fit"):
            return

        subplot_fit_point(
            fit,
            output_path=str(self.image_path),
            output_format=self.fmt,
            image_plane_lines=image_plane_lines,
            image_plane_line_colors=image_plane_line_colors,
            source_plane_lines=source_plane_lines,
            source_plane_line_colors=source_plane_line_colors,
            title_prefix=self.title_prefix,
            curves=curves,
            family_index=family_index,
        )

    def fit_per_image(
        self,
        fit_list,
        curves=None,
        solver=None,
        combined: bool = False,
        image=None,
        family_index=None,
    ):
        """
        Output a cluster diagnostic per family for a combined fit, or the single dataset's
        residual ranking, image-plane zoom and source-plane panel in ``fit_per_image.png``.
        """
        name = "fit_per_image_combined" if combined else "fit_per_image"
        if not _setting(["fit", "fit_point_dataset"], name):
            return
        if combined:
            from autolens.point.plot.family_fit_plots import plot_family_fits

            return plot_family_fits(
                fit_list,
                image=image,
                single_panel=_setting(
                    ["fit", "fit_point_dataset"], "family_fit_single_panel", default=False
                ),
                output_path=str(self.image_path),
                output_format=self.fmt,
                title_prefix=self.title_prefix,
            )
        multi_source_plots.subplot_fit_per_image(
            fit_list,
            curves=curves,
            solver=solver,
            output_path=str(self.image_path),
            output_format=self.fmt,
            title_prefix=self.title_prefix,
            family_index=family_index,
        )

    def dataset_point_combined(self, dataset_list, solver=None, image_extent=None):
        """
        Output ``dataset_point.png`` of every point dataset of a multi-dataset (e.g. cluster) fit in one panel,
        controlled by ``point_dataset: subplot_dataset_combined``.
        """
        if not _setting(["point_dataset"], "subplot_dataset_combined"):
            return
        multi_source_plots.plot_dataset_positions(
            dataset_list,
            solver=solver,
            image_extent=image_extent,
            output_path=str(self.image_path),
            output_format=self.fmt,
            title_prefix=self.title_prefix,
        )

    def fit_point_combined(
        self,
        fit_list,
        curves=None,
        solver=None,
        quick_update: bool = False,
        image_extent=None,
    ):
        """
        Output ``fit.png`` of every point dataset of a multi-dataset (e.g. cluster) fit in one panel ("Fit
        Positions", each source's curves at its own redshift), controlled by
        ``fit_point_dataset: subplot_fit_combined`` and always output on a quick update.
        """
        if not quick_update and not _setting(
            ["fit", "fit_point_dataset"], "subplot_fit_combined"
        ):
            return
        multi_source_plots.plot_fit_positions(
            fit_list,
            image_extent=image_extent,
            solver=solver,
            curves=curves,
            output_path=str(self.image_path),
            output_format=self.fmt,
            title_prefix=self.title_prefix,
        )

    def image_with_points_combined(self, image, dataset_list, fit_list=None, curves=None, solver=None):
        """
        Output ``image_with_points.png`` (no fit) or ``fit_image_with_points.png`` of every point dataset of a
        multi-dataset (e.g. cluster) fit over one observed image, controlled by
        ``point_dataset: image_with_points_combined`` and ``fit_point_dataset: fit_image_with_points_combined``.
        """
        if fit_list is None:
            section, name = ["point_dataset"], "image_with_points_combined"
        else:
            section, name = ["fit", "fit_point_dataset"], "fit_image_with_points_combined"
        if not _setting(section, name):
            return
        multi_source_plots.plot_image_with_points(
            image=image,
            datasets=dataset_list,
            fits=fit_list,
            curves=curves,
            solver=solver,
            output_path=str(self.image_path),
            output_format=self.fmt,
            title_prefix=self.title_prefix,
        )

    def point_solver(self, fit: FitPointDataset, image_plane_lines=None, family_index=None):
        """
        Output `point_solver.png`: how the `PointSolver` refines its triangles onto the images of the fit's model.

        The model's point source is re-solved with NumPy (the JAX path pads the steps and, inside `jax.jit`, hides
        them), so this is a diagnostic of the model-fit's own solver at this model: did the grid cover every image
        and how many triangles survive each step. A warning is logged when a step keeps nearly as many triangles as
        the JAX solver's `MAX_CONTAINING_SIZE`, which the JAX path truncates to.

        Controlled by the `fit_point_dataset: subplot_point_solver` setting, which is off by default as the
        re-solve and figure cost a few seconds per visualization (more for a many-galaxy multi-plane cluster). The
        figure is not output for fits without a `PointSolver` (e.g. source-plane fits of a model with no solver).

        Parameters
        ----------
        fit
            The maximum log likelihood `FitPointDataset` of the non-linear search.
        image_plane_lines
            Pre-computed critical-curve lines; `None` computes the critical curves of the source's own plane on the
            figure's extent.
        """
        try:
            if not plot_setting(
                section=["fit", "fit_point_dataset"], name="subplot_point_solver"
            ):
                return
        except KeyError:
            return

        solver = getattr(fit, "solver", None)
        fit_positions = getattr(fit, "positions", None)
        if solver is None or not hasattr(solver, "steps") or fit_positions is None:
            return

        data = point_solver_plots.point_solver_data_from(
            solver=solver,
            tracer=fit.tracer,
            source_plane_coordinate=fit_positions.source_plane_coordinate,
            positions=fit.dataset.positions,
            plane_redshift=fit_positions.plane_redshift,
            image_plane_lines=image_plane_lines,
        )
        point_solver_plots.warn_if_near_jax_cap(data)
        point_solver_plots.plot_point_solver_from_data(
            data=data,
            output_path=str(self.image_path),
            output_format=self.fmt,
            title_prefix=self.title_prefix,
            family_index=family_index,
        )
