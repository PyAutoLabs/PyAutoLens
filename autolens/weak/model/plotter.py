from autolens.analysis.plotter import Plotter
from autolens.analysis.plotter import plot_setting

from autolens.weak.dataset import WeakDataset
from autolens.weak.fit import FitWeak
from autolens.weak.plot.weak_dataset_plots import subplot_weak_dataset
from autolens.weak.plot.fit_weak_plots import subplot_fit_weak
from autolens.weak.plot.fit_weak_plots import subplot_fit_quick as subplot_fit_quick_weak


class PlotterWeak(Plotter):
    def dataset_weak(self, dataset: WeakDataset):
        """
        Output visualization of a `WeakDataset` shear catalogue.

        Parameters
        ----------
        dataset
            The weak-lensing dataset which is visualized.
        """

        def should_plot(name):
            return plot_setting(section=["weak_dataset"], name=name)

        output_path = str(self.image_path)
        fmt = self.fmt

        if should_plot("subplot_dataset"):
            subplot_weak_dataset(
                dataset,
                output_path=output_path,
                output_format=fmt,
                title_prefix=self.title_prefix,
            )

    def fit_weak(self, fit: FitWeak, quick_update: bool = False):
        """
        Visualizes a `FitWeak` object.

        Parameters
        ----------
        fit
            The maximum log likelihood `FitWeak` of the non-linear search.
        quick_update
            If `True`, a lighter-weight quick-update subplot is output instead of the full fit subplot.
        """

        def should_plot(name):
            return plot_setting(section=["fit", "fit_weak"], name=name)

        output_path = str(self.image_path)
        fmt = self.fmt

        if quick_update:
            subplot_fit_quick_weak(
                fit,
                output_path=output_path,
                output_format=fmt,
                title_prefix=self.title_prefix,
            )
            return

        if should_plot("subplot_fit"):
            subplot_fit_weak(
                fit,
                output_path=output_path,
                output_format=fmt,
                title_prefix=self.title_prefix,
            )
