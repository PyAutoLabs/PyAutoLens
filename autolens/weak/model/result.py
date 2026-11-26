import autoarray as aa

from autolens.analysis.result import Result


class ResultWeak(Result):
    @property
    def grid(self):
        return aa.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.1)

    @property
    def max_log_likelihood_fit(self):
        """
        The `FitWeak` of the maximum log likelihood model, built on the NumPy path (via
        `AnalysisWeak.fit_for_visualization`) so its `model_shear` is a `ShearYX2DIrregular` that the
        shear-field plotters can draw, whatever backend the likelihood ran on.
        """
        return self.analysis.fit_for_visualization(instance=self.instance)
