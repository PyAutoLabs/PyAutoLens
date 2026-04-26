"""
Analysis class for fitting a ``Tracer`` lens model to an imaging dataset.

``AnalysisImaging`` implements the ``log_likelihood_function`` that a ``PyAutoFit``
non-linear search calls on each iteration.  It:

1. Constructs a ``Tracer`` from the current model instance.
2. Optionally applies adaptive galaxy images to linear components.
3. Calls ``FitImaging`` to evaluate the log likelihood.
4. Returns the figure of merit (log likelihood or log evidence).

It also manages result output (``ResultImaging``), on-the-fly visualisation
(``VisualizerImaging``), and position-based priors via ``PositionLikelihood``.
"""
import logging

import autofit as af
import autogalaxy as ag

from autolens.analysis.analysis.dataset import AnalysisDataset
from autolens.imaging.model.result import ResultImaging
from autolens.imaging.model.visualizer import VisualizerImaging
from autolens.imaging.fit_imaging import FitImaging

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


class AnalysisImaging(AnalysisDataset):

    Result = ResultImaging
    Visualizer = VisualizerImaging

    def log_likelihood_function(self, instance: af.ModelInstance) -> float:
        """
        Given an instance of the model, where the model parameters are set via a non-linear search, fit the model
        instance to the imaging dataset.

        This function returns a log likelihood which is used by the non-linear search to guide the model-fit.

        For this analysis class, this function performs the following steps:

        1) If the analysis has a adapt image, associated the model galaxy images of this dataset to the galaxies in
           the model instance.

        2) Extract attributes which model aspects of the data reductions, like the scaling the background sky
           and background noise.

        3) Extracts all galaxies from the model instance and set up a `Tracer`, which includes ordering the galaxies
           by redshift to set up each `Plane`.

        4) Use the `Tracer` and other attributes to create a `FitImaging` object, which performs steps such as creating
           model images of every galaxy in the tracer, blurring them with the imaging dataset's PSF and computing
           residuals, a chi-squared statistic and the log likelihood.

        Certain models will fail to fit the dataset and raise an exception. For example if an `Inversion` is used, the
        linear algebra calculation may be invalid and raise an Exception. In such circumstances the model is discarded
        and its likelihood value is passed to the non-linear search in a way that it ignores it (for example, using a
        value of -1.0e99).

        Parameters
        ----------
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).

        Returns
        -------
        float
            The log likelihood indicating how well this model instance fitted the imaging data.
        """

        log_likelihood_penalty = self.log_likelihood_penalty_from(
            instance=instance,
        )

        if self._use_jax:
            return self.fit_from(instance=instance).figure_of_merit - log_likelihood_penalty

        try:
            return self.fit_from(instance=instance).log_likelihood - log_likelihood_penalty
        except Exception as e:
            raise af.exc.FitException

    def fit_from(
        self,
        instance: af.ModelInstance,
    ) -> FitImaging:
        """
        Given a model instance create a `FitImaging` object.

        This function is used in the `log_likelihood_function` to fit the model to the imaging data and compute the
        log likelihood.

        Parameters
        ----------
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).
        check_positions
            Whether the multiple image positions of the lensed source should be checked, i.e. whether they trace
            within the position threshold of one another in the source plane.

        Returns
        -------
        FitImaging
            The fit of the plane to the imaging dataset, which includes the log likelihood.
        """

        if self._use_jax:
            self._register_fit_imaging_pytrees()

        tracer = self.tracer_via_instance_from(
            instance=instance,
        )

        dataset_model = self.dataset_model_via_instance_from(instance=instance)

        adapt_images = self.adapt_images_via_instance_from(
            instance=instance, galaxies=tracer.galaxies
        )

        return FitImaging(
            dataset=self.dataset,
            tracer=tracer,
            dataset_model=dataset_model,
            adapt_images=adapt_images,
            settings=self.settings,
            xp=self._xp
        )

    @staticmethod
    def _register_fit_imaging_pytrees() -> None:
        """Register every type reachable from a ``FitImaging`` return value
        so ``jax.jit(fit_from)`` can flatten its output.

        ``dataset``, ``adapt_images`` and ``settings`` are constants per
        analysis — ride as aux so JAX does not recurse into them. Everything
        else (``tracer``, ``dataset_model`` and the autoarray wrappers they
        carry) is dynamic per fit.
        """
        from autoarray.abstract_ndarray import register_instance_pytree
        from autoarray.dataset.dataset_model import DatasetModel
        from autolens.lens.tracer import Tracer

        register_instance_pytree(
            FitImaging,
            no_flatten=("dataset", "adapt_images", "settings"),
        )
        register_instance_pytree(DatasetModel)
        # ``cosmology`` is a fixed physical constant per fit; ride as aux.
        register_instance_pytree(Tracer, no_flatten=("cosmology",))

