import logging
import numpy as np
import os
from typing import List, Optional

from autoconf import conf
from autoconf.dictable import output_to_json

import autofit as af
import autoarray as aa
import autogalaxy as ag

from autogalaxy.analysis.analysis.dataset import AnalysisDataset as AgAnalysisDataset

from autolens.analysis.analysis.lens import AnalysisLens
from autolens.analysis.result import ResultDataset
from autolens.analysis.positions import PositionsLH

from autolens import exc

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


class AnalysisDataset(AgAnalysisDataset, AnalysisLens):
    def __init__(
        self,
        dataset,
        positions_likelihood_list: Optional[List[PositionsLH]] = None,
        adapt_images: Optional[ag.AdaptImages] = None,
        cosmology: ag.cosmo.LensingCosmology = None,
        settings: aa.Settings = None,
        raise_inversion_positions_likelihood_exception: bool = True,
        title_prefix: str = None,
        use_jax: bool = True,
        **kwargs,
    ):
        """
        Fits a lens model to a dataset via a non-linear search.

        This abstract Analysis class has attributes and methods for all model-fits which fit the model to a dataset
        (e.g. imaging or interferometer data).

        This class stores the Cosmology used for the analysis and settings that control aspects of the calculation,
        including how pixelizations, inversions and lensing calculations are performed.

        Parameters
        ----------
        dataset
            The imaging, interferometer or other dataset that the model if fitted too.
        positions_likelihood_list
            Alters the likelihood function to include a term which accounts for whether image-pixel coordinates in
            arc-seconds corresponding to the multiple images of each lensed source galaxy trace close to one another in
            their source-plane. This is a list, as it may support multiple planes, where a positions likelihood object
            is input for each plane (e.g. double source plane lensing).
        adapt_images
            Contains the adapt-images which are used to make a pixelization's mesh and regularization adapt to the
            reconstructed galaxy's morphology.
        cosmology
            The Cosmology assumed for this analysis.
        settings
            Settings controlling how an inversion is fitted during the model-fit, for example which linear algebra
            formalism is used.
        raise_inversion_positions_likelihood_exception
            If an inversion is used without the `positions_likelihood_list` it is likely a systematic solution will
            be inferred, in which case an Exception is raised before the model-fit begins to inform the user
            of this. This exception is not raised if this input is False, allowing the user to perform the model-fit
            anyway.
        """

        super().__init__(
            dataset=dataset,
            adapt_images=adapt_images,
            cosmology=cosmology,
            settings=settings,
            title_prefix=title_prefix,
            use_jax=use_jax,
            **kwargs,
        )

        AnalysisLens.__init__(
            self=self,
            positions_likelihood_list=positions_likelihood_list,
            cosmology=cosmology,
            use_jax=use_jax,
        )

        self.raise_inversion_positions_likelihood_exception = (
            raise_inversion_positions_likelihood_exception
        )

        if os.environ.get("PYAUTOFIT_TEST_MODE") == "1":
            self.raise_inversion_positions_likelihood_exception = False

        # Can be deleted after relevent AutoFIT PR merged
        self._use_jax = use_jax

    def modify_before_fit(self, paths: af.DirectoryPaths, model: af.Collection):
        """
        This function is called immediately before the non-linear search begins and performs final tasks and checks
        before it begins.

        This function:

        - Checks that the adapt-dataset is consistent with previous adapt-datasets if the model-fit is being
          resumed from a previous run.

        - Checks the model and raises exceptions if certain critieria are not met.

        Once inherited from it also visualizes objects which do not change throughout the model fit like the dataset.

        Parameters
        ----------
        paths
            The paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.
        model
            The model object, which includes model components representing the galaxies that are fitted to
            the imaging data.
        """

        super().modify_before_fit(paths=paths, model=model)

        self.raise_exceptions(model=model)

        return self

    def raise_exceptions(self, model):
        has_pix = model.has_model(cls=(aa.Pixelization,)) or model.has_instance(
            cls=(aa.Pixelization,)
        )

        if has_pix:
            if (
                self.positions_likelihood_list is None
                and self.raise_inversion_positions_likelihood_exception
                and not conf.instance["general"]["test"][
                    "disable_positions_lh_inversion_check"
                ]
            ):
                raise exc.AnalysisException(
                    """
                    You have begun a model-fit which reconstructs the source using a pixelization.
                    However, you have not input a `positions_likelihood_list` object.
                    It is likely your model-fit will infer an inaccurate solution.
                    
                    Please read the following readthedocs page for a description of why this is, and how to set up
                    a positions likelihood object:
                    
                    https://pyautolens.readthedocs.io/en/latest/general/demagnified_solutions.html
                    """
                )

    def save_results(self, paths: af.DirectoryPaths, result: ResultDataset):
        """
        At the end of a model-fit, this routine saves attributes of the `Analysis` object to the `files`
        folder such that they can be loaded after the analysis using PyAutoFit's database and aggregator tools.

        For this analysis it outputs the following:

        - The maximum log likelihood tracer of the fit.

        Parameters
        ----------
        paths
            The paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.
        result
            The result of a model fit, including the non-linear search, samples and maximum likelihood tracer.
        """
        try:
            output_to_json(
                obj=result.max_log_likelihood_tracer,
                file_path=paths._files_path / "tracer.json",
            )
        except AttributeError:
            pass
