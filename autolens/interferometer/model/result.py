import autoarray as aa

from autogalaxy.analysis.adapt_images.adapt_images import (
    AdaptImages,
    galaxy_name_image_dict_via_result_from,
)
from autolens.lens.tracer import Tracer
from autolens.interferometer.fit_interferometer import FitInterferometer
from autolens.analysis.result import ResultDataset


class ResultInterferometer(ResultDataset):
    @property
    def max_log_likelihood_fit(self) -> FitInterferometer:
        """
        An instance of a `FitInterferometer` corresponding to the maximum log likelihood model inferred by the
        non-linear search.
        """
        return self.analysis.fit_from(instance=self.instance)

    @property
    def max_log_likelihood_tracer(self) -> Tracer:
        """
        An instance of a `Tracer` corresponding to the maximum log likelihood model inferred by the non-linear search.

        The `Tracer` is computed from the `max_log_likelihood_fit`, as this ensures that all linear light profiles
        are converted to normal light profiles with their `intensity` values updated.
        """
        return (
            self.max_log_likelihood_fit.model_obj_linear_light_profiles_to_light_profiles
        )

    @property
    def real_space_mask(self) -> aa.Mask2D:
        """
        The real space mask used by this model-fit.
        """
        return self.max_log_likelihood_fit.dataset.real_space_mask

    def adapt_images_from(self, use_model_images: bool = False) -> AdaptImages:
        """
        Returns the adapt-images which are used to make a pixelization's mesh and regularization adapt to the
        reconstructed galaxy's morphology.

        The adapt images use each galaxy's real-space model image in the best-fit model.
        For lensed galaxies this is the image-plane model image.
        The `use_model_images` argument is retained for compatibility.
        """

        return AdaptImages(
            galaxy_name_image_dict=galaxy_name_image_dict_via_result_from(
                result=self, use_model_images=True
            ),
        )
