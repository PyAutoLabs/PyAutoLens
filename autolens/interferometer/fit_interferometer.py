import numpy as np
from typing import Dict, List, Optional

from autoconf import cached_property

import autoarray as aa
import autogalaxy as ag

from autogalaxy.abstract_fit import AbstractFitInversion

from autolens.lens.tracer import Tracer
from autolens.lens.to_inversion import TracerToInversion


class FitInterferometer(aa.FitInterferometer, AbstractFitInversion):
    def __init__(
        self,
        dataset: aa.Interferometer,
        tracer: Tracer,
        dataset_model: Optional[aa.DatasetModel] = None,
        adapt_images: Optional[ag.AdaptImages] = None,
        settings: aa.Settings = None,
        xp=np,
    ):
        """
        Fits an interferometer dataset using a `Tracer` object.

        The fit performs the following steps:

        1) Compute the sum of all images of galaxy light profiles in the `Tracer`.

        2) Fourier transform this image with the transformer object and `uv_wavelengths` to create
           the `profile_visibilities`.

        3) Subtract these visibilities from the `data` to create the `profile_subtracted_visibilities`.

        4) If the `Tracer` has any linear algebra objects (e.g. linear light profiles, a pixelization / regulariation)
           fit the `profile_subtracted_visibilities` with these objects via an inversion.

        5) Compute the `model_data` as the sum of the `profile_visibilities` and `reconstructed_data` of the inversion
           (if an inversion is not performed the `model_data` is only the `profile_visibilities`.

        6) Subtract the `model_data` from the data and compute the residuals, chi-squared and likelihood via the
           noise-map (if an inversion is performed the `log_evidence`, including addition terms describing the linear
           algebra solution, is computed).

        When performing a model-fit` via ` AnalysisInterferometer` object the `figure_of_merit` of
        this object is called and returned in the `log_likelihood_function`.

        Parameters
        ----------
        dataset
            The interforometer dataset which is fitted by the galaxies in the tracer.
        tracer
            The tracer of galaxies whose light profile images are used to fit the interferometer data.
        dataset_model
            Attributes which allow for parts of a dataset to be treated as a model (e.g. the background sky level).
        adapt_images
            Contains the adapt-images which are used to make a pixelization's mesh and regularization adapt to the
            reconstructed galaxy's morphology.
        settings
            Settings controlling how an inversion is fitted for example which linear algebra formalism is used.
        """

        self.tracer = tracer

        self.adapt_images = adapt_images

        self.settings = settings

        super().__init__(
            dataset=dataset,
            dataset_model=dataset_model,
            xp=xp,
        )
        AbstractFitInversion.__init__(
            self=self, model_obj=tracer, settings=settings, xp=xp
        )

        self.use_jax = xp is not np

    @property
    def _xp(self):
        if self.use_jax:
            import jax.numpy as jnp

            return jnp
        return np

    @property
    def profile_visibilities(self) -> aa.Visibilities:
        """
        Returns the visibilities of every light profile in the tracer, which are computed by performing a Fourier
        transform to the sum of light profile images.
        """
        return self.tracer.visibilities_from(
            grid=self.grids.lp, transformer=self.dataset.transformer, xp=self._xp
        )

    @property
    def profile_subtracted_visibilities(self) -> aa.Visibilities:
        """
        Returns the interferometer dataset's visibilities with all transformed light profile images in the fit's
        tracer subtracted.
        """
        return self.data - self.profile_visibilities

    @property
    def tracer_to_inversion(self) -> TracerToInversion:
        dataset = aa.DatasetInterface(
            data=self.profile_subtracted_visibilities,
            noise_map=self.noise_map,
            grids=self.grids,
            transformer=self.dataset.transformer,
            sparse_operator=self.dataset.sparse_operator,
        )

        return TracerToInversion(
            dataset=dataset,
            tracer=self.tracer,
            adapt_images=self.adapt_images,
            settings=self.settings,
            xp=self._xp,
        )

    @cached_property
    def inversion(self) -> Optional[aa.AbstractInversion]:
        """
        If the tracer has linear objects which are used to fit the data (e.g. a linear light profile / pixelization)
        this function returns a linear inversion, where the flux values of these objects (e.g. the `intensity`
        of linear light profiles) are computed via linear matrix algebra.

        The data passed to this function is the dataset's image with all light profile images of the tracer subtracted,
        ensuring that the inversion only fits the data with ordinary light profiles subtracted.
        """
        if self.perform_inversion:
            return self.tracer_to_inversion.inversion

    @property
    def model_data(self) -> aa.Visibilities:
        """
        Returns the model data that is used to fit the data.

        If the tracer does not have any linear objects and therefore omits an inversion, the model data is the
        sum of all light profile images Fourier transformed to visibilities.

        If a inversion is included it is the sum of these visibilities and the inversion's reconstructed visibilities.
        """

        if self.perform_inversion:
            return (
                self.profile_visibilities
                + self.inversion.mapped_reconstructed_operated_data
            )

        return self.profile_visibilities

    @property
    def galaxy_image_dict(self) -> Dict[ag.Galaxy, np.ndarray]:
        """
        A dictionary which associates every galaxy in the tracer with its `image`.

        This image is the image of the sum of:

        - The images of all ordinary light profiles in that tracer summed.
        - The images of all linear objects (e.g. linear light profiles / pixelizations), where the images are solved
          for first via the inversion.

        For modeling, this dictionary is used to set up the `adapt_images` that adapt certain pixelizations to the
        data being fitted.
        """
        galaxy_image_dict = self.tracer.galaxy_image_2d_dict_from(
            grid=self.grids.lp, xp=self._xp
        )

        galaxy_linear_obj_image_dict = self.galaxy_linear_obj_data_dict_from(
            use_operated=False
        )

        return {**galaxy_image_dict, **galaxy_linear_obj_image_dict}

    @property
    def galaxy_signal_to_noise_map_dict(self) -> Dict[ag.Galaxy, np.ndarray]:
        """
        A dictionary which associates every galaxy in the tracer with its signal-to-noise map.

        This signal-to-noise map is the signal-to-noise map of the sum of:

        - The images of all ordinary light profiles in that tracer summed.
        - The images of all linear objects (e.g. linear light profiles / pixelizations), where the images are solved
          for first via the inversion.

        For modeling, this dictionary is used to set up the `adapt_images` that adapt certain pixelizations to the
        data being fitted.
        """
        galaxy_image_dict = self.galaxy_image_dict

        galaxy_signal_to_noise_map_dict = {}

        for galaxy, image in galaxy_image_dict.items():
            galaxy_signal_to_noise_map_dict[galaxy] = image / self.dirty_noise_map

        return galaxy_signal_to_noise_map_dict

    @property
    def galaxy_model_visibilities_dict(self) -> Dict[ag.Galaxy, np.ndarray]:
        """
        A dictionary which associates every galaxy in the tracer with its model visibilities.

        These visibilities are the sum of:

        - The visibilities of all ordinary light profiles in that tracer summed and Fourier transformed to visibilities
          space.
        - The visibilities of all linear objects (e.g. linear light profiles / pixelizations), where the visibilities
          are solved for first via the inversion.
        """
        galaxy_model_visibilities_dict = self.tracer.galaxy_visibilities_dict_from(
            grid=self.grids.lp, transformer=self.dataset.transformer, xp=self._xp
        )

        galaxy_linear_obj_visibilities_dict = self.galaxy_linear_obj_data_dict_from(
            use_operated=True
        )

        return {**galaxy_model_visibilities_dict, **galaxy_linear_obj_visibilities_dict}

    @property
    def model_visibilities_of_planes_list(self) -> List[aa.Visibilities]:
        """
        A list of every model image of every plane in the tracer.

        This image is the image of the sum of:

        - The images of all ordinary light profiles in that plane summed and convolved with the imaging data's PSF.
        - The images of all linear objects (e.g. linear light profiles / pixelizations), where the images are solved
          for first via the inversion.

        This is used to visualize the different contibutions of light from the image-plane, source-plane and other
        planes in a fit.
        """
        galaxy_model_visibilities_dict = self.galaxy_model_visibilities_dict

        model_visibilities_of_planes_list = [
            aa.Visibilities.zeros(shape_slim=(self.dataset.data.shape_slim,))
            for i in range(self.tracer.total_planes)
        ]

        for plane_index, galaxies in enumerate(self.tracer.planes):
            for galaxy in galaxies:
                model_visibilities_of_planes_list[
                    plane_index
                ] += galaxy_model_visibilities_dict[galaxy]

        return model_visibilities_of_planes_list

    @property
    def tracer_linear_light_profiles_to_light_profiles(self) -> Tracer:
        """
        The `Tracer` where all linear light profiles have been converted to ordinary light profiles, where their
        `intensity` values are set to the values inferred by this fit.

        This is typically used for visualization, because linear light profiles cannot be used in `LightProfilePlotter`
        or `GalaxyPlotter` objects.
        """
        return self.model_obj_linear_light_profiles_to_light_profiles
