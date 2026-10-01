"""
Interferometer fit class for strong gravitational lens modeling in the uv-plane.

``FitInterferometer`` extends the ``autogalaxy`` ``FitInterferometer`` base class to
work with a ``Tracer`` instead of a plain ``Galaxies`` collection.  The fit pipeline
mirrors the imaging analogue but operates entirely in the visibility (uv) domain:

1. Evaluate all light profiles of the tracer galaxies on the (ray-traced) real-space grid.
2. Apply the Fourier transform (DFT or NUFFT) to map the image to visibilities.
3. Subtract the predicted visibilities from the observed visibilities.
4. If the tracer contains linear light profiles or pixelizations, solve for their
   amplitudes via a linear inversion of the residual visibilities.
5. Combine direct and inversion visibilities into the ``model_data``.
6. Compute residuals, chi-squared, and log likelihood (or log evidence).

The ``TracerToInversion`` helper is used to assemble the linear system in step 4.
"""
import copy
import numpy as np
from typing import Dict, List, Optional

from autonerves import cached_property

import autoarray as aa
import autogalaxy as ag

from autogalaxy.abstract_fit import AbstractFitInversion
from autogalaxy.interferometer.fit_interferometer import (
    _has_light_profile_non_linear,
    _require_transformer,
    sparse_profile_terms_from,
    uses_precomputed_data_term_from,
)

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
        preloads=None,
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
        preloads
            An optional `PreloadsInterferometer` carrying channel-invariant inversion quantities (e.g. the
            `curvature_matrix` `F`) computed once and reused by this fit instead of being rebuilt. Used by the
            datacube shared-state path, where every spectral channel shares the lens model. `None` (the
            default) leaves the standard per-fit behaviour unchanged.
        """

        self.tracer = tracer

        self.adapt_images = adapt_images

        self.settings = settings

        self.preloads = preloads

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

    @cached_property
    def profile_image(self) -> aa.Array2D:
        """
        Returns the summed image of every ordinary (non-linear) light profile in the tracer, which is Fourier
        transformed to the `profile_visibilities`.
        """
        return self.tracer.image_2d_from(grid=self.grids.lp, xp=self._xp)

    @cached_property
    def profile_visibilities(self) -> Optional[aa.Visibilities]:
        """
        Returns the visibilities of every light profile in the tracer, which are computed by performing a Fourier
        transform to the sum of light profile images.

        If the tracer has no ordinary (non-linear) light profile (e.g. its light is entirely an MGE of linear
        Gaussians), the image is all zeros and the Fourier transform is skipped. This is decided structurally,
        so it is safe under `jax.jit`.

        On an array-free dataset (built by `Interferometer.from_stream` / `from_sparse_terms`, which has no
        `uv_wavelengths` and so no transformer) there are no visibilities to compute and this returns `None`.
        The likelihood never needs them there: ordinary light profiles enter through their real-space (lensed)
        `profile_image` and the data-term identity (`sparse_profile_terms_from`, `sparse_chi_squared`).
        """
        if self.dataset.transformer is None:
            return None

        if _has_light_profile_non_linear(galaxies=self.tracer.galaxies):
            return self.dataset.transformer.visibilities_from(
                image=self.profile_image, xp=self._xp
            )

        return aa.Visibilities.zeros(
            shape_slim=(self.dataset.transformer.uv_wavelengths.shape[0],)
        )

    @cached_property
    def profile_subtracted_visibilities(self) -> Optional[aa.Visibilities]:
        """
        Returns the interferometer dataset's visibilities with all transformed light profile images in the fit's
        tracer subtracted.

        On an array-free dataset there are no visibilities, so this is `None` (and `profile_visibilities` is not
        evaluated). The inversion of such a fit instead receives the profile-subtracted dirty image and data term
        (see `tracer_to_inversion`).
        """
        if self.data is None:
            return None

        return self.data - self.profile_visibilities

    @property
    def sparse_chi_squared(self):
        """
        The chi-squared of this fit computed from the dataset's `sparse_operator` without any visibility-sized
        array, which `chi_squared` (and so `log_likelihood` and, without an inversion, `figure_of_merit`) returns
        on an array-free dataset (see `aa.FitInterferometer.sparse_chi_squared`).

        - With an inversion it is the inversion's `fast_chi_squared`, whose data term is that of the
          profile-subtracted visibilities, so it equals `sum(|d - F i_p - F s|^2 / sigma^2)` up to the
          `s^T (eps I) s` the curvature matrix's `no_regularization_add_to_curvature_diag_value` adds for
          unregularized linear objects (the chi-squared convention `log_evidence` already uses; ~1e-7 relative
          on `log_likelihood` versus the dense residual-map chi-squared for e.g. an MGE).
        - Without one the model visibilities are only `F i_p`, the transform of the tracer's lensed ordinary
          light `profile_image`, and it is `data_term - 2 i_p^T d~ + i_p^T W~ i_p` (`sparse_profile_terms_from`);
          with no ordinary light either the model is zero and it is the operator's cached `data_term`.

        `None` when the dataset has no `sparse_operator`. Every branch is structural, so it is safe under
        `jax.jit`.
        """
        sparse_operator = self.dataset.sparse_operator

        if sparse_operator is None:
            return None

        if self.perform_inversion:
            return self.inversion.fast_chi_squared

        _, data_term = sparse_profile_terms_from(
            dataset=self.dataset,
            galaxies=self.tracer.galaxies,
            image=self.profile_image,
            xp=self._xp,
        )

        if data_term is None:
            return getattr(sparse_operator, "data_term", None)

        return data_term

    @property
    def _uses_precomputed_data_term(self) -> bool:
        """
        Whether this fit's inversion reads its data term from the scalar cached on the dataset's
        `sparse_operator` (see `uses_precomputed_data_term_from`), in which case `tracer_to_inversion` passes
        `data=None` and the likelihood never evaluates `profile_visibilities` or
        `profile_subtracted_visibilities`.
        """
        return uses_precomputed_data_term_from(
            dataset=self.dataset,
            galaxies=self.tracer.galaxies,
            data=self.data,
            noise_map=self.noise_map,
        )

    @property
    def tracer_to_inversion(self) -> TracerToInversion:
        """
        Returns the object which builds this fit's inversion from its tracer's linear objects.

        The inversion fits the `profile_subtracted_visibilities`, except when `_uses_precomputed_data_term`, where
        `data=None` is passed and the sparse inversion touches no visibility-sized array:

        - With no ordinary light profile nothing is subtracted, so the inversion takes its data vector from the
          operator's cached dirty image and the data term of its `fast_chi_squared` from the operator's cached
          scalar.
        - On an array-free dataset with ordinary light profiles (e.g. lens light), it is passed the dirty image
          `d~ - W~ i_p` and data term `data_term - 2 i_p^T d~ + i_p^T W~ i_p` of the profile-subtracted
          visibilities, both formed from one `W~ i_p` product (`sparse_profile_terms_from`), so the profile
          visibilities `F i_p` are never formed.

        On an in-memory sparse dataset with ordinary light profiles the subtracted dirty image is still supplied
        (the data vector must use it) alongside the subtracted visibilities. Where the visibilities exist they
        remain available to outputs via `fit.data`.
        """
        sparse_dirty_image, data_term = sparse_profile_terms_from(
            dataset=self.dataset,
            galaxies=self.tracer.galaxies,
            image=self.profile_image,
            xp=self._xp,
        )

        if self._uses_precomputed_data_term:
            data = None
        else:
            data = self.profile_subtracted_visibilities
            data_term = None

        dataset = aa.DatasetInterface(
            data=data,
            noise_map=self.noise_map,
            grids=self.grids,
            transformer=self.dataset.transformer,
            sparse_operator=self.dataset.sparse_operator,
            sparse_dirty_image=sparse_dirty_image,
            data_term=data_term,
        )

        return TracerToInversion(
            dataset=dataset,
            tracer=self.tracer,
            adapt_images=self.adapt_images,
            settings=self.settings,
            xp=self._xp,
            preloads=self._preloads_scoped,
        )

    @property
    def _preloads_scoped(self):
        """
        The preloads as consumed by this fit's inversion, scoped to its dataset type.

        Cross-dataset-type shared state (e.g. an imaging lead factor in a joint
        imaging + interferometer graph) is valid here only through its source-plane mesh
        geometry — a mapper / curvature matrix from another dataset type would embed that
        dataset's grids and silently corrupt the fit, so non-interferometer preloads are
        reduced to their mesh-geometry view.
        """
        if self.preloads is None or isinstance(self.preloads, aa.PreloadsInterferometer):
            return self.preloads

        return aa.PreloadsInterferometer(
            source_plane_mesh_grid=self.preloads.source_plane_mesh_grid,
            image_plane_mesh_grid=self.preloads.image_plane_mesh_grid,
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
    def inversion_with_data(self) -> Optional[aa.AbstractInversion]:
        """
        The fit's `inversion`, guaranteed to carry the visibilities it fitted as its dataset's `data`, for
        output quantities that read them (e.g. `data_subtracted_dict`, plotted by `subplot_of_mapper`).

        On the sparse path with no ordinary light profile (`_uses_precomputed_data_term`) the likelihood's
        inversion is built with `data=None`, so that it touches no visibility-sized array. Nothing was subtracted
        from the visibilities in that case, so the data it fitted are `fit.data`: this returns a shallow copy of
        the inversion (solved first, so it shares the reconstruction and every other cached quantity) whose dataset interface carries
        `fit.data`. In every other case it returns `inversion` itself.

        On an array-free dataset (built by `Interferometer.from_stream` / `from_sparse_terms`) `fit.data` is
        `None`, so there are no visibilities to carry and `inversion` itself is returned; output quantities
        that read the visibilities are unavailable on such a fit.
        """
        inversion = self.inversion

        if inversion is None or inversion.dataset.data is not None or self.data is None:
            return inversion

        # Solve first, so the copy shares the reconstruction (and everything it cached) rather than repeating it.
        inversion.reconstruction

        dataset = copy.copy(inversion.dataset)
        dataset.data = self.data

        inversion_with_data = copy.copy(inversion)
        inversion_with_data.dataset = dataset

        return inversion_with_data

    @property
    def model_data(self) -> aa.Visibilities:
        """
        Returns the model data that is used to fit the data.

        If the tracer does not have any linear objects and therefore omits an inversion, the model data is the
        sum of all light profile images Fourier transformed to visibilities.

        If a inversion is included it is the sum of these visibilities and the inversion's reconstructed visibilities.

        On an array-free dataset (built by `Interferometer.from_stream` / `from_sparse_terms`) there is no
        transformer to form model visibilities with, so this raises an `aa.exc.DatasetException`; the real-space
        `model_image_natural` and its natural dirty image `dirty_model_image_natural` describe the model there.
        """
        _require_transformer(fit=self, quantity="model_data")

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
    def model_image_natural(self) -> aa.Array2D:
        """
        The real-space (image-plane) model image `m` of the fit, on the dataset's `real_space_mask`: the lensed
        image of every ordinary (non-linear) light profile (`profile_image`) plus, when the fit has an
        inversion, the solved linear objects' reconstruction mapped to the image plane
        (`inversion.mapped_reconstructed_data`, linear light profiles and pixelized sources) -- the real-space
        image whose visibilities are `model_data`.

        It is built from these two terms rather than from `galaxy_image_dict`, whose entry for a galaxy with
        both ordinary and linear light holds only the linear reconstruction.

        It needs neither visibilities nor a transformer, so it is available on an array-free dataset (built by
        `Interferometer.from_stream` / `from_sparse_terms`), where it is the image the natural-weighted dirty
        model image `dirty_model_image_natural` is formed from.
        """
        image = np.asarray(
            getattr(self.profile_image, "array", self.profile_image), dtype=np.float64
        )

        if self.inversion is not None:
            reconstruction = self.inversion.mapped_reconstructed_data
            image = image + np.asarray(
                getattr(reconstruction, "array", reconstruction), dtype=np.float64
            )

        return aa.Array2D(
            values=image,
            mask=self.dataset.real_space_mask,
        )

    @property
    def dirty_model_image_natural(self) -> aa.Array2D:
        """
        The naturally weighted, normalised dirty image of the model visibilities, `W~ m / sum(w)`, formed from
        `model_image_natural` with the dataset's `sparse_operator` (see
        `autoarray.fit.fit_interferometer.dirty_model_image_natural_from`).

        It is the model counterpart of the dataset's `dirty_image_natural` and needs no visibilities, so it is
        how a fit on an array-free dataset is visualized. It is available on any dataset carrying a
        `sparse_operator` (array-free, or in-memory after `apply_sparse_operator()`); otherwise it raises an
        `aa.exc.DatasetException`.
        """
        return aa.fit.fit_interferometer.dirty_model_image_natural_from(
            dataset=self.dataset, image=self.model_image_natural
        )

    @property
    def dirty_residual_map_natural(self) -> aa.Array2D:
        """
        The naturally weighted dirty residual map, `dirty_image_natural - dirty_model_image_natural`, which is
        `Re(F^H W (d - F m)) / sum(w)`: the natural dirty image of the visibility residuals, computed without
        them.
        """
        return aa.Array2D(
            values=np.asarray(self.dataset.dirty_image_natural.array)
            - np.asarray(self.dirty_model_image_natural.array),
            mask=self.dataset.real_space_mask,
        )

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

        On an array-free dataset there is no transformer and this raises an `aa.exc.DatasetException`.
        """
        _require_transformer(fit=self, quantity="galaxy_model_visibilities_dict")

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

        On an array-free dataset there is no transformer and this raises an `aa.exc.DatasetException`.
        """
        _require_transformer(fit=self, quantity="model_visibilities_of_planes_list")

        galaxy_model_visibilities_dict = self.galaxy_model_visibilities_dict

        model_visibilities_of_planes_list = [
            aa.Visibilities.zeros(shape_slim=(self.dataset.data.shape_slim,))
            for i in range(self.tracer.total_planes)
        ]

        for plane_index, galaxies in enumerate(self.tracer.planes):
            for galaxy in galaxies:
                # A plane may hold a `MassField`, which is not a key of the galaxy dictionary (it has no
                # light, so `Tracer.galaxy_image_2d_dict_from` skips it); indexing it here would KeyError.
                if isinstance(galaxy, ag.MassField):
                    continue

                model_visibilities_of_planes_list[
                    plane_index
                ] += galaxy_model_visibilities_dict[galaxy]

        return model_visibilities_of_planes_list

    @property
    def tracer_linear_light_profiles_to_light_profiles(self) -> Tracer:
        """
        The `Tracer` where all linear light profiles have been converted to ordinary light profiles, where their
        `intensity` values are set to the values inferred by this fit.

        This is typically used for visualization, because linear light profiles cannot be used in `LightProfile`
        or `Galaxy` objects.
        """
        return self.model_obj_linear_light_profiles_to_light_profiles
