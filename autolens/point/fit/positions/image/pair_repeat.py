import numpy as np

import autoarray as aa
import autogalaxy as ag

from autolens.point.fit.positions.image.abstract import AbstractFitPositionsImagePair


class FitPositionsImagePairRepeat(AbstractFitPositionsImagePair):
    """
    Fits the positions of a a point source dataset using a `Tracer` object with an image-plane chi-squared where every
    model position of the point-source is paired with its closest observed position, allowing for repeated pairings of
    the same observed position to model positions.

    The fit performs the following steps:

    1) Determine the source-plane centre of the point source, which could be a free model parameter or computed
       as the barycenter of ray-traced positions in the source-plane, using name pairing (see below).

    2) Determine the image-plane model positions using the `PointSolver` and the source-plane centre of the point
       source (e.g. ray tracing triangles to and from  the image and source planes), including accounting for
       multi-plane ray-tracing.

    3) Pair each observed position with the closest model position, allowing for repeated pairings of the same
       model position to observed positions, to compute the `residual_map`.

    4) Apply the **unmatched-model policy** (see below): model-predicted images which no observed position paired
       to — the over-prediction case, e.g. spurious images from a wrong mass model — may contribute an additional
       chi-squared penalty.

    5) Compute the chi-squared of each position as the square of the residual divided by the RMS noise-map value,
       plus any policy penalty, and sum to compute the overall log likelihood of the fit.

    **Over- and under-prediction**

    A model tracer may predict more images than observed (over-prediction) or fewer (under-prediction):

    - **Under-prediction** is always penalized: every *observed* position contributes a residual to its nearest
      surviving model image, so a model that cannot produce an observed image pays the full image-plane distance
      to the images it does produce. If the solver returns no images at all, every observed position contributes
      the ``no_image_residual`` floor (a large, finite residual — loudly bad without breaking the sampler with
      NaNs / infs).

    - **Over-prediction** is controlled by the ``unmatched_model_policy`` class attribute:

      - ``"ignore"`` — extra model images contribute nothing (the historical behaviour, now explicit).
      - ``"penalize"`` — every finite model image which is not the nearest neighbour of any observed position
        contributes its distance to the nearest observed position as an additional residual.
      - ``"magnification_filter"`` (default) — as ``"penalize"``, but model images whose absolute magnification
        is below ``magnification_threshold`` are exempt first. This is the standard observational convention
        (e.g. Lenstool practice): a strongly demagnified extra image — typically the central image — is assumed
        to be below the detection limit and does not count against the model, while a bright unobserved image
        does.

    The policy attributes are **class attributes** (the ``Analysis.LATENT_BATCH_MODE`` pattern), so a model-fit
    can switch policy without new constructor plumbing::

        class FitQuiet(al.FitPositionsImagePairRepeat):
            unmatched_model_policy = "ignore"

        al.AnalysisPoint(dataset=dataset, solver=solver, fit_positions_cls=FitQuiet)

    Penalty terms are normalized by the mean of the position noise-map, and all policy computations are
    fixed-shape (NaN-padded model positions are masked, never dropped), so the fit remains JAX-compilable.

    Point source fitting uses name pairing, whereby the `name` of the `Point` object is paired to the name of the
    point source dataset to ensure that point source datasets are fitted to the correct point source.

    This fit object is used in the `FitPointDataset` to perform position based fitting of a `PointDataset`,
    which may also fit other components of the point dataset like fluxes or time delays.

    When performing a `model-fit`via an `AnalysisPoint` object the `figure_of_merit` of this object
    is called and returned in the `log_likelihood_function`.

    Parameters
    ----------
    name
        The name of the point source dataset which is paired to a `Point` profile.
    data
        The positions of the point source in the image-plane which are fitted.
    noise_map
        The noise-map of the positions which are used to compute the log likelihood of the positions.
    tracer
        The tracer of galaxies whose point source profile are used to fit the positions.
    solver
        Solves the lens equation in order to determine the image-plane positions of a point source by ray-tracing
        triangles to and from the source-plane.
    profile
        Manually input the profile of the point source, which is used instead of the one extracted from the
        tracer via name pairing if that profile is not found.
    """

    unmatched_model_policy = "magnification_filter"
    magnification_threshold = 0.1
    no_image_residual = 1.0e4

    @property
    def _distance_matrix(self):
        """
        The (n_observed, n_model) matrix of distances between every observed and model position, with
        non-finite (NaN-padded) model positions set to the ``no_image_residual`` sentinel so they are
        never selected as a nearest neighbour.
        """
        data = self._xp.asarray(np.asarray(self.data))
        model = self.model_data.array

        distances = self._xp.sqrt(
            self._xp.sum(
                (data[:, None, :] - model[None, :, :]) ** 2,
                axis=2,
            )
        )
        finite = self._xp.isfinite(distances)
        return self._xp.where(finite, distances, self.no_image_residual)

    @property
    def residual_map(self) -> aa.ArrayIrregular:
        """
        The distance of every observed position to its nearest model position (repeats allowed).

        If the solver returns no images (or only NaN-padded rows), every observed position contributes the
        ``no_image_residual`` floor.
        """
        model = self.model_data.array

        if model.shape[0] == 0:
            return aa.ArrayIrregular(
                values=self._xp.full(len(self.data), float(self.no_image_residual))
            )

        return aa.ArrayIrregular(values=self._xp.min(self._distance_matrix, axis=1))

    @property
    def unmatched_model_mask(self):
        """
        Boolean mask over the model positions marking the "extra" images the policy penalizes: finite model
        positions which are not the nearest neighbour of any observed position and (under
        ``"magnification_filter"``) whose absolute magnification is above ``magnification_threshold``.
        """
        model = self.model_data.array

        finite = self._xp.isfinite(model).all(axis=1)

        nearest_indexes = self._xp.argmin(self._distance_matrix, axis=1)
        matched = (
            self._xp.arange(model.shape[0])[None, :] == nearest_indexes[:, None]
        ).any(axis=0)

        unmatched = self._xp.logical_and(finite, self._xp.logical_not(matched))

        if self.unmatched_model_policy == "magnification_filter":
            use_multi_plane = len(self.tracer.planes) > 2
            plane_j = (
                self.tracer.extract_plane_index_of_profile(profile_name=self.name)
                if use_multi_plane
                else -1
            )
            lens_calc = ag.LensCalc.from_tracer(
                tracer=self.tracer,
                use_multi_plane=use_multi_plane,
                plane_j=plane_j,
            )
            safe_model = self._xp.where(
                finite[:, None], model, self._xp.zeros_like(model)
            )
            magnifications = lens_calc.magnification_2d_via_hessian_from(
                grid=safe_model, xp=self._xp
            )
            magnifications = (
                magnifications.array
                if hasattr(magnifications, "array")
                else magnifications
            )
            detectable = self._xp.abs(magnifications) >= self.magnification_threshold
            unmatched = self._xp.logical_and(unmatched, detectable)

        return unmatched

    @property
    def unmatched_model_penalty_map(self):
        """
        The penalty residual of every model position: its distance to the nearest observed position where the
        ``unmatched_model_mask`` is set, zero elsewhere.
        """
        distances_to_data = self._xp.min(self._distance_matrix, axis=0)
        return self._xp.where(
            self.unmatched_model_mask,
            distances_to_data,
            self._xp.zeros_like(distances_to_data),
        )

    @property
    def n_unmatched_model_positions(self) -> int:
        """
        The number of model positions the policy counts as unexplained extras — a useful fit-quality
        diagnostic alongside the chi-squared.
        """
        return self._xp.sum(self.unmatched_model_mask)

    @property
    def chi_squared(self) -> float:
        """
        The chi-squared of the fit: the observed-position chi-squared (as in all `AbstractFit` objects) plus,
        for policies other than ``"ignore"``, the unmatched-model penalty terms normalized by the mean position
        noise.
        """
        chi_squared = self._xp.sum(self.chi_squared_map.array)

        if (
            self.unmatched_model_policy == "ignore"
            or self.model_data.array.shape[0] == 0
        ):
            return chi_squared

        noise_mean = self._xp.mean(self._xp.asarray(np.asarray(self.noise_map)))

        return chi_squared + self._xp.sum(
            (self.unmatched_model_penalty_map / noise_mean) ** 2.0
        )
