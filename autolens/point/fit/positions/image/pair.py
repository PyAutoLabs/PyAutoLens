import numpy as np
from scipy.optimize import linear_sum_assignment

import autoarray as aa

from autolens.point.fit.positions.image.abstract import AbstractFitPositionsImagePair


class FitPositionsImagePair(AbstractFitPositionsImagePair):
    """
    Fits the positions of a point source dataset using a `Tracer` object with an image-plane chi-squared where every
    model position of the point-source is paired with its closest observed position, without allowing for repeated
    pairings of the same observed position to model positions.

    By not allowing for repeated pairings, this can produce behaviour such as a model position not being paired to
    its closest observed position, but instead being paired to a further observed position, if doing so
    means that the overall distances of pairings are reduced.

    **Under-prediction penalty**: the Hungarian assignment pairs ``min(n_observed, n_model)`` positions, which
    historically meant that a model predicting *fewer* images than observed silently dropped the unmatched
    observed positions from the chi-squared — rewarding under-prediction (samplers drove n_model below
    n_observed to shrink the chi-squared). Unmatched observed positions now contribute their distance to the
    nearest model position as a residual, and if the model predicts no images at all every observed position
    contributes the ``no_image_residual`` floor. ``FitPositionsImagePairRepeat`` remains the model-fit default;
    it additionally offers over-prediction policies.

    The fit performs the following steps:

    1) Determine the source-plane centre of the point source, which could be a free model parameter or computed
       as the barycenter of ray-traced positions in the source-plane, using name pairing (see below).

    2) Determine the image-plane model positions using the `PointSolver` and the source-plane centre of the point
       source (e.g. ray tracing triangles to and from  the image and source planes), including accounting for
       multi-plane ray-tracing.

    3) Pair each model position with the observed position, not allowing for repeated pairings of the same
       observed position to model positions, to compute the `residual_map`. This may result in some observed
       positions not being paired to their closest model position, if doing so reduces the overall distances of
       pairings.

    5) Compute the chi-squared of each position as the square of the residual divided by the RMS noise-map value.

    6) Sum the chi-squared values to compute the overall log likelihood of the fit.

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

    no_image_residual = 1.0e4

    @property
    def residual_map(self) -> aa.ArrayIrregular:

        model = np.asarray(self.model_data.array)
        model = model[np.isfinite(model).all(axis=1)]

        if model.shape[0] == 0:
            return aa.ArrayIrregular(
                values=[float(self.no_image_residual)] * len(self.data)
            )

        cost_matrix = np.linalg.norm(
            np.array(
                self.data,
            )[:, np.newaxis]
            - model,
            axis=2,
        )

        data_indexes, model_indexes = linear_sum_assignment(cost_matrix)

        # Residuals are ordered by observed position (matching the noise-map ordering): assigned
        # positions get their Hungarian pairing distance; positions the assignment could not pair
        # (under-prediction, n_model < n_observed) get their distance to the nearest model
        # position, so a model that cannot produce an observed image is penalized, not rewarded.
        residuals = np.min(cost_matrix, axis=1)

        for data_index, model_index in zip(data_indexes, model_indexes):
            residuals[data_index] = np.sqrt(
                self.square_distance(self.data[data_index], model[model_index])
            )

        residual_map = [float(r) for r in residuals]

        return aa.ArrayIrregular(values=residual_map)
