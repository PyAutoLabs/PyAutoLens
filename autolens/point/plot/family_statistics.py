"""Per-image position-fit contributions for family plots."""

import numpy as np
from scipy.special import logsumexp

from autolens.point.fit.positions.image.pair_repeat import FitPositionsImagePairRepeat
from autolens.point.fit.positions.source.separations import FitPositionsSource


def contribution_values(record) -> tuple[np.ndarray, float, str]:
    """Return observed-image terms, the extra predicted-image penalty and a label.

    Mixture terms use the fit's Gaussian density and equal component weights.
    If the solver predicts no images, they use the fit's squared residual floor.
    Source-plane terms are the fit's quadratic forms, before any likelihood
    normalization or marginalization.
    """
    positions_fit = record.fit.positions

    if record.mixture:
        noise = np.asarray(positions_fit.noise_map)
        model = np.asarray(record.model)
        if len(model) == 0:
            terms = (positions_fit.no_image_residual / noise) ** 2
        else:
            observed = np.asarray(positions_fit.data)
            log_components = positions_fit.log_p(
                observed.T[:, :, None], model.T[:, None, :], noise[:, None]
            )
            terms = -2.0 * (logsumexp(log_components, axis=1) - np.log(len(model)))
        return np.asarray(terms), 0.0, "-2 log mixture"

    terms = np.asarray(positions_fit.chi_squared_map.array)
    if isinstance(positions_fit, FitPositionsSource):
        return terms, 0.0, "Source-plane χ²"

    penalty = 0.0
    if (
        isinstance(positions_fit, FitPositionsImagePairRepeat)
        and positions_fit.unmatched_model_policy != "ignore"
        and len(record.model) > 0
    ):
        noise_mean = np.mean(np.asarray(positions_fit.noise_map))
        distances = np.asarray(positions_fit.unmatched_model_penalty_map)
        penalty = float(np.sum((distances / noise_mean) ** 2))

    return terms, penalty, "Image-plane χ²"
