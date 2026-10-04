import numpy as np

from autolens.point.fit.positions.image.abstract import AbstractFitPositionsImagePair
from autolens.point.fit.solved import SolvedCentre


class FitPositionsImagePairAll(AbstractFitPositionsImagePair):
    """
    Fits the positions of a a point source dataset using a `Tracer` object with an image-plane chi-squared where every
    model position of the point-source is paired with all other observed positions using the probability of each
    model posiition explaining each observed position.

    Pairing all model positions with all observed positions is a less intuitive and commonly used approach
    than other methods, for example pairing each position one-to-one. The scheme was proposed in the paper
    below and provides a number of benefits, for example being a fully Bayesian approach to the problem and
    linearizing aspects of the problem.

    https://arxiv.org/abs/2406.15280

    THIS IMPLEMENTATION DOES NOT CURRRENTLY BREAK DOWN THE CALCULATION INTO A RESIDUAL MAP AND CHI-SQUARED,
    GOING STRAIGHT TO A `log_likelihood`. FUTURE WORK WILL WORK OUT HOW TO EXPRESS THIS IN TERMS OF A CHI-SQUARED
    AND RESIDUAL MAP.

    The fit performs the following steps:

    1) Determine the source-plane centre of the point source, which is either a free model parameter read from
       the profile's `centre` (`ag.ps.Point` / `ag.ps.PointFlux`) or, for `FitPositionsImagePairAllSolved`,
       solved for analytically given the current tracer (see `autolens.point.fit.solved.SolvedCentre`), using
       name pairing (see below).

    2) Determine the image-plane model positions using the `PointSolver` and the source-plane centre of the point
       source (e.g. ray tracing triangles to and from  the image and source planes), including accounting for
       multi-plane ray-tracing.

    3) Pair every model position with every observed position and return the overall log likelihood of the fit.

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

    # The floor every observed position contributes when the solver returns no images, matching
    # `FitPositionsImagePair` and `FitPositionsImagePairRepeat`: loudly bad, but finite, so the
    # model is scored rather than silently resampled.
    no_image_residual = 1.0e4

    def log_p(
        self,
        data_position: np.ndarray,
        model_position: np.ndarray,
        sigma: float,
    ) -> float:
        """
        Compute the log probability of a given model coordinate explaining a given observed coordinate.

        Accounts for noise, with noiser image coordinates having a comparatively lower log probability.

        Parameters
        ----------
        data_position
            The observed coordinate.
        model_position
            The model coordinate.
        sigma
            The noise associated with the observed coordinate.

        Returns
        -------
            The log probability of the model coordinate explaining the observed coordinate.
        """
        chi2 = self.square_distance(data_position, model_position) / sigma**2
        return -self._xp.log(self._xp.sqrt(2 * self._xp.pi * sigma**2)) - 0.5 * chi2

    def all_permutations_log_likelihoods(self) -> np.ndarray:
        """
        Compute the log likelihood for each permutation whereby the model could explain the observed image coordinates.

        For example, if there are two observed image coordinates and two model image coordinates, the log likelihood
        for each permutation is:

        P(data_0 | model_0) * P(data_1 | model_1)
        P(data_0 | model_1) * P(data_1 | model_0)
        P(data_0 | model_0) * P(data_1 | model_0)
        P(data_0 | model_1) * P(data_1 | model_1)

        This is every way in which the coordinates generated by the model can explain the observed coordinates.

        The reduction over model positions is a max-shifted log-sum-exp rather than a literal
        `log(sum(exp(...)))`: exponentiating first underflows to 0 once the best model/observed pairing is
        ~38 sigma or worse, turning the log likelihood into `-inf` and killing gradient flow across the
        exact region gradient searches must traverse to find the basin. The shifted form is mathematically
        identical wherever the literal form is finite, and stays finite (the max term contributes exactly 0
        after the shift) at arbitrarily large mismatch.

        Padded model positions -- the solver's `inf` sentinel rows, kept because `model_data` solves with
        `remove_infinities=False` -- contribute `log_p = -inf`, i.e. nothing to the sum. Their value is right,
        but their forward-mode tangent is not: `square_distance` gives `(d - inf) ** 2` a tangent of
        `2 * (d - inf) * 0 = NaN`, which `jax.jacfwd` -- the default `gradient_mode="forward"` of
        `AnalysisPoint` -- carried through `exp` into every gradient component (PyAutoLens#767). Under JAX the
        likelihoods are therefore wrapped in a `jax.custom_jvp`: the primal is the unchanged per-position
        computation below (so jitted log likelihoods stay bit-identical), and the tangent is the JVP of
        `_log_likelihoods_padding_safe`, which scores padded rows on a finite placeholder and masks them out.
        """

        model_data = self.model_data.array

        if self._xp is np:
            return self._log_likelihoods_from(model_data)

        import jax

        @jax.custom_jvp
        def log_likelihoods(model_data):
            return self._log_likelihoods_from(model_data)

        @log_likelihoods.defjvp
        def log_likelihoods_jvp(primals, tangents):
            (model_data,), (model_data_dot,) = primals, tangents
            _, log_likelihoods_dot = jax.jvp(
                self._log_likelihoods_padding_safe, (model_data,), (model_data_dot,)
            )
            return self._log_likelihoods_from(model_data), log_likelihoods_dot

        return log_likelihoods(model_data)

    def _log_sum_exp(self, log_ps):
        """
        Max-shifted `log(sum(exp(log_ps)))` over the model positions (see `all_permutations_log_likelihoods`).
        """
        # `initial` covers the zero-model-positions case (an empty `log_ps`), where a bare `max`
        # raises: the -inf sentinel is clamped to 0 below, giving `log(sum of nothing) = -inf`,
        # exactly the literal form's result, which `chi_squared`'s `has_image` fallback replaces.
        max_log_p = self._xp.max(log_ps, initial=-np.inf)
        # With no finite model position every log_p is -inf, and shifting by a -inf max would
        # produce NaN (`-inf - -inf`) inside `exp` — including under `jax.grad`, where a NaN in
        # the branch `chi_squared`'s `xp.where` discards still poisons the gradient. Clamp the
        # shift to 0 so this case reduces to the literal form's `log(0) = -inf`, which the
        # `has_image` fallback in `chi_squared` then replaces.
        max_log_p = self._xp.where(self._xp.isfinite(max_log_p), max_log_p, 0.0)
        return max_log_p + self._xp.log(
            self._xp.sum(self._xp.exp(log_ps - max_log_p))
        )

    def _log_likelihoods_from(self, model_data) -> np.ndarray:
        """
        The per-observed-position log likelihoods for the given model positions (`inf` padding rows included):
        the primal of `all_permutations_log_likelihoods`.
        """
        return self._xp.array(
            [
                self._log_sum_exp(
                    self._xp.array(
                        [
                            self.log_p(
                                data_position,
                                model_position,
                                sigma,
                            )
                            for model_position in model_data
                        ]
                    )
                )
                for data_position, sigma in zip(self.data, self.noise_map)
            ]
        )

    def _log_likelihoods_padding_safe(self, model_data) -> np.ndarray:
        """
        `_log_likelihoods_from` with a forward-mode tangent that stays finite on padded model positions, used
        only for the tangent of `all_permutations_log_likelihoods` under JAX.

        Padded (`inf`) rows are scored on a finite placeholder and masked back to `log_p = -inf` (the
        double-`where` pattern), so no `inf` reaches `square_distance`. Each observed position is scored
        against all model positions in one vectorized `log_p` call: with one scalar `where` per model
        position instead, XLA's CPU fusion of the jitted forward-mode graph still leaked a non-finite tangent
        out of the discarded branch.
        """
        is_valid = self._xp.isfinite(model_data).all(axis=1)
        # Transposed so `square_distance`'s `coord2[0]` / `coord2[1]` index all x / all y coordinates.
        safe_model_data = self._xp.where(is_valid[:, None], model_data, 0.0).T

        return self._xp.array(
            [
                self._log_sum_exp(
                    self._xp.where(
                        is_valid,
                        self.log_p(data_position, safe_model_data, sigma),
                        -np.inf,
                    )
                )
                for data_position, sigma in zip(self.data, self.noise_map)
            ]
        )

    @property
    def chi_squared(self) -> float:
        """
        Compute the log likelihood of the model image coordinates explaining the observed image coordinates.

        This is the sum across all permutations of the observed image coordinates of the log probability of each
        model image coordinate explaining the observed image coordinate.

        For example, if there are two observed image coordinates and two model image coordinates, the log likelihood
        is the sum of the log probabilities:

        P(data_0 | model_0) * P(data_1 | model_1)
        + P(data_0 | model_1) * P(data_1 | model_0)
        + P(data_0 | model_0) * P(data_1 | model_0)
        + P(data_0 | model_1) * P(data_1 | model_1)

        This is every way in which the coordinates generated by the model can explain the observed coordinates.
        """
        n_non_nan_model_positions = self._xp.count_nonzero(
            self._xp.isfinite(
                self.model_data.array,
            ).any(axis=1)
        )

        # With no finite model positions `n_permutations` is 0, so `-log(0)` is `+inf` while the
        # permutation sum is `-inf`, and the two combine to NaN. `fitness.py` converts a NaN
        # log-likelihood into `resample_figure_of_merit`, so the model would be silently rejected
        # rather than scored -- the exact outcome `no_image_residual` exists to avoid.
        #
        # `FitPositionsImagePair` and `FitPositionsImagePairRepeat` both fall back to the
        # `no_image_residual` floor for every observed position; do the same here, on the same
        # (residual / noise) ** 2 scale their chi-squared ends up on.
        #
        # `n_non_nan_model_positions` is a traced value under `jax.jit`, so this selects with
        # `xp.where` rather than a Python `if`. `where` evaluates both branches, so the log is
        # taken on a clamped count to keep the NaN from forming in the discarded branch.
        noise_map = self._xp.asarray(np.asarray(self.noise_map))

        no_image_chi_squared = self._xp.sum(
            (self.no_image_residual / noise_map) ** 2.0
        )

        has_image = n_non_nan_model_positions > 0

        n_permutations = (
            self._xp.where(has_image, n_non_nan_model_positions, 1)
        ) ** len(self.data)

        chi_squared = -2.0 * (
            -self._xp.log(n_permutations)
            + self._xp.sum(self.all_permutations_log_likelihoods())
        )

        return self._xp.where(has_image, chi_squared, no_image_chi_squared)


class FitPositionsImagePairAllSolved(SolvedCentre, FitPositionsImagePairAll):
    """
    ``FitPositionsImagePairAll`` with the source-plane centre fed into the `PointSolver` forward solve
    (`model_data`, inherited unchanged from `AbstractFitPositionsImagePair`) solved for analytically
    (`SolvedCentre.source_plane_coordinate`, `β*`) rather than read from a free `centre` model parameter.

    This is **not** a result from Lombardi 2024 (arXiv:2406.15280) — the paper never substitutes a solved
    source-plane centre into an image-plane likelihood. It is an extension in the spirit of glafic's
    source-position-optimized image-plane chi-squared: the all-to-all pairing chi-squared itself
    (`chi_squared`, `all_permutations_log_likelihoods`) is completely unchanged from `FitPositionsImagePairAll`.

    Must be paired (by name) with a parameter-free profile such as `ag.ps.PointSolved`: a `centre`-bearing
    profile (`ag.ps.Point` / `ag.ps.PointFlux`) raises (see `SolvedCentre.source_plane_coordinate`), since its
    centre priors would otherwise be sampled but silently ignored.
    """

    _non_solved_alternative_name = "FitPositionsImagePairAll"
