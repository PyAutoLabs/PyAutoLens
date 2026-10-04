"""
Position-based likelihood penalties for strong lens model fitting.

``PositionsLH`` objects add a penalty term to the log likelihood whenever the observed
image-plane positions of a lensed point source do not self-consistently trace back to the
same source-plane location in the current model.

The check is performed by tracing each observed position through the tracer's deflection
map and computing the maximum pairwise separation in the source plane
(``SourceMaxSeparation``).  If this separation exceeds the user-specified ``threshold``,
a large negative penalty is added to the log likelihood:

  penalty = -log_likelihood_penalty_factor × (max_separation − threshold)

This forces the non-linear search to explore models where the image positions are
self-consistent, greatly improving convergence speed for systems with strong position
constraints (e.g. quad-lens quasars).
"""
import numpy as np
from typing import Optional
from pathlib import Path

import autoarray as aa
import autofit as af

from autofit.tools.util import open_

import autogalaxy as ag

from autogalaxy.analysis.analysis.dataset import AnalysisDataset

from autolens.lens.tracer import Tracer
from autolens.point.max_separation import (
    SourceMaxSeparation,
)

from autolens import exc


class PositionsLH:
    def __init__(
        self,
        positions: aa.Grid2DIrregular,
        threshold: float,
        log_likelihood_penalty_factor: float = 1e8,
        plane_redshift: Optional[float] = None,
    ):
        """
        The `PositionsLH` objects add a penalty term to the likelihood of the **PyAutoLens** `log_likelihood_function`
        defined in the `Analysis` classes.

        The penalty term inspects the distance that the locations of the multiple images of the lensed source galaxy
        trace within one another in the source-plane and penalizes solutions where they trace far from one another,
        on the basis that this indicates an unphysical or inaccurate mass model. If they trace within the
        threshold the penalty term is not applied.

        For the `PositionsLH` object, if the multiple image coordinates do not trace within the source-plane
        threshold of one another a penalty to the likelihood is applied:

        `log_Likelihood_penalty_base - log_likelihood_penalty_factor * (max_source_plane_separation - threshold)`

        The penalty term reduces as the source-plane coordinates trace closer to one another, meaning that the
        initial stages of the non-linear search can sample mass models that reduce the threshold.

        For example, for one penalty term, if the multiple image coordinates are defined
        via `positions=aa.Grid2DIrregular([(1.0, 0.0), (-1.0, 0.0)]` and they do not trace within `threshold=0.3` of
        one another, the mass model will receive a large likelihood penalty.

        The default behaviour assumes a single lens plane and single source plane, meaning the input `positions`
        are the image-plane coordinates of one source galaxy at a specifc redshift.

        For multiple source planes, the `plane_redshift` can be used to pair the image-plane positions with the
        redshift of the source plane they trace too.

        Parameters
        ----------
        positions
            The arcsecond coordinates of the lensed source multiple images which are used to compute the likelihood
            penalty.
        threshold
            If the maximum separation of any two source plane coordinates is above the threshold the penalty term
            is applied.
        log_likelihood_penalty_factor
            A factor which multiplies how far source pixels do not trace within the threshold of one another, with a
            larger factor producing a larger penalty making the non-linear parameter space gradient steeper.
        plane_redshift
            The plane redshift of the lensed source multiple images, which is only required if position threshold
            for a double source plane lens system is being used where the specific plane is required.
        """

        # A position that is not finite cannot be traced: a single (inf, inf) row makes
        # the maximum source-plane separation nan, `nan > threshold` is False, and the
        # penalty is zero for every model -- silently, since `output_positions_info`
        # uses a nan-safe maximum and reports the true separation. `Result.
        # positions_likelihood_from` passes exactly that: the point solver's
        # fixed-length output, padded with (inf, inf). Drop such rows.
        values = np.asarray(positions.array if hasattr(positions, "array") else positions)
        finite = np.isfinite(values).all(axis=1)

        if not finite.all():
            positions = aa.Grid2DIrregular(values[finite])

        self.positions = positions
        self.threshold = threshold
        self.plane_redshift = plane_redshift

        if len(positions) < 2:
            raise exc.PositionsException(
                f"The positions input into the PositionsLikelihood object have {len(positions)} finite "
                f"(y,x) coordinate(s) ({int((~finite).sum())} non-finite dropped), so there is nothing "
                f"to compare them with.\n\n"
                "Please input more positions into the Positions."
            )

        self.log_likelihood_penalty_factor = log_likelihood_penalty_factor

    def output_positions_info(
        self, output_path: str, tracer: Tracer, overwrite_file: bool = True
    ):
        """
        Outputs a `positions.info` file which summarises the positions penalty term for a model fit, including:

        - The arc second coordinates of the lensed source multiple images used for the model-fit.
        - The radial distance of these coordinates from (0.0, 0.0).
        - The threshold value used by the likelihood penalty.
        - The maximum source plane separation of the maximum likelihood tracer.

        Parameters
        ----------
        output_path
        tracer

        Returns
        -------

        """

        flag = "w+" if overwrite_file else "a+"

        with open_(Path(output_path) / "positions.info", flag) as f:

            positions_fit = SourceMaxSeparation(
                data=self.positions,
                noise_map=None,
                tracer=tracer,
                plane_redshift=self.plane_redshift,
            )

            distances = positions_fit.data.distances_to_coordinate_from(
                coordinate=(0.0, 0.0)
            )

            if self.plane_redshift is None:
                f.write(f"Plane Index: -1 \n")
            else:
                f.write(f"Plane Redshift: {self.plane_redshift} \n")

            f.write(f"Positions: \n {self.positions} \n\n")
            f.write(f"Radial Distance from (0.0, 0.0): \n {distances} \n\n")
            f.write(f"Threshold = {self.threshold} \n")
            f.write(
                f"Max Source Plane Separation of Maximum Likelihood Model = {positions_fit.max_separation_of_plane_positions}"
            )
            f.write("")

    def log_likelihood_penalty_from(
        self, instance: af.ModelInstance, analysis: AnalysisDataset, xp=np
    ) -> np.array:
        """
        Returns a log-likelihood penalty used to constrain lens models where multiple image-plane
        positions do not trace to within a threshold distance of one another in the source-plane.

        This penalty is intended for use in `Analysis` classes that include the `PenaltyLH` mixin. It adds a
        heavy penalty to the likelihood when the multiple images traces far apart in the source-plane, discouraging
        models where the mapped source-plane positions  are too widely separated.

        Specifically, if the maximum separation between traced positions in the source-plane exceeds
        a defined threshold, a penalty term is applied to the log likelihood:

            penalty = log_likelihood_penalty_factor * (max_separation - threshold)

        If the separation is within the threshold, no penalty is applied.

        JAX Compatibility
        -----------------
        Because this function may be jitted or differentiated using JAX, it uses `jax.lax.cond` to apply
        conditional logic in a way that is compatible with JAX's functional and tracing model.
        Both branches (penalty and zero) are evaluated at trace time, though only one is returned
        at runtime depending on the condition.

        Parameters
        ----------
        instance
            The current model instance evaluated during the non-linear search.
        analysis
            The `Analysis` object calling this function, from which the `tracer` and `dataset` are derived.

        Returns
        -------
        penalty
            A scalar log-likelihood penalty (≥ 0) if the max separation exceeds the threshold, or 0.0 otherwise.
        """
        tracer = analysis.tracer_via_instance_from(instance=instance)

        if not tracer.has(cls=ag.mp.MassProfile) or len(tracer.planes) == 1:
            return xp.array(0.0)

        positions_fit = SourceMaxSeparation(
            data=self.positions,
            noise_map=None,
            tracer=tracer,
            plane_redshift=self.plane_redshift,
            xp=xp,
        )

        max_separation = xp.max(
            positions_fit.furthest_separations_of_plane_positions.array
        )

        penalty = self.log_likelihood_penalty_factor * (max_separation - self.threshold)

        if xp.__name__.startswith("jax"):

            import jax

            return jax.lax.cond(
                max_separation > self.threshold,
                lambda: penalty,
                lambda: xp.array(0.0),
            )

        return penalty if max_separation > self.threshold else np.array(0.0)
