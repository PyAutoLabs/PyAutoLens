"""
Abstract base for image-plane position fitting strategies.

``AbstractFitPositionsImagePair`` solves for the predicted image positions using a
``PointSolver`` and then pairs each observed position with the closest predicted image,
computing a chi-squared from their separation.

Three concrete pairing strategies are provided in this sub-package:

- ``FitPositionsImagePair`` — pairs each observed to its nearest predicted image via
  the Hungarian (linear sum assignment) algorithm.
- ``FitPositionsImagePairAll`` — computes the chi-squared for every observed/predicted
  pair combination and takes the minimum.
- ``FitPositionsImagePairRepeat`` — allows predicted images to be paired to more than
  one observed position, useful for highly magnified systems where some images may be
  too close to separate.
"""
from abc import ABC
import numpy as np
from typing import Optional

import autoarray as aa
import autogalaxy as ag

from autolens.point.solver import PointSolver
from autolens.point.fit.positions.abstract import AbstractFitPositions
from autolens.lens.tracer import Tracer


class AbstractFitPositionsImagePair(AbstractFitPositions, ABC):
    def __init__(
        self,
        name: str,
        data: aa.Grid2DIrregular,
        noise_map: aa.ArrayIrregular,
        tracer: Tracer,
        solver: PointSolver,
        profile: Optional[ag.ps.Point] = None,
        xp=np,
    ):
        """
        Abstract class to fit the positions of a point source dataset using a `Tracer` object with an image-plane
        chi-squared, where the specific implementation of the image-plane chi-squared is defined in the sub-class.

        The fit performs the following steps:

        1) Determine the source-plane centre of the point source, which could be a free model parameter or computed
           as the barycenter of ray-traced positions in the source-plane, using name pairing (see below).

        2) Determine the image-plane model positions using the `PointSolver` and the source-plane centre of the point
           source (e.g. ray tracing triangles to and from  the image and source planes), including accounting for
           multi-plane ray-tracing.

        3) Using the sub-class specific chi-squared, compute the residuals of each image-plane position, chi-squared
           and overall log likelihood of the fit.

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

        super().__init__(
            name=name,
            data=data,
            noise_map=noise_map,
            tracer=tracer,
            solver=solver,
            profile=profile,
            xp=xp,
        )

    @staticmethod
    def square_distance(
        coord1: np.array,
        coord2: np.array,
    ) -> float:
        """
        Calculate the square distance between two points.

        Parameters
        ----------
        coord1
            The first point to calculate the distance between.
        coord2
            The second point to calculate the distance between.

        Returns
        -------
        The square distance between the two points
        """
        return (coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2

    @property
    def model_data(self) -> aa.Grid2DIrregular:
        """
        Returns the model positions, which are computed via the point solver.
        """
        return self.solver.solve(
            tracer=self.tracer,
            source_plane_coordinate=self.source_plane_coordinate,
            xp=self._xp,
            plane_redshift=self.plane_redshift,
            remove_infinities=False,
        )
