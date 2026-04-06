"""
Result classes for **PyAutoLens** model fits.

This module defines the ``Result`` and ``ResultDataset`` classes that wrap the output of
a ``PyAutoFit`` non-linear search for lensing model fits.  Key properties include:

- ``max_log_likelihood_tracer`` — reconstructs the best-fit ``Tracer`` from the stored
  model instance.
- ``source_plane_light_profile_centre_from`` — extracts the source-plane light-profile
  centre, used by automatic position updating to seed position priors for the next
  search in a SLaM pipeline.
- ``image_plane_multiple_image_positions_from`` — uses a ``PointSolver`` to compute the
  image-plane multiple-image positions predicted by the best-fit tracer, optionally
  filtered by a minimum flux threshold.

These results feed directly into downstream pipeline stages and post-processing scripts.
"""
import logging
import os
import numpy as np
from typing import List, Optional, Union

import autoarray as aa
import autogalaxy as ag

from autogalaxy.analysis.result import ResultDataset as AgResultDataset

from autolens.analysis.positions import PositionsLH
from autolens.point.max_separation import (
    SourceMaxSeparation,
)
from autolens.lens.tracer import Tracer
from autolens.point.solver import PointSolver
from autofit.non_linear.test_mode import is_test_mode

logger = logging.getLogger(__name__)


class Result(AgResultDataset):
    @property
    def max_log_likelihood_tracer(self) -> Tracer:
        """
        An instance of a `Tracer` corresponding to the maximum log likelihood model inferred by the non-linear search.
        """
        return self.analysis.tracer_via_instance_from(instance=self.instance)

    def source_plane_light_profile_centre_from(
        self, plane_redshift: Optional[float] = None
    ) -> aa.Grid2DIrregular:
        """
        Return a light profile centre of one of the a galaxies in the maximum log likelihood `Tracer`'s source-plane.
        If there are multiple light profiles, the first light profile's centre is returned.

        These centres are used by automatic position updating to determine the best-fit lens model's image-plane
        multiple-image positions.
        """
        try:
            plane_index = self.max_log_likelihood_tracer.plane_index_via_redshift_from(
                redshift=plane_redshift
            )
        except TypeError:
            plane_index = -1

        centre = self.max_log_likelihood_tracer.planes[plane_index].extract_attribute(
            cls=ag.LightProfile, attr_name="centre"
        )
        if centre is not None:
            return aa.Grid2DIrregular(values=[np.asarray(centre[0])])

    def source_plane_centre_from(
        self, plane_redshift: Optional[float] = None
    ) -> aa.Grid2DIrregular:
        """
        Return the centre of a source-plane galaxy via the following criteria:

        1) If the source plane contains only light profiles, return the first light's centre.
        2) If it contains an `LEq` return the centre of its brightest pixel instead.

        These centres are used by automatic position updating to determine the multiple-images of a best-fit lens model
        (and thus tracer) by back-tracing the centres to the image plane via the mass model.
        """
        return self.source_plane_light_profile_centre_from(
            plane_redshift=plane_redshift
        )

    def image_plane_multiple_image_positions(
        self, plane_redshift: Optional[float] = None
    ) -> aa.Grid2DIrregular:
        """
        Backwards ray-trace the source-plane centres (see above) to the image-plane via the mass model, to determine
        the multiple image position of the source(s) in the image-plane.

        These image-plane positions are used by the next search in a pipeline if automatic position updating is turned
        on.

        Parameters
        ----------
        plane_redshift
            The plane redshift of the lensed source, which is only required when a double (or triple) source plane lens
            system is being analysed where the specific plane via its redshift is required to define whihch source
            galaxy is used to compute the multiple images.
        """

        grid = self.analysis.dataset.mask.derive_grid.all_false

        solver = PointSolver.for_grid(
            grid=grid, pixel_scale_precision=0.001, xp=self.analysis._xp
        )

        source_plane_centre = self.source_plane_centre_from(
            plane_redshift=plane_redshift
        )

        multiple_images = solver.solve(
            tracer=self.max_log_likelihood_tracer,
            source_plane_coordinate=source_plane_centre.in_list[0],
            plane_redshift=plane_redshift,
        )

        if multiple_images.shape[0] == 1:
            return self.image_plane_multiple_image_positions_for_single_image_from()

        return aa.Grid2DIrregular(values=multiple_images)

    def image_plane_multiple_image_positions_for_single_image_from(
        self, plane_redshift: Optional[float] = None, increments: int = 20
    ) -> aa.Grid2DIrregular:
        """
        If the standard point solver only locates one multiple image, finds one or more additional images, which are
        not technically multiple image in the point source regime, but are close enough to it they can be used
        in a position threshold likelihood.

        This is performed by incrementally moving the source-plane centre's coordinates towards the centre of the
        source-plane at (0.0", 0.0"). This ensures that the centre will eventually go inside the caustic, where
        multiple images are formed.

        To move the source-plane centre, the original source-plane centre is multiplied by a factor that decreases
        from 1.0 to 0.0 in increments of 1/increments. For example, if the source-plane centre is (1.0", -0.5") and
        the `factor` is 0.5, the input source-plane centre is (0.5", -0.25").

        The multiple images are always computed for the same mass model, thus they will always be valid multiple images
        for the model being fitted, but as the factor decrease the multiple images may move furhter from their observed
        positions.

        Parameters
        ----------
        plane_redshift
            The plane redshift of the lensed source, which is only required when a double (or triple) source plane lens
            system is being analysed where the specific plane via its redshift is required to define whihch source
            galaxy is used to compute the multiple images.
        increments
            The number of increments the source-plane centre is moved to compute multiple images.
        """

        logger.info(
            """
        Could not find multiple images for maximum likelihood lens model.
        
        Incrementally moving source centre inwards towards centre of source-plane until caustic crossing occurs
        and multiple images are formed.        
        """
        )

        grid = self.analysis.dataset.mask.derive_grid.all_false

        centre = self.source_plane_centre_from(plane_redshift=plane_redshift).in_list[0]

        solver = PointSolver.for_grid(
            grid=grid, pixel_scale_precision=0.001, xp=self.analysis._xp
        )

        for i in range(1, increments):
            factor = 1.0 - (1.0 * (i / increments))

            multiple_images = solver.solve(
                tracer=self.max_log_likelihood_tracer,
                source_plane_coordinate=(centre[0] * factor, centre[1] * factor),
                plane_redshift=plane_redshift,
            )

            if multiple_images.shape[0] > 1:
                return aa.Grid2DIrregular(values=multiple_images)

        logger.info(
            """
        Could not find multiple images for maximum likelihood lens model, even after incrementally moving the source
        centre inwards to the centre of the source-plane.

        Set the multiple image postiions to two images at (1.0", 1.0") so code continues to run.
        """
        )
        return aa.Grid2DIrregular(values=[(1.0, 1.0), (1.0, 1.0)])

    def positions_threshold_from(
        self,
        factor=1.0,
        minimum_threshold=None,
        positions: Optional[aa.Grid2DIrregular] = None,
        plane_redshift: Optional[float] = None,
    ) -> float:
        """
        Compute a new position threshold from these results corresponding to the image-plane multiple image positions
        of the maximum log likelihood `Tracer` ray-traced to the source-plane.

        First, we ray-trace forward the multiple-image's to the source-plane via the mass model to determine how far
        apart they are separated. We take the maximum source-plane separation of these points and multiple this by
        the auto_positions_factor to determine a new positions threshold. This value may also be rounded up to the
        input `auto_positions_minimum_threshold`.

        This is used for non-linear search chaining, specifically updating the position threshold of a new model-fit
        using the maximum likelihood model of a previous search.

        The above behaviour assumes a single lens and single source plane, if there are multiple source planes (e.g.
        a double source plane lens) the input `plane_redshift_list` can be used to specify which source planea
        the position threshold is computed for.

        Parameters
        ----------
        factor
            The value the computed threshold is multiplied by to make the position threshold larger or smaller than the
            maximum log likelihood model's threshold.
        minimum_threshold
            The output threshold is rounded up to this value if it is below it, to avoid extremely small threshold
            values.
        positions
            If input, these positions are used instead of the computed multiple image positions from the lens mass
            model.
        plane_redshift
            The plane redshift of the lensed source, which is only required when a double (or triple) source plane lens
            system is being analysed where the specific plane via its redshift is required to define whihch source
            galaxy is used to compute the multiple images.

        Returns
        -------
        float
            The maximum source plane separation of this results maximum likelihood `Tracer` multiple images multiplied
            by `factor` and rounded up to the `threshold`.
        """

        tracer = Tracer(galaxies=self.max_log_likelihood_galaxies)

        positions = (
            self.image_plane_multiple_image_positions(plane_redshift=plane_redshift)
            if positions is None
            else positions
        )

        positions_fits = SourceMaxSeparation(
            data=positions, noise_map=None, tracer=tracer, plane_redshift=plane_redshift
        )

        threshold = factor * np.nanmax(positions_fits.max_separation_of_plane_positions)

        if minimum_threshold is not None:
            if threshold < minimum_threshold:
                return minimum_threshold

        return threshold

    def positions_likelihood_from(
        self,
        factor=1.0,
        minimum_threshold=None,
        positions: Optional[aa.Grid2DIrregular] = None,
        mass_centre_radial_distance_min: float = None,
        plane_redshift: Optional[float] = None,
    ) -> PositionsLH:
        """
        Returns a `PositionsLH` object from the result of a lens model-fit, where the maximum log likelihood mass
        and source models are used to determine the multiple image positions in the image-plane and source-plane
        and ray-trace them to the source-plane to determine the threshold.

        In chained fits, for example the SLaM pipelines, this means that a simple initial fit (e.g. SIE mass model,
        parametric source) can be used to determine the multiple image positions and threshold for a more complex
        subsequent fit (e.g. power-law mass model, pixelized source).

        The mass model central image is removed from the solution, as this is rarely physically observed and therefore
        should not be included in the likelihood penalty or resampling. It is removed by setting a positive
        magnification threshold in the `PointSolver`. For strange lens models the central image may still be
        solved for, in which case the `mass_centre_radial_distance_min` parameter can be used to remove it.

        Parameters
        ----------
        factor
            The value the computed threshold is multiplied by to make the position threshold larger or smaller than the
            maximum log likelihood model's threshold.
        minimum_threshold
            The output threshold is rounded up to this value if it is below it, to avoid extremely small threshold
            values.
        positions
            If input, these positions are used instead of the computed multiple image positions from the lens mass
            model.
        mass_centre_radial_distance_min
            The minimum radial distance from the mass model centre that a multiple image position must be to be
            included in the likelihood penalty or resampling. If `None` all positions are used. This is an additional
            method to remove central images that may make it through the point solver's magnification threshold.
        plane_redshift
            The plane redshift of the lensed source, which is only required when a double (or triple) source plane lens
            system is being analysed where the specific plane via its redshift is required to define whihch source
            galaxy is used to compute the multiple images.

        Returns
        -------
        The `PositionsLH` object used to apply a likelihood penalty or resample the positions.
        """

        if is_test_mode():
            return

        positions = (
            self.image_plane_multiple_image_positions(plane_redshift=plane_redshift)
            if positions is None
            else positions
        )

        if mass_centre_radial_distance_min is not None:
            mass_centre = self.max_log_likelihood_tracer.extract_attribute(
                cls=ag.mp.MassProfile, attr_name="centre"
            )

            distances = positions.distances_to_coordinate_from(
                coordinate=mass_centre[0]
            )

            positions = positions[distances > mass_centre_radial_distance_min]

        positions = aa.Grid2DIrregular(
            np.asarray(positions.array if hasattr(positions, "array") else positions)
        )

        threshold = self.positions_threshold_from(
            factor=factor,
            minimum_threshold=minimum_threshold,
            positions=positions,
            plane_redshift=plane_redshift,
        )

        return PositionsLH(
            positions=positions, threshold=threshold, plane_redshift=plane_redshift
        )


class ResultDataset(Result):
    @property
    def max_log_likelihood_tracer(self) -> Tracer:
        """
        An instance of a `Tracer` corresponding to the maximum log likelihood model inferred by the non-linear search.

        If a dataset is fitted the adapt images of the adapt image must first be associated with each galaxy.
        """
        instance = self.analysis.instance_with_associated_adapt_images_from(
            instance=self.instance
        )

        return self.analysis.tracer_via_instance_from(instance=instance)

    def source_plane_centre_from(
        self, plane_redshift: Optional[float] = None
    ) -> aa.Grid2DIrregular:
        """
        Return the centre of a source-plane galaxy via the following criteria:

        1) If the source plane contains only light profiles, return the first light's centre.
        2) If it contains an `LEq` return the centre of its brightest pixel instead.

        These centres are used by automatic position updating to determine the multiple-images of a best-fit lens model
        (and thus tracer) by back-tracing the centres to the image plane via the mass model.
        """
        source_plane_inversion_centre = self.source_plane_inversion_centre_from(
            plane_redshift=plane_redshift
        )

        if source_plane_inversion_centre is not None:
            return source_plane_inversion_centre

        return self.source_plane_light_profile_centre_from(
            plane_redshift=plane_redshift
        )

    def source_plane_inversion_centre_from(
        self, plane_redshift: Optional[float] = None
    ) -> aa.Grid2DIrregular:
        """
        Returns the centre of the brightest source pixel(s) of an `LEq`.

        These centres are used by automatic position updating to determine the best-fit lens model's image-plane
        multiple-image positions.
        """

        if self.max_log_likelihood_fit.inversion is not None:
            if self.max_log_likelihood_fit.inversion.has(cls=aa.Mapper):

                try:
                    plane_index = (
                        self.max_log_likelihood_tracer.plane_index_via_redshift_from(
                            redshift=plane_redshift
                        )
                    )
                    mapper_index = (
                        self.max_log_likelihood_tracer.plane_indexes_with_pixelizations[
                            plane_index
                        ]
                    )
                except TypeError:
                    mapper_index = 0

                inversion = self.max_log_likelihood_fit.inversion

                return inversion.max_pixel_centre(mapper_index=mapper_index)
