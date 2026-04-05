"""
Image-plane point-source solver for strong gravitational lensing.

Finding the multiple images of a point source requires solving the lens equation
θ = β + α(θ) for θ given a fixed source-plane position β.  This is an inverse
problem with no analytic solution for general mass distributions.

``PointSolver`` solves this numerically using a triangle-tiling approach:

1. The image plane is tiled with triangles.
2. Each triangle is ray-traced to the source plane.
3. Triangles that contain the source-plane coordinate β are refined recursively.
4. The centroids of the final refined triangles give the image-plane positions.

The output positions array is padded to a fixed size (``MAX_CONTAINING_SIZE``) using the
sentinel value ``inf`` for JAX compatibility — these ``inf`` entries are stripped by
default but can be retained for use inside a ``jax.jit``-traced function.
"""
import logging
from typing import Tuple, Optional

import numpy as np
import autoarray as aa
from autoarray.structures.triangles.shape import Point

from autolens.lens.tracer import Tracer
from .shape_solver import AbstractSolver


logger = logging.getLogger(__name__)


class PointSolver(AbstractSolver):

    def solve(
        self,
        tracer: Tracer,
        source_plane_coordinate: Tuple[float, float],
        plane_redshift: Optional[float] = None,
        remove_infinities: bool = True,
    ) -> aa.Grid2DIrregular:
        """
        Solve for the image plane coordinates that are traced to the source plane coordinate.

        This is done by tiling the image plane with triangles and checking if the source plane coordinate is contained
        within the triangle. The triangles are sub-sampled to increase the resolution with only the triangles that
        contain the source plane coordinate and their neighbours being kept.

        The means of the triangles are then filtered to keep only those with an absolute magnification above the
        threshold.

        The positions are stored on an array of fixed shape defined by `MAX_CONTAINING_SIZE`. This ensures the
        array is static, which is important for JAX compatibility. This array typically has many entries
        which use the sentinel value of `inf`, subsequent JAX calculations incorporated. By default, these
        sentinel values are removed from the output, for example general use outside of JAX when simulating
        strong lenses.

        Parameters
        ----------
        source_plane_coordinate
            The plane coordinate to trace to the image plane, which by default in the source-plane coordinate
            but could be a coordinate in another plane is `plane_redshift` is input.
        tracer
            The tracer that traces the image plane coordinates to the source plane
        plane_redshift
            The redshift of the plane coordinate, which for multi-plane systems may not be the source-plane.

        Returns
        -------
        A ``Grid2DIrregular`` of image-plane coordinates, always numpy-backed even when the
        solver uses a JAX backend internally.
        """
        kept_triangles = super().solve_triangles(
            tracer=tracer,
            shape=Point(*source_plane_coordinate),
            plane_redshift=plane_redshift,
        )

        filtered_means = self._filter_low_magnification(
            tracer=tracer, points=kept_triangles.means
        )

        solution = aa.Grid2DIrregular(
            [pair for pair in filtered_means], xp=self._xp
        ).array

        is_nan = self._xp.isnan(solution).any(axis=1)
        sentinel = self._xp.full_like(solution[0], fill_value=self._xp.inf)
        solution = self._xp.where(is_nan[:, None], sentinel, solution)

        if remove_infinities:

            solution = solution[~self._xp.isinf(solution).any(axis=1)]

        return aa.Grid2DIrregular(solution)
