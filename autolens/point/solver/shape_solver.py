import numpy as np
import logging
import math

from typing import Tuple, List, Iterator, Optional

import autoarray as aa

from autoarray.structures.triangles.shape import Shape

import autogalaxy as ag
from autolens.lens.tracer import Tracer
from .step import Step

logger = logging.getLogger(__name__)


class AbstractSolver:
    # noinspection PyPep8Naming
    def __init__(
        self,
        scale: float,
        initial_triangles,
        pixel_scale_precision: float,
        magnification_threshold=0.1,
        neighbor_degree: int = 1,
        xp=np,
    ):
        """
        Determine the image plane coordinates that are traced to be a source plane coordinate.

        This is performed efficiently by iteratively subdividing the image plane into triangles and checking if the
        source plane coordinate is contained within the triangle. The triangles are subsampled to increase the
        resolution

        Parameters
        ----------
        neighbor_degree
            The number of times recursively add neighbors for the triangles that contain
            the source plane coordinate.
        pixel_scale_precision
            The target pixel scale of the image grid.
        """
        self.scale = scale
        self.pixel_scale_precision = pixel_scale_precision
        self.magnification_threshold = magnification_threshold
        self.neighbor_degree = neighbor_degree

        self.initial_triangles = initial_triangles

        self.use_jax = xp is not np

    @property
    def _xp(self):
        if self.use_jax:
            import jax.numpy as jnp

            return jnp
        return np

    # noinspection PyPep8Naming
    @classmethod
    def for_grid(
        cls,
        grid: aa.Grid2D,
        pixel_scale_precision: float,
        magnification_threshold=0.1,
        neighbor_degree: int = 1,
        xp=np,
    ):
        """
        Create a solver for a given grid.

        The grid defines the limits of the image plane and the pixel scale.

        Parameters
        ----------
        grid
            The grid to use.
        pixel_scale_precision
            The precision to which the triangles should be subdivided.
        magnification_threshold
            The threshold for the magnification under which multiple images are filtered.
        max_containing_size
            Only applies to JAX. This is the maximum number of multiple images expected.
            We need to know this in advance to allocate memory for the JAX array.
        neighbor_degree
            The number of times recursively add neighbors for the triangles that contain

        Returns
        -------
        The solver.
        """
        scale = grid.pixel_scale

        y = grid[:, 0]
        x = grid[:, 1]

        y_min = y.min()
        y_max = y.max()
        x_min = x.min()
        x_max = x.max()

        return cls.for_limits_and_scale(
            y_min=y_min,
            y_max=y_max,
            x_min=x_min,
            x_max=x_max,
            scale=scale,
            pixel_scale_precision=pixel_scale_precision,
            magnification_threshold=magnification_threshold,
            neighbor_degree=neighbor_degree,
            xp=xp,
        )

    @classmethod
    def for_limits_and_scale(
        cls,
        y_min=-1.0,
        y_max=1.0,
        x_min=-1.0,
        x_max=1.0,
        scale=0.1,
        pixel_scale_precision: float = 0.001,
        magnification_threshold=0.1,
        neighbor_degree: int = 1,
        xp=np,
    ):
        """
        Create a solver for a given grid.

        The grid defines the limits of the image plane and the pixel scale.

        Parameters
        ----------
        y_min
        y_max
        x_min
        x_max
            The limits of the image plane in pixels.
        scale
            The pixel scale of the image plane. The initial triangles have this side length.
        pixel_scale_precision
            The precision to which the triangles should be subdivided.
        magnification_threshold
            The threshold for the magnification under which multiple images are filtered.
        neighbor_degree
            The number of times recursively add neighbors for the triangles that contain

        Returns
        -------
        The solver.
        """

        if xp.__name__.startswith("jax"):
            from autoarray.structures.triangles.coordinate_array import (
                CoordinateArrayTriangles as triangle_cls,
            )
        else:
            from autoarray.structures.triangles.coordinate_array_np import (
                CoordinateArrayTrianglesNp as triangle_cls,
            )

        initial_triangles = triangle_cls.for_limits_and_scale(
            y_min=y_min,
            y_max=y_max,
            x_min=x_min,
            x_max=x_max,
            scale=scale,
        )

        return cls(
            scale=scale,
            initial_triangles=initial_triangles,
            pixel_scale_precision=pixel_scale_precision,
            magnification_threshold=magnification_threshold,
            neighbor_degree=neighbor_degree,
            xp=xp,
        )

    @property
    def n_steps(self) -> int:
        """
        How many times should triangles be subdivided?
        """
        return math.ceil(math.log2(self.scale / self.pixel_scale_precision))

    def _plane_grid(
        self,
        tracer: Tracer,
        grid: aa.type.Grid2DLike,
        plane_redshift: Optional[float] = None,
    ) -> aa.type.Grid2DLike:
        """
        Calculate the source plane grid from the image plane grid.

        Parameters
        ----------
        grid
            The image plane grid.

        Returns
        -------
        The source plane grid computed by applying the deflections to the image plane grid.
        """
        if plane_redshift is None:
            plane_index = -1
        else:
            plane_index = tracer.plane_index_via_redshift_from(redshift=plane_redshift)

        deflections = tracer.deflections_between_planes_from(
            grid=grid, plane_i=0, plane_j=plane_index, xp=self._xp
        )
        # noinspection PyTypeChecker
        return grid.grid_2d_via_deflection_grid_from(deflection_grid=deflections)

    def solve_triangles(
        self,
        tracer: Tracer,
        shape: Shape,
        plane_redshift: Optional[float] = None,
    ):
        """
        Solve for the image plane coordinates that are traced to the source plane coordinate.

        This is done by tiling the image plane with triangles and checking if the source plane coordinate is contained
        within the triangle. The triangles are subsampled to increase the resolution with only the triangles that
        contain the source plane coordinate and their neighbours being kept.

        The means of the triangles are then filtered to keep only those with an absolute magnification above the
        threshold.

        Parameters
        ----------
        tracer
            The tracer to use to trace the image plane coordinates to the source plane.
        shape
            The shape in the source plane for which we want to identify the image plane coordinates.
        plane_redshift
            The redshift of the source plane.

        Returns
        -------
        A list of image plane coordinates that are traced to the source plane coordinate.
        """
        if self.n_steps == 0:
            raise ValueError(
                "The target pixel scale is too large to subdivide the triangles."
            )

        steps = list(
            self.steps(
                tracer=tracer,
                shape=shape,
                plane_redshift=plane_redshift,
            )
        )
        final_step = steps[-1]
        return final_step.filtered_triangles

    def _filter_low_magnification(
        self, tracer: Tracer, points: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """
        Filter the points to keep only those with an absolute magnification above the threshold.

        Parameters
        ----------
        points
            The points to filter.

        Returns
        -------
        The points with an absolute magnification above the threshold.
        """
        points = self._xp.array(points)
        magnifications = ag.LensCalc.from_mass_obj(
            tracer
        ).magnification_2d_via_hessian_from(
            grid=aa.Grid2DIrregular(points).array, xp=self._xp
        )
        mask = self._xp.abs(magnifications) > self.magnification_threshold
        return self._xp.where(mask[:, None], points, self._xp.nan)

    def _plane_triangles(
        self,
        tracer: Tracer,
        triangles: aa.AbstractTriangles,
        plane_redshift,
    ):
        """
        Filter the triangles to keep only those that meet the solver condition
        """
        plane_grid = self._plane_grid(
            tracer=tracer,
            grid=aa.Grid2DIrregular(triangles.vertices),
            plane_redshift=plane_redshift,
        )

        return triangles.with_vertices(plane_grid.array)

    def steps(
        self,
        tracer: Tracer,
        shape: Shape,
        plane_redshift: Optional[float] = None,
    ) -> Iterator[Step]:
        """
        Iterate over the steps of the triangle solver algorithm.

        Parameters
        ----------
        tracer
            The tracer to use to trace the image plane coordinates to the source plane.
        plane_redshift
            The redshift of the source plane.
        shape
            The shape in the source plane for which we want to identify the image plane coordinates.

        Returns
        -------
        An iterator over the steps of the triangle solver algorithm.
        """
        initial_triangles = self.initial_triangles

        for number in range(self.n_steps):
            plane_triangles = self._plane_triangles(
                tracer=tracer,
                triangles=initial_triangles,
                plane_redshift=plane_redshift,
            )

            indexes = plane_triangles.containing_indices(shape=shape)
            kept_triangles = initial_triangles.for_indexes(indexes=indexes)

            neighbourhood = kept_triangles
            for _ in range(self.neighbor_degree):
                neighbourhood = neighbourhood.neighborhood()

            up_sampled = neighbourhood.up_sample()

            yield Step(
                number=number,
                initial_triangles=initial_triangles,
                filtered_triangles=kept_triangles,
                neighbourhood=neighbourhood,
                up_sampled=up_sampled,
                plane_triangles=plane_triangles,
            )

            initial_triangles = up_sampled

    def tree_flatten(self):
        return (), (
            self.scale,
            self.pixel_scale_precision,
            self.magnification_threshold,
            self.initial_triangles,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            scale=aux_data[0],
            pixel_scale_precision=aux_data[1],
            magnification_threshold=aux_data[2],
            initial_triangles=aux_data[3],
        )


class ShapeSolver(AbstractSolver):
    def find_magnification(
        self,
        tracer: Tracer,
        shape: Shape,
        plane_redshift: Optional[float] = None,
    ) -> float:
        """
        Find the magnification of the shape in the source plane.

        Parameters
        ----------
        tracer
            A tracer that traces the image plane to the source plane.
        shape
            The shape of an image plane pixel.
        plane_redshift
            The redshift of the source plane.

        Returns
        -------
        The magnification of the shape in the source plane.
        """
        kept_triangles = super().solve_triangles(
            tracer=tracer,
            shape=shape,
            plane_redshift=plane_redshift,
        )
        return kept_triangles.area / shape.area
