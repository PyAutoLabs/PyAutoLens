"""
Abstract triangle-tiling solver and shape-based solver for point-source positions.

``AbstractSolver`` (and its concrete subclass ``ShapeSolver``) implement the hierarchical
triangle-refinement algorithm that underlies ``PointSolver``:

1. An initial grid of triangles covers the image plane.
2. Each triangle is ray-traced to the source plane; those that contain the target
   source coordinate are kept.
3. Kept triangles are sub-divided for the next refinement iteration.
4. After ``n_steps`` levels the centroids of the finest triangles give the image positions.

``ShapeSolver`` extends this base with support for fitting extended source *shapes*
(e.g. rings, arcs) rather than point coordinates, used for morphological constraints.
"""
import numpy as np
import logging
import math

from typing import Tuple, List, Iterator, Optional, Union

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
        y_min: float,
        y_max: float,
        x_min: float,
        x_max: float,
        scale: float,
        pixel_scale_precision: float,
        magnification_threshold=0.1,
        neighbor_degree: int = 1,
        use_jax: bool = False,
    ):
        """
        Determine the image plane coordinates that are traced to be a source plane coordinate.

        This is performed efficiently by iteratively subdividing the image plane into triangles and checking if the
        source plane coordinate is contained within the triangle. The triangles are subsampled to increase the
        resolution.

        The solver is stateless with respect to the array module (``xp``); the array
        module is chosen per ``.solve()`` call and the initial triangle tiling is
        built lazily from the stored geometry primitives.

        Parameters
        ----------
        y_min, y_max, x_min, x_max
            The extent of the image plane used to tile the initial triangles.
        scale
            The pixel scale of the image plane. The initial triangles have this side length.
        pixel_scale_precision
            The target pixel scale of the image grid.
        magnification_threshold
            The threshold for the magnification under which multiple images are filtered.
        neighbor_degree
            The number of times recursively add neighbors for the triangles that contain
            the source plane coordinate.
        use_jax
            If ``True``, ``.solve()`` defaults to ``xp=jnp`` and ``remove_infinities=False``
            (the JAX-static-shape contract), and registers ``Tracer`` plus the concrete
            galaxy / profile classes it carries as JAX pytrees on the first call. The
            user wraps the call in their own ``@jax.jit`` — see the ``lens_calc.py``
            workspace guide for the canonical pattern.
        """
        self.y_min = y_min
        self.y_max = y_max
        self.x_min = x_min
        self.x_max = x_max
        self.scale = scale
        self.pixel_scale_precision = pixel_scale_precision
        self.magnification_threshold = magnification_threshold
        self.neighbor_degree = neighbor_degree
        self.use_jax = use_jax

        # The grid ``for_grid`` was built from, when there was one. It is the image-plane
        # pixelization ``image_regions_from`` reports its regions on; a solver built via
        # ``for_limits_and_scale`` has only an extent and no pixel grid, so it stays `None`
        # and the grid must be passed to ``image_regions_from`` explicitly. It is
        # deliberately *not* part of the pytree (``tree_flatten``): it is used only by the
        # numpy-only, never-jitted mappings layer.
        self.grid = None

    @property
    def _xp(self):
        """The array module the solver runs against by default. ``jnp`` when
        ``use_jax=True``, ``np`` otherwise. ``.solve()`` falls back to this when
        the caller does not pass ``xp=`` explicitly."""
        if self.use_jax:
            import jax.numpy as jnp

            return jnp
        return np

    def _initial_triangles(self, xp):
        """
        Build the initial triangle tiling for the stored image-plane extent, using
        the JAX or NumPy triangle implementation depending on ``xp``.

        On the JAX path the tiling carries its static vertex table
        (``static_vertices=True``, see
        `autoarray.structures.triangles.coordinate_array.static_vertex_table`): the
        step-0 lattice depends only on the solver's geometry, which is static pytree
        aux data, so its geometrically unique vertices are a cached compile-time
        constant. Step 0's ``triangles.vertices`` -- the grid every step-0 deflection is
        evaluated on -- is then the ``(V, 2)`` unique table instead of the flat
        ``(3N, 2)`` one (11 859 rather than 69 849 rows for the +-9.9" / 0.2" lattice
        of a 100x100, 0.2" grid), and ``triangles.indices`` gathers the traced
        vertices back to the triangles. Later steps are data-dependent and keep the
        flat table. A change of geometry changes the pytree aux data, so it retraces
        with the matching table. The NumPy path is unchanged (its sibling deduplicates
        with dynamic shapes).
        """
        if xp.__name__.startswith("jax"):
            from autoarray.structures.triangles.coordinate_array import (
                CoordinateArrayTriangles,
            )

            return CoordinateArrayTriangles.for_limits_and_scale(
                y_min=self.y_min,
                y_max=self.y_max,
                x_min=self.x_min,
                x_max=self.x_max,
                scale=self.scale,
                static_vertices=True,
            )

        from autoarray.structures.triangles.coordinate_array_np import (
            CoordinateArrayTrianglesNp,
        )

        return CoordinateArrayTrianglesNp.for_limits_and_scale(
            y_min=self.y_min,
            y_max=self.y_max,
            x_min=self.x_min,
            x_max=self.x_max,
            scale=self.scale,
        )

    # noinspection PyPep8Naming
    @classmethod
    def for_grid(
        cls,
        grid: aa.Grid2D,
        pixel_scale_precision: float,
        magnification_threshold=0.1,
        neighbor_degree: int = 1,
        use_jax: bool = False,
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
        neighbor_degree
            The number of times recursively add neighbors for the triangles that contain
        use_jax
            Forwarded to the constructor; see ``__init__``.

        Returns
        -------
        The solver.
        """
        scale = grid.pixel_scale

        y = grid[:, 0]
        x = grid[:, 1]

        solver = cls.for_limits_and_scale(
            y_min=y.min(),
            y_max=y.max(),
            x_min=x.min(),
            x_max=x.max(),
            scale=scale,
            pixel_scale_precision=pixel_scale_precision,
            magnification_threshold=magnification_threshold,
            neighbor_degree=neighbor_degree,
            use_jax=use_jax,
        )

        # Remembered so ``ShapeSolver.image_regions_from`` can report image-plane regions
        # on the same pixelization the caller is plotting, without being handed the grid
        # a second time.
        solver.grid = grid

        return solver

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
        use_jax: bool = False,
    ):
        """
        Create a solver for an explicit image-plane extent.

        Parameters
        ----------
        y_min, y_max, x_min, x_max
            The limits of the image plane in pixels.
        scale
            The pixel scale of the image plane. The initial triangles have this side length.
        pixel_scale_precision
            The precision to which the triangles should be subdivided.
        magnification_threshold
            The threshold for the magnification under which multiple images are filtered.
        neighbor_degree
            The number of times recursively add neighbors for the triangles that contain
        use_jax
            Forwarded to the constructor; see ``__init__``.

        Returns
        -------
        The solver.
        """
        return cls(
            y_min=y_min,
            y_max=y_max,
            x_min=x_min,
            x_max=x_max,
            scale=scale,
            pixel_scale_precision=pixel_scale_precision,
            magnification_threshold=magnification_threshold,
            neighbor_degree=neighbor_degree,
            use_jax=use_jax,
        )

    @property
    def n_steps(self) -> int:
        """
        How many times should triangles be subdivided?
        """
        return math.ceil(math.log2(self.scale / self.pixel_scale_precision))

    @staticmethod
    def _plane_index(tracer: Tracer, plane_redshift: Optional[float] = None) -> int:
        """
        Resolve the index of the plane being solved for.

        Both the triangle search (``_plane_grid``) and the magnification filter
        (``_filter_low_magnification``) must agree on which plane is the source plane; when they
        disagreed, the search solved to the requested plane while the filter measured
        magnification at the tracer's last plane, and every candidate image was discarded
        (PyAutoLens #480). They therefore share this one resolution rather than each repeating it.

        Parameters
        ----------
        tracer
            The tracer whose planes are being solved through.
        plane_redshift
            The redshift of the source plane. ``None`` means the tracer's last plane, which is the
            single-source case and is expressed as the index ``-1``.

        Returns
        -------
        The index of the plane at ``plane_redshift``, or ``-1`` when it is ``None``.
        """
        if plane_redshift is None:
            return -1

        plane_index = tracer.plane_index_via_redshift_from(redshift=plane_redshift)

        if plane_index is None:
            raise ValueError(
                f"No plane in the tracer has redshift {plane_redshift}, so the solver cannot "
                f"determine which plane to solve for. The tracer's plane redshifts are "
                f"{list(tracer.plane_redshifts)}. Pass `plane_redshift` as one of these, or omit "
                f"it to solve for the last plane."
            )

        return plane_index

    def _plane_grid(
        self,
        tracer: Tracer,
        grid: aa.type.Grid2DLike,
        xp,
        plane_redshift: Optional[float] = None,
    ) -> aa.type.Grid2DLike:
        """
        Calculate the source plane grid from the image plane grid.

        Parameters
        ----------
        grid
            The image plane grid.
        xp
            The array module (``numpy`` or ``jax.numpy``) used for the deflection computation.

        Returns
        -------
        The source plane grid computed by applying the deflections to the image plane grid.
        """
        plane_index = self._plane_index(tracer=tracer, plane_redshift=plane_redshift)

        deflections = tracer.deflections_between_planes_from(
            grid=grid, plane_i=0, plane_j=plane_index, xp=xp
        )
        # noinspection PyTypeChecker
        return grid.grid_2d_via_deflection_grid_from(deflection_grid=deflections, xp=xp)

    def solve_triangles(
        self,
        tracer: Tracer,
        shape: Shape,
        xp=None,
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
        xp
            The array module (``numpy`` or ``jax.numpy``) the solve runs in. ``None`` (the
            default) falls back to ``self._xp``, which is ``jnp`` when the solver was
            constructed with ``use_jax=True`` and ``np`` otherwise -- the same contract as
            ``PointSolver.solve``.
        plane_redshift
            The redshift of the source plane.

        Returns
        -------
        A list of image plane coordinates that are traced to the source plane coordinate.
        """
        if xp is None:
            xp = self._xp

        # `n_steps` is `ceil(log2(scale / pixel_scale_precision))`, which is <= 0 whenever the
        # requested precision is coarser than (or equal to) the initial triangle scale. A negative
        # value used to slip past an `== 0` check and make `steps` an empty list, surfacing as
        # `IndexError: list index out of range` on `steps[-1]` below. Reject it here, naming the
        # parameter the caller controls -- do NOT clamp to 1, which would silently solve at a
        # precision the caller did not ask for.
        if self.n_steps <= 0:
            raise ValueError(
                f"""
                The requested `pixel_scale_precision` is too large to subdivide the triangles.

                pixel_scale_precision = {self.pixel_scale_precision}
                initial triangle scale = {self.scale}

                The solver refines triangles by repeated bisection, so it needs
                `pixel_scale_precision` to be smaller than the initial triangle scale; here it
                would require {self.n_steps} subdivision steps.

                Decrease `pixel_scale_precision` (e.g. to {self.scale / 10.0:.3g} or smaller) so the
                solver can refine the tiling.
                """
            )

        steps = list(
            self.steps(
                tracer=tracer,
                shape=shape,
                xp=xp,
                plane_redshift=plane_redshift,
            )
        )
        final_step = steps[-1]
        return final_step.filtered_triangles

    def _filter_low_magnification(
        self,
        tracer: Tracer,
        points: List[Tuple[float, float]],
        xp,
        plane_redshift: Optional[float] = None,
    ) -> List[Tuple[float, float]]:
        """
        Filter the points to keep only those with an absolute magnification above the threshold.

        The magnification is measured at the plane being solved for, which is the plane the
        triangle search traced to. Measuring it at the tracer's last plane instead de-magnifies
        every candidate image of an intermediate-plane source by orders of magnitude, so the
        threshold discards all of them and the solve returns no images (PyAutoLens #480).

        ``LensCalc.from_tracer`` binds ``tracer.deflections_between_planes_from`` for the
        requested plane, which is the same callable the search uses. For ``plane_redshift=None``
        this is not merely equivalent to the previous ``from_mass_obj(tracer)`` but identical:
        ``Tracer.deflections_yx_2d_from`` dispatches to ``deflections_between_planes_from`` with
        its ``plane_i=0, plane_j=-1`` defaults whenever the tracer has more than one plane, and a
        single-plane tracer has only the one plane to measure at.

        Parameters
        ----------
        points
            The points to filter.
        xp
            The array module used for the magnification calculation.
        plane_redshift
            The redshift of the plane being solved for. ``None`` measures at the last plane.

        Returns
        -------
        The points with an absolute magnification above the threshold.
        """
        points = xp.array(points)
        plane_index = self._plane_index(tracer=tracer, plane_redshift=plane_redshift)
        magnifications = ag.LensCalc.from_tracer(
            tracer,
            use_multi_plane=True,
            plane_i=0,
            plane_j=plane_index,
        ).magnification_2d_via_hessian_from(
            grid=aa.Grid2DIrregular(points).array, xp=xp
        )
        mask = xp.abs(magnifications) > self.magnification_threshold
        return xp.where(mask[:, None], points, xp.nan)

    def _plane_triangles(
        self,
        tracer: Tracer,
        triangles: aa.AbstractTriangles,
        xp,
        plane_redshift,
    ):
        """
        Filter the triangles to keep only those that meet the solver condition
        """
        plane_grid = self._plane_grid(
            tracer=tracer,
            grid=aa.Grid2DIrregular(triangles.vertices),
            xp=xp,
            plane_redshift=plane_redshift,
        )

        return triangles.with_vertices(plane_grid.array)

    def steps(
        self,
        tracer: Tracer,
        shape: Shape,
        xp=None,
        plane_redshift: Optional[float] = None,
    ) -> Iterator[Step]:
        """
        Iterate over the steps of the triangle solver algorithm.

        Parameters
        ----------
        tracer
            The tracer to use to trace the image plane coordinates to the source plane.
        shape
            The shape in the source plane for which we want to identify the image plane coordinates.
        xp
            The array module (``numpy`` or ``jax.numpy``) the step iteration runs in.
            ``None`` (the default) falls back to ``self._xp``.
        plane_redshift
            The redshift of the source plane.

        Returns
        -------
        An iterator over the steps of the triangle solver algorithm.

        Notes
        -----
        The area of ``step.filtered_triangles`` is **not** monotone over the steps. Step 0
        tiles the whole image plane at ``self.scale`` and keeps every triangle whose traced
        image meets the shape; each later step then filters the *neighbourhood* of the
        previous kept set, which is larger than that set, so the kept area oscillates while
        it converges. What is monotone is the search envelope: ``step.neighbourhood``
        shrinks at every step, because the rim it adds is one triangle wide and the
        triangles halve. The residual on the converged kept area is roughly one triangle
        edge per boundary triangle -- see ``ShapeSolver.find_magnification``.
        """
        if xp is None:
            xp = self._xp

        initial_triangles = self._initial_triangles(xp)

        for number in range(self.n_steps):
            plane_triangles = self._plane_triangles(
                tracer=tracer,
                triangles=initial_triangles,
                xp=xp,
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
            self.y_min,
            self.y_max,
            self.x_min,
            self.x_max,
            self.scale,
            self.pixel_scale_precision,
            self.magnification_threshold,
            self.neighbor_degree,
            self.use_jax,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            y_min,
            y_max,
            x_min,
            x_max,
            scale,
            pixel_scale_precision,
            magnification_threshold,
            neighbor_degree,
            use_jax,
        ) = aux_data
        return cls(
            y_min=y_min,
            y_max=y_max,
            x_min=x_min,
            x_max=x_max,
            scale=scale,
            pixel_scale_precision=pixel_scale_precision,
            magnification_threshold=magnification_threshold,
            neighbor_degree=neighbor_degree,
            use_jax=use_jax,
        )


class ShapeSolver(AbstractSolver):
    """
    The triangle solver for an extended source-plane `Shape`, rather than a point.

    JAX
    ---
    ``ShapeSolver`` is a NumPy-only solver and rejects ``use_jax=True`` / ``xp=jax.numpy``.
    The JAX triangle containers keep static shapes by truncating every refinement step to
    ``ArrayTriangles.MAX_CONTAINING_SIZE`` (15) triangles — ample for a ``Point``, which
    lies inside a handful of triangles, and meaningless for a shape with area, whose kept
    set grows with the size of its images. Before this was found, ``use_jax=True`` was
    silently ignored (``find_magnification`` hardcoded ``xp=np``); routing it through
    ``self._xp`` instead would have replaced a silently-ignored flag with a silently wrong
    number — an Isothermal sphere whose true magnification is 6.86 measures 0.13 under
    JAX. Lifting the cap is a redesign of the JAX containers, not a fix here, so the flag
    raises instead. See ``test_shape_solver.py::test_jax_and_numpy_kept_triangles_agree``.
    """

    # The one message both rejection routes raise, so a caller sees the same explanation
    # whether they set `use_jax=True` or passed `xp=jax.numpy` -- and, since it is a plain
    # string, whether or not JAX is installed.
    _JAX_REJECTED_MESSAGE = (
        "ShapeSolver does not support JAX. The JAX triangle containers truncate "
        "every refinement step to ArrayTriangles.MAX_CONTAINING_SIZE (15) "
        "triangles to keep static shapes, which is enough for a Point but not for "
        "a Shape with area: the kept triangles of an extended source number in the "
        "thousands, so the JAX path silently measures a small fraction of the "
        "image and returns a magnification orders of magnitude too small. Use "
        "the NumPy path (`use_jax=False`, the default), or `PointSolver` if you "
        "want the JAX-differentiable positions of a point source."
    )

    def _xp_from(self, xp):
        """
        Resolve the array module for a solve, rejecting JAX.

        The ``use_jax=True`` case is rejected **before** ``self._xp`` is consulted, because
        ``self._xp`` imports ``jax.numpy``. Resolving first and checking afterwards made a
        JAX-free install raise ``ModuleNotFoundError: No module named 'jax'`` from inside
        the solver rather than the explanation below -- the wrong error, and one which
        blamed the environment for a decision the solver had already taken. The
        explicitly-passed-module check reads only ``__name__``, so it imports nothing
        either.

        Parameters
        ----------
        xp
            The caller's array module, or ``None`` to use ``self._xp``.

        Returns
        -------
        The array module, which is always ``numpy``.
        """
        if xp is None:
            if self.use_jax:
                raise NotImplementedError(self._JAX_REJECTED_MESSAGE)

            return self._xp

        if getattr(xp, "__name__", "").startswith("jax"):
            raise NotImplementedError(self._JAX_REJECTED_MESSAGE)

        return xp

    def solve_triangles(
        self,
        tracer: Tracer,
        shape: Shape,
        xp=None,
        plane_redshift: Optional[float] = None,
    ):
        """
        The kept image-plane triangles whose traced images meet `shape`.

        See ``AbstractSolver.solve_triangles``; this override only rejects the JAX path
        (see the class docstring).
        """
        return super().solve_triangles(
            tracer=tracer,
            shape=shape,
            xp=self._xp_from(xp),
            plane_redshift=plane_redshift,
        )

    def steps(
        self,
        tracer: Tracer,
        shape: Shape,
        xp=None,
        plane_redshift: Optional[float] = None,
    ) -> Iterator[Step]:
        """
        Iterate over the refinement steps of the solve.

        See ``AbstractSolver.steps``; this override only rejects the JAX path (see the
        class docstring).
        """
        return super().steps(
            tracer=tracer,
            shape=shape,
            xp=self._xp_from(xp),
            plane_redshift=plane_redshift,
        )

    def find_magnification(
        self,
        tracer: Tracer,
        shape: Shape,
        xp=None,
        plane_redshift: Optional[float] = None,
        per_image: bool = False,
    ) -> Union[float, List[float]]:
        """
        Find the magnification of the shape in the source plane.

        The magnification is the total image-plane area the shape's images cover divided by
        the shape's own area, measured on the kept triangles of the finest refinement step.

        Convergence and error
        ---------------------
        A triangle is kept when its *traced* image meets the shape, so the kept set covers
        the true image region plus a rim of boundary triangles which only partly overlap
        it. The area therefore approaches the true value **from above**, with an error of
        order ``perimeter * side_length`` — one triangle edge per boundary triangle. Since
        ``side_length`` at the finest step is ``pixel_scale_precision``, halving
        ``pixel_scale_precision`` roughly halves the error. For a source much larger than
        ``pixel_scale_precision`` this is a sub-percent effect; for a source comparable to
        it the answer is unreliable, and the caller should refine further.

        The other error is physical, not numerical: this measures the magnification of a
        *finite* shape, which equals the point magnification only in the limit of a small
        shape. An Isothermal sphere with ``einstein_radius=1.0`` and a source at
        ``beta=0.3`` has an analytic total magnification ``2 * theta_E / beta = 6.67``; at
        ``pixel_scale_precision=0.005`` a ``Circle`` of radius 0.2 gives 7.12 (7% high,
        finite-source), radius 0.1 gives 6.76 (1.3%) and radius 0.05 gives 6.66 (0.1%).

        Parameters
        ----------
        tracer
            A tracer that traces the image plane to the source plane.
        shape
            The shape of an image plane pixel.
        xp
            The array module the magnification calculation runs in. ``None`` (the default)
            falls back to ``self._xp``. This previously defaulted to ``np``
            unconditionally, so a solver built with ``use_jax=True`` silently ran in NumPy;
            it now consults the flag and raises, because the JAX path is wrong for a shape
            rather than merely unused (see the class docstring).
        plane_redshift
            The redshift of the source plane.
        per_image
            If ``True`` the kept triangles are grouped into their connected components (the
            separate multiple images) and one magnification is returned per image, largest
            first, dropping any component whose magnification does not exceed
            ``self.magnification_threshold``. If ``False`` (the default) the single total
            over every kept triangle is returned, unfiltered.

        Returns
        -------
        The magnification of the shape in the source plane, or the list of per-image
        magnifications when ``per_image=True``.

        Notes
        -----
        ``sum(per_image)`` is not identical to the total: the total is unfiltered, so it
        also carries any component the threshold rejects. For an Isothermal *sphere* the
        rejected component is the demagnified central image the profile's singular centre
        produces, worth ~0.004 of a magnification of 6.7.
        """
        kept_triangles = self.solve_triangles(
            tracer=tracer,
            shape=shape,
            xp=xp,
            plane_redshift=plane_redshift,
        )

        if not per_image:
            return kept_triangles.area / shape.area

        return [
            magnification
            for magnification in sorted(
                (
                    component.area / shape.area
                    for component in _connected_components_from(kept_triangles)
                ),
                reverse=True,
            )
            if magnification > self.magnification_threshold
        ]

    def image_regions_from(
        self,
        tracer: Tracer,
        shape: Shape,
        plane_redshift: Optional[float] = None,
        grid: Optional[aa.type.Grid2DLike] = None,
        xp=None,
    ) -> List[aa.ImageRegion]:
        """
        The image-plane regions (the multiple images) a source-plane shape maps to.

        This is the parametric-source counterpart of
        ``autoarray.inversion.mappings.image_regions_from``, which does the same job for a
        pixelized source via a ``Mapper``. Both return the same ``ImageRegion`` objects, so
        ``brightest_coordinate_from`` / ``centroid_from`` / ``flux_from`` / ``area`` behave
        identically whichever engine produced them.

        The kept triangles of the finest refinement step are grouped into connected
        components by shared vertices — each component is one multiple image — and each
        component is converted to an image-plane pixel region by marking every pixel of
        `grid` which contains a kept-triangle vertex or centroid. Because the triangles
        carry the geometry, a source far smaller than a pixel still produces regions.

        Components whose magnification (their share of the kept area, divided by the
        shape's area) does not exceed ``self.magnification_threshold`` are dropped. That
        threshold was previously accepted by the constructor and never used by
        ``ShapeSolver`` at all; without it the singular centre of an Isothermal sphere
        contributes a spurious two-triangle "image" at the lens centre.

        Parameters
        ----------
        tracer
            The tracer which traces the image plane to the source plane.
        shape
            The source-plane shape whose images are found.
        plane_redshift
            The redshift of the plane `shape` lives in. ``None`` is the tracer's last plane.
        grid
            The image-plane grid whose pixels the regions are reported on. ``None`` uses the
            grid the solver was built from by ``for_grid``; a solver built by
            ``for_limits_and_scale`` has no grid and must be passed one.
        xp
            Unused except for symmetry with the rest of the solver: this is diagnostic /
            visualization code, so it is numpy-only and never jitted. Passing ``jax.numpy``
            raises.

        Returns
        -------
        One ``ImageRegion`` per multiple image, ordered by decreasing area.
        """
        if xp is not None and xp is not np:
            raise ValueError(
                "`ShapeSolver.image_regions_from` is numpy-only and is never jitted: it "
                "builds `Mask2D` and contour objects for plotting and diagnostics. Pass "
                "`xp=None` (or `xp=numpy`); use `solve_triangles` for the JAX path."
            )

        grid = self._grid_from(grid=grid)

        kept_triangles = self.solve_triangles(
            tracer=tracer,
            shape=shape,
            xp=np,
            plane_redshift=plane_redshift,
        )

        components = _connected_components_from(kept_triangles)

        shape_area = shape.area

        regions = []

        for component in components:
            if (
                shape_area > 0.0
                and component.area / shape_area <= self.magnification_threshold
            ):
                continue

            region = _image_region_from_triangles(
                triangles=component.triangles, mask=grid.mask
            )

            if region is not None:
                regions.append(region)

        regions.sort(key=lambda region: -len(region.slim_indexes))

        return regions

    def mapping_from(
        self,
        tracer: Tracer,
        shape: Shape,
        plane_redshift: Optional[float] = None,
        grid: Optional[aa.type.Grid2DLike] = None,
    ) -> aa.Mapping:
        """
        The ``Mapping`` pairing a source-plane shape with the image-plane regions it maps to.

        The source-plane side of a shape has no mesh pixels, so ``pix_indexes`` is empty and
        ``peak_value`` is ``None``; the shape's own boundary is its one source contour, which
        is what the source-plane panels of ``subplot_mappings`` draw.

        Parameters
        ----------
        tracer
            The tracer which traces the image plane to the source plane.
        shape
            The source-plane shape which is mapped.
        plane_redshift
            The redshift of the plane `shape` lives in. ``None`` is the tracer's last plane.
        grid
            The image-plane grid the regions are reported on; ``None`` uses the solver's own.

        Returns
        -------
        The ``Mapping`` of the shape.
        """
        return aa.Mapping(
            pix_indexes=np.array([], dtype=int),
            source_contours=[shape.boundary()],
            # `Shape.x` is element 0 of a coordinate pair, which for every PyAuto grid is
            # the `y` coordinate — see the `Shape` class docstring. A `Circle` at a light
            # profile centre `(y_c, x_c)` is built positionally as `Circle(y_c, x_c, r)`.
            source_centre=(float(shape.x), float(shape.y)),
            image_regions=self.image_regions_from(
                tracer=tracer,
                shape=shape,
                plane_redshift=plane_redshift,
                grid=grid,
            ),
            peak_value=None,
        )

    def _grid_from(self, grid: Optional[aa.type.Grid2DLike]) -> aa.type.Grid2DLike:
        """
        Resolve the image-plane grid the regions are reported on.

        Parameters
        ----------
        grid
            An explicit grid, or ``None`` to use the one ``for_grid`` remembered.

        Returns
        -------
        The grid.
        """
        grid = self.grid if grid is None else grid

        if grid is None:
            raise ValueError(
                "This solver was built with `for_limits_and_scale`, which has an image-plane "
                "extent but no pixel grid, so `image_regions_from` does not know which pixels "
                "to report its regions on. Pass `grid=` explicitly, or build the solver with "
                "`ShapeSolver.for_grid(grid=..., ...)`."
            )

        return grid


def _vertex_labels_from(triangles: np.ndarray) -> np.ndarray:
    """
    Label the vertices of `triangles` so that coincident vertices share a label.

    Exact float equality (what ``AbstractTriangles.indices`` uses, via ``np.unique``) does
    **not** identify coincident vertices here: a lattice point reached from two different
    triangles is computed by two different sequences of floating point operations and the
    results differ in the last ulp. Grouping the kept triangles of an Isothermal sphere on
    exact equality splits its two images into 397 fragments rather than 2. The vertices are
    therefore quantised to a thousandth of a triangle edge — far below the spacing of
    distinct vertices, far above the ulp — before being deduplicated.

    Parameters
    ----------
    triangles
        An ``(n, 3, 2)`` array of triangle vertices.

    Returns
    -------
    An ``(n, 3)`` integer array of vertex labels.
    """
    vertices = np.asarray(triangles).reshape(-1, 2)

    edges = np.linalg.norm(
        np.asarray(triangles)[:, 0] - np.asarray(triangles)[:, 1], axis=-1
    )

    tolerance = float(np.median(edges)) * 1.0e-3

    if not np.isfinite(tolerance) or tolerance <= 0.0:
        tolerance = 1.0e-12

    _, labels = np.unique(np.round(vertices / tolerance), axis=0, return_inverse=True)

    return np.asarray(labels).reshape(-1, 3)


def _connected_components_from(triangles) -> List["_TriangleComponent"]:
    """
    Group a set of triangles into connected components which share vertices.

    Two triangles are in the same component when a chain of triangles between them each
    share a vertex with the next. For the image-plane triangles of a lensed source, each
    component is one multiple image.

    Parameters
    ----------
    triangles
        An ``AbstractTriangles`` (or anything with a ``triangles`` ``(n, 3, 2)`` array).

    Returns
    -------
    One ``_TriangleComponent`` per component.
    """
    array = np.asarray(triangles.triangles)

    array = array[~np.isnan(array).any(axis=(1, 2))]

    if array.shape[0] == 0:
        return []

    labels = _vertex_labels_from(array)

    total_triangles = labels.shape[0]

    parent = np.arange(total_triangles)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int):
        left, right = find(left), find(right)
        if left != right:
            parent[left] = right

    # The first triangle seen at each vertex label is unioned with every later triangle
    # sharing it, which connects the whole star of triangles around that vertex.
    first_at_vertex = {}

    for triangle_index, row in enumerate(labels):
        for label in row:
            label = int(label)
            if label in first_at_vertex:
                union(first_at_vertex[label], triangle_index)
            else:
                first_at_vertex[label] = triangle_index

    roots = np.array([find(index) for index in range(total_triangles)])

    return [
        _TriangleComponent(triangles=array[roots == root]) for root in np.unique(roots)
    ]


class _TriangleComponent:
    def __init__(self, triangles: np.ndarray):
        """
        One connected component of a set of triangles.

        Exposes the same ``triangles`` / ``area`` interface as ``AbstractTriangles`` so a
        component can be measured the same way the whole kept set is, without depending on
        which triangle container the solve produced.

        Parameters
        ----------
        triangles
            An ``(n, 3, 2)`` array of the component's triangle vertices.
        """
        self.triangles = triangles

    @property
    def area(self) -> float:
        """
        The total area covered by the component's triangles.
        """
        triangles = self.triangles

        return float(
            0.5
            * np.abs(
                (triangles[:, 0, 0] * (triangles[:, 1, 1] - triangles[:, 2, 1]))
                + (triangles[:, 1, 0] * (triangles[:, 2, 1] - triangles[:, 0, 1]))
                + (triangles[:, 2, 0] * (triangles[:, 0, 1] - triangles[:, 1, 1]))
            ).sum()
        )


def _image_region_from_triangles(
    triangles: np.ndarray, mask: aa.Mask2D
) -> Optional[aa.ImageRegion]:
    """
    The ``ImageRegion`` covered by a component of image-plane triangles.

    Every pixel of `mask` containing a triangle vertex or centroid is in the region. The
    region's boundary contours are the pixel-edge loops of that coverage (the same outline
    engine A draws for a pixelized source), so a ring is drawn as an annulus rather than as
    the filled disc a convex hull would give.

    Parameters
    ----------
    triangles
        An ``(n, 3, 2)`` array of ``(y, x)`` image-plane triangle vertices.
    mask
        The ``Mask2D`` of the image-plane grid, whose ``False`` entries are the data pixels.

    Returns
    -------
    The region, or ``None`` when no unmasked pixel of `mask` is covered.
    """
    from autoarray.inversion.mappings.mapping import contours_from_bool_native
    from autoarray.mask.mask_2d import Mask2D

    points = np.concatenate([triangles.reshape(-1, 2), triangles.mean(axis=1)], axis=0)

    mask_arr = np.asarray(mask)

    geometry = mask.geometry

    rows = np.floor(
        (geometry.scaled_maxima[0] - points[:, 0]) / geometry.pixel_scales[0]
    ).astype(int)
    columns = np.floor(
        (points[:, 1] - geometry.scaled_minima[1]) / geometry.pixel_scales[1]
    ).astype(int)

    inside = (
        (rows >= 0)
        & (rows < mask_arr.shape[0])
        & (columns >= 0)
        & (columns < mask_arr.shape[1])
    )

    region_native = np.zeros(shape=mask_arr.shape, dtype=bool)
    region_native[rows[inside], columns[inside]] = True
    region_native &= ~mask_arr

    if not region_native.any():
        return None

    slim_index_native = np.full(shape=mask_arr.shape, fill_value=-1, dtype=int)
    slim_index_native[~mask_arr] = np.arange(int(np.sum(~mask_arr)))

    centroids = triangles.mean(axis=1)

    return aa.ImageRegion(
        slim_indexes=np.sort(slim_index_native[region_native]),
        mask=Mask2D(
            mask=~region_native,
            pixel_scales=mask.pixel_scales,
            origin=mask.origin,
        ),
        contours=contours_from_bool_native(
            bool_native=region_native, geometry=geometry
        ),
        # The mean of the triangle centroids, not of the covered pixel centres: the
        # triangles resolve the image far below the pixel scale, so this stays meaningful
        # for an image which covers only one or two pixels.
        centre=(float(np.mean(centroids[:, 0])), float(np.mean(centroids[:, 1]))),
    )
