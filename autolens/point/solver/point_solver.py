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
import os
from typing import Tuple, Optional

import numpy as np
import autoarray as aa
from autoarray.structures.triangles.shape import Point

from autolens.lens.tracer import Tracer
from .shape_solver import AbstractSolver


logger = logging.getLogger(__name__)

# One-shot latch for the ``PYAUTO_SMALL_DATASETS`` short-circuit warning in
# ``PointSolver.solve``. Module-level rather than per-instance: a vmap batch calls
# ``solve`` once per sampled parameter set, and the message is about the process
# environment, not about any one solver.
_SMALL_DATASETS_WARNED = False


class PointSolver(AbstractSolver):

    def solve(
        self,
        tracer: Tracer,
        source_plane_coordinate: Tuple[float, float],
        xp=None,
        plane_redshift: Optional[float] = None,
        remove_infinities: Optional[bool] = None,
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
        tracer
            The tracer that traces the image plane coordinates to the source plane.
        source_plane_coordinate
            The plane coordinate to trace to the image plane, which by default in the source-plane coordinate
            but could be a coordinate in another plane is `plane_redshift` is input.
        xp
            The array module (``numpy`` or ``jax.numpy``) the solve runs in. ``AnalysisPoint``
            passes ``jax.numpy`` when ``use_jax=True`` is set on the analysis. When ``None`` (the
            default), falls back to ``self._xp`` — which is ``jnp`` if the solver was constructed
            with ``use_jax=True`` and ``np`` otherwise. Pass explicitly to override.
        plane_redshift
            The redshift of the plane coordinate, which for multi-plane systems may not be the source-plane.
        remove_infinities
            Whether to strip the ``inf`` sentinel rows from the output. When ``None`` (the default),
            defaults to ``True`` on the NumPy path and ``False`` on the JAX path. The JAX path
            keeps the padded static shape so the output crosses a ``jax.jit`` boundary cleanly;
            strip the infinities outside the jit if needed. The default follows the effective
            ``xp`` for this call, including an explicit override of the constructor backend.
            An explicit ``True`` or ``False`` takes precedence over this default.

        Returns
        -------
        A ``Grid2DIrregular`` of image-plane coordinates. NumPy-backed on the default path,
        ``jax.Array``-backed when ``use_jax=True`` (or ``xp=jnp``).

        Notes
        -----
        Gradients (JAX path): the returned positions carry a ``jax.custom_jvp`` implicit
        fixed-point rule (``dθ = A⁻¹ (dα + dβ)`` at the solved positions — the gravity.jl
        / Lombardi 2024 Eq. 30 mechanism), so ``jax.grad`` through a solver-chained
        likelihood is exact between image-count events instead of identically zero. See
        ``autolens.point.solver.implicit_diff`` for the differentiability contract
        (staircase quantization, caustic-crossing seams, near-critical divergence).

        Smoke-test short-circuit (``PYAUTO_SMALL_DATASETS``): the triangle-tiling solve
        is the dominant cost in many simulator scripts and is meaningless on the
        downsized grids used for fast smoke tests. When ``PYAUTO_SMALL_DATASETS=1`` is
        set the solver returns the fixed pair ``[(1.0, 0.0), (0.0, 1.0)]`` immediately,
        skipping ``solve_triangles`` entirely. The two coordinates are well separated
        so any downstream ``positions_likelihood_from`` / threshold calculation behaves
        normally. ``PYAUTO_SMALL_DATASETS`` is a smoke-test-only flag and is never set
        inside a ``jax.jit`` trace, so a plain numpy-backed ``Grid2DIrregular`` is safe
        here even when the surrounding analysis uses ``xp=jnp``.

        The short-circuit announces itself with a ``logger.warning`` the first time it
        fires in a process. It returns the same two coordinates for every lens model, so
        anything derived from them — a likelihood, a chi-squared, a position pairing — is
        model-independent, and a parity script that compares such a value against a pinned
        literal is measuring nothing. That failure mode was silent until it cost a real
        investigation (PyAutoLens#710), hence the warning rather than a bare return.
        """
        if xp is None:
            xp = self._xp

        if remove_infinities is None:
            remove_infinities = xp is np

        # NOTE: pytree registration is the user's responsibility (call
        # `autolens.jax.register_tracer_classes(tracer)` once before wrapping
        # in @jax.jit). Auto-registering inside solve() doesn't help because
        # JAX flattens function arguments at trace time — before entering
        # this method — so registration must run before the first jitted
        # call. See the `lens_calc.py` workspace guide for the canonical
        # JIT-it-yourself pattern.

        if os.environ.get("PYAUTO_SMALL_DATASETS") == "1":
            global _SMALL_DATASETS_WARNED
            if not _SMALL_DATASETS_WARNED:
                _SMALL_DATASETS_WARNED = True
                logger.warning(
                    "PointSolver.solve is short-circuited: PYAUTO_SMALL_DATASETS=1 is set, "
                    "so the triangle-tiling solve is skipped and the fixed pair "
                    "[(1.0, 0.0), (0.0, 1.0)] is returned for EVERY lens model. Any "
                    "likelihood, chi-squared or position-pairing value computed from these "
                    "positions is independent of the model and must not be compared against "
                    "a pinned literal. Scripts that need a real solve should declare "
                    "`ENV: full_datasets` (workspace test-harness) or unset the flag. This "
                    "warning is issued once per process."
                )
            return aa.Grid2DIrregular(values=[(1.0, 0.0), (0.0, 1.0)])

        if xp is not np:
            from .implicit_diff import solve_padded_factory, tracer_is_jax_compatible

            if not tracer_is_jax_compatible(tracer):
                # Unregistered tracer (e.g. a simulator script's hand-built Tracer):
                # cannot cross the custom_jvp boundary. Plain forward solve, gradient
                # behaviour unchanged (identically zero, as before this rule existed).
                solution = self._solve_array(
                    tracer=tracer,
                    source_plane_coordinate=source_plane_coordinate,
                    xp=xp,
                    plane_redshift=plane_redshift,
                    remove_infinities=remove_infinities,
                )
                return aa.Grid2DIrregular(solution)

            plane_index = (
                -1
                if plane_redshift is None
                else tracer.plane_index_via_redshift_from(redshift=plane_redshift)
            )
            solve_padded = solve_padded_factory(
                solver=self,
                plane_redshift=plane_redshift,
                plane_index=plane_index,
                xp=xp,
            )
            beta = xp.stack(
                [
                    xp.asarray(source_plane_coordinate[0]),
                    xp.asarray(source_plane_coordinate[1]),
                ]
            )
            solution = solve_padded(tracer, beta)

            if remove_infinities:
                solution = solution[~xp.isinf(solution).any(axis=1)]

            return aa.Grid2DIrregular(solution)

        solution = self._solve_array(
            tracer=tracer,
            source_plane_coordinate=source_plane_coordinate,
            xp=xp,
            plane_redshift=plane_redshift,
            remove_infinities=remove_infinities,
        )

        # Warn on the *final* result rather than only on the empty branch inside
        # `_solve_array`, because there are two distinct routes to an empty answer:
        #
        #   1. no triangle contained the coordinate  -> `filtered_means` is already length 0
        #   2. every candidate failed the magnification threshold -> `_filter_low_magnification`
        #      preserves the length and writes NaN rows, which become `inf` and are then stripped
        #      by `remove_infinities`, landing here at length 0
        #
        # Only reachable on the NumPy path: the JAX path keeps its padded static shape, so
        # `len(solution)` is non-zero there and this reads no traced value.
        if len(solution) == 0:

            logger.warning(
                f"PointSolver.solve found no images for source-plane coordinate "
                f"{tuple(source_plane_coordinate)}, so an empty grid is returned. This means "
                f"either that the coordinate lies outside the region traced by the image-plane "
                f"grid, or that every candidate image was rejected by "
                f"`magnification_threshold` (currently {self.magnification_threshold})."
            )

        return aa.Grid2DIrregular(solution)

    def _solve_array(
        self,
        tracer: Tracer,
        source_plane_coordinate: Tuple[float, float],
        xp,
        plane_redshift: Optional[float] = None,
        remove_infinities: bool = False,
    ):
        """
        The raw solve: triangle refinement, magnification filtering and sentinel padding,
        returning the bare ``xp`` positions array (no ``Grid2DIrregular`` wrap, no logging).

        On the JAX path this is the primal of the ``custom_jvp`` wrapper built by
        ``implicit_diff.solve_padded_factory`` — gradients are supplied by the implicit
        fixed-point rule at the solved positions, never by differentiating through the
        refinement iteration (see ``implicit_diff``'s module docstring for the contract).
        """
        kept_triangles = super().solve_triangles(
            tracer=tracer,
            shape=Point(*source_plane_coordinate),
            xp=xp,
            plane_redshift=plane_redshift,
        )

        filtered_means = self._filter_low_magnification(
            tracer=tracer,
            points=kept_triangles.means,
            xp=xp,
            plane_redshift=plane_redshift,
        )

        # When no triangle traces to the source-plane coordinate -- e.g. the coordinate lies
        # outside the region the image-plane grid tiles -- `filtered_means` comes back with shape
        # (0, 2). The `[pair for pair in ...]` comprehension below then yields `[]`, and the
        # `Grid2DIrregular` built from it exposes `.array` as a bare list rather than a 2D array,
        # so the `axis=1` reductions raised
        # `numpy.exceptions.AxisError: axis 1 is out of bounds for array of dimension 1`.
        #
        # Zero images is a legitimate answer, not a failure, so build a correctly-shaped empty
        # grid instead.
        #
        # `_filter_low_magnification` returns an `xp` array (its `List[Tuple[float, float]]`
        # annotation is inaccurate), so `len(...)` is a leading-dimension query. Array shapes are
        # static under a `jax.jit` trace, so this reads no traced *value* and the branch is
        # trace-safe. In practice the JAX path builds its triangles at a padded fixed size
        # (`MAX_CONTAINING_SIZE`), so it is the NumPy path that reaches this.
        if len(filtered_means) == 0:

            solution = xp.zeros((0, 2))

        else:

            solution = aa.Grid2DIrregular(
                [pair for pair in filtered_means], xp=xp
            ).array

            is_nan = xp.isnan(solution).any(axis=1)
            sentinel = xp.full_like(solution[0], fill_value=xp.inf)
            solution = xp.where(is_nan[:, None], sentinel, solution)

            if remove_infinities:

                solution = solution[~xp.isinf(solution).any(axis=1)]

        return solution
