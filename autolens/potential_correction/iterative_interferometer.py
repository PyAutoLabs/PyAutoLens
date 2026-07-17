"""
The visibility-space iterative solver of the gravitational-imaging
(potential correction) technique: ``IterFitDpsiSrcInterferometer`` jointly
optimizes the pixelized source and potential corrections dpsi against an
`Interferometer` dataset with a Levenberg-Marquardt loop, re-ray-tracing the
real-space grid through the corrected lens each accepted step.

Scaling design (the sparse-operator / w-tilde route): the LM loop runs
entirely in the real-space normal-equation space — the cost, gradient and
Hessian of the penalized objective are expressed through the joint curvature
``F = A^T (T^H C^-1 T) A`` (built via the dataset's
``InterferometerSparseOperator`` FFT machinery), the data vector
``D = A^T T^H C^-1 d`` (the dirty image) and the precomputed scalar
``d^H C^-1 d``:

    P(x) = 0.5 (d^H C^-1 d - 2 x^T D + x^T F x) + 0.5 x^T R x

so no per-candidate NUFFT is required and cost scales with real-space mask
pixels, independent of the visibility count.

Ported and extended from the ``potential_correction`` package of Cao et al.
2025 (https://github.com/caoxiaoyue/lensing_potential_correction). If you use
this functionality in your research, please cite Cao et al. 2025; citation
materials are provided at
https://github.com/caoxiaoyue/potential_correction_paper.
"""

import logging
from typing import Optional

import numpy as np

import autofit as af
import autoarray as aa
from autoarray import exc
from autoarray.inversion.mappers import mapper_util

from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.profiles.mass.input.input_potential import InputPotential
from autogalaxy.analysis.adapt_images.adapt_images import AdaptImages

from autolens.lens.tracer import Tracer
from autolens.interferometer.fit_interferometer import FitInterferometer
from autolens.potential_correction import dense_util
from autolens.potential_correction import util as pc_util
from autolens.potential_correction.pixelization import (
    DpsiLinearObj,
    DpsiPixelization,
    DpsiSrcPixelization,
)
from autolens.potential_correction.fit_interferometer import (
    _subset_rows_in_full_from,
)
from autolens.potential_correction.src_factory import PixSrcFactoryITP

logger = logging.getLogger(__name__)


class IterFitDpsiSrcInterferometer:
    def __init__(
        self,
        dataset,
        lens_start: Galaxy,
        dpsi_pixelization: DpsiPixelization,
        src_pixelization: aa.Pixelization,
        gauge_constraints: bool = False,
        src_image_mesh=None,
        dpsi_mask=None,
        settings_inversion: Optional[aa.Settings] = None,
        preloads: Optional[dict] = None,
        n_iter: int = 20,
        tol: float = 1e-6,
        reg_optimize_every: Optional[int] = None,
        reg_optimize_grid: int = 5,
        verbose: bool = False,
    ):
        """
        Iteratively solves for the pixelized source s and potential
        corrections dpsi of an interferometer dataset, minimizing the
        penalized visibility objective with a Levenberg-Marquardt loop on
        the combined state x = [s | dpsi]. Each accepted step re-ray-traces
        the real-space grid through the corrected lens (the macro model plus
        an ``InputPotential`` built from the current dpsi), so the
        corrections feed back into the source mapping.

        Requires the dataset's sparse operator
        (``dataset.apply_sparse_operator()``) — the loop runs through the
        w-tilde normal-equation space and never materializes
        visibility-space matrices.

        Parameters
        ----------
        dataset
            The ``al.Interferometer`` dataset carrying a sparse operator.
        lens_start
            The macro lens galaxy the corrections perturb.
        dpsi_pixelization
            The dpsi mesh + regularization model, on the real-space mask.
        src_pixelization
            The source pixelization.
        gauge_constraints
            Whether to impose <dpsi, 1> = <dpsi, x> = <dpsi, y> = 0 via an
            equality-constrained KKT step.
        src_image_mesh
            An image mesh whose image-plane mesh grid is preloaded into the
            source inversion.
        dpsi_mask
            An optional 2D bool mask restricting the dpsi mesh to a sub-region
            of the real-space mask (typically an arc-tracing mask from
            ``al.pc.util.arc_mask_from``): the corrections are only
            constrained where the lensed arcs are, and restricting the mesh
            there stabilises the inversion. Defaults to the full real-space
            mask.
        settings_inversion
            The inversion settings; defaults to the border relocator without
            the positive-only solver (the LM state is signed).
        preloads
            Precomputed attributes set directly onto the fit.
        n_iter
            The maximum number of outer LM iterations.
        tol
            The step-norm convergence tolerance.
        reg_optimize_every
            When set, every N accepted outer iterations (and once at the
            start) the regularization strength multipliers (a_src, a_dpsi)
            are re-optimized by maximising the Laplace evidence at the
            current normal equations — the per-iteration objective control
            of Koopmans 2005 / Vegetti & Koopmans 2009 that prevents the
            source block absorbing potential perturbations. Each candidate
            costs one Cholesky solve at fixed F, D.
        reg_optimize_grid
            The number of log-spaced multipliers per axis of the
            re-optimization grid (spanning 1e-2..1e2 around the current
            scales).
        verbose
            Whether to log per-iteration costs.
        """
        self.dataset = dataset
        self.lens_start = lens_start
        self.dpsi_pixelization = dpsi_pixelization
        self.src_pixelization = src_pixelization
        self.gauge_constraints = gauge_constraints
        self.src_image_mesh = src_image_mesh
        self.dpsi_mask = dpsi_mask
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.reg_optimize_every = reg_optimize_every
        self.reg_optimize_grid = int(reg_optimize_grid)
        self.reg_scales = (1.0, 1.0)
        self.verbose = bool(verbose)

        if settings_inversion is None:
            self.settings_inversion = aa.Settings(
                use_positive_only_solver=False,
                use_border_relocator=True,
            )
        else:
            self.settings_inversion = settings_inversion

        if getattr(dataset, "sparse_operator", None) is None:
            raise exc.InversionException(
                "IterFitDpsiSrcInterferometer requires the dataset's sparse "
                "operator: call dataset.apply_sparse_operator() first."
            )

        if preloads is not None:
            for key, value in preloads.items():
                setattr(self, key, value)

    # -----------------------------
    # Static (state-independent) pieces
    # -----------------------------

    @property
    def pair_dpsi_data_obj(self):
        if not hasattr(self, "_pair_dpsi_data_obj"):
            mask = (
                np.asarray(self.dpsi_mask)
                if self.dpsi_mask is not None
                else np.asarray(self.dataset.real_space_mask)
            )
            self._pair_dpsi_data_obj = self.dpsi_pixelization.pair_dpsi_data_mesh(
                mask,
                self.dataset.real_space_mask.pixel_scales[0],
            )
        return self._pair_dpsi_data_obj

    @property
    def dpsi_gradient_matrix(self):
        if not hasattr(self, "dpsi_grad_mat"):
            self.dpsi_grad_mat = pc_util.dpsi_gradient_matrix_from(
                self.pair_dpsi_data_obj.itp_mat,
                self.pair_dpsi_data_obj.Hx_dpsi,
                self.pair_dpsi_data_obj.Hy_dpsi,
            )
        return self.dpsi_grad_mat

    @property
    def dpsi_points(self):
        if not hasattr(self, "_dpsi_points"):
            self._dpsi_points = np.vstack(
                [
                    self.pair_dpsi_data_obj.ygrid_dpsi_1d,
                    self.pair_dpsi_data_obj.xgrid_dpsi_1d,
                ]
            ).T
        return self._dpsi_points

    @property
    def dpsi_regularization_matrix(self):
        if not hasattr(self, "dpsi_reg_mat"):
            linear_obj = DpsiLinearObj(
                mask=self.pair_dpsi_data_obj.mask_dpsi, points=self.dpsi_points
            )
            self.dpsi_reg_mat = (
                self.dpsi_pixelization.regularization.regularization_matrix_from(
                    linear_obj=linear_obj
                )
            )
        return self.dpsi_reg_mat

    @property
    def data_weighted_norm(self) -> float:
        """The scalar d^H C^-1 d over real and imaginary components."""
        if not hasattr(self, "_data_weighted_norm"):
            data = np.asarray(self.dataset.data)
            noise = np.asarray(self.dataset.noise_map)
            self._data_weighted_norm = float(
                np.sum((data.real / noise.real) ** 2)
                + np.sum((data.imag / noise.imag) ** 2)
            )
        return self._data_weighted_norm

    @property
    def _extent_index(self):
        if not hasattr(self, "_extent_index_cached"):
            self._extent_index_cached = np.asarray(
                self.dataset.real_space_mask.extent_index_for_masked_pixel
            )
        return self._extent_index_cached

    # -----------------------------
    # Per-state constructions
    # -----------------------------

    def _updated_lens_galaxies_from_dpsi(self, dpsi_vec: np.ndarray):
        mask_dpsi_aa = self.pair_dpsi_data_obj.mask_dpsi_aa
        grid_dpsi = aa.Grid2D.from_mask(mask=mask_dpsi_aa)

        # zero-fill extrapolation: the correction vanishes outside its mesh
        # (nearest extrapolation would smear constant deflections across the
        # full re-trace grid when the dpsi mesh is a sub-region of it)
        pix_mass_profile = InputPotential(
            lensing_potential=dpsi_vec,
            image_plane_grid=np.asarray(grid_dpsi),
            mask=mask_dpsi_aa,
            extrapolate="zero",
        )

        lens_macro = self.lens_start
        lens_pix = Galaxy(
            redshift=getattr(lens_macro, "redshift", 0.2), mass=pix_mass_profile
        )
        return [lens_macro, lens_pix]

    def _build_pix_src_tracer(self, lens_galaxies):
        source_galaxy = Galaxy(redshift=1.0, pixelization=self.src_pixelization)

        adapt_images = None
        if self.src_image_mesh is not None:
            self.image_plane_mesh_grid = self.src_image_mesh.image_plane_mesh_grid_from(
                mask=self.dataset.real_space_mask
            )
            adapt_images = AdaptImages(
                galaxy_image_plane_mesh_grid_dict={
                    source_galaxy: self.image_plane_mesh_grid
                }
            )

        tracer = Tracer(galaxies=[*lens_galaxies, source_galaxy])
        return tracer, adapt_images

    def _init_source_inversion(self, lens_galaxies):
        tracer, adapt_images = self._build_pix_src_tracer(lens_galaxies)

        src_fit = FitInterferometer(
            dataset=self.dataset,
            tracer=tracer,
            adapt_images=adapt_images,
            settings=self.settings_inversion,
        )

        self.tracer = tracer
        mapper = src_fit.inversion.linear_obj_list[0]
        self._mapper = mapper
        self.source_plane_mesh_grid = mapper.source_plane_mesh_grid
        self.image_plane_mesh_grid = mapper.image_plane_mesh_grid
        self._src_mesh = mapper.mesh
        self._border_relocator = (
            self.dataset.grids.border_relocator
            if self.settings_inversion.use_border_relocator
            else None
        )
        self._src_regularization = self.src_pixelization.regularization
        self.src_reg_mat = src_fit.inversion.regularization_matrix

    def _update_source_mapper(self, lens_galaxies):
        """
        Re-ray-traces the real-space grid through the updated lens, keeping
        the source mesh fixed, and rebuilds the (untransformed) mapper.
        """
        tracer, _ = self._build_pix_src_tracer(lens_galaxies)
        self.tracer = tracer

        source_plane_data_grid = tracer.traced_grid_2d_list_from(
            grid=self.dataset.grid
        )[-1]

        interpolator = self._src_mesh.interpolator_from(
            source_plane_data_grid=source_plane_data_grid,
            source_plane_mesh_grid=self.source_plane_mesh_grid,
            border_relocator=self._border_relocator,
            adapt_data=None,
        )

        self._mapper = aa.Mapper(
            interpolator=interpolator,
            regularization=self._src_regularization,
            settings=self.settings_inversion,
            image_plane_mesh_grid=self.image_plane_mesh_grid,
        )

    @property
    def _dpsi_rows_in_full(self):
        if not hasattr(self, "_dpsi_rows_in_full_cached"):
            self._dpsi_rows_in_full_cached = _subset_rows_in_full_from(
                np.asarray(self.dataset.real_space_mask),
                self.pair_dpsi_data_obj.mask_data,
            )
        return self._dpsi_rows_in_full_cached

    def _dpsi_response_matrix(self, s: np.ndarray):
        """
        The sparse real-space correction response G = -D_s D_psi at the
        current source state (gradients of the pixelized source
        reconstruction, evaluated at the current ray-traced positions),
        scattered into the full real-space row space when the dpsi mesh is
        arc-restricted.
        """
        from scipy.sparse import coo_matrix as _coo

        source_factory = PixSrcFactoryITP(
            points=self.source_plane_mesh_grid, values=np.asarray(s)
        )
        traced = np.asarray(
            self.dataset.grid.slim
            - self.tracer.deflections_yx_2d_from(self.dataset.grid.slim)
        )[self._dpsi_rows_in_full]
        source_gradients = source_factory.eval_grad(traced[:, 1], traced[:, 0])
        src_grad_mat = pc_util.source_gradient_matrix_from(source_gradients)
        G_sub = (-1.0 * src_grad_mat @ self.dpsi_gradient_matrix).tocoo()
        n_full = int(np.count_nonzero(~np.asarray(self.dataset.real_space_mask)))
        return _coo(
            (G_sub.data, (self._dpsi_rows_in_full[G_sub.row], G_sub.col)),
            shape=(n_full, G_sub.shape[1]),
        )

    def _normal_equations_from(self, s, dpsi):
        """
        The joint curvature F, data vector D and dense real-space response
        [f | G] at the state (s, dpsi), after updating the lens and mapper.
        """
        lens_galaxies = self._updated_lens_galaxies_from_dpsi(np.asarray(dpsi))
        self._update_source_mapper(lens_galaxies)
        mapper = self._mapper

        rows_src, cols_src, vals_src = mapper_util.sparse_triplets_from(
            pix_indexes_for_sub=mapper.pix_indexes_for_sub_slim_index,
            pix_weights_for_sub=mapper.pix_weights_for_sub_slim_index,
            slim_index_for_sub=mapper.slim_index_for_sub_slim_index,
            fft_index_for_masked_pixel=self._extent_index,
            sub_fraction_slim=mapper.over_sampler.sub_fraction.array,
            return_rows_slim=False,
        )

        G = self._dpsi_response_matrix(s)
        n_src = mapper.params
        rows = np.concatenate([np.asarray(rows_src), self._extent_index[G.row]])
        cols = np.concatenate([np.asarray(cols_src), G.col + n_src])
        vals = np.concatenate([np.asarray(vals_src), G.data])

        n_dpsi = self.dpsi_regularization_matrix.shape[0]
        F = np.asarray(
            self.dataset.sparse_operator.curvature_matrix_diag_from(
                rows=rows, cols=cols, vals=vals, S=n_src + n_dpsi
            )
        )

        A = np.hstack(
            [np.asarray(mapper.mapping_matrix), np.asarray(G.todense())]
        )
        D = np.asarray(A.T @ np.asarray(self.dataset.sparse_operator.dirty_image))

        return F, D, A

    def _regularization_matrix(self):
        return dense_util.dense_block_diag_from(
            self.src_reg_mat, self.dpsi_regularization_matrix
        )

    def _cost_from(self, x, F, D, R):
        """
        The penalized objective. The chi^2 uses the SOURCE block only
        (model = T f(psi0 + dpsi) s): the state's dpsi enters the model
        through the re-traced lens, so including the linear G dpsi term as
        well would double-count the correction (the linearization G belongs
        to the gradient/Hessian, which model the next increment). Through
        the normal-equation identity:
        0.5 (d^H C^-1 d - 2 s^T D_s + s^T F_ss s) + 0.5 x^T R x.
        """
        n_s = self.src_reg_mat.shape[0]
        s_vec = x[:n_s]
        chi2 = 0.5 * (
            self.data_weighted_norm
            - 2.0 * float(s_vec @ D[:n_s])
            + float(s_vec @ (F[:n_s, :n_s] @ s_vec))
        )
        reg = 0.5 * float(x @ (R @ x))
        return chi2 + reg, chi2, reg

    def _effective_regularization(self):
        """The block-diagonal regularization at the current strength scales."""
        a_s, a_d = self.reg_scales
        return dense_util.dense_block_diag_from(
            a_s * dense_util.as_dense(self.src_reg_mat),
            a_d * np.asarray(self.dpsi_regularization_matrix),
        )

    def _optimize_regularization(self, F, D):
        """
        Re-optimizes the regularization strength multipliers (a_src, a_dpsi)
        by maximising the Laplace evidence at the current (fixed) normal
        equations F, D — the objective control of Koopmans 2005 / Vegetti &
        Koopmans 2009. Returns the new scales and the solution under them.
        """
        n_s = self.src_reg_mat.shape[0]
        R_s = dense_util.as_dense(self.src_reg_mat)
        R_d = np.asarray(self.dpsi_regularization_matrix)
        n_d = R_d.shape[0]
        logdet_Rs = pc_util.log_det_mat(R_s, sparse=False)
        try:
            logdet_Rd = pc_util.log_det_mat(R_d, sparse=False)
        except np.linalg.LinAlgError:
            logdet_Rd = pc_util.log_det_mat(R_d, sparse=True)

        candidates = np.logspace(-2.0, 2.0, self.reg_optimize_grid)
        best = None
        for a_s in candidates * self.reg_scales[0]:
            for a_d in candidates * self.reg_scales[1]:
                R = dense_util.dense_block_diag_from(a_s * R_s, a_d * R_d)
                curve = F + R
                try:
                    L_chol = np.linalg.cholesky(curve)
                except np.linalg.LinAlgError:
                    continue
                from scipy.linalg import cho_solve

                x = cho_solve((L_chol, True), D)
                logdet_curve = 2.0 * float(np.sum(np.log(np.diag(L_chol))))
                chi2 = (
                    self.data_weighted_norm
                    - 2.0 * float(x[:n_s] @ D[:n_s])
                    + float(x[:n_s] @ (F[:n_s, :n_s] @ x[:n_s]))
                )
                reg_pen = float(x @ (R @ x))
                evidence = 0.5 * (
                    (n_s * np.log(a_s) + logdet_Rs)
                    + (n_d * np.log(a_d) + logdet_Rd)
                    - logdet_curve
                    - chi2
                    - reg_pen
                )
                if best is None or evidence > best[0]:
                    best = (evidence, a_s, a_d, x)
        if best is None:
            raise exc.InversionException(
                "Regularization re-optimization found no positive-definite "
                "candidate."
            )
        return best

    # -----------------------------
    # The Levenberg-Marquardt loop
    # -----------------------------

    def solve_joint_optimization(self, x0=None):
        """
        Runs the Levenberg-Marquardt loop on the combined state
        x = [s | dpsi] in the real-space normal-equation space, accepting
        steps that decrease the penalized cost. Returns the optimized
        (s, dpsi); also stored as ``s_opt`` / ``dpsi_opt``.
        """
        n_dpsi = self.dpsi_regularization_matrix.shape[0]
        lens_galaxies = self._updated_lens_galaxies_from_dpsi(np.zeros(n_dpsi))
        self._init_source_inversion(lens_galaxies)
        n_s = self.src_reg_mat.shape[0]

        x = np.zeros(n_s + n_dpsi) if x0 is None else np.asarray(x0, dtype=float).copy()
        if x.shape != (n_s + n_dpsi,):
            raise ValueError(
                f"x0 has shape {x.shape}; expected ({n_s + n_dpsi},)"
            )
        R = np.asarray(self._effective_regularization())

        constraint_matrix = None
        if self.gauge_constraints:
            G_c = np.zeros((3, n_dpsi))
            G_c[0, :] = 1.0 / n_dpsi
            G_c[1, :] = self.dpsi_points[:, 1] / n_dpsi
            G_c[2, :] = self.dpsi_points[:, 0] / n_dpsi
            constraint_matrix = np.hstack([np.zeros((3, n_s)), G_c])

        F, D, A = self._normal_equations_from(x[:n_s], x[n_s:])

        if self.reg_optimize_every is not None:
            ev, a_s, a_d, x_star = self._optimize_regularization(F, D)
            self.reg_scales = (a_s, a_d)
            R = np.asarray(self._effective_regularization())
            # the linear solve's dpsi block is an increment on the lens state
            x_new = np.asarray(x_star).copy()
            x_new[n_s:] = x[n_s:] + x_star[n_s:]
            x = x_new
            F, D, A = self._normal_equations_from(x[:n_s], x[n_s:])
            if self.verbose:
                logger.info(
                    "reg re-optimization: a_src=%.3e a_dpsi=%.3e evidence=%.4e",
                    a_s, a_d, ev,
                )

        current_cost, chi2, reg = self._cost_from(x, F, D, R)

        mu = 1.0

        if self.verbose:
            logger.info(
                "Starting joint LM optimization (interferometer, sparse route): "
                "%d iterations, initial cost %.4e",
                self.n_iter,
                current_cost,
            )

        for i in range(self.n_iter):
            H = F + R
            # Gauss-Newton gradient: the residual is d - T f(psi0+dpsi) s —
            # the state's dpsi lives in the re-traced lens, so the model term
            # uses the source block only; J^T C^-1 r = D - F @ [s | 0].
            x_model = np.zeros_like(x)
            x_model[:n_s] = x[:n_s]
            minus_gradient = D - F @ x_model - R @ x

            if self.verbose:
                logger.info(
                    "Iter %d: cost=%.4e chi2=%.4e reg=%.4e mu=%.1e",
                    i, current_cost, chi2, reg, mu,
                )

            step_accepted = False
            while not step_accepted:
                delta_x = None
                try:
                    delta_x = dense_util.solve_lm_step_from(
                        H, minus_gradient, mu,
                        constraint_matrix=constraint_matrix, x=x,
                    )
                    if np.any(np.isnan(np.asarray(delta_x))):
                        delta_x = None
                except np.linalg.LinAlgError:
                    delta_x = None

                if delta_x is not None:
                    x_new = x + delta_x
                    F_new, D_new, A_new = self._normal_equations_from(
                        x_new[:n_s], x_new[n_s:]
                    )
                    cost_new, chi2_new, reg_new = self._cost_from(
                        x_new, F_new, D_new, R
                    )

                    if cost_new < current_cost:
                        x, F, D, A = x_new, F_new, D_new, A_new
                        current_cost, chi2, reg = cost_new, chi2_new, reg_new
                        mu = max(1e-15, mu / 3.0)
                        step_accepted = True

                        if (
                            self.reg_optimize_every is not None
                            and (i + 1) % self.reg_optimize_every == 0
                        ):
                            ev, a_s, a_d, x_star = self._optimize_regularization(F, D)
                            self.reg_scales = (a_s, a_d)
                            R = np.asarray(self._effective_regularization())
                            x_new = np.asarray(x_star).copy()
                            x_new[n_s:] = x[n_s:] + x_star[n_s:]
                            x = x_new
                            F, D, A = self._normal_equations_from(x[:n_s], x[n_s:])
                            current_cost, chi2, reg = self._cost_from(x, F, D, R)
                            if self.verbose:
                                logger.info(
                                    "reg re-optimization @iter %d: a_src=%.3e "
                                    "a_dpsi=%.3e evidence=%.4e cost=%.4e",
                                    i, a_s, a_d, ev, current_cost,
                                )

                        if float(np.linalg.norm(delta_x)) < self.tol:
                            if self.verbose:
                                logger.info(
                                    "Converged at iteration %d (step tolerance).",
                                    i,
                                )
                            self.s_opt = x[:n_s]
                            self.dpsi_opt = x[n_s:]
                            self._final_state = (F, D, A, R)
                            return self.s_opt, self.dpsi_opt
                    else:
                        mu *= 5.0
                        if mu > 1e15:
                            logger.warning(
                                "LM damping parameter exceeded 1e15; stopping."
                            )
                            break
                else:
                    mu *= 5.0
                    if mu > 1e15:
                        logger.warning("LM solver failed repeatedly; stopping.")
                        break

            if not step_accepted:
                break

        self.s_opt = x[:n_s]
        self.dpsi_opt = x[n_s:]
        self._final_state = (F, D, A, R)
        return self.s_opt, self.dpsi_opt

    def log_evidence(
        self,
        s: Optional[np.ndarray] = None,
        dpsi: Optional[np.ndarray] = None,
        include_noise_normalization: bool = True,
    ) -> float:
        """
        The Laplace-approximation log evidence at (s, dpsi) (defaulting to
        the optimized state): 0.5 [ logdet R_s + logdet R_dpsi
        - logdet(F + R) - chi^2 - x^T R x ] (+ the complex-noise
        normalization), with chi^2 through the normal-equation identity.
        """
        if s is None or dpsi is None:
            if hasattr(self, "s_opt") and hasattr(self, "dpsi_opt"):
                s = self.s_opt
                dpsi = self.dpsi_opt
            else:
                raise ValueError(
                    "Provide s and dpsi, or run solve_joint_optimization() first."
                )

        x = np.concatenate([np.asarray(s), np.asarray(dpsi)])
        F, D, A = self._normal_equations_from(s, dpsi)
        R = np.asarray(self._effective_regularization())

        n_s_split = self.src_reg_mat.shape[0]
        s_vec = x[:n_s_split]
        chi2 = (
            self.data_weighted_norm
            - 2.0 * float(s_vec @ D[:n_s_split])
            + float(s_vec @ (F[:n_s_split, :n_s_split] @ s_vec))
        )
        reg_val = float(x @ (R @ x))

        a_s, a_d = self.reg_scales
        n_src = self.src_reg_mat.shape[0]
        n_dpsi = np.asarray(self.dpsi_regularization_matrix).shape[0]
        logdet_Rs = (
            pc_util.log_det_mat(self.src_reg_mat, sparse=True)
            + n_src * np.log(a_s)
        )
        try:
            sign, logdet_Rd = np.linalg.slogdet(
                np.asarray(self.dpsi_regularization_matrix)
            )
            if sign != 1:
                raise np.linalg.LinAlgError(
                    "The dpsi regularization matrix is not positive definite."
                )
        except (np.linalg.LinAlgError, TypeError, ValueError):
            logdet_Rd = pc_util.log_det_mat(
                self.dpsi_regularization_matrix, sparse=True
            )
        logdet_Rd = logdet_Rd + n_dpsi * np.log(a_d)

        sign, logdet_H = np.linalg.slogdet(F + R)
        if sign != 1:
            raise np.linalg.LinAlgError(
                "The curvature+regularization matrix is not positive definite."
            )

        log_evidence = 0.5 * (logdet_Rs + logdet_Rd - logdet_H - chi2 - reg_val)

        if include_noise_normalization:
            noise = np.asarray(self.dataset.noise_map)
            log_evidence += -0.5 * float(
                aa.util.fit.noise_normalization_complex_from(noise_map=noise)
            )

        return float(log_evidence)


class IterDpsiSrcInvInterferometerAnalysis(af.Analysis):
    def __init__(
        self,
        dataset,
        lens_start: Galaxy,
        n_iter: int = 20,
        tol: float = 1e-6,
        src_image_mesh=None,
        settings_inversion: Optional[aa.Settings] = None,
        preloads: Optional[dict] = None,
        gauge_constraints: bool = True,
        verbose: bool = False,
    ):
        """
        Samples the joint source+dpsi pixelization hyper-parameters of the
        visibility-space iterative LM solver, with its Laplace evidence as
        the likelihood.
        """
        self.dataset = dataset
        self.lens_start = lens_start
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.src_image_mesh = src_image_mesh
        self.gauge_constraints = bool(gauge_constraints)
        self.verbose = bool(verbose)
        if settings_inversion is None:
            self.settings_inversion = aa.Settings(
                use_positive_only_solver=False,
                use_border_relocator=True,
            )
        else:
            self.settings_inversion = settings_inversion
        self.preloads = preloads

    def log_likelihood_function(self, instance: DpsiSrcPixelization):
        fit = IterFitDpsiSrcInterferometer(
            dataset=self.dataset,
            lens_start=self.lens_start,
            dpsi_pixelization=instance.dpsi_pixelization,
            src_pixelization=instance.src_pixelization,
            src_image_mesh=self.src_image_mesh,
            gauge_constraints=self.gauge_constraints,
            n_iter=self.n_iter,
            tol=self.tol,
            verbose=self.verbose,
            settings_inversion=self.settings_inversion,
            preloads=self.preloads,
        )
        try:
            s_opt, dpsi_opt = fit.solve_joint_optimization()
            return fit.log_evidence(s=s_opt, dpsi=dpsi_opt)
        except exc.InversionException:
            # a failed inversion is a valid (very bad) sample, not a crash
            logger.exception(
                "InversionException during visibility-space iterative "
                "source+dpsi optimization; returning penalty likelihood."
            )
            return -1e30
