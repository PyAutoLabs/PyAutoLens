"""
The iterative solver of the gravitational-imaging (potential correction)
technique: ``IterFitDpsiSrcImaging`` jointly optimizes the pixelized source
and the pixelized potential corrections dpsi with a Levenberg-Marquardt
loop, re-ray-tracing the image grid through the corrected lens (the macro
model plus an ``InputPotential`` built from the current dpsi) at every step.

The linear algebra runs through the ``xp``-parameterized kernels of
``dense_util``: pass ``xp=jax.numpy`` to ``solve_joint_optimization`` /
``log_evidence`` for accelerator-ready execution, or leave the numpy default.

Ported from the ``potential_correction`` package of Cao et al. 2025
(https://github.com/caoxiaoyue/lensing_potential_correction). If you use this
functionality in your research, please cite Cao et al. 2025; citation
materials are provided at
https://github.com/caoxiaoyue/potential_correction_paper.
"""

import logging
from typing import Optional

import numpy as np

import autofit as af
import autoarray as aa
from autoarray import exc

from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.profiles.mass.input.input_potential import InputPotential
from autogalaxy.analysis.adapt_images.adapt_images import AdaptImages

from autolens.lens.tracer import Tracer
from autolens.imaging.fit_imaging import FitImaging
from autolens.potential_correction import dense_util
from autolens.potential_correction import util as pc_util
from autolens.potential_correction.pixelization import (
    DpsiLinearObj,
    DpsiPixelization,
    DpsiSrcPixelization,
)
from autolens.potential_correction.src_factory import PixSrcFactoryITP, SrcFactory

logger = logging.getLogger(__name__)


class IterFitDpsiSrcImaging:
    def __init__(
        self,
        masked_imaging,
        lens_start: Galaxy,
        dpsi_pixelization: DpsiPixelization,
        src_pixelization: aa.Pixelization,
        gauge_constraints: bool = False,
        adapt_image=None,
        src_image_mesh=None,
        settings_inversion: Optional[aa.Settings] = None,
        preloads: Optional[dict] = None,
        n_iter: int = 20,
        tol: float = 1e-6,
        verbose: bool = False,
        visualize_output_dir: Optional[str] = None,
        visualize_every_n: int = 1000000,
    ):
        """
        Iteratively solves for the pixelized source s and potential
        corrections dpsi minimizing the penalized objective
        P(s, dpsi) = 0.5 [ (d - L(psi0+dpsi) s)^T C^-1 (d - L(psi0+dpsi) s)
        + s^T R_s s + dpsi^T R_dpsi dpsi ], with a Levenberg-Marquardt loop
        on the combined state x = [s | dpsi]. Unlike the one-shot
        ``FitDpsiSrcImaging`` linearization, each accepted step re-ray-traces
        the image grid through the corrected lens, so the corrections feed
        back into the source mapping.

        Parameters
        ----------
        masked_imaging
            The masked ``al.Imaging`` dataset.
        lens_start
            The macro lens galaxy the corrections perturb.
        dpsi_pixelization
            The dpsi mesh + regularization model.
        src_pixelization
            The source pixelization.
        gauge_constraints
            Whether to impose the gauge constraints <dpsi, 1> = <dpsi, x> =
            <dpsi, y> = 0 (removing the constant / deflection-drift
            degeneracies) via an equality-constrained KKT step.
        adapt_image
            The adapt image of the source pixelization's image mesh.
        src_image_mesh
            An image mesh whose image-plane mesh grid is preloaded into the
            source inversion.
        settings_inversion
            The inversion settings; defaults to the border relocator without
            the positive-only solver (the LM state is signed).
        preloads
            Precomputed attributes set directly onto the fit.
        n_iter
            The maximum number of outer LM iterations.
        tol
            The step-norm convergence tolerance.
        verbose
            Whether to log per-iteration costs.
        visualize_output_dir
            Directory the intermediate ``show`` figures are written to.
        visualize_every_n
            Visualize every n-th iteration (default effectively never).
        """
        self.masked_imaging = masked_imaging
        self.lens_start = lens_start
        self.dpsi_pixelization = dpsi_pixelization
        self.src_pixelization = src_pixelization
        self.gauge_constraints = gauge_constraints
        self.adapt_image = adapt_image
        self.src_image_mesh = src_image_mesh
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.verbose = bool(verbose)
        self.visualize_output_dir = visualize_output_dir
        self.visualize_every_n = int(visualize_every_n)

        if settings_inversion is None:
            self.settings_inversion = aa.Settings(
                use_positive_only_solver=False,
                use_border_relocator=True,
            )
        else:
            self.settings_inversion = settings_inversion

        if preloads is not None:
            for key, value in preloads.items():
                setattr(self, key, value)

        self.masked_imaging = self.masked_imaging.apply_over_sampling(
            over_sample_size_lp=4,
            over_sample_size_pixelization=4,
        )

    # -----------------------------
    # Cached low-level operators
    # -----------------------------

    @property
    def inverse_noise_variance(self):
        if not hasattr(self, "inv_noise_var"):
            self.inv_noise_var = np.asarray(
                1.0 / self.masked_imaging.noise_map.slim**2
            )
        return self.inv_noise_var

    @property
    def psf_matrix(self):
        if not hasattr(self, "psf_mat"):
            self.psf_mat = pc_util.psf_matrix_from(
                np.asarray(self.masked_imaging.psf.kernel.native),
                np.asarray(self.masked_imaging.mask),
            )
        return self.psf_mat

    @property
    def pair_dpsi_data_obj(self):
        if not hasattr(self, "_pair_dpsi_data_obj"):
            self._pair_dpsi_data_obj = self.dpsi_pixelization.pair_dpsi_data_mesh(
                self.masked_imaging.mask,
                self.masked_imaging.pixel_scales[0],
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

    # -----------------------------
    # Per-iteration constructions
    # -----------------------------

    def _updated_lens_galaxies_from_dpsi(self, dpsi_vec: np.ndarray):
        """
        The macro lens plus a pixelized ``InputPotential`` correction built
        from the current dpsi vector, as two galaxies on the lens plane.
        """
        mask_dpsi_aa = self.pair_dpsi_data_obj.mask_dpsi_aa
        grid_dpsi = aa.Grid2D.from_mask(mask=mask_dpsi_aa)

        pix_mass_profile = InputPotential(
            lensing_potential=dpsi_vec,
            image_plane_grid=np.asarray(grid_dpsi),
            mask=mask_dpsi_aa,
        )

        lens_macro = self.lens_start
        lens_pix = Galaxy(
            redshift=getattr(lens_macro, "redshift", 0.2), mass=pix_mass_profile
        )

        return [lens_macro, lens_pix]

    def _build_pix_src_tracer(self, lens_galaxies):
        source_galaxy = Galaxy(redshift=1.0, pixelization=self.src_pixelization)

        adapt_kwargs = {}
        if self.adapt_image is not None:
            adapt_kwargs["galaxy_name_image_dict"] = {"source": self.adapt_image}
        if self.src_image_mesh is not None:
            self.image_plane_mesh_grid = self.src_image_mesh.image_plane_mesh_grid_from(
                mask=self.masked_imaging.mask
            )
            adapt_kwargs["galaxy_image_plane_mesh_grid_dict"] = {
                source_galaxy: self.image_plane_mesh_grid
            }

        adapt_images = AdaptImages(**adapt_kwargs) if adapt_kwargs else None

        tracer = Tracer(galaxies=[*lens_galaxies, source_galaxy])

        return tracer, adapt_images

    def _init_source_inversion(self, lens_galaxies):
        """
        Runs one full source inversion at the given lens galaxies, caching
        the source mesh, mapping and regularization matrices the iteration
        keeps fixed (the source mesh does not move between LM steps).
        """
        tracer, adapt_images = self._build_pix_src_tracer(lens_galaxies)

        src_fit = FitImaging(
            dataset=self.masked_imaging,
            tracer=tracer,
            adapt_images=adapt_images,
            settings=self.settings_inversion,
        )

        self.tracer = tracer
        mapper = src_fit.inversion.linear_obj_list[0]
        self.source_plane_mesh_grid = mapper.source_plane_mesh_grid
        self.image_plane_mesh_grid = mapper.image_plane_mesh_grid
        self._src_mesh = mapper.mesh
        self._border_relocator = (
            self.masked_imaging.grids.border_relocator
            if self.settings_inversion.use_border_relocator
            else None
        )
        self._src_regularization = self.src_pixelization.regularization
        self.src_reg_mat = src_fit.inversion.regularization_matrix
        self.src_map_mat = src_fit.inversion.operated_mapping_matrix

    def _update_source_inversion(self, lens_galaxies):
        """
        Re-ray-traces the data grid through the updated lens, keeping the
        source mesh fixed, and rebuilds the PSF-convolved source mapping
        matrix.
        """
        tracer, _ = self._build_pix_src_tracer(lens_galaxies)
        self.tracer = tracer

        source_plane_data_grid = tracer.traced_grid_2d_list_from(
            grid=self.masked_imaging.grid
        )[-1]

        interpolator = self._src_mesh.interpolator_from(
            source_plane_data_grid=source_plane_data_grid,
            source_plane_mesh_grid=self.source_plane_mesh_grid,
            border_relocator=self._border_relocator,
            adapt_data=None,
        )

        mapper = aa.Mapper(
            interpolator=interpolator,
            regularization=self._src_regularization,
            settings=self.settings_inversion,
            image_plane_mesh_grid=self.image_plane_mesh_grid,
        )

        self.src_map_mat = self.masked_imaging.psf.convolved_mapping_matrix_from(
            mapping_matrix=mapper.mapping_matrix,
            mask=self.masked_imaging.mask,
        )

    def _source_plane_data_grid(self, tracer):
        defl = tracer.deflections_yx_2d_from(self.masked_imaging.grid.slim)
        return self.masked_imaging.grid.slim - defl

    def _source_gradient_matrix(self, tracer, source_factory: SrcFactory):
        src_grid = self._source_plane_data_grid(tracer)
        src_grad_vals = source_factory.eval_grad(src_grid[:, 1], src_grid[:, 0])
        return pc_util.source_gradient_matrix_from(src_grad_vals)

    def _source_factory_from(self, s: np.ndarray):
        return PixSrcFactoryITP(points=self.source_plane_mesh_grid, values=s)

    def get_L_Js_Jdpsi(self, s, dpsi, xp=np):
        """
        The source mapping matrix L and the Jacobians (J_s = L, J_dpsi) of
        the model image w.r.t. the source and dpsi at the state (s, dpsi),
        after updating the lens for the current corrections.
        """
        lens_galaxies = self._updated_lens_galaxies_from_dpsi(np.asarray(dpsi))
        source_factory = self._source_factory_from(np.asarray(s))
        self._update_source_inversion(lens_galaxies)
        src_grad_mat = self._source_gradient_matrix(self.tracer, source_factory)
        dpsi_map_mat = dense_util.dpsi_mapping_matrix_from(
            self.psf_matrix, src_grad_mat, self.dpsi_gradient_matrix, xp=xp
        )
        L = dense_util.as_dense(self.src_map_mat, xp=xp)
        return L, L, dpsi_map_mat

    def _regularization_matrix(self, xp=np):
        return dense_util.dense_block_diag_from(
            self.src_reg_mat, self.dpsi_regularization_matrix, xp=xp
        )

    def _init_joint_optimization(self):
        n_dpsi = self.dpsi_regularization_matrix.shape[0]
        lens_galaxies = self._updated_lens_galaxies_from_dpsi(np.zeros(n_dpsi))
        self._init_source_inversion(lens_galaxies)
        n_s = self.src_reg_mat.shape[0]
        return n_s, n_dpsi

    # -----------------------------
    # The Levenberg-Marquardt loop
    # -----------------------------

    def solve_joint_optimization(self, xp=np):
        """
        Runs the Levenberg-Marquardt loop on the combined state
        x = [s | dpsi], accepting steps that decrease the penalized cost and
        adapting the damping mu. Returns the optimized (s, dpsi); they are
        also stored as ``s_opt`` / ``dpsi_opt``.

        Parameters
        ----------
        xp
            The array backend of the dense kernels: ``numpy`` (default) or
            ``jax.numpy`` for accelerator-ready execution. The autolens
            ray-tracing / source-mesh updates between steps always run on
            the host.
        """
        n_s, n_dpsi = self._init_joint_optimization()

        x = xp.zeros(n_s + n_dpsi)

        data_slim = xp.asarray(np.asarray(self.masked_imaging.data.slim))
        inv_var = xp.asarray(self.inverse_noise_variance)
        src_reg_mat = dense_util.as_dense(self.src_reg_mat, xp=xp)
        dpsi_reg_mat = dense_util.as_dense(self.dpsi_regularization_matrix, xp=xp)

        constraint_matrix = None
        if self.gauge_constraints:
            # gauge: <dpsi, 1> = <dpsi, x> = <dpsi, y> = 0, removing the
            # constant and deflection-drift degeneracies of the potential
            G = np.zeros((3, n_dpsi))
            G[0, :] = 1.0 / n_dpsi
            G[1, :] = self.dpsi_points[:, 1] / n_dpsi
            G[2, :] = self.dpsi_points[:, 0] / n_dpsi
            constraint_matrix = xp.asarray(
                np.hstack([np.zeros((3, n_s)), G])
            )

        s = np.asarray(x[:n_s])
        dpsi = np.asarray(x[n_s:])
        L, J_s, J_dpsi = self.get_L_Js_Jdpsi(s, dpsi, xp=xp)

        current_cost, chi2, reg_s, reg_dpsi = (
            float(v)
            for v in dense_util.lm_cost_from(
                data_slim, inv_var, x[:n_s], x[n_s:], L, src_reg_mat, dpsi_reg_mat,
                xp=xp,
            )
        )

        mu = 1.0

        if self.verbose:
            logger.info(
                "Starting joint LM optimization: %d iterations, initial cost %.4e",
                self.n_iter,
                current_cost,
            )

        for i in range(self.n_iter):
            H, minus_gradient, residual, chi2, reg_s, reg_dpsi, current_cost = (
                dense_util.lm_hessian_and_gradient_from(
                    data_slim, inv_var, x, L, J_dpsi, src_reg_mat, dpsi_reg_mat,
                    xp=xp,
                )
            )
            current_cost = float(current_cost)

            if self.visualize_output_dir is not None and i % self.visualize_every_n == 0:
                self.show(
                    np.asarray(L), np.asarray(x[:n_s]), np.asarray(x[n_s:]),
                    output=f"iter_{i}.png",
                )

            if self.verbose:
                logger.info(
                    "Iter %d: cost=%.4e chi2=%.4e reg_s=%.4e mu=%.1e",
                    i, current_cost, float(chi2), float(reg_s), mu,
                )

            step_accepted = False
            while not step_accepted:
                delta_x = None
                try:
                    delta_x = dense_util.solve_lm_step_from(
                        H, minus_gradient, mu,
                        constraint_matrix=constraint_matrix, x=x, xp=xp,
                    )
                    if np.any(np.isnan(np.asarray(delta_x))):
                        delta_x = None
                except np.linalg.LinAlgError:
                    delta_x = None

                if delta_x is not None:
                    x_new = x + delta_x
                    s_new = np.asarray(x_new[:n_s])
                    dpsi_new = np.asarray(x_new[n_s:])

                    L_new, J_s_new, J_dpsi_new = self.get_L_Js_Jdpsi(
                        s_new, dpsi_new, xp=xp
                    )
                    cost_new, chi2_new, reg_s_new, reg_dpsi_new = (
                        dense_util.lm_cost_from(
                            data_slim, inv_var, x_new[:n_s], x_new[n_s:],
                            L_new, src_reg_mat, dpsi_reg_mat, xp=xp,
                        )
                    )
                    cost_new = float(cost_new)

                    if cost_new < current_cost:
                        x = x_new
                        L, J_s, J_dpsi = L_new, J_s_new, J_dpsi_new
                        current_cost = cost_new
                        mu = max(1e-15, mu / 3.0)
                        step_accepted = True

                        if float(xp.linalg.norm(delta_x)) < self.tol:
                            if self.verbose:
                                logger.info(
                                    "Converged at iteration %d (step tolerance).", i
                                )
                            self.s_opt = np.asarray(x[:n_s])
                            self.dpsi_opt = np.asarray(x[n_s:])
                            return self.s_opt, self.dpsi_opt
                    else:
                        mu *= 5.0
                        if mu > 1e15:
                            logger.warning(
                                "LM damping parameter exceeded 1e15; stopping."
                            )
                            self.s_opt = np.asarray(x[:n_s])
                            self.dpsi_opt = np.asarray(x[n_s:])
                            return self.s_opt, self.dpsi_opt
                else:
                    mu *= 5.0
                    if mu > 1e15:
                        logger.warning("LM solver failed repeatedly; stopping.")
                        self.s_opt = np.asarray(x[:n_s])
                        self.dpsi_opt = np.asarray(x[n_s:])
                        return self.s_opt, self.dpsi_opt

        self.s_opt = np.asarray(x[:n_s])
        self.dpsi_opt = np.asarray(x[n_s:])
        return self.s_opt, self.dpsi_opt

    def log_evidence(
        self,
        s: Optional[np.ndarray] = None,
        dpsi: Optional[np.ndarray] = None,
        include_noise_normalization: bool = True,
        xp=np,
    ) -> float:
        """
        The Laplace-approximation log evidence at (s, dpsi) (defaulting to
        the optimized state of ``solve_joint_optimization``).
        """
        if s is None or dpsi is None:
            if hasattr(self, "s_opt") and hasattr(self, "dpsi_opt"):
                s = self.s_opt
                dpsi = self.dpsi_opt
            else:
                raise ValueError(
                    "Provide s and dpsi, or run solve_joint_optimization() first."
                )

        L, J_s, J_dpsi = self.get_L_Js_Jdpsi(s, dpsi, xp=xp)

        log_evidence = float(
            dense_util.log_evidence_lm_from(
                data_slim=np.asarray(self.masked_imaging.data.slim),
                noise_slim=np.asarray(self.masked_imaging.noise_map.slim),
                s=s,
                dpsi=dpsi,
                L=L,
                J_s=J_s,
                J_dpsi=J_dpsi,
                src_reg_matrix=self.src_reg_mat,
                dpsi_reg_matrix=self.dpsi_regularization_matrix,
                xp=xp,
            )
        )

        if include_noise_normalization:
            noise_1d = np.asarray(self.masked_imaging.noise_map.slim)
            log_evidence += float(np.sum(np.log(2.0 * np.pi * noise_1d**2.0))) * (
                -0.5
            )

        return log_evidence

    def show(self, L, s, dpsi, output="show", interpolate=False, show_src_grid=False):
        """
        The nine-panel summary of the current LM state (data, noise, SNR,
        model, residuals, dpsi/dkappa maps and the source).
        """
        import copy
        import os

        from matplotlib import pyplot as plt

        from autolens.potential_correction.visualize import (
            imshow_masked_data,
            show_image_irregular,
            show_image_irregular_interpolate,
        )

        fig = plt.figure(figsize=(15, 10))
        cmap = copy.copy(plt.get_cmap("jet"))
        cmap.set_bad(color="white")
        myargs_data = {
            "origin": "upper",
            "cmap": cmap,
            "extent": self.pair_dpsi_data_obj.data_bound,
        }
        xlimit = [
            self.pair_dpsi_data_obj.xgrid_data_1d.min(),
            self.pair_dpsi_data_obj.xgrid_data_1d.max(),
        ]
        ylimit = [
            self.pair_dpsi_data_obj.ygrid_data_1d.min(),
            self.pair_dpsi_data_obj.ygrid_data_1d.max(),
        ]

        def plot_subplot(pos, data, mask, title):
            plt.subplot(pos)
            ax = plt.gca()
            imshow_masked_data(data, mask, ax=ax, **myargs_data)
            ax.set_title(title)
            ax.set_xlim(*xlimit)
            ax.set_ylim(*ylimit)
            return ax

        plot_subplot(331, self.masked_imaging.data, self.masked_imaging.mask, "Data")
        plot_subplot(332, self.masked_imaging.noise_map, self.masked_imaging.mask, "Noise")
        plot_subplot(
            333,
            self.masked_imaging.data / self.masked_imaging.noise_map,
            self.masked_imaging.mask,
            "SNR",
        )

        model_image_slim = np.asarray(L) @ np.asarray(s)
        ax = plot_subplot(334, model_image_slim, self.masked_imaging.mask, "Model")
        if show_src_grid:
            ax.scatter(
                self.image_plane_mesh_grid[:, 1], self.image_plane_mesh_grid[:, 0],
                c="black", s=0.5, alpha=0.5,
            )

        residual = self.masked_imaging.data - model_image_slim
        plot_subplot(335, residual, self.masked_imaging.mask, "Residual")
        plot_subplot(
            336, residual / self.masked_imaging.noise_map, self.masked_imaging.mask,
            "Norm Residual",
        )

        plot_subplot(337, dpsi, self.pair_dpsi_data_obj.mask_dpsi, "Dpsi Map")
        plot_subplot(
            338,
            self.pair_dpsi_data_obj.hamiltonian_dpsi @ np.asarray(dpsi),
            self.pair_dpsi_data_obj.mask_dpsi,
            "Dkappa Map",
        )

        plt.subplot(339)
        ax = plt.gca()
        if interpolate:
            show_image_irregular_interpolate(
                self.source_plane_mesh_grid, s, ax=ax, enlarge_factor=1.1,
                npixels=100, cmap="jet",
            )
        else:
            show_image_irregular(
                self.source_plane_mesh_grid, s, enlarge_factor=1.1, cmap="jet",
                ax=ax, title="Source",
            )
        if show_src_grid:
            ax.scatter(
                self.source_plane_mesh_grid[:, 1], self.source_plane_mesh_grid[:, 0],
                c="black", s=0.1, alpha=0.5,
            )
        ax.set_title("Source")

        plt.tight_layout()

        if output == "show":
            return fig
        if self.visualize_output_dir is None:
            raise ValueError("visualize_output_dir must be set when saving figures")
        os.makedirs(self.visualize_output_dir, exist_ok=True)
        fig.savefig(
            os.path.join(self.visualize_output_dir, output), bbox_inches="tight"
        )
        plt.close(fig)


class IterDpsiSrcInvAnalysis(af.Analysis):
    def __init__(
        self,
        masked_imaging,
        lens_start: Galaxy,
        n_iter: int = 20,
        tol: float = 1e-6,
        adapt_image=None,
        settings_inversion: Optional[aa.Settings] = None,
        preloads: Optional[dict] = None,
        gauge_constraints: bool = True,
        verbose: bool = False,
        xp=np,
    ):
        """
        Samples the joint source+dpsi pixelization hyper-parameters with the
        iterative LM solver's Laplace evidence as the likelihood.

        Parameters
        ----------
        masked_imaging
            The masked ``al.Imaging`` dataset.
        lens_start
            The macro lens galaxy the corrections perturb.
        n_iter, tol, gauge_constraints, verbose
            Forwarded to ``IterFitDpsiSrcImaging``.
        adapt_image
            The adapt image of the source pixelization's image mesh.
        settings_inversion
            The inversion settings; defaults to the border relocator without
            the positive-only solver, with the adaptive image-mesh
            thresholds of the original implementation.
        preloads
            Precomputed fit attributes shared across evaluations.
        xp
            The array backend of the dense LM kernels (``numpy`` or
            ``jax.numpy``).
        """
        self.masked_imaging = masked_imaging
        self.lens_start = lens_start
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.adapt_image = adapt_image
        self.gauge_constraints = bool(gauge_constraints)
        self.verbose = bool(verbose)
        self.xp = xp
        if settings_inversion is None:
            self.settings_inversion = aa.Settings(
                use_positive_only_solver=False,
                use_border_relocator=True,
                image_mesh_min_mesh_pixels_per_pixel=3,
                image_mesh_min_mesh_number=5,
                image_mesh_adapt_background_percent_threshold=0.1,
                image_mesh_adapt_background_percent_check=0.8,
            )
        else:
            self.settings_inversion = settings_inversion
        self.preloads = preloads

    def _fit_from(self, instance: DpsiSrcPixelization, visualize_output_dir=None):
        return IterFitDpsiSrcImaging(
            masked_imaging=self.masked_imaging,
            lens_start=self.lens_start,
            dpsi_pixelization=instance.dpsi_pixelization,
            src_pixelization=instance.src_pixelization,
            adapt_image=self.adapt_image,
            gauge_constraints=self.gauge_constraints,
            n_iter=self.n_iter,
            tol=self.tol,
            verbose=self.verbose,
            settings_inversion=self.settings_inversion,
            preloads=self.preloads,
            visualize_output_dir=visualize_output_dir,
        )

    def log_likelihood_function(self, instance: DpsiSrcPixelization):
        fit = self._fit_from(instance)
        try:
            s_opt, dpsi_opt = fit.solve_joint_optimization(xp=self.xp)
            return fit.log_evidence(s=s_opt, dpsi=dpsi_opt, xp=self.xp)
        except exc.InversionException:
            # a failed inversion is a valid (very bad) sample, not a crash
            logger.exception(
                "InversionException during iterative source+dpsi optimization; "
                "returning penalty likelihood."
            )
            return -1e30

    def visualize(self, paths: af.DirectoryPaths, instance, during_analysis=True):
        fit = self._fit_from(instance, visualize_output_dir=str(paths.image_path))
        s_opt, dpsi_opt = fit.solve_joint_optimization(xp=self.xp)
        L, _, _ = fit.get_L_Js_Jdpsi(s_opt, dpsi_opt, xp=self.xp)
        fit.show(L, s_opt, dpsi_opt, output="final.png")
