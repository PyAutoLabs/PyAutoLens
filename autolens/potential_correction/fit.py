"""
The linear fits of the gravitational-imaging (potential correction)
technique: ``FitDpsiImaging`` inverts an image residual for pixelized
corrections dpsi to the lensing potential; ``FitDpsiSrcImaging`` jointly
inverts the image for the pixelized source and dpsi, fully accounting for
their covariance, with the Bayesian evidence used to set the regularization
strengths of both.

Ported from the ``potential_correction`` package of Cao et al. 2025
(https://github.com/caoxiaoyue/lensing_potential_correction). If you use this
functionality in your research, please cite Cao et al. 2025; citation
materials are provided at
https://github.com/caoxiaoyue/potential_correction_paper.
"""

from typing import Optional

import numpy as np
from scipy.sparse import block_diag
from scipy.spatial import Delaunay

import autoarray as aa

from autogalaxy.profiles.mass.input.interp import LinearNDInterpolatorExt
from autogalaxy.galaxy.galaxy import Galaxy

from autolens.lens.tracer import Tracer
from autolens.imaging.fit_imaging import FitImaging
from autogalaxy.analysis.adapt_images.adapt_images import AdaptImages
from autolens.potential_correction import util as pc_util
from autolens.potential_correction.pixelization import (
    DpsiLinearObj,
    DpsiPixelization,
)
from autolens.potential_correction.src_factory import SrcFactory


class FitDpsiImaging:
    def __init__(
        self,
        masked_imaging,
        image_residual: np.ndarray,
        source_gradient: np.ndarray,
        dpsi_pixelization: DpsiPixelization,
        anchor_points: Optional[np.ndarray] = None,
        preloads: Optional[dict] = None,
    ):
        """
        A linear inversion of an image residual for pixelized corrections
        dpsi to the lensing potential, at fixed source: the residual is
        modelled as -B D_s D_psi dpsi (PSF blur matrix B, source-gradient
        matrix D_s, dpsi-gradient operator D_psi), regularized on the dpsi
        mesh and solved for the maximum-evidence correction.

        Parameters
        ----------
        masked_imaging
            The masked ``al.Imaging`` dataset.
        image_residual
            The 1D (slim) image residual of the smooth-model fit, of shape
            [n_unmasked_data_pixels].
        source_gradient
            The [n_unmasked_data_pixels, 2] (dS/dy, dS/dx) source gradients
            at the ray-traced positions of the image pixels.
        dpsi_pixelization
            The dpsi mesh + regularization model.
        anchor_points
            The [3, 2] (y, x) anchor positions of the dpsi rescaling scheme.
        preloads
            Precomputed attributes set directly onto the fit (e.g. the
            ``psf_mat`` shared across evaluations).
        """
        self.masked_imaging = masked_imaging
        self.input_image_residual = image_residual
        self.source_gradient = source_gradient
        self.anchor_points = anchor_points
        self.dpsi_pixelization = dpsi_pixelization

        if preloads is not None:
            for key, value in preloads.items():
                setattr(self, key, value)

        self.masked_imaging = self.masked_imaging.apply_over_sampling(
            over_sample_size_lp=4,
            over_sample_size_pixelization=4,
        )

    @property
    def inverse_noise_covariance_matrix(self):
        noise_1d = self.masked_imaging.noise_map.slim
        return pc_util.inverse_covariance_matrix_from(noise_1d)

    @property
    def psf_matrix(self):
        return pc_util.psf_matrix_from(
            np.asarray(self.masked_imaging.psf.kernel.native),
            np.asarray(self.masked_imaging.mask),
        )

    @property
    def source_gradient_matrix(self):
        return pc_util.source_gradient_matrix_from(self.source_gradient)

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
        if not hasattr(self, "itp_mat"):
            self.itp_mat = self.pair_dpsi_data_obj.itp_mat
        return pc_util.dpsi_gradient_matrix_from(
            self.itp_mat,
            self.pair_dpsi_data_obj.Hx_dpsi,
            self.pair_dpsi_data_obj.Hy_dpsi,
        )

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
    def dpsi_linear_obj(self):
        return DpsiLinearObj(
            mask=self.pair_dpsi_data_obj.mask_dpsi, points=self.dpsi_points
        )

    @property
    def dpsi_regularization_matrix(self):
        if not hasattr(self, "dpsi_reg_mat"):
            self.dpsi_reg_mat = (
                self.dpsi_pixelization.regularization.regularization_matrix_from(
                    linear_obj=self.dpsi_linear_obj
                )
            )
        return self.dpsi_reg_mat

    @property
    def mapping_matrix(self):
        # np.asarray guards against dense @ sparse products returning np.matrix
        return np.asarray(-1.0 * self.psf_mat @ self.src_grad_mat @ self.dpsi_grad_mat)

    @property
    def data_vector(self):
        return np.asarray(
            self.map_mat.T @ self.inv_cov_mat @ self.input_image_residual
        ).ravel()

    @property
    def curvature_regularization_matrix(self):
        return np.asarray(
            self.map_mat.T @ self.inv_cov_mat @ self.map_mat + self.reg_mat
        )

    def construct_useful_matrices(self):
        if not hasattr(self, "psf_mat"):
            self.psf_mat = self.psf_matrix
        if not hasattr(self, "inv_cov_mat"):
            self.inv_cov_mat = self.inverse_noise_covariance_matrix
        if not hasattr(self, "src_grad_mat"):
            self.src_grad_mat = self.source_gradient_matrix
        if not hasattr(self, "dpsi_grad_mat"):
            self.dpsi_grad_mat = self.dpsi_gradient_matrix
        if not hasattr(self, "reg_mat"):
            self.reg_mat = self.dpsi_regularization_matrix
        if not hasattr(self, "map_mat"):
            self.map_mat = self.mapping_matrix
        if not hasattr(self, "d_vec"):
            self.d_vec = self.data_vector
        self.curve_reg_mat = self.curvature_regularization_matrix

    def solve_dpsi(self, return_error: bool = False):
        self.construct_useful_matrices()
        if return_error:
            return (
                np.linalg.solve(self.curve_reg_mat, self.d_vec),
                np.linalg.inv(self.curve_reg_mat),
            )
        return np.linalg.solve(self.curve_reg_mat, self.d_vec)

    @property
    def log_evidence(self):
        """
        The Bayesian evidence of the dpsi inversion: noise normalization,
        the log-determinants of the curvature+regularization and
        regularization matrices, the regularization penalty of the solution
        and the chi-squared of the residual fit.
        """
        if not hasattr(self, "dpsi_slim") or not hasattr(
            self, "model_image_residual_slim"
        ):
            self.dpsi_slim = self.solve_dpsi()
            self.model_image_residual_slim = self.map_mat @ self.dpsi_slim

        noise_slim = np.asarray(self.masked_imaging.noise_map.slim)

        self.noise_term = float(np.sum(np.log(2 * np.pi * noise_slim**2.0))) * (-0.5)

        sign, logval = np.linalg.slogdet(self.curve_reg_mat)
        if sign != 1:
            raise np.linalg.LinAlgError(
                "The curvature+regularization matrix is not positive definite."
            )
        self.log_det_curve_reg_term = logval * (-0.5)

        try:
            sign, logval = np.linalg.slogdet(self.reg_mat)
            if sign != 1:
                raise np.linalg.LinAlgError(
                    "The regularization matrix is not positive definite."
                )
            self.log_det_reg_term = logval * 0.5
        except (np.linalg.LinAlgError, TypeError, ValueError):
            self.log_det_reg_term = pc_util.log_det_mat(self.reg_mat, sparse=True) * 0.5

        reg_cov_term = self.dpsi_slim.T @ self.reg_mat @ self.dpsi_slim
        self.reg_cov_term = float(reg_cov_term) * (-0.5)

        residual_of_image_residual = (
            self.input_image_residual - self.model_image_residual_slim
        )
        norm_residual = residual_of_image_residual / noise_slim
        self.chi2_term = float(np.sum(norm_residual**2)) * (-0.5)

        return (
            self.noise_term
            + self.log_det_curve_reg_term
            + self.log_det_reg_term
            + self.reg_cov_term
            + self.chi2_term
        )

    @property
    def rescaled_dpsi(self):
        """
        The dpsi solution rescaled by the plane zeroing it at the three
        anchor points (Suyu et al.'s scheme), plus the (a_y, a_x, c)
        coefficients. Falls back to the unrescaled solution when no valid
        anchor points are set.
        """
        if self.anchor_points is None or np.shape(self.anchor_points) != (3, 2):
            return self.dpsi_slim, 0.0, 0.0, 0.0
        if not hasattr(self, "dpsi_at_anchors"):
            tri = Delaunay(np.fliplr(self.dpsi_points))
            self.dpsi_interpl = LinearNDInterpolatorExt(tri, self.dpsi_slim)
            self.dpsi_at_anchors = self.dpsi_interpl(
                self.anchor_points[:, 1], self.anchor_points[:, 0]
            )
        ay, ax, c = pc_util.dpsi_rescale_factors_from(
            self.anchor_points, self.dpsi_at_anchors
        )
        dpsi_new = (
            ay * self.anchor_points[:, 0]
            + ax * self.anchor_points[:, 1]
            + c
            + self.dpsi_slim
        )
        return dpsi_new, ay, ax, c


class FitDpsiSrcImaging:
    def __init__(
        self,
        masked_imaging,
        lens_start: Galaxy,
        source_start: SrcFactory,
        dpsi_pixelization: DpsiPixelization,
        src_pixelization: aa.Pixelization,
        anchor_points: Optional[np.ndarray] = None,
        adapt_image=None,
        src_image_mesh=None,
        settings_inversion: Optional[aa.Settings] = None,
        preloads: Optional[dict] = None,
    ):
        """
        A joint linear inversion of an image for the pixelized source and
        the pixelized corrections dpsi to the lensing potential, fully
        accounting for their covariance: the mapping matrix is the
        horizontal block [F_src | -B D_s D_psi] and the regularization the
        block diagonal of the source and dpsi regularization matrices, with
        the Bayesian evidence used to set both regularization strengths.

        Parameters
        ----------
        masked_imaging
            The masked ``al.Imaging`` dataset.
        lens_start
            The lens galaxy of the smooth-model fit the corrections perturb.
        source_start
            The source factory evaluated for the source gradients (from the
            smooth-model fit's source).
        dpsi_pixelization
            The dpsi mesh + regularization model.
        src_pixelization
            The source pixelization of the joint inversion.
        anchor_points
            The [3, 2] (y, x) anchor positions of the dpsi rescaling scheme.
        adapt_image
            The adapt image of the source pixelization's image mesh.
        src_image_mesh
            An image mesh whose image-plane mesh grid is preloaded into the
            source inversion.
        settings_inversion
            The inversion settings; defaults to the positive-only solver
            with the border relocator.
        preloads
            Precomputed attributes set directly onto the fit.
        """
        self.masked_imaging = masked_imaging
        self.anchor_points = anchor_points
        self.lens_start = lens_start
        self.source_start = source_start
        self.dpsi_pixelization = dpsi_pixelization
        self.src_pixelization = src_pixelization
        self.adapt_image = adapt_image
        self.src_image_mesh = src_image_mesh
        if settings_inversion is None:
            self.settings_inversion = aa.Settings(
                use_positive_only_solver=True,
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

    def do_source_inversion(self):
        """
        Runs the standard autolens source inversion at the starting lens
        model, caching its mapper, operated mapping matrix and
        regularization matrix as the source blocks of the joint inversion.
        """
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

        self.adapt_images = AdaptImages(**adapt_kwargs) if adapt_kwargs else None

        tracer = Tracer(galaxies=[self.lens_start, source_galaxy])

        self.src_fit = FitImaging(
            dataset=self.masked_imaging,
            tracer=tracer,
            adapt_images=self.adapt_images,
            settings=self.settings_inversion,
        )

        self.src_mapper = self.src_fit.inversion.linear_obj_list[0]
        self.src_map_mat = self.src_fit.inversion.operated_mapping_matrix
        self.src_reg_mat = self.src_fit.inversion.regularization_matrix

    @property
    def inverse_noise_covariance_matrix(self):
        if not hasattr(self, "inv_cov_mat"):
            noise_1d = self.masked_imaging.noise_map
            self.inv_cov_mat = pc_util.inverse_covariance_matrix_from(noise_1d)
        return self.inv_cov_mat

    @property
    def psf_matrix(self):
        if not hasattr(self, "psf_mat"):
            self.psf_mat = pc_util.psf_matrix_from(
                np.asarray(self.masked_imaging.psf.kernel.native),
                np.asarray(self.masked_imaging.mask),
            )
        return self.psf_mat

    @property
    def source_plane_data_grid(self):
        # single-plane ray tracing is assumed here
        return self.masked_imaging.grid.slim - self.lens_start.deflections_yx_2d_from(
            self.masked_imaging.grid.slim
        )

    @property
    def source_plane_source_gradient(self):
        return self.source_start.eval_grad(
            self.source_plane_data_grid[:, 1], self.source_plane_data_grid[:, 0]
        )

    @property
    def source_gradient_matrix(self):
        if not hasattr(self, "src_grad_mat"):
            self.src_grad_mat = pc_util.source_gradient_matrix_from(
                self.source_plane_source_gradient
            )
        return self.src_grad_mat

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
        if not hasattr(self, "itp_mat"):
            self.itp_mat = self.pair_dpsi_data_obj.itp_mat
        if not hasattr(self, "dpsi_grad_mat"):
            self.dpsi_grad_mat = pc_util.dpsi_gradient_matrix_from(
                self.itp_mat,
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
    def dpsi_linear_obj(self):
        return DpsiLinearObj(
            mask=self.pair_dpsi_data_obj.mask_dpsi, points=self.dpsi_points
        )

    @property
    def dpsi_regularization_matrix(self):
        if not hasattr(self, "dpsi_reg_mat"):
            self.dpsi_reg_mat = (
                self.dpsi_pixelization.regularization.regularization_matrix_from(
                    linear_obj=self.dpsi_linear_obj
                )
            )
        return self.dpsi_reg_mat

    @property
    def dpsi_mapping_matrix(self):
        if not hasattr(self, "dpsi_map_mat"):
            # np.asarray guards against dense @ sparse returning np.matrix
            self.dpsi_map_mat = np.asarray(
                -1.0
                * self.psf_matrix
                @ self.source_gradient_matrix
                @ self.dpsi_gradient_matrix
            )
        return self.dpsi_map_mat

    @property
    def src_regularization_matrix(self):
        if not hasattr(self, "src_reg_mat"):
            if hasattr(self, "src_reg_base_mat"):
                coeff = self.src_pixelization.regularization.coefficient
                self.src_reg_mat = coeff * self.src_reg_base_mat
            else:
                self.do_source_inversion()
        return self.src_reg_mat

    @property
    def src_mapping_matrix(self):
        if not hasattr(self, "src_map_mat"):
            self.do_source_inversion()
        return self.src_map_mat

    @property
    def mapping_matrix(self):
        if not hasattr(self, "map_mat"):
            self.map_mat = np.hstack(
                (self.src_mapping_matrix, self.dpsi_mapping_matrix)
            )
        return self.map_mat

    @property
    def regularization_matrix(self):
        if not hasattr(self, "reg_mat"):
            self.reg_mat = block_diag(
                [self.src_regularization_matrix, self.dpsi_regularization_matrix]
            )
        return self.reg_mat

    @property
    def data_vector(self):
        if not hasattr(self, "d_vec"):
            self.d_vec = np.asarray(
                self.mapping_matrix.T
                @ self.inverse_noise_covariance_matrix
                @ np.asarray(self.masked_imaging.data)
            ).ravel()
        return self.d_vec

    @property
    def curvature_matrix(self):
        if not hasattr(self, "curv_mat"):
            self.curv_mat = np.asarray(
                self.mapping_matrix.T
                @ self.inverse_noise_covariance_matrix
                @ self.mapping_matrix
            )
        return self.curv_mat

    @property
    def curvature_regularization_matrix(self):
        return self.curvature_matrix + self.regularization_matrix

    def solve_src_dpsi(self, return_error: bool = False):
        curve_reg_mat = np.asarray(self.curvature_regularization_matrix)
        data_vector = np.asarray(self.data_vector)
        if return_error:
            return (
                np.linalg.solve(curve_reg_mat, data_vector),
                np.linalg.inv(curve_reg_mat),
            )
        return np.linalg.solve(curve_reg_mat, data_vector)

    @property
    def log_evidence(self):
        """
        The Bayesian evidence of the joint source+dpsi inversion: noise
        normalization, the log-determinants of the curvature+regularization
        matrix and of the source and dpsi regularization matrices, the joint
        regularization penalty of the solution and the chi-squared of the
        image fit.
        """
        if not hasattr(self, "src_dpsi_slim") or not hasattr(
            self, "model_image_slim"
        ):
            self.src_dpsi_slim = self.solve_src_dpsi()
            self.model_image_slim = self.mapping_matrix @ self.src_dpsi_slim

        noise_slim = np.asarray(self.masked_imaging.noise_map)

        self.noise_term = float(np.sum(np.log(2 * np.pi * noise_slim**2.0))) * (-0.5)

        curve_reg_mat = np.asarray(self.curvature_regularization_matrix)
        sign, logval = np.linalg.slogdet(curve_reg_mat)
        if sign != 1:
            raise np.linalg.LinAlgError(
                "The curvature+regularization matrix is not positive definite."
            )
        self.log_det_curve_reg_term = logval * (-0.5)

        self.log_det_reg_term_src = (
            pc_util.log_det_mat(self.src_regularization_matrix, sparse=True) * 0.5
        )
        try:
            sign, logval = np.linalg.slogdet(
                np.asarray(self.dpsi_regularization_matrix)
            )
            if sign != 1:
                raise np.linalg.LinAlgError(
                    "The dpsi regularization matrix is not positive definite."
                )
            self.log_det_reg_term_dpsi = logval * 0.5
        except (np.linalg.LinAlgError, TypeError, ValueError):
            self.log_det_reg_term_dpsi = (
                pc_util.log_det_mat(self.dpsi_regularization_matrix, sparse=True) * 0.5
            )
        self.log_det_reg_term = self.log_det_reg_term_src + self.log_det_reg_term_dpsi

        reg_cov_term = (
            self.src_dpsi_slim.T @ self.regularization_matrix @ self.src_dpsi_slim
        )
        self.reg_cov_term = float(reg_cov_term) * (-0.5)

        image_residual = np.asarray(self.masked_imaging.data) - self.model_image_slim
        norm_residual = image_residual / noise_slim
        self.chi2_term = float(np.sum(norm_residual**2)) * (-0.5)

        return (
            self.noise_term
            + self.log_det_curve_reg_term
            + self.log_det_reg_term
            + self.reg_cov_term
            + self.chi2_term
        )

    def draw_random_solutions(self, n_solutions: int = 300, return_dkappa: bool = True):
        """
        Draws source/dpsi solutions consistent with the data at the level
        permitted by the noise and regularizations; when ``return_dkappa``
        the dpsi block is transformed to convergence corrections via the
        dpsi mesh's Laplacian operator.
        """
        mean, cov_mat = self.solve_src_dpsi(return_error=True)

        L = np.diag(np.ones_like(mean))
        if return_dkappa:
            n_src = self.src_regularization_matrix.shape[0]
            L[n_src:, n_src:] = self.pair_dpsi_data_obj.hamiltonian_dpsi.toarray()

        cov_mat = L @ cov_mat @ L.T
        mean = L @ mean
        return np.random.multivariate_normal(
            mean, cov_mat, size=n_solutions, check_valid="warn", tol=1e-5
        )

    @property
    def best_fit_source(self):
        n_s = self.src_regularization_matrix.shape[0]
        return self.src_dpsi_slim[0:n_s]

    @property
    def best_fit_dpsi(self):
        n_s = self.src_regularization_matrix.shape[0]
        return self.src_dpsi_slim[n_s:]

    @property
    def rescaled_dpsi(self):
        """
        The dpsi solution rescaled by the plane zeroing it at the three
        anchor points (Suyu et al.'s scheme), plus the (a_y, a_x, c)
        coefficients. Falls back to the unrescaled solution when no valid
        anchor points are set.
        """
        if self.anchor_points is None or np.shape(self.anchor_points) != (3, 2):
            return self.best_fit_dpsi, 0.0, 0.0, 0.0
        if not hasattr(self, "dpsi_at_anchors"):
            tri = Delaunay(np.fliplr(self.dpsi_points))
            self.dpsi_interpl = LinearNDInterpolatorExt(tri, self.best_fit_dpsi)
            self.dpsi_at_anchors = self.dpsi_interpl(
                self.anchor_points[:, 1], self.anchor_points[:, 0]
            )
        ay, ax, c = pc_util.dpsi_rescale_factors_from(
            self.anchor_points, self.dpsi_at_anchors
        )
        dpsi_new = (
            ay * self.anchor_points[:, 0]
            + ax * self.anchor_points[:, 1]
            + c
            + self.best_fit_dpsi
        )
        return dpsi_new, ay, ax, c
