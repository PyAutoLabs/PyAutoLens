"""
The visibility-space fits of the gravitational-imaging (potential correction)
technique: ``FitDpsiSrcInterferometer`` jointly inverts an `Interferometer`
dataset for the pixelized source and pixelized corrections dpsi to the
lensing potential.

Two routes build the joint linear system:

- **Sparse-operator route (primary)**: the curvature blocks are computed in
  real space through the dataset's ``InterferometerSparseOperator`` (the
  w-tilde formalism, ``F = A^T W~ A`` via FFT convolution) and the data
  vector through its dirty image — cost scales with the number of real-space
  mask pixels, independent of the number of visibilities. Requires
  ``dataset.apply_sparse_operator()``.
- **Dense route (reference)**: the real-space response is pushed through
  ``transformer.transform_mapping_matrix`` and the curvature assembled from
  the stacked real/imag visibility blocks — viable only for small
  visibility counts, retained as the parity reference for the sparse route.

Ported and extended from the ``potential_correction`` package of Cao et al.
2025 (https://github.com/caoxiaoyue/lensing_potential_correction). If you use
this functionality in your research, please cite Cao et al. 2025; citation
materials are provided at
https://github.com/caoxiaoyue/potential_correction_paper. The visibility-space
formulation follows the gravitational-imaging methodology behind the
JVAS B1938+666 detections (Powell et al. 2025; Vegetti et al. 2026).
"""

from typing import Optional

import numpy as np
from scipy.sparse import block_diag, coo_matrix

import autoarray as aa
from autoarray.inversion.mappers import mapper_util

from autogalaxy.galaxy.galaxy import Galaxy

from autolens.lens.tracer import Tracer
from autolens.interferometer.fit_interferometer import FitInterferometer
from autolens.potential_correction import util as pc_util
from autolens.potential_correction.pixelization import (
    DpsiLinearObj,
    DpsiPixelization,
)
from autolens.potential_correction.src_factory import SrcFactory


class FitDpsiSrcInterferometer:
    def __init__(
        self,
        dataset,
        lens_start: Galaxy,
        source_start: SrcFactory,
        dpsi_pixelization: DpsiPixelization,
        src_pixelization: aa.Pixelization,
        src_image_mesh=None,
        settings_inversion: Optional[aa.Settings] = None,
        use_sparse_operator: bool = True,
        preloads: Optional[dict] = None,
    ):
        """
        A joint linear inversion of interferometer visibilities for the
        pixelized source and the pixelized corrections dpsi to the lensing
        potential, fully accounting for their covariance.

        The real-space responses are the source mapper's mapping matrix f and
        the correction response G = D_s D_psi (source-gradient matrix times
        dpsi-gradient operator — no PSF: the measurement operator here is the
        non-uniform Fourier transform, applied by the route machinery). The
        joint curvature is [f | G]^T (T^H C^-1 T) [f | G] and the data vector
        [f | G]^T T^H C^-1 d, with the Bayesian evidence setting both
        regularization strengths.

        Parameters
        ----------
        dataset
            The ``al.Interferometer`` dataset. For the sparse-operator route
            (the default, which scales to large visibility counts) it must
            carry a sparse operator — call ``dataset.apply_sparse_operator()``
            first.
        lens_start
            The smooth lens galaxy of the starting model the corrections
            perturb.
        source_start
            The source factory evaluated for the source gradients.
        dpsi_pixelization
            The dpsi mesh + regularization model, built on the dataset's
            real-space mask.
        src_pixelization
            The source pixelization of the joint inversion.
        src_image_mesh
            An image mesh whose image-plane mesh grid is preloaded into the
            source inversion.
        settings_inversion
            The inversion settings; defaults to the positive-only solver
            with the border relocator.
        use_sparse_operator
            Whether the curvature/data-vector are built through the sparse
            w-tilde operator (default) or the dense transformed mapping
            matrix (small visibility counts only; the parity reference).
        preloads
            Precomputed attributes set directly onto the fit.
        """
        self.dataset = dataset
        self.lens_start = lens_start
        self.source_start = source_start
        self.dpsi_pixelization = dpsi_pixelization
        self.src_pixelization = src_pixelization
        self.src_image_mesh = src_image_mesh
        self.use_sparse_operator = use_sparse_operator
        if settings_inversion is None:
            self.settings_inversion = aa.Settings(
                use_positive_only_solver=True,
                use_border_relocator=True,
            )
        else:
            self.settings_inversion = settings_inversion

        if use_sparse_operator and getattr(dataset, "sparse_operator", None) is None:
            raise aa.exc.InversionException(
                "The sparse-operator route requires the dataset's sparse operator: "
                "call dataset.apply_sparse_operator() first (or pass "
                "use_sparse_operator=False for the dense small-visibility route)."
            )

        if preloads is not None:
            for key, value in preloads.items():
                setattr(self, key, value)

    # -----------------------------
    # Source inversion blocks
    # -----------------------------

    def do_source_inversion(self):
        """
        Runs the standard autolens interferometer source inversion at the
        starting lens model, caching its mapper and regularization matrix as
        the source blocks of the joint inversion.
        """
        source_galaxy = Galaxy(redshift=1.0, pixelization=self.src_pixelization)

        adapt_images = None
        if self.src_image_mesh is not None:
            self.image_plane_mesh_grid = self.src_image_mesh.image_plane_mesh_grid_from(
                mask=self.dataset.real_space_mask
            )
            from autogalaxy.analysis.adapt_images.adapt_images import AdaptImages

            adapt_images = AdaptImages(
                galaxy_image_plane_mesh_grid_dict={
                    source_galaxy: self.image_plane_mesh_grid
                }
            )

        tracer = Tracer(galaxies=[self.lens_start, source_galaxy])

        self.src_fit = FitInterferometer(
            dataset=self.dataset,
            tracer=tracer,
            adapt_images=adapt_images,
            settings=self.settings_inversion,
        )

        self.src_mapper = self.src_fit.inversion.linear_obj_list[0]
        self.src_reg_mat = self.src_fit.inversion.regularization_matrix

    @property
    def src_regularization_matrix(self):
        if not hasattr(self, "src_reg_mat"):
            self.do_source_inversion()
        return self.src_reg_mat

    @property
    def mapper(self):
        if not hasattr(self, "src_mapper"):
            self.do_source_inversion()
        return self.src_mapper

    # -----------------------------
    # Correction response (real space)
    # -----------------------------

    @property
    def pair_dpsi_data_obj(self):
        if not hasattr(self, "_pair_dpsi_data_obj"):
            self._pair_dpsi_data_obj = self.dpsi_pixelization.pair_dpsi_data_mesh(
                np.asarray(self.dataset.real_space_mask),
                self.dataset.real_space_mask.pixel_scales[0],
            )
        return self._pair_dpsi_data_obj

    @property
    def source_plane_data_grid(self):
        # single-plane ray tracing of the real-space grid is assumed
        return self.dataset.grid.slim - self.lens_start.deflections_yx_2d_from(
            self.dataset.grid.slim
        )

    @property
    def source_gradient_matrix(self):
        if not hasattr(self, "src_grad_mat"):
            source_gradients = self.source_start.eval_grad(
                self.source_plane_data_grid[:, 1], self.source_plane_data_grid[:, 0]
            )
            self.src_grad_mat = pc_util.source_gradient_matrix_from(source_gradients)
        return self.src_grad_mat

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
    def dpsi_response_matrix(self):
        """
        The sparse real-space correction response G = -D_s D_psi of shape
        [n_real_space_pixels, n_dpsi]: the change of the (unconvolved,
        untransformed) real-space image per unit dpsi.
        """
        if not hasattr(self, "dpsi_response_mat"):
            self.dpsi_response_mat = (
                -1.0 * self.source_gradient_matrix @ self.dpsi_gradient_matrix
            ).tocoo()
        return self.dpsi_response_mat

    # -----------------------------
    # Regularization
    # -----------------------------

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
    def regularization_matrix(self):
        if not hasattr(self, "reg_mat"):
            self.reg_mat = np.asarray(
                block_diag(
                    [self.src_regularization_matrix, self.dpsi_regularization_matrix]
                ).toarray()
            )
        return self.reg_mat

    # -----------------------------
    # Joint system — sparse (w-tilde) route
    # -----------------------------

    def _joint_triplets(self):
        """
        The COO triplets of the joint real-space response A = [f | G] with
        rows indexed on the sparse operator's rectangular (unmasked-extent)
        grid, mirroring the sparse interferometer inversion's assembly.
        """
        mapper = self.mapper
        extent_index = np.asarray(
            self.dataset.real_space_mask.extent_index_for_masked_pixel
        )

        rows_src, cols_src, vals_src = mapper_util.sparse_triplets_from(
            pix_indexes_for_sub=mapper.pix_indexes_for_sub_slim_index,
            pix_weights_for_sub=mapper.pix_weights_for_sub_slim_index,
            slim_index_for_sub=mapper.slim_index_for_sub_slim_index,
            fft_index_for_masked_pixel=extent_index,
            sub_fraction_slim=mapper.over_sampler.sub_fraction.array,
            return_rows_slim=False,
        )

        G = self.dpsi_response_matrix
        n_src = self.mapper.params
        rows_dpsi = extent_index[G.row]
        cols_dpsi = G.col + n_src
        vals_dpsi = G.data

        rows = np.concatenate([np.asarray(rows_src), rows_dpsi])
        cols = np.concatenate([np.asarray(cols_src), cols_dpsi])
        vals = np.concatenate([np.asarray(vals_src), vals_dpsi])
        return rows, cols, vals, n_src

    @property
    def real_space_mapping_matrix(self):
        """
        The dense real-space joint response [f | G] on the slim mask pixels,
        used for the data vector and to materialize model images.
        """
        if not hasattr(self, "_real_space_mapping_matrix"):
            f = np.asarray(self.mapper.mapping_matrix)
            G = np.asarray(self.dpsi_response_matrix.todense())
            self._real_space_mapping_matrix = np.hstack([f, G])
        return self._real_space_mapping_matrix

    @property
    def curvature_matrix(self):
        if not hasattr(self, "curv_mat"):
            if self.use_sparse_operator:
                rows, cols, vals, n_src = self._joint_triplets()
                self.curv_mat = np.asarray(
                    self.dataset.sparse_operator.curvature_matrix_diag_from(
                        rows=rows,
                        cols=cols,
                        vals=vals,
                        S=n_src + self.dpsi_regularization_matrix.shape[0],
                    )
                )
            else:
                operated = self.operated_mapping_matrix
                inv_var = self._stacked_inverse_variance
                self.curv_mat = operated.T @ (operated * inv_var[:, None])
        return self.curv_mat

    @property
    def data_vector(self):
        if not hasattr(self, "d_vec"):
            if self.use_sparse_operator:
                self.d_vec = np.asarray(
                    self.real_space_mapping_matrix.T
                    @ np.asarray(self.dataset.sparse_operator.dirty_image)
                )
            else:
                operated = self.operated_mapping_matrix
                inv_var = self._stacked_inverse_variance
                self.d_vec = operated.T @ (inv_var * self._stacked_data)
        return self.d_vec

    # -----------------------------
    # Dense (reference) route
    # -----------------------------

    @property
    def operated_mapping_matrix(self):
        """
        The dense visibility-space joint response, real and imaginary parts
        stacked row-wise: T([f | G]) of shape [2 n_vis, n_src + n_dpsi].
        Small visibility counts only — the parity reference for the sparse
        route.
        """
        if not hasattr(self, "_operated_mapping_matrix"):
            transformed = self.dataset.transformer.transform_mapping_matrix(
                self.real_space_mapping_matrix
            )
            self._operated_mapping_matrix = np.vstack(
                [np.real(transformed), np.imag(transformed)]
            )
        return self._operated_mapping_matrix

    @property
    def _stacked_data(self):
        data = np.asarray(self.dataset.data)
        return np.concatenate([data.real, data.imag])

    @property
    def _stacked_noise(self):
        noise = np.asarray(self.dataset.noise_map)
        return np.concatenate([noise.real, noise.imag])

    @property
    def _stacked_inverse_variance(self):
        return 1.0 / self._stacked_noise**2

    # -----------------------------
    # Solve + evidence
    # -----------------------------

    @property
    def curvature_regularization_matrix(self):
        return self.curvature_matrix + self.regularization_matrix

    def solve_src_dpsi(self, return_error: bool = False):
        curve_reg = np.asarray(self.curvature_regularization_matrix)
        d_vec = np.asarray(self.data_vector)
        if return_error:
            return (
                np.linalg.solve(curve_reg, d_vec),
                np.linalg.inv(curve_reg),
            )
        return np.linalg.solve(curve_reg, d_vec)

    @property
    def model_visibilities(self):
        """
        The model visibilities of the solved joint system, via one forward
        transform of the reconstructed real-space image (cost: a single
        NUFFT, independent of the route).
        """
        if not hasattr(self, "_model_visibilities"):
            model_image_slim = self.real_space_mapping_matrix @ self.src_dpsi_slim
            model_image = aa.Array2D(
                values=model_image_slim, mask=self.dataset.real_space_mask
            )
            self._model_visibilities = np.asarray(
                self.dataset.transformer.visibilities_from(image=model_image)
            )
        return self._model_visibilities

    @property
    def log_evidence(self):
        """
        The Bayesian evidence of the joint visibility-space inversion: the
        complex-noise normalization, the log-determinants of the
        curvature+regularization and regularization matrices, the joint
        regularization penalty and the visibility chi-squared (real and
        imaginary parts).
        """
        if not hasattr(self, "src_dpsi_slim"):
            self.src_dpsi_slim = self.solve_src_dpsi()

        noise = np.asarray(self.dataset.noise_map)

        self.noise_term = -0.5 * float(
            aa.util.fit.noise_normalization_complex_from(noise_map=noise)
        )

        curve_reg = np.asarray(self.curvature_regularization_matrix)
        sign, logval = np.linalg.slogdet(curve_reg)
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

        data = np.asarray(self.dataset.data)
        residual = data - self.model_visibilities
        chi2 = float(
            np.sum((residual.real / noise.real) ** 2)
            + np.sum((residual.imag / noise.imag) ** 2)
        )
        self.chi2_term = -0.5 * chi2

        return (
            self.noise_term
            + self.log_det_curve_reg_term
            + self.log_det_reg_term
            + self.reg_cov_term
            + self.chi2_term
        )

    @property
    def best_fit_source(self):
        n_s = self.src_regularization_matrix.shape[0]
        return self.src_dpsi_slim[0:n_s]

    @property
    def best_fit_dpsi(self):
        n_s = self.src_regularization_matrix.shape[0]
        return self.src_dpsi_slim[n_s:]
