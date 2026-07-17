import numpy as np
import pytest
from scipy.sparse import csr_matrix

import autoarray as aa
import autolens as al
from autolens.potential_correction.fit import FitDpsiImaging, FitDpsiSrcImaging
from autolens.potential_correction.pixelization import DpsiLinearObj


class MockNoiseMap:
    def __init__(self, values):
        self.slim = values


class MockMaskedImaging:
    def __init__(self, noise_values):
        self.noise_map = MockNoiseMap(noise_values)


def injected_dpsi_fit():
    """
    A FitDpsiImaging with every cached matrix pre-injected (the pattern of
    the original package's unit tests), so the evidence terms can be
    verified against hand-computed linear algebra with no imaging data.
    """
    fit = object.__new__(FitDpsiImaging)

    fit.anchor_points = None
    fit.input_image_residual = np.array([0.1, -0.2, 0.3, -0.1])
    fit.map_mat = np.array(
        [
            [0.3, 0.1, -0.2],
            [-0.1, 0.4, 0.1],
            [0.2, -0.3, 0.5],
            [0.0, 0.2, -0.4],
        ]
    )
    w = np.array([1.0, 2.0, 3.0, 4.0])
    fit.inv_cov_mat = np.diag(w)
    fit.reg_mat = csr_matrix(np.diag([0.5, 0.5, 0.5]))
    fit.d_vec = fit.map_mat.T @ fit.inv_cov_mat @ fit.input_image_residual
    fit.curve_reg_mat = (
        fit.map_mat.T @ fit.inv_cov_mat @ fit.map_mat + fit.reg_mat.toarray()
    )
    fit.psf_mat = np.eye(4)
    fit.src_grad_mat = np.eye(4)
    fit.dpsi_grad_mat = np.ones((4, 3)) * 0.1

    noise = 1.0 / np.sqrt(w)
    fit.masked_imaging = MockMaskedImaging(noise_values=noise)

    return fit


def test__fit_dpsi_imaging__solve_matches_normal_equations():
    fit = injected_dpsi_fit()

    dpsi = fit.solve_dpsi()

    assert fit.curve_reg_mat @ dpsi == pytest.approx(fit.d_vec, abs=1.0e-12)


def test__fit_dpsi_imaging__log_evidence_matches_hand_computed_terms():
    fit = injected_dpsi_fit()

    evidence = fit.log_evidence

    noise = fit.masked_imaging.noise_map.slim
    dpsi = np.linalg.solve(fit.curve_reg_mat, fit.d_vec)
    model = fit.map_mat @ dpsi
    residual = fit.input_image_residual - model

    noise_term = -0.5 * np.sum(np.log(2 * np.pi * noise**2))
    log_det_curve = -0.5 * np.linalg.slogdet(fit.curve_reg_mat)[1]
    log_det_reg = 0.5 * np.linalg.slogdet(fit.reg_mat.toarray())[1]
    reg_cov = -0.5 * float(dpsi.T @ fit.reg_mat @ dpsi)
    chi2 = -0.5 * float(np.sum((residual / noise) ** 2))

    assert evidence == pytest.approx(
        noise_term + log_det_curve + log_det_reg + reg_cov + chi2, rel=1.0e-10
    )


def test__fit_dpsi_imaging__rescaled_dpsi_without_anchors_is_identity():
    fit = injected_dpsi_fit()
    fit.log_evidence

    dpsi_new, ay, ax, c = fit.rescaled_dpsi

    assert dpsi_new == pytest.approx(fit.dpsi_slim)
    assert (ay, ax, c) == (0.0, 0.0, 0.0)


def test__dpsi_linear_obj__works_for_mask_and_kernel_regularizations():
    mask = np.ones((6, 6), dtype=bool)
    mask[1:5, 1:5] = False
    ys, xs = np.where(~mask)
    points = np.vstack([ys.astype(float), xs.astype(float)]).T

    linear_obj = DpsiLinearObj(mask=mask, points=points)

    assert linear_obj.params == 16

    reg_curvature = aa.reg.CurvatureMask(coefficient=2.0)
    matrix_curvature = reg_curvature.regularization_matrix_from(linear_obj=linear_obj)
    assert matrix_curvature.shape == (16, 16)
    assert matrix_curvature == pytest.approx(matrix_curvature.T, abs=1.0e-12)

    reg_kernel = aa.reg.MaternKernel(coefficient=1.0, scale=1.0, nu=0.5)
    matrix_kernel = reg_kernel.regularization_matrix_from(linear_obj=linear_obj)
    assert matrix_kernel.shape == (16, 16)
    assert np.isfinite(matrix_kernel).all()


def test__fit_dpsi_src_imaging__end_to_end_evidence_is_finite(masked_imaging_7x7):
    lens = al.Galaxy(
        redshift=0.5,
        mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.0),
    )
    source_galaxy = al.Galaxy(
        redshift=1.0,
        bulge=al.lp.SersicSph(
            centre=(0.0, 0.0), intensity=1.0, effective_radius=0.5
        ),
    )
    source_start = al.pc.AnalyticSrcFactory(source_galaxy=source_galaxy)

    dpsi_pixelization = al.pc.DpsiPixelization(
        mesh=al.pc.RegularDpsiMesh(factor=1),
        regularization=aa.reg.CurvatureMask(coefficient=1.0),
    )
    src_pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )

    fit = al.pc.FitDpsiSrcImaging(
        masked_imaging=masked_imaging_7x7,
        lens_start=lens,
        source_start=source_start,
        dpsi_pixelization=dpsi_pixelization,
        src_pixelization=src_pixelization,
    )

    evidence = fit.log_evidence

    assert np.isfinite(evidence)

    n_src = fit.src_regularization_matrix.shape[0]
    n_dpsi = np.count_nonzero(~fit.pair_dpsi_data_obj.mask_dpsi)
    assert fit.best_fit_source.shape == (n_src,)
    assert fit.best_fit_dpsi.shape == (n_dpsi,)
    assert fit.mapping_matrix.shape[1] == n_src + n_dpsi


def test__fit_dpsi_imaging__end_to_end_evidence_is_finite(masked_imaging_7x7):
    n_data = int(np.count_nonzero(~np.asarray(masked_imaging_7x7.mask)))

    rng = np.random.default_rng(3)
    image_residual = rng.normal(scale=0.1, size=n_data)
    source_gradient = np.ones((n_data, 2))

    dpsi_pixelization = al.pc.DpsiPixelization(
        mesh=al.pc.RegularDpsiMesh(factor=1),
        regularization=aa.reg.CurvatureMask(coefficient=1.0),
    )

    fit = al.pc.FitDpsiImaging(
        masked_imaging=masked_imaging_7x7,
        image_residual=image_residual,
        source_gradient=source_gradient,
        dpsi_pixelization=dpsi_pixelization,
    )

    evidence = fit.log_evidence

    assert np.isfinite(evidence)
    assert fit.dpsi_slim.shape == (
        np.count_nonzero(~fit.pair_dpsi_data_obj.mask_dpsi),
    )
