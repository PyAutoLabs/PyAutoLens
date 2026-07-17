import numpy as np
import pytest

import autoarray as aa
import autolens as al
from autolens.potential_correction.fit_interferometer import (
    FitDpsiSrcInterferometer,
)


def fit_from(dataset, use_sparse_operator):
    lens = al.Galaxy(
        redshift=0.5,
        mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.0),
    )
    source_galaxy = al.Galaxy(
        redshift=1.0,
        bulge=al.lp.GaussianSph(centre=(0.0, 0.0), intensity=1.0, sigma=0.5),
    )
    return FitDpsiSrcInterferometer(
        dataset=dataset,
        lens_start=lens,
        source_start=al.pc.AnalyticSrcFactory(source_galaxy=source_galaxy),
        dpsi_pixelization=al.pc.DpsiPixelization(
            mesh=al.pc.RegularDpsiMesh(factor=1),
            regularization=aa.reg.CurvatureMask(coefficient=1.0),
        ),
        src_pixelization=al.Pixelization(
            mesh=al.mesh.RectangularUniform(shape=(3, 3)),
            regularization=al.reg.Constant(coefficient=1.0),
        ),
        use_sparse_operator=use_sparse_operator,
    )


def test__dense_route__end_to_end_evidence_is_finite(interferometer_7):
    fit = fit_from(interferometer_7, use_sparse_operator=False)

    evidence = fit.log_evidence

    assert np.isfinite(evidence)

    n_src = fit.src_regularization_matrix.shape[0]
    n_dpsi = np.count_nonzero(~fit.pair_dpsi_data_obj.mask_dpsi)
    assert fit.best_fit_source.shape == (n_src,)
    assert fit.best_fit_dpsi.shape == (n_dpsi,)
    assert fit.operated_mapping_matrix.shape == (
        2 * np.asarray(interferometer_7.data).shape[0],
        n_src + n_dpsi,
    )


def test__sparse_route__matches_dense_route(interferometer_7):
    dataset_sparse = interferometer_7.apply_sparse_operator()

    fit_dense = fit_from(interferometer_7, use_sparse_operator=False)
    fit_sparse = fit_from(dataset_sparse, use_sparse_operator=True)

    curvature_dense = np.asarray(fit_dense.curvature_matrix)
    curvature_sparse = np.asarray(fit_sparse.curvature_matrix)
    assert curvature_sparse == pytest.approx(curvature_dense, rel=1e-4, abs=1e-6)

    data_vector_dense = np.asarray(fit_dense.data_vector)
    data_vector_sparse = np.asarray(fit_sparse.data_vector)
    assert data_vector_sparse == pytest.approx(data_vector_dense, rel=1e-4, abs=1e-6)

    evidence_dense = fit_dense.log_evidence
    evidence_sparse = fit_sparse.log_evidence
    assert evidence_sparse == pytest.approx(evidence_dense, rel=1e-4)


def test__sparse_route__requires_sparse_operator(interferometer_7):
    with pytest.raises(aa.exc.InversionException):
        fit_from(interferometer_7, use_sparse_operator=True)


def test__evidence_terms__match_hand_computed_dense_formulation(interferometer_7):
    fit = fit_from(interferometer_7, use_sparse_operator=False)
    evidence = fit.log_evidence

    noise = np.asarray(interferometer_7.noise_map)
    data = np.asarray(interferometer_7.data)

    operated = fit.operated_mapping_matrix
    inv_var = fit._stacked_inverse_variance
    reg = fit.regularization_matrix
    curve_reg = operated.T @ (operated * inv_var[:, None]) + reg
    solution = np.linalg.solve(
        curve_reg, operated.T @ (inv_var * fit._stacked_data)
    )

    residual = data - fit.model_visibilities
    chi2 = -0.5 * float(
        np.sum((residual.real / noise.real) ** 2)
        + np.sum((residual.imag / noise.imag) ** 2)
    )

    assert fit.src_dpsi_slim == pytest.approx(solution, rel=1e-8)
    assert fit.chi2_term == pytest.approx(chi2, rel=1e-8)
    assert fit.noise_term == pytest.approx(
        -0.5 * float(aa.util.fit.noise_normalization_complex_from(noise_map=noise)),
        rel=1e-10,
    )
    assert np.isfinite(evidence)
