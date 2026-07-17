import numpy as np
import pytest
from scipy.linalg import block_diag as scipy_block_diag
from scipy.sparse import csr_matrix

from autolens.potential_correction import dense_util
from autolens.potential_correction import util as pc_util


def test__as_dense__handles_sparse_and_dense():
    dense = np.array([[1.0, 2.0], [3.0, 4.0]])

    assert dense_util.as_dense(dense) == pytest.approx(dense)
    assert dense_util.as_dense(csr_matrix(dense)) == pytest.approx(dense)


def test__dense_block_diag_from__matches_scipy():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0]])
    c = np.eye(3) * 7.0

    result = dense_util.dense_block_diag_from(a, b, c)

    assert result == pytest.approx(scipy_block_diag(a, b, c))


def test__source_gradient_matrix_dense_from__matches_sparse_version():
    rng = np.random.default_rng(0)
    source_gradient = rng.normal(size=(5, 2))

    dense = dense_util.source_gradient_matrix_dense_from(source_gradient)
    sparse = pc_util.source_gradient_matrix_from(source_gradient).toarray()

    assert dense == pytest.approx(sparse)


def test__dpsi_gradient_matrix_dense_from__matches_sparse_version():
    rng = np.random.default_rng(1)
    itp = csr_matrix(rng.normal(size=(6, 4)))
    Hx = csr_matrix(rng.normal(size=(4, 4)))
    Hy = csr_matrix(rng.normal(size=(4, 4)))

    dense = dense_util.dpsi_gradient_matrix_dense_from(itp, Hx, Hy)
    sparse = pc_util.dpsi_gradient_matrix_from(itp, Hx, Hy).toarray()

    assert dense == pytest.approx(sparse)


def test__dpsi_mapping_matrix_from__is_minus_psf_srcgrad_dpsigrad():
    rng = np.random.default_rng(2)
    psf = rng.normal(size=(4, 4))
    src_grad = rng.normal(size=(4, 8))
    dpsi_grad = rng.normal(size=(8, 3))

    result = dense_util.dpsi_mapping_matrix_from(psf, src_grad, dpsi_grad)

    assert result == pytest.approx(-1.0 * psf @ src_grad @ dpsi_grad)


def joint_problem(n_data=6, n_src=3, n_dpsi=2, seed=3):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=n_data)
    noise = rng.uniform(0.5, 1.5, size=n_data)
    mapping = rng.normal(size=(n_data, n_src + n_dpsi))
    A = rng.normal(size=(n_src, n_src))
    src_reg = A @ A.T + n_src * np.eye(n_src)
    B = rng.normal(size=(n_dpsi, n_dpsi))
    dpsi_reg = B @ B.T + n_dpsi * np.eye(n_dpsi)
    return data, noise, mapping, src_reg, dpsi_reg


def reference_joint_evidence(data, noise, mapping, src_reg, dpsi_reg):
    """The phase-3 numpy evidence formulation, computed directly."""
    reg = scipy_block_diag(src_reg, dpsi_reg)
    inv_cov = np.diag(1.0 / noise**2)
    curve_reg = mapping.T @ inv_cov @ mapping + reg
    d_vec = mapping.T @ inv_cov @ data
    solution = np.linalg.solve(curve_reg, d_vec)
    model = mapping @ solution

    noise_term = -0.5 * np.sum(np.log(2 * np.pi * noise**2))
    logdet_curve = -0.5 * np.linalg.slogdet(curve_reg)[1]
    logdet_reg = 0.5 * (
        np.linalg.slogdet(src_reg)[1] + np.linalg.slogdet(dpsi_reg)[1]
    )
    reg_cov = -0.5 * float(solution @ reg @ solution)
    chi2 = -0.5 * float(np.sum(((data - model) / noise) ** 2))
    return noise_term + logdet_curve + logdet_reg + reg_cov + chi2, solution


def test__log_evidence_joint_dense_from__matches_reference_formulation():
    data, noise, mapping, src_reg, dpsi_reg = joint_problem()

    result = dense_util.log_evidence_joint_dense_from(
        data, noise, mapping, src_reg, dpsi_reg
    )

    expected, solution = reference_joint_evidence(
        data, noise, mapping, src_reg, dpsi_reg
    )
    assert bool(result["valid"])
    assert float(result["evidence"]) == pytest.approx(expected, rel=1.0e-10)
    assert result["solution"] == pytest.approx(solution, rel=1.0e-10)


def test__log_evidence_from_fixed_curvature__matches_joint_dense():
    data, noise, mapping, src_reg, dpsi_reg = joint_problem()

    full = dense_util.log_evidence_joint_dense_from(
        data, noise, mapping, src_reg, dpsi_reg
    )

    inv_var = 1.0 / noise**2
    curvature = mapping.T @ (mapping * inv_var[:, None])
    data_vector = mapping.T @ (inv_var * data)
    noise_term = -0.5 * np.sum(np.log(2 * np.pi * noise**2))

    fast = dense_util.log_evidence_from_fixed_curvature(
        curvature_matrix=curvature,
        data_vector=data_vector,
        data_slim=data,
        mapping_matrix=mapping,
        inv_var=inv_var,
        noise_term=noise_term,
        src_reg_matrix=src_reg,
        dpsi_reg_matrix=dpsi_reg,
    )

    assert bool(fast["valid"])
    assert float(fast["evidence"]) == pytest.approx(
        float(full["evidence"]), rel=1.0e-10
    )
    assert fast["solution"] == pytest.approx(np.asarray(full["solution"]), rel=1.0e-8)


def test__log_evidence_from_fixed_curvature__invalid_on_non_pd_regularization():
    data, noise, mapping, src_reg, dpsi_reg = joint_problem()

    result = dense_util.log_evidence_from_fixed_curvature(
        curvature_matrix=mapping.T @ mapping,
        data_vector=mapping.T @ data,
        data_slim=data,
        mapping_matrix=mapping,
        inv_var=1.0 / noise**2,
        noise_term=0.0,
        src_reg_matrix=-1.0 * src_reg,  # not positive definite
        dpsi_reg_matrix=dpsi_reg,
    )

    assert not bool(result["valid"])


def test__log_evidence_dpsi_dense_from__matches_reference_formulation():
    rng = np.random.default_rng(4)
    n_data, n_dpsi = 5, 3
    data = rng.normal(size=n_data)
    noise = rng.uniform(0.5, 1.5, size=n_data)
    mapping = rng.normal(size=(n_data, n_dpsi))
    reg = np.diag(rng.uniform(0.5, 2.0, size=n_dpsi))

    result = dense_util.log_evidence_dpsi_dense_from(data, noise, mapping, reg)

    inv_cov = np.diag(1.0 / noise**2)
    curve_reg = mapping.T @ inv_cov @ mapping + reg
    solution = np.linalg.solve(curve_reg, mapping.T @ inv_cov @ data)
    model = mapping @ solution
    expected = (
        -0.5 * np.sum(np.log(2 * np.pi * noise**2))
        - 0.5 * np.linalg.slogdet(curve_reg)[1]
        + 0.5 * np.linalg.slogdet(reg)[1]
        - 0.5 * float(solution @ reg @ solution)
        - 0.5 * float(np.sum(((data - model) / noise) ** 2))
    )

    assert bool(result["valid"])
    assert float(result["evidence"]) == pytest.approx(expected, rel=1.0e-10)


def test__lm_cost_and_hessian__hand_computed():
    data, noise, mapping, src_reg, dpsi_reg = joint_problem()
    n_src = src_reg.shape[0]
    inv_var = 1.0 / noise**2
    L = mapping[:, :n_src]
    J_dpsi = mapping[:, n_src:]

    rng = np.random.default_rng(5)
    x = rng.normal(size=mapping.shape[1])
    s, dpsi = x[:n_src], x[n_src:]

    cost, chi2, reg_s, reg_dpsi = dense_util.lm_cost_from(
        data, inv_var, s, dpsi, L, src_reg, dpsi_reg
    )
    residual = data - L @ s
    assert float(chi2) == pytest.approx(0.5 * np.sum(inv_var * residual**2))
    assert float(reg_s) == pytest.approx(0.5 * s @ src_reg @ s)
    assert float(cost) == pytest.approx(float(chi2) + float(reg_s) + float(reg_dpsi))

    H, minus_gradient, res, chi2_h, _, _, cost_h = (
        dense_util.lm_hessian_and_gradient_from(
            data, inv_var, x, L, J_dpsi, src_reg, dpsi_reg
        )
    )
    J = np.hstack([L, J_dpsi])
    R = scipy_block_diag(src_reg, dpsi_reg)
    assert H == pytest.approx(J.T @ (J * inv_var[:, None]) + R)
    assert minus_gradient == pytest.approx(J.T @ (inv_var * residual) - R @ x)
    assert float(cost_h) == pytest.approx(float(cost))


def test__solve_lm_step_from__unconstrained_and_constrained():
    data, noise, mapping, src_reg, dpsi_reg = joint_problem()
    n_src = src_reg.shape[0]
    inv_var = 1.0 / noise**2
    x = np.zeros(mapping.shape[1])

    H, minus_gradient, *_ = dense_util.lm_hessian_and_gradient_from(
        data, inv_var, x, mapping[:, :n_src], mapping[:, n_src:], src_reg, dpsi_reg
    )

    mu = 0.7
    step = dense_util.solve_lm_step_from(H, minus_gradient, mu)
    assert (H + mu * np.eye(H.shape[0])) @ step == pytest.approx(
        minus_gradient, rel=1.0e-8
    )

    # a single constraint: the dpsi block sums to zero after the step
    C = np.zeros((1, mapping.shape[1]))
    C[0, n_src:] = 1.0
    step_c = dense_util.solve_lm_step_from(
        H, minus_gradient, mu, constraint_matrix=C, x=x
    )
    assert float((C @ (x + step_c))[0]) == pytest.approx(0.0, abs=1.0e-8)


def test__log_evidence_lm_from__matches_hand_computed():
    data, noise, mapping, src_reg, dpsi_reg = joint_problem()
    n_src = src_reg.shape[0]
    L = mapping[:, :n_src]
    J_dpsi = mapping[:, n_src:]

    rng = np.random.default_rng(6)
    s = rng.normal(size=n_src)
    dpsi = rng.normal(size=dpsi_reg.shape[0])

    result = dense_util.log_evidence_lm_from(
        data, noise, s, dpsi, L, L, J_dpsi, src_reg, dpsi_reg
    )

    inv_var = 1.0 / noise**2
    residual = data - L @ s
    chi2 = np.sum(inv_var * residual**2)
    reg_val = s @ src_reg @ s + dpsi @ dpsi_reg @ dpsi
    J = np.hstack([L, J_dpsi])
    R = scipy_block_diag(src_reg, dpsi_reg)
    H = J.T @ (J * inv_var[:, None]) + R
    expected = 0.5 * (
        np.linalg.slogdet(src_reg)[1]
        + np.linalg.slogdet(dpsi_reg)[1]
        - np.linalg.slogdet(H)[1]
        - chi2
        - reg_val
    )

    assert float(result) == pytest.approx(expected, rel=1.0e-10)
