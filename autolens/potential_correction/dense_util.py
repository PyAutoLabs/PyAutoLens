"""
Dense linear-algebra kernels of the gravitational-imaging (potential
correction) technique, written once against the PyAuto ``xp`` API: every
function takes ``xp=np`` and runs identically under numpy (the default) or
``jax.numpy`` (pass ``xp=jnp`` for JIT-able, accelerator-ready execution).
Sparse-matrix inputs are densified on entry; no scipy.sparse arithmetic
happens inside the kernels.

These kernels replace the scipy.sparse + numpy paths of ``fit.py`` in the
performance-critical settings of the iterative solver (``iterative.py``) and
evidence-based hyper-parameter sampling.

Ported from the ``potential_correction`` package of Cao et al. 2025
(https://github.com/caoxiaoyue/lensing_potential_correction), refactored from
its ``jax_ops`` module onto the ecosystem ``xp`` convention. If you use this
functionality in your research, please cite Cao et al. 2025; citation
materials are provided at
https://github.com/caoxiaoyue/potential_correction_paper.
"""

import numpy as np


def as_dense(matrix, xp=np):
    """
    Converts a numpy / scipy-sparse / jax array to a dense 2D ``xp`` array.
    """
    toarray = getattr(matrix, "toarray", None)
    if toarray is not None:
        matrix = toarray()
    return xp.asarray(matrix)


def dense_block_diag_from(*blocks, xp=np):
    """
    A dense block-diagonal matrix from two or more dense square blocks.
    """
    blocks = tuple(as_dense(b, xp=xp) for b in blocks)
    sizes = [b.shape[0] for b in blocks]
    total = sum(sizes)
    rows = []
    offset = 0
    for b, size in zip(blocks, sizes):
        left = xp.zeros((size, offset), dtype=b.dtype)
        right = xp.zeros((size, total - offset - size), dtype=b.dtype)
        rows.append(xp.concatenate([left, b, right], axis=1))
        offset += size
    return xp.concatenate(rows, axis=0)


def inverse_noise_variance_from(noise_slim, xp=np):
    """
    The diagonal of the inverse noise covariance as a 1D vector, 1/sigma^2.
    """
    noise = xp.asarray(noise_slim)
    return 1.0 / (noise * noise)


def source_gradient_matrix_dense_from(source_gradient, xp=np):
    """
    The dense source-gradient matrix D_s of shape [n_data, 2 * n_data]:
    column 2i holds the x-derivative and column 2i+1 the y-derivative of the
    source at data pixel i (the dense form of
    ``util.source_gradient_matrix_from``).
    """
    sg = xp.asarray(source_gradient)
    n_data = sg.shape[0]
    rows = xp.arange(n_data)
    matrix = xp.zeros((n_data, 2 * n_data), dtype=sg.dtype)
    if xp is np:
        matrix[rows, 2 * rows] = sg[:, 1]
        matrix[rows, 2 * rows + 1] = sg[:, 0]
    else:
        matrix = matrix.at[rows, 2 * rows].set(sg[:, 1])
        matrix = matrix.at[rows, 2 * rows + 1].set(sg[:, 0])
    return matrix


def dpsi_gradient_matrix_dense_from(itp_mat, Hx, Hy, xp=np):
    """
    The dense dpsi-gradient operator D_psi of shape [2 * n_data, n_dpsi],
    interleaving the x and y gradient rows per data pixel (the dense form of
    ``util.dpsi_gradient_matrix_from``).
    """
    itp = as_dense(itp_mat, xp=xp)
    Hx_d = as_dense(Hx, xp=xp)
    Hy_d = as_dense(Hy, xp=xp)
    Hx_itp = itp @ Hx_d
    Hy_itp = itp @ Hy_d
    return xp.stack((Hx_itp, Hy_itp), axis=1).reshape(
        2 * itp.shape[0], Hx_d.shape[1]
    )


def dpsi_mapping_matrix_from(psf_mat, src_grad_mat, dpsi_grad_mat, xp=np):
    """
    The dpsi mapping matrix -B D_s D_psi of shape [n_data, n_dpsi].
    """
    psf = as_dense(psf_mat, xp=xp)
    sg = as_dense(src_grad_mat, xp=xp)
    dg = as_dense(dpsi_grad_mat, xp=xp)
    return -1.0 * psf @ sg @ dg


def log_evidence_joint_dense_from(
    data_slim, noise_slim, mapping_matrix, src_reg_matrix, dpsi_reg_matrix, xp=np
):
    """
    The Bayesian evidence of the joint source+dpsi inversion, decomposed into
    its terms, from dense inputs.

    Returns
    -------
    A dict with keys ``evidence``, ``valid`` (positive determinant signs),
    ``solution``, ``model_image``, ``curvature_reg_matrix``, ``noise_term``,
    ``logdet_curve``, ``logdet_src``, ``logdet_dpsi``, ``chi2`` and
    ``reg_cov``. Under ``xp=jax.numpy`` invalid decompositions surface as
    ``valid=False`` rather than raising (JIT-compatible); callers must check
    it.
    """
    data = xp.asarray(data_slim)
    noise = xp.asarray(noise_slim)
    mapping = as_dense(mapping_matrix, xp=xp)
    src_reg = as_dense(src_reg_matrix, xp=xp)
    dpsi_reg = as_dense(dpsi_reg_matrix, xp=xp)

    reg_matrix = dense_block_diag_from(src_reg, dpsi_reg, xp=xp)
    inv_var = inverse_noise_variance_from(noise, xp=xp)

    weighted_mapping = mapping * inv_var[:, None]
    curvature_matrix = mapping.T @ weighted_mapping
    data_vector = mapping.T @ (inv_var * data)

    curvature_reg_matrix = curvature_matrix + reg_matrix
    solution = xp.linalg.solve(curvature_reg_matrix, data_vector)
    model_image = mapping @ solution

    sign_curve, logdet_curve = xp.linalg.slogdet(curvature_reg_matrix)
    sign_src, logdet_src = xp.linalg.slogdet(src_reg)
    sign_dpsi, logdet_dpsi = xp.linalg.slogdet(dpsi_reg)

    residual = data - model_image
    reg_cov = solution @ (reg_matrix @ solution)
    chi2 = xp.sum((residual / noise) ** 2.0)
    noise_term = -0.5 * xp.sum(xp.log(2.0 * xp.pi * noise * noise))

    evidence = (
        noise_term
        - 0.5 * logdet_curve
        + 0.5 * (logdet_src + logdet_dpsi)
        - 0.5 * reg_cov
        - 0.5 * chi2
    )

    valid = (sign_curve > 0) & (sign_src > 0) & (sign_dpsi > 0)

    return {
        "evidence": evidence,
        "valid": valid,
        "solution": solution,
        "model_image": model_image,
        "curvature_reg_matrix": curvature_reg_matrix,
        "noise_term": noise_term,
        "logdet_curve": logdet_curve,
        "logdet_src": logdet_src,
        "logdet_dpsi": logdet_dpsi,
        "chi2": chi2,
        "reg_cov": reg_cov,
    }


def log_evidence_from_fixed_curvature(
    curvature_matrix,
    data_vector,
    data_slim,
    mapping_matrix,
    inv_var,
    noise_term,
    src_reg_matrix,
    dpsi_reg_matrix,
    xp=np,
):
    """
    The joint evidence with the curvature matrix M^T C^-1 M and data vector
    precomputed — the fast path when only the regularization matrices change
    between hyper-parameter samples. Log-determinants come from Cholesky
    factors (one O(n^3) each); a failed factorization surfaces as
    ``valid=False``.

    Returns the same dict as ``log_evidence_joint_dense_from`` (without
    ``curvature_reg_matrix``).
    """
    if xp is np:
        from scipy.linalg import cho_solve
    else:
        from jax.scipy.linalg import cho_solve

    F = as_dense(curvature_matrix, xp=xp)
    d_vec = xp.asarray(data_vector)
    data = xp.asarray(data_slim)
    mapping = as_dense(mapping_matrix, xp=xp)
    iv = xp.asarray(inv_var)
    src_reg = as_dense(src_reg_matrix, xp=xp)
    dpsi_reg = as_dense(dpsi_reg_matrix, xp=xp)

    reg_matrix = dense_block_diag_from(src_reg, dpsi_reg, xp=xp)
    curve_reg = F + reg_matrix

    if xp is np:
        try:
            L_cr = np.linalg.cholesky(curve_reg)
            L_src = np.linalg.cholesky(src_reg)
            L_dpsi = np.linalg.cholesky(dpsi_reg)
        except np.linalg.LinAlgError:
            return {"evidence": -np.inf, "valid": False}
    else:
        L_cr = xp.linalg.cholesky(curve_reg)
        L_src = xp.linalg.cholesky(src_reg)
        L_dpsi = xp.linalg.cholesky(dpsi_reg)

    solution = cho_solve((L_cr, True), d_vec)
    logdet_curve = 2.0 * xp.sum(xp.log(xp.diag(L_cr)))
    logdet_src = 2.0 * xp.sum(xp.log(xp.diag(L_src)))
    logdet_dpsi = 2.0 * xp.sum(xp.log(xp.diag(L_dpsi)))

    model_image = mapping @ solution
    residual = data - model_image
    chi2 = xp.sum(iv * residual * residual)
    reg_cov = solution @ (reg_matrix @ solution)

    evidence = (
        noise_term
        - 0.5 * logdet_curve
        + 0.5 * (logdet_src + logdet_dpsi)
        - 0.5 * reg_cov
        - 0.5 * chi2
    )

    if xp is np:
        valid = bool(
            np.all(np.diag(L_cr) > 0)
            & np.all(np.diag(L_src) > 0)
            & np.all(np.diag(L_dpsi) > 0)
        )
    else:
        valid = (
            xp.all(xp.diag(L_cr) > 0)
            & xp.all(xp.diag(L_src) > 0)
            & xp.all(xp.diag(L_dpsi) > 0)
        )

    return {
        "evidence": evidence,
        "valid": valid,
        "solution": solution,
        "model_image": model_image,
        "noise_term": noise_term,
        "logdet_curve": logdet_curve,
        "logdet_src": logdet_src,
        "logdet_dpsi": logdet_dpsi,
        "chi2": chi2,
        "reg_cov": reg_cov,
    }


def log_evidence_dpsi_dense_from(
    data_slim, noise_slim, mapping_matrix, reg_matrix, xp=np
):
    """
    The evidence of the dpsi-only inversion of an image residual, from dense
    inputs.

    Returns a dict with ``evidence``, ``valid``, ``solution``,
    ``model_image`` and ``curvature_reg_matrix``.
    """
    data = xp.asarray(data_slim)
    noise = xp.asarray(noise_slim)
    mapping = as_dense(mapping_matrix, xp=xp)
    reg = as_dense(reg_matrix, xp=xp)

    inv_var = inverse_noise_variance_from(noise, xp=xp)
    weighted_mapping = mapping * inv_var[:, None]
    curvature_matrix = mapping.T @ weighted_mapping
    data_vector = mapping.T @ (inv_var * data)

    curvature_reg_matrix = curvature_matrix + reg
    solution = xp.linalg.solve(curvature_reg_matrix, data_vector)
    model_image = mapping @ solution

    sign_curve, logdet_curve = xp.linalg.slogdet(curvature_reg_matrix)
    sign_reg, logdet_reg = xp.linalg.slogdet(reg)

    residual = data - model_image
    reg_cov = solution @ (reg @ solution)
    chi2 = xp.sum((residual / noise) ** 2.0)
    noise_term = -0.5 * xp.sum(xp.log(2.0 * xp.pi * noise * noise))

    evidence = (
        noise_term
        - 0.5 * logdet_curve
        + 0.5 * logdet_reg
        - 0.5 * reg_cov
        - 0.5 * chi2
    )

    return {
        "evidence": evidence,
        "valid": (sign_curve > 0) & (sign_reg > 0),
        "solution": solution,
        "model_image": model_image,
        "curvature_reg_matrix": curvature_reg_matrix,
    }


def lm_cost_from(data_slim, inv_var, s, dpsi, L, src_reg_matrix, dpsi_reg_matrix, xp=np):
    """
    The Levenberg-Marquardt cost of a candidate (s, dpsi):
    0.5 chi^2 + 0.5 s^T R_s s + 0.5 dpsi^T R_dpsi dpsi.

    Returns (cost, chi2, reg_s, reg_dpsi).
    """
    data = xp.asarray(data_slim)
    iv = xp.asarray(inv_var)
    s_arr = xp.asarray(s)
    dpsi_arr = xp.asarray(dpsi)
    L_d = as_dense(L, xp=xp)
    src_reg = as_dense(src_reg_matrix, xp=xp)
    dpsi_reg = as_dense(dpsi_reg_matrix, xp=xp)

    residual = data - L_d @ s_arr
    chi2 = 0.5 * xp.sum(iv * residual * residual)
    reg_s = 0.5 * (s_arr @ (src_reg @ s_arr))
    reg_dpsi = 0.5 * (dpsi_arr @ (dpsi_reg @ dpsi_arr))
    cost = chi2 + reg_s + reg_dpsi

    return cost, chi2, reg_s, reg_dpsi


def lm_hessian_and_gradient_from(
    data_slim, inv_var, x, L, J_dpsi, src_reg_matrix, dpsi_reg_matrix, xp=np
):
    """
    The LM Hessian H = J^T C^-1 J + R, the negated gradient
    J^T C^-1 r - R x, and the current residual/cost terms, for the combined
    state x = [s | dpsi].

    Returns (H, minus_gradient, residual, chi2, reg_s, reg_dpsi, cost).
    """
    data = xp.asarray(data_slim)
    iv = xp.asarray(inv_var)
    x_arr = xp.asarray(x)
    L_d = as_dense(L, xp=xp)
    J_d = as_dense(J_dpsi, xp=xp)
    src_reg = as_dense(src_reg_matrix, xp=xp)
    dpsi_reg = as_dense(dpsi_reg_matrix, xp=xp)

    n_s = src_reg.shape[0]
    s = x_arr[:n_s]
    dpsi = x_arr[n_s:]

    J_combined = xp.concatenate([L_d, J_d], axis=1)
    R = dense_block_diag_from(src_reg, dpsi_reg, xp=xp)

    residual = data - L_d @ s
    weighted_J = J_combined * iv[:, None]
    H = J_combined.T @ weighted_J + R
    minus_gradient = J_combined.T @ (iv * residual) - R @ x_arr

    chi2 = 0.5 * xp.sum(iv * residual * residual)
    reg_s = 0.5 * (s @ (src_reg @ s))
    reg_dpsi = 0.5 * (dpsi @ (dpsi_reg @ dpsi))
    cost = chi2 + reg_s + reg_dpsi

    return H, minus_gradient, residual, chi2, reg_s, reg_dpsi, cost


def solve_lm_step_from(H, minus_gradient, mu, constraint_matrix=None, x=None, xp=np):
    """
    The damped LM step delta_x solving (H + mu D) dx = -g with Marquardt
    scaling D = diag(diag(H)) (clipped below at the mean diagonal times
    1e-12 so zero diagonal entries stay damped) — scale-invariant damping,
    required when H's magnitude varies over many orders between datasets
    (e.g. visibility-weighted interferometer curvatures ~1e11 vs imaging
    ~1e4). When a ``constraint_matrix`` C is given, the equality-constrained
    step solves the KKT system enforcing C (x + dx) = 0.
    """
    H_d = as_dense(H, xp=xp)
    g = xp.asarray(minus_gradient)
    n_x = H_d.shape[0]
    diag = xp.diag(H_d)
    diag = xp.clip(diag, 1e-12 * xp.mean(xp.abs(diag)), None)
    H_lm = H_d + mu * xp.diag(diag)

    if constraint_matrix is None:
        return xp.linalg.solve(H_lm, g)

    C = xp.asarray(constraint_matrix)
    x_arr = xp.asarray(x)
    n_c = C.shape[0]
    top = xp.concatenate([H_lm, C.T], axis=1)
    bottom = xp.concatenate(
        [C, xp.zeros((n_c, n_c), dtype=H_d.dtype)], axis=1
    )
    H_kkt = xp.concatenate([top, bottom], axis=0)
    rhs = xp.concatenate([g, -(C @ x_arr)])
    solution = xp.linalg.solve(H_kkt, rhs)
    return solution[:n_x]


def log_evidence_lm_from(
    data_slim,
    noise_slim,
    s,
    dpsi,
    L,
    J_s,
    J_dpsi,
    src_reg_matrix,
    dpsi_reg_matrix,
    xp=np,
):
    """
    The Laplace-approximation log evidence at an LM solution (s, dpsi),
    without the noise-normalization term (add it separately):
    0.5 [ logdet R_s + logdet R_dpsi - logdet H - chi^2 - s^T R_s s
    - dpsi^T R_dpsi dpsi ].
    """
    data = xp.asarray(data_slim)
    noise = xp.asarray(noise_slim)
    s_arr = xp.asarray(s)
    dpsi_arr = xp.asarray(dpsi)
    L_d = as_dense(L, xp=xp)
    J_s_d = as_dense(J_s, xp=xp)
    J_dpsi_d = as_dense(J_dpsi, xp=xp)
    src_reg = as_dense(src_reg_matrix, xp=xp)
    dpsi_reg = as_dense(dpsi_reg_matrix, xp=xp)

    inv_var = inverse_noise_variance_from(noise, xp=xp)

    residual = data - L_d @ s_arr
    chi2 = xp.sum(inv_var * residual * residual)

    reg_val = s_arr @ (src_reg @ s_arr) + dpsi_arr @ (dpsi_reg @ dpsi_arr)

    J_combined = xp.concatenate([J_s_d, J_dpsi_d], axis=1)
    R = dense_block_diag_from(src_reg, dpsi_reg, xp=xp)
    H = J_combined.T @ (J_combined * inv_var[:, None]) + R

    _, logdet_H = xp.linalg.slogdet(H)
    _, logdet_Rs = xp.linalg.slogdet(src_reg)
    _, logdet_Rd = xp.linalg.slogdet(dpsi_reg)

    return 0.5 * ((logdet_Rs + logdet_Rd) - logdet_H - chi2 - reg_val)
