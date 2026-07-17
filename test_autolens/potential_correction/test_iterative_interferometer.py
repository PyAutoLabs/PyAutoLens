import numpy as np
import pytest

import autoarray as aa
import autolens as al


def iter_fit_from(dataset, gauge_constraints=False, n_iter=2):
    lens = al.Galaxy(
        redshift=0.5,
        mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.0),
    )
    dpsi_pixelization = al.pc.DpsiPixelization(
        mesh=al.pc.RegularDpsiMesh(factor=1),
        regularization=aa.reg.CurvatureMask(coefficient=1.0),
    )
    src_pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )
    return al.pc.IterFitDpsiSrcInterferometer(
        dataset=dataset,
        lens_start=lens,
        dpsi_pixelization=dpsi_pixelization,
        src_pixelization=src_pixelization,
        gauge_constraints=gauge_constraints,
        n_iter=n_iter,
    )


def test__requires_sparse_operator(interferometer_7):
    with pytest.raises(aa.exc.InversionException):
        iter_fit_from(interferometer_7)


def test__solve_joint_optimization__finite_state_and_decreasing_cost(
    interferometer_7,
):
    dataset = interferometer_7.apply_sparse_operator()
    fit = iter_fit_from(dataset)

    s_opt, dpsi_opt = fit.solve_joint_optimization()

    n_dpsi = np.count_nonzero(~fit.pair_dpsi_data_obj.mask_dpsi)
    assert s_opt.shape == (fit.src_reg_mat.shape[0],)
    assert dpsi_opt.shape == (n_dpsi,)
    assert np.isfinite(s_opt).all()
    assert np.isfinite(dpsi_opt).all()

    # the optimized state must beat the zero starting state, whose penalized
    # cost is 0.5 d^H C^-1 d
    x = np.concatenate([s_opt, dpsi_opt])
    F, D, _ = fit._normal_equations_from(s_opt, dpsi_opt)
    R = np.asarray(fit._regularization_matrix())
    cost_opt, _, _ = fit._cost_from(x, F, D, R)
    assert cost_opt < 0.5 * fit.data_weighted_norm


def test__cost_identity_matches_direct_visibility_chi2(interferometer_7):
    """
    The normal-equation chi^2 identity (d^H C^-1 d - 2 x^T D + x^T F x) must
    equal the directly-computed visibility chi^2 at the same state.
    """
    dataset = interferometer_7.apply_sparse_operator()
    fit = iter_fit_from(dataset)

    s_opt, dpsi_opt = fit.solve_joint_optimization()
    x = np.concatenate([s_opt, dpsi_opt])
    F, D, A = fit._normal_equations_from(s_opt, dpsi_opt)
    R = np.asarray(fit._regularization_matrix())
    _, chi2_half, _ = fit._cost_from(x, F, D, R)

    n_s = fit.src_reg_mat.shape[0]
    model_image = aa.Array2D(
        values=A[:, :n_s] @ x[:n_s], mask=dataset.real_space_mask
    )
    model_visibilities = np.asarray(
        dataset.transformer.visibilities_from(image=model_image)
    )
    data = np.asarray(dataset.data)
    noise = np.asarray(dataset.noise_map)
    residual = data - model_visibilities
    chi2_direct = 0.5 * (
        np.sum((residual.real / noise.real) ** 2)
        + np.sum((residual.imag / noise.imag) ** 2)
    )

    assert chi2_half == pytest.approx(chi2_direct, rel=1e-3)


def test__gauge_constraints_are_satisfied(interferometer_7):
    dataset = interferometer_7.apply_sparse_operator()
    fit = iter_fit_from(dataset, gauge_constraints=True)

    s_opt, dpsi_opt = fit.solve_joint_optimization()

    n_dpsi = dpsi_opt.shape[0]
    G = np.zeros((3, n_dpsi))
    G[0, :] = 1.0 / n_dpsi
    G[1, :] = fit.dpsi_points[:, 1] / n_dpsi
    G[2, :] = fit.dpsi_points[:, 0] / n_dpsi

    assert G @ dpsi_opt == pytest.approx(np.zeros(3), abs=1.0e-6)


def test__log_evidence__finite_at_optimum(interferometer_7):
    dataset = interferometer_7.apply_sparse_operator()
    fit = iter_fit_from(dataset)

    fit.solve_joint_optimization()
    evidence = fit.log_evidence()

    assert np.isfinite(evidence)

    with pytest.raises(ValueError):
        iter_fit_from(dataset).log_evidence()
