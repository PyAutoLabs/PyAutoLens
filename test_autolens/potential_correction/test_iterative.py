import numpy as np
import pytest

import autoarray as aa
import autolens as al


def iter_fit_from(masked_imaging, gauge_constraints=False, n_iter=2):
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
    return al.pc.IterFitDpsiSrcImaging(
        masked_imaging=masked_imaging,
        lens_start=lens,
        dpsi_pixelization=dpsi_pixelization,
        src_pixelization=src_pixelization,
        gauge_constraints=gauge_constraints,
        n_iter=n_iter,
    )


def test__solve_joint_optimization__returns_finite_state_and_decreases_cost(
    masked_imaging_7x7,
):
    fit = iter_fit_from(masked_imaging_7x7)

    s_opt, dpsi_opt = fit.solve_joint_optimization()

    n_dpsi = np.count_nonzero(~fit.pair_dpsi_data_obj.mask_dpsi)
    assert s_opt.shape == (fit.src_reg_mat.shape[0],)
    assert dpsi_opt.shape == (n_dpsi,)
    assert np.isfinite(s_opt).all()
    assert np.isfinite(dpsi_opt).all()

    # the optimized state must beat the zero starting state
    data = np.asarray(fit.masked_imaging.data.slim)
    inv_var = fit.inverse_noise_variance
    L, _, _ = fit.get_L_Js_Jdpsi(s_opt, dpsi_opt)
    from autolens.potential_correction import dense_util

    cost_opt, *_ = dense_util.lm_cost_from(
        data, inv_var, s_opt, dpsi_opt, L, fit.src_reg_mat,
        fit.dpsi_regularization_matrix,
    )
    cost_zero = 0.5 * float(np.sum(inv_var * data**2))
    assert float(cost_opt) < cost_zero


def test__solve_joint_optimization__gauge_constraints_are_satisfied(
    masked_imaging_7x7,
):
    fit = iter_fit_from(masked_imaging_7x7, gauge_constraints=True)

    s_opt, dpsi_opt = fit.solve_joint_optimization()

    n_dpsi = dpsi_opt.shape[0]
    G = np.zeros((3, n_dpsi))
    G[0, :] = 1.0 / n_dpsi
    G[1, :] = fit.dpsi_points[:, 1] / n_dpsi
    G[2, :] = fit.dpsi_points[:, 0] / n_dpsi

    assert G @ dpsi_opt == pytest.approx(np.zeros(3), abs=1.0e-6)


def test__gauge_project_dpsi__zeroes_the_three_functionals(masked_imaging_7x7):
    fit = iter_fit_from(masked_imaging_7x7, gauge_constraints=True)
    fit._init_joint_optimization()

    x = fit.dpsi_points[:, 1]
    y = fit.dpsi_points[:, 0]
    n_dpsi = fit.dpsi_points.shape[0]

    # an arbitrary field carrying a deliberate constant + linear (gauge) part
    rng = np.random.default_rng(0)
    dpsi = rng.standard_normal(n_dpsi) + 4.0 + 2.0 * x - 3.0 * y

    projected = fit._gauge_project_dpsi(dpsi)

    functionals = np.array(
        [np.sum(projected), np.sum(x * projected), np.sum(y * projected)]
    )
    assert functionals == pytest.approx(np.zeros(3), abs=1.0e-8)
    # removing only constant + linear modes: a already-gauged field is unchanged
    assert fit._gauge_project_dpsi(projected) == pytest.approx(projected, abs=1.0e-8)


def test__solve_joint_optimization__gauge_project_x0__gauges_an_ungauged_warm_start(
    masked_imaging_7x7,
):
    # a cold solve gives a gauge-satisfied optimum; offset its dpsi by a pure
    # constant to build a warm start that sits on the optimum but violates the
    # <dpsi,1> constraint (an un-gauged x0, as the one-shot solution is).
    fit0 = iter_fit_from(masked_imaging_7x7, gauge_constraints=True)
    s_opt, dpsi_opt = fit0.solve_joint_optimization()
    x0 = np.concatenate([s_opt, dpsi_opt + 0.7])

    def gauge_of(fit, dpsi):
        n_dpsi = dpsi.shape[0]
        G = np.zeros((3, n_dpsi))
        G[0, :] = 1.0 / n_dpsi
        G[1, :] = fit.dpsi_points[:, 1] / n_dpsi
        G[2, :] = fit.dpsi_points[:, 0] / n_dpsi
        return G @ dpsi

    assert np.abs(gauge_of(fit0, dpsi_opt + 0.7)[0]) > 1.0e-2  # x0 is un-gauged

    fit_on = iter_fit_from(masked_imaging_7x7, gauge_constraints=True)
    _, dpsi_on = fit_on.solve_joint_optimization(x0=x0, gauge_project_x0=True)
    assert gauge_of(fit_on, dpsi_on) == pytest.approx(np.zeros(3), abs=1.0e-6)


def test__log_evidence__finite_at_optimum(masked_imaging_7x7):
    fit = iter_fit_from(masked_imaging_7x7)

    fit.solve_joint_optimization()
    evidence = fit.log_evidence()

    assert np.isfinite(evidence)


def test__log_evidence__requires_state_or_solve(masked_imaging_7x7):
    fit = iter_fit_from(masked_imaging_7x7)

    with pytest.raises(ValueError):
        fit.log_evidence()
