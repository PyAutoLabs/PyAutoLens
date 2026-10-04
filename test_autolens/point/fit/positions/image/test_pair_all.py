import numpy as np
import pytest

import autolens as al


point = al.ps.Point(centre=(0.1, 0.1))
galaxy = al.Galaxy(redshift=1.0, point_0=point)
tracer = al.Tracer(galaxies=[al.Galaxy(redshift=0.5), galaxy])


@pytest.fixture
def data():
    return np.array([(0.0, 0.0), (1.0, 0.0)])


@pytest.fixture
def noise_map():
    return np.array([1.0, 1.0])


@pytest.fixture
def fit(data, noise_map):
    model_positions = al.Grid2DIrregular(
        [
            (-1.0749, -1.1),
            (1.19117, 1.175),
        ]
    )

    return al.FitPositionsImagePairAll(
        name="point_0",
        data=data,
        noise_map=noise_map,
        tracer=tracer,
        solver=al.mock.MockPointSolver(model_positions),
    )


def test__fit_positions_image_pair_all__two_model_positions__per_permutation_likelihoods_and_chi_squared_correct(
    fit,
):
    assert np.allclose(
        fit.all_permutations_log_likelihoods(),
        [
            -1.51114426,
            -1.50631469,
        ],
    )
    assert fit.chi_squared == -2.0 * -4.40375330990644


def test__fit_positions_image_pair_all__model_has_inf_position__inf_excluded_from_permutations(
    data,
    noise_map,
):
    model_positions = al.Grid2DIrregular(
        [
            (-1.0749, -1.1),
            (1.19117, 1.175),
            (np.inf, np.inf),
        ]
    )
    fit = al.FitPositionsImagePairAll(
        name="point_0",
        data=data,
        noise_map=noise_map,
        tracer=tracer,
        solver=al.mock.MockPointSolver(model_positions),
    )

    assert np.allclose(
        fit.all_permutations_log_likelihoods(),
        [
            -1.51114426,
            -1.50631469,
        ],
    )
    assert fit.chi_squared == -2.0 * -4.40375330990644


def test__fit_positions_image_pair_all__model_has_duplicate_position__duplicate_permutations_handled(
    data,
    noise_map,
):
    model_positions = al.Grid2DIrregular(
        [
            (-1.0749, -1.1),
            (1.19117, 1.175),
            (1.19117, 1.175),
        ]
    )
    fit = al.FitPositionsImagePairAll(
        name="point_0",
        data=data,
        noise_map=noise_map,
        tracer=tracer,
        solver=al.mock.MockPointSolver(model_positions),
    )

    assert np.allclose(
        fit.all_permutations_log_likelihoods(),
        [-1.14237812, -0.87193683],
    )
    assert fit.chi_squared == -2.0 * -4.211539531047171


def test__fit_positions_image_pair_all__penalty_regression__n_permutations_and_monotonic_likelihood(
    data, noise_map
):
    # Test 5: n_permutations must equal n_finite_model_positions ** n_observed, and adding a
    # spurious (but finite) extra model position must strictly lower the likelihood -- an
    # extra candidate explanation dilutes each permutation's probability without improving
    # the fit to any observed position.
    two_model_positions = al.Grid2DIrregular(
        [(-1.0749, -1.1), (1.19117, 1.175)]
    )
    fit_two = al.FitPositionsImagePairAll(
        name="point_0",
        data=data,
        noise_map=noise_map,
        tracer=tracer,
        solver=al.mock.MockPointSolver(two_model_positions),
    )

    n_non_nan = np.count_nonzero(np.isfinite(two_model_positions.array).any(axis=1))
    assert n_non_nan == 2
    n_permutations = n_non_nan ** len(data)
    assert n_permutations == 2 ** len(data)

    three_model_positions = al.Grid2DIrregular(
        [(-1.0749, -1.1), (1.19117, 1.175), (5.0, 5.0)]
    )
    fit_three = al.FitPositionsImagePairAll(
        name="point_0",
        data=data,
        noise_map=noise_map,
        tracer=tracer,
        solver=al.mock.MockPointSolver(three_model_positions),
    )

    n_non_nan_three = np.count_nonzero(
        np.isfinite(three_model_positions.array).any(axis=1)
    )
    assert n_non_nan_three == 3
    assert n_non_nan_three ** len(data) == 3 ** len(data)

    # log_likelihood = -0.5 * chi_squared, so a strictly lower likelihood is a strictly
    # higher chi_squared.
    assert fit_three.chi_squared > fit_two.chi_squared


def test__fit_positions_image_pair_all_solved__source_plane_coordinate_feeds_solver(
    data, noise_map
):
    galaxy_solved = al.Galaxy(redshift=1.0, point_0=al.ps.PointSolved())
    tracer_solved = al.Tracer(galaxies=[al.Galaxy(redshift=0.5), galaxy_solved])

    model_positions = al.Grid2DIrregular([(-1.0749, -1.1), (1.19117, 1.175)])

    fit = al.FitPositionsImagePairAllSolved(
        name="point_0",
        data=data,
        noise_map=noise_map,
        tracer=tracer_solved,
        solver=al.mock.MockPointSolver(model_positions),
    )

    assert np.isfinite(fit.source_plane_coordinate[0])
    assert np.isfinite(fit.source_plane_coordinate[1])
    # The pairing chi-squared itself is untouched (solver is mocked, so model positions are
    # fixed regardless of the solved centre): matches the plain FitPositionsImagePairAll
    # value from the fixture-equivalent test above.
    assert fit.chi_squared == -2.0 * -4.40375330990644


def test__no_model_positions__finite_no_image_floor_matching_siblings(data, noise_map):
    """
    Regression: with no model positions, `n_permutations` is 0, so `-log(0)` is `+inf` while the
    permutation sum is `-inf` and the two combined to NaN.

    `fitness.py` converts a NaN log-likelihood into `resample_figure_of_merit`, so the model was
    silently resampled rather than scored -- exactly what the `no_image_residual` floor exists to
    prevent. `FitPositionsImagePair` and `FitPositionsImagePairRepeat` already applied that floor;
    `FitPositionsImagePairAll` did not.

    Surfaced by @rhayes777's PyAutoLens#531 once `PointSolver.solve` began returning an empty grid
    instead of raising.
    """

    no_positions = al.Grid2DIrregular(np.zeros((0, 2)))

    kwargs = dict(
        name="point_0",
        data=data,
        noise_map=noise_map,
        tracer=tracer,
        solver=al.mock.MockPointSolver(no_positions),
    )

    fit_all = al.FitPositionsImagePairAll(**kwargs)

    # `log_likelihood` is not asserted here: the module's `noise_map` fixture is a raw ndarray,
    # which `fit_dataset.noise_normalization` cannot consume. That predates this fix and is why
    # every test in this file asserts `chi_squared` only. `log_likelihood` finiteness is covered
    # end-to-end against a real solver and an `ArrayIrregular` noise-map.
    assert np.isfinite(fit_all.chi_squared)

    # the floor is on the same scale as the siblings', not merely finite
    assert fit_all.chi_squared == pytest.approx(
        al.FitPositionsImagePair(**kwargs).chi_squared
    )
    assert fit_all.chi_squared == pytest.approx(
        al.FitPositionsImagePairRepeat(**kwargs).chi_squared
    )


def test__extreme_mismatch__log_sum_exp_stays_finite(data, noise_map):
    """
    Regression: the literal `log(sum(exp(log_p)))` underflows to `log(0) = -inf` once the best
    model/observed pairing is ~38 sigma or worse (`exp` underflows below the smallest float64),
    strangling gradient flow across the exact region gradient searches traverse to find the basin.
    The max-shifted log-sum-exp must stay finite at arbitrarily large mismatch and equal the
    directly-computed shifted reduction.
    """
    model_positions = al.Grid2DIrregular([(40.0, 40.0), (50.0, 50.0)])

    fit = al.FitPositionsImagePairAll(
        name="point_0",
        data=data,
        noise_map=noise_map,
        tracer=tracer,
        solver=al.mock.MockPointSolver(model_positions),
    )

    log_likelihoods = fit.all_permutations_log_likelihoods()

    assert np.all(np.isfinite(log_likelihoods))
    assert np.isfinite(fit.chi_squared)

    # ~56 sigma worst pairing: every log_p is far below the ~-745 underflow threshold of exp().
    for data_position, sigma, log_likelihood in zip(data, noise_map, log_likelihoods):
        log_ps = np.array(
            [
                fit.log_p(data_position, model_position, sigma)
                for model_position in model_positions.array
            ]
        )
        assert np.all(log_ps < -745.0)
        expected = log_ps.max() + np.log(np.sum(np.exp(log_ps - log_ps.max())))
        assert log_likelihood == pytest.approx(expected, rel=1.0e-12)


def test__moderate_mismatch__log_sum_exp_matches_literal_form(data, noise_map):
    """Where the literal `log(sum(exp(...)))` is finite, the shifted form must equal it."""

    model_positions = al.Grid2DIrregular([(-1.0749, -1.1), (1.19117, 1.175)])

    fit = al.FitPositionsImagePairAll(
        name="point_0",
        data=data,
        noise_map=noise_map,
        tracer=tracer,
        solver=al.mock.MockPointSolver(model_positions),
    )

    for data_position, sigma, log_likelihood in zip(
        data, noise_map, fit.all_permutations_log_likelihoods()
    ):
        literal = np.log(
            np.sum(
                np.exp(
                    np.array(
                        [
                            fit.log_p(data_position, model_position, sigma)
                            for model_position in model_positions.array
                        ]
                    )
                )
            )
        )
        assert log_likelihood == pytest.approx(literal, rel=1.0e-14)


def test__model_positions_present__chi_squared_unchanged(data, noise_map):
    """The no-image branch must not perturb the ordinary path."""

    fit = al.FitPositionsImagePairAll(
        name="point_0",
        data=data,
        noise_map=noise_map,
        tracer=tracer,
        solver=al.mock.MockPointSolver(
            al.Grid2DIrregular([(-1.0749, -1.1), (1.19117, 1.175)])
        ),
    )

    assert fit.chi_squared == -2.0 * -4.40375330990644


# Real model positions of the forward-mode gradient tests below, followed by solver-style `inf`
# padding rows (the `PointSolver` pads to a fixed size and is called with `remove_infinities=False`).
_GRAD_BASE_POSITIONS = np.array(
    [(-1.03, -1.09), (0.35, 1.63), (1.57, 0.43), (1.25, 1.22)]
    + [(np.inf, np.inf)] * 6
)
_GRAD_DATA = al.Grid2DIrregular([(-1.0, -1.1), (0.4, 1.6), (1.6, 0.45), (1.2, 1.25)])
_GRAD_NOISE_MAP = al.ArrayIrregular([0.05, 0.05, 0.05, 0.05])


class _PaddedSolver:
    """Model positions that move smoothly with `theta` (scale, shift_0, shift_1), with `inf` padding
    rows whose tangent is exactly 0 -- the shape `PointSolver.solve` returns under JAX."""

    def __init__(self, theta):
        self.theta = theta

    def solve(self, tracer, source_plane_coordinate, xp=np, plane_redshift=None, remove_infinities=True):
        import jax.numpy as jnp

        base = jnp.asarray(_GRAD_BASE_POSITIONS)
        is_padding = ~jnp.isfinite(base).all(axis=1)
        moved = jnp.where(is_padding[:, None], 0.0, base) * self.theta[0] + self.theta[1:3]
        return type("Positions", (), {"array": jnp.where(is_padding[:, None], jnp.inf, moved)})()


@pytest.mark.parametrize(
    "fit_cls, point",
    [
        (al.FitPositionsImagePairAll, al.ps.Point(centre=(0.07, 0.07))),
        (al.FitPositionsImagePairAllSolved, al.ps.PointSolved()),
    ],
)
def test__forward_mode_gradient_with_padded_model_positions__finite_and_matches_reverse_and_fd(
    fit_cls, point
):
    """
    Regression (PyAutoLens#767): `AnalysisPoint` differentiates with `jax.jacfwd` by default, and the
    forward gradient of the all-pairs likelihood was NaN in every component. The padded (`inf`) model
    positions gave `square_distance` a tangent of `2 * (d - inf) * 0 = NaN`, which `exp` carried into
    the log likelihood; reverse mode happened to discard it.

    Jitted, as gradient searches run it (XLA's fusion of the jitted graph is where a per-position
    guard still leaked NaN), over eight parameter draws: `jacfwd` must be finite, non-zero, equal to
    `grad`, and equal to a central finite difference of the (smooth, mocked-solver) likelihood.
    """
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    lens = al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.6, ell_comps=(0.05, 0.0)),
    )
    tracer = al.Tracer(galaxies=[lens, al.Galaxy(redshift=1.0, point_0=point)])

    def log_likelihood(theta):
        return fit_cls(
            name="point_0",
            data=_GRAD_DATA,
            noise_map=_GRAD_NOISE_MAP,
            tracer=tracer,
            solver=_PaddedSolver(theta),
            xp=jnp,
        ).log_likelihood

    value = jax.jit(log_likelihood)
    forward = jax.jit(jax.jacfwd(log_likelihood))
    reverse = jax.jit(jax.grad(log_likelihood))

    for key in range(8):
        theta = jnp.array([1.0, 0.0, 0.0]) + jax.random.uniform(
            jax.random.PRNGKey(key), (3,), minval=-0.02, maxval=0.02
        )

        grad_forward = np.asarray(forward(theta))
        grad_reverse = np.asarray(reverse(theta))

        h = 1.0e-6
        grad_fd = np.array(
            [
                (float(value(theta.at[i].add(h))) - float(value(theta.at[i].add(-h))))
                / (2.0 * h)
                for i in range(3)
            ]
        )

        assert np.all(np.isfinite(grad_forward)), (key, grad_forward)
        assert np.any(grad_forward != 0.0)

        scale = np.max(np.abs(grad_reverse))
        np.testing.assert_allclose(grad_forward, grad_reverse, rtol=1e-6, atol=1e-12 * scale)
        np.testing.assert_allclose(grad_forward, grad_fd, rtol=1e-6, atol=1e-6 * scale)


def test__padded_model_positions__log_likelihoods_identical_to_padding_removed():
    """
    The padding guard changes tangents only: under JAX (and NumPy) the per-position log likelihoods
    with `inf` padding rows are exactly those of the real rows alone.
    """
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    real = al.Grid2DIrregular(_GRAD_BASE_POSITIONS[:4])

    def fit_for(solver, xp):
        return al.FitPositionsImagePairAll(
            name="point_0",
            data=_GRAD_DATA,
            noise_map=_GRAD_NOISE_MAP,
            tracer=tracer,
            solver=solver,
            xp=xp,
        )

    expected = fit_for(al.mock.MockPointSolver(real), np).all_permutations_log_likelihoods()

    padded_np = fit_for(
        al.mock.MockPointSolver(al.Grid2DIrregular(_GRAD_BASE_POSITIONS)), np
    ).all_permutations_log_likelihoods()
    padded_jax = jax.jit(
        lambda theta: fit_for(_PaddedSolver(theta), jnp).all_permutations_log_likelihoods()
    )(jnp.array([1.0, 0.0, 0.0]))

    np.testing.assert_array_equal(padded_np, expected)
    np.testing.assert_allclose(np.asarray(padded_jax), expected, rtol=1e-14)
