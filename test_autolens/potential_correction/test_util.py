import numpy as np
import pytest
from scipy.sparse import csr_matrix

import autolens as al
from autolens.potential_correction import util as pc_util


def test__gradient_points_from__cross_layout():
    points = np.array([[1.0, 2.0], [3.0, 4.0]])

    grad_points = pc_util.gradient_points_from(points, cross_size=0.1)

    assert grad_points.shape == (8, 2)
    # per-point order is (+y, -y, +x, -x)
    assert grad_points[0] == pytest.approx([1.1, 2.0])
    assert grad_points[1] == pytest.approx([0.9, 2.0])
    assert grad_points[2] == pytest.approx([1.0, 2.1])
    assert grad_points[3] == pytest.approx([1.0, 1.9])
    assert grad_points[4] == pytest.approx([3.1, 4.0])


def test__source_gradient_from__exact_on_linear_function():
    points = np.array([[0.5, -0.3], [1.0, 2.0], [-1.5, 0.7]])
    grad_points = pc_util.gradient_points_from(points, cross_size=0.01)

    # S = 2 y + 3 x has gradient (2, 3) everywhere
    values = 2.0 * grad_points[:, 0] + 3.0 * grad_points[:, 1]

    gradient = pc_util.source_gradient_from(values, grad_points)

    assert gradient[:, 0] == pytest.approx(2.0, abs=1.0e-10)
    assert gradient[:, 1] == pytest.approx(3.0, abs=1.0e-10)


def test__source_gradient_matrix_from__interleaved_x_y_structure():
    source_gradient = np.array([[2.0, 3.0], [4.0, 5.0]])  # (dS/dy, dS/dx) rows

    matrix = pc_util.source_gradient_matrix_from(source_gradient).toarray()

    expected = np.array(
        [
            [3.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 4.0],
        ]
    )
    assert matrix == pytest.approx(expected)


def test__dpsi_gradient_matrix_from__interleaves_x_then_y_rows():
    itp_mat = csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]))
    Hx = csr_matrix(np.array([[1.0, -1.0], [0.0, 2.0]]))
    Hy = csr_matrix(np.array([[3.0, 0.0], [0.0, 4.0]]))

    matrix = pc_util.dpsi_gradient_matrix_from(itp_mat, Hx, Hy).toarray()

    Hx_itp = (itp_mat @ Hx).toarray()
    Hy_itp = (itp_mat @ Hy).toarray()
    expected = np.empty((6, 2))
    expected[0::2] = Hx_itp
    expected[1::2] = Hy_itp
    assert matrix == pytest.approx(expected)

    # acting on a dpsi vector gives interleaved (x, y) gradients per pixel
    dpsi = np.array([1.0, 2.0])
    result = matrix @ dpsi
    assert result[0::2] == pytest.approx(Hx_itp @ dpsi)
    assert result[1::2] == pytest.approx(Hy_itp @ dpsi)


def test__psf_matrix_from__identity_kernel_gives_identity_matrix():
    mask = np.ones((5, 5), dtype=bool)
    mask[1:4, 1:4] = False

    kernel = np.zeros((3, 3))
    kernel[1, 1] = 1.0

    psf_mat = pc_util.psf_matrix_from(kernel, mask)

    assert psf_mat == pytest.approx(np.eye(9))


def test__psf_matrix_from__blur_matches_direct_convolution():
    mask = np.ones((5, 5), dtype=bool)
    mask[1:4, 1:4] = False

    kernel = np.array(
        [
            [0.0, 0.1, 0.0],
            [0.1, 0.6, 0.1],
            [0.0, 0.1, 0.0],
        ]
    )

    psf_mat = pc_util.psf_matrix_from(kernel, mask)

    # blur a delta at the central unmasked pixel (index 4 of the 3x3 block)
    image = np.zeros(9)
    image[4] = 1.0
    blurred = psf_mat @ image

    expected = np.array([0.0, 0.1, 0.0, 0.1, 0.6, 0.1, 0.0, 0.1, 0.0])
    assert blurred == pytest.approx(expected)


def test__psf_matrix_from__even_kernel_raises():
    mask = np.zeros((4, 4), dtype=bool)
    with pytest.raises(Exception):
        pc_util.psf_matrix_from(np.ones((4, 4)) / 16.0, mask)


def test__inverse_covariance_matrix_from__diagonal_of_inverse_variance():
    noise = np.array([0.5, 2.0, 1.0])

    inv_cov = pc_util.inverse_covariance_matrix_from(noise).toarray()

    assert inv_cov == pytest.approx(np.diag([4.0, 0.25, 1.0]))


def test__log_det_mat__matches_slogdet_dense_and_sparse():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 6))
    spd = A @ A.T + 6.0 * np.eye(6)

    expected = np.linalg.slogdet(spd)[1]

    assert pc_util.log_det_mat(spd) == pytest.approx(expected, rel=1.0e-10)
    assert pc_util.log_det_mat(csr_matrix(spd), sparse=True) == pytest.approx(
        expected, rel=1.0e-10
    )


def test__log_det_mat__raises_on_negative_determinant():
    with pytest.raises(np.linalg.LinAlgError):
        pc_util.log_det_mat(np.diag([1.0, -1.0]))


def test__dpsi_rescale_factors_from__exact_plane_through_anchors():
    anchor_points = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    # dpsi at the anchors follows the plane 2 y + 3 x + 1
    dpsi_values = 2.0 * anchor_points[:, 0] + 3.0 * anchor_points[:, 1] + 1.0

    a_y, a_x, c = pc_util.dpsi_rescale_factors_from(anchor_points, dpsi_values)

    # the rescaling plane cancels the input plane exactly
    assert a_y == pytest.approx(-2.0, abs=1.0e-10)
    assert a_x == pytest.approx(-3.0, abs=1.0e-10)
    assert c == pytest.approx(-1.0, abs=1.0e-10)


def test__split_cross_from__cross_layout_and_positive_lengths():
    rng = np.random.default_rng(1)
    points = rng.uniform(-1.0, 1.0, size=(12, 2))

    split = pc_util.split_cross_from(points)

    assert split.shape == (48, 2)
    split = split.reshape(12, 4, 2)
    # arm 0/1 move y only; arm 2/3 move x only, symmetrically
    assert split[:, 0, 1] == pytest.approx(points[:, 1])
    assert split[:, 2, 0] == pytest.approx(points[:, 0])
    assert split[:, 0, 0] - points[:, 0] == pytest.approx(
        points[:, 0] - split[:, 1, 0]
    )
    assert (split[:, 0, 0] > points[:, 0]).all()


def test__split_cross_from__all_unbounded_cells_yield_finite_arms():
    # with 4 points every Voronoi cell is unbounded; the arm length must fall
    # back to the nearest-neighbour spacing rather than go NaN (the original
    # implementation took a percentile over the -1 sentinels)
    points = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])

    split = pc_util.split_cross_from(points)

    assert np.isfinite(split).all()
    split = split.reshape(4, 4, 2)
    assert (split[:, 0, 0] > points[:, 0]).all()


def test__arc_mask_from__keeps_large_islands_drops_small_ones():
    snr_map = np.zeros((24, 24))
    snr_map[6:14, 6:14] = 10.0  # 64-pixel island: kept
    snr_map[20, 20] = 10.0  # single pixel island: dropped

    mask = pc_util.arc_mask_from(snr_map, threshold=3.0, ignore_size=25, ext_size=3)

    # the large island (dilated) is unmasked; the small one stays masked
    assert not mask[10, 10]
    assert mask[20, 20]
    # dilation grows the unmasked region beyond the raw island
    assert not mask[5, 10]
