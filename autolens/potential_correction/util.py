"""
Linear-algebra utilities of the gravitational-imaging (potential correction)
technique: source-gradient and dpsi-gradient operator assembly, the explicit
PSF blur matrix, noise covariance, log-determinants, anchor-point rescaling
and arc-mask generation.

Ported from the ``potential_correction`` package of Cao et al. 2025
(https://github.com/caoxiaoyue/lensing_potential_correction). If you use this
functionality in your research, please cite Cao et al. 2025; citation
materials are provided at
https://github.com/caoxiaoyue/potential_correction_paper.
"""

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix, csc_matrix, diags, vstack
from scipy.sparse.linalg import splu

from autoarray import numba_util
from autoarray.operators.derivative_util import cleaned_mask_from


def gradient_points_from(points: np.ndarray, cross_size: float = 0.001) -> np.ndarray:
    """
    The cross-shaped evaluation points used to estimate function gradients at
    a set of (y, x) positions by central differences.

    Parameters
    ----------
    points
        The [n_points, 2] array of (y, x) positions.
    cross_size
        The half-length of each cross arm.

    Returns
    -------
    The [n_points * 4, 2] array of evaluation points, in per-point order
    (+y, -y, +x, -x).
    """
    points = np.asarray(points)
    cross = np.array(
        [
            [cross_size, 0.0],
            [-cross_size, 0.0],
            [0.0, cross_size],
            [0.0, -cross_size],
        ]
    )
    return (points[:, None, :] + cross[None, :, :]).reshape((-1, 2))


def source_gradient_from(
    values_on_gradient_points: np.ndarray, gradient_points: np.ndarray
) -> np.ndarray:
    """
    The (dS/dy, dS/dx) central-difference gradients of a source function from
    its values on the cross points of ``gradient_points_from``.
    """
    values_on_gradient_points = np.asarray(values_on_gradient_points).reshape((-1, 4))
    gradient_points = np.asarray(gradient_points).reshape((-1, 4, 2))

    step_y = gradient_points[:, 0, 0] - gradient_points[:, 1, 0]
    step_x = gradient_points[:, 2, 1] - gradient_points[:, 3, 1]
    y_diff = (
        values_on_gradient_points[:, 0] - values_on_gradient_points[:, 1]
    ) / step_y
    x_diff = (
        values_on_gradient_points[:, 2] - values_on_gradient_points[:, 3]
    ) / step_x

    return np.stack((y_diff, x_diff), axis=1)


@numba_util.jit()
def source_gradient_matrix_triplets_from(source_gradient):
    """
    The (rows, cols, values) sparse triplets of the source-gradient matrix
    (see ``source_gradient_matrix_from``).
    """
    n_unmasked_data_points = source_gradient.shape[0]
    rows_idx = np.full(n_unmasked_data_points * 2, -1, dtype=np.int64)
    cols_idx = np.full(n_unmasked_data_points * 2, -1, dtype=np.int64)
    values = np.full(n_unmasked_data_points * 2, 0.0, dtype=np.float64)

    count = 0
    for i in range(n_unmasked_data_points):
        rows_idx[count] = i
        cols_idx[count] = i * 2
        values[count] = source_gradient[i, 1]  # x-derivative
        count += 1
        rows_idx[count] = i
        cols_idx[count] = i * 2 + 1
        values[count] = source_gradient[i, 0]  # y-derivative
        count += 1

    return rows_idx, cols_idx, values


def source_gradient_matrix_from(source_gradient: np.ndarray) -> csr_matrix:
    """
    The sparse source-gradient matrix D_s of shape
    [n_unmasked_data_points, 2 * n_unmasked_data_points], which multiplies the
    interleaved (x, y) dpsi-gradient vector to produce the brightness
    correction of each image pixel (eq. 9 of the potential-correction
    formalism).

    Parameters
    ----------
    source_gradient
        The [n_unmasked_data_points, 2] array of (dS/dy, dS/dx) source
        gradients at the ray-traced positions of the image pixels.
    """
    source_gradient = np.asarray(source_gradient)
    rows_idx, cols_idx, values = source_gradient_matrix_triplets_from(source_gradient)
    return csr_matrix(
        (values, (rows_idx, cols_idx)),
        shape=(source_gradient.shape[0], 2 * source_gradient.shape[0]),
    )


def dpsi_gradient_matrix_from(itp_mat, Hx, Hy) -> csr_matrix:
    """
    The sparse dpsi-gradient operator D_psi of shape
    [2 * n_unmasked_data_points, n_unmasked_dpsi_points]: interpolates the
    dpsi mesh onto the data grid and takes its (x, y) gradients, with the
    per-pixel rows interleaved as (x_0, y_0, x_1, y_1, ...) (eq. 8 of the
    potential-correction formalism).

    Parameters
    ----------
    itp_mat
        The sparse [n_unmasked_data_points, n_unmasked_dpsi_points]
        coarse-to-fine interpolation matrix.
    Hx
        The sparse x first-derivative operator of the dpsi mesh.
    Hy
        The sparse y first-derivative operator of the dpsi mesh.
    """
    n_unmasked_data_points = itp_mat.shape[0]
    Hx_itp = itp_mat @ Hx
    Hy_itp = itp_mat @ Hy

    indices = np.empty(2 * n_unmasked_data_points, dtype=int)
    indices[0::2] = np.arange(n_unmasked_data_points)
    indices[1::2] = np.arange(n_unmasked_data_points, 2 * n_unmasked_data_points)

    return vstack([Hx_itp, Hy_itp]).tocsr()[indices, :]


@numba_util.jit()
def psf_matrix_from(psf_kernel, mask):
    """
    The explicit PSF blur matrix B of shape [n_unmasked, n_unmasked]: entry
    (j, i) is the fraction of pixel i's flux blurred into pixel j. The kernel
    must be odd-sized and is renormalized to unit sum if required.

    Parameters
    ----------
    psf_kernel
        The 2D PSF kernel (odd-sized).
    mask
        The 2D bool mask (``True`` = masked) defining the fitted pixels.
    """
    psf_hw = int(psf_kernel.shape[0] / 2)
    if psf_hw * 2 + 1 != psf_kernel.shape[0]:
        raise ValueError("The psf kernel size is not an odd number")

    if not np.isclose(np.sum(psf_kernel), 1.0):
        psf_kernel = psf_kernel / np.sum(psf_kernel)

    mask_ext = np.ones(
        (mask.shape[0] + psf_hw * 2, mask.shape[1] + psf_hw * 2), dtype="bool"
    )
    mask_ext[psf_hw:-psf_hw, psf_hw:-psf_hw] = mask
    image_ext_shape = mask_ext.shape

    indice_0, indice_1 = np.nonzero(~mask_ext)
    n_unmasked_pix = len(indice_0)
    psf_mat = np.zeros((n_unmasked_pix, n_unmasked_pix), dtype=np.float64)

    for ii in range(n_unmasked_pix):
        image_unit = np.zeros(image_ext_shape, dtype=np.float64)
        image_unit[
            indice_0[ii] - psf_hw : indice_0[ii] + psf_hw + 1,
            indice_1[ii] - psf_hw : indice_1[ii] + psf_hw + 1,
        ] = psf_kernel[:, :]
        for jj in range(n_unmasked_pix):
            psf_mat[jj, ii] = image_unit[indice_0[jj], indice_1[jj]]

    return psf_mat


def inverse_covariance_matrix_from(noise_slim) -> csr_matrix:
    """
    The sparse diagonal inverse noise-covariance matrix 1/sigma_i^2 of a 1D
    (slim) noise map.
    """
    noise_slim = np.asarray(noise_slim)
    return diags(1.0 / noise_slim**2, format="csr")


def log_det_mat(square_matrix, sparse: bool = False) -> float:
    """
    The log-determinant of a positive-definite matrix, via sparse LU
    decomposition when ``sparse`` (falling back to dense ``slogdet``).

    Raises ``np.linalg.LinAlgError`` if the determinant is not positive.
    """
    if sparse:
        try:
            lu = splu(csc_matrix(square_matrix))
            diagL = lu.L.diagonal().astype(np.complex128)
            diagU = lu.U.diagonal().astype(np.complex128)
            return float(np.real(np.log(diagL).sum() + np.log(diagU).sum()))
        except RuntimeError:
            pass

    if not isinstance(square_matrix, np.ndarray):
        square_matrix = square_matrix.toarray()
    sign, logdet = np.linalg.slogdet(square_matrix)
    if sign <= 0:
        raise np.linalg.LinAlgError(
            "The matrix is not positive definite: determinant sign is non-positive"
        )
    return float(logdet)


def dpsi_rescale_factors_from(anchor_points, dpsi_values):
    """
    The rescaling plane (a_y, a_x, c) which zeroes the dpsi solution at three
    anchor points — Suyu et al.'s rescaling scheme, removing the wandering
    source position and the unconstrained constant of the lensing potential.

    Parameters
    ----------
    anchor_points
        The [3, 2] array of (y, x) anchor positions.
    dpsi_values
        The dpsi values interpolated at the anchor positions.
    """
    anchor_points = np.asarray(anchor_points)
    dpsi_values = np.asarray(dpsi_values)
    A_matrix = np.hstack(
        [anchor_points, np.ones((3, 1), dtype=anchor_points.dtype)]
    )
    b_vector = -1.0 * dpsi_values
    a_y, a_x, c = np.linalg.solve(A_matrix, b_vector)
    return a_y, a_x, c


def split_cross_from(points: np.ndarray) -> np.ndarray:
    """
    Cross-shaped gradient-evaluation points for an irregular grid, with
    per-point arm lengths set by the square root of each point's Voronoi cell
    area (clipped at the 90th percentile; unbounded cells use the clip value).

    Parameters
    ----------
    points
        The [n_points, 2] array of (y, x) positions.

    Returns
    -------
    The [n_points * 4, 2] array of evaluation points, in per-point order
    (+y, -y, +x, -x).
    """
    from scipy.spatial import Voronoi

    vor = Voronoi(points)
    n_pixels = len(points)
    region_areas = np.zeros(n_pixels)

    for i in range(n_pixels):
        region_vertices_indexes = vor.regions[vor.point_region[i]]
        if -1 in region_vertices_indexes or len(region_vertices_indexes) == 0:
            region_areas[i] = -1
        else:
            polygon = vor.vertices[region_vertices_indexes]
            x = polygon[:, 0]
            y = polygon[:, 1]
            region_areas[i] = 0.5 * np.abs(
                np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
            )

    # the percentile must be taken over bounded cells only: with many
    # unbounded cells (small meshes) a percentile over the -1 sentinels goes
    # negative and the arm lengths become NaN
    positive_areas = region_areas[region_areas > 0]
    if positive_areas.size == 0:
        from scipy.spatial import cKDTree

        distances, _ = cKDTree(points).query(points, k=2)
        max_area = float(np.median(distances[:, 1])) ** 2
    else:
        max_area = np.percentile(positive_areas, 90.0)
    region_areas[region_areas <= 0] = max_area
    region_areas[region_areas > max_area] = max_area

    half_lengths = 0.5 * np.sqrt(region_areas)

    splitted_array = np.zeros((n_pixels, 4, 2))
    splitted_array[:, 0, 0] = points[:, 0] + half_lengths
    splitted_array[:, 0, 1] = points[:, 1]
    splitted_array[:, 1, 0] = points[:, 0] - half_lengths
    splitted_array[:, 1, 1] = points[:, 1]
    splitted_array[:, 2, 0] = points[:, 0]
    splitted_array[:, 2, 1] = points[:, 1] + half_lengths
    splitted_array[:, 3, 0] = points[:, 0]
    splitted_array[:, 3, 1] = points[:, 1] - half_lengths

    return splitted_array.reshape(n_pixels * 4, 2)


def arc_mask_from(
    snr_map: np.ndarray,
    threshold: float = 3.0,
    ignore_size: int = 25,
    ext_size: int = 5,
) -> np.ndarray:
    """
    A cleaned mask tracing the lensed arcs of a signal-to-noise map: pixels
    above ``threshold`` are kept, connected islands smaller than
    ``ignore_size`` pixels are dropped, the result is dilated by an
    ``ext_size`` square footprint and finally cleaned so every unmasked pixel
    supports a finite-difference scheme.

    Implemented with ``scipy.ndimage`` (8-connectivity labelling, matching
    the original's ``skimage`` behaviour) to avoid a scikit-image dependency.
    """
    bool_map = snr_map > threshold

    labels, _ = ndimage.label(bool_map, structure=np.ones((3, 3), dtype=int))
    label_sizes = np.bincount(labels.ravel())
    small_islands = np.where(label_sizes < ignore_size)[0]
    small_islands = small_islands[small_islands != 0]

    mask = np.copy(bool_map)
    for island_label in small_islands:
        mask[labels == island_label] = 0
    mask = ndimage.binary_dilation(
        mask, structure=np.ones((ext_size, ext_size), dtype=bool)
    )

    mask, _ = cleaned_mask_from(~mask, max_iter=50)
    return mask
