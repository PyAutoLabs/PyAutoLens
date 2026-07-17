import numpy as np
import pytest

import autolens as al


def circular_mask(shape=(20, 20), pixel_scale=0.5, r_min=0.0, r_max=3.6):
    cy = (shape[0] - 1) / 2.0
    cx = (shape[1] - 1) / 2.0
    mask = np.ones(shape, dtype=bool)
    for i in range(shape[0]):
        for j in range(shape[1]):
            r = np.sqrt(((i - cy) * pixel_scale) ** 2 + ((j - cx) * pixel_scale) ** 2)
            if r_min <= r <= r_max:
                mask[i, j] = False
    return mask


def test__pair_regular_dpsi_mesh__shapes_and_grids():
    mask = circular_mask()
    pair = al.pc.PairRegularDpsiMesh(mask, pixel_scale=0.5, dpsi_factor=2)

    n_data = np.count_nonzero(~pair.mask_data)
    n_dpsi = np.count_nonzero(~pair.mask_dpsi)

    assert pair.shape_2d_dpsi == (10, 10)
    assert pair.dpix_dpsi == pytest.approx(1.0)
    assert pair.xgrid_data_1d.shape == (n_data,)
    assert pair.ygrid_dpsi_1d.shape == (n_dpsi,)
    assert pair.itp_mat.shape == (n_data, n_dpsi)
    assert pair.Hx_dpsi.shape == (n_dpsi, n_dpsi)
    assert pair.hamiltonian_data.shape == (n_data, n_data)


def test__pair_regular_dpsi_mesh__itp_matrix_partition_of_unity():
    mask = circular_mask()
    pair = al.pc.PairRegularDpsiMesh(mask, pixel_scale=0.5, dpsi_factor=2)

    row_sums = np.asarray(pair.itp_mat.sum(axis=1)).ravel()
    assert row_sums == pytest.approx(np.ones_like(row_sums), abs=1.0e-10)


def test__pair_regular_dpsi_mesh__interpolation_exact_on_linear_function():
    mask = circular_mask()
    pair = al.pc.PairRegularDpsiMesh(mask, pixel_scale=0.5, dpsi_factor=2)

    dpsi_values = 2.0 * pair.ygrid_dpsi_1d + 3.0 * pair.xgrid_dpsi_1d
    data_values = pair.itp_mat @ dpsi_values

    assert data_values == pytest.approx(
        2.0 * pair.ygrid_data_1d + 3.0 * pair.xgrid_data_1d, abs=1.0e-10
    )


def test__pair_regular_dpsi_mesh__indivisible_shape_raises():
    mask = circular_mask(shape=(21, 21))
    with pytest.raises(ValueError):
        al.pc.PairRegularDpsiMesh(mask, pixel_scale=0.5, dpsi_factor=2)


def test__pair_regular_dpsi_mesh__too_sparse_raises():
    mask = circular_mask(shape=(20, 20), r_min=0.7, r_max=2.0)
    with pytest.raises(ValueError):
        al.pc.PairRegularDpsiMesh(mask, pixel_scale=0.5, dpsi_factor=10)


def test__regular_dpsi_mesh__equality():
    assert al.pc.RegularDpsiMesh(factor=2) == al.pc.RegularDpsiMesh(factor=2)
    assert al.pc.RegularDpsiMesh(factor=2) != al.pc.RegularDpsiMesh(factor=3)
