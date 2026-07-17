"""
The dpsi mesh of the gravitational-imaging (potential correction) technique:
a rectangular mesh a factor coarser than the data grid, on whose unmasked
pixels the pixelized corrections to the lensing potential are defined, paired
to the data grid by a sparse bilinear interpolation matrix.

Ported from the ``potential_correction`` package of Cao et al. 2025
(https://github.com/caoxiaoyue/lensing_potential_correction). If you use this
functionality in your research, please cite Cao et al. 2025; citation
materials are provided at
https://github.com/caoxiaoyue/potential_correction_paper.
"""

import numpy as np

import autoarray as aa
from autoarray.operators import coarse_interp_util
from autoarray.operators import derivative_util


class RegularDpsiMesh:
    def __init__(self, factor: int = 1):
        """
        The mesh component of a ``DpsiPixelization`` model: pixelized
        potential corrections are defined on a rectangular mesh ``factor``
        times coarser than the data grid.

        Parameters
        ----------
        factor
            How many times coarser than the data grid the dpsi mesh is.
        """
        self.factor = int(factor)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__ and self.__class__ is other.__class__

    def __str__(self):
        return "\n".join(["{}: {}".format(k, v) for k, v in self.__dict__.items()])

    def __repr__(self):
        return "{}\n{}".format(self.__class__.__name__, str(self))


class PairRegularDpsiMesh:
    def __init__(self, mask, pixel_scale: float, dpsi_factor: int = 2):
        """
        Pairs the dpsi mesh to the data grid: cleans the data mask, bins it
        onto the coarser dpsi mesh (cleaning again), builds the bilinear
        dpsi-to-data interpolation matrix and the sparse first/second
        derivative operators of both grids.

        Parameters
        ----------
        mask
            The 2D bool data mask (``True`` = masked), typically an
            annular-like region tracing the lensed arcs.
        pixel_scale
            The pixel size of the data grid, in arcsec.
        dpsi_factor
            How many times coarser than the data grid the dpsi mesh is; both
            mask dimensions must be divisible by it.
        """
        self.mask_data, self.diff_types_data = derivative_util.cleaned_mask_from(
            np.asarray(mask)
        )
        self.dpix_data = pixel_scale
        self.mask_data_aa = aa.Mask2D(mask=self.mask_data, pixel_scales=self.dpix_data)
        grid_data = aa.Grid2D.from_mask(mask=self.mask_data_aa)
        self.xgrid_data = np.asarray(grid_data.native[:, :, 1])
        self.ygrid_data = np.asarray(grid_data.native[:, :, 0])
        limit = (self.mask_data.shape[0] * self.dpix_data) * 0.5
        self.data_bound = [-limit, limit, -limit, limit]

        self.dpsi_factor = dpsi_factor
        if (
            self.mask_data.shape[0] % self.dpsi_factor != 0
            or self.mask_data.shape[1] % self.dpsi_factor != 0
        ):
            raise ValueError("both mask dimensions must be divisible by dpsi_factor")
        self.shape_2d_dpsi = (
            int(self.mask_data.shape[0] / self.dpsi_factor),
            int(self.mask_data.shape[1] / self.dpsi_factor),
        )
        self.dpix_dpsi = float(2.0 * limit / self.shape_2d_dpsi[0])
        self.mask_dpsi = coarse_interp_util.binned_mask_from(
            self.mask_data, self.dpsi_factor
        )
        self.mask_dpsi, self.diff_types_dpsi = derivative_util.cleaned_mask_from(
            self.mask_dpsi
        )
        self.mask_dpsi_aa = aa.Mask2D(mask=self.mask_dpsi, pixel_scales=self.dpix_dpsi)
        grid_dpsi = aa.Grid2D.from_mask(mask=self.mask_dpsi_aa)
        self.xgrid_dpsi = np.asarray(grid_dpsi.native[:, :, 1])
        self.ygrid_dpsi = np.asarray(grid_dpsi.native[:, :, 0])

        self.grid_1d_from_mask()

        self.get_itp_box_ctr()
        self.get_dpsi2data_mapping()

        self.get_gradient_operator_data()
        self.get_gradient_operator_dpsi()
        self.get_hamiltonian_operator_data()
        self.get_hamiltonian_operator_dpsi()

    def grid_1d_from_mask(self):
        self.idx_1d_data = np.where((~self.mask_data).flatten())[0]
        self.xgrid_data_1d = self.xgrid_data.flatten()[self.idx_1d_data]
        self.ygrid_data_1d = self.ygrid_data.flatten()[self.idx_1d_data]

        self.idx_1d_dpsi = np.where((~self.mask_dpsi).flatten())[0]
        self.xgrid_dpsi_1d = self.xgrid_dpsi.flatten()[self.idx_1d_dpsi]
        self.ygrid_dpsi_1d = self.ygrid_dpsi.flatten()[self.idx_1d_dpsi]

    def get_itp_box_ctr(self):
        ctr_itp_box_mask = aa.Mask2D(
            mask=np.full(
                (self.shape_2d_dpsi[0] - 1, self.shape_2d_dpsi[1] - 1), False
            ),
            pixel_scales=self.dpix_dpsi,
        )
        ctr_itp_box = aa.Grid2D.from_mask(mask=ctr_itp_box_mask)
        self.xc_itp_box = np.asarray(ctr_itp_box.native[:, :, 1])
        self.yc_itp_box = np.asarray(ctr_itp_box.native[:, :, 0])

        self.mask_itp_box = coarse_interp_util.interp_box_mask_from(self.mask_dpsi)
        self.xc_itp_box_1d = self.xc_itp_box[~self.mask_itp_box]
        self.yc_itp_box_1d = self.yc_itp_box[~self.mask_itp_box]

        if len(self.xc_itp_box_1d) == 0:
            raise ValueError(
                "The dpsi grid is too sparse. "
                "Try decreasing the dpsi_factor to smaller values."
            )

    def get_dpsi2data_mapping(self):
        """
        The sparse matrix of shape
        [n_unmasked_data_pixels, n_unmasked_dpsi_pixels] mapping a vector on
        the coarser dpsi mesh to the finer data grid by bilinear
        interpolation.
        """
        self.itp_mat = coarse_interp_util.coarse_interp_matrix_from(
            self.mask_itp_box,
            self.xc_itp_box,
            self.yc_itp_box,
            self.xgrid_data_1d,
            self.ygrid_data_1d,
            self.xgrid_dpsi,
            self.ygrid_dpsi,
            self.mask_dpsi,
        )

    def get_gradient_operator_data(self):
        self.Hy_data, self.Hx_data = derivative_util.derivative_1st_operators_from(
            self.mask_data, pixel_scale=self.dpix_data
        )

    def get_gradient_operator_dpsi(self):
        self.Hy_dpsi, self.Hx_dpsi = derivative_util.derivative_1st_operators_from(
            self.mask_dpsi, pixel_scale=self.dpix_dpsi
        )

    def get_hamiltonian_operator_data(self):
        self.Hyy_data, self.Hxx_data = derivative_util.derivative_2nd_operators_from(
            self.mask_data, pixel_scale=self.dpix_data
        )
        self.hamiltonian_data = self.Hxx_data + self.Hyy_data

    def get_hamiltonian_operator_dpsi(self):
        self.Hyy_dpsi, self.Hxx_dpsi = derivative_util.derivative_2nd_operators_from(
            self.mask_dpsi, pixel_scale=self.dpix_dpsi
        )
        self.hamiltonian_dpsi = self.Hxx_dpsi + self.Hyy_dpsi

    def show_grid(self, output_file="grid.png"):
        from matplotlib import pyplot as plt

        plt.figure(figsize=(20, 20))
        plt.plot(
            self.xgrid_data.flatten(), self.ygrid_data.flatten(), "*", color="black", label="data"
        )
        plt.plot(
            self.xgrid_dpsi.flatten(), self.ygrid_dpsi.flatten(), "*", color="red", label="dpsi"
        )
        plt.plot(self.xgrid_data_1d, self.ygrid_data_1d, "o", color="black", label="data-unmask")
        plt.plot(self.xgrid_dpsi_1d, self.ygrid_dpsi_1d, "p", color="red", label="dpsi-unmask")
        plt.plot(self.xc_itp_box.flatten(), self.yc_itp_box.flatten(), "+", color="blue", label="dpsi-box")
        plt.plot(self.xc_itp_box_1d, self.yc_itp_box_1d, "+", color="red", label="dpsi-box-unmask")
        plt.legend()
        plt.savefig(output_file, bbox_inches="tight")
        plt.close()
