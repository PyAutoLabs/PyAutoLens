"""
Model components of the gravitational-imaging (potential correction)
technique: the dpsi pixelization (mesh + regularization), the joint
source+dpsi pixelization, and the linear-object adapter through which any
``AbstractRegularization`` builds the dpsi regularization matrix.

Ported from the ``potential_correction`` package of Cao et al. 2025
(https://github.com/caoxiaoyue/lensing_potential_correction). If you use this
functionality in your research, please cite Cao et al. 2025; citation
materials are provided at
https://github.com/caoxiaoyue/potential_correction_paper.
"""

import numpy as np

import autoarray as aa
from autoarray.inversion.regularization.abstract import AbstractRegularization

from autolens.potential_correction import mesh as dpsi_mesh


class DpsiMeshGrid:
    def __init__(self, points: np.ndarray):
        """
        Minimal grid wrapper exposing the ``array`` attribute the
        kernel regularizations read their pixel positions from.
        """
        self.array = points

    @property
    def shape(self):
        return self.array.shape

    def __array__(self, dtype=None):
        return np.asarray(self.array, dtype=dtype)


class DpsiLinearObj:
    def __init__(self, mask: np.ndarray, points: np.ndarray):
        """
        The linear-object adapter of the dpsi mesh, exposing the attributes
        the regularization schemes read: ``mask`` (the rectangular dpsi-mesh
        mask, used by ``aa.reg.CurvatureMask`` / ``aa.reg.FourthOrderMask``),
        ``source_plane_mesh_grid`` (the unmasked dpsi pixel positions, used
        by the kernel regularizations, e.g. ``aa.reg.MaternKernel``) and
        ``params`` (the parameter count, used by the regularization
        weights).

        Through this adapter every dpsi regularization is built via the
        standard ``regularization_matrix_from(linear_obj=...)`` interface —
        no scheme-specific dispatch is needed in the fit.

        Parameters
        ----------
        mask
            The 2D bool mask of the dpsi mesh (``True`` = masked).
        points
            The [n_unmasked_dpsi, 2] array of (y, x) positions of the
            unmasked dpsi pixels.
        """
        self.mask = mask
        self.source_plane_mesh_grid = DpsiMeshGrid(points=np.asarray(points))

    @property
    def params(self) -> int:
        return int(np.count_nonzero(~np.asarray(self.mask)))


class DpsiPixelization:
    def __init__(
        self, mesh: dpsi_mesh.RegularDpsiMesh, regularization: AbstractRegularization
    ):
        """
        The pixelization of the potential corrections: the dpsi mesh (its
        coarsening factor) plus the regularization scheme applied to the
        corrections (``aa.reg.CurvatureMask``, ``aa.reg.FourthOrderMask`` or
        a kernel regularization such as ``aa.reg.MaternKernel``).

        Parameters
        ----------
        mesh
            The ``RegularDpsiMesh`` defining the dpsi-mesh coarsening factor.
        regularization
            The regularization scheme applied to the potential corrections.
        """
        self.mesh = mesh
        self.regularization = regularization

    def pair_dpsi_data_mesh(self, mask, pixel_scale: float):
        return dpsi_mesh.PairRegularDpsiMesh(mask, pixel_scale, self.mesh.factor)


class DpsiSrcPixelization:
    def __init__(
        self, dpsi_pixelization: DpsiPixelization, src_pixelization: aa.Pixelization
    ):
        """
        The joint pixelization of a source+dpsi inversion: the dpsi
        pixelization of the potential corrections and the standard autolens
        pixelization of the source.

        Parameters
        ----------
        dpsi_pixelization
            The pixelization of the potential corrections.
        src_pixelization
            The pixelization of the source reconstruction.
        """
        self.dpsi_pixelization = dpsi_pixelization
        self.src_pixelization = src_pixelization
