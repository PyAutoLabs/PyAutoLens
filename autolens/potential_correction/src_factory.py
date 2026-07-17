"""
Source factories of the gravitational-imaging (potential correction)
technique: callables evaluating the source brightness and its gradients at
arbitrary source-plane positions, from an analytic galaxy or a pixelized
reconstruction.

Ported from the ``potential_correction`` package of Cao et al. 2025
(https://github.com/caoxiaoyue/lensing_potential_correction). If you use this
functionality in your research, please cite Cao et al. 2025; citation
materials are provided at
https://github.com/caoxiaoyue/potential_correction_paper.
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial import Delaunay

import autoarray as aa

from autogalaxy.profiles.mass.input.interp import LinearNDInterpolatorExt
from autogalaxy.galaxy.galaxy import Galaxy

from autolens.potential_correction import util as pc_util

MESH2D_VORONOI_CLS = getattr(aa, "Mesh2DVoronoi", None)


class SrcFactory(ABC):
    @abstractmethod
    def eval_func(self, xgrid: np.ndarray, ygrid: np.ndarray) -> np.ndarray:
        """The source brightness at the input (x, y) positions."""

    def eval_grad(self, xgrid: np.ndarray, ygrid: np.ndarray, cross_size=0.001):
        """
        The (dS/dy, dS/dx) source gradients at the input (x, y) positions,
        by central differences over a cross of half-length ``cross_size``.
        """
        if xgrid.shape != ygrid.shape:
            raise ValueError("xgrid and ygrid must have the same shape")
        origin_shape = xgrid.shape
        points = np.vstack((np.ravel(ygrid), np.ravel(xgrid))).T
        grad_points = pc_util.gradient_points_from(points, cross_size=cross_size)
        values_at_grad_points = self.eval_func(grad_points[:, 1], grad_points[:, 0])
        src_grad_values = pc_util.source_gradient_from(
            values_at_grad_points, grad_points
        )
        return src_grad_values.reshape(*origin_shape, 2)


class AnalyticSrcFactory(SrcFactory):
    def __init__(self, source_galaxy: Galaxy):
        """
        A source factory evaluating an analytic source galaxy's light
        profiles directly.
        """
        self.source_galaxy = source_galaxy

    def eval_func(self, xgrid: np.ndarray, ygrid: np.ndarray):
        grid_flat = np.vstack([np.ravel(ygrid), np.ravel(xgrid)]).T
        func_values = self.source_galaxy.image_2d_from(
            grid=aa.Grid2DIrregular(values=grid_flat)
        )
        return np.asarray(func_values).reshape(np.shape(xgrid))


class PixSrcFactoryITP(SrcFactory):
    def __init__(self, points: np.ndarray, values: np.ndarray):
        """
        A source factory interpolating a pixelized source reconstruction
        (Delaunay linear interpolation with nearest-neighbour fallback).
        Gradient evaluation uses per-point cross sizes set by the Voronoi
        cell areas of the reconstruction's mesh.

        Parameters
        ----------
        points
            The [n_points, 2] source-plane positions of the reconstruction,
            in autolens (y, x) order.
        values
            The reconstructed source brightness at those positions.
        """
        self.points = points
        self.values = values
        try:
            if MESH2D_VORONOI_CLS is None:
                raise AttributeError("aa.Mesh2DVoronoi is unavailable")
            self.vor_mesh = MESH2D_VORONOI_CLS(values=self.points)
            self._split_cross = self.vor_mesh.split_cross
        except Exception:
            self._split_cross = pc_util.split_cross_from(self.points)

    def eval_func(self, xgrid: np.ndarray, ygrid: np.ndarray):
        if not hasattr(self, "tri"):
            self.tri = Delaunay(np.fliplr(self.points))
        if not hasattr(self, "interp_func"):
            self.interp_func = LinearNDInterpolatorExt(self.tri, self.values)
        return self.interp_func(xgrid, ygrid)

    def eval_grad(self, xgrid: np.ndarray, ygrid: np.ndarray, cross_size=None):
        if cross_size is None:
            grad_points = self._split_cross
        else:
            grad_points = pc_util.gradient_points_from(
                self.points, cross_size=cross_size
            )
        values_at_grad_points = self.eval_func(grad_points[:, 1], grad_points[:, 0])
        if not hasattr(self, "interp_grad_func"):
            src_grad_values = pc_util.source_gradient_from(
                values_at_grad_points, grad_points
            )
            self.interp_grad_func = LinearNDInterpolatorExt(self.tri, src_grad_values)
        return self.interp_grad_func(xgrid, ygrid)
