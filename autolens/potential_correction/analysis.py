"""
The ``af.Analysis`` classes of the gravitational-imaging (potential
correction) technique, through which a non-linear search samples the dpsi
mesh factor and regularization hyper-parameters by maximising the Bayesian
evidence of the (joint) inversion.

Ported from the ``potential_correction`` package of Cao et al. 2025
(https://github.com/caoxiaoyue/lensing_potential_correction). If you use this
functionality in your research, please cite Cao et al. 2025; citation
materials are provided at
https://github.com/caoxiaoyue/potential_correction_paper.
"""

import logging
import os
from typing import Optional

import numpy as np

import autofit as af
import autoarray as aa
from autoarray import exc

from autogalaxy.galaxy.galaxy import Galaxy

from autolens.potential_correction.fit import FitDpsiImaging, FitDpsiSrcImaging
from autolens.potential_correction.pixelization import DpsiSrcPixelization
from autolens.potential_correction.src_factory import SrcFactory

logger = logging.getLogger(__name__)


class DpsiInvAnalysis(af.Analysis):
    def __init__(
        self,
        masked_imaging,
        image_residual: np.ndarray,
        source_gradient: np.ndarray,
        anchor_points: Optional[np.ndarray] = None,
        preloads: Optional[dict] = None,
    ):
        """
        Samples the dpsi pixelization (mesh factor + regularization
        hyper-parameters) of a dpsi-only inversion of an image residual,
        with the inversion's Bayesian evidence as the likelihood.

        Parameters
        ----------
        masked_imaging
            The masked ``al.Imaging`` dataset.
        image_residual
            The 1D (slim) image residual of the smooth-model fit.
        source_gradient
            The [n_unmasked, 2] source gradients at the ray-traced image
            pixels.
        anchor_points
            The [3, 2] (y, x) anchor positions of the dpsi rescaling scheme.
        preloads
            Precomputed fit attributes shared across evaluations.
        """
        self.masked_imaging = masked_imaging
        self.image_residual = image_residual
        self.source_gradient = source_gradient
        self.anchor_points = anchor_points
        self.preloads = preloads

    def log_likelihood_function(self, instance):
        fit = FitDpsiImaging(
            masked_imaging=self.masked_imaging,
            image_residual=self.image_residual,
            source_gradient=self.source_gradient,
            anchor_points=self.anchor_points,
            dpsi_pixelization=instance,
            preloads=self.preloads,
        )
        return fit.log_evidence


class DpsiSrcInvAnalysis(af.Analysis):
    def __init__(
        self,
        masked_imaging,
        lens_start: Galaxy,
        source_start: SrcFactory,
        anchor_points: Optional[np.ndarray] = None,
        adapt_image=None,
        src_image_mesh=None,
        settings_inversion: Optional[aa.Settings] = None,
        preloads: Optional[dict] = None,
    ):
        """
        Samples the joint source+dpsi pixelization (source pixelization and
        dpsi mesh/regularization hyper-parameters) of a joint inversion,
        with the inversion's Bayesian evidence as the likelihood.

        Parameters
        ----------
        masked_imaging
            The masked ``al.Imaging`` dataset.
        lens_start
            The lens galaxy of the smooth-model fit the corrections perturb.
        source_start
            The source factory evaluated for the source gradients.
        anchor_points
            The [3, 2] (y, x) anchor positions of the dpsi rescaling scheme.
        adapt_image
            The adapt image of the source pixelization's image mesh.
        src_image_mesh
            An image mesh whose image-plane mesh grid is preloaded into the
            source inversion.
        settings_inversion
            The inversion settings; defaults to the positive-only solver
            with the border relocator.
        preloads
            Precomputed fit attributes shared across evaluations.
        """
        self.masked_imaging = masked_imaging
        self.anchor_points = anchor_points
        self.lens_start = lens_start
        self.source_start = source_start
        self.adapt_image = adapt_image
        self.src_image_mesh = src_image_mesh
        if settings_inversion is None:
            self.settings_inversion = aa.Settings(
                use_positive_only_solver=True,
                use_border_relocator=True,
            )
        else:
            self.settings_inversion = settings_inversion
        self.preloads = preloads

    def _fit_from(self, instance: DpsiSrcPixelization) -> FitDpsiSrcImaging:
        return FitDpsiSrcImaging(
            masked_imaging=self.masked_imaging,
            anchor_points=self.anchor_points,
            lens_start=self.lens_start,
            source_start=self.source_start,
            dpsi_pixelization=instance.dpsi_pixelization,
            src_pixelization=instance.src_pixelization,
            adapt_image=self.adapt_image,
            src_image_mesh=self.src_image_mesh,
            settings_inversion=self.settings_inversion,
            preloads=self.preloads,
        )

    def log_likelihood_function(self, instance: DpsiSrcPixelization):
        fit = self._fit_from(instance)
        try:
            return fit.log_evidence
        except exc.InversionException:
            # a failed inversion is a valid (very bad) sample, not a crash
            logger.exception(
                "InversionException during joint source+dpsi evidence evaluation; "
                "returning penalty likelihood."
            )
            return -1e8

    def visualize(self, paths: af.DirectoryPaths, instance, during_analysis=True):
        from autolens.potential_correction.visualize import show_fit_dpsi_src

        fit = self._fit_from(instance)
        fit.log_evidence

        os.makedirs(paths.image_path, exist_ok=True)
        show_fit_dpsi_src(fit=fit, output=f"{paths.image_path}/fit_dpsi_src.png")
