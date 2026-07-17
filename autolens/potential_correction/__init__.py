"""
The gravitational-imaging (potential correction) technique: pixelized
corrections dpsi to the lensing potential, reconstructed on a coarse
rectangular mesh — alone from an image residual (``FitDpsiImaging``) or
jointly with the pixelized source (``FitDpsiSrcImaging``) — with the
regularization strengths of both set by maximising the Bayesian evidence.

Ported from the ``potential_correction`` package of Cao et al. 2025
(https://github.com/caoxiaoyue/lensing_potential_correction). If you use this
functionality in your research, please cite Cao et al. 2025; citation
materials are provided at
https://github.com/caoxiaoyue/potential_correction_paper.
"""

from autolens.potential_correction import util
from autolens.potential_correction import visualize
from autolens.potential_correction.mesh import RegularDpsiMesh
from autolens.potential_correction.mesh import PairRegularDpsiMesh
from autolens.potential_correction.pixelization import DpsiLinearObj
from autolens.potential_correction.pixelization import DpsiPixelization
from autolens.potential_correction.pixelization import DpsiSrcPixelization
from autolens.potential_correction.src_factory import SrcFactory
from autolens.potential_correction.src_factory import AnalyticSrcFactory
from autolens.potential_correction.src_factory import PixSrcFactoryITP
from autolens.potential_correction.fit import FitDpsiImaging
from autolens.potential_correction.fit import FitDpsiSrcImaging
from autolens.potential_correction.analysis import DpsiInvAnalysis
from autolens.potential_correction.analysis import DpsiSrcInvAnalysis
