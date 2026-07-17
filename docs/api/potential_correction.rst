====================
Potential Correction
====================

The gravitational-imaging (potential correction) technique: pixelized
corrections to the lensing potential, reconstructed alone from an image
residual or jointly with the pixelized source by maximising the Bayesian
evidence.

Ported from the ``potential_correction`` package of Cao et al. 2025
(https://github.com/caoxiaoyue/lensing_potential_correction). If you use this
functionality in your research, please cite Cao et al. 2025; citation
materials are provided at
https://github.com/caoxiaoyue/potential_correction_paper.

.. currentmodule:: autolens.pc

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   RegularDpsiMesh
   PairRegularDpsiMesh
   DpsiPixelization
   DpsiSrcPixelization
   SrcFactory
   AnalyticSrcFactory
   PixSrcFactoryITP
   FitDpsiImaging
   FitDpsiSrcImaging
   DpsiInvAnalysis
   DpsiSrcInvAnalysis
