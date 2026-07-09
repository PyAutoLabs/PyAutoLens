=============
Point Sources
=============

Point sources arise when the background object is compact (e.g. a quasar, supernova, or
compact radio source) and is modelled by its image-plane positions, flux ratios, and/or
time delays rather than by a resolved surface-brightness distribution.

Dataset
-------

Data structures holding the observed positions, fluxes, and time delays of one or more
named point sources.

.. currentmodule:: autolens

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   PointDataset

Solver
------

The ``PointSolver`` finds the image-plane positions that correspond to a given
source-plane coordinate by solving the lens equation numerically via a triangle-tiling
approach.

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   PointSolver

Fitting
-------

Fit classes for point-source positions, flux ratios, and time delays.  ``FitPointDataset``
orchestrates all active components; the individual fit classes can also be used standalone.

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   FitPointDataset
   FitPositionsImagePair
   FitPositionsImagePairAll
   FitPositionsImagePairRepeat
   FitPositionsSource
   FitFluxes
   FitTimeDelays

Position Likelihood
-------------------

``PositionsLH`` adds a penalty to the log likelihood when the observed image positions do
not self-consistently trace back to the same source-plane location, guiding the non-linear
search toward physically consistent mass models.

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   PositionsLH

Analysis
--------

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   AnalysisPoint