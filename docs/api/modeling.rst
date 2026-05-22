=============
Lens Modeling
=============

Analysis
========

The ``Analysis`` objects define the ``log_likelihood_function`` of how a lens model is fitted to a dataset.

It acts as an interface between the data, model and the non-linear search.

.. currentmodule:: autolens

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   AnalysisImaging
   AnalysisInterferometer

Non-linear Searches
-------------------

A non-linear search is an algorithm which fits a model to data.

**PyAutoGalaxy** currently supports three types of non-linear search algorithms: nested samplers,
Markov Chain Monte Carlo (MCMC) and Maximum Likelihood Estimaotrs (MLE).

.. currentmodule:: autofit

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   Nautilus
   LBFGS
   BFGS
   DynestyDynamic
   Emcee
   PySwarmsLocal
   PySwarmsGlobal

Priors
------

The priors of parameters of every component of a model, which is fitted to data, are customized using ``Prior`` objects.

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   UniformPrior
   GaussianPrior
   LogUniformPrior
   LogGaussianPrior

Adapt
-----

.. currentmodule:: autolens

.. autosummary::
   :toctree: generated/

   SetupAdapt