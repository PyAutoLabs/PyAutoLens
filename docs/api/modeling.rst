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

   AdaptImages

Model Utilities
---------------

Convenience functions which compose commonly used ``af.Model`` objects.

``mass_field_from`` builds the ``af.Model(al.MassField, ...)`` external-field model that
goes in a model's ``fields`` collection, sharing the centre of its ``MassSheet`` /
``ExternalPotential`` components with the lens galaxy model's mass centre.
The primary form is ``fields=af.Collection(field=field)``. For a single field,
``fields=field`` is also supported. It shortens prior paths from
``fields.field.shear.gamma_1`` to ``fields.shear.gamma_1`` and therefore gives a
different result identifier by design. Existing collection-form identifiers are unchanged.

.. currentmodule:: autolens.analysis.model_util

.. autosummary::
   :toctree: generated/

   mass_field_from
   mge_model_from
   mge_point_model_from
   hilbert_pixels_from_pixel_scale
