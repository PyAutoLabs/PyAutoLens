===============
Galaxy / Tracer
===============

Galaxy / Tracer
---------------

``Galaxy`` and ``Galaxies`` model individual galaxies with light and mass profiles at a
given redshift.  ``Tracer`` groups galaxies by redshift into planes and performs
multi-plane gravitational lensing ray-tracing, computing lensed images, convergence,
deflection angles, magnification, critical curves, and caustics.

``MassField`` holds the mass of everything *outside* the modelled system — an external
shear, a mass sheet, an external potential — at its own redshift and with no light of its
own; pass a list of them as ``Tracer(galaxies=..., fields=[field])`` and each is placed in
the plane at its redshift, contributing its mass to every lensing calculation while every
per-galaxy surface still sees galaxies alone. Attaching these components to a ``Galaxy``
instead remains fully supported.

.. currentmodule:: autolens

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   Galaxy
   Galaxies
   MassField
   Tracer

To treat the redshift of a galaxy as a free parameter in a model, the ``Redshift`` object must
be used.

This is because **PyAutoFit** (which handles model-fitting), requires all parameters to be a Python class.

The ``Redshift`` object does not need to be used for general **PyAutoGalaxy** use.

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

    Redshift

Galaxy Catalogues (CSV)
-----------------------

Load many galaxies from a ``y, x, luminosity`` CSV catalogue — the input format of the
scaling-relation galaxy tier used at multi-galaxy, group and cluster scale (see the
``multi_galaxy``, ``group`` and ``cluster`` packages of the ``autolens_workspace``).
``galaxy_table_from_csv`` reads the catalogue; the two ``*_from_csv_tables`` functions
build instances or model components from it.

.. autosummary::
   :toctree: _autosummary

   galaxy_table_from_csv
   galaxies_from_csv_tables
   galaxy_af_models_from_csv_tables