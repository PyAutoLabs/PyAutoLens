.. _galaxy-tracer:

Galaxies and ray tracing
========================

Build galaxies from light and mass profiles, then use a ``Tracer`` to calculate how their light is gravitationally lensed. The objects below separate individual galaxies, collections of galaxies and multi-plane ray tracing.

.. _id1:

Galaxies and collections
------------------------

``Galaxy`` holds components at a specified redshift. Component names such as ``bulge``, ``disk`` and ``mass`` let you identify their contributions. ``Galaxies`` evaluates a collection, combining its light or mass quantities as appropriate.

.. currentmodule:: autolens

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   Galaxy
   Galaxies

Use ``Tracer`` for lensing between redshift planes; a ``Galaxies`` collection alone does not perform multi-plane ray tracing.

Ray tracing
-----------

``Tracer`` groups galaxies into planes by redshift and propagates coordinates through those planes using the lensing cosmology. It provides lensed images and lensing quantities including deflections, convergence, magnification, critical curves and caustics.

.. code-block:: python

    import autolens as al

    lens = al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(einstein_radius=1.0),
    )
    source = al.Galaxy(
        redshift=1.0,
        bulge=al.lp.Sersic(intensity=1.0, effective_radius=0.2),
    )
    tracer = al.Tracer(galaxies=[lens, source])

This constructs a model; it does not fit data. The `ray-tracing guide <https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/guides/tracer.ipynb>`_ shows how to evaluate and plot it.

.. currentmodule:: autolens

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   Tracer

External mass fields
--------------------

``MassField`` holds mass components at their own redshift without assigning them galaxy light. Use it for environmental contributions such as external shear, a mass sheet or an external potential. It also accepts other mass profiles when they represent the environment.

Pass fields through ``Tracer(galaxies=..., fields=[field])``. Each field is placed at its redshift and contributes to the lensing calculations, while remaining separate from the galaxies. A single ``fields=field`` is also accepted; ``tracer.fields`` is stored as a list.

Attaching external mass components directly to a ``Galaxy`` remains supported.

.. currentmodule:: autolens

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   MassField

Fixed and fitted redshifts
--------------------------

For a known redshift, pass a number directly to ``Galaxy(redshift=...)``. When the redshift itself is to be inferred, ``Redshift`` provides a model component whose value can be assigned a prior.

.. code-block:: python

    import autofit as af

    redshift = af.Model(al.Redshift)
    redshift.redshift = af.UniformPrior(lower_limit=0.8, upper_limit=1.2)
    galaxy_model = af.Model(al.Galaxy, redshift=redshift)

The prior limits are illustrative, not a default recommendation. For lensing, choose redshifts and priors consistent with the intended ordering of lens and source planes. Add the light or mass components required by your model before fitting.

.. currentmodule:: autolens

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   Redshift

.. _galaxy-catalogues-csv:

Galaxy catalogues
-----------------

Load catalogue entries with ``y``, ``x`` and ``luminosity`` columns to construct populations of galaxies. ``galaxy_table_from_csv`` reads a table; ``galaxies_from_csv_tables`` and ``galaxy_af_models_from_csv_tables`` build galaxy instances or model components from the supplied tables.

The `scaling-relation examples <https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/scaling_relation/start_here.ipynb>`_ explain how measured luminosities can constrain the mass model of a companion population. The `multi-galaxy <https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/start_here.ipynb>`_, `group <https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/start_here.ipynb>`_ and `cluster <https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/cluster/start_here.ipynb>`_ introductions show the different modelling workflows.

.. currentmodule:: autolens

.. autosummary::
   :toctree: _autosummary

   galaxy_table_from_csv
   galaxies_from_csv_tables
   galaxy_af_models_from_csv_tables
