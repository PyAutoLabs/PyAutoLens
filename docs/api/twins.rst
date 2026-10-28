=====
Twins
=====

``al.twins`` turns a fitted lens into controlled, reproducible simulations
("twins") that carry their ground truth and provenance.

A fit is ingested as a ``ReferenceModel`` (the tracer with its solved
intensities, the ``ObservationSetup`` -- mask, PSF, observed noise map,
over-sampling -- and a provenance record with file hashes and library
versions). A ``Twin`` is that tracer and setup after a chain of
transformations, each leaving a ``TransformationRecord``. Simulating a twin with
a seed gives a ``SimulatedTwin``, written as FITS beside its truth tracer and a
``twin.json`` provenance file.

Three invariants:

1. A reference model is immutable, and one reference has many twins.
2. Preserving any parent property is a recorded choice, never implicit.
3. Every simulated observation carries its ground truth, transformation chain,
   seed and library versions beside the FITS.

The default noise is the observed noise map (Gaussian noise drawn with
``numpy.random.default_rng(seed)``); Poisson noise (``noise="poisson"``) needs
the setup's ``exposure_time`` and ``background_sky_level``.

.. code-block:: python

    import autolens as al

    reference = al.twins.ReferenceModel.from_fit_directory(
        "output/lens_fit/<identifier>", lens_id="lens_001"
    )
    twin = al.twins.Twin.baseline(reference).apply(al.twins.transform.Identity())
    simulated = twin.simulate(seed=1)
    simulated.output_to("twins/lens_001/baseline_seed_1")
    reloaded = al.twins.SimulatedTwin.from_directory("twins/lens_001/baseline_seed_1")
    resimulated = reloaded.resimulate()  # bit-identical to reloaded.dataset.data

Galaxies are addressed by name: a tracer loaded from a fit's ``files/tracer.json``
keeps the keys of the model's ``af.Collection`` (e.g. ``"lens"``, ``"source"``), and a
tracer built from a plain list names its galaxies ``"galaxy_0"``, ``"galaxy_1"``, ....
A transformation also accepts a role from the reference's provenance ``roles``
(``roles={"lens": "main_deflector"}`` lets ``galaxy="main_deflector"`` resolve to
``"lens"``). Component paths in a record read ``"galaxies.lens.mass"``.

Stellar plus dark matter
------------------------

``transform.ReplaceTotalMassWithStellarPlusDark`` replaces a galaxy's total-mass
profile(s) (every mass profile that is not light and not an ``ExternalShear``) with
stellar mass following its MGE light plus a dark-matter halo. The split follows the
mass-normalisation convention of PyAutoGalaxy's Mass Budget page
(``PyAutoGalaxy/docs/api/mass_budget.rst``, rulings R3 and R4):

- the aperture is a circle of radius theta_E,parent, the Einstein radius of the whole
  parent tracer (``LensCalc``, on the setup's masked grid), centred on the replaced
  galaxy's mass centre (``einstein_radius_aperture_from``);
- M_ref is the projected mass of the replaced component(s) inside it; external shear
  and other galaxies are untouched and excluded;
- ``ag.MassBudget`` solves one mass-to-light ratio for the light ``Basis`` (which is
  re-expressed as ``lmp.Gaussian`` members with identical light, so the image of the
  lens light is unchanged) and the ``kappa_s`` of an ``NFWSph`` (or ``gNFWSph`` with a
  fixed ``inner_slope``) centred on the light, with ``scale_radius_kpc`` converted to
  arc-seconds by the setup cosmology (or the tracer's) at the galaxy redshift.

The record's ``preserved`` names the one deliberate preservation, the galaxy's
projected mass inside theta_E,parent. Its ``procedure`` quotes the convention and the
derived numbers, and its ``truth`` (mass-to-light ratio, ``kappa_s``, scale radius in
arc-seconds and kpc, inner slope, dark fraction) is written flat under ``"truth"`` in
``twin.json``. A ``gNFW`` convergence does one ``quad`` per point: pass coarse
``aperture_settings`` for it. Budgets naming several galaxies raise
``NotImplementedError`` (ruling R7).

.. code-block:: python

    baseline = al.twins.Twin.baseline(reference)
    twin = baseline.apply(
        al.twins.transform.ReplaceTotalMassWithStellarPlusDark(
            galaxy="lens", light="bulge", scale_radius_kpc=20.0, f_dark=0.5
        )
    )
    twin.records[-1].truth  # flat ground truth

Diagnostics
-----------

``compare(parent, twin)`` returns a ``TwinComparison``: convergence and deflection
difference maps, enclosed-mass profiles M(<R) of both, tangential critical curves and
Einstein radii of both, and ``delta_chi_squared_image``, the noise-free
sum of (I_twin - I_parent)^2 / sigma^2 on the parent's masked grid with its PSF and noise
map. ``compare_to_data(twin, dataset)`` is a different number,
chi^2(twin | data) - chi^2(parent | data) from two ``al.FitImaging`` of the observed data
(parametric sources only; pixelized sources raise ``NotImplementedError``).
``aplt.subplot_twin_comparison`` plots a comparison.

.. code-block:: python

    import autolens.plot as aplt

    comparison = al.twins.compare(baseline, twin)
    comparison.delta_chi_squared_image
    delta_chi_squared_data = al.twins.compare_to_data(
        twin, al.twins.ObservationSetup.dataset_from_fit_directory(fit_path)
    )
    aplt.subplot_twin_comparison(comparison, output_path="plots", output_format="png")

API
---

.. currentmodule:: autolens.twins

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   ObservationSetup
   ReferenceModel
   Twin
   SimulatedTwin
   TransformationRecord
   TwinComparison
   transform.Identity
   transform.ReplaceTotalMassWithStellarPlusDark

.. autosummary::
   :toctree: _autosummary

   compare
   compare_to_data
   einstein_radius_aperture_from

.. currentmodule:: autolens.plot

.. autosummary::
   :toctree: _autosummary

   subplot_twin_comparison
