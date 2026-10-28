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
   transform.Identity
