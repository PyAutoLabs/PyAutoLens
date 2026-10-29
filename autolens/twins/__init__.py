"""
``al.twins``: controlled, provenance-carrying simulations ("twins") of fitted lenses.

A fitted lens is ingested as a ``ReferenceModel`` (tracer with solved intensities,
``ObservationSetup`` and provenance). A ``Twin`` is that tracer and setup after a
chain of ``transform`` steps, each leaving a ``TransformationRecord``; simulating a
twin with a seed gives a ``SimulatedTwin`` whose FITS files sit beside its truth
tracer and ``twin.json``.

Three invariants:

1. A reference model is immutable, and one reference has many twins.
2. Preserving any parent property is a recorded choice, never implicit
   (``TransformationRecord.preserved`` is empty unless a transformation fills it).
3. Every simulated observation carries its ground truth, transformation chain,
   seed and library versions beside the FITS.

Default noise is the observed noise map (Gaussian, ``numpy.random.default_rng(seed)``);
Poisson noise is used only when the setup sets ``exposure_time`` and
``background_sky_level`` and ``noise="poisson"`` is requested.

``transform.ReplaceTotalMassWithStellarPlusDark`` replaces a galaxy's total mass with
stellar mass following its MGE light plus a dark halo, normalised inside the parent's
Einstein radius (``einstein_radius_aperture_from``); ``compare`` and ``compare_to_data``
report what a transformation changed, and ``aplt.subplot_twin_comparison`` plots it.

``model.stellar_dark_model_from`` builds the model that fits a stellar-plus-dark twin
with the parameterization it was made with (fixed MGE light, one shared mass-to-light
ratio, a dark halo centred on the light, free external shear; broad fixed priors that
never know the truth), and ``recovery.recovery_table_from`` compares a fit's posterior
medians and ``sigma`` bounds with the twin's ``truth`` (a ``RecoveryTable`` with
``all_inside`` and ``to_csv``). Both map ``truth`` keys onto model paths through
``model.truth_paths_from``.

NumPy only: nothing here supports JAX.
"""

from autolens.twins.record import TransformationRecord
from autolens.twins.setup import ObservationSetup
from autolens.twins.reference import ReferenceModel
from autolens.twins.twin import Twin
from autolens.twins.simulate import SimulatedTwin
from autolens.twins.aperture import einstein_radius_aperture_from
from autolens.twins.compare import TwinComparison, compare, compare_to_data
from autolens.twins.model import stellar_dark_model_from, truth_paths_from
from autolens.twins.recovery import RecoveryRow, RecoveryTable, recovery_table_from
from autolens.twins import model, recovery, transform

__all__ = [
    "ObservationSetup",
    "ReferenceModel",
    "Twin",
    "SimulatedTwin",
    "TransformationRecord",
    "TwinComparison",
    "compare",
    "compare_to_data",
    "einstein_radius_aperture_from",
    "stellar_dark_model_from",
    "truth_paths_from",
    "RecoveryRow",
    "RecoveryTable",
    "recovery_table_from",
    "model",
    "recovery",
    "transform",
]
