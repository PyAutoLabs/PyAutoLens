"""
Model construction utilities for **PyAutoLens** example scripts and pipelines.

This module provides convenience functions that build pre-configured ``af.Model``
objects for common lens modeling scenarios.  They are primarily intended for use in
the autolens_workspace ``start_here.py`` scripts and SLaM pipeline templates, where
a sensible default model is needed without the user having to specify every prior
explicitly.

Key functions re-exported from ``autogalaxy``:
- ``mge_model_from`` — build an MGE (Multi-Gaussian Expansion) light profile model.
- ``mge_point_model_from`` — MGE model for point-source fitting.
- ``hilbert_pixels_from_pixel_scale`` — estimate Hilbert image-mesh pixel count.

PyAutoLens-specific:
- ``random_galaxies_for_simulation_from`` — sample concrete (lens, source) ``Galaxy``
  instances for synthetic-data generation in ``start_here`` scripts.
- ``mass_field_from`` — build the ``af.Model(al.MassField, ...)`` external-field model that
  goes in a model's ``fields`` collection.
"""
from typing import Optional, Tuple

import numpy as np

import autofit as af

import autolens as al

from autogalaxy.analysis.model_util import mge_model_from
from autogalaxy.analysis.model_util import mge_point_model_from
from autogalaxy.analysis.model_util import hilbert_pixels_from_pixel_scale


SIMULATOR_RANDOM_LENS_SUMMARY = (
    "Each simulated strong lens draws fresh truths from: "
    "lens bulge SNR in [20, 60] (when included), "
    "lens mass einstein_radius in [0.2, 1.8] with normal-clipped ellipticity, "
    "external shear ~ Normal(0, 0.05), "
    "source bulge SNR in [10, 30] / point-source flux in [0.0, 2.0] (mode dependent)."
)


def _clipped_ell_comp(rng: np.random.Generator) -> float:
    return float(np.clip(rng.normal(0.0, 0.2), -1.0, 1.0))


def random_galaxies_for_simulation_from(
    include_lens_light: bool = True,
    use_point_source: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Tuple["al.Galaxy", "al.Galaxy"]:
    """
    Sample a ``(lens_galaxy, source_galaxy)`` pair for synthetic strong-lens
    data generation.

    Each parameter is drawn directly from a numpy ``Generator`` and used to
    construct concrete profile instances — no ``af.Model`` priors are involved.
    SNR-normalised Sersic profiles (``lp_snr.Sersic``) are used for diffuse
    light components so that simulator output lands at a controlled target
    SNR; the SNR appears as a profile attribute on the *instance*, never as a
    fitting parameter.

    Do **not** use the returned galaxies as fitting models. They are
    instances, suitable for ``Tracer`` / ``simulator.via_tracer_from``.

    Parameters
    ----------
    include_lens_light
        If True (default), give the lens galaxy an ``lp_snr.Sersic`` bulge.
        If False, the lens is mass-only.
    use_point_source
        If True, source is a ``PointFlux`` with random centre and flux. If
        False (default), source is an ``lp_snr.Sersic``.
    rng
        Optional ``numpy.random.Generator``. If ``None`` a fresh
        ``default_rng()`` is created on each call.

    Returns
    -------
    (Galaxy, Galaxy)
        ``(lens_galaxy, source_galaxy)`` at redshifts 0.5 and 1.0 respectively.
    """
    rng = rng if rng is not None else np.random.default_rng()

    if include_lens_light:
        lens_bulge = al.lp_snr.Sersic(
            centre=(0.0, 0.0),
            ell_comps=(_clipped_ell_comp(rng), _clipped_ell_comp(rng)),
            effective_radius=float(rng.uniform(1.0, 5.0)),
            sersic_index=float(rng.uniform(3.5, 4.5)),
            signal_to_noise_ratio=float(rng.uniform(20.0, 60.0)),
        )
    else:
        lens_bulge = None

    mass = al.mp.Isothermal(
        centre=(0.0, 0.0),
        ell_comps=(_clipped_ell_comp(rng), _clipped_ell_comp(rng)),
        einstein_radius=float(rng.uniform(0.2, 1.8)),
    )

    shear = al.mp.ExternalShear(
        gamma_1=float(rng.normal(0.0, 0.05)),
        gamma_2=float(rng.normal(0.0, 0.05)),
    )

    lens = al.Galaxy(redshift=0.5, bulge=lens_bulge, mass=mass, shear=shear)

    if use_point_source:
        point_0 = al.ps.PointFlux(
            centre=(float(rng.normal(0.0, 0.3)), float(rng.normal(0.0, 0.3))),
            flux=float(rng.uniform(0.0, 2.0)),
        )
        source = al.Galaxy(redshift=1.0, point_0=point_0)
    else:
        source_bulge = al.lp_snr.Sersic(
            centre=(float(rng.normal(0.0, 0.3)), float(rng.normal(0.0, 0.3))),
            ell_comps=(_clipped_ell_comp(rng), _clipped_ell_comp(rng)),
            effective_radius=float(rng.uniform(0.01, 3.0)),
            sersic_index=float(rng.uniform(1.5, 2.5)),
            signal_to_noise_ratio=float(rng.uniform(10.0, 30.0)),
        )
        source = al.Galaxy(redshift=1.0, bulge=source_bulge)

    return lens, source


def mass_field_from(
    lens: af.Model,
    shear: bool = True,
    mass_sheet: bool = False,
    potential: bool = False,
    redshift: Optional[float] = None,
) -> af.Model:
    """
    Build the ``af.Model(al.MassField, ...)`` describing the tidal field of everything
    outside the modelled system, with its geometry tied to a lens galaxy model.

    The external field is composed of the components no galaxy owns — an ``ExternalShear``,
    a ``MassSheet``, an ``ExternalPotential`` — and is placed in a model's ``fields``
    collection rather than bolted onto a galaxy::

        lens = af.Model(al.Galaxy, redshift=0.5, mass=af.Model(al.mp.Isothermal))

        model = af.Collection(
            galaxies=af.Collection(lens=lens, source=source),
            fields=af.Collection(field=al.model_util.mass_field_from(lens=lens)),
        )

    For a single field, ``fields=al.model_util.mass_field_from(lens=lens)`` is also
    supported. This shortens prior paths (``fields.shear.gamma_1`` instead of
    ``fields.field.shear.gamma_1``), so the two forms have different result identifiers
    by design. The collection form above remains the primary example and retains its
    existing identifier.

    The ``MassSheet`` and ``ExternalPotential`` have a meaningful centre (unlike an
    ``ExternalShear``, whose deflection field is constant), and that centre is physically
    the centre of the system the field is expanded about. It is therefore **shared** with
    the lens galaxy's mass centre — the same prior object, not a copy — so the two move
    together in the non-linear search and cost no extra dimensions.

    In a multi-deflector system, ``lens`` is the *primary* deflector: the galaxy whose
    centre the external expansion is about. Pass that galaxy's model, not a secondary one.

    The galaxy-attached form (``af.Model(al.Galaxy, ..., shear=af.Model(al.mp.ExternalShear))``)
    remains fully supported and is not deprecated; this is a second way to write the same
    physics, one which keeps the external field out of the galaxy.

    Parameters
    ----------
    lens
        The ``af.Model`` of the (primary) lens galaxy. Its ``mass.centre`` is shared with the
        field's centred components and, unless ``redshift`` is given, its redshift is the
        field's redshift.
    shear
        If True (default), the field includes an ``ExternalShear``.
    mass_sheet
        If True, the field includes a ``MassSheet``, centred on the lens mass.
    potential
        If True, the field includes an ``ExternalPotential``, centred on the lens mass.
    redshift
        The redshift of the field. Defaults to the lens galaxy model's redshift.

    Returns
    -------
    The ``af.Model`` of the ``MassField``, to be placed in the model's ``fields`` collection
    or supplied directly as ``fields`` for a single field.
    """
    if not hasattr(lens, "mass"):
        raise ValueError(
            f"The `lens` model passed to `mass_field_from` has no `mass` attribute: "
            f"{lens!r}. The external field's centred components are tied to the lens "
            f"galaxy's mass centre, so the lens model must have a `mass`, e.g. "
            f"af.Model(al.Galaxy, redshift=0.5, mass=af.Model(al.mp.Isothermal))."
        )

    redshift = lens.redshift if redshift is None else redshift

    components = {}

    if shear:
        components["shear"] = af.Model(al.mp.ExternalShear)

    if mass_sheet:
        components["mass_sheet"] = af.Model(al.mp.MassSheet)

    if potential:
        components["potential"] = af.Model(al.mp.ExternalPotential)

    field = af.Model(al.MassField, redshift=redshift, **components)

    # The *same* prior object, not a copy: `field.potential.centre is lens.mass.centre` must
    # hold, so the external expansion and the lens mass share one centre in the search.
    if mass_sheet:
        field.mass_sheet.centre = lens.mass.centre

    if potential:
        field.potential.centre = lens.mass.centre

    return field
