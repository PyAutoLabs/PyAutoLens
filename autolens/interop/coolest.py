"""
Import / export of lens models to COOLEST JSON template files
(https://github.com/aymgal/COOLEST).

This module writes and reads the analytic profile parameters of a lens model
— galaxies with light and mass profiles — as a COOLEST template, so PyAutoLens
results can be exchanged with other lens modeling codes (lenstronomy,
herculens, ...). Deflection / convergence maps and other derived data products
are not exported; use the profile objects returned by ``from_coolest`` to
compute them in either code.

The parameter-level conversions (ellipticity, position angle, intermediate-axis
radii, Einstein radius rescaling) live in ``autogalaxy.interop.coolest`` — see
that package's docstrings for the exact conventions and derivations.

Einstein radius definition
--------------------------

The ``theta_E`` written for SIE / PEMD profiles is the profile *parameter* in
the COOLEST convention: the Einstein radius of the equivalent circular profile
defined via the intermediate axis r = sqrt(a b) of the elliptical isodensity
contours. It is a property of the individual mass profile and does **not**
include external shear or other profiles. This differs from curve-based
definitions such as the Euclid DR1 catalogue's Einstein radius (the effective
radius of the area enclosed by the tangential critical curve, which responds
to shear); convert between the two through the ray-tracing API, not by
rescaling parameters.

NFW profiles require a physical density normalization (``rho_c``), so their
conversion uses the critical surface mass density between the galaxy's
redshift and the highest galaxy redshift in the model, computed from the
model's cosmology in units of solar masses per arcsec**2 (``rho_c`` is then
solar masses per arcsec**3).

The ``coolest`` package (``pip install autolens[coolest]``) is required only
by this module and is imported lazily.
"""

import os
from typing import Dict, List, Optional, Union

import autogalaxy as ag

from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.interop.coolest.light import coolest_dict_from_light
from autogalaxy.interop.coolest.light import light_profile_from
from autogalaxy.interop.coolest.mass import coolest_dict_from_mass
from autogalaxy.interop.coolest.mass import mass_profile_from
from autogalaxy.profiles.light.abstract import LightProfile
from autogalaxy.profiles.mass.abstract.abstract import MassProfile
from autogalaxy.profiles.mass.dark.nfw import NFW
from autogalaxy.profiles.mass.sheets.external_shear import ExternalShear
from autogalaxy.profiles.mass.sheets.mass_sheet import MassSheet

from autolens.lens.tracer import Tracer


def _coolest_modules():
    try:
        from coolest.template.classes.parameter import PointEstimate
        from coolest.template import lazy
        from coolest.template.standard import COOLEST
        from coolest.template.json import JSONSerializer
    except ImportError as e:
        raise ImportError(
            "The COOLEST interop module requires the `coolest` package, which "
            "is an optional dependency of PyAutoLens. Install it via "
            "`pip install autolens[coolest]` or `pip install coolest`."
        ) from e
    return PointEstimate, lazy, COOLEST, JSONSerializer


def _path_no_ext(file_path: str) -> str:
    # The coolest JSONSerializer rejects relative paths.
    file_path = os.path.abspath(file_path)
    if file_path.endswith(".json"):
        return file_path[: -len(".json")]
    return file_path


def _sigma_crit_from(cosmology, redshift_0: float, redshift_1: float) -> float:
    """
    The critical surface mass density between two redshifts in units of solar
    masses per arcsec**2 (the unit system the NFW converter documents).
    """
    sigma_crit_kpc2 = (
        cosmology.critical_surface_density_between_redshifts_solar_mass_per_kpc2_from(
            redshift_0=redshift_0, redshift_1=redshift_1
        )
    )
    kpc_per_arcsec = cosmology.kpc_per_arcsec_from(redshift=redshift_0)
    return float(sigma_crit_kpc2 * kpc_per_arcsec**2)


def _set_parameters(coolest_profile, parameters: Dict, point_estimate_cls):
    for name, value in parameters.items():
        coolest_profile.parameters[name].set_point_estimate(
            point_estimate_cls(float(value))
        )


def to_coolest(
    galaxies: Union[List[Galaxy], Tracer],
    file_path: str,
    cosmology: Optional[ag.cosmo.LensingCosmology] = None,
    mode: str = "MAP",
    pixel_size: float = 0.1,
    metadata: Optional[Dict] = None,
) -> str:
    """
    Export a lens model's analytic profile parameters to a COOLEST JSON
    template file.

    Every galaxy becomes a COOLEST ``Galaxy`` lensing entity with its light
    and mass profiles converted to COOLEST conventions. ``ExternalShear`` and
    ``MassSheet`` profiles are exported as COOLEST ``MassField`` entities (the
    standard treats external fields separately from galaxies).

    Parameters
    ----------
    galaxies
        The galaxies of the lens model (each with a ``redshift``), or a
        ``Tracer`` whose galaxies (and cosmology) are used.
    file_path
        The output path of the template; a ``.json`` extension is appended by
        the COOLEST serializer if not present.
    cosmology
        The cosmology written to the template and used to compute the critical
        surface density NFW profiles need. If a ``Tracer`` is passed its
        cosmology is used; defaults to ``ag.cosmo.Planck15``.
    mode
        The COOLEST template mode, e.g. "MAP" for an inferred model or "MOCK"
        for a simulated one.
    pixel_size
        The instrument pixel size (arcsec) written to the template's minimal
        instrument block.
    metadata
        Extra metadata stored in the template.

    Returns
    -------
    The path of the written ``.json`` template file.
    """
    PointEstimate, lazy, COOLEST, JSONSerializer = _coolest_modules()

    if isinstance(galaxies, Tracer):
        cosmology = cosmology or galaxies.cosmology
        galaxies = list(galaxies.galaxies)
    else:
        galaxies = list(galaxies)
        cosmology = cosmology or ag.cosmo.Planck15()

    redshift_max = max(galaxy.redshift for galaxy in galaxies)

    entities = []

    for i, galaxy in enumerate(galaxies):
        light_dicts = [
            coolest_dict_from_light(profile=profile)
            for profile in galaxy.cls_list_from(cls=LightProfile)
        ]

        mass_profiles = galaxy.cls_list_from(cls=MassProfile)

        sigma_crit = None
        if any(isinstance(profile, NFW) for profile in mass_profiles):
            if galaxy.redshift < redshift_max:
                sigma_crit = _sigma_crit_from(
                    cosmology=cosmology,
                    redshift_0=galaxy.redshift,
                    redshift_1=redshift_max,
                )

        field_profiles = []
        mass_dicts = []

        for profile in mass_profiles:
            if isinstance(profile, (ExternalShear, MassSheet)):
                field_profiles.append(profile)
            else:
                mass_dicts.append(
                    coolest_dict_from_mass(profile=profile, sigma_crit=sigma_crit)
                )

        light_model = lazy.LightModel(*[d["type"] for d in light_dicts])
        mass_model = lazy.MassModel(*[d["type"] for d in mass_dicts])

        for coolest_profile, profile_dict in zip(
            list(light_model) + list(mass_model), light_dicts + mass_dicts
        ):
            _set_parameters(
                coolest_profile=coolest_profile,
                parameters=profile_dict["parameters"],
                point_estimate_cls=PointEstimate,
            )

        entities.append(
            lazy.Galaxy(
                f"galaxy_{i}",
                float(galaxy.redshift),
                light_model=light_model,
                mass_model=mass_model,
            )
        )

        if field_profiles:
            field_dicts = [
                coolest_dict_from_mass(profile=profile)
                for profile in field_profiles
            ]
            field_model = lazy.MassModel(*[d["type"] for d in field_dicts])
            for coolest_profile, profile_dict in zip(
                list(field_model), field_dicts
            ):
                _set_parameters(
                    coolest_profile=coolest_profile,
                    parameters=profile_dict["parameters"],
                    point_estimate_cls=PointEstimate,
                )
            entities.append(
                lazy.MassField(
                    f"mass_field_{i}",
                    float(galaxy.redshift),
                    mass_model=field_model,
                )
            )

    h0 = getattr(cosmology.H0, "value", cosmology.H0)
    om0 = getattr(cosmology.Om0, "value", cosmology.Om0)

    metadata = dict(metadata or {})
    metadata.setdefault(
        "generated_by",
        "PyAutoLens (autolens.interop.coolest); theta_E is the profile's "
        "intermediate-axis Einstein radius, NFW rho_c is in solar masses per "
        "arcsec**3 with Sigma_crit in solar masses per arcsec**2.",
    )

    root = COOLEST(
        mode,
        lazy.CoordinatesOrigin(),
        lazy.LensingEntityList(*entities),
        lazy.Observation(),
        lazy.Instrument(pixel_size),
        cosmology=lazy.Cosmology(H0=float(h0), Om0=float(om0)),
        metadata=metadata,
    )

    path_no_ext = _path_no_ext(file_path)
    os.makedirs(os.path.dirname(path_no_ext), exist_ok=True)
    JSONSerializer(path_no_ext, obj=root, check_external_files=False).dump_simple()

    return f"{path_no_ext}.json"


def _parameters_from(coolest_profile) -> Dict:
    parameters = {}
    for name, parameter in coolest_profile.parameters.items():
        point_estimate = parameter.point_estimate
        value = None if point_estimate is None else point_estimate.value
        if value is None:
            raise ValueError(
                f"The COOLEST profile '{coolest_profile.type}' has no point "
                f"estimate for its parameter '{name}' — the template does not "
                "define a complete model."
            )
        parameters[name] = float(value)
    return parameters


def from_coolest(
    file_path: str,
    cosmology: Optional[ag.cosmo.LensingCosmology] = None,
) -> Tracer:
    """
    Build a ``Tracer`` from the analytic profiles of a COOLEST JSON template
    file.

    Every COOLEST ``Galaxy`` entity becomes a PyAutoLens galaxy with its light
    profiles (``light_0``, ``light_1``, ...) and mass profiles (``mass_0``,
    ...); every ``MassField`` entity becomes a galaxy holding its external
    mass profiles. All parameters are converted from COOLEST conventions to
    PyAutoLens conventions (see ``autogalaxy.interop.coolest``).

    Parameters
    ----------
    file_path
        The path of the ``.json`` template file.
    cosmology
        The cosmology of the returned ``Tracer``. Defaults to a
        ``FlatLambdaCDM`` built from the template's H0 / Om0 (or
        ``ag.cosmo.Planck15`` if the template has none).
    """
    _, _, _, JSONSerializer = _coolest_modules()

    root = JSONSerializer(
        _path_no_ext(file_path), check_external_files=False
    ).load(verbose=False)

    if cosmology is None:
        if root.cosmology is not None:
            cosmology = ag.cosmo.FlatLambdaCDM(
                H0=float(root.cosmology.H0), Om0=float(root.cosmology.Om0)
            )
        else:
            cosmology = ag.cosmo.Planck15()

    entities = list(root.lensing_entities)

    redshift_max = max(float(entity.redshift) for entity in entities)

    galaxies = []

    for entity in entities:
        profiles = {}

        light_model = getattr(entity, "light_model", None) or []
        for i, coolest_profile in enumerate(light_model):
            profiles[f"light_{i}"] = light_profile_from(
                profile_type=coolest_profile.type,
                parameters=_parameters_from(coolest_profile),
            )

        mass_model = getattr(entity, "mass_model", None) or []
        redshift = float(entity.redshift)

        sigma_crit = None
        if (
            any(coolest_profile.type == "NFW" for coolest_profile in mass_model)
            and redshift < redshift_max
        ):
            sigma_crit = _sigma_crit_from(
                cosmology=cosmology,
                redshift_0=redshift,
                redshift_1=redshift_max,
            )

        for i, coolest_profile in enumerate(mass_model):
            profiles[f"mass_{i}"] = mass_profile_from(
                profile_type=coolest_profile.type,
                parameters=_parameters_from(coolest_profile),
                sigma_crit=sigma_crit,
            )

        galaxies.append(
            Galaxy(redshift=float(entity.redshift), **profiles)
        )

    return Tracer(galaxies=galaxies, cosmology=cosmology)
