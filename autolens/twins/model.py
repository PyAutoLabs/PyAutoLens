"""
``stellar_dark_model_from``: the inference model of a stellar-plus-dark twin, with no
misspecification.

A twin made by ``transform.ReplaceTotalMassWithStellarPlusDark`` has a lens galaxy whose
mass is a ``Basis`` of ``lmp.Gaussian`` (its MGE light times one mass-to-light ratio)
plus a dark halo. ``stellar_dark_model_from`` returns the ``af.Collection`` that fits
that twin with the same parameterization: the light of each Gaussian is fixed to the
twin's, and only the mass-to-light ratio, the dark halo and the external shear are
free. It is the twin-based counterpart of
``ag.analysis.chaining_util.mass_light_dark_basis_from``, which builds the same basis
from a light-pipeline ``Result``.

``truth_paths_from`` maps the transformation's ``truth`` keys onto the model paths of
the free parameters; ``al.twins.recovery.recovery_table_from`` uses it, so the model
and the recovery table cannot disagree about where a parameter lives.

NumPy only: nothing here supports JAX.
"""

import inspect
from typing import Dict, Optional, Tuple

import autofit as af
import autogalaxy as ag
from autogalaxy.profiles.mass.abstract.abstract import MassProfile

from autolens.twins._galaxies import galaxy_dict_from, galaxy_name_from
from autolens.twins.transform.stellar_dark import NAME as STELLAR_DARK_NAME

# The default priors of the free parameters, by ``truth`` key. They are broad, fixed
# and never derived from the truth (a prior centred on the answer defeats the
# experiment):
#
# - ``mass_to_light_ratio``: log-uniform over 1e-6 to 1e6 (the PyAutoGalaxy default
#   for stellar profiles). Its units are convergence per unit image intensity, so its
#   scale depends on the data units; twelve decades cover any of them.
# - ``kappa_s``: log-uniform over 1e-4 to 10.
# - ``scale_radius_arcsec``: uniform over 0.1 to 30 arc-seconds (the NFW default's
#   upper limit; roughly 1 to 100 kpc at the redshifts of galaxy-scale lenses).
# - ``inner_slope``: uniform over 0 to 2 (cored to steeper than isothermal).
DEFAULT_PRIORS = {
    "mass_to_light_ratio": lambda: af.LogUniformPrior(
        lower_limit=1.0e-6, upper_limit=1.0e6
    ),
    "kappa_s": lambda: af.LogUniformPrior(lower_limit=1.0e-4, upper_limit=10.0),
    "scale_radius_arcsec": lambda: af.UniformPrior(lower_limit=0.1, upper_limit=30.0),
    "inner_slope": lambda: af.UniformPrior(lower_limit=0.0, upper_limit=2.0),
}

# The ``truth`` key of each recoverable parameter and the name of the model parameter
# that holds it; ``truth_paths_from`` finds each name under ``galaxies.<galaxy>``.
TRUTH_PARAMETER_NAMES = {
    "mass_to_light_ratio": "mass_to_light_ratio",
    "kappa_s": "kappa_s",
    "scale_radius_arcsec": "scale_radius",
    "inner_slope": "inner_slope",
}

_DARK_FREE_NAMES = ("kappa_s", "scale_radius", "inner_slope")


def _has_free_inner_slope(dark_cls) -> bool:
    """Whether ``inner_slope`` is a constructor argument (``NFWSph`` fixes it to 1)."""
    return "inner_slope" in inspect.signature(dark_cls.__init__).parameters


def _is_external_shear(profile) -> bool:
    return isinstance(profile, ag.mp.ExternalShear)


def stellar_dark_record_from(twin, galaxy: Optional[str] = None):
    """
    The last ``ReplaceTotalMassWithStellarPlusDark`` record of ``twin`` (for ``galaxy``,
    a galaxy name or role, when given).

    Raises
    ------
    ValueError
        If the twin has no such record.
    """
    name = None if galaxy is None else galaxy_name_from(twin, galaxy)
    for record in reversed(twin.records):
        if record.name != STELLAR_DARK_NAME:
            continue
        if name is None or record.truth.get("galaxy") == name:
            return record
    raise ValueError(
        f"The twin has no {STELLAR_DARK_NAME} record"
        + ("" if name is None else f" for galaxy {name!r}")
        + f" (records: {[record.name for record in twin.records]}); apply "
        f"al.twins.transform.ReplaceTotalMassWithStellarPlusDark first."
    )


def _prior_from(key: str, priors: Optional[Dict]):
    if priors is not None and key in priors:
        return priors[key]
    return DEFAULT_PRIORS[key]()


def _stellar_model_from(light_basis, mass_to_light_ratio_prior) -> af.Model:
    """
    A ``Basis`` of ``lmp.Gaussian`` models with every member's centre, ``ell_comps``,
    ``sigma`` and intensity fixed to the twin's and one mass-to-light ratio shared by
    all members (as ``mass_light_dark_basis_from`` ties it).
    """
    model_list = []
    for profile in light_basis.profile_list:
        model = af.Model(ag.lmp.Gaussian)
        model.centre = tuple(float(value) for value in profile.centre)
        model.ell_comps = tuple(float(value) for value in profile.ell_comps)
        model.intensity = float(profile.intensity)
        model.sigma = float(profile.sigma)
        model.mass_to_light_ratio = mass_to_light_ratio_prior
        model_list.append(model)
    return af.Model(ag.lp_basis.Basis, profile_list=model_list)


def _dark_model_from(dark, truth, free_scale_radius, free_inner_slope, priors):
    """
    The dark profile's model: centre fixed to the light centre, free ``kappa_s``, free
    (or truth-fixed) ``scale_radius`` and, if the class has one, a free (or
    truth-fixed) ``inner_slope``. Any other constructor argument is fixed to the twin's.
    """
    dark_cls = type(dark)
    model = af.Model(dark_cls)

    for name in inspect.signature(dark_cls.__init__).parameters:
        if name in ("self", "centre") or name in _DARK_FREE_NAMES:
            continue
        if hasattr(dark, name):
            setattr(model, name, getattr(dark, name))

    model.centre = tuple(float(value) for value in truth["dark_centre"])
    model.kappa_s = _prior_from("kappa_s", priors)
    model.scale_radius = (
        _prior_from("scale_radius_arcsec", priors)
        if free_scale_radius
        else float(truth["scale_radius_arcsec"])
    )
    if _has_free_inner_slope(dark_cls):
        model.inner_slope = (
            _prior_from("inner_slope", priors)
            if free_inner_slope
            else float(truth["inner_slope"])
        )
    return model


def _field_model_from(field, free_shear):
    """A ``MassField`` model whose ``ExternalShear`` components are free if asked."""
    components = {
        key: (
            af.Model(ag.mp.ExternalShear)
            if free_shear and _is_external_shear(profile)
            else profile
        )
        for key, profile in field.profile_dict.items()
    }
    return af.Model(ag.MassField, redshift=field.redshift, **components)


def _source_name_from(galaxies: Dict, lens_name: str) -> str:
    """``"source"`` when the tracer has one, else the single highest-redshift galaxy."""
    if "source" in galaxies:
        return "source"
    redshifts = {
        name: galaxy.redshift for name, galaxy in galaxies.items() if name != lens_name
    }
    if not redshifts:
        raise ValueError("The twin's tracer has no galaxy besides the lens.")
    highest = max(redshifts.values())
    names = [name for name, redshift in redshifts.items() if redshift == highest]
    if len(names) > 1:
        raise ValueError(
            f"Several galaxies ({names}) share the highest redshift; pass source_model "
            f"as a dictionary keyed by galaxy name."
        )
    return names[0]


def stellar_dark_model_from(
    twin,
    galaxy: Optional[str] = None,
    *,
    source_model,
    free_scale_radius: bool = True,
    free_inner_slope: Optional[bool] = None,
    free_shear: bool = True,
    priors: Optional[Dict] = None,
) -> af.Collection:
    """
    The model that fits a stellar-plus-dark twin with the parameterization it was made
    with, as ``af.Collection(galaxies=af.Collection(<name>=..., ...), fields=...)``.

    Everything is read from the last ``ReplaceTotalMassWithStellarPlusDark`` record of
    the twin (its ``requested`` light and dark attribute names, and its ``truth``) and
    from the twin's tracer, not by guessing attribute names:

    - the lens galaxy's light attribute (``requested["light"]``, e.g. ``bulge``) is a
      ``Basis`` of ``af.Model(lmp.Gaussian)``, each member's centre, ``ell_comps``,
      ``sigma`` and intensity fixed to the twin's, all sharing ONE free
      ``mass_to_light_ratio`` (the same prior object, so the model has one parameter);
    - the dark attribute (``requested["dark_name"]``) is ``af.Model`` of the twin's dark
      class, centred on the light (``truth["dark_centre"]``, fixed), with free
      ``kappa_s``, free ``scale_radius`` (fixed to the truth when
      ``free_scale_radius=False``) and, for a class with an ``inner_slope`` (e.g.
      ``gNFWSph``), a free ``inner_slope`` unless ``free_inner_slope=False``;
    - an ``ExternalShear`` attribute of the lens galaxy, or an ``ExternalShear``
      component of a tracer ``MassField`` (the workspace convention,
      ``fields=af.Model(al.MassField, redshift=..., shear=af.Model(al.mp.ExternalShear))``),
      becomes a free ``af.Model(ExternalShear)`` when ``free_shear`` (else it is fixed to
      the twin's). One field is passed as ``fields=<model>``; several as
      ``fields=af.Collection(field_0=..., ...)``. The field's component names are kept;
      the tracer does not keep field names, so the single-field slot is ``fields``;
    - any other attribute of the lens galaxy is fixed to the twin's;
    - ``source_model`` replaces the source galaxy (the galaxy named ``"source"``, else
      the single highest-redshift galaxy), or, as a dictionary, the galaxies it names;
    - every other galaxy is kept as its fixed instance under its name.

    Default priors (``DEFAULT_PRIORS``, by ``truth`` key) are broad, fixed and never
    derived from the truth: ``mass_to_light_ratio`` log-uniform over [1e-6, 1e6],
    ``kappa_s`` log-uniform over [1e-4, 10], ``scale_radius_arcsec`` uniform over
    [0.1, 30] arc-seconds and ``inner_slope`` uniform over [0, 2].

    Parameters
    ----------
    twin
        A ``Twin`` with a ``ReplaceTotalMassWithStellarPlusDark`` record.
    galaxy
        The transformed galaxy (name or role); defaults to the galaxy of the last
        stellar-dark record.
    source_model
        The source galaxy model (e.g. ``af.Model(al.Galaxy, redshift=1.0, bulge=...)``)
        or a dictionary ``{galaxy name: model}``.
    free_scale_radius
        Fit the dark scale radius (default) or fix it to the truth.
    free_inner_slope
        Fit the inner slope of a dark class that has one. ``None`` (default) means free
        when the class has an ``inner_slope``. ``True`` for a class without one raises.
    free_shear
        Fit the external shear (default) or fix it to the twin's.
    priors
        Overrides of ``DEFAULT_PRIORS``, by ``truth`` key, e.g.
        ``{"kappa_s": af.LogUniformPrior(lower_limit=1e-3, upper_limit=1.0)}``.

    Raises
    ------
    ValueError
        If the twin has no ``ReplaceTotalMassWithStellarPlusDark`` record, or a
        ``priors`` key is not a ``DEFAULT_PRIORS`` key.
    """
    if priors is not None:
        unknown = sorted(set(priors) - set(DEFAULT_PRIORS))
        if unknown:
            raise ValueError(
                f"Unknown priors keys {unknown}; the keys are {list(DEFAULT_PRIORS)}."
            )

    record = stellar_dark_record_from(twin, galaxy)
    truth = record.truth
    lens_name = truth["galaxy"]
    light_name = record.requested["light"]
    dark_name = record.requested["dark_name"]

    tracer = twin.tracer
    galaxies = galaxy_dict_from(tracer)
    lens = galaxies[lens_name]

    dark = getattr(lens, dark_name, None)
    if free_inner_slope is None:
        free_inner_slope = dark is not None and _has_free_inner_slope(type(dark))
    elif free_inner_slope and not _has_free_inner_slope(type(dark)):
        raise ValueError(
            f"free_inner_slope=True but the dark profile "
            f"{type(dark).__name__} has no inner_slope."
        )

    lens_components = {}
    for key, profile in lens.profile_dict.items():
        if key == light_name and all(
            isinstance(member, ag.lmp.Gaussian)
            for member in getattr(profile, "profile_list", [])
        ):
            lens_components[key] = _stellar_model_from(
                light_basis=profile,
                mass_to_light_ratio_prior=_prior_from("mass_to_light_ratio", priors),
            )
        elif key == dark_name and isinstance(profile, MassProfile):
            lens_components[key] = _dark_model_from(
                dark=profile,
                truth=truth,
                free_scale_radius=free_scale_radius,
                free_inner_slope=free_inner_slope,
                priors=priors,
            )
        elif free_shear and _is_external_shear(profile):
            lens_components[key] = af.Model(ag.mp.ExternalShear)
        else:
            lens_components[key] = profile

    if isinstance(source_model, dict):
        source_models = dict(source_model)
    else:
        source_models = {_source_name_from(galaxies, lens_name): source_model}
    unknown = sorted(set(source_models) - set(galaxies))
    if unknown:
        raise ValueError(
            f"source_model names galaxies {unknown} that are not in the twin's tracer "
            f"({list(galaxies)})."
        )

    galaxy_models = {}
    for name, instance in galaxies.items():
        if name == lens_name:
            galaxy_models[name] = af.Model(
                ag.Galaxy, redshift=lens.redshift, **lens_components
            )
        elif name in source_models:
            galaxy_models[name] = source_models[name]
        else:
            galaxy_models[name] = instance

    model = af.Collection(galaxies=af.Collection(**galaxy_models))

    fields = list(getattr(tracer, "fields", None) or [])
    if len(fields) == 1:
        model.fields = _field_model_from(fields[0], free_shear=free_shear)
    elif len(fields) > 1:
        model.fields = af.Collection(
            **{
                f"field_{index}": _field_model_from(field, free_shear=free_shear)
                for index, field in enumerate(fields)
            }
        )

    return model


def truth_paths_from(model, galaxy: str) -> Dict[str, Tuple[str, ...]]:
    """
    The model path of each recoverable ``truth`` key (``TRUTH_PARAMETER_NAMES``) that is
    a free parameter of ``model`` under ``galaxies.<galaxy>``.

    A tied parameter (the shared ``mass_to_light_ratio``) has one path per member; the
    first is returned, and any of them indexes the same sample column. A truth key whose
    parameter is fixed (e.g. ``scale_radius_arcsec`` with ``free_scale_radius=False``,
    or ``inner_slope`` of an ``NFWSph``) is left out.

    Raises
    ------
    ValueError
        If a parameter name is held by more than one free parameter of the galaxy.
    """
    paths = {}
    for key, parameter in TRUTH_PARAMETER_NAMES.items():
        matches = [
            prior_paths[0]
            for prior_paths in model.all_paths
            if prior_paths[0][:2] == ("galaxies", galaxy)
            and prior_paths[0][-1] == parameter
        ]
        if len(matches) > 1:
            raise ValueError(
                f"Several free parameters of galaxies.{galaxy} are named {parameter!r} "
                f"({matches}); pass the paths explicitly."
            )
        if matches:
            paths[key] = tuple(matches[0])
    return paths
