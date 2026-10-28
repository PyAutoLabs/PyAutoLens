"""
``ReplaceTotalMassWithStellarPlusDark``: replace a galaxy's total-mass profile(s) with
stellar mass that follows its MGE light plus a dark-matter halo, normalised so their
projected mass inside the parent's Einstein radius equals that of the replaced mass.

The normalisation convention is PyAutoGalaxy's ``docs/api/mass_budget.rst`` (rulings
R3 and R4, 2026-10-28); the solve is ``ag.MassBudget``.
"""

import copy
from typing import Dict, Optional, Union

import numpy as np

import autoarray as aa
import autogalaxy as ag
from autogalaxy.profiles.light.abstract import LightProfile
from autogalaxy.profiles.mass.abstract.abstract import MassProfile
from autonerves.class_path import get_class_path

from autolens.twins._galaxies import galaxy_dict_from, galaxy_name_from
from autolens.twins.aperture import (
    check_aperture_in_grid,
    einstein_radius_aperture_from,
    einstein_radius_from,
)
from autolens.twins.record import TransformationRecord

NAME = "ReplaceTotalMassWithStellarPlusDark"

CONVENTION_PAGE = "PyAutoGalaxy docs/api/mass_budget.rst (rulings R3, R4)"

CONVENTION_TEXT = (
    "Aperture (R3): a circle of radius theta_E,parent, the Einstein radius of the whole "
    "parent lens model (tracer-level, from the area of its tangential critical curve via "
    "LensCalc), centred on the mass centre of the replaced galaxy. Reference mass (R3): "
    "M_ref is the projected mass of the replaced mass component(s) inside the aperture, "
    "by grid integration of their convergence (mass_angular_within_circle_via_grid_from); "
    "external shear and all other galaxies are left untouched and do not enter M_ref. "
    "Stellar component: a Basis of lmp.Gaussian copied from the galaxy's light basis "
    "(centre, ell_comps, sigma and solved intensity of every Gaussian) with one shared "
    "mass-to-light ratio Upsilon = f_* M_ref / M_light(theta_E), M_light being the same "
    "grid integral of the basis at Upsilon = 1. Dark component (R4): an NFWSph (or a "
    "gNFWSph with a fixed inner_slope) centred on the light with a fixed scale radius, "
    "given in kpc and converted to arc-seconds with the setup cosmology at the galaxy "
    "redshift; kappa_s = (1 - f_*) M_ref / M(kappa_s = 1), exact because the convergence "
    "is linear in kappa_s. All masses are angular (convergence units, arcsec^2). "
    f"Source: {CONVENTION_PAGE}."
)

DEFAULT_APERTURE_SETTINGS = {
    "sub_size_interior": 8,
    "sub_size_edge": 16,
    "sub_size_centre": 64,
    "centre_refine_pixels": 2.0,
}


def _plain(value):
    """
    JSON-able plain data for a record: NumPy scalars become floats / ints / bools and
    tuples become lists. ``al.from_dict`` drops ``None`` values inside dictionaries, so
    a ``None`` here is a bug and raises.
    """
    if value is None:
        raise ValueError("A TransformationRecord value must not be None.")
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, str):
        return value
    raise TypeError(
        f"A TransformationRecord value must be plain data, got {type(value).__name__}."
    )


def _has_mass(profile) -> bool:
    if isinstance(profile, ag.lp_basis.Basis):
        return len(profile.mass_profile_list) > 0
    return isinstance(profile, MassProfile)


def _is_external_shear(profile) -> bool:
    return isinstance(profile, ag.mp.ExternalShear)


def _grid_geometry(grid: aa.Grid2D) -> Dict:
    return {
        "shape_native": list(grid.mask.shape_native),
        "pixel_scales": [float(scale) for scale in grid.mask.pixel_scales],
        "origin": [float(value) for value in grid.mask.origin],
    }


class ReplaceTotalMassWithStellarPlusDark:
    def __init__(
        self,
        galaxy: str,
        light: str,
        *,
        scale_radius_kpc: float,
        f_dark: float = 0.5,
        budget: Optional[Union[ag.MassBudget, Dict[str, Dict[str, float]]]] = None,
        dark_cls=None,
        keep_shear: bool = True,
        aperture_settings: Optional[Dict] = None,
        einstein_radius_pixel_scale: float = 0.05,
        grid: Optional[aa.Grid2D] = None,
        dark_name: str = "dark",
        **dark_fixed,
    ):
        """
        Replace the total-mass profile(s) of one galaxy by stellar mass following its MGE
        light plus a dark-matter halo, keeping the replaced galaxy's projected mass inside
        theta_E,parent (the one deliberate preservation, recorded in ``preserved``).

        ``apply(twin)`` returns ``(tracer, setup, record)`` and mutates nothing:

        - every ``MassProfile`` attribute of the galaxy that is not light and not an
          ``ExternalShear`` (e.g. its ``Isothermal``) is removed;
        - the light basis attribute ``light`` is replaced by the stellar ``Basis`` of
          ``lmp.Gaussian`` that ``ag.MassBudget`` solves. Its members copy the centre,
          ``ell_comps``, ``sigma`` and intensity of the light Gaussians, so the galaxy's
          light is unchanged and its stellar mass is ``Upsilon`` times that light. (Adding
          the stellar basis beside the light basis would count the light twice.)
        - the dark profile is added as the attribute ``dark_name``, centred on the light;
        - external shear (galaxy attribute or tracer field), the source and all other
          galaxies are unchanged (``keep_shear=False`` removes the replaced galaxy's
          ``ExternalShear`` attributes and the tracer's ``ExternalShear`` fields).

        The aperture is ``einstein_radius_aperture_from(tracer, setup.grid, centre)``: the
        whole parent tracer's Einstein radius, centred on the replaced mass centre.

        Parameters
        ----------
        galaxy
            The galaxy to transform: its name in the tracer (the key of the fit's
            ``af.Collection``, e.g. ``"lens"``, or ``"galaxy_<index>"`` for a tracer built
            from a list), or a role from the reference's provenance ``roles``.
        light
            The name of the galaxy's MGE light attribute: a ``Basis`` of solved-intensity
            ``lp.Gaussian`` profiles (linear, operated or non-Gaussian members raise
            ``ProfileException``: they have no ``lmp.Gaussian`` counterpart).
        scale_radius_kpc
            The fixed dark-profile scale radius, in kpc (ruling R4). It is converted to
            arc-seconds with ``cosmology.arcsec_per_kpc_from(redshift=galaxy.redshift)``,
            the cosmology being the setup's, or the tracer's when the setup has none.
        f_dark
            The dark fraction of the replaced mass inside theta_E,parent (stellar
            fraction ``1 - f_dark``). Ignored when ``budget`` is given.
        budget
            Alternatively an ``ag.MassBudget`` or its ``fractions`` dictionary,
            ``{galaxy: {"stellar": f_*, "dark": f_dark}}``; only its fractions are used
            (the aperture is always the Einstein-radius aperture). A budget naming more
            than one galaxy raises ``NotImplementedError`` (ruling R7).
        dark_cls
            The dark profile class; ``al.mp.NFWSph`` by default, or e.g. ``al.mp.gNFWSph``
            with ``inner_slope`` passed in ``dark_fixed``.
        keep_shear
            Keep the external shear (default). It never enters M_ref.
        aperture_settings
            ``sub_size_interior``, ``sub_size_edge``, ``sub_size_centre`` and
            ``centre_refine_pixels`` of the aperture integrator (defaults: those of
            ``ag.CircularAperture``). A ``gNFW`` convergence does one ``quad`` per point,
            so pass coarse sub-sizes for it (e.g. 2, 4, 4).
        einstein_radius_pixel_scale
            The resolution of the grid on which the critical curve, and so theta_E, is
            traced.
        grid
            The uniform grid on which aperture masses are integrated; defaults to the
            setup mask's full (unmasked) geometry. The aperture must lie within it.
        dark_name
            The galaxy attribute the dark profile is stored as.
        dark_fixed
            Further fixed constructor arguments of ``dark_cls``, e.g. ``inner_slope``.
        """
        if budget is not None:
            fractions = (
                budget.fractions if isinstance(budget, ag.MassBudget) else dict(budget)
            )
            # Validates the fractions; raises NotImplementedError for several galaxies.
            ag.MassBudget(aperture=None, fractions=fractions)
            self.budget_galaxy = next(iter(fractions))
            self.fractions = {
                "stellar": float(fractions[self.budget_galaxy]["stellar"]),
                "dark": float(fractions[self.budget_galaxy]["dark"]),
            }
        else:
            if not 0.0 <= f_dark <= 1.0:
                raise ValueError(f"f_dark must lie in [0, 1], got {f_dark}.")
            self.budget_galaxy = None
            self.fractions = {"stellar": 1.0 - float(f_dark), "dark": float(f_dark)}

        if not scale_radius_kpc > 0.0:
            raise ValueError(
                f"scale_radius_kpc must be positive, got {scale_radius_kpc}."
            )

        self.galaxy = galaxy
        self.light = light
        self.scale_radius_kpc = float(scale_radius_kpc)
        self.dark_cls = ag.mp.NFWSph if dark_cls is None else dark_cls
        self.keep_shear = bool(keep_shear)
        self.aperture_settings = {
            **DEFAULT_APERTURE_SETTINGS,
            **dict(aperture_settings or {}),
        }
        self.einstein_radius_pixel_scale = float(einstein_radius_pixel_scale)
        self.grid = grid
        self.dark_name = dark_name
        self.dark_fixed = dict(dark_fixed)

    @property
    def f_dark(self) -> float:
        return self.fractions["dark"]

    def requested(self) -> Dict:
        """The requested arguments, as plain data (``grid`` as its geometry)."""
        return _plain(
            {
                "galaxy": self.galaxy,
                "light": self.light,
                "scale_radius_kpc": self.scale_radius_kpc,
                "f_dark": self.f_dark,
                "fractions": self.fractions,
                "dark_cls": get_class_path(self.dark_cls),
                "dark_fixed": self.dark_fixed,
                "keep_shear": self.keep_shear,
                "aperture_settings": self.aperture_settings,
                "einstein_radius_pixel_scale": self.einstein_radius_pixel_scale,
                "grid": (
                    "setup mask geometry"
                    if self.grid is None
                    else _grid_geometry(self.grid)
                ),
                "dark_name": self.dark_name,
            }
        )

    def apply(self, twin):
        tracer = twin.tracer
        setup = twin.setup

        name = galaxy_name_from(twin, self.galaxy)
        if self.budget_galaxy is not None and (
            galaxy_name_from(twin, self.budget_galaxy) != name
        ):
            raise ValueError(
                f"The budget names galaxy {self.budget_galaxy!r} but the transformation "
                f"replaces {name!r}."
            )
        galaxy = galaxy_dict_from(tracer)[name]
        path = f"galaxies.{name}"

        light_basis = getattr(galaxy, self.light, None)
        if not isinstance(light_basis, ag.lp_basis.Basis):
            raise ValueError(
                f"{path}.{self.light} must be an MGE light Basis, got "
                f"{type(light_basis).__name__}."
            )
        # Raises ProfileException for members with no lmp.Gaussian counterpart, before
        # any expensive step.
        light_basis.mass_basis_from(mass_to_light_ratio=1.0)

        profile_dict = galaxy.profile_dict
        stellar_already = [
            key
            for key, profile in profile_dict.items()
            if isinstance(profile, LightProfile) and _has_mass(profile)
        ]
        if stellar_already:
            raise ValueError(
                f"{path} already has light-and-mass components {stellar_already}; "
                f"{NAME} replaces a total-mass profile only."
            )
        replaced = {
            key: profile
            for key, profile in profile_dict.items()
            if isinstance(profile, MassProfile)
            and not isinstance(profile, LightProfile)
            and not _is_external_shear(profile)
        }
        if not replaced:
            raise ValueError(f"{path} has no total-mass profile to replace.")
        if self.dark_name in profile_dict and self.dark_name not in replaced:
            raise ValueError(
                f"{path} already has an attribute {self.dark_name!r}; pass another "
                f"dark_name."
            )
        shear_keys = [
            key for key, profile in profile_dict.items() if _is_external_shear(profile)
        ]
        shear_field_indexes = [
            index
            for index, field in enumerate(getattr(tracer, "fields", []))
            if _is_external_shear(field)
        ]

        mass_centre = tuple(
            float(value) for value in next(iter(replaced.values())).centre
        )
        light_centre = tuple(
            float(value) for value in light_basis.profile_list[0].centre
        )

        aperture = einstein_radius_aperture_from(
            tracer=tracer,
            grid=setup.grid,
            centre=mass_centre,
            pixel_scale=self.einstein_radius_pixel_scale,
            **self.aperture_settings,
        )
        einstein_radius_parent = aperture.radius

        grid = self.grid
        if grid is None:
            grid = aa.Grid2D.uniform(
                shape_native=setup.mask.shape_native,
                pixel_scales=setup.mask.pixel_scales,
                origin=setup.mask.origin,
                over_sample_size=1,
            )
        check_aperture_in_grid(aperture=aperture, grid=grid)

        cosmology = setup.cosmology if setup.cosmology is not None else tracer.cosmology
        arcsec_per_kpc = float(cosmology.arcsec_per_kpc_from(redshift=galaxy.redshift))
        scale_radius_arcsec = self.scale_radius_kpc * arcsec_per_kpc

        result = ag.MassBudget(
            aperture=aperture, fractions={name: self.fractions}
        ).solve(
            reference_mass=ag.Galaxy(redshift=galaxy.redshift, **replaced),
            light_basis=light_basis,
            dark_cls=self.dark_cls,
            scale_radius=scale_radius_arcsec,
            centre=light_centre,
            grid=grid,
            **self.dark_fixed,
        )

        new_tracer = copy.deepcopy(tracer)
        new_galaxy = galaxy_dict_from(new_tracer)[name]

        changed = []
        removed = {}
        added = {}
        for key, profile in replaced.items():
            delattr(new_galaxy, key)
            changed.append(f"{path}.{key}")
            removed[f"{path}.{key}"] = get_class_path(type(profile))
        if result.stellar is not None:
            setattr(new_galaxy, self.light, result.stellar)
            changed.append(f"{path}.{self.light}")
            added[f"{path}.{self.light}"] = "Basis of lmp.Gaussian"
        if result.dark is not None:
            setattr(new_galaxy, self.dark_name, result.dark)
            if f"{path}.{self.dark_name}" not in changed:
                changed.append(f"{path}.{self.dark_name}")
            added[f"{path}.{self.dark_name}"] = get_class_path(type(result.dark))

        side_effects = []
        shear_paths = [f"{path}.{key}" for key in shear_keys] + [
            f"fields.{index}" for index in shear_field_indexes
        ]
        if not self.keep_shear:
            for key in shear_keys:
                delattr(new_galaxy, key)
            new_tracer.fields = [
                field
                for index, field in enumerate(new_tracer.fields)
                if index not in shear_field_indexes
            ]
            changed.extend(shear_paths)
            for shear_path in shear_paths:
                removed[shear_path] = get_class_path(ag.mp.ExternalShear)
            if shear_paths:
                side_effects.append(f"external shear removed: {shear_paths}")
        elif shear_paths:
            side_effects.append(
                f"external shear kept unchanged and excluded from M_ref: {shear_paths}"
            )

        try:
            einstein_radius_twin = einstein_radius_from(
                tracer=new_tracer,
                grid=setup.grid,
                pixel_scale=self.einstein_radius_pixel_scale,
            )
            side_effects.append(
                f"Einstein radius of the twin {einstein_radius_twin:.6f} arcsec (parent "
                f"{einstein_radius_parent:.6f} arcsec), recomputed on the same grid; "
                f"reported, not asserted"
            )
        except ValueError:
            einstein_radius_twin = None
            side_effects.append(
                "the twin has no tangential critical curve inside the setup grid; its "
                "Einstein radius is undefined (reported, not asserted)"
            )
        if result.dark is not None:
            side_effects.append(
                f"dark halo centred on the light centre {list(light_centre)}; the "
                f"replaced mass centre was {list(mass_centre)}"
            )
        if result.stellar is not None:
            side_effects.append(
                f"light basis {path}.{self.light} re-expressed as lmp.Gaussian with "
                f"identical centre, ell_comps, intensity and sigma, so the light is "
                f"unchanged"
            )
        side_effects.append("the source and all other galaxies are unchanged")

        derived = dict(result.derived)
        derived["fixed"] = dict(derived.get("fixed") or {})
        dark_profile = result.dark
        inner_slope = self.dark_fixed.get(
            "inner_slope", getattr(dark_profile, "inner_slope", 1.0)
        )

        truth = {
            "galaxy": name,
            "redshift": galaxy.redshift,
            "mass_to_light_ratio": derived["mass_to_light_ratio"],
            "kappa_s": derived["kappa_s"],
            "scale_radius_arcsec": scale_radius_arcsec,
            "scale_radius_kpc": self.scale_radius_kpc,
            "inner_slope": inner_slope,
            "dark_cls": get_class_path(self.dark_cls),
            "f_dark": derived["f_dark"],
            "f_stellar": self.fractions["stellar"],
            "einstein_radius_parent": einstein_radius_parent,
            "mass_reference": derived["mass_reference"],
            "dark_centre": list(light_centre),
        }

        procedure = {
            "text": CONVENTION_TEXT,
            "convention": CONVENTION_PAGE,
            "galaxy": name,
            "derived": derived,
            "einstein_radius_parent": einstein_radius_parent,
            "einstein_radius_grid": "setup.grid",
            "einstein_radius_pixel_scale": self.einstein_radius_pixel_scale,
            "aperture": {
                "radius": aperture.radius,
                "centre": list(aperture.centre),
                **self.aperture_settings,
            },
            "aperture_grid": _grid_geometry(grid),
            "scale_radius_kpc": self.scale_radius_kpc,
            "scale_radius_arcsec": scale_radius_arcsec,
            "arcsec_per_kpc": arcsec_per_kpc,
            "redshift": galaxy.redshift,
            "cosmology": get_class_path(type(cosmology)),
            "cosmology_source": "setup" if setup.cosmology is not None else "tracer",
            "removed": removed,
            "added": added,
            "truth": truth,
        }
        if einstein_radius_twin is not None:
            procedure["einstein_radius_twin"] = einstein_radius_twin

        record = TransformationRecord(
            name=NAME,
            requested=self.requested(),
            procedure=_plain(procedure),
            changed=changed,
            side_effects=side_effects,
            preserved=[f"projected mass of {name} inside theta_E,parent"],
        )

        return new_tracer, setup, record
