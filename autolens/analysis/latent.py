"""
Latent variables for PyAutoLens analyses.

All latents take a generic ``fit`` argument and access ``fit.tracer``,
``fit.galaxy_image_dict`` and ``fit.dataset.grids.lp`` — APIs that exist
identically on both ``FitImaging`` (``autolens/imaging/fit_imaging.py:176``)
and ``FitInterferometer`` (``autolens/interferometer/fit_interferometer.py:176``).
The registry is dataset-agnostic; a future ``AnalysisInterferometer``
wiring can reuse it without code duplication.

User-level enable/disable: each key in ``autolens/config/latent.yaml`` maps
to a bool. The raw-flux latents (``total_lens_flux``,
``total_lensed_source_flux``, ``total_source_flux``) require no instrument
inputs and default ``true``. The microjansky variants require ``magzero``
on the Analysis; they default ``false`` and return NaN with a single
warning per process if enabled without ``magzero`` (rather than raising,
which would kill the post-fit metric write of an otherwise-converged
search).
"""
import importlib
import logging
from typing import Callable, Dict, List, Optional

import numpy as np

from autoconf import conf
from autogalaxy.imaging.model.latent import (
    ab_mag_via_flux_from,
    flux_mujy_via_ab_mag_from,
)

logger = logging.getLogger(__name__)

# Latent names that have already emitted a "magzero missing" warning in this
# process. Used by ``_maybe_magzero_warn`` to deduplicate the message across
# the many fit evaluations a single search performs.
_MAGZERO_WARNED: set = set()

# Set to True the first time ``effective_einstein_radius`` falls back from
# the JAX path to the NumPy path because ``jax_zero_contour`` is missing.
# Deduplicates the fallback warning across the many fit evaluations a
# single search performs.
_JAX_ZERO_CONTOUR_FALLBACK_WARNED: bool = False


def _jax_zero_contour_available() -> bool:
    """
    Return True if ``jax_zero_contour`` can be imported; False otherwise.
    The first False return emits one warning per process noting that
    ``effective_einstein_radius`` will use the slower NumPy path.
    """
    global _JAX_ZERO_CONTOUR_FALLBACK_WARNED
    try:
        importlib.import_module("jax_zero_contour")
        return True
    except ModuleNotFoundError:
        if not _JAX_ZERO_CONTOUR_FALLBACK_WARNED:
            logger.warning(
                "jax_zero_contour not installed; effective_einstein_radius "
                "falling back to NumPy path (slower). "
                "pip install jax_zero_contour to enable the JIT path."
            )
            _JAX_ZERO_CONTOUR_FALLBACK_WARNED = True
        return False


def _maybe_magzero_warn(magzero, name) -> bool:
    """
    Return True when ``magzero`` is missing (and emit a one-time-per-process
    warning for ``name``); False otherwise.

    Callers that get True must early-return ``xp.nan`` — the µJy conversion
    is meaningless without a zero-point, but a search-killing raise here
    would discard otherwise-converged fits.
    """
    if magzero is None:
        if name not in _MAGZERO_WARNED:
            logger.warning(
                "magzero not set on Analysis; '%s' latent will be NaN. "
                "Pass magzero=<value> to AnalysisImaging to enable it, "
                "or disable in config/latent.yaml to silence this warning.",
                name,
            )
            _MAGZERO_WARNED.add(name)
        return True
    return False


def total_lens_flux(fit, magzero=None, xp=np):
    """
    Total integrated flux of the lens galaxy (``fit.tracer.galaxies[0]``),
    in the raw image units the fit was performed in.

    Requires no instrument inputs — ``magzero`` is accepted for uniform
    dispatcher context but ignored. See the workspace flux guide
    (``scripts/guides/units/flux.py``) for how to convert to microjanskies.

    Returns NaN when galaxy 0 has no light profile.
    """
    try:
        image = fit.galaxy_image_dict[fit.tracer.galaxies[0]]
    except (AttributeError, KeyError, IndexError):
        return xp.nan
    return xp.sum(image.array)


def total_lensed_source_flux(fit, magzero=None, xp=np):
    """
    Image-plane integrated flux of the source galaxy after lensing
    (``fit.galaxy_image_dict[fit.tracer.galaxies[-1]]``), in raw image
    units. ``magzero`` is accepted but ignored.
    """
    try:
        image = fit.galaxy_image_dict[fit.tracer.galaxies[-1]]
    except (AttributeError, KeyError, IndexError):
        return xp.nan
    return xp.sum(image.array)


def total_source_flux(fit, magzero=None, xp=np):
    """
    Source-plane intrinsic flux of the source galaxy, in raw image units.

    Reads from ``fit.tracer_linear_light_profiles_to_light_profiles`` so
    that linear light profiles (whose ``intensity`` is solved by the
    inversion) contribute the correct image — same tracer-conversion
    handling as :func:`total_source_flux_mujy`.

    ``magzero`` is accepted but ignored.
    """
    try:
        tracer = fit.tracer_linear_light_profiles_to_light_profiles
        source_image = tracer.galaxies[-1].image_2d_from(
            grid=fit.dataset.grids.lp, xp=xp
        )
    except (AttributeError, IndexError):
        return xp.nan
    return xp.sum(source_image.array)


def total_lens_flux_mujy(fit, magzero, xp=np):
    """
    Total integrated flux of the lens galaxy (``fit.tracer.galaxies[0]``),
    magzero-converted to microjanskies.

    Returns NaN — with a one-time-per-process warning — when ``magzero``
    is missing, rather than raising. The µJy conversion is meaningless
    without a zero-point, but a hard raise during post-fit latent
    computation would discard the result of an otherwise-converged
    multi-hour search.

    Also returns NaN when galaxy 0 has no light profile.
    """
    if _maybe_magzero_warn(magzero, "total_lens_flux_mujy"):
        return xp.nan
    try:
        image = fit.galaxy_image_dict[fit.tracer.galaxies[0]]
    except (AttributeError, KeyError, IndexError):
        return xp.nan
    total_flux = xp.sum(image.array)
    return flux_mujy_via_ab_mag_from(
        ab_mag=ab_mag_via_flux_from(flux=total_flux, magzero=magzero, xp=xp),
        xp=xp,
    )


def total_lensed_source_flux_mujy(fit, magzero, xp=np):
    """
    Image-plane integrated flux of the source galaxy after lensing
    (``fit.galaxy_image_dict[fit.tracer.galaxies[-1]]``), in microjanskies.

    Returns NaN + one warning when ``magzero`` is missing; see
    :func:`total_lens_flux_mujy` for the rationale.
    """
    if _maybe_magzero_warn(magzero, "total_lensed_source_flux_mujy"):
        return xp.nan
    try:
        image = fit.galaxy_image_dict[fit.tracer.galaxies[-1]]
    except (AttributeError, KeyError, IndexError):
        return xp.nan
    total_flux = xp.sum(image.array)
    return flux_mujy_via_ab_mag_from(
        ab_mag=ab_mag_via_flux_from(flux=total_flux, magzero=magzero, xp=xp),
        xp=xp,
    )


def total_source_flux_mujy(fit, magzero, xp=np):
    """
    Source-plane intrinsic flux of the source galaxy, in microjanskies.

    Reads from ``fit.tracer_linear_light_profiles_to_light_profiles`` rather
    than ``fit.tracer`` so that linear light profiles (whose ``intensity``
    is solved by the inversion at fit time) contribute the correct image.
    For non-linear fits this property is a no-op pass-through (returns
    ``fit.tracer``), so the numpy-only and JAX paths both work uniformly.

    Returns NaN + one warning when ``magzero`` is missing; see
    :func:`total_lens_flux_mujy` for the rationale.
    """
    if _maybe_magzero_warn(magzero, "total_source_flux_mujy"):
        return xp.nan
    try:
        tracer = fit.tracer_linear_light_profiles_to_light_profiles
        source_image = tracer.galaxies[-1].image_2d_from(
            grid=fit.dataset.grids.lp, xp=xp
        )
    except (AttributeError, IndexError):
        return xp.nan
    total_flux = xp.sum(source_image.array)
    return flux_mujy_via_ab_mag_from(
        ab_mag=ab_mag_via_flux_from(flux=total_flux, magzero=magzero, xp=xp),
        xp=xp,
    )


def magnification(fit, magzero, xp=np):
    """
    Ratio of image-plane to source-plane source flux — the integrated
    magnification implied by the lens model and source light profile.

    ``magzero`` is accepted but unused (the µJy conversions cancel in the
    ratio). It's still required in the signature so the dispatcher can
    pass a uniform context dict to every latent function.
    """
    lensed = total_lensed_source_flux_mujy(fit=fit, magzero=magzero, xp=xp)
    intrinsic = total_source_flux_mujy(fit=fit, magzero=magzero, xp=xp)
    return lensed / intrinsic


def effective_einstein_radius(fit, magzero, xp=np):
    """
    Effective Einstein radius via the tangential critical curve.

    JAX path: ``LensCalc.einstein_radius_jit_from(init_guess=fan)``, where
    ``fan`` is a fixed 4-seed fan at ±1 arcsec from the lens centre — the
    JIT-compatible variant required because ``ZeroSolver`` (line 1520 of
    ``autogalaxy/operate/lens_calc.py``) uses ``lax.cond`` /
    ``lax.while_loop`` early termination that is incompatible with
    ``jax.vmap`` but fine under ``jax.jit``.

    NumPy path: ``LensCalc.einstein_radius_from(grid=fit.dataset.grids.lp)``.
    """
    from autogalaxy.operate.lens_calc import LensCalc

    try:
        lens_calc = LensCalc.from_mass_obj(fit.tracer)
        if xp is not np and _jax_zero_contour_available():
            import jax.numpy as jnp
            init_guess = jnp.array(
                [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
            )
            return lens_calc.einstein_radius_jit_from(init_guess=init_guess)
        return lens_calc.einstein_radius_from(grid=fit.dataset.grids.lp)
    except (ValueError, AttributeError):
        return xp.nan


LATENT_FUNCTIONS: Dict[str, Callable] = {
    "total_lens_flux": total_lens_flux,
    "total_lensed_source_flux": total_lensed_source_flux,
    "total_source_flux": total_source_flux,
    "total_lens_flux_mujy": total_lens_flux_mujy,
    "total_lensed_source_flux_mujy": total_lensed_source_flux_mujy,
    "total_source_flux_mujy": total_source_flux_mujy,
    "magnification": magnification,
    "effective_einstein_radius": effective_einstein_radius,
}


def latent_keys_enabled(yaml_config: Optional[Dict[str, bool]] = None) -> List[str]:
    """
    Return the ordered list of enabled latent keys.

    Reads ``conf.instance["latent"]`` (a flat ``key: bool`` dict from
    ``autolens/config/latent.yaml``) unless ``yaml_config`` is passed
    explicitly — tests pass a literal dict to avoid pushing a temporary
    config directory.

    Unknown keys (present in the yaml but not in :data:`LATENT_FUNCTIONS`)
    are dropped with a logger warning rather than raising — yaml carries
    forward-compat entries for latents that ship in later releases.
    """
    if yaml_config is None:
        yaml_config = dict(conf.instance["latent"])

    enabled: List[str] = []
    for key, on in yaml_config.items():
        if not on:
            continue
        if key not in LATENT_FUNCTIONS:
            logger.warning(
                "latent.yaml lists '%s' but no such latent is registered; "
                "dropping. Known latents: %s",
                key,
                sorted(LATENT_FUNCTIONS),
            )
            continue
        enabled.append(key)
    return enabled
