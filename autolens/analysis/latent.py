"""
Latent variables for PyAutoLens analyses.

All latents take a generic ``fit`` argument and access ``fit.tracer``,
``fit.galaxy_image_dict`` and ``fit.dataset.grids.lp`` — APIs that exist
identically on both ``FitImaging`` (``autolens/imaging/fit_imaging.py:176``)
and ``FitInterferometer`` (``autolens/interferometer/fit_interferometer.py:176``).
The registry is dataset-agnostic; a future ``AnalysisInterferometer``
wiring can reuse it without code duplication.

User-level enable/disable: each key in ``autolens/config/latent.yaml`` maps
to a bool. All five default ``false`` because ``compute_latent_samples``
runs on every fit (``latent_after_fit: true`` in autofit's default
``output.yaml``) and the latents that require ``magzero`` would otherwise
crash existing fits where ``magzero`` is not passed.
"""
import logging
from typing import Callable, Dict, List, Optional

import numpy as np

from autoconf import conf
from autogalaxy.imaging.model.latent import (
    ab_mag_via_flux_from,
    flux_mujy_via_ab_mag_from,
)

logger = logging.getLogger(__name__)


def _require_magzero(magzero, name):
    if magzero is None:
        raise ValueError(
            f"magzero must be passed to the Analysis via kwargs to compute "
            f"the '{name}' latent. Disable it in config/latent.yaml or "
            f"pass magzero=<value>."
        )


def total_lens_flux_mujy(fit, magzero, xp=np):
    """
    Total integrated flux of the lens galaxy (``fit.tracer.galaxies[0]``),
    magzero-converted to microjanskies.

    Returns NaN when galaxy 0 has no light profile (raises ``KeyError`` /
    ``AttributeError`` inside ``fit.galaxy_image_dict``).
    """
    _require_magzero(magzero, "total_lens_flux_mujy")
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
    (``fit.galaxy_image_dict[fit.tracer.galaxies[-1]]``).
    """
    _require_magzero(magzero, "total_lensed_source_flux_mujy")
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
    Source-plane intrinsic flux of the source galaxy, via
    ``fit.tracer.galaxies[-1].image_2d_from(grid=fit.dataset.grids.lp)``.
    """
    _require_magzero(magzero, "total_source_flux_mujy")
    try:
        source_image = fit.tracer.galaxies[-1].image_2d_from(
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
        if xp is not np:
            import jax.numpy as jnp
            init_guess = jnp.array(
                [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
            )
            return lens_calc.einstein_radius_jit_from(init_guess=init_guess)
        return lens_calc.einstein_radius_from(grid=fit.dataset.grids.lp)
    except (ValueError, AttributeError):
        return xp.nan


LATENT_FUNCTIONS: Dict[str, Callable] = {
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
