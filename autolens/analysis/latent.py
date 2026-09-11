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

import autoarray as aa
import autofit as af
from autonerves import conf
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

# Set to True the first time ``_pixelized_source_flux`` meets a mapper whose
# mesh geometry exposes no ``areas_for_magnification``. Deduplicates the
# warning across the many fit evaluations a single search performs.
_MESH_AREAS_WARNED: bool = False

# Set to True the first time ``_pixelized_source_flux`` cannot resolve the
# data's pixel area, and so cannot express the mesh-integrated flux in this
# module's per-data-pixel convention. Deduplicates that warning across the
# many fit evaluations a single search performs.
_PIXEL_AREA_WARNED: bool = False


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


def _data_pixel_area(fit) -> Optional[float]:
    """
    The data's pixel area ``A_pix`` in arcsec², or ``None`` if it cannot be
    resolved from the fit.

    Read from ``fit.dataset.grids.lp`` — the *same* grid the light-profile
    flux term of :func:`total_source_flux` is evaluated on — so both terms of
    that sum share one convention. ``Grid2D.pixel_area`` is the product of the
    grid's ``pixel_scales`` and is a plain Python float, so dividing by it
    introduces no array op and is trace-safe inside ``jax.jit``. A grid type
    that publishes no ``pixel_area`` falls back to multiplying out
    ``pixel_scales``, on the grid and then on the dataset itself.
    """
    dataset = getattr(fit, "dataset", None)
    grid = getattr(getattr(dataset, "grids", None), "lp", None)

    pixel_area = getattr(grid, "pixel_area", None)
    if pixel_area is not None:
        return float(pixel_area)

    for obj in (grid, dataset):
        pixel_scales = getattr(obj, "pixel_scales", None)
        if pixel_scales is not None:
            return float(pixel_scales[0]) * float(pixel_scales[1])

    return None


def _pixelized_source_flux(fit, xp=np):
    """
    Source-plane integrated flux of the source galaxy's *pixelized*
    (inversion) component, in raw image units **per data pixel**.

    ``Galaxy.image_2d_from`` returns zeros for a galaxy whose only light
    model is a ``Pixelization`` — the reconstruction lives on the inversion's
    mesh, not on a light profile. This helper integrates that reconstruction
    over the mesh and re-expresses it in this module's per-data-pixel flux
    convention: ``Σ_i s_i A_i / A_pix``, where ``s_i`` are the reconstructed
    source-pixel values (``inversion.reconstruction_dict[mapper]``), ``A_i``
    the exact per-pixel areas the mesh geometry publishes as
    ``areas_for_magnification`` (barycentric dual areas for a Delaunay mesh —
    PyAutoArray#524 — transformed cell areas for a rectangular mesh) and
    ``A_pix`` the data pixel area (:func:`_data_pixel_area`).

    **Why the division by ``A_pix``.** Every other flux latent here is a sum
    over *data pixels* — ``Σ galaxy_image_dict[galaxy]``,
    ``Σ galaxy.image_2d_from(grid)`` — i.e. ``∫ I d²θ / A_pix``. The
    reconstruction ``s_i`` is in those same per-data-pixel units (the mapping
    matrix maps ``s`` onto image-pixel values), so ``Σ_i s_i A_i`` is the
    plain surface integral ``∫ S d²β`` and must be divided by ``A_pix`` to
    join the convention. Only then is :func:`magnification` —
    ``Σ image / (Σ s A / A_pix)`` — the physical ``∫ I d²θ / ∫ S d²β``.
    Undivided it scales with the data's pixel scale: on a 0.1"/pixel dataset
    the pixelized magnification came out exactly 100× too large. The 1"
    fixtures have ``A_pix = 1``, which is why this was invisible in tests.

    The mapper → galaxy association is read from
    ``inversion.linear_obj_galaxy_dict`` — the dict the inversion is given at
    construction (``autolens/lens/to_inversion.py``), whose galaxy values are
    the tracer's own galaxy objects. ``fit.tracer_to_inversion`` must *not* be
    used here: on ``FitImaging`` it is an uncached property that rebuilds a
    ``TracerToInversion`` per call, whose mappers are new objects and
    therefore not keys of ``reconstruction_dict``.

    Returns ``0.0`` when the fit has no inversion or no mapper belongs to the
    source galaxy (the light-profile path already covers those fits), and
    ``NaN`` — with a one-time-per-process warning — when a mapper's mesh
    geometry publishes no exact quadrature areas, or when the data pixel area
    cannot be resolved (an unscaled number would silently be wrong by the
    pixel area).

    All branching is on Python structure (``None`` checks, ``isinstance``,
    dict membership), never on array values, so the function is safe inside
    the per-sample ``jax.jit`` the latent batch path uses.
    """
    global _MESH_AREAS_WARNED
    global _PIXEL_AREA_WARNED

    inversion = getattr(fit, "inversion", None)
    if inversion is None:
        return 0.0

    linear_obj_galaxy_dict = getattr(inversion, "linear_obj_galaxy_dict", None)
    if not linear_obj_galaxy_dict:
        return 0.0

    try:
        source_galaxy = fit.tracer.galaxies[-1]
    except (AttributeError, IndexError):
        return 0.0

    total = 0.0
    found_mapper = False

    for linear_obj, galaxy in linear_obj_galaxy_dict.items():
        if galaxy is not source_galaxy:
            continue
        if not isinstance(linear_obj, aa.Mapper):
            continue

        areas = getattr(
            getattr(linear_obj, "mesh_geometry", None),
            "areas_for_magnification",
            None,
        )
        if areas is None:
            if not _MESH_AREAS_WARNED:
                logger.warning(
                    "The mesh geometry '%s' publishes no "
                    "'areas_for_magnification', so the source-plane flux of "
                    "the pixelized source cannot be integrated; "
                    "'total_source_flux' and 'magnification' will be NaN.",
                    type(getattr(linear_obj, "mesh_geometry", None)).__name__,
                )
                _MESH_AREAS_WARNED = True
            return xp.nan

        found_mapper = True
        total = total + xp.sum(
            xp.asarray(inversion.reconstruction_dict[linear_obj]) * xp.asarray(areas)
        )

    if not found_mapper:
        return 0.0

    pixel_area = _data_pixel_area(fit)
    if pixel_area is None:
        if not _PIXEL_AREA_WARNED:
            logger.warning(
                "The data pixel area could not be resolved from the fit "
                "(no 'fit.dataset.grids.lp.pixel_area' and no 'pixel_scales'), "
                "so the pixelized source flux cannot be expressed per data "
                "pixel; 'total_source_flux' and 'magnification' will be NaN."
            )
            _PIXEL_AREA_WARNED = True
        return xp.nan

    return total / pixel_area


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

    Two contributions are summed, so that a source modelled with light
    profiles, a pixelization, or both is measured correctly:

    - **Light profiles** — read from
      ``fit.tracer_linear_light_profiles_to_light_profiles`` so that linear
      light profiles (whose ``intensity`` is solved by the inversion)
      contribute the correct image.
    - **Pixelization** — the inversion's reconstruction integrated with its
      exact per-pixel ``areas_for_magnification`` (barycentric dual areas for
      a Delaunay mesh, PyAutoArray#524; transformed cell areas for a
      rectangular mesh) and expressed per data pixel (divided by the data
      pixel area), which is the convention the light-profile term above is
      already in; see :func:`_pixelized_source_flux`. Without this term a
      purely pixelized source has ``image_2d_from`` zeros and a zero source
      flux, making :func:`magnification` infinite (PyAutoLens#726).

    ``magzero`` is accepted but ignored.
    """
    try:
        tracer = fit.tracer_linear_light_profiles_to_light_profiles
        source_image = tracer.galaxies[-1].image_2d_from(
            grid=fit.dataset.grids.lp, xp=xp
        )
    except (AttributeError, IndexError):
        return xp.nan
    return xp.sum(source_image.array) + _pixelized_source_flux(fit=fit, xp=xp)


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

    Same definition as :func:`total_source_flux` — light-profile flux plus
    the pixelized (inversion) flux — magzero-converted.

    The light-profile term reads from
    ``fit.tracer_linear_light_profiles_to_light_profiles`` rather than
    ``fit.tracer`` so that linear light profiles (whose ``intensity`` is
    solved by the inversion at fit time) contribute the correct image. For
    non-linear fits this property is a no-op pass-through (returns
    ``fit.tracer``), so the numpy-only and JAX paths both work uniformly.
    The pixelized term integrates the inversion's reconstruction with its
    ``areas_for_magnification`` (barycentric dual areas for a Delaunay mesh —
    PyAutoArray#524 — transformed cell areas for a rectangular mesh) and
    expresses it per data pixel (divided by the data pixel area), matching the
    light-profile term's convention; see :func:`_pixelized_source_flux`.

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
    total_flux = xp.sum(source_image.array) + _pixelized_source_flux(fit=fit, xp=xp)
    return flux_mujy_via_ab_mag_from(
        ab_mag=ab_mag_via_flux_from(flux=total_flux, magzero=magzero, xp=xp),
        xp=xp,
    )


def magnification(fit, magzero, xp=np):
    """
    Ratio of image-plane to source-plane source flux — the integrated
    magnification implied by the lens model and the source's light.

    The denominator is :func:`total_source_flux_mujy`: light-profile flux
    plus, for a pixelized source, the inversion's reconstruction integrated
    with its ``areas_for_magnification`` (barycentric dual areas for a
    Delaunay mesh — PyAutoArray#524 — transformed cell areas for a rectangular
    mesh) and expressed per data pixel (divided by the data pixel area). Both
    numerator and denominator are then per-data-pixel sums, so the ratio is
    the physical ``∫ I d²θ / ∫ S d²β`` and does not depend on the data's pixel
    scale. A source whose only light model is a ``Pixelization`` therefore has
    a finite magnification rather than an infinite one (PyAutoLens#726).

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

    JAX path: ``LensCalc.einstein_radius_jit_from()`` with no seed — the
    JIT-compatible variant, which since PyAutoGalaxy#614 finds its own Newton
    seed inside the trace (argmin of the tangential eigen value on a coarse
    grid, ±3 arcsec by default) instead of needing a caller-supplied
    ``init_guess``. The earlier fixed 4-seed fan at ±1 arcsec from the origin
    could not converge for off-centre lenses or Einstein radii outside its
    basin. ``ZeroSolver`` uses ``lax.cond`` / ``lax.while_loop`` early
    termination that is incompatible with ``jax.vmap`` but fine under
    ``jax.jit``.

    NumPy path: ``LensCalc.einstein_radius_from(grid=fit.dataset.grids.lp)``.
    """
    from autogalaxy.operate.lens_calc import LensCalc

    try:
        lens_calc = LensCalc.from_mass_obj(fit.tracer)
        if xp is not np and _jax_zero_contour_available():
            return lens_calc.einstein_radius_jit_from()
        return lens_calc.einstein_radius_from(grid=fit.dataset.grids.lp)
    except (ValueError, AttributeError):
        return xp.nan


def latent_instance_from(model, parameters, xp=np):
    """
    Build the model instance a latent function is evaluated on, on either
    backend, the way ``autofit.non_linear.fitness.Fitness`` builds the one the
    likelihood is evaluated on.

    On the NumPy path the model's assertions are checked as usual: a sample
    that violates one raises ``af.exc.FitException``, which the latent engine
    turns into a NaN row and drops. Under JAX that check applies a Python
    ``not`` to a traced boolean and raises ``TracerBoolConversionError``
    inside the per-sample ``jax.jit`` the latent batch path uses. The engine
    swallowed that raise into a NaN row for **every** sample, so a model with
    any assertion (e.g. the ordered two-basis MGE lens light of the Euclid
    ``vis_lp`` stage, PyAutoFit#1583) wrote no latent output at all. Under JAX
    the assertions are therefore skipped here, as ``Fitness`` skips them
    (every sample the search accepted already satisfied them), and ``xp`` is
    threaded through so the instance is built on the traced backend.

    Subclasses of :class:`LatentLens` that build their own instance (rather
    than composing ``LatentLens.variables``) must go through this helper.
    """
    if xp is np:
        return model.instance_from_vector(vector=parameters)
    return model.instance_from_vector(
        vector=parameters, ignore_assertions=True, xp=xp
    )


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


class LatentLens(af.Latent):
    """
    Latent-variable catalogue for strong-lens fits, declared on
    ``AnalysisImaging`` as ``Latent = LatentLens`` (mirrors ``Visualizer`` /
    ``Result``).

    :meth:`keys` returns the config-enabled latent names; :meth:`variables`
    dispatches the :data:`LATENT_FUNCTIONS` registry on a per-sample fit.
    Subclass to add project-specific latents (e.g. the Euclid pipeline's
    aperture fluxes), composing the library values via
    ``LatentLens.variables(analysis, parameters, model)``. A subclass that
    builds the instance itself must use :func:`latent_instance_from`, which
    skips the model's assertions under JAX — checking them inside the
    per-sample ``jax.jit`` raises, and the engine turns that raise into a NaN
    row for every sample (no latent output at all).
    """

    # The lensing Einstein-radius latent routes through ``ZeroSolver``
    # (uses ``lax.cond`` / ``lax.while_loop``, vmap-incompatible), so latents
    # use the per-sample jit batch path.
    BATCH_MODE = "jit"

    @staticmethod
    def keys(analysis) -> List[str]:
        return latent_keys_enabled()

    @staticmethod
    def variables(analysis, parameters, model):
        keys = latent_keys_enabled()
        if not keys:
            raise NotImplementedError
        xp = analysis._xp
        instance = latent_instance_from(model=model, parameters=parameters, xp=xp)
        fit = analysis.fit_from(instance=instance)
        magzero = analysis.kwargs.get("magzero", None)
        context = {"fit": fit, "magzero": magzero, "xp": xp}
        return tuple(LATENT_FUNCTIONS[k](**context) for k in keys)
