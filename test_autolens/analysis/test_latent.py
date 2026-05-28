import importlib
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

import autofit as af
import autolens as al
from autolens.analysis import latent as _latent_module
from autolens.analysis.latent import (
    LATENT_FUNCTIONS,
    effective_einstein_radius,
    latent_keys_enabled,
    magnification,
    total_lens_flux,
    total_lens_flux_mujy,
    total_lensed_source_flux,
    total_lensed_source_flux_mujy,
    total_source_flux,
    total_source_flux_mujy,
)


# ---------------------------------------------------------------------------
# Latent function tests — SimpleNamespace/MagicMock-built fits
# ---------------------------------------------------------------------------

def _fit_with_galaxy_images(images_by_galaxy, **extras):
    """Build a SimpleNamespace fit whose galaxy_image_dict resolves the
    given (galaxy_index → image) mapping."""
    galaxies = [object() for _ in range(max(images_by_galaxy.keys()) + 1)]
    galaxy_image_dict = {
        galaxies[i]: SimpleNamespace(array=np.asarray(arr))
        for i, arr in images_by_galaxy.items()
    }
    fit = SimpleNamespace(
        tracer=SimpleNamespace(galaxies=galaxies),
        galaxy_image_dict=galaxy_image_dict,
        **extras,
    )
    return fit


def test_total_lens_flux_mujy_against_known_image():
    fit = _fit_with_galaxy_images({0: [1.0, 2.0, 3.0, 4.0]})
    value = total_lens_flux_mujy(fit=fit, magzero=25.0)

    # Sanity check against the AB-mag → muJy formula. flux=10, magzero=25.
    expected_ab_mag = -2.5 * np.log10(10.0) + 25.0
    expected_muJy = 10 ** ((23.9 - expected_ab_mag) / 2.5)
    assert value == pytest.approx(expected_muJy)


def test_total_lens_flux_mujy_returns_nan_when_no_light_profile():
    fit = MagicMock()
    fit.galaxy_image_dict.__getitem__.side_effect = KeyError("no light")
    fit.tracer.galaxies = [object()]

    assert np.isnan(total_lens_flux_mujy(fit=fit, magzero=25.0))


def test_total_lens_flux_mujy_missing_magzero_returns_nan_and_warns(caplog):
    _latent_module._MAGZERO_WARNED.discard("total_lens_flux_mujy")
    caplog.set_level(logging.WARNING)
    fit = _fit_with_galaxy_images({0: [1.0, 2.0, 3.0, 4.0]})

    value = total_lens_flux_mujy(fit=fit, magzero=None)

    assert np.isnan(value)
    assert any(
        "magzero" in rec.message and "total_lens_flux_mujy" in rec.message
        for rec in caplog.records
    )


def test_total_lens_flux_against_known_image():
    fit = _fit_with_galaxy_images({0: [1.0, 2.0, 3.0, 4.0]})

    # Raw sum — no AB-mag conversion. `magzero` accepted but ignored.
    assert total_lens_flux(fit=fit) == pytest.approx(10.0)
    assert total_lens_flux(fit=fit, magzero=25.0) == pytest.approx(10.0)


def test_total_lens_flux_returns_nan_when_no_light_profile():
    fit = MagicMock()
    fit.galaxy_image_dict.__getitem__.side_effect = KeyError("no light")
    fit.tracer.galaxies = [object()]

    assert np.isnan(total_lens_flux(fit=fit))


def test_total_lensed_source_flux_mujy_against_known_image():
    # Two galaxies; source at index -1 (i.e. index 1).
    fit = _fit_with_galaxy_images({0: [0.0, 0.0], 1: [5.0, 5.0]})
    value = total_lensed_source_flux_mujy(fit=fit, magzero=25.0)

    expected_ab_mag = -2.5 * np.log10(10.0) + 25.0
    expected_muJy = 10 ** ((23.9 - expected_ab_mag) / 2.5)
    assert value == pytest.approx(expected_muJy)


def test_total_lensed_source_flux_mujy_missing_magzero_returns_nan_and_warns(caplog):
    _latent_module._MAGZERO_WARNED.discard("total_lensed_source_flux_mujy")
    caplog.set_level(logging.WARNING)
    fit = _fit_with_galaxy_images({0: [0.0, 0.0], 1: [5.0, 5.0]})

    value = total_lensed_source_flux_mujy(fit=fit, magzero=None)

    assert np.isnan(value)
    assert any(
        "magzero" in rec.message and "total_lensed_source_flux_mujy" in rec.message
        for rec in caplog.records
    )


def test_total_lensed_source_flux_against_known_image():
    fit = _fit_with_galaxy_images({0: [0.0, 0.0], 1: [5.0, 5.0]})

    assert total_lensed_source_flux(fit=fit) == pytest.approx(10.0)


def test_total_source_flux_mujy_against_known_image():
    source = SimpleNamespace(
        image_2d_from=lambda grid, xp=np: SimpleNamespace(
            array=np.array([2.0, 3.0, 5.0])
        )
    )
    # Both `tracer` and `tracer_linear_light_profiles_to_light_profiles`
    # point at the same galaxies for non-linear fits — the conversion
    # property is a no-op pass-through.
    galaxies_namespace = SimpleNamespace(galaxies=[object(), source])
    fit = SimpleNamespace(
        tracer=galaxies_namespace,
        tracer_linear_light_profiles_to_light_profiles=galaxies_namespace,
        dataset=SimpleNamespace(grids=SimpleNamespace(lp=object())),
    )
    value = total_source_flux_mujy(fit=fit, magzero=25.0)

    expected_ab_mag = -2.5 * np.log10(10.0) + 25.0
    expected_muJy = 10 ** ((23.9 - expected_ab_mag) / 2.5)
    assert value == pytest.approx(expected_muJy)


def test_total_source_flux_mujy_uses_converted_tracer_for_linear_profiles():
    """When the source has a linear light profile, ``fit.tracer.galaxies[-1]``
    is un-solved (``image_2d_from`` returns zeros). The library must read from
    ``fit.tracer_linear_light_profiles_to_light_profiles`` where intensities
    are filled in from the inversion."""

    unsolved_source = SimpleNamespace(
        image_2d_from=lambda grid, xp=np: SimpleNamespace(array=np.zeros(4))
    )
    solved_source = SimpleNamespace(
        image_2d_from=lambda grid, xp=np: SimpleNamespace(
            array=np.array([1.0, 2.0, 3.0, 4.0])
        )
    )
    fit = SimpleNamespace(
        tracer=SimpleNamespace(galaxies=[object(), unsolved_source]),
        tracer_linear_light_profiles_to_light_profiles=SimpleNamespace(
            galaxies=[object(), solved_source]
        ),
        dataset=SimpleNamespace(grids=SimpleNamespace(lp=object())),
    )

    value = total_source_flux_mujy(fit=fit, magzero=25.0)

    # Expected from the solved source (sum = 10):
    expected_ab_mag = -2.5 * np.log10(10.0) + 25.0
    expected_muJy = 10 ** ((23.9 - expected_ab_mag) / 2.5)
    assert value == pytest.approx(expected_muJy)
    # Confirm we did NOT read from the unsolved tracer (which would give 0).
    assert value != 0.0


def test_total_source_flux_mujy_missing_magzero_returns_nan_and_warns(caplog):
    _latent_module._MAGZERO_WARNED.discard("total_source_flux_mujy")
    caplog.set_level(logging.WARNING)

    # Source-plane fit fixture (matches the positive test above).
    source = SimpleNamespace(
        image_2d_from=lambda grid, xp=np: SimpleNamespace(
            array=np.array([2.0, 3.0, 5.0])
        )
    )
    galaxies_namespace = SimpleNamespace(galaxies=[object(), source])
    fit = SimpleNamespace(
        tracer=galaxies_namespace,
        tracer_linear_light_profiles_to_light_profiles=galaxies_namespace,
        dataset=SimpleNamespace(grids=SimpleNamespace(lp=object())),
    )

    value = total_source_flux_mujy(fit=fit, magzero=None)

    assert np.isnan(value)
    assert any(
        "magzero" in rec.message and "total_source_flux_mujy" in rec.message
        for rec in caplog.records
    )


def test_total_source_flux_against_known_image():
    source = SimpleNamespace(
        image_2d_from=lambda grid, xp=np: SimpleNamespace(
            array=np.array([2.0, 3.0, 5.0])
        )
    )
    galaxies_namespace = SimpleNamespace(galaxies=[object(), source])
    fit = SimpleNamespace(
        tracer=galaxies_namespace,
        tracer_linear_light_profiles_to_light_profiles=galaxies_namespace,
        dataset=SimpleNamespace(grids=SimpleNamespace(lp=object())),
    )

    assert total_source_flux(fit=fit) == pytest.approx(10.0)


def test_maybe_magzero_warn_logs_only_once_per_name(caplog):
    """Sibling of the per-latent NaN/warn tests — asserts the module-level
    dedup set really suppresses repeat warnings for the same name."""
    _latent_module._MAGZERO_WARNED.discard("total_lens_flux_mujy")
    caplog.set_level(logging.WARNING)

    fit = _fit_with_galaxy_images({0: [1.0, 2.0, 3.0, 4.0]})
    for _ in range(3):
        total_lens_flux_mujy(fit=fit, magzero=None)

    matching = [
        r for r in caplog.records if "total_lens_flux_mujy" in r.message
    ]
    assert len(matching) == 1


def test_magnification_is_lensed_over_intrinsic():
    # Image-plane lensed source flux = 10, source-plane intrinsic = 2.
    # The µJy conversions cancel in the ratio, so the result is 5.0.

    class _FakeSourceGalaxy:
        # Plain class so instances are hashable (used as galaxy_image_dict key).
        def image_2d_from(self, grid, xp=np):
            return SimpleNamespace(array=np.array([2.0]))

    source = _FakeSourceGalaxy()
    # Same fakes for tracer and the converted-tracer property (no-op
    # pass-through for non-linear profile fixtures).
    galaxies_namespace = SimpleNamespace(galaxies=[object(), source])
    fit = SimpleNamespace(
        tracer=galaxies_namespace,
        tracer_linear_light_profiles_to_light_profiles=galaxies_namespace,
        galaxy_image_dict={source: SimpleNamespace(array=np.array([10.0]))},
        dataset=SimpleNamespace(grids=SimpleNamespace(lp=object())),
    )

    value = magnification(fit=fit, magzero=25.0)
    assert value == pytest.approx(5.0)


def test_effective_einstein_radius_calls_lens_calc_numpy_path(monkeypatch):
    calls = {}

    class _SpyLensCalc:
        def einstein_radius_from(self, grid):
            calls["grid"] = grid
            return 1.234

        def einstein_radius_jit_from(self, init_guess):
            calls["init_guess"] = init_guess
            raise AssertionError("numpy path must not use jit_from")

    monkeypatch.setattr(
        "autogalaxy.operate.lens_calc.LensCalc.from_mass_obj",
        classmethod(lambda cls, tracer: _SpyLensCalc()),
    )
    fit = SimpleNamespace(
        tracer=object(),
        dataset=SimpleNamespace(grids=SimpleNamespace(lp="sentinel_grid")),
    )

    value = effective_einstein_radius(fit=fit, magzero=None, xp=np)

    assert value == pytest.approx(1.234)
    assert calls["grid"] == "sentinel_grid"


def test_effective_einstein_radius_jax_path_falls_back_to_numpy_when_dep_missing(
    monkeypatch, caplog
):
    """
    When ``xp is not np`` but ``jax_zero_contour`` isn't installed, the
    function must fall through to ``einstein_radius_from`` (the NumPy path)
    instead of crashing or returning NaN — caller-side fallback yields a
    real Einstein radius value. One warning is emitted per process.
    """
    _latent_module._JAX_ZERO_CONTOUR_FALLBACK_WARNED = False

    real_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "jax_zero_contour":
            raise ModuleNotFoundError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(_latent_module.importlib, "import_module", fake_import)

    calls = {}

    class _SpyLensCalc:
        def einstein_radius_from(self, grid):
            calls["grid"] = grid
            return 5.678

        def einstein_radius_jit_from(self, init_guess):
            raise AssertionError(
                "jit path must not run when jax_zero_contour is missing"
            )

    monkeypatch.setattr(
        "autogalaxy.operate.lens_calc.LensCalc.from_mass_obj",
        classmethod(lambda cls, tracer: _SpyLensCalc()),
    )
    fit = SimpleNamespace(
        tracer=object(),
        dataset=SimpleNamespace(grids=SimpleNamespace(lp="sentinel_grid")),
    )

    sentinel_xp = MagicMock()  # truthy `xp is not np`
    with caplog.at_level(logging.WARNING, logger=_latent_module.__name__):
        value = effective_einstein_radius(
            fit=fit, magzero=None, xp=sentinel_xp
        )

    assert value == pytest.approx(5.678)
    assert calls["grid"] == "sentinel_grid"
    fallback_warnings = [
        r for r in caplog.records if "falling back to NumPy" in r.message
    ]
    assert len(fallback_warnings) == 1


def test_effective_einstein_radius_returns_nan_on_value_error(monkeypatch):
    def _raise(cls, tracer):
        raise ValueError("singular mass model")

    monkeypatch.setattr(
        "autogalaxy.operate.lens_calc.LensCalc.from_mass_obj",
        classmethod(_raise),
    )
    fit = SimpleNamespace(
        tracer=object(),
        dataset=SimpleNamespace(grids=SimpleNamespace(lp=object())),
    )

    assert np.isnan(effective_einstein_radius(fit=fit, magzero=None))


# ---------------------------------------------------------------------------
# Registry / config-reader tests
# ---------------------------------------------------------------------------

def test_latent_keys_enabled_filters_disabled():
    enabled = latent_keys_enabled(
        yaml_config={"total_lens_flux_mujy": False, "magnification": False}
    )
    assert enabled == []


def test_latent_keys_enabled_preserves_yaml_order():
    yaml_config = {
        "magnification": True,
        "total_lens_flux_mujy": True,
        "effective_einstein_radius": True,
    }
    enabled = latent_keys_enabled(yaml_config=yaml_config)
    assert enabled == [
        "magnification",
        "total_lens_flux_mujy",
        "effective_einstein_radius",
    ]


def test_latent_keys_enabled_drops_unknown_with_warning(caplog):
    caplog.set_level(logging.WARNING)
    enabled = latent_keys_enabled(
        yaml_config={
            "never_registered_latent": True,
            "total_lens_flux_mujy": True,
        }
    )

    assert enabled == ["total_lens_flux_mujy"]
    assert any("never_registered_latent" in rec.message for rec in caplog.records)


def test_latent_functions_registry_keys():
    assert set(LATENT_FUNCTIONS) == {
        "total_lens_flux",
        "total_lensed_source_flux",
        "total_source_flux",
        "total_lens_flux_mujy",
        "total_lensed_source_flux_mujy",
        "total_source_flux_mujy",
        "magnification",
        "effective_einstein_radius",
    }


# ---------------------------------------------------------------------------
# AnalysisImaging end-to-end
# ---------------------------------------------------------------------------

def test_analysis_imaging_compute_latent_variables_aligns_with_keys(
    masked_imaging_7x7,
):
    lens_galaxy = al.Galaxy(redshift=0.5, light=al.lp.Sersic(intensity=0.1))
    source_galaxy = al.Galaxy(redshift=1.0, light=al.lp.Sersic(intensity=0.05))
    model = af.Collection(
        galaxies=af.Collection(lens=lens_galaxy, source=source_galaxy)
    )

    analysis = al.AnalysisImaging(
        dataset=masked_imaging_7x7, use_jax=False, magzero=25.0
    )

    parameters = np.array(model.physical_values_from_prior_medians)
    values = analysis.compute_latent_variables(parameters=parameters, model=model)

    assert isinstance(values, tuple)
    assert len(values) == len(analysis.LATENT_KEYS)
    # test_autolens/config/latent.yaml enables the three raw-flux keys plus
    # total_lens_flux_mujy. magzero=25.0 above keeps the µJy column finite.
    assert analysis.LATENT_KEYS == [
        "total_lens_flux",
        "total_lensed_source_flux",
        "total_source_flux",
        "total_lens_flux_mujy",
    ]
    assert all(np.isfinite(v) for v in values)


def test_analysis_imaging_compute_latent_variables_raises_when_empty(monkeypatch):
    monkeypatch.setattr(
        al.AnalysisImaging,
        "LATENT_KEYS",
        property(lambda self: []),
    )
    analysis = al.AnalysisImaging(dataset=MagicMock(), use_jax=False)

    with pytest.raises(NotImplementedError):
        analysis.compute_latent_variables(parameters=np.array([]), model=MagicMock())


def test_analysis_imaging_latent_keys_property_reads_config():
    # The autouse `set_config_path` fixture in test_autolens/conftest.py
    # pushes test_autolens/config/latent.yaml — the three raw-flux keys
    # and total_lens_flux_mujy are enabled there.
    analysis = al.AnalysisImaging(dataset=MagicMock(), use_jax=False)
    assert analysis.LATENT_KEYS == [
        "total_lens_flux",
        "total_lensed_source_flux",
        "total_source_flux",
        "total_lens_flux_mujy",
    ]
