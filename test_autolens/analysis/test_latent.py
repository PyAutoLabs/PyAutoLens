import importlib
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

import autoarray as aa
import autofit as af
import autolens as al
from autogalaxy.imaging.model.latent import (
    ab_mag_via_flux_from,
    flux_mujy_via_ab_mag_from,
)
from autolens.analysis import latent as _latent_module
from autolens.analysis.latent import (
    LATENT_FUNCTIONS,
    LatentLens,
    _pixelized_source_flux,
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

        def einstein_radius_jit_from(self, init_guess=None):
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

        def einstein_radius_jit_from(self, init_guess=None):
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
# Pixelized (inversion) source flux — PyAutoLens#726
# ---------------------------------------------------------------------------

def _lens_galaxy():
    return al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=1.0),
    )


# The scene the pixel-scale-stability test re-observes at two pixel scales.
_SCALE_TEST_LENS = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.0),
)
_SCALE_TEST_SOURCE = al.Galaxy(
    redshift=1.0,
    light=al.lp.Sersic(
        centre=(0.05, 0.05),
        intensity=1.0,
        effective_radius=0.2,
        sersic_index=1.0,
    ),
)


def _delaunay_fit(dataset):
    """A real ``FitImaging`` whose source galaxy's only light model is a
    Delaunay pixelization. The Delaunay mesh does not build its own
    image-plane mesh grid, so one is supplied via ``adapt_images``."""
    pixelization = al.Pixelization(
        mesh=al.mesh.Delaunay(pixels=9, zeroed_pixels=0),
        regularization=al.reg.Constant(coefficient=0.01),
    )
    source = al.Galaxy(redshift=1.0, pixelization=pixelization)

    image_plane_mesh_grid = al.image_mesh.Overlay(
        shape=(3, 3)
    ).image_plane_mesh_grid_from(mask=dataset.mask)

    adapt_images = al.AdaptImages(
        galaxy_image_dict={source: dataset.data},
        galaxy_image_plane_mesh_grid_dict={source: image_plane_mesh_grid},
    )

    fit = al.FitImaging(
        dataset=dataset,
        tracer=al.Tracer(galaxies=[_lens_galaxy(), source]),
        adapt_images=adapt_images,
    )
    return fit, source


def _rectangular_fit(dataset):
    """A real ``FitImaging`` whose source galaxy's only light model is a
    rectangular pixelization (which builds its own mesh grid)."""
    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=0.01),
    )
    source = al.Galaxy(redshift=1.0, pixelization=pixelization)

    fit = al.FitImaging(
        dataset=dataset,
        tracer=al.Tracer(galaxies=[_lens_galaxy(), source]),
    )
    return fit, source


_MASK_7X7 = np.array(
    [
        [True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True],
        [True, True, False, False, False, True, True],
        [True, True, False, False, False, True, True],
        [True, True, False, False, False, True, True],
        [True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True],
    ]
)


def _masked_imaging_7x7_at(pixel_scales):
    """The ``masked_imaging_7x7`` fixture's geometry at arbitrary pixel
    scales. The fixture itself is 1.0"/pixel, where a division by the data
    pixel area is numerically invisible."""
    dataset = aa.Imaging(
        data=aa.Array2D.no_mask(values=np.ones((7, 7)), pixel_scales=pixel_scales),
        psf=aa.Convolver(
            kernel=aa.Array2D.no_mask(
                values=[[0.0, 0.5, 0.0], [0.5, 1.0, 0.5], [0.0, 0.5, 0.0]],
                pixel_scales=pixel_scales,
            )
        ),
        noise_map=aa.Array2D.no_mask(
            values=2.0 * np.ones((7, 7)), pixel_scales=pixel_scales
        ),
        over_sample_size_lp=1,
    )
    return dataset.apply_mask(mask=aa.Mask2D(mask=_MASK_7X7, pixel_scales=pixel_scales))


def _mesh_integrated_flux(fit, source):
    """Σ_mappers Σ_i s_i A_i for the mappers belonging to ``source`` — the
    plain source-plane surface integral ∫ S d²β, computed here independently
    of the library implementation."""
    inversion = fit.inversion
    total = 0.0
    for linear_obj, galaxy in inversion.linear_obj_galaxy_dict.items():
        if galaxy is not source or not isinstance(linear_obj, aa.Mapper):
            continue
        total += np.sum(
            np.asarray(inversion.reconstruction_dict[linear_obj])
            * np.asarray(linear_obj.mesh_geometry.areas_for_magnification)
        )
    return total


def _expected_pixelized_flux(fit, source):
    """Σ_i s_i A_i / A_pix — the pixelized source flux in the module's
    per-data-pixel convention, the one every other flux latent uses. ``A_pix``
    is read from the same grid the light-profile term is evaluated on."""
    mesh_integral = _mesh_integrated_flux(fit=fit, source=source)
    return mesh_integral / fit.dataset.grids.lp.pixel_area


def test_pixelized_source_flux_zero_when_fit_has_no_inversion():
    """The light-profile path already covers a fit with no inversion, so the
    pixelized term must contribute exactly nothing (not NaN, not a sentinel)."""
    fit = SimpleNamespace(
        tracer=SimpleNamespace(galaxies=[object(), object()]),
        inversion=None,
    )
    assert _pixelized_source_flux(fit=fit) == 0.0

    # A fit object with no ``inversion`` attribute at all behaves identically.
    fit_no_attr = SimpleNamespace(
        tracer=SimpleNamespace(galaxies=[object(), object()])
    )
    assert _pixelized_source_flux(fit=fit_no_attr) == 0.0


def test_delaunay_pixelized_source_magnification_is_finite(masked_imaging_7x7):
    """Regression for PyAutoLens#726: a source whose only light model is a
    pixelization had zero source-plane flux and therefore infinite
    magnification."""
    fit, source = _delaunay_fit(masked_imaging_7x7)

    denominator = _expected_pixelized_flux(fit=fit, source=source)
    lensed = np.sum(fit.galaxy_image_dict[source].array)

    assert denominator != 0.0
    assert total_source_flux(fit=fit) == pytest.approx(denominator)
    assert _pixelized_source_flux(fit=fit) == pytest.approx(denominator)

    value = magnification(fit=fit, magzero=25.0)
    assert np.isfinite(value)
    assert value == pytest.approx(lensed / denominator)


def test_rectangular_pixelized_source_magnification_is_finite(masked_imaging_7x7):
    fit, source = _rectangular_fit(masked_imaging_7x7)

    denominator = _expected_pixelized_flux(fit=fit, source=source)
    lensed = np.sum(fit.galaxy_image_dict[source].array)

    assert denominator != 0.0
    assert total_source_flux(fit=fit) == pytest.approx(denominator)

    value = magnification(fit=fit, magzero=25.0)
    assert np.isfinite(value)
    assert value == pytest.approx(lensed / denominator)


def test_total_source_flux_sums_light_profile_and_pixelization(masked_imaging_7x7):
    """A source carrying BOTH a linear light profile and a pixelization must
    contribute both terms — neither may swallow the other."""
    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=0.01),
    )
    source = al.Galaxy(
        redshift=1.0,
        light=al.lp_linear.Sersic(),
        pixelization=pixelization,
    )
    fit = al.FitImaging(
        dataset=masked_imaging_7x7,
        tracer=al.Tracer(galaxies=[_lens_galaxy(), source]),
    )

    tracer = fit.tracer_linear_light_profiles_to_light_profiles
    light_profile_flux = np.sum(
        tracer.galaxies[-1].image_2d_from(grid=fit.dataset.grids.lp).array
    )
    pixelized_flux = _expected_pixelized_flux(fit=fit, source=source)

    assert light_profile_flux != 0.0
    assert pixelized_flux != 0.0
    assert _pixelized_source_flux(fit=fit) == pytest.approx(pixelized_flux)
    assert total_source_flux(fit=fit) == pytest.approx(
        light_profile_flux + pixelized_flux
    )


def test_pixelized_source_flux_is_per_data_pixel():
    """The pixelized term is Σ_i s_i A_i / A_pix, not Σ_i s_i A_i — it must
    join the per-data-pixel convention every other flux latent uses.

    At the 1" fixtures A_pix = 1 and the two are indistinguishable, so this
    test builds the same fit on 0.5"/pixel data (A_pix = 0.25), where the
    division is a factor of 4."""
    dataset = _masked_imaging_7x7_at(pixel_scales=(0.5, 0.5))
    assert dataset.grids.lp.pixel_area == pytest.approx(0.25)

    fit, source = _rectangular_fit(dataset)

    mesh_integral = _mesh_integrated_flux(fit=fit, source=source)
    assert mesh_integral != 0.0

    expected = mesh_integral / 0.25
    assert _pixelized_source_flux(fit=fit) == pytest.approx(expected)
    assert total_source_flux(fit=fit) == pytest.approx(expected)

    # The division is real, not a no-op inherited from a 1" fixture.
    assert total_source_flux(fit=fit) != pytest.approx(mesh_integral)

    lensed = np.sum(fit.galaxy_image_dict[source].array)
    assert magnification(fit=fit, magzero=25.0) == pytest.approx(lensed / expected)


def _simulated_lens_dataset(pixel_scales, shape_native):
    """A noise-free image of ``_SCALE_TEST_LENS`` lensing ``_SCALE_TEST_SOURCE``,
    masked to a 2" circle — the same physical scene at whatever pixel scale is
    asked for, so long as ``pixel_scales * shape_native`` is held fixed."""
    grid = al.Grid2D.uniform(shape_native=shape_native, pixel_scales=pixel_scales)
    tracer = al.Tracer(galaxies=[_SCALE_TEST_LENS, _SCALE_TEST_SOURCE])

    dataset = aa.Imaging(
        data=aa.Array2D(
            values=tracer.image_2d_from(grid=grid).native,
            mask=aa.Mask2D.all_false(
                shape_native=shape_native, pixel_scales=pixel_scales
            ),
        ),
        psf=aa.Convolver(
            kernel=aa.Array2D.no_mask(
                values=[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                pixel_scales=pixel_scales,
            )
        ),
        noise_map=aa.Array2D.no_mask(
            values=0.01 * np.ones(shape_native), pixel_scales=pixel_scales
        ),
        over_sample_size_lp=1,
    )
    return dataset.apply_mask(
        mask=aa.Mask2D.circular(
            shape_native=shape_native, pixel_scales=pixel_scales, radius=2.0
        )
    )


def test_pixelized_magnification_is_stable_across_pixel_scales():
    """The physical statement behind the per-data-pixel convention: fit the
    *same* scene at two pixel scales and the pixelized magnification stays put.

    What it pins is deliberately loose — a rectangular reconstruction of a
    compact Sersic over a 6.4" field is coarse, so only the order of magnitude
    is meaningful and the window here is a factor of two. It is exactly what
    the unscaled ``Sum_i s_i A_i`` violated: undivided, the two answers differ
    by the pixel-area ratio, 4x, which the last assertion checks is outside
    the window (so the window is not vacuous).

    It pins nothing about agreement with a light-profile source's
    magnification, which at this coarseness is not itself scale-stable.
    """
    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(20, 20)),
        regularization=al.reg.Constant(coefficient=1.0),
    )

    values = {}
    for pixel_scales, shape_native in (((0.4, 0.4), (16, 16)), ((0.2, 0.2), (32, 32))):
        dataset = _simulated_lens_dataset(
            pixel_scales=pixel_scales, shape_native=shape_native
        )
        fit = al.FitImaging(
            dataset=dataset,
            tracer=al.Tracer(
                galaxies=[
                    _SCALE_TEST_LENS,
                    al.Galaxy(redshift=1.0, pixelization=pixelization),
                ]
            ),
        )
        values[pixel_scales[0]] = magnification(fit=fit, magzero=25.0)

    assert all(np.isfinite(v) for v in values.values())

    ratio = values[0.4] / values[0.2]
    assert 0.5 < ratio < 2.0

    # Without the division by the data pixel area the same two fits would
    # differ by the ratio of pixel areas (0.16 / 0.04), i.e. off the window.
    unscaled_ratio = ratio * (0.16 / 0.04)
    assert not 0.5 < unscaled_ratio < 2.0


def test_pixelized_source_flux_nan_when_pixel_area_unresolvable(caplog):
    """A fit whose dataset publishes neither a grid ``pixel_area`` nor
    ``pixel_scales`` yields NaN plus one warning per process — never a
    silently unscaled number."""
    _latent_module._PIXEL_AREA_WARNED = False
    caplog.set_level(logging.WARNING)

    dataset = _masked_imaging_7x7_at(pixel_scales=(0.5, 0.5))
    fit, _ = _rectangular_fit(dataset)
    fit_no_pixel_area = SimpleNamespace(
        tracer=fit.tracer,
        inversion=fit.inversion,
        dataset=SimpleNamespace(grids=SimpleNamespace(lp=SimpleNamespace())),
    )

    values = [_pixelized_source_flux(fit=fit_no_pixel_area) for _ in range(3)]

    assert all(np.isnan(v) for v in values)
    matching = [r for r in caplog.records if "data pixel area" in r.message]
    assert len(matching) == 1


def test_light_profile_only_fit_source_flux_unchanged(fit_imaging_x2_plane_7x7):
    """The light-profile-only path is untouched: the pixelized term is exactly
    zero, so ``total_source_flux`` is still the plain light-profile sum."""
    fit = fit_imaging_x2_plane_7x7

    tracer = fit.tracer_linear_light_profiles_to_light_profiles
    light_profile_flux = np.sum(
        tracer.galaxies[-1].image_2d_from(grid=fit.dataset.grids.lp).array
    )

    assert fit.inversion is None
    assert _pixelized_source_flux(fit=fit) == 0.0
    assert total_source_flux(fit=fit) == pytest.approx(light_profile_flux)
    assert total_source_flux_mujy(fit=fit, magzero=25.0) == pytest.approx(
        flux_mujy_via_ab_mag_from(
            ab_mag=ab_mag_via_flux_from(flux=light_profile_flux, magzero=25.0)
        )
    )
    assert np.isfinite(magnification(fit=fit, magzero=25.0))


class _NoAreasMapper(aa.Mapper):
    """A mapper whose mesh geometry publishes no ``areas_for_magnification``
    (no exact quadrature areas), bypassing ``Mapper.__init__``."""

    def __init__(self):
        pass

    @property
    def mesh_geometry(self):
        return SimpleNamespace()


def test_pixelized_source_flux_nan_and_one_warning_without_mesh_areas(caplog):
    _latent_module._MESH_AREAS_WARNED = False
    caplog.set_level(logging.WARNING)

    source = object()
    mapper = _NoAreasMapper()
    fit = SimpleNamespace(
        tracer=SimpleNamespace(galaxies=[object(), source]),
        inversion=SimpleNamespace(
            linear_obj_galaxy_dict={mapper: source},
            reconstruction_dict={mapper: np.ones(4)},
        ),
    )

    values = [_pixelized_source_flux(fit=fit) for _ in range(3)]

    assert all(np.isnan(v) for v in values)
    matching = [r for r in caplog.records if "areas_for_magnification" in r.message]
    assert len(matching) == 1


def test_pixelized_source_flux_zero_when_no_mapper_belongs_to_source():
    """A mapper belonging to another galaxy (e.g. a pixelized lens) must not
    leak into the source-plane flux."""
    source = object()
    other_galaxy = object()
    mapper = _NoAreasMapper()
    fit = SimpleNamespace(
        tracer=SimpleNamespace(galaxies=[other_galaxy, source]),
        inversion=SimpleNamespace(
            linear_obj_galaxy_dict={mapper: other_galaxy},
            reconstruction_dict={mapper: np.ones(4)},
        ),
    )

    assert _pixelized_source_flux(fit=fit) == 0.0


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

def test_latent_lens_variables_aligns_with_keys(
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
    values = LatentLens.variables(analysis, parameters=parameters, model=model)
    keys = LatentLens.keys(analysis)

    assert isinstance(values, tuple)
    assert len(values) == len(keys)
    # test_autolens/config/latent.yaml enables the three raw-flux keys plus
    # total_lens_flux_mujy. magzero=25.0 above keeps the µJy column finite.
    assert keys == [
        "total_lens_flux",
        "total_lensed_source_flux",
        "total_source_flux",
        "total_lens_flux_mujy",
    ]
    assert all(np.isfinite(v) for v in values)


def test_latent_lens_variables_raises_when_empty(monkeypatch):
    monkeypatch.setattr(_latent_module, "latent_keys_enabled", lambda *a, **k: [])
    analysis = al.AnalysisImaging(dataset=MagicMock(), use_jax=False)

    with pytest.raises(NotImplementedError):
        LatentLens.variables(analysis, parameters=np.array([]), model=MagicMock())


def test_analysis_imaging_declares_latent_lens_and_keys_read_config():
    # The autouse `set_config_path` fixture in test_autolens/conftest.py
    # pushes test_autolens/config/latent.yaml — the three raw-flux keys
    # and total_lens_flux_mujy are enabled there.
    analysis = al.AnalysisImaging(dataset=MagicMock(), use_jax=False)
    assert al.AnalysisImaging.Latent is LatentLens
    assert LatentLens.keys(analysis) == [
        "total_lens_flux",
        "total_lensed_source_flux",
        "total_source_flux",
        "total_lens_flux_mujy",
    ]


# ---------------------------------------------------------------------------
# latent_instance_from: assertions on the NumPy path, skipped under JAX
# ---------------------------------------------------------------------------

def _sersic_model_with_assertion():
    model = af.Model(al.lp.Sersic)
    model.add_assertion(model.effective_radius > model.sersic_index)
    return model


def test_latent_instance_from_numpy_checks_assertions():
    model = _sersic_model_with_assertion()
    medians = np.array(model.physical_values_from_prior_medians)

    # sersic_index above effective_radius violates the assertion -> the NumPy
    # path raises, exactly as the likelihood path does (Fitness -> resample).
    bad = medians.copy()
    bad[model.paths.index(("effective_radius",))] = 0.1
    bad[model.paths.index(("sersic_index",))] = 4.0
    with pytest.raises(af.exc.FitException):
        _latent_module.latent_instance_from(model=model, parameters=bad, xp=np)

    good = medians.copy()
    good[model.paths.index(("effective_radius",))] = 4.0
    good[model.paths.index(("sersic_index",))] = 1.0
    instance = _latent_module.latent_instance_from(
        model=model, parameters=good, xp=np
    )
    assert instance.effective_radius == pytest.approx(4.0)


def test_latent_instance_from_jax_skips_assertions_under_jit():
    """
    The bug: the latent engine evaluates ``Latent.variables`` under a
    per-sample ``jax.jit``, and ``instance_from_vector`` with its default
    assertion check applies a Python ``not`` to a traced boolean there. The
    raise was swallowed into a NaN row for every sample, so a model with any
    assertion wrote no latent output at all.
    """
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    model = _sersic_model_with_assertion()
    vector = np.array(model.physical_values_from_prior_medians)
    vector[model.paths.index(("effective_radius",))] = 4.0
    vector[model.paths.index(("sersic_index",))] = 1.0

    # The default instance construction cannot be traced.
    with pytest.raises(jax.errors.TracerBoolConversionError):
        jax.jit(lambda v: model.instance_from_vector(vector=v).effective_radius)(
            vector
        )

    # The helper mirrors Fitness's JAX path: assertions skipped, xp threaded.
    value = jax.jit(
        lambda v: _latent_module.latent_instance_from(
            model=model, parameters=v, xp=jnp
        ).effective_radius
    )(vector)
    assert float(value) == pytest.approx(4.0)


def test_latent_lens_variables_traces_under_jit_with_model_assertion(
    masked_imaging_7x7,
):
    """
    End to end: ``LatentLens.variables`` on a model carrying an assertion,
    evaluated exactly as the engine's ``jit`` batch path evaluates it, returns
    finite values for every enabled latent (instead of raising and being
    masked to a NaN row).
    """
    jax = pytest.importorskip("jax")

    lens = af.Model(al.Galaxy, redshift=0.5, light=af.Model(al.lp.Sersic))
    source = af.Model(al.Galaxy, redshift=1.0, light=af.Model(al.lp.Sersic))
    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))
    model.add_assertion(
        model.galaxies.lens.light.intensity > model.galaxies.source.light.intensity
    )

    vector = np.array(model.physical_values_from_prior_medians)
    vector[model.paths.index(("galaxies", "lens", "light", "intensity"))] = 0.1
    vector[model.paths.index(("galaxies", "source", "light", "intensity"))] = 0.05

    analysis = al.AnalysisImaging(dataset=masked_imaging_7x7, use_jax=True, magzero=25.0)
    keys = LatentLens.keys(analysis)

    values = jax.jit(
        lambda v: LatentLens.variables(analysis, parameters=v, model=model)
    )(vector)

    assert len(values) == len(keys)
    assert all(np.isfinite(float(v)) for v in values)

    # And they agree with the NumPy evaluation of the same sample.
    analysis_np = al.AnalysisImaging(
        dataset=masked_imaging_7x7, use_jax=False, magzero=25.0
    )
    values_np = LatentLens.variables(analysis_np, parameters=vector, model=model)
    assert np.allclose(
        [float(v) for v in values], [float(v) for v in values_np], rtol=1e-6
    )
