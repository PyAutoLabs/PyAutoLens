import numpy as np
import pytest

import autolens as al
from autogalaxy.cosmology import FlatLambdaCDM, Planck15
from autolens.lens import los


def _approx_coefficients(n_planes):
    """
    Fixed placeholder mass-function and mass-concentration coefficients, for
    tests that exercise the sampling machinery rather than the colossus fits.
    They are not physically accurate (no redshift evolution).
    """
    mf = np.tile([-1.9, 8.0], (n_planes, 1))
    mc = np.tile([-3.0, 40.0], (n_planes, 1))
    return mf, mc


def test__negative_kappa_from__loose_quad_matches_reference_and_is_negative():
    """
    The ``quad_limit`` / ``quad_epsrel`` knobs that test mode lowers must thread
    through to both ``quad`` calls and still produce a finite, negative kappa.
    A coarse integration (limit=1) should agree with a finer one to a few
    percent — the value is unused in test mode, so loose accuracy is fine.
    """
    cosmology = Planck15()
    _, centres = los.los_planes_from(
        z_lens=0.5, z_source=1.0, planes_before_lens=4, planes_after_lens=4
    )
    mf, mc = _approx_coefficients(len(centres))

    kwargs = dict(
        z_centre=centres[0],
        comoving_volume_per_arcsec2=1.0,
        A_mf=mf[0, 0],
        B_mf=mf[0, 1],
        A_mc=mc[0, 0],
        B_mc=mc[0, 1],
        m_min=1e7,
        m_max=1e10,
        z_source=1.0,
        truncation_factor=100.0,
        c_scatter=0.15,
        cosmology=cosmology,
    )

    reference = los.negative_kappa_from(quad_limit=10, quad_epsrel=1e-3, **kwargs)
    coarse = los.negative_kappa_from(quad_limit=1, quad_epsrel=1e-1, **kwargs)

    assert reference < 0.0
    assert coarse < 0.0
    assert coarse == pytest.approx(reference, rel=0.05)


def test__galaxies_from__test_mode_caps_halos_per_plane(monkeypatch):
    """
    Under ``PYAUTO_TEST_MODE`` ``galaxies_from`` must cap the halo population to
    a handful per plane (so the downstream multi-plane ray tracing stays cheap)
    while still emitting one negative-kappa ``MassSheet`` galaxy per plane.
    """
    monkeypatch.setenv("PYAUTO_TEST_MODE", "2")

    cosmology = Planck15()
    _, centres = los.los_planes_from(
        z_lens=0.5, z_source=1.0, planes_before_lens=4, planes_after_lens=4
    )
    n_planes = len(centres)
    mf, mc = _approx_coefficients(n_planes)

    sampler = los.LOSSampler(
        z_lens=0.5,
        z_source=1.0,
        planes_before_lens=4,
        planes_after_lens=4,
        m_min=1e7,
        m_max=1e10,
        cone_radius_arcsec=5.0,
        c_scatter=0.15,
        truncation_factor=100.0,
        cosmology=cosmology,
        mass_function_coefficients=mf,
        mass_concentration_coefficients=mc,
        seed=42,
    )

    galaxies = sampler.galaxies_from()

    halos = [
        g
        for g in galaxies
        if hasattr(g, "mass") and isinstance(g.mass, al.mp.NFWTruncatedSph)
    ]
    sheets = [
        g
        for g in galaxies
        if hasattr(g, "mass_sheet") and isinstance(g.mass_sheet, al.mp.MassSheet)
    ]

    # One negative-kappa sheet per plane, all with negative convergence.
    assert len(sheets) == n_planes
    assert all(g.mass_sheet.kappa < 0.0 for g in sheets)

    # Halos are capped to at most three per plane (grouped by plane redshift).
    counts = {}
    for g in halos:
        counts[g.redshift] = counts.get(g.redshift, 0) + 1

    assert len(counts) > 0
    assert all(count <= 3 for count in counts.values())


def _test_mode_sampler(monkeypatch):
    monkeypatch.setenv("PYAUTO_TEST_MODE", "2")

    cosmology = Planck15()
    _, centres = los.los_planes_from(
        z_lens=0.5, z_source=1.0, planes_before_lens=4, planes_after_lens=4
    )
    n_planes = len(centres)
    mf, mc = _approx_coefficients(n_planes)

    sampler = los.LOSSampler(
        z_lens=0.5,
        z_source=1.0,
        planes_before_lens=4,
        planes_after_lens=4,
        m_min=1e7,
        m_max=1e10,
        cone_radius_arcsec=5.0,
        c_scatter=0.15,
        truncation_factor=100.0,
        cosmology=cosmology,
        mass_function_coefficients=mf,
        mass_concentration_coefficients=mc,
        seed=42,
    )

    return sampler, n_planes


def test__fields_from__sheets_are_mass_fields_with_the_legacy_kappa_per_plane(
    monkeypatch,
):
    """
    A negative-kappa sheet is external mass, not a galaxy. `fields_from` returns exactly the
    sheets `galaxies_from()` has always returned — same redshift, same kappa, same order —
    wrapped in a `MassField` instead of a `Galaxy`.
    """
    sampler, n_planes = _test_mode_sampler(monkeypatch)

    galaxies = sampler.galaxies_from()
    fields = sampler.fields_from()

    sheet_galaxies = [
        g
        for g in galaxies
        if hasattr(g, "mass_sheet") and isinstance(g.mass_sheet, al.mp.MassSheet)
    ]

    assert len(fields) == n_planes
    assert all(isinstance(field, al.MassField) for field in fields)

    for field, sheet_galaxy in zip(fields, sheet_galaxies):
        assert field.redshift == sheet_galaxy.redshift
        assert field.mass_sheet.kappa == pytest.approx(
            sheet_galaxy.mass_sheet.kappa, rel=1e-12
        )


def test__galaxies_from__sheets_as_fields_returns_halos_only(monkeypatch):
    sampler, n_planes = _test_mode_sampler(monkeypatch)

    everything = sampler.galaxies_from()
    halos_only = sampler.galaxies_from(sheets_as_fields=True)

    assert len(halos_only) == len(everything) - n_planes
    assert all(
        hasattr(g, "mass") and isinstance(g.mass, al.mp.NFWTruncatedSph)
        for g in halos_only
    )

    # The test-mode halo cap still applies on this path.
    counts = {}
    for g in halos_only:
        counts[g.redshift] = counts.get(g.redshift, 0) + 1

    assert len(counts) > 0
    assert all(count <= 3 for count in counts.values())


def test__galaxies_and_fields_from__is_the_pair_the_tracer_takes(monkeypatch):
    sampler, n_planes = _test_mode_sampler(monkeypatch)

    galaxies, fields = sampler.galaxies_and_fields_from()

    assert galaxies == sampler.galaxies_from(sheets_as_fields=True)
    assert len(fields) == n_planes

    tracer = al.Tracer(
        galaxies=galaxies
        + [
            al.Galaxy(redshift=0.5, mass=al.mp.IsothermalSph(einstein_radius=1.0)),
            al.Galaxy(redshift=1.0, bulge=al.lp.SersicSph(intensity=1.0)),
        ],
        fields=fields,
    )

    assert len(tracer.fields) == n_planes
    assert all(not isinstance(galaxy, al.MassField) for galaxy in tracer.galaxies)


def test__the_population_is_sampled_once_and_shared(monkeypatch):
    """
    `galaxies_from()` and `fields_from()` must describe the same line of sight, so the
    sample is cached on the instance rather than redrawn per call.
    """
    sampler, _ = _test_mode_sampler(monkeypatch)

    first = sampler.galaxies_from()
    second = sampler.galaxies_from()

    assert [g.redshift for g in first] == [g.redshift for g in second]
    assert first[0] is second[0]


def test__los_sampler__default_coefficients_match_sheth_tormen_reference(monkeypatch):
    """
    With no precomputed coefficients, ``LOSSampler`` fits the Sheth-Tormen mass function from
    colossus. The per-plane slope must sit near the reference -1.906 and the expected number of
    1e7 to 1e10 M_sun halos in a 5 arcsec cone (z_lens 0.5, z_source 1.0, 4 + 4 planes) near the
    reference 1282 halos measured with the former hmf Sheth-Mo-Tormen fit.

    Test mode only caps the *drawn* halos; the expected count ``n_bar`` each plane passes to
    ``number_of_halos_from`` is the uncapped population, which is what is checked here.
    """
    pytest.importorskip("colossus")

    monkeypatch.setenv("PYAUTO_TEST_MODE", "2")

    slopes = []
    n_bars = []

    mass_function_ab_from = los.mass_function_ab_from
    number_of_halos_from = los.number_of_halos_from

    def recording_mass_function_ab_from(*args, **kwargs):
        A, B = mass_function_ab_from(*args, **kwargs)
        slopes.append(A)
        return A, B

    def recording_number_of_halos_from(*args, **kwargs):
        n_bar = number_of_halos_from(*args, **kwargs)
        n_bars.append(n_bar)
        return n_bar

    monkeypatch.setattr(los, "mass_function_ab_from", recording_mass_function_ab_from)
    monkeypatch.setattr(los, "number_of_halos_from", recording_number_of_halos_from)

    sampler = los.LOSSampler(z_lens=0.5, z_source=1.0, seed=1)
    sampler.galaxies_and_fields_from()

    assert len(slopes) == 9
    assert all(A == pytest.approx(-1.906, abs=0.02) for A in slopes)
    assert sum(n_bars) == pytest.approx(1282.0, rel=0.05)


def test__mass_function_ab_from__missing_colossus_raises_clear_import_error(
    monkeypatch,
):
    """
    Without colossus installed the coefficient fits must stop with an ``ImportError`` that names
    colossus and how to install it, never a bare ``ModuleNotFoundError`` for another package.
    """
    import sys

    monkeypatch.setitem(sys.modules, "colossus", None)
    monkeypatch.setitem(sys.modules, "colossus.cosmology", None)

    with pytest.raises(ImportError, match="pip install colossus"):
        los.mass_function_ab_from(redshift=0.5)

    with pytest.raises(ImportError, match="pip install colossus"):
        los.mass_concentration_ab_from(redshift=0.5)


def test__mass_function_ab_from__cosmology_astropy_is_deprecated():
    pytest.importorskip("colossus")

    from astropy.cosmology import Planck15 as AstropyPlanck15

    with pytest.warns(DeprecationWarning, match="cosmology_astropy"):
        A, B = los.mass_function_ab_from(
            redshift=0.5, cosmology_astropy=AstropyPlanck15
        )

    assert A == pytest.approx(-1.907, abs=0.01)


def test__mass_function_ab_from__mismatched_cosmology_astropy_raises():
    """
    A ``cosmology_astropy`` that disagrees with the colossus cosmology must not be silently
    ignored (it would give the Planck15 mass function for a WMAP9 caller): it raises a
    ``ValueError`` naming both. A matching colossus name works.
    """
    pytest.importorskip("colossus")
    from astropy.cosmology import WMAP9

    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValueError, match="WMAP9") as exc:
            los.mass_function_ab_from(redshift=0.5, cosmology_astropy=WMAP9)
    assert "planck15" in str(exc.value)

    with pytest.warns(DeprecationWarning):
        A, B = los.mass_function_ab_from(
            redshift=0.5, cosmology_astropy=WMAP9, cosmology_name="WMAP9"
        )
    assert B == pytest.approx(8.108, abs=0.002)


def test__los_sampler__mismatched_cosmology_raises():
    """
    ``LOSSampler`` checks both ``cosmology`` and ``cosmology_astropy`` against
    ``cosmology_name_colossus`` when it computes the halo statistics with colossus, and raises a
    ``ValueError`` naming both on a mismatch. Matching pairs, and precomputed coefficients (no
    colossus), work unchanged.
    """
    pytest.importorskip("colossus")
    from astropy.cosmology import WMAP9

    wmap9_like = FlatLambdaCDM(H0=69.32, Om0=0.2865)

    with pytest.raises(ValueError, match="`cosmology`") as exc:
        los.LOSSampler(
            z_lens=0.5, z_source=1.0, cosmology=wmap9_like, seed=1
        ).galaxies_from()
    assert "planck15" in str(exc.value)

    with pytest.warns(DeprecationWarning):
        sampler = los.LOSSampler(
            z_lens=0.5, z_source=1.0, cosmology_astropy=WMAP9, seed=1
        )
    with pytest.raises(ValueError, match="WMAP9") as exc:
        sampler.galaxies_from()
    assert "planck15" in str(exc.value)

    mf, mc = _approx_coefficients(n_planes=9)
    los.LOSSampler(
        z_lens=0.5,
        z_source=1.0,
        cosmology=wmap9_like,
        mass_function_coefficients=mf,
        mass_concentration_coefficients=mc,
        seed=1,
    ).galaxies_from()


def test__los_sampler__matching_cosmology_pair_is_accepted(monkeypatch):
    pytest.importorskip("colossus")

    monkeypatch.setenv("PYAUTO_TEST_MODE", "2")

    wmap9_like = FlatLambdaCDM(H0=69.32, Om0=0.2865)

    galaxies = los.LOSSampler(
        z_lens=0.5,
        z_source=1.0,
        cosmology=wmap9_like,
        cosmology_name_colossus="WMAP9",
        seed=1,
    ).galaxies_from()

    assert len(galaxies) > 0
