import numpy as np
import pytest

import autolens as al
from autogalaxy.cosmology import Planck15
from autolens.lens import los


def _approx_coefficients(n_planes):
    """
    Approximate per-plane mass-function and mass-concentration coefficients,
    matching the pre-computed fallback used by the los_halos workspace
    simulators (avoids the optional ``hmf`` / ``colossus`` dependencies).
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
