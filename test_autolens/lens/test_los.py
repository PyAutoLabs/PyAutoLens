"""
Tests for the line-of-sight halo sampling module ``autolens.lens.los``.
"""

import numpy as np
import pytest

import autogalaxy as ag
from autogalaxy.cosmology import Planck15

from autolens.lens.los import (
    comoving_distance_mpc_from,
    comoving_volume_mpc3_from,
    los_planes_from,
    negative_kappa_from,
    number_of_halos_from,
    sample_halo_masses,
    sample_positions_in_circle,
    sample_concentrations,
    _mass_ratio_from_concentration_and_tau,
    LOSSampler,
)


@pytest.fixture
def cosmology():
    return Planck15()


# --------------------------------------------------------------------------
# Comoving distance / volume
# --------------------------------------------------------------------------


class TestComovingDistance:
    def test_zero_redshift(self, cosmology):
        assert comoving_distance_mpc_from(0.0, cosmology) == pytest.approx(
            0.0, abs=1e-6
        )

    def test_positive_redshift(self, cosmology):
        d = comoving_distance_mpc_from(0.5, cosmology)
        assert 1900.0 < d < 2000.0

    def test_monotonic(self, cosmology):
        d1 = comoving_distance_mpc_from(0.3, cosmology)
        d2 = comoving_distance_mpc_from(0.6, cosmology)
        d3 = comoving_distance_mpc_from(1.0, cosmology)
        assert d1 < d2 < d3


class TestComovingVolume:
    def test_zero(self, cosmology):
        assert comoving_volume_mpc3_from(0.0, cosmology) == pytest.approx(
            0.0, abs=1e-6
        )

    def test_positive(self, cosmology):
        v = comoving_volume_mpc3_from(0.5, cosmology)
        assert v > 0

    def test_monotonic(self, cosmology):
        v1 = comoving_volume_mpc3_from(0.3, cosmology)
        v2 = comoving_volume_mpc3_from(0.6, cosmology)
        assert v2 > v1


# --------------------------------------------------------------------------
# Plane slicing
# --------------------------------------------------------------------------


class TestLosPlanes:
    def test_boundary_endpoints(self, cosmology):
        boundaries, centres = los_planes_from(
            z_lens=0.5,
            z_source=1.0,
            planes_before_lens=4,
            planes_after_lens=4,
        )
        assert boundaries[0] == pytest.approx(0.0)
        assert boundaries[-1] == pytest.approx(1.0)

    def test_number_of_planes(self, cosmology):
        boundaries, centres = los_planes_from(
            z_lens=0.5,
            z_source=1.0,
            planes_before_lens=4,
            planes_after_lens=4,
        )
        n_planes = len(centres)
        # 4 front + 5 back (back includes the lens plane itself)
        assert n_planes == 9
        assert len(boundaries) == n_planes + 1

    def test_centres_within_boundaries(self, cosmology):
        boundaries, centres = los_planes_from(
            z_lens=0.5,
            z_source=1.0,
            planes_before_lens=4,
            planes_after_lens=4,
        )
        assert np.all(centres > 0.0)
        assert np.all(centres < 1.0)

    def test_boundaries_sorted(self, cosmology):
        boundaries, centres = los_planes_from(
            z_lens=0.5,
            z_source=1.0,
            planes_before_lens=4,
            planes_after_lens=4,
        )
        assert np.all(np.diff(boundaries) > 0)
        assert np.all(np.diff(centres) > 0)

    def test_matches_los_pipes_slicing_bounder(self):
        """
        Verify against the reference slicing_bounder function from los_pipes.
        """
        z_lens = 0.5
        z_source = 1.0
        pls = [4, 4]

        gap_front = z_lens / (pls[0] + 1.0)
        gap_back = (z_source - z_lens) / (pls[1] + 1.0)
        z_front_ref = np.arange(1.5 * gap_front, z_lens, gap_front)
        z_back_ref = np.arange(
            z_lens + 0.5 * gap_back, z_source - 1.0 * gap_back, gap_back
        )
        boundaries_ref = np.concatenate(([0.0], z_front_ref, z_back_ref, [z_source]))

        centres_front_ref = np.linspace(0.0, z_lens, pls[0] + 2)[1:-1]
        centres_back_ref = np.linspace(z_lens, z_source, pls[1] + 2)[:-1]
        centres_ref = np.concatenate((centres_front_ref, centres_back_ref))

        boundaries, centres = los_planes_from(
            z_lens=z_lens,
            z_source=z_source,
            planes_before_lens=pls[0],
            planes_after_lens=pls[1],
        )

        np.testing.assert_allclose(boundaries, boundaries_ref, atol=1e-12)
        np.testing.assert_allclose(centres, centres_ref, atol=1e-12)


# --------------------------------------------------------------------------
# Mass ratio
# --------------------------------------------------------------------------


class TestMassRatio:
    def test_known_values(self):
        """
        Compare against los_pipes scale_c function for c=10, tau from cti=100.
        """
        from scipy.optimize import fsolve

        c = 10.0
        cti = 100.0
        delta_c = 200.0 / 3.0 * (c ** 3 / (np.log(1 + c) - c / (1 + c)))
        tau = fsolve(
            lambda t, dc, ct: ct / 3.0 * (t ** 3 / (np.log(1 + t) - t / (1 + t))) - dc,
            10.0,
            args=(delta_c, cti),
        )[0]

        ratio = _mass_ratio_from_concentration_and_tau(c, tau)

        tau2 = tau ** 2
        expected = (
            tau2
            / (tau2 + 1.0) ** 2
            * ((tau2 - 1.0) * np.log(tau) + tau * np.pi - (tau2 + 1.0))
        ) / (np.log(1 + c) - c / (1 + c))

        assert ratio == pytest.approx(expected, rel=1e-10)

    def test_mass_ratio_less_than_one(self):
        ratio = _mass_ratio_from_concentration_and_tau(10.0, 25.0)
        assert 0 < ratio < 2.0


# --------------------------------------------------------------------------
# Sampling functions
# --------------------------------------------------------------------------


class TestNumberOfHalos:
    def test_positive(self):
        n = number_of_halos_from(
            A=-1.9, B=8.0, m_min=1e7, m_max=1e10, volume=1.0
        )
        assert n > 0

    def test_scales_with_volume(self):
        n1 = number_of_halos_from(A=-1.9, B=8.0, m_min=1e7, m_max=1e10, volume=1.0)
        n2 = number_of_halos_from(A=-1.9, B=8.0, m_min=1e7, m_max=1e10, volume=2.0)
        assert n2 == pytest.approx(2.0 * n1, rel=1e-10)


class TestSampleHaloMasses:
    def test_within_range(self):
        masses = sample_halo_masses(
            n=1000, m_min=1e7, m_max=1e10, A=-1.9, B=8.0, seed=42
        )
        assert len(masses) == 1000
        assert np.all(masses >= 1e7)
        assert np.all(masses <= 1e10)

    def test_reproducible(self):
        m1 = sample_halo_masses(n=100, m_min=1e7, m_max=1e10, A=-1.9, B=8.0, seed=42)
        m2 = sample_halo_masses(n=100, m_min=1e7, m_max=1e10, A=-1.9, B=8.0, seed=42)
        np.testing.assert_array_equal(m1, m2)

    def test_empty(self):
        masses = sample_halo_masses(n=0, m_min=1e7, m_max=1e10, A=-1.9, B=8.0)
        assert len(masses) == 0

    def test_skew_toward_low_mass(self):
        masses = sample_halo_masses(
            n=10000, m_min=1e7, m_max=1e10, A=-1.9, B=8.0, seed=42
        )
        median = np.median(masses)
        assert median < 1e9


class TestSamplePositions:
    def test_within_radius(self):
        pos = sample_positions_in_circle(n=1000, radius=5.0, seed=42)
        r = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
        assert np.all(r <= 5.0)

    def test_shape(self):
        pos = sample_positions_in_circle(n=100, radius=5.0, seed=42)
        assert pos.shape == (100, 2)

    def test_empty(self):
        pos = sample_positions_in_circle(n=0, radius=5.0)
        assert pos.shape == (0, 2)


class TestSampleConcentrations:
    def test_clipped(self):
        c = sample_concentrations(
            log10_masses=np.array([7.0, 8.0, 9.0, 10.0]),
            A_mc=-3.0,
            B_mc=40.0,
            c_scatter=0.15,
            seed=42,
        )
        assert np.all(c >= 0.1)
        assert np.all(c <= 200.0)

    def test_reasonable_values(self):
        c = sample_concentrations(
            log10_masses=np.full(1000, 9.0),
            A_mc=-3.0,
            B_mc=40.0,
            c_scatter=0.15,
            seed=42,
        )
        median_c = np.median(c)
        expected_mean = -3.0 * 9.0 + 40.0
        assert 0.5 * expected_mean < median_c < 2.0 * expected_mean


# --------------------------------------------------------------------------
# Negative kappa
# --------------------------------------------------------------------------


class TestNegativeKappa:
    def test_returns_negative(self, cosmology):
        kappa = negative_kappa_from(
            z_centre=0.25,
            comoving_volume_per_arcsec2=30.0,
            A_mf=-1.9,
            B_mf=8.0,
            A_mc=-3.0,
            B_mc=40.0,
            m_min=1e7,
            m_max=1e10,
            z_source=1.0,
            truncation_factor=100.0,
            c_scatter=0.15,
            cosmology=cosmology,
        )
        assert kappa < 0.0

    def test_magnitude_reasonable(self, cosmology):
        kappa = negative_kappa_from(
            z_centre=0.25,
            comoving_volume_per_arcsec2=30.0,
            A_mf=-1.9,
            B_mf=8.0,
            A_mc=-3.0,
            B_mc=40.0,
            m_min=1e7,
            m_max=1e10,
            z_source=1.0,
            truncation_factor=100.0,
            c_scatter=0.15,
            cosmology=cosmology,
        )
        assert -0.1 < kappa < 0.0

    def test_scales_with_volume(self, cosmology):
        kwargs = dict(
            z_centre=0.25,
            A_mf=-1.9,
            B_mf=8.0,
            A_mc=-3.0,
            B_mc=40.0,
            m_min=1e7,
            m_max=1e10,
            z_source=1.0,
            truncation_factor=100.0,
            c_scatter=0.15,
            cosmology=cosmology,
        )
        k1 = negative_kappa_from(comoving_volume_per_arcsec2=10.0, **kwargs)
        k2 = negative_kappa_from(comoving_volume_per_arcsec2=20.0, **kwargs)
        assert k2 == pytest.approx(2.0 * k1, rel=1e-6)


# --------------------------------------------------------------------------
# LOSSampler
# --------------------------------------------------------------------------


class TestLOSSampler:
    """
    LOSSampler tests use [2,2] planes which produces 5 planes
    (2 front + 3 back including the lens plane).
    """

    N_PLANES_2_2 = 5

    def _make_sampler(self, cosmology, seed=42):
        n = self.N_PLANES_2_2
        return LOSSampler(
            z_lens=0.5,
            z_source=1.0,
            planes_before_lens=2,
            planes_after_lens=2,
            m_min=1e7,
            m_max=1e10,
            cone_radius_arcsec=3.0,
            c_scatter=0.15,
            truncation_factor=100.0,
            cosmology=cosmology,
            mass_function_coefficients=np.tile([-1.9, 8.0], (n, 1)),
            mass_concentration_coefficients=np.tile([-3.0, 40.0], (n, 1)),
            seed=seed,
        )

    def test_returns_galaxies(self, cosmology):
        galaxies = self._make_sampler(cosmology).galaxies_from()
        assert len(galaxies) > 0

        has_halo = any(
            hasattr(g, "mass") and isinstance(g.mass, ag.mp.NFWTruncatedSph)
            for g in galaxies
        )
        has_sheet = any(
            hasattr(g, "mass_sheet") and isinstance(g.mass_sheet, ag.mp.MassSheet)
            for g in galaxies
        )
        assert has_halo
        assert has_sheet

    def test_number_of_sheets_matches_planes(self, cosmology):
        n_front = 3
        n_back = 2
        _, centres = los_planes_from(
            z_lens=0.5, z_source=1.0,
            planes_before_lens=n_front, planes_after_lens=n_back,
        )
        n_planes = len(centres)

        sampler = LOSSampler(
            z_lens=0.5,
            z_source=1.0,
            planes_before_lens=n_front,
            planes_after_lens=n_back,
            m_min=1e7,
            m_max=1e10,
            cone_radius_arcsec=3.0,
            c_scatter=0.15,
            truncation_factor=100.0,
            cosmology=cosmology,
            mass_function_coefficients=np.tile([-1.9, 8.0], (n_planes, 1)),
            mass_concentration_coefficients=np.tile([-3.0, 40.0], (n_planes, 1)),
            seed=42,
        )
        galaxies = sampler.galaxies_from()

        n_sheets = sum(
            1
            for g in galaxies
            if hasattr(g, "mass_sheet")
            and isinstance(g.mass_sheet, ag.mp.MassSheet)
        )
        assert n_sheets == n_planes

    def test_negative_kappa_sheets_are_negative(self, cosmology):
        galaxies = self._make_sampler(cosmology).galaxies_from()

        for g in galaxies:
            if hasattr(g, "mass_sheet") and isinstance(
                g.mass_sheet, ag.mp.MassSheet
            ):
                assert g.mass_sheet.kappa < 0.0

    def test_reproducible(self, cosmology):
        g1 = self._make_sampler(cosmology, seed=123).galaxies_from()
        g2 = self._make_sampler(cosmology, seed=123).galaxies_from()
        assert len(g1) == len(g2)

    def test_halo_redshifts_match_plane_centres(self, cosmology):
        galaxies = self._make_sampler(cosmology).galaxies_from()

        _, centres = los_planes_from(
            z_lens=0.5,
            z_source=1.0,
            planes_before_lens=2,
            planes_after_lens=2,
        )

        for g in galaxies:
            found = any(
                abs(g.redshift - c) < 1e-10 for c in centres
            )
            assert found, (
                f"Galaxy redshift {g.redshift} does not match any plane centre"
            )
