"""
Line-of-sight (LOS) halo sampling and negative kappa sheet computation.

This module implements the full pipeline for populating a light cone with
line-of-sight dark matter halos and computing compensatory negative convergence
sheets to maintain mass conservation.  It follows the methodology of
He et al. (2022, MNRAS 511, 3046).

The main entry point is :class:`LOSSampler`, which orchestrates plane slicing,
halo sampling, unit conversion, and negative kappa computation to produce a
list of :class:`autogalaxy.Galaxy` objects ready for inclusion in a
multi-plane :class:`autolens.Tracer`.
"""

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from typing import List, Optional, Tuple

import autogalaxy as ag
from autogalaxy.cosmology import Planck15


def comoving_distance_mpc_from(z, cosmology):
    """
    Comoving distance in Mpc to redshift *z*.

    Derived from ``cosmology.angular_diameter_distance_kpc_z1z2``
    via D_c = D_A(0, z) * (1 + z), converted from kpc to Mpc.
    """
    d_a_kpc = cosmology.angular_diameter_distance_kpc_z1z2(0.0, z)
    return d_a_kpc * (1.0 + z) / 1.0e3


def comoving_volume_mpc3_from(z, cosmology):
    """
    Comoving volume in Mpc^3 enclosed within redshift *z* (full sky).

    V_c = (4/3) pi D_c(z)^3  for a flat universe.
    """
    d_c = comoving_distance_mpc_from(z, cosmology)
    return (4.0 / 3.0) * np.pi * d_c ** 3


def los_planes_from(
    z_lens,
    z_source,
    planes_before_lens,
    planes_after_lens,
    cosmology=None,
):
    """
    Return plane boundaries and centres for the LOS light-cone geometry.

    Plane boundaries are determined by the ``slicing_bounder`` algorithm from
    los_pipes: the redshift interval [0, z_lens] is split into
    ``planes_before_lens`` slices and [z_lens, z_source] into
    ``planes_after_lens`` slices, each of roughly equal comoving-distance
    width.

    Parameters
    ----------
    z_lens
        Redshift of the main lens.
    z_source
        Redshift of the background source.
    planes_before_lens
        Number of LOS planes between the observer and the lens.
    planes_after_lens
        Number of LOS planes between the lens and the source.
    cosmology
        A ``LensingCosmology`` instance. Defaults to ``Planck15()``.

    Returns
    -------
    boundaries : ndarray of shape (n_planes + 1,)
        Redshift boundaries of each plane, starting at 0 and ending
        at z_source.
    centres : ndarray of shape (n_planes,)
        Redshift centres of each plane.
    """
    gap_front = z_lens / (planes_before_lens + 1.0)
    gap_back = (z_source - z_lens) / (planes_after_lens + 1.0)

    z_front = np.arange(1.5 * gap_front, z_lens, gap_front)
    z_back = np.arange(
        z_lens + 0.5 * gap_back, z_source - 1.0 * gap_back, gap_back
    )

    boundaries = np.concatenate(([0.0], z_front, z_back, [z_source]))

    centres_front = np.linspace(0.0, z_lens, planes_before_lens + 2)[1:-1]
    centres_back = np.linspace(z_lens, z_source, planes_after_lens + 2)[:-1]
    centres = np.concatenate((centres_front, centres_back))

    return boundaries, centres


def mass_function_ab_from(
    redshift,
    cosmology_astropy,
    m_min_log10=5.9,
    m_max_log10=10.1,
    dlog10m=0.1,
):
    """
    Linear fit to the Sheth-Mo-Tormen halo mass function at a given redshift.

    Returns coefficients (A, B) such that::

        log10(dn/dm) ≈ A * log10(m) + B

    where dn/dm is in units of h^4 Mpc^-3 M_sun^-1 (with h factors folded
    in to give masses in M_sun rather than M_sun/h).

    Parameters
    ----------
    redshift
        Redshift at which to evaluate the mass function.
    cosmology_astropy
        An ``astropy.cosmology`` instance (e.g. ``Planck15``), required by
        the ``hmf`` library.
    m_min_log10
        Lower bound of log10(m / M_sun) for the fit range.
    m_max_log10
        Upper bound of log10(m / M_sun) for the fit range.
    dlog10m
        Mass bin spacing in dex.

    Returns
    -------
    A : float
        Slope of the linear fit.
    B : float
        Intercept of the linear fit.
    """
    import hmf
    import hmf.fitting_functions

    h = cosmology_astropy.h

    m_min_h = m_min_log10 + np.log10(h)
    m_max_h = m_max_log10 + np.log10(h)

    mf = hmf.MassFunction(
        Mmin=m_min_h,
        Mmax=m_max_h,
        z=redshift,
        hmf_model=hmf.fitting_functions.SMT,
        dlog10m=dlog10m,
        cosmo_model=cosmology_astropy,
    )

    x = np.arange(m_min_log10, m_max_log10, dlog10m)
    y = np.log10(mf.dndm * (h ** 4.0))

    design = np.vstack([x, np.ones(len(x))]).T
    A, B = np.linalg.lstsq(design, y, rcond=None)[0]

    return A, B


def mass_concentration_ab_from(
    redshift,
    m_min_log10=5.8,
    m_max_log10=10.1,
    n_points=100,
    cosmology_name="planck15",
):
    """
    Linear fit to the Ludlow+16 concentration-mass relation.

    Returns coefficients (A, B) such that::

        c(m) ≈ A * log10(m / M_sun) + B

    Parameters
    ----------
    redshift
        Redshift at which to evaluate the relation.
    m_min_log10
        Lower bound of log10(m / M_sun).
    m_max_log10
        Upper bound of log10(m / M_sun).
    n_points
        Number of mass points for the fit.
    cosmology_name
        Colossus cosmology name (e.g. ``"planck15"``).

    Returns
    -------
    A : float
        Slope.
    B : float
        Intercept.
    """
    from colossus.cosmology import cosmology as col_cosmology
    from colossus.halo.concentration import concentration as col_concentration

    col_cosmo = col_cosmology.setCosmology(cosmology_name)

    lgm_range = np.linspace(m_min_log10, m_max_log10, n_points)
    mass_input = 10 ** lgm_range * col_cosmo.h
    cc = col_concentration(mass_input, "200c", redshift, model="ludlow16")

    design = np.vstack([lgm_range, np.ones(n_points)]).T
    A, B = np.linalg.lstsq(design, cc, rcond=None)[0]

    return A, B


def _mass_ratio_from_concentration_and_tau(concentration, tau):
    """
    M_tNFW / M_200 for a truncated NFW with scale tau = r_t / r_s.

    This is the ``scale_c`` function from los_pipes, used inside the
    negative kappa integrand.
    """
    tau2 = tau * tau
    tau_factor = (
        tau2
        / (tau2 + 1.0) ** 2
        * ((tau2 - 1.0) * np.log(tau) + tau * np.pi - (tau2 + 1.0))
    )
    c_factor = np.log(1.0 + concentration) - concentration / (
        1.0 + concentration
    )
    return tau_factor / c_factor


def negative_kappa_from(
    z_centre,
    comoving_volume_per_arcsec2,
    A_mf,
    B_mf,
    A_mc,
    B_mc,
    m_min,
    m_max,
    z_source,
    truncation_factor,
    c_scatter,
    cosmology,
):
    """
    Compute the negative convergence sheet for a single LOS plane.

    This is the compensatory convergence that accounts for the average
    contribution of *all* halos in the mass range, ensuring mass conservation.

    The computation performs a double integral over halo mass and
    concentration scatter:

    .. math::

        \\kappa_{\\rm neg} = -\\frac{1}{\\Sigma_{\\rm cr}}
        \\int_{m_{\\rm min}}^{m_{\\rm max}} \\frac{dn}{dm} \\, m \\,
        \\left\\langle \\frac{M_{\\rm tNFW}}{M_{200}} \\right\\rangle_{P(c|m)} \\, dm
        \\times V_{\\rm com/arcsec^2}

    Parameters
    ----------
    z_centre
        Redshift of the plane centre.
    comoving_volume_per_arcsec2
        Comoving volume of the slice per square arcsecond (Mpc^3 / arcsec^2).
    A_mf, B_mf
        Mass function linear fit coefficients.
    A_mc, B_mc
        Mass-concentration linear fit coefficients.
    m_min, m_max
        Halo mass range in solar masses.
    z_source
        Source redshift.
    truncation_factor
        Overdensity factor defining truncation radius (e.g. 100 for r_100).
    c_scatter
        Log-normal scatter in concentration (sigma in dex, e.g. 0.15).
    cosmology
        A ``LensingCosmology`` instance.

    Returns
    -------
    kappa_neg : float
        Negative convergence value (a negative number).
    """
    from scipy.optimize import fsolve

    sigma_cr = cosmology.critical_surface_density_between_redshifts_solar_mass_per_kpc2_from(
        redshift_0=z_centre,
        redshift_1=z_source,
    )
    sigma_cr_mpc2 = sigma_cr * 1.0e6

    def _solve_tau(c, delta_c, cti):
        return cti / 3.0 * (c ** 3 / (np.log(1 + c) - c / (1 + c))) - delta_c

    def _integrand_concentration(lgc, m, lgc_centre):
        c_val = 10 ** lgc
        delta_c = 200.0 / 3.0 * (
            c_val ** 3 / (np.log(1.0 + c_val) - c_val / (1.0 + c_val))
        )
        tau = fsolve(_solve_tau, 10.0, args=(delta_c, truncation_factor))[0]
        mass_ratio = _mass_ratio_from_concentration_and_tau(c_val, tau)

        prob = (
            1.0
            / (np.sqrt(2.0 * np.pi) * c_scatter)
            * np.exp(-((lgc - lgc_centre) ** 2) / (2.0 * c_scatter ** 2))
        )
        return mass_ratio * prob

    def _integrand_mass(m):
        lgc_centre = np.log10(A_mc * np.log10(m) + B_mc)
        lgc_lo = lgc_centre - 4.0 * c_scatter
        lgc_hi = lgc_centre + 4.0 * c_scatter

        c_integral = quad(
            _integrand_concentration, lgc_lo, lgc_hi, args=(m, lgc_centre)
        )[0]

        dndm = 10 ** B_mf * m ** A_mf
        return dndm * m * c_integral

    mass_integral = quad(_integrand_mass, m_min, m_max)[0]

    kappa = mass_integral * comoving_volume_per_arcsec2 / sigma_cr_mpc2

    return -kappa


def number_of_halos_from(A, B, m_min, m_max, volume):
    """
    Expected number of halos in a volume, from the integrated mass function.

    N_bar = volume * integral_{m_min}^{m_max} dn/dm dm
           = volume * 10^B * (m_max^{1+A} - m_min^{1+A}) / (1 + A)

    Parameters
    ----------
    A, B
        Mass function linear fit coefficients.
    m_min, m_max
        Mass range in solar masses.
    volume
        Comoving volume of the region (Mpc^3).

    Returns
    -------
    n_bar : float
        Mean expected number of halos.
    """
    cumulative = lambda m: 10 ** B * m ** (1.0 + A) / (1.0 + A)
    return volume * (cumulative(m_max) - cumulative(m_min))


def sample_halo_masses(n, m_min, m_max, A, B, seed=None):
    """
    Draw halo masses from the power-law mass function via inverse CDF.

    Parameters
    ----------
    n
        Number of masses to draw.
    m_min, m_max
        Mass range in solar masses.
    A, B
        Mass function linear fit coefficients.
    seed
        Optional RNG seed for reproducibility.

    Returns
    -------
    masses : ndarray of shape (n,)
        Sampled halo masses in solar masses.
    """
    if n == 0:
        return np.array([])

    cumulative = lambda m: 10 ** B * m ** (1.0 + A) / (1.0 + A)

    m_range = np.logspace(np.log10(m_min), np.log10(m_max), 10000)
    v_range = cumulative(m_range)
    v_range = (v_range - v_range[0]) / (v_range[-1] - v_range[0])

    draw_func = interp1d(v_range, m_range, kind="cubic")

    rng = np.random.RandomState(seed)
    u = rng.random(n)

    return draw_func(u)


def sample_positions_in_circle(n, radius, seed=None):
    """
    Draw positions uniformly within a circle of given radius.

    Parameters
    ----------
    n
        Number of positions to draw.
    radius
        Radius of the circle in arcsec.
    seed
        Optional RNG seed.

    Returns
    -------
    positions : ndarray of shape (n, 2)
        (y, x) positions in arcsec.
    """
    if n == 0:
        return np.empty((0, 2))

    rng = np.random.RandomState(seed)
    r = radius * np.sqrt(rng.random(n))
    theta = 2.0 * np.pi * rng.random(n)

    y = r * np.cos(theta)
    x = r * np.sin(theta)

    return np.column_stack((y, x))


def sample_concentrations(log10_masses, A_mc, B_mc, c_scatter, seed=None):
    """
    Sample concentrations from a log-normal distribution around the
    mass-concentration relation.

    Parameters
    ----------
    log10_masses
        Array of log10(m / M_sun) for each halo.
    A_mc, B_mc
        Mass-concentration relation coefficients:
        ``c_mean(m) = A_mc * log10(m) + B_mc``.
    c_scatter
        Scatter in log10(c) (e.g. 0.15 dex).
    seed
        Optional RNG seed.

    Returns
    -------
    concentrations : ndarray
        Sampled concentrations, clipped to [0.1, 200].
    """
    if len(log10_masses) == 0:
        return np.array([])

    rng = np.random.RandomState(seed)

    lgc_centre = np.log10(A_mc * log10_masses + B_mc)
    lgc_draw = rng.normal(loc=lgc_centre, scale=c_scatter)
    c_draw = 10 ** lgc_draw

    return np.clip(c_draw, 0.1, 200.0)


def light_cone_radius_at_z(z, z_lens, cone_radius_arcsec, cosmology):
    """
    Effective light-cone radius at redshift *z*.

    The cone has constant angular radius in front of the lens and shrinks
    behind it following the geometry of the convergent light cone.

    Parameters
    ----------
    z
        Redshift at which to evaluate.
    z_lens
        Lens redshift.
    cone_radius_arcsec
        Angular radius of the cone in arcsec (at/before the lens).
    cosmology
        A ``LensingCosmology`` instance.

    Returns
    -------
    radius : float
        Effective radius in arcsec.
    """
    if z <= z_lens:
        return cone_radius_arcsec

    d_ls = cosmology.angular_diameter_distance_kpc_z1z2(z_lens, z)
    d_s = cosmology.angular_diameter_distance_kpc_z1z2(0.0, z)
    d_source = cosmology.angular_diameter_distance_kpc_z1z2(
        0.0, cosmology._z_source_cache
        if hasattr(cosmology, "_z_source_cache")
        else z
    )
    d_ls_source = cosmology.angular_diameter_distance_kpc_z1z2(
        z_lens, cosmology._z_source_cache
        if hasattr(cosmology, "_z_source_cache")
        else z
    )

    return cone_radius_arcsec - 1.0 * (d_ls / d_s * d_source / d_ls_source)


class LOSSampler:
    """
    Sample line-of-sight halos and negative kappa sheets for multi-plane
    gravitational lensing simulations.

    This class orchestrates the full LOS pipeline:

    1. Slice the light cone into redshift planes.
    2. For each plane: compute the mass function and mass-concentration
       relation coefficients.
    3. For each plane: sample halo masses, positions, and concentrations.
    4. Convert each halo to an ``NFWTruncatedSph`` profile.
    5. Compute the negative kappa sheet for each plane.
    6. Return a list of ``Galaxy`` objects (halos + sheets) for the ``Tracer``.

    Parameters
    ----------
    z_lens
        Redshift of the main lens.
    z_source
        Redshift of the source.
    planes_before_lens
        Number of LOS planes between observer and lens.
    planes_after_lens
        Number of LOS planes between lens and source.
    m_min
        Minimum halo mass in M_sun (e.g. ``1e7``).
    m_max
        Maximum halo mass in M_sun (e.g. ``1e10``).
    cone_radius_arcsec
        Angular radius of the light cone in arcsec.
    c_scatter
        Log-normal scatter in concentration (dex, e.g. 0.15).
    truncation_factor
        Overdensity factor for truncation radius (e.g. 100 for r_100).
    cosmology
        A ``LensingCosmology`` instance. Defaults to ``Planck15()``.
    cosmology_astropy
        An ``astropy.cosmology`` instance for ``hmf``. Required if
        ``mass_function_ab_from`` is used (i.e. ``mass_function_coefficients``
        are not provided).
    cosmology_name_colossus
        Colossus cosmology name (e.g. ``"planck15"``).
    mass_function_coefficients
        Optional pre-computed (A, B) per plane, shape (n_planes, 2).
        If provided, skips ``hmf`` computation.
    mass_concentration_coefficients
        Optional pre-computed (A, B) per plane, shape (n_planes, 2).
        If provided, skips ``colossus`` computation.
    seed
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        z_lens: float,
        z_source: float,
        planes_before_lens: int = 4,
        planes_after_lens: int = 4,
        m_min: float = 1e7,
        m_max: float = 1e10,
        cone_radius_arcsec: float = 5.0,
        c_scatter: float = 0.15,
        truncation_factor: float = 100.0,
        cosmology=None,
        cosmology_astropy=None,
        cosmology_name_colossus: str = "planck15",
        mass_function_coefficients=None,
        mass_concentration_coefficients=None,
        seed: Optional[int] = None,
    ):
        self.z_lens = z_lens
        self.z_source = z_source
        self.planes_before_lens = planes_before_lens
        self.planes_after_lens = planes_after_lens
        self.m_min = m_min
        self.m_max = m_max
        self.cone_radius_arcsec = cone_radius_arcsec
        self.c_scatter = c_scatter
        self.truncation_factor = truncation_factor
        self.cosmology = cosmology or Planck15()
        self.cosmology_astropy = cosmology_astropy
        self.cosmology_name_colossus = cosmology_name_colossus
        self.mass_function_coefficients = mass_function_coefficients
        self.mass_concentration_coefficients = mass_concentration_coefficients
        self.seed = seed

    def galaxies_from(self) -> List[ag.Galaxy]:
        """
        Sample LOS halos and negative kappa sheets.

        Returns
        -------
        galaxies : list of Galaxy
            A list containing:
            - One ``Galaxy`` per sampled halo, each with an
              ``NFWTruncatedSph`` mass profile.
            - One ``Galaxy`` per plane with a ``MassSheet`` for the
              negative kappa correction.
        """
        cosmology = self.cosmology
        rng = np.random.RandomState(self.seed)

        boundaries, centres = los_planes_from(
            z_lens=self.z_lens,
            z_source=self.z_source,
            planes_before_lens=self.planes_before_lens,
            planes_after_lens=self.planes_after_lens,
            cosmology=cosmology,
        )

        n_planes = len(centres)

        all_sky_arcsec2 = 4.0 * np.pi * (180.0 / np.pi * 3600.0) ** 2
        arcsec2_fraction = 1.0 / all_sky_arcsec2

        d_s = cosmology.angular_diameter_distance_kpc_z1z2(0.0, self.z_source)
        d_ls = cosmology.angular_diameter_distance_kpc_z1z2(
            self.z_lens, self.z_source
        )

        mf_coeffs = self.mass_function_coefficients
        mc_coeffs = self.mass_concentration_coefficients

        if mf_coeffs is None:
            mf_coeffs = np.zeros((n_planes, 2))
            for i in range(n_planes):
                A, B = mass_function_ab_from(
                    redshift=centres[i],
                    cosmology_astropy=self.cosmology_astropy,
                )
                mf_coeffs[i] = [A, B]

        if mc_coeffs is None:
            mc_coeffs = np.zeros((n_planes, 2))
            for i in range(n_planes):
                A, B = mass_concentration_ab_from(
                    redshift=centres[i],
                    cosmology_name=self.cosmology_name_colossus,
                )
                mc_coeffs[i] = [A, B]

        galaxies = []

        for i in range(n_planes):
            z_lo = boundaries[i]
            z_hi = boundaries[i + 1]
            z_cen = centres[i]

            vol_all_sky = (
                comoving_volume_mpc3_from(z_hi, cosmology)
                - comoving_volume_mpc3_from(z_lo, cosmology)
            )
            vol_per_arcsec2 = vol_all_sky * arcsec2_fraction

            d_c_cen = comoving_distance_mpc_from(z_cen, cosmology)
            arcsec_to_mpc_comoving = (
                np.pi / 180.0 / 3600.0 * d_c_cen / (1.0 + z_cen)
            )
            vol_depth = vol_per_arcsec2 / arcsec_to_mpc_comoving ** 2

            if z_cen <= self.z_lens:
                r_cone = self.cone_radius_arcsec
            else:
                d_l_z = cosmology.angular_diameter_distance_kpc_z1z2(
                    self.z_lens, z_cen
                )
                d_z = cosmology.angular_diameter_distance_kpc_z1z2(0.0, z_cen)
                r_cone = self.cone_radius_arcsec - 1.0 * (
                    d_l_z / d_z * d_s / d_ls
                )

            cone_area = np.pi * r_cone ** 2
            plane_volume = cone_area * vol_per_arcsec2

            n_bar = number_of_halos_from(
                A=mf_coeffs[i, 0],
                B=mf_coeffs[i, 1],
                m_min=self.m_min,
                m_max=self.m_max,
                volume=plane_volume,
            )
            n_halos = rng.poisson(n_bar)

            if n_halos > 0:
                masses = sample_halo_masses(
                    n=n_halos,
                    m_min=self.m_min,
                    m_max=self.m_max,
                    A=mf_coeffs[i, 0],
                    B=mf_coeffs[i, 1],
                    seed=rng.randint(0, 2 ** 31),
                )
                positions = sample_positions_in_circle(
                    n=n_halos,
                    radius=r_cone,
                    seed=rng.randint(0, 2 ** 31),
                )
                concentrations = sample_concentrations(
                    log10_masses=np.log10(masses),
                    A_mc=mc_coeffs[i, 0],
                    B_mc=mc_coeffs[i, 1],
                    c_scatter=self.c_scatter,
                    seed=rng.randint(0, 2 ** 31),
                )

                for j in range(n_halos):
                    halo = ag.mp.NFWTruncatedSph.from_m200_concentration(
                        centre=(positions[j, 0], positions[j, 1]),
                        m200_solar_mass=masses[j],
                        concentration=concentrations[j],
                        redshift_halo=z_cen,
                        redshift_source=self.z_source,
                        cosmology=cosmology,
                        truncation_factor=self.truncation_factor,
                    )
                    galaxies.append(
                        ag.Galaxy(redshift=z_cen, mass=halo)
                    )

            kappa_neg = negative_kappa_from(
                z_centre=z_cen,
                comoving_volume_per_arcsec2=vol_depth,
                A_mf=mf_coeffs[i, 0],
                B_mf=mf_coeffs[i, 1],
                A_mc=mc_coeffs[i, 0],
                B_mc=mc_coeffs[i, 1],
                m_min=self.m_min,
                m_max=self.m_max,
                z_source=self.z_source,
                truncation_factor=self.truncation_factor,
                c_scatter=self.c_scatter,
                cosmology=cosmology,
            )
            galaxies.append(
                ag.Galaxy(
                    redshift=z_cen,
                    mass_sheet=ag.mp.MassSheet(kappa=kappa_neg),
                )
            )

        return galaxies
