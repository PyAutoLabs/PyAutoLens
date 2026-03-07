import numpy as np
from typing import List, Optional

import autoarray as aa
import autogalaxy as ag
import autogalaxy.plot as aplt

from autolens import exc


def plane_redshifts_from(galaxies: List[ag.Galaxy]) -> List[float]:
    """
    Returns a list of plane redshifts from a list of galaxies, using the redshifts of the galaxies to determine the
    unique redshifts of the planes.

    Each plane redshift corresponds to a unique redshift in the list of galaxies, such that the returned list of
    redshifts contains no duplicate values. This means multiple galaxies at the same redshift are assigned to the
    same plane.

    For example, if the input is three galaxies, two at redshift 1.0 and one at redshift 2.0, the returned list of
    redshifts would be [1.0, 2.0].

    Parameters
    ----------
    galaxies
        The list of galaxies used to determine the unique redshifts of the planes.

    Returns
    -------
    The list of unique redshifts of the planes.
    """

    galaxies_ascending_redshift = sorted(galaxies, key=lambda galaxy: galaxy.redshift)

    # Coerce to float to avoid issues with other float types not being hashable.
    plane_redshifts = [float(galaxy.redshift) for galaxy in galaxies_ascending_redshift]

    return list(dict.fromkeys(plane_redshifts))


def planes_from(
    galaxies: List[ag.Galaxy], plane_redshifts: Optional[List[float]] = None
) -> List[ag.Galaxies]:
    """
    Returns a list of list of galaxies grouped into their planes, where planes contained all galaxies at the same
    unique redshift.

    Each plane redshift corresponds to a unique redshift in the list of galaxies, such that the returned list of
    redshifts contains no duplicate values. This means multiple galaxies at the same redshift are assigned to the
    same plane.

    If the plane redshifts are not input, the redshifts of the galaxies are used to determine the unique redshifts of
    the planes.

    For example, if the input is three galaxies, two at redshift 1.0 and one at redshift 2.0, the returned list of
    list of galaxies would be [[g1, g2], g3]].

    Parameters
    ----------
    galaxies
        The list of galaxies used to determine the unique redshifts of the planes.
    plane_redshifts
        The redshifts of the planes, which are used to group the galaxies into their respective planes. If not input,
        the redshifts of the galaxies are used to determine the unique redshifts of the planes.

    Returns
    -------
    The list of list of galaxies grouped into their planes.
    """

    galaxies_ascending_redshift = sorted(galaxies, key=lambda galaxy: galaxy.redshift)

    if plane_redshifts is None:
        plane_redshifts = plane_redshifts_from(galaxies=galaxies_ascending_redshift)

    planes = [[] for i in range(len(plane_redshifts))]

    for galaxy in galaxies_ascending_redshift:
        index = (np.abs(np.asarray(plane_redshifts) - galaxy.redshift)).argmin()
        planes[index].append(galaxy)

    for index in range(len(planes)):
        planes[index] = ag.Galaxies(galaxies=planes[index])

    return planes


def traced_grid_2d_list_from(
    planes: List[List[ag.Galaxy]],
    grid: aa.type.Grid2DLike,
    cosmology: ag.cosmo.LensingCosmology = None,
    plane_index_limit: int = Optional[None],
    xp=np,
):
    """
    Returns a ray-traced grid of 2D Cartesian (y,x) coordinates which accounts for multi-plane ray-tracing.

    This uses the redshifts and mass profiles of the galaxies contained within the tracer to perform the multi-plane
    ray-tracing calculation.

    This function returns a list of 2D (y,x) grids, corresponding to each redshift in the input list of planes. The
    plane redshifts are determined from the redshifts of the galaxies in each plane, whereby there is a unique plane
    at each redshift containing all galaxies at the same redshift.

    For example, if the `planes` list contains three lists of galaxies with `redshift`'s z0.5, z=1.0 and z=2.0, the
    returned list of traced grids will contain three entries corresponding to the input grid after ray-tracing to
    redshifts 0.5, 1.0 and 2.0.

    An input cosmology object can change the cosmological model, which is used to compute the scaling
    factors between planes (which are derived from their redshifts and angular diameter distances). It is these
    scaling factors that account for multi-plane ray tracing effects.

    The calculation can be terminated early by inputting a `plane_index_limit`. All planes whose integer indexes are
    above this value are omitted from the calculation and not included in the returned list of grids (the size of
    this list is reduced accordingly).

    For example, if `planes` has 3 lists of galaxies, but `plane_index_limit=1`, the third plane (corresponding to
    index 2) will not be calculated. The `plane_index_limit` is used to avoid uncessary ray tracing calculations
    of higher redshift planes whose galaxies do not have mass profile (and only have light profiles).

    Parameters
    ----------
    galaxies
        The galaxies whose mass profiles are used to perform multi-plane ray-tracing, where the list of galaxies
        has an index for each plane, correspond to each unique redshift in the multi-plane system.
    grid
        The 2D (y, x) coordinates on which multi-plane ray-tracing calculations are performed.
    cosmology
        The cosmology used for ray-tracing from which angular diameter distances between planes are computed.
    plane_index_limit
        The integer index of the last plane which is used to perform ray-tracing, all planes with an index above
        this value are omitted.

    Returns
    -------
    traced_grid_list
        A list of 2D (y,x) grids each of which are the input grid ray-traced to a redshift of the input list of planes.
    """
    cosmology = cosmology or ag.cosmo.Planck15()

    traced_grid_list = []
    traced_deflection_list = []

    redshift_list = [galaxies[0].redshift for galaxies in planes]

    for plane_index, galaxies in enumerate(planes):

        scaled_grid = grid.array

        if plane_index > 0:

            for previous_plane_index in range(plane_index):
                scaling_factor = cosmology.scaling_factor_between_redshifts_from(
                    redshift_0=redshift_list[previous_plane_index],
                    redshift_1=galaxies[0].redshift,
                    redshift_final=redshift_list[-1],
                    xp=xp,
                )

                scaled_deflections = (
                    scaling_factor * traced_deflection_list[previous_plane_index].array
                )

                scaled_grid = scaled_grid - scaled_deflections

        scaled_grid = aa.Grid2DIrregular(
            values=scaled_grid,
        )

        traced_grid_list.append(scaled_grid)

        if plane_index_limit is not None:
            if plane_index == plane_index_limit:
                return traced_grid_list

        deflections_yx_2d = sum(
            (g.deflections_yx_2d_from(grid=scaled_grid, xp=xp) for g in galaxies)
        )

        traced_deflection_list.append(deflections_yx_2d)

    return traced_grid_list


def grid_2d_at_redshift_from(
    redshift: float,
    galaxies: List[ag.Galaxy],
    grid: aa.type.Grid2DLike,
    cosmology: ag.cosmo.LensingCosmology = None,
    xp=np,
) -> aa.type.Grid2DLike:
    """
    Returns a ray-traced grid of 2D Cartesian (y,x) coordinates, which accounts for multi-plane ray-tracing, at a
    specified input redshift which may be different to the redshifts of all planes.

    Given a list of galaxies whose redshifts define a multi-plane lensing system and an input grid of (y,x) arc-second
    coordinates (e.g. an image-plane grid), ray-trace the grid to an input redshift in of the multi-plane system.

    This is performed using multi-plane ray-tracing and a list of galaxies which are converted into a list of planes
    at a set of redshift. The galaxy mass profiles are used to compute deflection angles. Any redshift can be input
    even if a plane does not exist there, including redshifts before the first plane of the lens system.

    An input cosmology object can change the cosmological model, which is used to compute the scaling
    factors between planes (which are derived from their redshifts and angular diameter distances). It is these
    scaling factors that account for multi-plane ray tracing effects.

    There are two ways the calculation may be performed:

    1) If the input redshift is the same as the redshift of a plane in the multi-plane system, the grid is ray-traced
    to that plane and the traced grid returned.

    2) If the input redshift is not the same as the redshift of a plane in the multi-plane system, a plane is inserted
    at this redshift and the grid is ray-traced to this plane.

    For example, the input list `galaxies` may contained three `ag.Galaxy` objects at redshifts z=0.5, z=1.0 and z=2.0.
    We can input an image-plane grid and request that its coordinates are ray-traced to a plane at z=1.75 in this
    multi-plane system. This will insert a plane at z=1.75 and use the galaxy's at z=0.5 and z=1.0 to compute
    deflection angles, alongside accounting for multi-plane lensing effects via the angular diameter distances
    between the different galaxy redshifts.

    Parameters
    ----------
    redshift
        The redshift the input (image-plane) grid is traced too.
    galaxies
        A list of galaxies which make up a multi-plane strong lens ray-tracing system.
    grid
        The 2D (y, x) coordinates which is ray-traced to the input redshift.
    cosmology
        The cosmology used for ray-tracing from which angular diameter distances between planes are computed.
    """
    cosmology = cosmology or ag.cosmo.Planck15()

    plane_redshifts = plane_redshifts_from(galaxies=galaxies)

    if redshift <= plane_redshifts[0]:
        return grid.copy()

    planes = planes_from(galaxies=galaxies, plane_redshifts=plane_redshifts)

    plane_index_with_redshift = [
        plane_index
        for plane_index, galaxies in enumerate(planes)
        if galaxies[0].redshift == redshift
    ]

    if plane_index_with_redshift:
        traced_grid_list = traced_grid_2d_list_from(
            planes=planes, grid=grid, cosmology=cosmology, xp=xp
        )

        return traced_grid_list[plane_index_with_redshift[0]]

    for plane_index, plane_redshift in enumerate(plane_redshifts):
        if redshift > plane_redshift:
            plane_index_insert = plane_index + 1

    planes.insert(plane_index_insert, [ag.Galaxy(redshift=redshift)])

    traced_grid_list = traced_grid_2d_list_from(
        planes=planes, grid=grid, cosmology=cosmology, xp=xp
    )

    return traced_grid_list[plane_index_insert]


def time_delays_from(
    galaxies: List[ag.Galaxy],
    grid: aa.type.Grid2DLike,
    xp=np,
    cosmology: ag.cosmo.LensingCosmology = None,
) -> aa.type.Grid2DLike:
    """
    Returns the gravitational lensing time delay in days for a grid of 2D (y, x) coordinates.

    This function calculates the time delay at each image-plane position due to both geometric and gravitational
    (Shapiro) effects, as described by the Fermat potential, which are computed using the deflection angles of the
    galaxies in the lens system.

    It requires a two-plane system (lens and source), and does not currently support multi-plane time delay
    calculations involving more than two planes, but it could be extended to do so in the future.

    The time delay is computed as:

    .. math::
        \Delta t(\boldsymbol{\theta}) = \frac{D_{\Delta t}}{c} \, \phi(\boldsymbol{\theta})

    where:

    - \( \boldsymbol{\theta} \): image-plane coordinate
    - \( \phi(\boldsymbol{\theta}) \): Fermat potential at each coordinate
    - \( c \): speed of light
    - \( D_{\Delta t} \): time-delay distance

    The time-delay distance is given by:

    .. math::
        D_{\Delta t} = (1 + z_l) \frac{D_d D_s}{D_{ds}}

    with \( D_d, D_s, D_{ds} \) the angular diameter distances to the lens, to the source, and from lens to source.

    The time delay is computed using the Fermat potential, as described by the equations above.

    An input cosmology object can change the cosmological model, which is used to compute the scaling
    factors between planes (which are derived from their redshifts and angular diameter distances). It is these
    scaling factors that account for multi-plane ray tracing effects.

    Parameters
    ----------
    galaxies
        List of galaxies whose mass profiles define the lens and source planes. Must contain exactly two redshifts.
    grid
        The 2D (y, x) image-plane coordinates where the time delay is computed.
    cosmology
        The cosmological model used to calculate angular diameter distances. Defaults to Planck15.

    Returns
    -------
    The time delay at each (y, x) coordinate in the input grid, in units of days.
    """
    cosmology = cosmology or ag.cosmo.Planck15()

    plane_redshifts = plane_redshifts_from(galaxies=galaxies)

    if len(plane_redshifts) != 2:
        raise exc.RayTracingException(
            "The time delay calculation requires exactly two planes, but the input galaxies have "
            f"{len(plane_redshifts)} planes with redshifts {plane_redshifts}."
        )

    z_l, z_s = plane_redshifts[0], plane_redshifts[1]

    # -----------------
    # Constants (SI)
    # -----------------
    kpc_in_m = xp.asarray(3.085677581491367e19)  # kpc in meters
    arcsec_to_rad = xp.pi / 648000.0  # arcsec -> rad (pi / (180*3600))
    seconds_per_day = xp.asarray(86400.0)
    c = xp.asarray(299792458.0)  # m/s

    # This factor converts Fermat potential in arcsec^2 into days once multiplied by D_dt/c
    factor = (arcsec_to_rad * arcsec_to_rad) / seconds_per_day

    # -----------------
    # Angular diameter distances (kpc)
    # -----------------
    Dd_kpc = cosmology.angular_diameter_distance_to_earth_in_kpc_from(z_l, xp=xp)
    Ds_kpc = cosmology.angular_diameter_distance_to_earth_in_kpc_from(z_s, xp=xp)
    Dds_kpc = cosmology.angular_diameter_distance_between_redshifts_in_kpc_from(
        redshift_0=z_l, redshift_1=z_s, xp=xp
    )

    # Time-delay distance in meters: (1+z_l) * Dd * Ds / Dds
    D_dt_m = (1.0 + z_l) * (Dd_kpc * Ds_kpc / Dds_kpc) * kpc_in_m

    # Fermat potential (should be in arcsec^2 for this formula)
    import autogalaxy as ag

    fermat_potential = ag.LensCalc.from_mass_obj(galaxies).fermat_potential_from(
        grid=grid, xp=xp
    )

    # Final time delay in days
    return (D_dt_m / c) * fermat_potential * factor


def ordered_plane_redshifts_with_slicing_from(
    lens_redshifts, planes_between_lenses, source_plane_redshift
):
    """
    Given a set of lens plane redshifts, the source-plane redshift and the number of planes between each, setup the \
    plane redshifts using these values. A lens redshift corresponds to the 'main' lens galaxy(s),
    whereas the slices collect line-of-sight halos over a range of redshifts.

    The source-plane redshift is removed from the ordered plane redshifts that are returned, so that galaxies are not \
    planed at the source-plane redshift.

    For example, if the main plane redshifts are [1.0, 2.0], and the bin sizes are [1,3], the following redshift \
    slices for planes will be used:

    z=0.5
    z=1.0
    z=1.25
    z=1.5
    z=1.75
    z=2.0

    Parameters
    ----------
    lens_redshifts : [float]
        The redshifts of the main-planes (e.g. the lens galaxy), which determine where redshift intervals are placed.
    planes_between_lenses : [int]
        The number of slices between each main plane. The first entry in this list determines the number of slices \
        between Earth (redshift 0.0) and main plane 0, the next between main planes 0 and 1, etc.
    source_plane_redshift
        The redshift of the source-plane, which is input explicitly to ensure galaxies are not placed in the \
        source-plane.
    """

    # Check that the number of slices between lens planes is equal to the number of intervals between the lens planes.
    if len(lens_redshifts) != len(planes_between_lenses) - 1:
        raise exc.PlaneException(
            "The number of lens_plane_redshifts input is not equal to the number of "
            "slices_between_lens_planes+1."
        )

    plane_redshifts = []

    # Add redshift 0.0 and the source plane redshifit to the lens plane redshifts, so that calculation below can use
    # them when dividing slices. These will be removed by the return function at the end from the plane redshifts.

    lens_redshifts.insert(0, 0.0)
    lens_redshifts.append(source_plane_redshift)

    for lens_plane_index in range(1, len(lens_redshifts)):
        previous_plane_redshift = lens_redshifts[lens_plane_index - 1]
        plane_redshift = lens_redshifts[lens_plane_index]
        slice_total = planes_between_lenses[lens_plane_index - 1]
        plane_redshifts += list(
            np.linspace(previous_plane_redshift, plane_redshift, slice_total + 2)
        )[1:]

    return plane_redshifts[0:-1]


def visuals_2d_of_planes_list_from(tracer, grid) -> aplt.Visuals2D:

    visuals_2d_of_planes_list = []

    for plane_index in range(len(tracer.planes)):

        visuals_2d_of_planes_list.append(
            aplt.Visuals2D().add_critical_curves_or_caustics(
                mass_obj=tracer,
                grid=grid,
                plane_index=plane_index,
            )
        )

    return visuals_2d_of_planes_list
