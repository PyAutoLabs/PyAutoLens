"""
The core gravitational-lensing ray-tracing module for **PyAutoLens**.

The central class is ``Tracer``, which groups a list of ``Galaxy`` objects by redshift
into a series of *planes* and performs multi-plane gravitational lensing calculations.
Key capabilities:

- **Multi-plane ray tracing** — images are traced from the source plane through
  intermediate lens planes to the observer using the cosmology-dependent angular
  diameter distances between each pair of planes.
- **Lensed images** — the light of each galaxy is ray-traced to its correct position
  and the contributions from all planes are summed.
- **Lensing maps** — convergence, shear, magnification, deflection angles, and potential
  maps can all be computed on arbitrary ``Grid2D`` grids.
- **Critical curves & caustics** — the image-plane loci where magnification diverges and
  their source-plane counterparts.
- **Lens modeling** — the ``Tracer`` is the core object used by
  ``FitImaging`` / ``FitInterferometer`` / ``FitPointDataset`` when performing a
  Bayesian model fit via a ``PyAutoFit`` non-linear search.
"""
from abc import ABC
import numpy as np
import warnings
from scipy.interpolate import griddata
from typing import Dict, List, Optional, Type, Union

import autofit as af
import autoarray as aa
import autogalaxy as ag

from autoarray import validate

from autogalaxy.profiles.geometry_profiles import GeometryProfile
from autogalaxy.profiles.light.abstract import LightProfile
from autogalaxy.profiles.light.snr import LightProfileSNR
from autogalaxy.galaxy.mass_field import MassField
from autogalaxy.profiles.mass.abstract.abstract import MassProfile
from autogalaxy.profiles.point_sources import Point, PointSolved

from autolens.lens import tracer_util


LENSABLE_CLS = (LightProfile, aa.Pixelization, Point, PointSolved)
"""
Everything a tracer can gravitationally lens.

Emphatically **not** just ``LightProfile``: a source reconstructed by a
``Pixelization`` carries no light profile, and a point source carries neither — yet
both are light to be lensed. Enumerating only light profiles made the redshift-ordering
warning below fire on every pixelized-source and point-source configuration in the test
suite, which is precisely the "would a warning be noise?" failure it must avoid.
"""


class MultiPlaneRedshiftWarning(UserWarning):
    """
    Warned when a tracer's redshifts describe a system in which no light lies behind
    any mass, so nothing in it can be gravitationally lensed.

    This has its own category so it can be silenced with a single filter, without
    suppressing unrelated warnings:

    ::

        import warnings
        from autolens.lens.tracer import MultiPlaneRedshiftWarning

        warnings.filterwarnings("ignore", category=MultiPlaneRedshiftWarning)

    It is a **warning and never an error**, deliberately. Multi-plane ray tracing
    genuinely supports geometries that look wrong under two-plane "lens and source"
    naming, so a configuration this flags may still be exactly what the user intended.
    """


def _warn_if_no_light_is_behind_any_mass(galaxies, fields=None):
    """
    Warn if every light-bearing galaxy lies in front of (or level with) every
    mass-bearing galaxy, across more than one redshift plane.

    In that configuration no light can be deflected by any mass, so the tracer
    produces an unlensed image while looking like a lens model. The usual cause is
    the lens and source redshifts being the wrong way round — the case reported on
    PyAutoLens#532, where ``z_lens=1.0`` with ``z_source=0.5`` returned a finite image
    and said nothing.

    __Why this warns and does not raise__

    Multi-plane lensing legitimately supports geometries that look inverted under
    two-plane naming, so this cannot be an error. The test on the ordering is
    deliberately narrow for the same reason: it fires only when *no* light at all
    sits behind *any* mass. A system with mass at z=1.0 and light at both z=0.5 and
    z=1.5 is a real multi-plane configuration and is not flagged, because some of its
    light is genuinely lensed.

    Single-plane systems (one distinct redshift) are never flagged — a galaxy with its
    own light and mass at one redshift is ordinary PyAutoGalaxy usage, not a mistake.

    __Tracer safety__

    Redshifts can be free model parameters (e.g. a subhalo redshift under
    ``jax.jit``), so only concrete redshifts are compared; a traced redshift makes the
    check skip rather than coerce a traced boolean.

    Parameters
    ----------
    galaxies
        The galaxies the tracer was constructed with.
    fields
        The `MassField` objects the tracer was constructed with, if any. A field is
        mass-bearing exactly like a galaxy's mass profiles, so it participates in the
        geometry check; passing no fields leaves the check byte-identical to the
        galaxies-only behaviour it had before fields existed.
    """
    try:
        galaxy_list = list(galaxies)
    except TypeError:
        return

    if fields is not None:
        try:
            galaxy_list = galaxy_list + list(fields)
        except TypeError:
            return

    mass_redshifts = []
    light_redshifts = []
    all_redshifts = []

    for galaxy in galaxy_list:
        redshift = getattr(galaxy, "redshift", None)

        if not validate.is_concrete_scalar(redshift):
            return

        all_redshifts.append(redshift)

        has = getattr(galaxy, "has", None)

        if has is None:
            return

        is_mass = has(cls=MassProfile)
        is_lensable = any(has(cls=cls) for cls in LENSABLE_CLS)

        # An entirely empty galaxy is a scaffold — a placeholder in a model being
        # composed, or a source not yet filled in. We cannot judge the geometry of a
        # system that is still being built, so say nothing rather than warn someone
        # mid-construction.
        if not is_mass and not is_lensable:
            return

        if is_mass:
            mass_redshifts.append(redshift)

        if is_lensable:
            light_redshifts.append(redshift)

    if not mass_redshifts or not light_redshifts:
        return

    if len(set(all_redshifts)) < 2:
        return

    if max(light_redshifts) > min(mass_redshifts):
        return

    warnings.warn(
        f"No light in this tracer lies behind any mass, so nothing in it is "
        f"gravitationally lensed: the light-bearing galaxies are at redshifts "
        f"{sorted(set(light_redshifts))} and the mass-bearing galaxies at "
        f"{sorted(set(mass_redshifts))}. The usual cause is the lens and source "
        f"redshifts being the wrong way round.\n\n"
        f"This is a warning, not an error — multi-plane ray tracing supports "
        f"geometries that look inverted under two-plane naming, so this may be "
        f"intended. Silence it with:\n\n"
        f"    warnings.filterwarnings('ignore', category=MultiPlaneRedshiftWarning)",
        MultiPlaneRedshiftWarning,
        stacklevel=3,
    )


def _validate_galaxies(galaxies):
    """
    Raise if ``galaxies`` is not something the tracer can treat as a collection of
    galaxies.

    Without this, ``Tracer(galaxies="not a list")`` constructs happily and fails much
    later with ``AttributeError: 'str' object has no attribute 'redshift'`` — an error
    naming nothing the caller passed. Integers, ``None`` and dicts were accepted at
    construction too.

    A string is the trap worth naming: it *is* iterable, so an ``isinstance(x,
    Iterable)`` check alone does not catch it. The element type is what matters, not
    iterability.

    ``af.ModelInstance`` is accepted unexamined — that is how PyAutoFit hands a model's
    galaxies to the tracer during a fit.

    Elements are duck-typed on ``redshift`` rather than required to be exactly
    ``ag.Galaxy``, so mocks and ``Galaxy`` subclasses keep working. ``redshift`` is
    precisely the attribute whose absence produced the original confusing failure.

    No tracer gate is needed here: this validates the *container*. A JAX trace makes a
    galaxy's parameters traced, never the list of galaxies itself.

    Parameters
    ----------
    galaxies
        The input to validate.
    """
    if isinstance(galaxies, af.ModelInstance):
        return

    if isinstance(galaxies, (str, bytes)):
        raise TypeError(
            f"galaxies must be an iterable of Galaxy objects, but a "
            f"{type(galaxies).__name__} was input: {galaxies!r}. A string is itself "
            f"iterable, so this is not caught by an iterability check — pass a list "
            f"of Galaxy objects, e.g. Tracer(galaxies=[lens, source])"
        )

    if not isinstance(galaxies, (list, tuple)):
        raise TypeError(
            f"galaxies must be an iterable of Galaxy objects, but a "
            f"{type(galaxies).__name__} was input: {galaxies!r}. Pass a list of "
            f"Galaxy objects, e.g. Tracer(galaxies=[lens, source])"
        )

    for index, galaxy in enumerate(galaxies):
        if not hasattr(galaxy, "redshift"):
            raise TypeError(
                f"galaxies must be an iterable of Galaxy objects, but the entry at "
                f"index {index} is a {type(galaxy).__name__}: {galaxy!r}. Every entry "
                f"needs a redshift, which is what the ray-tracing calculation groups "
                f"galaxies into planes by"
            )


def _validate_fields(fields):
    """
    Raise if ``fields`` is not something the tracer can treat as a collection of
    ``MassField`` objects.

    A bare ``MassField`` is accepted as a one-element collection. Otherwise this
    mirrors ``_validate_galaxies`` — same container checks, same
    string trap, same ``af.ModelInstance`` escape hatch — but names ``MassField``
    in every message, because the two arguments are not interchangeable and the
    most likely mistake is passing one where the other belongs.

    A ``Galaxy`` in ``fields`` is called out by name: it is a valid lensing object
    in the wrong argument, and saying "not a MassField" would not tell the caller
    where it should have gone.

    Parameters
    ----------
    fields
        The input to validate.
    """
    if isinstance(fields, af.ModelInstance):
        return

    if isinstance(fields, MassField):
        fields = [fields]

    if isinstance(fields, (str, bytes)):
        raise TypeError(
            f"fields must be an iterable of MassField objects, but a "
            f"{type(fields).__name__} was input: {fields!r}. A string is itself "
            f"iterable, so this is not caught by an iterability check — pass a list "
            f"of MassField objects, e.g. Tracer(galaxies=[lens, source], "
            f"fields=[field])"
        )

    if not isinstance(fields, (list, tuple)):
        raise TypeError(
            f"fields must be an iterable of MassField objects, but a "
            f"{type(fields).__name__} was input: {fields!r}. Pass a list of "
            f"MassField objects, e.g. Tracer(galaxies=[lens, source], fields=[field])"
        )

    for index, field in enumerate(fields):
        if isinstance(field, ag.Galaxy):
            raise TypeError(
                f"fields must be an iterable of MassField objects, but the entry at "
                f"index {index} is a Galaxy: {field!r}. Galaxies go in the galaxies "
                f"argument, e.g. Tracer(galaxies=[lens, source], fields=[field])"
            )

        if not hasattr(field, "redshift"):
            raise TypeError(
                f"fields must be an iterable of MassField objects, but the entry at "
                f"index {index} is a {type(field).__name__}: {field!r}. Every entry "
                f"needs a redshift, which is what the ray-tracing calculation groups "
                f"the tracer's members into planes by"
            )


class Tracer(ABC, ag.OperateImageGalaxies):
    def __init__(
        self,
        galaxies: Union[List[ag.Galaxy], af.ModelInstance],
        cosmology: ag.cosmo.LensingCosmology = None,
        fields: Optional[Union[MassField, List[MassField], af.ModelInstance]] = None,
    ):
        """
        Performs gravitational lensing ray-tracing calculations based on an input list of galaxies and a cosmology.

        The tracer stores the input galaxies in their input order, which may not be in ascending redshift order.
        However, for all ray-tracing calculations, the tracer orders the input galaxies in ascending order of redshift,
        as this is required for the multi-plane ray-tracing calculations.

        The tracer then creates a series of planes, where each plane is a collection of galaxies at the same redshift.

        The redshifts of these planes are determined by the redshifts of the galaxies, such that there is a unique
        plane redshift for every unique galaxy redshift (galaxies with identical redshifts are put in the same plane).

        Gravitational lensing calculations are then performed individually for each plane and combined to produce the
        correct overall lensing calculation. This includes the calculations like the deflection angles, create
        images of the galaxies at different planes, and the overall lensed image of all galaxies.

        Multi-plane ray-tracing work natively, whereby the redshifts of the planes are used to perform multi-plane
        ray-tracing calculations. This uses the input cosmology so that deflection-angles are rescaled according to
        the lens-geometry of the multi-plane system.

        The `Tracer` object is also the core of the lens modeling API, whereby a model tracer is created via
        the `PyAutoFit` `af.Model` object.

        Parameters
        ----------
        galaxies
            The list of galaxies which make up the gravitational lensing ray-tracing system.
        cosmology
            The cosmology used to perform ray-tracing calculations.
        fields
            The `MassField` objects describing the tidal field of everything outside the modelled system (e.g. an
            external shear, a mass sheet, an external potential). Pass a list or a single
            `MassField`; the tracer stores a fresh list in either case. Each is placed at its own redshift in the
            multi-plane calculation exactly like a galaxy's mass, but contributes no light. Attaching these
            components to a `Galaxy` instead remains fully supported and is not deprecated.
        """

        # if isinstance(galaxies, af.ModelInstance):
        #     galaxies = list(galaxies.values())

        _validate_galaxies(galaxies=galaxies)

        if fields is not None:
            _validate_fields(fields=fields)

        fields = tracer_util.fields_list_from(fields)

        _warn_if_no_light_is_behind_any_mass(galaxies=galaxies, fields=fields)

        self.galaxies = galaxies

        # Stored as a plain list (never None) so every member-walking property below can concatenate it
        # unconditionally, and so a tracer with no fields serialises as `fields: []` rather than a null.
        self.fields = fields

        self.cosmology = cosmology or ag.cosmo.Planck15()

    @property
    def members(self) -> List:
        """
        Returns every object the tracer places in a plane: its galaxies followed by its fields.

        A `MassField` is not a galaxy and never appears in `galaxies`, but it *is* a redshift-bearing,
        mass-bearing member of the lensing system. Every lensing-quantity calculation (deflections,
        convergence, potential, ray-tracing) therefore runs over `members`, whereas every surface which
        enumerates *galaxies* (e.g. the per-galaxy image dictionary) runs over `galaxies` alone.

        Returns
        -------
        The galaxies and fields of the tracer, in input order, galaxies first.
        """
        return list(self.galaxies) + list(self.fields)

    @property
    def galaxies_ascending_redshift(self) -> List[ag.Galaxy]:
        """
        Returns the galaxies in the tracer in ascending redshift order.

        Multi-plane ray tracing calculations begin from the first lowest redshift plane and perform calculations in
        planes of increasing redshift. Thus, the galaxies are sorted by redshift in ascending order to aid this
        calculation.

        When any galaxy has a JAX-traced ``redshift`` (e.g. a free-parameter
        subhalo redshift under ``jax.jit``), Python's ``sorted`` cannot order
        the list because the key comparator would coerce a traced boolean.
        In that case, we trust the input order — the caller (typically
        ``af.Collection(galaxies=...)``) is expected to declare galaxies in
        ascending-redshift order, with each traced-redshift galaxy placed at
        its intended plane position. See ``tracer_util.plane_redshifts_from``
        for the matching rule.

        Returns
        -------
        The galaxies in the tracer in ascending redshift order.
        """
        if not tracer_util._any_traced(self.galaxies):
            return sorted(self.galaxies, key=lambda galaxy: galaxy.redshift)

        return list(self.galaxies)

    @property
    def members_ascending_redshift(self) -> List:
        """
        Returns every member of the tracer -- its galaxies and its fields -- in ascending redshift order.

        This is the list the plane calculation is built from, so that a `MassField` sits in the plane at its
        own redshift and contributes its mass to the multi-plane ray-tracing exactly as a galaxy's mass does.

        `galaxies_ascending_redshift` keeps its galaxies-only meaning and is unchanged; the two properties
        agree whenever a tracer has no fields.

        The traced-redshift bypass matches `galaxies_ascending_redshift`: if any member has a JAX-traced
        `redshift`, the input order is trusted (galaxies first, then fields), because a traced value cannot
        be used as a sort key.

        Returns
        -------
        The galaxies and fields in the tracer in ascending redshift order.
        """
        members = self.members

        if not tracer_util._any_traced(members):
            return sorted(members, key=lambda member: member.redshift)

        return members

    @property
    def plane_redshifts(self) -> List[float]:
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
        # Members, not galaxies: a `MassField` at a redshift no galaxy occupies is a plane of its own.
        return tracer_util.plane_redshifts_from(
            galaxies=self.members_ascending_redshift
        )

    @property
    def planes(self) -> List[ag.Galaxies]:
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
        # Members, not galaxies: a plane may hold a `MassField` beside its galaxies, so that its mass enters
        # every plane-level lensing calculation while every light calculation sees the field's zeros.
        return tracer_util.planes_from(
            galaxies=self.members_ascending_redshift,
            plane_redshifts=self.plane_redshifts,
        )

    @classmethod
    def sliced_tracer_from(
        cls,
        lens_galaxies: List[ag.Galaxy],
        line_of_sight_galaxies: List[ag.Galaxy],
        source_galaxies: List[ag.Galaxy],
        planes_between_lenses: List[int],
        cosmology: ag.cosmo.LensingCosmology = None,
        fields: Optional[Union[MassField, List[MassField]]] = None,
    ):
        """
        Returns a tracer where the lens system is split into planes with specified redshift distances between them.

        This is used for ray-tracing systems with many galaxies at different redshifts (e.g. hundreds or more). If
        each galaxy redshift is treated indepedently, this would require many planes to be created, and the multi-plane
        ray-tracing calculation would be computationally slow.

        To speed the calculation up, the galaxies are grouped into planes with redshifts separated by the inputs.
        To achieve this, the galaxies have their redshifts reassigned from their original values to the nearest
        value of a sliced plane redshift. This ensures that every galaxy is in a subset of planes.

        The redshifts of the planes are determines as follows:

        - Use the redshifts of the lens galaxies to determine the redshifts of the planes, where a lens galaxy is
        expected to have a large mass and thus contribute to a significant portion of the overall lensing. This ensures
        the main lens galaxies have a redshift and plane to themselves, ensuring calculation accuracy.

        - Use the redshift of the source galaxies to determine the redshift of the source plane, ensuring the source
        galaxies also have a dedicated redshift and plane for calculation accuracy.

        - Create N planes between Earth and the first lens galaxy, the lens galaxy and the next lens galaxy (and so on)
        up to the source galaxy. The number of planes between each set of galaxies is specified by the input
        `planes_between_lenses`, where for a lens / source system `planes_between_lenses=[2,3]` would mean there are 2
        planes between Earth and the lens galaxy and 3 planes between the lens and source galaxy.

        - The `line_of_sight_galaxies` are placed in the planes corresponding to their closest redshift.

        Parameters
        ----------
        lens_galaxies
            The lens galaxies in the ray-tracing calculation. Most use cases will have only one lens galaxy, but the
            API supports multiple lens galaxies (e.g. double Einstein ring systems).
        line_of_sight_galaxies
            The galaxies in the line-of-sight to the primary lens galaxy, which may have many different redshifts
            and therefore create computational expensive multi-plane ray-tracing calculations without the plane
            grouping provided by this method.
        source_galaxies
            The source galaxies in the ray-tracing calculation. The API only supports one source galaxy (input multiple
            lens galaxies to build a multi-plane system).
        planes_between_lenses
            The number of slices between each main plane. The first entry in this list determines the number of slices
            between Earth (redshift 0.0) and the first lens galaxy, the next between the lens and source, etc.
        cosmology
            The cosmology used to perform ray-tracing calculations.
        fields
            The `MassField` objects describing the external tidal field. Each has its redshift snapped to the
            nearest sliced plane redshift, exactly as the `line_of_sight_galaxies` do.
            Accepts a list or a single `MassField`.
        """
        cosmology = cosmology or ag.cosmo.Planck15()

        lens_redshifts = tracer_util.plane_redshifts_from(galaxies=lens_galaxies)

        plane_redshifts = tracer_util.ordered_plane_redshifts_with_slicing_from(
            lens_redshifts=lens_redshifts,
            planes_between_lenses=planes_between_lenses,
            source_plane_redshift=source_galaxies[0].redshift,
        )

        plane_redshifts.append(source_galaxies[0].redshift)

        galaxies = lens_galaxies + line_of_sight_galaxies + source_galaxies

        fields = tracer_util.fields_list_from(fields)

        # Fields are snapped exactly as `line_of_sight_galaxies` are: a field is a redshift-bearing member of
        # the system, so leaving it off a sliced plane would defeat the whole point of the slicing.
        for galaxy in galaxies + fields:
            redshift_differences = list(
                map(lambda z: abs(z - galaxy.redshift), plane_redshifts)
            )
            galaxy.redshift = plane_redshifts[
                redshift_differences.index(min(redshift_differences))
            ]

        return Tracer(galaxies=galaxies, cosmology=cosmology, fields=fields)

    @property
    def total_planes(self) -> int:
        return len(self.plane_redshifts)

    @aa.decorators.to_grid
    def traced_grid_2d_list_from(
        self, grid: aa.type.Grid2DLike, xp=np, plane_index_limit: Optional[int] = None
    ) -> List[aa.type.Grid2DLike]:
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

        see `autolens.lens.tracer.tracer_util.traced_grid_2d_list_from()` for the full calculation.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates on which multi-plane ray-tracing calculations are performed.
        plane_index_limit
            The integer index of the last plane which is used to perform ray-tracing, all planes with an index above
            this value are omitted.

        Returns
        -------
        traced_grid_list
            A list of 2D (y,x) grids each of which are the input grid ray-traced to a redshift of the input list of
            planes.
        """

        grid_2d_list = tracer_util.traced_grid_2d_list_from(
            planes=self.planes,
            grid=grid,
            cosmology=self.cosmology,
            plane_index_limit=plane_index_limit,
            xp=xp,
        )

        if isinstance(grid, aa.Grid2D):
            # At a uniform over sample size of 1 every sub pixel is the pixel itself, so
            # `grid.over_sampled` is the slim grid in the same order and tracing it again
            # doubles every deflection angle calculation for a bit-identical result. The
            # over sample size is host numpy, so this branch is static at JAX trace time.

            sub_size_all_one = bool(np.all(np.asarray(grid.over_sample_size.array) == 1))

            if sub_size_all_one:
                grid_2d_over_sampled_list = [
                    aa.Grid2DIrregular(values=grid_2d.array, xp=xp)
                    for grid_2d in grid_2d_list
                ]
            else:
                grid_2d_over_sampled_list = tracer_util.traced_grid_2d_list_from(
                    planes=self.planes,
                    grid=grid.over_sampled,
                    cosmology=self.cosmology,
                    plane_index_limit=plane_index_limit,
                    xp=xp,
                )

            grid_2d_new_list = []

            for i in range(len(grid_2d_list)):
                grid_2d_new = aa.Grid2D(
                    values=grid_2d_list[i],
                    mask=grid.mask,
                    over_sampled=grid_2d_over_sampled_list[i],
                    over_sample_size=grid.over_sample_size,
                    over_sampler=grid.over_sampler,
                )

                grid_2d_new_list.append(grid_2d_new)

            return grid_2d_new_list

        return grid_2d_list

    def grid_2d_at_redshift_from(
        self, grid: aa.type.Grid2DLike, redshift: float
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

        For example, the input list `galaxies` may contained three `Galaxy` objects at redshifts z=0.5, z=1.0 and z=2.0.
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
        # A lensing-quantity path: the deflections used to trace to the input redshift must include the
        # tracer's fields, so this walks members rather than galaxies.
        return tracer_util.grid_2d_at_redshift_from(
            redshift=redshift,
            galaxies=self.members_ascending_redshift,
            grid=grid,
            cosmology=self.cosmology,
        )

    @property
    def upper_plane_index_with_light_profile(self) -> int:
        """
        Returns the index of the highest redshift plane in the tracer which has a light profile.

        When computing the image of a tracer, we only need to trace rays to the highest redshift plane which has a
        light profile. This upper index is therefore used to do this, and ensure faster computation by avoiding
        ray-tracing to planes which do not have light profiles.

        Returns
        -------
        The index of the highest redshift plane in the tracer which has a light profile.
        """
        return max(
            [
                (
                    plane_index
                    if any([galaxy.has(cls=ag.LightProfile) for galaxy in galaxies])
                    else 0
                )
                for (plane_index, galaxies) in enumerate(self.planes)
            ]
        )

    def image_2d_list_from(
        self,
        grid: aa.type.Grid2DLike,
        xp=np,
        operated_only: Optional[bool] = None,
    ) -> List[aa.Array2D]:
        """
        Returns a list of the 2D images for each plane from a 2D grid of Cartesian (y,x) coordinates.

        The image of each plane is computed by ray-tracing the grid using te mass profiles of each galaxies and then
        summing the images of all galaxies in that plane. If a plane has no galaxies, or if the galaxies in a plane
        has no light profiles, a numpy array of zeros is returned.

        For example, if the tracer's planes contain galaxies at redshifts z=0.5, z=1.0 and z=2.0, and the galaxies
        at redshifts z=0.5 and z=1.0 have light and mass profiles, the returned list of images will be the image of the
        galaxies at z=0.5 and z=1.0, where the image at redshift z=1.0 will include the lensing effects of the galaxies
        at z=0.5. The image at redshift z=2.0 will be a numpy array of zeros.

        The `plane_index` input is used to return a specific image of a plane, as opposed to a list of images
        of all planes. This can save on computational time when only the image of a specific plane is needed,
        and is used to perform iterative over-sampling calculations.

        The images output by this function do not include instrument operations, such as PSF convolution (for imaging
        data) or a Fourier transform (for interferometer data).

        Inherited methods in the `autogalaxy.operate.image` package can apply these operations to the images.
        These functions may have the `operated_only` input passed to them, which is why this function includes
        the `operated_only` input.

        If the `operated_only` input is included, the function omits light profiles which are parents of
        the `LightProfileOperated` object, which signifies that the light profile represents emission that has
        already had the instrument operations (e.g. PSF convolution, a Fourier transform) applied to it and therefore
        that operation is not performed again.

        See the `autogalaxy.profiles.light` package for details of how images are computed from a light
        profile.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates where values of the image are evaluated.
        operated_only
            The returned list from this function contains all light profile images, and they are never operated on
            (e.g. via the imaging PSF). However, inherited methods in the `autogalaxy.operate.image` package can
            apply these operations to the images, which may have the `operated_only` input passed to them. This input
            therefore is used to pass the `operated_only` input to these methods.
        """

        traced_grid_list = self.traced_grid_2d_list_from(
            grid=grid,
            xp=xp,
            plane_index_limit=self.upper_plane_index_with_light_profile,
        )

        image_2d_list = []

        for plane_index in range(len(traced_grid_list)):
            galaxies = self.planes[plane_index]

            image_2d = sum(
                [
                    galaxy.image_2d_from(
                        grid=traced_grid_list[plane_index],
                        operated_only=operated_only,
                        xp=xp,
                    )
                    for galaxy in galaxies
                ]
            )

            image_2d_list.append(image_2d)

        if self.upper_plane_index_with_light_profile < self.total_planes - 1:
            if isinstance(grid, aa.Grid2D):
                image_2d = aa.Array2D(
                    values=np.zeros(shape=grid.shape[0]), mask=grid.mask
                )
            else:
                image_2d = aa.ArrayIrregular(values=np.zeros(grid.shape[0]))

            for plane_index in range(
                self.upper_plane_index_with_light_profile, self.total_planes - 1
            ):
                image_2d_list.append(image_2d)

        return image_2d_list

    @aa.decorators.to_array
    def image_2d_from(
        self,
        grid: aa.type.Grid2DLike,
        xp=np,
        operated_only: Optional[bool] = None,
    ) -> aa.Array2D:
        """
        Returns the 2D image of this ray-tracing strong lens system from a 2D grid of Cartesian (y,x) coordinates.

        This function first computes the image of each plane in the tracer, via the function `image_2d_list_from`. The
        images are then summed to give the overall image of the tracer.

        Refer to the function `image_2d_list_from` for a full description of the calculation and how the `operated_only`
        input is used.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates where values of the image are evaluated.
        operated_only
            The returned list from this function contains all light profile images, and they are never operated on
            (e.g. via the imaging PSF). However, inherited methods in the `autogalaxy.operate.image` package can
            apply these operations to the images, which may have the `operated_only` input passed to them. This input
            therefore is used to pass the `operated_only` input to these methods.
        """
        return sum(
            self.image_2d_list_from(grid=grid, operated_only=operated_only, xp=xp)
        )

    def image_2d_via_input_plane_image_from(
        self,
        grid: aa.type.Grid2DLike,
        plane_image: aa.Array2D,
        xp=np,
        plane_index: int = -1,
        include_other_planes: bool = True,
    ) -> aa.Array2D:
        """
        Returns the lensed image of a plane or galaxy, where the input image is uniform and interpolated to compute
        the lensed image.

        The typical use case is inputting the image of an irregular galaxy in the source-plane (whose values are
        on a uniform array) and using this function computing the lensed image of this source galaxy.

        By default, this function computes the lensed image of the final plane, which is the source-plane, by using
        `plane_index=-1`. For multi-plane lens systems, the lensed image of any planes can be computed by setting
        `plane_index` to the index of the plane in the lens system.

        The emission of all other planes and galaxies can be included or omitted setting the `include_other_planes`
        bool. If there are multiple planes in a multi-plane lens system, the emission of the other planes are fully
        lensed.

        __Source Plane Interpolation__

        We use the scipy interpolation function `griddata` to create the lensed source galaxy image.

        In brief, we trace light rays to the source plane and calculate values based on where those light rays land in
        the source plane via interpolation.

        In more detail:

        - `points`: The 2D grid of (y,x) coordinates representing the location of every pixel of the source galaxy
          image in the source-plane, from which we are creating the lensed source image. These coordinates are the
          uniform source-plane grid computed after interpolating the irregular mesh the original source reconstruction
          used.

        - `values`: The intensity values of the source galaxy image which is used to create the lensed source image.
           These values are the flux values of the interpolated source galaxy image computed after interpolating the
           irregular mesh the original source reconstruction used.

        - `xi`: The image-plane grid ray traced to the source-plane. This evaluates the flux of each image-plane
          lensed source-pixel by ray-tracing it to the source-plane grid and computing its value by interpolating the
          source galaxy image.

        Parameters
        ----------
        grid
            The image-plane grid which is traced to the plane where the image is computed, where these values are
            used to perform the interpolation.
        plane_image
            The image of the plane or galaxy which is interpolated to compute the lensed image.
        plane_index
            The index of the plane the image is computed, where the default (-1) computes the image in the last plane
            and therefore the source-plane.

        Returns
        -------
        The lensed image of the plane or galaxy computed by interpolating its image to the image-plane.
        """

        plane_grid = aa.Grid2D.uniform(
            shape_native=plane_image.shape_native,
            pixel_scales=plane_image.pixel_scales,
        )

        traced_grid = self.traced_grid_2d_list_from(
            grid=grid, plane_index_limit=plane_index, xp=xp
        )[plane_index]

        image = griddata(
            points=plane_grid.array,
            values=plane_image.array,
            xi=traced_grid.over_sampled.array,
            fill_value=0.0,
            method="linear",
        )

        if isinstance(grid, aa.Grid2D):
            image = grid.over_sampler.binned_array_2d_from(array=image, xp=xp)

        if include_other_planes:
            image_list = self.image_2d_list_from(grid=grid, xp=xp, operated_only=False)

            if plane_index < 0:
                plane_index = self.total_planes + plane_index

            for plane_lp_index in range(self.total_planes):
                if plane_lp_index != plane_index:
                    image += image_list[plane_lp_index]

        return aa.Array2D(values=image, mask=grid.mask, xp=xp)

    def galaxy_image_2d_dict_from(
        self, grid: aa.type.Grid2DLike, xp=np, operated_only: Optional[bool] = None
    ) -> Dict[ag.Galaxy, np.ndarray]:
        """
        Returns a dictionary associating every `Galaxy` object in the `Tracer` with its corresponding 2D image, using
        the instance of each galaxy as the dictionary keys.

        This object is used for adaptive-features, which use the image of each galaxy in a model-fit in order to
        adapt quantities like a pixelization or regularization scheme to the surface brightness of the galaxies being
        fitted.

        By inheriting from `OperateImageGalaxies` functions which apply operations of this dictionary are accessible,
        for example convolving every image with a PSF or applying a Fourier transform to create a galaxy-visibilities
        dictionary.

        Parameters
        ----------
        grid
            The 2D (y,x) coordinates of the (masked) grid, in its original geometric reference frame.

        Returns
        -------
        A dictionary associated every galaxy in the tracer with its corresponding 2D image.
        """

        galaxy_image_2d_dict = dict()

        traced_grid_list = self.traced_grid_2d_list_from(grid=grid, xp=xp)

        for plane_index, galaxies in enumerate(self.planes):
            # A plane may hold a `MassField` beside its galaxies. This dictionary is keyed by *galaxy*, and
            # consumed by adaptive features and per-galaxy visualisation, so a field -- whose image is zeros
            # by construction -- is skipped rather than added as a key with a zeros image.
            plane_galaxies = [
                galaxy for galaxy in galaxies if not isinstance(galaxy, MassField)
            ]

            image_2d_list = [
                galaxy.image_2d_from(
                    grid=traced_grid_list[plane_index],
                    operated_only=operated_only,
                    xp=xp,
                )
                for galaxy in plane_galaxies
            ]

            for galaxy_index, galaxy in enumerate(plane_galaxies):
                galaxy_image_2d_dict[galaxy] = image_2d_list[galaxy_index]

        return galaxy_image_2d_dict

    @aa.decorators.to_vector_yx
    def deflections_yx_2d_from(
        self, grid: aa.type.Grid2DLike, xp=np
    ) -> Union[aa.VectorYX2D, aa.VectorYX2DIrregular]:
        """
        Returns the 2D deflection angles of all galaxies in the tracer, from the image-plane to the source-plane,
        accounting for multi-plane ray tracing and from a 2D grid of Cartesian (y,x) coordinates.

        The multi-plane ray tracing calculations are performed in the function `traced_2d_grid_list_from` and its
        sub-functions in the `tracer_util` module. This includes performing recursive ray-tracing between planes
        based on the planes redshifts and using the cosmological distances between them to scale the deflection angles.
        Users should refer to these functions for details on how the ray-tracing is performed.

        This function simply computes the corresponding multi-plane deflection angles by subtracting the image-plane
        grid (e.g. before lensing) from the source-plane grid (e.g. after lensing).

        If there is only one plane in the tracer, the deflections are computed by summation of the deflections of all
        galaxies in that plane. This is identical too, but computationally faster than, using the multi-plane
        ray-tracing calculation.

        See the `autogalaxy.profiles.mass` package for details of how deflections are computed from a mass profile.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates where values of the deflections are evaluated.
        """
        if self.total_planes > 1:
            return self.deflections_between_planes_from(grid=grid, xp=xp)
        return self.deflections_of_planes_summed_from(grid=grid, xp=xp)

    @aa.decorators.to_vector_yx
    def deflections_of_planes_summed_from(
        self, grid: aa.type.Grid2DLike, xp=np
    ) -> Union[aa.VectorYX2D, aa.VectorYX2DIrregular]:
        """
        Returns the summed 2D deflections angles of all galaxies in the tracer, not accounting for multi-plane ray
        tracing, from a 2D grid of Cartesian (y,x) coordinates.

        The deflections of each plane is computed by summing the deflections of all galaxies in that plane. If a
        plane has no galaxies, or if the galaxies in a plane has no mass profiles, a numpy array of zeros is returned.

        This calculation does not account for multi-plane ray-tracing effects, it is simply the sum of the deflections
        of all galaxies. The function `deflections_between_planes_from` performs the calculation whilst
        accounting for multi-plane ray-tracing effects.

        For example, if the tracer's planes contain galaxies at redshifts z=0.5, z=1.0 and z=2.0, and the galaxies
        at redshifts z=0.5 and z=1.0 have mass profiles, the returned deflections will be the sum of the deflections
        of the galaxies at z=0.5 and z=1.0.

        The deflections of a tracer do not depend on ray-tracing between grids. This is why the deflections of the
        tracer is the sum of the deflections of all planes, and does not need to account for multi-plane ray-tracing
        effects (in the way that deflection angles and images do).

        See the `autogalaxy.profiles.mass` package for details of how deflections are computed from a mass profile.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates where values of the deflections are evaluated.
        """
        # Members, not galaxies: the deflections of the tracer are the deflections of every mass it holds,
        # and a `MassField`'s mass is mass.
        return sum(
            [galaxy.deflections_yx_2d_from(grid=grid, xp=xp) for galaxy in self.members]
        )

    @aa.decorators.to_vector_yx
    def deflections_between_planes_from(
        self, grid: aa.type.Grid2DLike, xp=np, plane_i=0, plane_j=-1
    ) -> Union[aa.VectorYX2D, aa.VectorYX2DIrregular]:
        """
        Returns the summed 2D deflections angles between two input planes in the tracer, accounting for multi-plane
        ray tracing, from a 2D grid of Cartesian (y,x) coordinates.

        The multi-plane ray tracing calculations are performed in the function `traced_2d_grid_list_from` and its
        sub-functions in the `tracer_util` module. This includes performing recursive ray-tracing between planes
        based on the planes redshifts and using the cosmological distances between them to scale the deflection angles.
        Users should refer to these functions for details on how the ray-tracing is performed.

        This function simply computes the corresponding multi-plane deflection angles by subtracting the grid
        of index `plane_i` to that of index `plane_j`. The default inputs subtract the image-plane grid `plane_i=0`
        (e.g. before lensing) from the source-plane grid `plane_j=-1` (e.g. after lensing).

        See the `autogalaxy.profiles.mass` package for details of how deflections are computed from a mass profile.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates where values of the deflections are evaluated.
        """

        traced_grids_list = self.traced_grid_2d_list_from(grid=grid, xp=xp)

        return traced_grids_list[plane_i] - traced_grids_list[plane_j]

    @aa.decorators.to_array
    def convergence_2d_from(self, grid: aa.type.Grid2DLike, xp=np) -> aa.Array2D:
        """
        Returns the summed 2D convergence of all galaxies in the tracer from a 2D grid of Cartesian (y,x) coordinates.

        The convergence of each plane is computed by summing the convergences of all galaxies in that plane. If a
        plane has no galaxies, or if the galaxies in a plane has no mass profiles, a numpy array of zeros is returned.

        For example, if the tracer's planes contain galaxies at redshifts z=0.5, z=1.0 and z=2.0, and the galaxies
        at redshifts z=0.5 and z=1.0 have mass profiles, the returned convergence will be the sum of the convergences
        of the galaxies at z=0.5 and z=1.0.

        The convergences of a tracer do not depend on ray-tracing between grids. This is why the convergence of the
        tracer is the sum of the convergences of all planes, and does not need to account for multi-plane ray-tracing
        effects (in the way that deflection angles and images do).

        See the `autogalaxy.profiles.mass` package for details of how convergences are computed from a mass profile.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates where values of the convergence are evaluated.
        """
        # Members, not galaxies: a field's convergence (e.g. a `MassSheet`) is part of the tracer's convergence.
        return sum(
            [galaxy.convergence_2d_from(grid=grid, xp=xp) for galaxy in self.members]
        )

    @aa.decorators.to_array
    def potential_2d_from(self, grid: aa.type.Grid2DLike, xp=np) -> aa.Array2D:
        """
        Returns the summed 2D potential of all galaxies in the tracer from a 2D grid of Cartesian (y,x) coordinates.

        The potential of each plane is computed by summing the potentials of all galaxies in that plane. If a
        plane has no galaxies, or if the galaxies in a plane has no mass profiles, a numpy array of zeros is returned.

        For example, if the tracer's planes contain galaxies at redshifts z=0.5, z=1.0 and z=2.0, and the galaxies
        at redshifts z=0.5 and z=1.0 have mass profiles, the returned potential will be the sum of the potentials
        of the galaxies at z=0.5 and z=1.0.

        The potentials of a tracer do not depend on ray-tracing between grids. This is why the potential of the
        tracer is the sum of the potentials of all planes, and does not need to account for multi-plane ray-tracing
        effects (in the way that deflection angles and images do).

        See the `autogalaxy.profiles.mass` package for details of how potentials are computed from a mass profile.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates where values of the potential are evaluated.
        """
        # Members, not galaxies: a field's potential (e.g. an `ExternalPotential`) is part of the tracer's.
        return sum(
            [galaxy.potential_2d_from(grid=grid, xp=xp) for galaxy in self.members]
        )

    @aa.decorators.to_array
    def time_delays_from(self, grid: aa.type.Grid2DLike, xp=np) -> aa.Array2D:
        """
        Returns the gravitational lensing time delay in days, for a grid of 2D (y, x) coordinates.

        This function calculates the time delay at each image-plane position due to both geometric and gravitational
        (Shapiro) effects, as described by the Fermat potential, which are computed using the deflection angles of the
        galaxies in the lens system.

        Time dleays are computed from a reference point, which is the delay one would compute without a mass model.
        This means a time delay could be negative, which means the light travel time is faster when lensing is
        accounted for. When performing fitting, all time-delays are subtracted by the time-delay of the
        shortest time-delay, such that its value is zero and all other time-delays are positive.

        A full description of the calculation is given in the `autolens.lens.tracer.tracer_util.time_delays_from`
        function, which performs the calculation and has full latex documentation of the equations used.
        """
        return tracer_util.time_delays_from(
            # Members, not galaxies: a time delay is a lensing quantity computed from the tracer's full mass.
            galaxies=ag.Galaxies(self.members_ascending_redshift),
            grid=grid,
            xp=xp,
            cosmology=self.cosmology,
        )

    def has(self, cls: Type) -> bool:
        """
        Returns a bool specifying whether this tracer has a galaxy with a certain class type.

        For example, for the input `cls=ag.LightProfile`, this function returns True if any galaxy in the tracer has a
        light profile and false if no galaxy has a light profile.

        This function is used to check for mass profiles and specific types of profiles, like the linear light profile.

        Parameters
        ----------
        cls
            The class type of the galaxy which is checked for in the tracer.

        Returns
        -------
        True if any galaxy in the tracer has the input class type, else False.
        """
        # Members, not galaxies: `tracer.has(cls=al.mp.ExternalShear)` must be True for a shear held by a
        # field, exactly as it is for a shear attached to a galaxy.
        return any(map(lambda galaxy: galaxy.has(cls=cls), self.members))

    def cls_list_from(self, cls: Type) -> List:
        """
        Returns a list of objects in the tracer which are an instance of the input `cls`.

        For example:

        - If the input is `cls=ag.LightProfile`, a list containing all light profiles in the tracer is returned.

        Returns
        -------
            The list of objects in the tracer that inherit from input `cls`.
        """
        cls_list = []

        # Members, not galaxies: this lists *profiles*, and a field's profiles are profiles of the tracer.
        for galaxy in self.members:
            if galaxy.has(cls=cls):
                for cls_galaxy in galaxy.cls_list_from(cls=cls):
                    cls_list.append(cls_galaxy)

        return cls_list

    def plane_index_via_redshift_from(self, redshift: float) -> Optional[int]:
        """
        Returns the index of a plane at a given redshift.

        This is used to determine the index of a plane in the tracer, which is useful for multi-plane ray-tracing
        calculations. The index of the plane may, for example, be used to extract grid of a specific plane in the
        tracer after multi-plane ray-tracing calculations.

        A tolerance of 1e-8 is used to determine if the input redshift is close to the redshift of a plane. If
        no matching plane is found, None is returned.

        Parameters
        ----------
        redshift
            The redshift of the plane to find the index of.

        Returns
        -------
            The index of the plane that matches the input redshift.
        """

        for plane_index, plane_redshift in enumerate(self.plane_redshifts):
            if np.isclose(redshift, plane_redshift, atol=1e-8):
                return plane_index

    @property
    def plane_indexes_with_pixelizations(self) -> List[int]:
        """
        Returns a list of integer indexes of the indexes of planes which use a `Pixelization` to reconstruct the
        source galaxy.

        This list is used to set up an inversion, whereby each pixelization is extracted from the tracer with its
        corresponding ray-traced grid and passed to the PyAutoArray `inversion` module.

        Returns
        -------
        The list of integer indexes of the planes which use a `Pixelization` to reconstruct the source galaxy.
        """
        plane_indexes_with_inversions = [
            plane_index if plane.has(cls=aa.Pixelization) else None
            for (plane_index, plane) in enumerate(self.planes)
        ]
        return [
            plane_index
            for plane_index in plane_indexes_with_inversions
            if plane_index is not None
        ]

    @property
    def plane_indexes_with_images(self):
        """
        Returns a list of integer indexes of the indexes of planes which create an image, meaning they either
        have a `LightProfile` or `Pixelization`.

        This list is used to visualize double source plane lenses, whereby a fit for every plane with a
        `LightProfile` or `Pixelization` is created.

        Returns
        -------
        The list of integer indexes of the planes which create an image.
        """
        plane_indexes_with_images = [
            plane_index if plane.has(cls=ag.LightProfile) else None
            for (plane_index, plane) in enumerate(self.planes)
        ] + self.plane_indexes_with_pixelizations

        plane_indexes_with_images = list(dict.fromkeys(plane_indexes_with_images))

        return [
            plane_index
            for plane_index in plane_indexes_with_images
            if plane_index is not None
        ]

    @property
    def perform_inversion(self) -> bool:
        """
        Returns a bool specifying whether this fit object performs an inversion.

        This is based on whether any of the galaxies have a `Pixelization` or `LightProfileLinear` object, in which
        case an inversion is performed.

        Returns
        -------
            A bool which is True if an inversion is performed.
        """
        return any(plane.perform_inversion for plane in self.planes)

    def extract_attribute(
        self, cls: Type, attr_name: str, filter_nones: Optional[bool] = False
    ) -> List[Union[aa.ArrayIrregular, aa.Grid2DIrregular]]:
        """
        Returns an extracted attribute of a class in the tracer as a `ValueIrregular` or `Grid2DIrregular` object.

        For example, if a tracer has a galaxy with two light profiles, the input:

           `tracer.extract_attribute(cls=LightProfile, name="axis_ratio")`

        Returns

            `ArrayIrregular(values=[axis_ratio_0, axis_ratio_1])`

        If the image plane has two galaxies with two mass profiles and the source plane another galaxy with a
        mass profile, the input:

            `tracer.extract_attribute(cls=MassProfile, name="centre")`

        Returns

            GridIrregular2D(grid=[(centre_y_0, centre_x_0), (centre_y_1, centre_x_1), (centre_y_2, centre_x_2)])

        The primary use of this function is to extract the attributes of profiles for visualization, for example
        plotting the centres of all mass profiles colored by their profile over the tracer's image.

        Parameters
        ----------
        cls
            The class type of object whose attribute is extracted (e.g. light profile, mass profile).
        attr_name
            The name of the attribute which is extracted from the class type (e.g. axis_ratio, centre).
        """

        def extract(value, name):
            try:
                return getattr(value, name)
            except (AttributeError, IndexError):
                return None

        # Members, not galaxies: this lists the *attributes of profiles* (e.g. every mass profile centre),
        # so a field's profiles are included alongside the galaxies'.
        attributes = [
            extract(value, attr_name)
            for galaxy in self.members
            for value in galaxy.__dict__.values()
            if isinstance(value, cls)
        ]

        if attributes == []:
            return None
        elif isinstance(attributes[0], float):
            return aa.ArrayIrregular(values=attributes)
        elif isinstance(attributes[0], tuple):
            return aa.Grid2DIrregular(values=attributes)

    def extract_attributes_of_planes(
        self, cls: Type, attr_name: str, filter_nones: Optional[bool] = False
    ) -> List[Union[aa.ArrayIrregular, aa.Grid2DIrregular]]:
        """
        Returns an extracted attribute of a class in the tracer as a list of `ValueIrregular` or `Grid2DIrregular`
        objects, where the indexes of the list correspond to the tracer's planes.

        For example, if a tracer has an image-plane with a galaxy with a light profile and a source-plane with a galaxy
        with a light profile, the input:

            `tracer.extract_attributes_of_planes(cls=LightProfile, name="axis_ratio")`

        Returns:

            [ArrayIrregular(values=[axis_ratio_0]), ArrayIrregular(values=[axis_ratio_1])]

        If the image plane has two galaxies with a mass profile each and the source plane another galaxy with a
        mass profile, input:

            `tracer.extract_attributes_of_planes(cls=MassProfile, name="centres")`

        Returns:

            [
                Grid2DIrregular(values=[(centre_y_0, centre_x_0)]),
                Grid2DIrregular(values=[(centre_y_0, centre_x_0), (centre_y_1, centre_x_1)])
            ]

        If a profile does not have a certain entry, it is replaced with a None. The Nones can be removed
        by setting `filter_nones=True`.

        The primary use of this function is to extract the attributes of profiles for visualization, for example
        plotting the centres of all mass profiles colored by their profile over the tracer's image.

        Parameters
        ----------
        cls
            The class type of object whose attribute is extracted (e.g. light profile, mass profile).
        attr_name
            The name of the attribute which is extracted from the class type (e.g. axis_ratio, centre).
        filter_nones
            If True, None entries are removed from the list.
        """

        attribute_list = []

        # This walks *plane members*, so a `MassField` in a plane contributes its own entry (its mass profile
        # attributes for `cls=MassProfile`, a `None` for a class it does not hold, filtered out by
        # `filter_nones=True`). That is the member-aware half of the pair: `extract_attributes_of_galaxies`
        # below enumerates galaxies and so stays galaxy-only.
        for plane in self.planes:
            for galaxy in plane:
                attribute_list += [
                    galaxy.extract_attribute(cls=cls, attr_name=attr_name)
                ]

        if filter_nones:
            return list(filter(None, attribute_list))

        return attribute_list

    def extract_attributes_of_galaxies(
        self, cls: Type, attr_name: str, filter_nones: Optional[bool] = False
    ) -> List[Union[aa.ArrayIrregular, aa.Grid2DIrregular]]:
        """
        Returns an attribute of a class in the tracer as a list of `ValueIrregular` or `Grid2DIrregular` objects, where
        the indexes of the list correspond to the tracer's galaxies. If a plane has multiple galaxies it will have a
        list with each galaxy as an entry.

        For example, if a tracer has an image-plane with a galaxy with a light profile and a source-plane with a galaxy
        with a light profile, the input:

            `tracer.extract_attributes_of_galaxies(cls=LightProfile, name="axis_ratio")`

        Returns:

            [ArrayIrregular(values=[axis_ratio_0]), ArrayIrregular(values=[axis_ratio_1])]

        If the image plane has two galaxies with a mass profile each and the source plane another galaxy with a
        mass profile, the input:

            `tracer.extract_attributes_of_galaxies(cls=MassProfile, name="centres")`

        Returns:

            [
                Grid2DIrregular(values=[(centre_y_0, centre_x_0)]),
                Grid2DIrregular(values=[(centre_y_0, centre_x_0)])
                Grid2DIrregular(values=[(centre_y_0, centre_x_0)])
            ]

        If the first galaxy in the image plane in the example above had two mass profiles as well as the galaxy in the
        source plane it would return:

            [
                Grid2DIrregular(values=[(centre_y_0, centre_x_0), (centre_y_1, centre_x_1)]),
                Grid2DIrregular(values=[(centre_y_0, centre_x_0)])
                Grid2DIrregular(values=[(centre_y_0, centre_x_0, (centre_y_1, centre_x_1))])
            ]

        If a profile does not have a certain entry, it is replaced with a None. The Nones can be removed
        by setting `filter_nones=True`.

        The primary use of this function is to extract the attributes of profiles for visualization, for example
        plotting the centres of all mass profiles colored by their profile over the tracer's image.

        Parameters
        ----------
        cls
            The class type of object whose attribute is extracted (e.g. light profile, mass profile).
        attr_name
            The name of the attribute which is extracted from the class type (e.g. axis_ratio, centre).
        filter_nones
        """
        # Galaxies, not members: the returned list is indexed *by galaxy*, so a `MassField` would shift every
        # index and break the caller's pairing. Use `extract_attribute` (whole tracer) or
        # `extract_attributes_of_planes` (per plane) when a field's profiles must be included.
        if filter_nones:
            return [
                galaxy.extract_attribute(cls=cls, attr_name=attr_name)
                for galaxy in self.galaxies
                if galaxy.extract_attribute(cls=cls, attr_name=attr_name) is not None
            ]

        else:
            return [
                galaxy.extract_attribute(cls=cls, attr_name=attr_name)
                for galaxy in self.galaxies
            ]

    def extract_profile(self, profile_name: str) -> GeometryProfile:
        """
        Returns a profile (e.g. a `LightProfile`, `MassProfile`, `Point`) from the tracer using the name of that
        component.

        For example, if a tracer has two galaxies named `lens` and `source`, where `lens` has a light profile
        named `light_0` and `source` has a light profile named `light_1`, the input:

            `tracer.extract_profile(profile_name="light_1")`

        Return the light profile of the source galaxy.

        This primarily used for point-source modeling, where the locations that the point-sources tracer to in
        different planes must be paired to their corresponding point-source `Point` profile.

        Parameters
        ----------
        profile_name
            The name of the profile component in the tracer.
        """
        # Members, not galaxies: a profile is looked up by its component name, and a field's components are
        # named exactly as a galaxy's are. `extract_plane_index_of_profile` already walks planes (and so
        # already sees fields), so the two must agree.
        for galaxy in self.members:
            try:
                return galaxy.__dict__[profile_name]
            except KeyError:
                pass

    def extract_plane_index_of_profile(self, profile_name: str) -> int:
        """
        Returns the plane index of a profile (e.g. a `LightProfile`, `MassProfile`, `Point`) from the tracer using
        the name of that component.

        For example, if a tracer has two galaxies named `lens` and `source`, where `lens` has a light profile
        named `light_0` and `source` has a light profile named `light_1`, the input:

            `tracer.extract_profile(profile_name="light_1")`

        Would return `plane_index=1` corresponding to the profile in the source plane.

        This primarily used for point-source modeling, where the locations that the point-sources tracer to in
        different planes must be paired to their corresponding point-source `Point` profile.

        Parameters
        ----------
        profile_name
            The name of the profile component in the tracer.
        """
        for plane_index, galaxies in enumerate(self.planes):
            for galaxy in galaxies:
                if profile_name in galaxy.__dict__:
                    return plane_index

    def set_snr_of_snr_light_profiles(
        self,
        grid: aa.type.Grid2DLike,
        exposure_time: float,
        background_sky_level: float = 0.0,
        psf: Optional[aa.Convolver] = None,
    ):
        """
        Iterate over every `LightProfileSNR` in the tracer and set their `intensity` values to values which give
        their input `signal_to_noise_ratio` value, which is performed as follows:

        - Evaluate the image of each light profile on the input grid.
        - Blur this image with a PSF, if included.
        - Take the value of the brightest pixel.
        - Use an input `exposure_time` and `background_sky` (e.g. from the `SimulatorImaging` object) to determine
          what value of `intensity` gives the desired signal to noise ratio for the image.

        The intensity is set using an input grid, meaning that for strong lensing calculations the ray-traced grid
        can be used such that the S/N accounts for the magnification of a source galaxy.

        Parameters
        ----------
        grid
            The (y, x) coordinates in the original reference frame of the grid.
        exposure_time
            The exposure time of the simulated imaging.
        background_sky_level
            The level of the background sky of the simulated imaging.
        psf
            The psf of the simulated imaging which can change the S/N of the light profile due to spreading out
            the emission.
        """
        has_snr_profile = any(
            isinstance(light_profile, LightProfileSNR)
            for galaxies in self.planes
            for galaxy in galaxies
            for light_profile in galaxy.cls_list_from(cls=ag.LightProfile)
        )
        if not has_snr_profile:
            return

        grid = aa.Grid2D.uniform(
            shape_native=grid.shape_native,
            pixel_scales=grid.pixel_scales,
        )

        traced_grids_of_planes_list = self.traced_grid_2d_list_from(grid=grid)

        for plane_index, galaxies in enumerate(self.planes):
            for galaxy in galaxies:
                for light_profile in galaxy.cls_list_from(cls=ag.LightProfile):
                    if isinstance(light_profile, LightProfileSNR):
                        light_profile.set_intensity_from(
                            grid=traced_grids_of_planes_list[plane_index],
                            exposure_time=exposure_time,
                            background_sky_level=background_sky_level,
                            psf=psf,
                        )
