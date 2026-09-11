"""
The three lens acceptance models of the ``model-figures`` epic, built from the
**real** PyAutoLens classes.

Phases 1 and 2 rendered these three models from *structural doubles* inside
``test_autofit`` (classes with matching constructor signatures, so PyAutoFit
could render them without importing PyAutoLens).  Phase 3 replaces the doubles
with the real thing: every model below is the composition of a shipped
``autolens_workspace`` script, so the figure a reader sees in the docs is the
model the workspace composes.

* :func:`simple_lens` -- ``scripts/imaging/modeling.py``
* :func:`mge_pixelized` -- ``scripts/imaging/features/multi_gaussian_expansion/modeling.py``
  with the pixelized source of ``scripts/imaging/features/pixelization/modeling.py``
  (no shipped script has both, so the two are assembled here)
* :func:`group_scale` -- ``scripts/group/modeling.py`` with eight synthetic
  extra-galaxy centres (the shipped dataset has one; the epic's acceptance case
  is eight)

The builders take no arguments and load no data, so they are importable from the
docs ``make_figures.py`` script as well as from the tests.
"""

import numpy as np

import autofit as af
import autolens as al

__all__ = [
    "simple_lens",
    "mge_pixelized",
    "group_scale",
    "EXTRA_GALAXY_CENTRES",
]

#: The eight synthetic extra-galaxy centres of the group-scale acceptance case.
EXTRA_GALAXY_CENTRES = [
    (1.0, 1.0),
    (-1.0, 1.0),
    (1.0, -1.0),
    (-1.0, -1.0),
    (2.0, 0.0),
    (-2.0, 0.0),
    (0.0, 2.0),
    (0.0, -2.0),
]


def simple_lens():
    """
    Acceptance case (a) -- the simple lens of ``scripts/imaging/modeling.py``.

    A lens galaxy at ``z=0.5`` with a ``Sersic`` bulge, an ``Isothermal`` mass
    profile and an ``ExternalShear``, and a source galaxy at ``z=1.0`` with a
    ``SersicCore`` bulge.
    """
    bulge = af.Model(al.lp.Sersic)
    mass = af.Model(al.mp.Isothermal)
    shear = af.Model(al.mp.ExternalShear)

    lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass, shear=shear)

    bulge = af.Model(al.lp.SersicCore)

    source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def mge_basis(total_gaussians: int = 30, gaussian_per_basis: int = 2):
    """
    The MGE ``Basis`` of ``features/multi_gaussian_expansion/modeling.py``.

    ``gaussian_per_basis`` bases of ``total_gaussians`` linear Gaussians each.
    All 60 Gaussians share one ``centre_0`` and one ``centre_1`` prior; the
    Gaussians of each basis share that basis's own ``ell_comps`` pair (which is
    what splits the figure into two plates); every ``sigma`` is fixed to its own
    log-spaced value.
    """
    mask_radius = 3.0
    pixel_scale = 0.1

    log10_sigma_list = np.linspace(
        np.log10(pixel_scale / 10.0), np.log10(mask_radius), total_gaussians
    )

    centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
    centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

    bulge_gaussian_list = []

    for _ in range(gaussian_per_basis):
        gaussian_list = af.Collection(
            af.Model(al.lp_linear.Gaussian) for _ in range(total_gaussians)
        )

        for i, gaussian in enumerate(gaussian_list):
            gaussian.centre.centre_0 = centre_0
            gaussian.centre.centre_1 = centre_1
            gaussian.ell_comps = gaussian_list[0].ell_comps
            gaussian.sigma = 10 ** log10_sigma_list[i]

        bulge_gaussian_list += gaussian_list

    return af.Model(al.lp_basis.Basis, profile_list=bulge_gaussian_list)


def mge_pixelized():
    """
    Acceptance case (b) -- MGE lens light, pixelized source.

    The lens is the 2 x 30 MGE of
    ``features/multi_gaussian_expansion/modeling.py`` plus an ``Isothermal`` and
    an ``ExternalShear``; the source is the ``Pixelization`` of
    ``features/pixelization/modeling.py`` built on a ``Delaunay`` mesh, whose
    ``areas_factor`` has no prior configured and is therefore ``missing``.
    """
    lens = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=mge_basis(),
        mass=af.Model(al.mp.Isothermal),
        shear=af.Model(al.mp.ExternalShear),
    )

    pixelization = af.Model(
        al.Pixelization,
        mesh=af.Model(al.mesh.Delaunay, pixels=500, zeroed_pixels=0),
        regularization=af.Model(al.reg.ConstantSplit),
    )

    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def group_scale(centres=None):
    """
    Acceptance case (e) -- the group-scale model of ``scripts/group/modeling.py``.

    One main lens galaxy (``Sersic`` + ``Isothermal`` + ``ExternalShear``), a
    ``SersicCore`` source, and eight extra galaxies whose ``dPIEMassSph``
    centres are fixed to their own measured positions -- a plate of eight whose
    ``centre`` is ``fixed, varies by member`` and whose ``sigma`` priors are
    independent.
    """
    centres = EXTRA_GALAXY_CENTRES if centres is None else centres

    lens = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=af.Model(al.lp.Sersic),
        mass=af.Model(al.mp.Isothermal),
        shear=af.Model(al.mp.ExternalShear),
    )

    extra_galaxies_list = []

    for centre in centres:
        bulge = af.Model(al.lp.SersicSph)

        mass = af.Model(al.mp.dPIEMassSph)
        mass.centre = centre
        mass.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=300.0)
        mass.r_core = 0.0
        mass.r_cut = 10.0
        mass.redshift_object = 0.5
        mass.redshift_source = 1.0
        mass.H0 = 67.66
        mass.Om0 = 0.30966

        extra_galaxies_list.append(
            af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)
        )

    source = af.Model(al.Galaxy, redshift=1.0, bulge=af.Model(al.lp.SersicCore))

    return af.Collection(
        galaxies=af.Collection(lens=lens, source=source),
        extra_galaxies=af.Collection(extra_galaxies_list),
    )
