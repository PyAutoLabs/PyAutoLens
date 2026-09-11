"""
Regenerate the committed model-figure renders for the PyAutoLens docs.

Run from the PyAutoLens repo root, with the worktree environment active::

    python docs/general/images/model_cookbook/make_figures.py

Every PNG is written at ``width=14.0`` inches and DPI 100, so none exceeds the
1400 px width budget of the ``model-figures`` epic.

Two families of figure are written:

* the **cookbook stages** -- one per ``print(model.info)`` stage of
  ``autolens_workspace/scripts/guides/modeling/cookbook.py``, plus the
  ``Redshift Free`` and ``Solved Parameters`` stages, into this directory and
  embedded by ``docs/general/model_cookbook.md``;
* the **acceptance trio** of the epic (the simple lens, the MGE + pixelized
  source and the group-scale model), imported from
  ``test_autolens/model_figure/lens_models.py`` so the docs and the acceptance
  tests render the identical models.

``docs/overview/images/overview_3/`` additionally receives the three feature
figures embedded by ``docs/overview/overview_3_features.md``.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "test_autolens" / "model_figure"))

import autofit as af
import autolens as al

import lens_models

OUTPUT_PATH = Path(__file__).resolve().parent
OVERVIEW_PATH = REPO_ROOT / "docs" / "overview" / "images" / "overview_3"
WIDTH = 14.0


def write(model, name: str, path: Path = OUTPUT_PATH, **kwargs):
    """Render one model to ``<name>.png`` in ``path``."""
    path.mkdir(parents=True, exist_ok=True)

    af.ModelPlotter(model).figure(
        path=str(path),
        filename=name,
        format="png",
        width=WIDTH,
        **kwargs,
    )
    print(f"wrote {path.name}/{name}.png")


# ---------------------------------------------------------------------------
# the cookbook stages -- `scripts/guides/modeling/cookbook.py`
# ---------------------------------------------------------------------------


def simple_lens():
    """``__Simple Lens Model__`` -- cookbook.py:53-82."""
    bulge = af.Model(al.lp_linear.Sersic)
    mass = af.Model(al.mp.Isothermal)

    lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)

    source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.SersicCore)

    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def complex_lens():
    """``__More Complex Lens Models__`` -- cookbook.py:89-117."""
    bulge = af.Model(al.lp_linear.Sersic)
    disk = af.Model(al.lp_linear.Exponential)

    mass = af.Model(al.mp.Isothermal)
    shear = af.Model(al.mp.ExternalShear)

    lens = af.Model(
        al.Galaxy, redshift=0.5, bulge=bulge, disk=disk, mass=mass, shear=shear
    )

    bulge = af.Model(al.lp_linear.SersicCore)
    disk = af.Model(al.lp_linear.ExponentialCore)

    source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge, disk=disk)

    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def four_galaxies():
    """``__More Complex Lens Models__``, four galaxies -- cookbook.py:123-166."""
    lens_0 = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=af.Model(al.lp_linear.Sersic),
        mass=af.Model(al.mp.Isothermal),
    )
    lens_1 = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=af.Model(al.lp_linear.Sersic),
        mass=af.Model(al.mp.Isothermal),
    )
    source_0 = af.Model(
        al.Galaxy, redshift=1.0, bulge=af.Model(al.lp_linear.SersicCore)
    )
    source_1 = af.Model(
        al.Galaxy, redshift=1.0, bulge=af.Model(al.lp_linear.SersicCore)
    )

    return af.Collection(
        galaxies=af.Collection(
            lens_0=lens_0, lens_1=lens_1, source_0=source_0, source_1=source_1
        ),
    )


def concise():
    """``__Concise API__`` -- cookbook.py:183-200."""
    lens = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=al.lp_linear.Sersic,
        disk=al.lp_linear.Sersic,
        mass=al.mp.Isothermal,
        shear=al.mp.ExternalShear,
    )

    source = af.Model(
        al.Galaxy,
        redshift=1.0,
        bulge=al.lp_linear.SersicCore,
        disk=al.lp_linear.ExponentialCore,
    )

    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def prior_custom():
    """``__Prior Customization__`` -- cookbook.py:221-256."""
    bulge = af.Model(al.lp_linear.Sersic)
    bulge.sersic_index = af.TruncatedGaussianPrior(
        mean=4.0, sigma=1.0, lower_limit=1.0, upper_limit=8.0
    )

    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.1, lower_limit=-0.5, upper_limit=0.5
    )
    mass.centre.centre_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.1, lower_limit=-0.5, upper_limit=0.5
    )
    mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=8.0)

    lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)

    bulge = af.Model(al.lp_linear.SersicCore)

    source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)
    source.effective_radius = af.TruncatedGaussianPrior(
        mean=0.1, sigma=0.05, lower_limit=0.0, upper_limit=1.0
    )

    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def model_custom():
    """``__Model Customization__`` -- cookbook.py:263-319."""
    bulge = af.Model(al.lp_linear.Sersic)
    disk = af.Model(al.lp_linear.Exponential)

    bulge.centre = disk.centre
    bulge.sersic_index = 4.0

    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = bulge.centre.centre_0 + 0.1
    mass.centre.centre_1 = bulge.centre.centre_1 + 0.1

    shear = af.Model(al.mp.ExternalShear)

    lens = af.Model(
        al.Galaxy, redshift=0.5, bulge=bulge, disk=disk, mass=mass, shear=shear
    )

    bulge = af.Model(al.lp_linear.SersicCore)
    disk = af.Model(al.lp_linear.ExponentialCore)

    source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge, disk=disk)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    model.add_assertion(
        model.galaxies.lens.bulge.effective_radius
        > model.galaxies.lens.disk.effective_radius
    )
    model.add_assertion(model.galaxies.lens.mass.einstein_radius < 3.0)

    return model


def redshift_free():
    """``__Redshift Free__`` -- cookbook.py:327-329, brought into a full model."""
    redshift = af.Model(al.Redshift)
    redshift.redshift = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)

    lens = af.Model(al.Galaxy, redshift=redshift, mass=al.mp.Isothermal)

    source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.SersicCore)

    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def solved_parameters():
    """
    ``__Solved Parameters__`` -- the three solved encodings in one model.

    A linear ``Sersic`` bulge (solved ``intensity``), a pixelized source (solved
    ``reconstruction``, and a ``Delaunay`` mesh whose ``areas_factor`` is
    ``missing``) and a ``PointSolved`` point source (solved ``centre``).
    """
    lens = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=af.Model(al.lp_linear.Sersic),
        mass=af.Model(al.mp.Isothermal),
    )

    pixelization = af.Model(
        al.Pixelization,
        mesh=af.Model(al.mesh.Delaunay, pixels=500, zeroed_pixels=0),
        regularization=af.Model(al.reg.ConstantSplit),
    )

    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

    point_galaxy = af.Model(al.Galaxy, redshift=1.0, point=af.Model(al.ps.PointSolved))

    return af.Collection(
        galaxies=af.Collection(lens=lens, source=source, point_source=point_galaxy)
    )


# ---------------------------------------------------------------------------
# the overview_3 feature figures
# ---------------------------------------------------------------------------


def mge_overview():
    """The MGE lens light model of ``features/multi_gaussian_expansion/modeling.py``."""
    lens = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=lens_models.mge_basis(),
        mass=af.Model(al.mp.Isothermal),
        shear=af.Model(al.mp.ExternalShear),
    )

    source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.SersicCore)

    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def pixelization_overview():
    """The pixelized source model of ``features/pixelization/modeling.py``:274-289."""
    mass = af.Model(al.mp.PowerLaw)
    shear = af.Model(al.mp.ExternalShear)

    lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

    mesh = af.Model(al.mesh.RectangularBilinearAdaptDensity, shape=(30, 30))
    regularization = af.Model(al.reg.Constant)

    pixelization = af.Model(al.Pixelization, mesh=mesh, regularization=regularization)

    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def point_overview():
    """A point-source lens model -- an ``Isothermal`` lens and an ``al.ps.Point``."""
    lens = af.Model(al.Galaxy, redshift=0.5, mass=af.Model(al.mp.Isothermal))

    source = af.Model(al.Galaxy, redshift=1.0, point_0=af.Model(al.ps.Point))

    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def main():
    write(simple_lens(), "simple_lens")
    write(complex_lens(), "complex_lens")
    write(four_galaxies(), "four_galaxies")
    write(concise(), "concise")
    write(prior_custom(), "prior_custom")
    write(model_custom(), "model_custom")
    write(redshift_free(), "redshift_free")
    write(solved_parameters(), "solved_parameters")

    write(lens_models.simple_lens(), "acceptance_simple_lens")
    write(lens_models.mge_pixelized(), "acceptance_mge_pixelized")
    write(lens_models.group_scale(), "acceptance_group_scale")

    write(mge_overview(), "mge_model", path=OVERVIEW_PATH)
    write(pixelization_overview(), "pixelization_model", path=OVERVIEW_PATH)
    write(point_overview(), "point_model", path=OVERVIEW_PATH)


if __name__ == "__main__":
    main()
