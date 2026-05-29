"""
Regression test for the HowToLens chapter-1 ``tutorial_3_more_ray_tracing``
crash: ``ValueError: Axis limits cannot be NaN or Inf during plotting``.

The tutorial builds a multi-plane tracer whose lens mass profiles are singular
isothermal spheres centred on grid pixels. Historically the deflection angles
evaluated at those singular centres produced NaN/Inf, so the traced
source-plane grid contained non-finite coordinates; the tracer plotters then
derived their axis limits from those coordinates and matplotlib raised
``ValueError: Axis limits cannot be NaN or Inf``.

The fix was made at the *producer*: the mass-profile deflection code now handles
``r = 0`` (see autogalaxy "handle r=0 in ... deflections" and the NaN-safe
masking changes), so the traced grids and images stay finite and the plotters
receive finite extents.

These tests lock that in by reproducing the tutorial-3 four-galaxy, two-plane
tracer on a grid whose pixels coincide with the singular mass centres, and
asserting (a) the traced grids and image remain finite and (b) the tracer
subplots render without raising.
"""
import numpy as np
import pytest

import autolens as al
from autolens.lens.plot.tracer_plots import (
    subplot_tracer,
    subplot_galaxies_images,
)


@pytest.fixture(name="grid_singular_centres")
def make_grid_singular_centres():
    # An odd ``shape_native`` with 0.5" pixels places pixels exactly on the
    # singular mass centres at (0.0, 0.0) and (1.0, 0.0) -- the historical
    # NaN/Inf trigger for the deflection angles.
    return al.Grid2D.uniform(shape_native=(7, 7), pixel_scales=0.5)


@pytest.fixture(name="tracer_tutorial_3")
def make_tracer_tutorial_3():
    # The "Multi Galaxy Ray Tracing" tracer from HowToLens tutorial 3: two lens
    # galaxies at z=0.5 (a main lens with a singular isothermal + external shear
    # and a satellite) and two source galaxies at z=1.0.
    lens = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.SersicSph(
            centre=(0.0, 0.0), intensity=2.0, effective_radius=0.5, sersic_index=2.5
        ),
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0), ell_comps=(0.0, -0.111111), einstein_radius=1.6
        ),
        shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.0),
    )
    lens_satellite = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.DevVaucouleursSph(
            centre=(1.0, 0.0), intensity=2.0, effective_radius=0.2
        ),
        mass=al.mp.IsothermalSph(centre=(1.0, 0.0), einstein_radius=0.4),
    )
    source_0 = al.Galaxy(
        redshift=1.0,
        bulge=al.lp.DevVaucouleursSph(
            centre=(0.1, 0.2), intensity=0.3, effective_radius=0.3
        ),
        disk=al.lp.ExponentialCore(
            centre=(0.1, 0.2),
            ell_comps=(0.111111, 0.0),
            intensity=3.0,
            effective_radius=2.0,
        ),
    )
    source_1 = al.Galaxy(
        redshift=1.0,
        disk=al.lp.ExponentialCore(
            centre=(-0.3, -0.5),
            ell_comps=(0.1, 0.0),
            intensity=8.0,
            effective_radius=1.0,
        ),
    )
    return al.Tracer(
        galaxies=[lens, lens_satellite, source_0, source_1],
        cosmology=al.cosmo.Planck15(),
    )


def test__traced_grids_and_image_are_finite_at_singular_mass_centres(
    tracer_tutorial_3, grid_singular_centres
):
    # Producer-side guard: deflections at the singular mass centres must not
    # introduce NaN/Inf, otherwise the plotters below would derive non-finite
    # axis limits and crash (the original tutorial-3 failure).
    traced = tracer_tutorial_3.traced_grid_2d_list_from(grid=grid_singular_centres)

    for plane_grid in traced:
        assert np.isfinite(np.asarray(plane_grid.array)).all()

    image = tracer_tutorial_3.image_2d_from(grid=grid_singular_centres)

    assert np.isfinite(np.asarray(image.array)).all()


def test__subplot_tracer__singular_mass_centres__does_not_raise(
    tracer_tutorial_3, grid_singular_centres, tmp_path, plot_patch
):
    subplot_tracer(
        tracer=tracer_tutorial_3,
        grid=grid_singular_centres,
        output_path=tmp_path,
        output_format="png",
    )

    assert str(tmp_path / "tracer.png") in plot_patch.paths


def test__subplot_galaxies_images__singular_mass_centres__does_not_raise(
    tracer_tutorial_3, grid_singular_centres, tmp_path, plot_patch
):
    subplot_galaxies_images(
        tracer=tracer_tutorial_3,
        grid=grid_singular_centres,
        output_path=tmp_path,
        output_format="png",
    )

    assert str(tmp_path / "galaxies_images.png") in plot_patch.paths
