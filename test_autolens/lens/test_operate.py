import numpy as np
import pytest
from os import path

import autogalaxy as ag
import autolens as al

test_path = path.join("{}".format(path.dirname(path.realpath(__file__))), "files")


def test__operate_image__blurred_images_2d_via_psf_from__for_tracer_gives_list_of_planes(
    grid_2d_7x7, blurring_grid_2d_7x7, psf_3x3
):
    g0 = al.Galaxy(
        redshift=0.5,
        light_profile=al.lp.Sersic(intensity=1.0),
        mass_profile=al.mp.IsothermalSph(einstein_radius=1.0),
    )
    g1 = al.Galaxy(redshift=1.0, light_profile=al.lp.Sersic(intensity=2.0))

    blurred_image_0 = g0.blurred_image_2d_from(
        grid=grid_2d_7x7, psf=psf_3x3, blurring_grid=blurring_grid_2d_7x7
    )

    source_grid_2d_7x7 = g0.traced_grid_2d_from(grid=grid_2d_7x7)
    source_blurring_grid_2d_7x7 = g0.traced_grid_2d_from(grid=blurring_grid_2d_7x7)

    blurred_image_1 = g1.blurred_image_2d_from(
        grid=source_grid_2d_7x7, psf=psf_3x3, blurring_grid=source_blurring_grid_2d_7x7
    )

    tracer = al.Tracer(galaxies=[g0, g1], cosmology=al.cosmo.Planck15())

    blurred_image = tracer.blurred_image_2d_from(
        grid=grid_2d_7x7, psf=psf_3x3, blurring_grid=blurring_grid_2d_7x7
    )

    assert blurred_image.native == pytest.approx(
        blurred_image_0.native.array + blurred_image_1.native.array, 1.0e-4
    )

    blurred_image_list = tracer.blurred_image_2d_list_from(
        grid=grid_2d_7x7, psf=psf_3x3, blurring_grid=blurring_grid_2d_7x7
    )

    assert (blurred_image_list[0].slim == blurred_image_0.slim).all()
    assert (blurred_image_list[1].slim == blurred_image_1.slim).all()

    assert (blurred_image_list[0].native == blurred_image_0.native).all()
    assert (blurred_image_list[1].native == blurred_image_1.native).all()


def test__operate_image__visibilities_of_planes_from_grid_and_transformer(
    grid_2d_7x7, transformer_7x7_7
):
    g0 = al.Galaxy(redshift=0.5, light_profile=al.lp.Sersic(intensity=1.0))
    g1 = al.Galaxy(redshift=1.0, light_profile=al.lp.Sersic(intensity=2.0))

    visibilities_0 = g0.visibilities_from(
        grid=grid_2d_7x7, transformer=transformer_7x7_7
    )

    visibilities_1 = g1.visibilities_from(
        grid=grid_2d_7x7, transformer=transformer_7x7_7
    )

    tracer = al.Tracer(galaxies=[g0, g1], cosmology=al.cosmo.Planck15())

    visibilities = tracer.visibilities_list_from(
        grid=grid_2d_7x7, transformer=transformer_7x7_7
    )

    assert (visibilities[0] == visibilities_0).all()
    assert (visibilities[1] == visibilities_1).all()


def test__operate_image__galaxy_blurred_image_2d_dict_from(
    grid_2d_7x7, blurring_grid_2d_7x7, psf_3x3
):
    g0 = al.Galaxy(redshift=0.5, light_profile=al.lp.Sersic(intensity=1.0))
    g1 = al.Galaxy(
        redshift=0.5,
        mass_profile=al.mp.IsothermalSph(einstein_radius=1.0),
        light_profile=al.lp.Sersic(intensity=2.0),
    )

    g2 = al.Galaxy(redshift=0.5, light_profile=al.lp.Sersic(intensity=3.0))

    g3 = al.Galaxy(redshift=1.0, light_profile=al.lp.Sersic(intensity=5.0))

    g0_blurred_image = g0.blurred_image_2d_from(
        grid=grid_2d_7x7,
        psf=psf_3x3,
        blurring_grid=blurring_grid_2d_7x7,
    )

    g1_blurred_image = g1.blurred_image_2d_from(
        grid=grid_2d_7x7,
        psf=psf_3x3,
        blurring_grid=blurring_grid_2d_7x7,
    )

    g2_blurred_image = g2.blurred_image_2d_from(
        grid=grid_2d_7x7,
        psf=psf_3x3,
        blurring_grid=blurring_grid_2d_7x7,
    )

    source_grid_2d_7x7 = g1.traced_grid_2d_from(grid=grid_2d_7x7)
    source_blurring_grid_2d_7x7 = g1.traced_grid_2d_from(grid=blurring_grid_2d_7x7)

    g3_blurred_image = g3.blurred_image_2d_from(
        grid=source_grid_2d_7x7,
        psf=psf_3x3,
        blurring_grid=source_blurring_grid_2d_7x7,
    )

    tracer = al.Tracer(galaxies=[g3, g1, g0, g2], cosmology=al.cosmo.Planck15())

    blurred_image_dict = tracer.galaxy_blurred_image_2d_dict_from(
        grid=grid_2d_7x7,
        psf=psf_3x3,
        blurring_grid=blurring_grid_2d_7x7,
    )

    assert blurred_image_dict[g0].slim == pytest.approx(
        g0_blurred_image.slim.array, 1.0e-4
    )
    assert blurred_image_dict[g1].slim == pytest.approx(
        g1_blurred_image.slim.array, 1.0e-4
    )
    assert blurred_image_dict[g2].slim == pytest.approx(
        g2_blurred_image.slim.array, 1.0e-4
    )
    assert blurred_image_dict[g3].slim == pytest.approx(
        g3_blurred_image.slim.array, 1.0e-4
    )


def test__operate_image__galaxy_visibilities_dict_from_grid_and_transformer(
    grid_2d_7x7, transformer_7x7_7
):
    g0 = al.Galaxy(redshift=0.5, light_profile=al.lp.Sersic(intensity=1.0))
    g1 = al.Galaxy(
        redshift=0.5,
        mass_profile=al.mp.IsothermalSph(einstein_radius=1.0),
        light_profile=al.lp.Sersic(intensity=2.0),
    )
    g2 = al.Galaxy(redshift=0.5, light_profile=al.lp.Sersic(intensity=3.0))
    g3 = al.Galaxy(redshift=1.0, light_profile=al.lp.Sersic(intensity=5.0))

    g0_visibilities = g0.visibilities_from(
        grid=grid_2d_7x7, transformer=transformer_7x7_7
    )
    g1_visibilities = g1.visibilities_from(
        grid=grid_2d_7x7, transformer=transformer_7x7_7
    )

    g2_visibilities = g2.visibilities_from(
        grid=grid_2d_7x7, transformer=transformer_7x7_7
    )

    source_grid_2d_7x7 = g1.traced_grid_2d_from(grid=grid_2d_7x7)

    g3_visibilities = g3.visibilities_from(
        grid=source_grid_2d_7x7, transformer=transformer_7x7_7
    )

    tracer = al.Tracer(galaxies=[g3, g1, g0, g2], cosmology=al.cosmo.Planck15())

    visibilities_dict = tracer.galaxy_visibilities_dict_from(
        grid=grid_2d_7x7, transformer=transformer_7x7_7
    )

    assert (visibilities_dict[g0] == g0_visibilities).all()
    assert (visibilities_dict[g1] == g1_visibilities).all()
    assert (visibilities_dict[g2] == g2_visibilities).all()
    assert (visibilities_dict[g3] == g3_visibilities).all()


def test__operate_lens__sums_individual_quantities():
    grid = al.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.15)

    sis_0 = al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=0.2)
    sis_1 = al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=0.4)
    sis_2 = al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=0.6)
    sis_3 = al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=0.8)

    galaxy_0 = al.Galaxy(mass_profile_0=sis_0, mass_profile_1=sis_1, redshift=0.5)
    galaxy_1 = al.Galaxy(mass_profile_0=sis_2, mass_profile_1=sis_3, redshift=0.5)

    tracer = al.Tracer(
        galaxies=[galaxy_0, galaxy_1],
        cosmology=al.cosmo.Planck15(),
    )

    einstein_mass = ag.LensCalc.from_mass_obj(
        tracer
    ).einstein_mass_angular_from(grid=grid)

    assert einstein_mass == pytest.approx(np.pi * 2.0**2.0, 1.0e-1)
