import numpy as np
import pytest
from os import path

import autolens as al

test_path = path.join("{}".format(path.dirname(path.realpath(__file__))), "files")


def test__lp_linear_func_galaxy_dict_from(masked_imaging_7x7):
    # TODO : use pytest.parameterize

    galaxy_no_pix = al.Galaxy(redshift=0.5)

    tracer = al.Tracer(galaxies=[galaxy_no_pix, galaxy_no_pix])

    tracer_to_inversion = al.TracerToInversion(
        dataset=masked_imaging_7x7,
        tracer=tracer,
    )

    lp_linear_func_galaxy_dict = tracer_to_inversion.lp_linear_func_list_galaxy_dict

    assert lp_linear_func_galaxy_dict == {}

    lp_linear_0 = al.lp_linear.LightProfileLinear()
    lp_linear_1 = al.lp_linear.LightProfileLinear()
    lp_linear_2 = al.lp_linear.LightProfileLinear()

    galaxy_no_linear = al.Galaxy(redshift=0.5)
    galaxy_linear_0 = al.Galaxy(
        redshift=0.5, lp_linear=lp_linear_0, mass=al.mp.IsothermalSph()
    )

    galaxy_linear_1 = al.Galaxy(
        redshift=1.0, lp_linear=lp_linear_1, mass=al.mp.IsothermalSph()
    )
    galaxy_linear_2 = al.Galaxy(redshift=2.0, lp_linear=lp_linear_2)

    tracer = al.Tracer(
        galaxies=[galaxy_no_linear, galaxy_linear_0, galaxy_linear_1, galaxy_linear_2]
    )

    tracer_to_inversion = al.TracerToInversion(
        dataset=masked_imaging_7x7, tracer=tracer
    )

    lp_linear_func_galaxy_dict = tracer_to_inversion.lp_linear_func_list_galaxy_dict

    lp_linear_func_list = list(lp_linear_func_galaxy_dict.keys())

    assert lp_linear_func_galaxy_dict[lp_linear_func_list[0]] == galaxy_linear_0
    assert lp_linear_func_galaxy_dict[lp_linear_func_list[1]] == galaxy_linear_1
    assert lp_linear_func_galaxy_dict[lp_linear_func_list[2]] == galaxy_linear_2

    assert lp_linear_func_list[0].light_profile_list[0] == lp_linear_0
    assert lp_linear_func_list[1].light_profile_list[0] == lp_linear_1
    assert lp_linear_func_list[2].light_profile_list[0] == lp_linear_2

    traced_grid_list = tracer.traced_grid_2d_list_from(grid=masked_imaging_7x7.grids.lp)

    assert lp_linear_func_list[0].grid == pytest.approx(
        masked_imaging_7x7.grids.lp, 1.0e-4
    )
    assert lp_linear_func_list[1].grid == pytest.approx(
        traced_grid_list[1].array, 1.0e-4
    )
    assert lp_linear_func_list[2].grid == pytest.approx(
        traced_grid_list[2].array, 1.0e-4
    )

    lp_linear_3 = al.lp_linear.LightProfileLinear()
    lp_linear_4 = al.lp_linear.LightProfileLinear()

    basis_0 = al.lp_basis.Basis(profile_list=[lp_linear_0, lp_linear_1])

    galaxy_linear_0 = al.Galaxy(redshift=0.5, bulge=basis_0, mass=al.mp.IsothermalSph())

    galaxy_linear_1 = al.Galaxy(redshift=1.0, mass=al.mp.IsothermalSph())

    galaxy_linear_2 = al.Galaxy(redshift=2.0, lp_linear=lp_linear_2)

    basis_1 = al.lp_basis.Basis(profile_list=[lp_linear_3, lp_linear_4])

    galaxy_linear_3 = al.Galaxy(redshift=2.0, bulge=basis_1)

    tracer = al.Tracer(
        galaxies=[
            galaxy_no_linear,
            galaxy_linear_0,
            galaxy_linear_1,
            galaxy_linear_2,
            galaxy_linear_3,
        ]
    )

    tracer_to_inversion = al.TracerToInversion(
        dataset=masked_imaging_7x7,
        tracer=tracer,
    )

    lp_linear_func_galaxy_dict = tracer_to_inversion.lp_linear_func_list_galaxy_dict

    lp_linear_func_list = list(lp_linear_func_galaxy_dict.keys())

    assert lp_linear_func_galaxy_dict[lp_linear_func_list[0]] == galaxy_linear_0
    assert lp_linear_func_galaxy_dict[lp_linear_func_list[1]] == galaxy_linear_2
    assert lp_linear_func_galaxy_dict[lp_linear_func_list[2]] == galaxy_linear_3

    assert lp_linear_func_list[0].light_profile_list[0] == lp_linear_0
    assert lp_linear_func_list[0].light_profile_list[1] == lp_linear_1
    assert lp_linear_func_list[1].light_profile_list[0] == lp_linear_2
    assert lp_linear_func_list[2].light_profile_list[0] == lp_linear_3


def test__cls_pg_list_from(masked_imaging_7x7, grid_2d_7x7):
    mesh_0 = al.mesh.RectangularUniform(shape=(3, 3))

    pixelization_0 = al.Pixelization(mesh=mesh_0)

    galaxy_pix = al.Galaxy(redshift=1.0, pixelization=pixelization_0)
    galaxy_no_pix = al.Galaxy(redshift=0.5)

    tracer = al.Tracer(galaxies=[galaxy_no_pix, galaxy_pix])

    tracer_to_inversion = al.TracerToInversion(
        dataset=masked_imaging_7x7, tracer=tracer
    )

    pixelization_list = tracer_to_inversion.cls_pg_list_from(cls=al.Pixelization)

    assert pixelization_list[0] == []
    assert pixelization_list[1][0].mesh.pixels == 9

    mesh_1 = al.mesh.RectangularUniform(shape=(4, 3))

    pixelization_1 = al.Pixelization(mesh=mesh_1)

    mesh_2 = al.mesh.RectangularUniform(shape=(4, 4))

    pixelization_2 = al.Pixelization(mesh=mesh_2)

    galaxy_pix_0 = al.Galaxy(redshift=0.5, pixelization=pixelization_0)

    galaxy_pix_1 = al.Galaxy(redshift=1.0, pixelization=pixelization_1)

    galaxy_pix_2 = al.Galaxy(redshift=1.0, pixelization=pixelization_2)

    tracer = al.Tracer(galaxies=[galaxy_pix_0, galaxy_pix_1, galaxy_pix_2])

    tracer_to_inversion = al.TracerToInversion(
        dataset=masked_imaging_7x7, tracer=tracer
    )

    pixelization_list = tracer_to_inversion.cls_pg_list_from(cls=al.Pixelization)

    assert pixelization_list[0][0].mesh.pixels == 9
    assert pixelization_list[1][0].mesh.pixels == 12
    assert pixelization_list[1][1].mesh.pixels == 16

    galaxy_no_pix = al.Galaxy(redshift=0.5)

    tracer = al.Tracer(galaxies=[galaxy_no_pix, galaxy_no_pix])

    tracer_to_inversion = al.TracerToInversion(
        dataset=masked_imaging_7x7, tracer=tracer
    )

    pixelization_list = tracer_to_inversion.cls_pg_list_from(cls=al.Pixelization)

    assert pixelization_list == [[]]


def test__adapt_galaxy_image_pg_list(masked_imaging_7x7, grid_2d_7x7):
    gal = al.Galaxy(redshift=0.5)

    tracer = al.Tracer(galaxies=[gal, gal])

    tracer_to_inversion = al.TracerToInversion(
        dataset=masked_imaging_7x7, tracer=tracer
    )

    assert tracer_to_inversion.adapt_galaxy_image_pg_list == [[]]

    pixelization = al.Pixelization(
        mesh=al.m.MockMesh(), regularization=al.m.MockRegularization()
    )

    gal_pix = al.Galaxy(redshift=0.5, pixelization=pixelization)

    tracer = al.Tracer(galaxies=[gal_pix, gal_pix])

    tracer_to_inversion = al.TracerToInversion(
        dataset=masked_imaging_7x7, tracer=tracer
    )

    assert tracer_to_inversion.adapt_galaxy_image_pg_list == [[None, None]]

    gal_pix = al.Galaxy(redshift=0.5, pixelization=pixelization)

    adapt_images = al.AdaptImages(galaxy_image_dict={gal_pix: 1})

    tracer = al.Tracer(galaxies=[gal_pix, gal])

    tracer_to_inversion = al.TracerToInversion(
        dataset=masked_imaging_7x7, tracer=tracer, adapt_images=adapt_images
    )

    assert tracer_to_inversion.adapt_galaxy_image_pg_list == [[1]]

    gal0 = al.Galaxy(redshift=0.25)
    gal1 = al.Galaxy(redshift=0.75)
    gal2 = al.Galaxy(redshift=1.5)

    gal_pix0 = al.Galaxy(redshift=0.5, pixelization=pixelization)

    gal_pix1 = al.Galaxy(redshift=2.0, pixelization=pixelization)

    gal_pix2 = al.Galaxy(redshift=2.0, pixelization=pixelization)

    adapt_images = al.AdaptImages(
        galaxy_image_dict={gal_pix0: 1, gal_pix1: 2, gal_pix2: 3}
    )

    tracer = al.Tracer(galaxies=[gal0, gal1, gal2, gal_pix0, gal_pix1, gal_pix2])

    tracer_to_inversion = al.TracerToInversion(
        dataset=masked_imaging_7x7, tracer=tracer, adapt_images=adapt_images
    )

    assert tracer_to_inversion.adapt_galaxy_image_pg_list == [[], [1], [], [], [2, 3]]


def test__image_plane_mesh_grid_pg_list(masked_imaging_7x7):
    # Test Correct

    image_plane_mesh_grid_0 = np.array([[1.0, 1.0]])

    galaxy_pix = al.Galaxy(redshift=1.0, pixelization=al.m.MockPixelization())
    galaxy_no_pix = al.Galaxy(redshift=0.5)

    adapt_images = al.AdaptImages(
        galaxy_image_dict={galaxy_pix: 2},
        galaxy_image_plane_mesh_grid_dict={galaxy_pix: image_plane_mesh_grid_0},
    )

    tracer = al.Tracer(galaxies=[galaxy_no_pix, galaxy_pix])

    tracer_to_inversion = al.TracerToInversion(
        dataset=masked_imaging_7x7,
        tracer=tracer,
        adapt_images=adapt_images,
    )

    mesh_grids = tracer_to_inversion.image_plane_mesh_grid_pg_list

    assert mesh_grids[0] == None
    assert (mesh_grids[1] == np.array([[1.0, 1.0]])).all()

    # Test for extra galaxies

    galaxy_pix_0 = al.Galaxy(redshift=1.0, pixelization=al.m.MockPixelization())

    image_plane_mesh_grid_1 = np.array([[2.0, 2.0]])

    galaxy_pix_1 = al.Galaxy(redshift=2.0, pixelization=al.m.MockPixelization())

    galaxy_no_pix_0 = al.Galaxy(redshift=0.25)
    galaxy_no_pix_1 = al.Galaxy(redshift=0.5)
    galaxy_no_pix_2 = al.Galaxy(redshift=1.5)

    adapt_images = al.AdaptImages(
        galaxy_image_dict={galaxy_pix_0: 2, galaxy_pix_1: 3},
        galaxy_image_plane_mesh_grid_dict={
            galaxy_pix_0: image_plane_mesh_grid_0,
            galaxy_pix_1: image_plane_mesh_grid_1,
        },
    )

    tracer = al.Tracer(
        galaxies=[
            galaxy_pix_0,
            galaxy_pix_1,
            galaxy_no_pix_0,
            galaxy_no_pix_1,
            galaxy_no_pix_2,
        ]
    )

    tracer_to_inversion = al.TracerToInversion(
        dataset=masked_imaging_7x7,
        tracer=tracer,
        adapt_images=adapt_images,
    )

    mesh_grids = tracer_to_inversion.image_plane_mesh_grid_pg_list

    assert mesh_grids[0] == None
    assert mesh_grids[1] == None
    assert (mesh_grids[2] == np.array([[1.0, 1.0]])).all()
    assert mesh_grids[3] == None
    assert (mesh_grids[4] == np.array([[2.0, 2.0]])).all()


def test__traced_mesh_grid_pg_list(masked_imaging_7x7):
    # Test Multi plane

    galaxy_no_pix = al.Galaxy(
        redshift=0.5,
        mass_profile=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=0.5),
    )

    image_plane_mesh_grid_0 = al.Grid2D.no_mask(
        values=[[[1.0, 0.0]]], pixel_scales=(1.0, 1.0)
    )

    galaxy_pix_0 = al.Galaxy(redshift=1.0, pixelization=al.m.MockPixelization())

    image_plane_mesh_grid_1 = al.Grid2D.no_mask(
        values=[[[2.0, 0.0]]], pixel_scales=(1.0, 1.0)
    )

    galaxy_pix_1 = al.Galaxy(redshift=1.0, pixelization=al.m.MockPixelization())

    tracer = al.Tracer(galaxies=[galaxy_no_pix, galaxy_pix_0, galaxy_pix_1])

    adapt_images = al.AdaptImages(
        galaxy_image_dict={galaxy_pix_0: 2, galaxy_pix_1: 3},
        galaxy_image_plane_mesh_grid_dict={
            galaxy_pix_0: image_plane_mesh_grid_0,
            galaxy_pix_1: image_plane_mesh_grid_1,
        },
    )

    tracer_to_inversion = al.TracerToInversion(
        dataset=masked_imaging_7x7, tracer=tracer, adapt_images=adapt_images
    )

    traced_mesh_grids_list_of_planes = tracer_to_inversion.traced_mesh_grid_pg_list

    assert traced_mesh_grids_list_of_planes[0] == None
    assert traced_mesh_grids_list_of_planes[1][0] == pytest.approx(
        np.array([[1.0 - 0.5, 0.0]]), 1.0e-4
    )
    assert traced_mesh_grids_list_of_planes[1][1] == pytest.approx(
        np.array([[2.0 - 0.5, 0.0]]), 1.0e-4
    )

    # Test Extra Galaxies

    galaxy_pix_0 = al.Galaxy(redshift=1.0, pixelization=al.m.MockPixelization())
    galaxy_pix_1 = al.Galaxy(redshift=2.0, pixelization=al.m.MockPixelization())

    adapt_images = al.AdaptImages(
        galaxy_image_dict={galaxy_pix_0: 2, galaxy_pix_1: 3},
        galaxy_image_plane_mesh_grid_dict={
            galaxy_pix_0: image_plane_mesh_grid_0,
            galaxy_pix_1: image_plane_mesh_grid_1,
        },
    )

    galaxy_no_pix_0 = al.Galaxy(
        redshift=0.25,
        mass_profile=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=0.5),
    )
    galaxy_no_pix_1 = al.Galaxy(redshift=0.5)
    galaxy_no_pix_2 = al.Galaxy(redshift=1.5)

    tracer = al.Tracer(
        galaxies=[
            galaxy_pix_0,
            galaxy_pix_1,
            galaxy_no_pix_0,
            galaxy_no_pix_1,
            galaxy_no_pix_2,
        ]
    )

    tracer_to_inversion = al.TracerToInversion(
        dataset=masked_imaging_7x7, tracer=tracer, adapt_images=adapt_images
    )

    traced_mesh_grids_list_of_planes = tracer_to_inversion.traced_mesh_grid_pg_list

    traced_grid_pix_0 = tracer.traced_grid_2d_list_from(
        grid=al.Grid2DIrregular(values=[[1.0, 0.0]])
    )[2]
    traced_grid_pix_1 = tracer.traced_grid_2d_list_from(
        grid=al.Grid2DIrregular(values=[[2.0, 0.0]])
    )[4]

    assert traced_mesh_grids_list_of_planes[0] == None
    assert traced_mesh_grids_list_of_planes[1] == None
    assert (traced_mesh_grids_list_of_planes[2][0] == traced_grid_pix_0).all()
    assert traced_mesh_grids_list_of_planes[3] == None
    assert (traced_mesh_grids_list_of_planes[4][0] == traced_grid_pix_1).all()


def test__mapper_galaxy_dict(masked_imaging_7x7):
    galaxy_no_pix = al.Galaxy(redshift=0.5)

    tracer = al.Tracer(galaxies=[galaxy_no_pix, galaxy_no_pix])

    tracer_to_inversion = al.TracerToInversion(
        dataset=masked_imaging_7x7, tracer=tracer
    )

    mapper_galaxy_dict = tracer_to_inversion.mapper_galaxy_dict
    assert mapper_galaxy_dict == {}

    galaxy_no_pix = al.Galaxy(redshift=0.5)

    pixelization_0 = al.m.MockPixelization(
        mesh=al.mesh.RectangularUniform(shape=(3, 3))
    )

    galaxy_pix_0 = al.Galaxy(redshift=0.5, pixelization=pixelization_0)

    pixelization_1 = al.m.MockPixelization(
        mesh=al.mesh.RectangularUniform(shape=(4, 3))
    )

    galaxy_pix_1 = al.Galaxy(redshift=1.0, pixelization=pixelization_1)

    pixelization_2 = al.m.MockPixelization(
        mesh=al.mesh.RectangularUniform(shape=(4, 4))
    )

    galaxy_pix_2 = al.Galaxy(redshift=1.0, pixelization=pixelization_2)

    tracer = al.Tracer(
        galaxies=[galaxy_no_pix, galaxy_pix_0, galaxy_pix_1, galaxy_pix_2]
    )

    tracer_to_inversion = al.TracerToInversion(
        dataset=masked_imaging_7x7, tracer=tracer
    )

    mapper_galaxy_dict = tracer_to_inversion.mapper_galaxy_dict

    mapper_list = list(mapper_galaxy_dict.keys())

    assert mapper_galaxy_dict[mapper_list[0]] == galaxy_pix_0
    assert mapper_galaxy_dict[mapper_list[1]] == galaxy_pix_1
    assert mapper_galaxy_dict[mapper_list[2]] == galaxy_pix_2

    assert mapper_list[0].pixels == 9
    assert mapper_list[1].pixels == 12
    assert mapper_list[2].pixels == 16

    galaxy_no_pix_0 = al.Galaxy(
        redshift=0.25,
        mass_profile=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=0.5),
    )
    galaxy_no_pix_1 = al.Galaxy(redshift=0.5)
    galaxy_no_pix_2 = al.Galaxy(redshift=1.5)

    galaxy_pix_0 = al.Galaxy(redshift=0.75, pixelization=pixelization_0)
    galaxy_pix_1 = al.Galaxy(redshift=2.0, pixelization=pixelization_1)

    tracer = al.Tracer(
        galaxies=[
            galaxy_no_pix_0,
            galaxy_no_pix_1,
            galaxy_no_pix_2,
            galaxy_pix_0,
            galaxy_pix_1,
        ]
    )

    tracer_to_inversion = al.TracerToInversion(
        dataset=masked_imaging_7x7, tracer=tracer
    )

    mapper_galaxy_dict = tracer_to_inversion.mapper_galaxy_dict

    mapper_list = list(mapper_galaxy_dict.keys())

    assert mapper_galaxy_dict[mapper_list[0]] == galaxy_pix_0
    assert mapper_galaxy_dict[mapper_list[1]] == galaxy_pix_1

    assert mapper_galaxy_dict[mapper_list[0]] == galaxy_pix_0
    assert mapper_galaxy_dict[mapper_list[1]] == galaxy_pix_1


def test__inversion_imaging_from(grid_2d_7x7, masked_imaging_7x7):
    grids = al.GridsInterface(
        lp=masked_imaging_7x7.grids.lp,
        pixelization=masked_imaging_7x7.grids.pixelization,
        blurring=masked_imaging_7x7.grids.blurring,
        border_relocator=masked_imaging_7x7.grids.border_relocator,
    )

    dataset = al.DatasetInterface(
        data=masked_imaging_7x7.data,
        noise_map=masked_imaging_7x7.noise_map,
        grids=grids,
        psf=masked_imaging_7x7.psf,
    )

    g_linear = al.Galaxy(
        redshift=0.5, light_linear=al.lp_linear.Sersic(centre=(0.05, 0.05))
    )

    tracer = al.Tracer(galaxies=[al.Galaxy(redshift=0.5), g_linear])

    tracer_to_inversion = al.TracerToInversion(
        dataset=dataset,
        tracer=tracer,
    )

    inversion = tracer_to_inversion.inversion

    assert inversion.reconstruction[0] == pytest.approx(0.186868464426, 1.0e-2)

    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=0.0),
    )

    g0 = al.Galaxy(redshift=0.5, pixelization=pixelization)

    tracer = al.Tracer(galaxies=[al.Galaxy(redshift=0.5), g0])

    tracer_to_inversion = al.TracerToInversion(
        dataset=dataset,
        tracer=tracer,
    )

    inversion = tracer_to_inversion.inversion

    assert inversion.mapped_reconstructed_operated_data == pytest.approx(
        masked_imaging_7x7.data, 1.0e-2
    )


def test__inversion_interferometer_from(grid_2d_7x7, interferometer_7):
    interferometer_7.data = al.Visibilities.ones(shape_slim=(7,))

    grids = al.GridsInterface(
        lp=interferometer_7.grids.lp,
        pixelization=interferometer_7.grids.pixelization,
        blurring=interferometer_7.grids.blurring,
        border_relocator=interferometer_7.grids.border_relocator,
    )

    dataset = al.DatasetInterface(
        data=interferometer_7.data,
        noise_map=interferometer_7.noise_map,
        grids=grids,
        transformer=interferometer_7.transformer,
    )

    g_linear = al.Galaxy(
        redshift=0.5, light_linear=al.lp_linear.Sersic(centre=(0.05, 0.05))
    )

    tracer = al.Tracer(galaxies=[al.Galaxy(redshift=0.5), g_linear])

    tracer_to_inversion = al.TracerToInversion(
        dataset=dataset,
        tracer=tracer,
    )

    inversion = tracer_to_inversion.inversion

    assert inversion.reconstruction[0] == pytest.approx(0.0412484695, 1.0e-5)

    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(7, 7)),
        regularization=al.reg.Constant(coefficient=0.0),
    )

    g0 = al.Galaxy(redshift=0.5, pixelization=pixelization)

    tracer = al.Tracer(galaxies=[al.Galaxy(redshift=0.5), g0])

    tracer_to_inversion = al.TracerToInversion(
        dataset=dataset,
        tracer=tracer,
    )

    inversion = tracer_to_inversion.inversion

    assert inversion.reconstruction[0] == pytest.approx(-0.095834752881, 1.0e-4)
