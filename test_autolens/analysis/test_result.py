import os
import numpy as np
import pytest

import autofit as af
import autolens as al
from autoarray import Array2D

from autolens.analysis import result as res
from autolens.imaging.model.result import ResultImaging

directory = os.path.dirname(os.path.realpath(__file__))


def test__max_log_likelihood_tracer(
    analysis_imaging_7x7,
    tracer_x2_plane_7x7,
):
    model = af.Collection(
        galaxies=af.Collection(
            lens=al.Galaxy(redshift=0.5, mass=al.mp.Isothermal),
            source=al.Galaxy(redshift=1.0, light=al.lp.Sersic),
        )
    )

    search = al.m.MockSearch(name="test_search_2")

    result = search.fit(model=model, analysis=analysis_imaging_7x7)

    assert isinstance(result.max_log_likelihood_tracer, al.Tracer)
    assert isinstance(result.max_log_likelihood_tracer.galaxies[0], al.Galaxy)


def test__source_plane_light_profile_centre(analysis_imaging_7x7):
    lens = al.Galaxy(redshift=0.5, light=al.lp.SersicSph(intensity=1.0))

    source = al.Galaxy(
        redshift=1.0, light=al.lp.SersicSph(centre=(1.0, 2.0), intensity=2.0)
    )

    tracer = al.Tracer(galaxies=[lens, source])

    samples_summary = al.m.MockSamplesSummary(max_log_likelihood_instance=tracer)

    result = res.Result(samples_summary=samples_summary, analysis=analysis_imaging_7x7)

    assert result.source_plane_light_profile_centre_from().in_list == [(1.0, 2.0)]

    source_0 = al.Galaxy(
        redshift=1.0,
        light=al.lp.SersicSph(centre=(1.0, 2.0), intensity=2.0),
        light1=al.lp.SersicSph(centre=(3.0, 4.0), intensity=2.0),
    )

    source_1 = al.Galaxy(
        redshift=1.0, light=al.lp.SersicSph(centre=(5.0, 6.0), intensity=2.0)
    )

    tracer = al.Tracer(galaxies=[lens, source_0, source_1])

    samples_summary = al.m.MockSamplesSummary(max_log_likelihood_instance=tracer)

    result = res.Result(samples_summary=samples_summary, analysis=analysis_imaging_7x7)

    assert result.source_plane_light_profile_centre_from().in_list == [(1.0, 2.0)]

    source_0 = al.Galaxy(
        redshift=1.0, light=al.lp.SersicSph(centre=(1.0, 2.0), intensity=2.0)
    )

    source_1 = al.Galaxy(
        redshift=2.0, light=al.lp.SersicSph(centre=(5.0, 6.0), intensity=2.0)
    )

    tracer = al.Tracer(galaxies=[lens, source_0, source_1])

    samples_summary = al.m.MockSamplesSummary(max_log_likelihood_instance=tracer)

    result = res.Result(samples_summary=samples_summary, analysis=analysis_imaging_7x7)

    assert result.source_plane_light_profile_centre_from().in_list == [(5.0, 6.0)]

    tracer = al.Tracer(galaxies=[al.Galaxy(redshift=0.5)])

    samples_summary = al.m.MockSamplesSummary(max_log_likelihood_instance=tracer)

    result = res.Result(samples_summary=samples_summary, analysis=analysis_imaging_7x7)

    assert result.source_plane_light_profile_centre_from() == None


def test__source_plane_inversion_centre(analysis_imaging_7x7):
    lens = al.Galaxy(redshift=0.5, light=al.lp.SersicSph(intensity=1.0))

    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform((3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )

    source = al.Galaxy(redshift=1.0, pixelization=pixelization)

    tracer = al.Tracer(galaxies=[lens, source])

    samples_summary = al.m.MockSamplesSummary(max_log_likelihood_instance=tracer)

    result = ResultImaging(
        samples_summary=samples_summary, analysis=analysis_imaging_7x7
    )

    inversion = result.max_log_likelihood_fit.inversion

    assert (
        result.source_plane_inversion_centre_from().in_list[0]
        == inversion.max_pixel_centre().in_list[0]
    )

    lens = al.Galaxy(redshift=0.5, light=al.lp.SersicSph(intensity=1.0))
    source = al.Galaxy(redshift=1.0)

    tracer = al.Tracer(galaxies=[lens, source])

    samples_summary = al.m.MockSamplesSummary(max_log_likelihood_instance=tracer)

    result = ResultImaging(
        samples_summary=samples_summary, analysis=analysis_imaging_7x7
    )

    assert result.source_plane_inversion_centre_from() == None

    lens = al.Galaxy(redshift=0.5, light=al.lp_linear.Sersic())
    source = al.Galaxy(redshift=1.0)

    tracer = al.Tracer(galaxies=[lens, source])

    samples_summary = al.m.MockSamplesSummary(max_log_likelihood_instance=tracer)

    result = ResultImaging(
        samples_summary=samples_summary, analysis=analysis_imaging_7x7
    )

    assert result.source_plane_inversion_centre_from() == None


def test__source_plane_centre(analysis_imaging_7x7):
    lens = al.Galaxy(redshift=0.5, light=al.lp.SersicSph(intensity=1.0))

    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform((3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )

    source = al.Galaxy(
        redshift=1.0,
        light=al.lp.SersicSph(centre=(9.0, 8.0), intensity=2.0),
        pixelization=pixelization,
    )

    tracer = al.Tracer(galaxies=[lens, source])

    samples_summary = al.m.MockSamplesSummary(max_log_likelihood_instance=tracer)

    result = ResultImaging(
        samples_summary=samples_summary, analysis=analysis_imaging_7x7
    )

    assert result.source_plane_centre_from().in_list[0] == pytest.approx(
        (-0.916666673333, -0.916666), 1.0e-4
    )


def test__image_plane_multiple_image_positions(analysis_imaging_7x7):
    lens = al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(
            centre=(0.001, 0.001), einstein_radius=1.0, ell_comps=(0.0, 0.111111)
        ),
    )

    source = al.Galaxy(
        redshift=1.0,
        light1=al.lp.SersicSph(centre=(0.0, 0.05), intensity=2.0),
    )

    tracer = al.Tracer(galaxies=[lens, source])

    samples_summary = al.m.MockSamplesSummary(max_log_likelihood_instance=tracer)

    result = ResultImaging(
        samples_summary=samples_summary, analysis=analysis_imaging_7x7
    )

    multiple_images = result.image_plane_multiple_image_positions()

    assert pytest.approx((0.968719, 0.366210), 1.0e-2) in multiple_images.in_list


def test__positions_threshold_from(analysis_imaging_7x7):
    tracer = al.Tracer(
        galaxies=[
            al.Galaxy(
                redshift=0.5,
                mass=al.mp.Isothermal(
                    centre=(0.1, 0.0), einstein_radius=1.0, ell_comps=(0.0, 0.0)
                ),
            ),
            al.Galaxy(redshift=1.0, bulge=al.lp.SersicSph(centre=(0.0, 0.0))),
        ]
    )

    samples_summary = al.m.MockSamplesSummary(max_log_likelihood_instance=tracer)

    result = res.Result(samples_summary=samples_summary, analysis=analysis_imaging_7x7)

    assert result.positions_threshold_from() == pytest.approx(0.0019501455, 1.0e-4)
    assert result.positions_threshold_from(factor=5.0) == pytest.approx(
        0.0097507278035, 1.0e-4
    )
    assert result.positions_threshold_from(minimum_threshold=10.0) == pytest.approx(
        10.0, 1.0e-4
    )
    assert result.positions_threshold_from(
        positions=al.Grid2DIrregular([(0.0, 0.0)])
    ) == pytest.approx(0.0, 1.0e-4)


def test__positions_likelihood_from(analysis_imaging_7x7):
    tracer = al.Tracer(
        galaxies=[
            al.Galaxy(
                redshift=0.5,
                mass=al.mp.Isothermal(
                    centre=(0.1, 0.0), einstein_radius=1.0, ell_comps=(0.0, 0.0)
                ),
            ),
            al.Galaxy(redshift=1.0, bulge=al.lp.SersicSph(centre=(0.0, 0.0))),
        ]
    )

    samples_summary = al.m.MockSamplesSummary(max_log_likelihood_instance=tracer)

    result = res.Result(samples_summary=samples_summary, analysis=analysis_imaging_7x7)

    positions_likelihood = result.positions_likelihood_from(
        factor=0.1, minimum_threshold=0.2
    )

    assert isinstance(positions_likelihood, al.PositionsLH)
    assert positions_likelihood.threshold == pytest.approx(0.2, 1.0e-4)


def test__positions_likelihood_from__mass_centre_radial_distance_min(
    analysis_imaging_7x7,
):
    tracer = al.Tracer(
        galaxies=[
            al.Galaxy(
                redshift=0.5,
                mass=al.mp.Isothermal(
                    centre=(0.1, 0.0), einstein_radius=1.0, ell_comps=(0.0, 0.0)
                ),
            ),
            al.Galaxy(redshift=1.0, bulge=al.lp.SersicSph(centre=(0.0, 0.0))),
        ]
    )

    samples_summary = al.m.MockSamplesSummary(max_log_likelihood_instance=tracer)

    result = res.Result(samples_summary=samples_summary, analysis=analysis_imaging_7x7)

    positions_likelihood = result.positions_likelihood_from(
        factor=0.1, minimum_threshold=0.2, mass_centre_radial_distance_min=0.1
    )

    assert isinstance(positions_likelihood, al.PositionsLH)
    assert len(positions_likelihood.positions) == 2
    assert positions_likelihood.positions[0] == pytest.approx(
        (-1.00097656e00, 5.63818622e-04), 1.0e-4
    )
    assert positions_likelihood.positions[1] == pytest.approx(
        (1.00097656e00, -5.63818622e-04), 1.0e-4
    )


def test__results_include_mask__available_as_property(
    analysis_imaging_7x7, masked_imaging_7x7, samples_summary_with_result
):
    result = res.ResultDataset(
        samples_summary=samples_summary_with_result,
        analysis=analysis_imaging_7x7,
    )

    assert (result.mask == masked_imaging_7x7.mask).all()


def test___image_dict(analysis_imaging_7x7):
    galaxies = af.ModelInstance()
    galaxies.lens = al.Galaxy(redshift=0.5)
    galaxies.source = al.Galaxy(redshift=1.0)

    instance = af.ModelInstance()
    instance.galaxies = galaxies

    result = ResultImaging(
        samples_summary=al.m.MockSamplesSummary(max_log_likelihood_instance=instance),
        analysis=analysis_imaging_7x7,
    )

    image_dict = result.model_image_galaxy_dict

    assert isinstance(image_dict[str(("galaxies", "lens"))], Array2D)
    assert isinstance(image_dict[str(("galaxies", "source"))], Array2D)

    result.instance.galaxies.lens = al.Galaxy(redshift=0.5)

    image_dict = result.model_image_galaxy_dict

    assert (image_dict[str(("galaxies", "lens"))].native == np.zeros((7, 7))).all()
    assert isinstance(image_dict[str(("galaxies", "source"))], Array2D)
