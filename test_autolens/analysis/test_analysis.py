import numpy as np
from pathlib import Path
import os
import pytest

from autonerves import conf
from autonerves.dictable import from_json

import autofit as af
import autolens as al
from autolens import exc

directory = Path(__file__).resolve().parent


def test__tracer_for_instance(analysis_imaging_7x7):
    model = af.Collection(
        galaxies=af.Collection(
            lens=al.Galaxy(
                redshift=0.5,
                light=al.lp.SersicSph(intensity=2.0),
                mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.0),
            ),
            source=al.Galaxy(redshift=1.0),
        )
        + af.Collection(
            extra_galaxy=al.Galaxy(
                redshift=0.5,
                light=al.lp.SersicSph(intensity=0.1),
                mass=al.mp.IsothermalSph(einstein_radius=0.2),
            )
        ),
    )

    instance = model.instance_from_unit_vector([])
    tracer = analysis_imaging_7x7.tracer_via_instance_from(instance=instance)

    assert tracer.galaxies[0].redshift == 0.5
    assert tracer.galaxies[0].light.intensity == 2.0
    assert tracer.galaxies[0].mass.centre == pytest.approx((0.0, 0.0), 1.0e-4)
    assert tracer.galaxies[0].mass.einstein_radius == 1.0
    assert tracer.galaxies[2].redshift == 0.5
    assert tracer.galaxies[2].light.intensity == 0.1
    assert tracer.galaxies[2].mass.einstein_radius == 0.2


def test__tracer_for_instance__subhalo_redshift_rescale_used(analysis_imaging_7x7):
    model = af.Collection(
        galaxies=af.Collection(
            lens=al.Galaxy(
                redshift=0.5,
                mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.0),
            ),
            subhalo=al.Galaxy(redshift=0.25, mass=al.mp.NFWSph(centre=(0.1, 0.2))),
            source=al.Galaxy(redshift=1.0),
        )
    )

    instance = model.instance_from_unit_vector([])
    tracer = analysis_imaging_7x7.tracer_via_instance_from(instance=instance)

    assert tracer.galaxies[1].mass.centre == pytest.approx((0.1, 0.2), 1.0e-4)

    model = af.Collection(
        galaxies=af.Collection(
            lens=al.Galaxy(
                redshift=0.5,
                mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.0),
            ),
            subhalo=al.Galaxy(redshift=0.75, mass=al.mp.NFWSph(centre=(0.1, 0.2))),
            source=al.Galaxy(redshift=1.0),
        )
    )

    instance = model.instance_from_unit_vector([])
    tracer = analysis_imaging_7x7.tracer_via_instance_from(instance=instance)

    assert tracer.galaxies[1].mass.centre == pytest.approx((-0.19959, -0.39919), 1.0e-4)


def test__use_border_relocator__determines_if_border_pixel_relocation_is_used(
    masked_imaging_7x7,
):
    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=al.Galaxy(
                redshift=0.5, mass=al.mp.IsothermalSph(einstein_radius=100.0)
            ),
            source=al.Galaxy(redshift=1.0, pixelization=pixelization),
        )
    )

    masked_imaging_7x7.grids.lp.over_sampled[4] = np.array([300.0, 0.0])

    analysis = al.AnalysisImaging(
        dataset=masked_imaging_7x7,
        settings=al.Settings(use_border_relocator=False),
        use_jax=False,
    )

    instance = model.instance_from_unit_vector([])
    fit = analysis.fit_from(instance=instance)

    grid = fit.inversion.linear_obj_list[0].source_plane_data_grid.over_sampled

    assert grid[2] == pytest.approx([-82.99114877, 52.81254922], 1.0e-4)

    analysis = al.AnalysisImaging(
        dataset=masked_imaging_7x7,
        settings=al.Settings(use_border_relocator=True),
        use_jax=False,
    )

    instance = model.instance_from_unit_vector([])
    fit = analysis.fit_from(instance=instance)

    grid = fit.inversion.linear_obj_list[0].source_plane_data_grid.over_sampled

    assert grid[2] == pytest.approx([-82.991148773, 52.81254921], 1.0e-4)


def test__modify_before_fit__inversion_no_positions_likelihood__raises_exception(
    masked_imaging_7x7,
):
    lens = al.Galaxy(redshift=0.5, mass=al.mp.IsothermalSph())

    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(), regularization=al.reg.Constant()
    )

    source = al.Galaxy(redshift=1.0, pixelization=pixelization)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    analysis = al.AnalysisImaging(dataset=masked_imaging_7x7, use_jax=False)

    with pytest.raises(exc.AnalysisException):
        analysis.modify_before_fit(paths=af.DirectoryPaths(), model=model)

    positions_likelihood = al.PositionsLH(
        positions=al.Grid2DIrregular([(1.0, 100.0), (200.0, 2.0)]), threshold=0.01
    )

    analysis = al.AnalysisImaging(
        dataset=masked_imaging_7x7,
        positions_likelihood_list=[positions_likelihood],
        use_jax=False,
    )
    analysis.modify_before_fit(paths=af.DirectoryPaths(), model=model)


def test__save_results__tracer_output_to_json(analysis_imaging_7x7):
    lens = al.Galaxy(redshift=0.5)
    source = al.Galaxy(redshift=1.0)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    tracer = al.Tracer(galaxies=[lens, source])

    paths = af.DirectoryPaths()

    analysis_imaging_7x7.save_results(
        paths=paths,
        result=al.m.MockResult(max_log_likelihood_tracer=tracer, model=model),
    )

    tracer = from_json(file_path=paths._files_path / "tracer.json")

    assert tracer.galaxies[0].redshift == 0.5
    assert tracer.galaxies[1].redshift == 1.0

    os.remove(paths._files_path / "tracer.json")
