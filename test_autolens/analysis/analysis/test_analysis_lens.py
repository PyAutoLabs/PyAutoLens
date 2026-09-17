from pathlib import Path
import pytest

import autofit as af
import autolens as al

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
            extra_galaxyp=al.Galaxy(
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


def test__tracer_for_instance__fields_collection_is_folded_into_tracer_fields(
    analysis_imaging_7x7,
):
    """
    A model may declare an external field beside its galaxies, as
    `af.Collection(galaxies=..., fields=af.Collection(field=af.Model(al.MassField, ...)))`.

    The field must land in `tracer.fields`, and never in `tracer.galaxies`: a `MassField` is
    not a galaxy, and every per-galaxy surface downstream is keyed on that distinction.
    """
    model = af.Collection(
        galaxies=af.Collection(
            lens=al.Galaxy(
                redshift=0.5,
                mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.0),
            ),
            source=al.Galaxy(redshift=1.0, bulge=al.lp.SersicSph(intensity=1.0)),
        ),
        fields=af.Collection(
            field=af.Model(
                al.MassField, redshift=0.5, shear=af.Model(al.mp.ExternalShear)
            )
        ),
    )

    instance = model.instance_from_prior_medians()
    tracer = analysis_imaging_7x7.tracer_via_instance_from(instance=instance)

    assert len(tracer.fields) == 1
    assert isinstance(tracer.fields[0], al.MassField)
    assert tracer.fields[0].redshift == 0.5
    assert all(not isinstance(galaxy, al.MassField) for galaxy in tracer.galaxies)
    assert len(list(tracer.galaxies)) == 2


def test__log_likelihood_function__a_field_shear_matches_the_galaxy_attached_shear(
    analysis_imaging_7x7,
):
    """
    The point of the whole feature: an external shear held by a `MassField` must give the
    same likelihood as the identical shear attached to the lens galaxy. Both instances are
    built at the same prior-median parameter values, so the two models are the same physics
    expressed two ways.
    """
    shear = af.Model(al.mp.ExternalShear)

    field_model = af.Collection(
        galaxies=af.Collection(
            lens=al.Galaxy(
                redshift=0.5,
                mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.0),
            ),
            source=al.Galaxy(redshift=1.0, bulge=al.lp.SersicSph(intensity=1.0)),
        ),
        fields=af.Collection(field=af.Model(al.MassField, redshift=0.5, shear=shear)),
    )

    galaxy_model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=0.5,
                mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.0),
                shear=shear,
            ),
            source=al.Galaxy(redshift=1.0, bulge=al.lp.SersicSph(intensity=1.0)),
        ),
    )

    field_likelihood = analysis_imaging_7x7.log_likelihood_function(
        instance=field_model.instance_from_prior_medians()
    )
    galaxy_likelihood = analysis_imaging_7x7.log_likelihood_function(
        instance=galaxy_model.instance_from_prior_medians()
    )

    assert field_likelihood == pytest.approx(galaxy_likelihood, 1.0e-8)
