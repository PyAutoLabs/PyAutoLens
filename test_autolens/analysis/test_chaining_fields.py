import autofit as af
import autolens as al


def test_chained_mass_and_fields_keep_external_shear_in_model_and_tracer():
    stage_1 = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(al.Galaxy, redshift=0.5, mass=af.Model(al.mp.Isothermal)),
            source=af.Model(al.Galaxy, redshift=1.0, bulge=af.Model(al.lp.Sersic)),
        ),
        fields=af.Model(
            al.MassField, redshift=0.5, shear=af.Model(al.mp.ExternalShear)
        ),
    )

    mass, field = al.util.chaining.mass_and_fields_from(
        mass=af.Model(al.mp.PowerLaw),
        mass_result=stage_1.galaxies.lens.mass,
        fields_result=stage_1.fields,
    )
    stage_2 = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(al.Galaxy, redshift=0.5, mass=mass),
            source=stage_1.galaxies.source,
        ),
        fields=field,
    )

    assert ("fields", "shear", "gamma_1") in stage_2.unique_prior_paths

    instance = stage_2.instance_from_prior_medians()
    tracer = al.Tracer(galaxies=instance.galaxies, fields=instance.fields)
    assert len(tracer.fields) == 1
    assert isinstance(tracer.fields[0], al.MassField)

    _, fixed_field = al.util.chaining.mass_and_fields_from(
        mass=af.Model(al.mp.PowerLaw),
        mass_result=stage_1.galaxies.lens.mass,
        fields_result=stage_1.instance_from_prior_medians().fields,
    )
    fixed_stage = af.Collection(galaxies=stage_2.galaxies, fields=fixed_field)
    assert ("fields", "shear", "gamma_1") not in fixed_stage.unique_prior_paths
