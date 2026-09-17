"""
Tests for `al.model_util.mass_field_from` — the composer for the external-field model that
goes in a lens model's `fields` collection.

The behaviour worth pinning is the *shared* centre: the `MassSheet` and `ExternalPotential`
in the field are expanded about the primary deflector's centre, and they must share the lens
mass's centre prior object rather than get a copy of it. A copy would silently add two free
parameters and let the two centres drift apart in the search.
"""

import pytest

import autofit as af
import autolens as al


@pytest.fixture(name="lens")
def make_lens():
    return af.Model(al.Galaxy, redshift=0.5, mass=af.Model(al.mp.Isothermal))


def test__shear_only_is_the_default(lens):
    field = al.model_util.mass_field_from(lens=lens)

    assert field.cls is al.MassField
    assert hasattr(field, "shear")
    assert not hasattr(field, "mass_sheet")
    assert not hasattr(field, "potential")


def test__mass_sheet_alone(lens):
    field = al.model_util.mass_field_from(lens=lens, shear=False, mass_sheet=True)

    assert not hasattr(field, "shear")
    assert hasattr(field, "mass_sheet")
    assert not hasattr(field, "potential")


def test__potential_alone(lens):
    field = al.model_util.mass_field_from(lens=lens, shear=False, potential=True)

    assert not hasattr(field, "shear")
    assert not hasattr(field, "mass_sheet")
    assert hasattr(field, "potential")


def test__all_three_together(lens):
    field = al.model_util.mass_field_from(
        lens=lens, shear=True, mass_sheet=True, potential=True
    )

    assert hasattr(field, "shear")
    assert hasattr(field, "mass_sheet")
    assert hasattr(field, "potential")


def test__the_centres_are_the_same_prior_object_as_the_lens_mass_centre(lens):
    field = al.model_util.mass_field_from(
        lens=lens, shear=True, mass_sheet=True, potential=True
    )

    assert field.mass_sheet.centre is lens.mass.centre
    assert field.potential.centre is lens.mass.centre


def test__the_shared_centre_costs_no_extra_dimensions(lens):
    tied = af.Collection(
        galaxies=af.Collection(lens=lens),
        fields=af.Collection(
            field=al.model_util.mass_field_from(
                lens=lens, shear=False, mass_sheet=True, potential=True
            )
        ),
    )

    untied = af.Collection(
        galaxies=af.Collection(lens=lens),
        fields=af.Collection(
            field=af.Model(
                al.MassField,
                redshift=0.5,
                mass_sheet=af.Model(al.mp.MassSheet),
                potential=af.Model(al.mp.ExternalPotential),
            )
        ),
    )

    # Two centres, two components each, shared away.
    assert tied.prior_count == untied.prior_count - 4


def test__a_composed_collection_builds_an_instance_with_matching_centres(lens):
    model = af.Collection(
        galaxies=af.Collection(lens=lens),
        fields=af.Collection(
            field=al.model_util.mass_field_from(lens=lens, potential=True)
        ),
    )

    instance = model.instance_from_prior_medians()

    assert isinstance(instance.fields.field, al.MassField)
    assert instance.fields.field.potential.centre == pytest.approx(
        instance.galaxies.lens.mass.centre, 1.0e-12
    )


def test__redshift_defaults_to_the_lens_and_can_be_overridden(lens):
    assert al.model_util.mass_field_from(lens=lens).redshift == 0.5
    assert al.model_util.mass_field_from(lens=lens, redshift=0.8).redshift == 0.8


def test__a_lens_model_with_no_mass_raises_and_names_the_model():
    lens = af.Model(al.Galaxy, redshift=0.5, bulge=af.Model(al.lp.Sersic))

    with pytest.raises(ValueError, match="no `mass` attribute"):
        al.model_util.mass_field_from(lens=lens)
