from types import SimpleNamespace

import pytest
import autofit as af
import autolens as al

from autolens.aggregator.tracer import _tracer_from


@pytest.mark.parametrize("form", ["flat", "collection", "absent", "none", "empty"])
@pytest.mark.parametrize("override", [False, True])
def test__tracer_from__normalizes_saved_and_override_fields(form, override):
    field = af.Model(al.MassField, redshift=0.5, shear=al.mp.ExternalShear)
    model = af.Collection(
        galaxies=af.Collection(
            lens=al.Galaxy(redshift=0.5, mass=al.mp.IsothermalSph()),
            source=al.Galaxy(redshift=1.0, bulge=al.lp.SersicSph()),
        ),
        cosmology=al.cosmo.Planck15(),
    )
    if form == "flat":
        model.fields = field
    elif form == "collection":
        model.fields = af.Collection(field=field)
    elif form == "none":
        model.fields = None
    elif form == "empty":
        model.fields = af.Collection()
    instance = model.instance_from_prior_medians()
    fit = SimpleNamespace(children=[], instance=instance)

    tracers = _tracer_from(fit=fit, instance=instance if override else None)

    assert len(tracers) == 1
    expected = (
        [instance.fields]
        if form == "flat"
        else [instance.fields.field] if form == "collection" else []
    )
    assert tracers[0].fields == expected
    assert len(tracers[0].galaxies) == 2
