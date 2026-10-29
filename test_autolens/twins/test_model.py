import numpy as np
import pytest

import autofit as af
import autolens as al
from autolens.twins.model import DEFAULT_PRIORS, stellar_dark_record_from

from test_autolens.twins.conftest import COARSE_APERTURE, GNFW_APERTURE


def _truth_vector_from(model, twin, record):
    """
    The parameter vector of ``model`` at the twin's values: the record's truth for the
    stellar-dark parameters and the twin's shear for the free shear components.
    """
    truth = record.truth
    tracer = twin.tracer
    values = []
    for paths in model.all_paths:
        path = paths[0]
        name = path[-1]
        if name == "mass_to_light_ratio":
            values.append(truth["mass_to_light_ratio"])
        elif name == "kappa_s":
            values.append(truth["kappa_s"])
        elif name == "scale_radius":
            values.append(truth["scale_radius_arcsec"])
        elif name == "inner_slope":
            values.append(truth["inner_slope"])
        elif name in ("gamma_1", "gamma_2"):
            if path[0] == "fields":
                shear = getattr(tracer.fields[0], path[1])
            else:
                shear = getattr(getattr(tracer.galaxies, path[1]), path[2])
            values.append(getattr(shear, name))
        else:
            raise AssertionError(f"unexpected free parameter {path}")
    return values


def _grid():
    return al.Grid2D.uniform(shape_native=(8, 8), pixel_scales=0.3, over_sample_size=1)
    # An even shape keeps (0, 0), where a gNFW convergence diverges, off the grid.


def _assert_lens_reproduced(model, twin, record):
    instance = model.instance_from_vector(
        vector=_truth_vector_from(model, twin, record)
    )
    name = record.truth["galaxy"]
    grid = _grid()

    twin_lens = getattr(twin.tracer.galaxies, name)
    model_lens = getattr(instance.galaxies, name)

    convergence_twin = twin_lens.convergence_2d_from(grid=grid).array
    convergence_model = model_lens.convergence_2d_from(grid=grid).array
    assert np.max(np.abs(convergence_model - convergence_twin)) < 1.0e-10

    deflections_twin = twin_lens.deflections_yx_2d_from(grid=grid).array
    deflections_model = model_lens.deflections_yx_2d_from(grid=grid).array
    assert np.max(np.abs(deflections_model - deflections_twin)) < 1.0e-10

    return instance


def _reference_from(tracer, masked_imaging_7x7):
    return al.twins.ReferenceModel.from_result(
        result=al.m.MockResult(max_log_likelihood_tracer=tracer),
        dataset=masked_imaging_7x7,
        lens_id="model_test",
    )


def _field_shear_tracer():
    """The witness lens with its external shear held in a tracer MassField."""
    from test_autolens.twins.conftest import make_witness_tracer

    witness = make_witness_tracer()
    lens = witness.galaxies.lens
    lens = al.Galaxy(redshift=lens.redshift, bulge=lens.bulge, mass=lens.mass)
    field = al.MassField(
        redshift=0.5, shear=al.mp.ExternalShear(gamma_1=0.03, gamma_2=-0.02)
    )
    return al.Tracer(
        galaxies=af.ModelInstance({"lens": lens, "source": witness.galaxies.source}),
        fields=[field],
    )


def test__stellar_dark_model__layout_and_prior_count(witness_twin):
    source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.SersicSph)
    model = al.twins.stellar_dark_model_from(witness_twin, source_model=source)

    # M/L (one, shared) + kappa_s + scale_radius + shear (2) + source Sersic (5).
    assert model.prior_count == 1 + 2 + 2 + 5

    lens = model.galaxies.lens
    members = lens.bulge.profile_list
    assert len(members) == 3
    assert all(member.cls is al.lmp.Gaussian for member in members)
    assert len({id(member.mass_to_light_ratio) for member in members}) == 1
    assert isinstance(members[0].mass_to_light_ratio, af.LogUniformPrior)

    twin_bulge = witness_twin.tracer.galaxies.lens.bulge
    for member, profile in zip(members, twin_bulge.profile_list):
        assert member.sigma == profile.sigma
        assert member.intensity == profile.intensity
        assert tuple(member.centre) == tuple(profile.centre)

    assert lens.dark.cls is al.mp.NFWSph
    assert tuple(lens.dark.centre) == (0.0, 0.0)
    assert lens.shear.cls is al.mp.ExternalShear
    assert not hasattr(lens, "mass")
    assert model.galaxies.source is source

    # Prior medians build an instance and a tracer.
    instance = model.instance_from_prior_medians()
    al.Tracer(galaxies=list(instance.galaxies))


def test__stellar_dark_model__instance_at_truth_reproduces_the_twin(witness_twin):
    source = witness_twin.tracer.galaxies.source
    model = al.twins.stellar_dark_model_from(witness_twin, "lens", source_model=source)
    record = stellar_dark_record_from(witness_twin)

    assert model.prior_count == 1 + 2 + 2

    instance = _assert_lens_reproduced(model, witness_twin, record)

    grid = _grid()
    tracer = al.Tracer(galaxies=list(instance.galaxies))
    assert (
        np.max(
            np.abs(
                tracer.deflections_yx_2d_from(grid=grid).array
                - witness_twin.tracer.deflections_yx_2d_from(grid=grid).array
            )
        )
        < 1.0e-10
    )


def test__stellar_dark_model__fixed_scale_radius_shear_and_prior_overrides(
    witness_twin,
):
    source = witness_twin.tracer.galaxies.source
    kappa_s_prior = af.LogUniformPrior(lower_limit=1.0e-3, upper_limit=1.0)
    model = al.twins.stellar_dark_model_from(
        witness_twin,
        source_model=source,
        free_scale_radius=False,
        free_shear=False,
        priors={"kappa_s": kappa_s_prior},
    )
    record = stellar_dark_record_from(witness_twin)

    assert model.prior_count == 2
    assert model.galaxies.lens.dark.kappa_s is kappa_s_prior
    assert model.galaxies.lens.dark.scale_radius == record.truth["scale_radius_arcsec"]
    assert isinstance(model.galaxies.lens.shear, al.mp.ExternalShear)

    with pytest.raises(ValueError, match="Unknown priors keys"):
        al.twins.stellar_dark_model_from(
            witness_twin, source_model=source, priors={"kapa_s": kappa_s_prior}
        )


def test__stellar_dark_model__default_priors_do_not_know_the_truth(witness_twin):
    source = witness_twin.tracer.galaxies.source
    model = al.twins.stellar_dark_model_from(witness_twin, source_model=source)
    dark = model.galaxies.lens.dark

    for prior, key in (
        (
            model.galaxies.lens.bulge.profile_list[0].mass_to_light_ratio,
            "mass_to_light_ratio",
        ),
        (dark.kappa_s, "kappa_s"),
        (dark.scale_radius, "scale_radius_arcsec"),
    ):
        default = DEFAULT_PRIORS[key]()
        assert type(prior) is type(default)
        assert prior.lower_limit == default.lower_limit
        assert prior.upper_limit == default.upper_limit


def test__stellar_dark_model__gnfw_free_inner_slope(witness_baseline):
    twin = witness_baseline.apply(
        al.twins.transform.ReplaceTotalMassWithStellarPlusDark(
            galaxy="lens",
            light="bulge",
            scale_radius_kpc=10.0,
            f_dark=0.5,
            dark_cls=al.mp.gNFWSph,
            inner_slope=1.3,
            aperture_settings=GNFW_APERTURE,
        )
    )
    source = twin.tracer.galaxies.source
    record = stellar_dark_record_from(twin)

    model = al.twins.stellar_dark_model_from(twin, source_model=source)
    assert model.galaxies.lens.dark.cls is al.mp.gNFWSph
    assert isinstance(model.galaxies.lens.dark.inner_slope, af.UniformPrior)
    assert model.prior_count == 1 + 3 + 2
    assert al.twins.truth_paths_from(model, "lens")["inner_slope"] == (
        "galaxies",
        "lens",
        "dark",
        "inner_slope",
    )

    _assert_lens_reproduced(model, twin, record)

    fixed = al.twins.stellar_dark_model_from(
        twin, source_model=source, free_inner_slope=False
    )
    assert fixed.galaxies.lens.dark.inner_slope == 1.3
    assert fixed.prior_count == 1 + 2 + 2


def test__stellar_dark_model__nfw_rejects_free_inner_slope(witness_twin):
    with pytest.raises(ValueError, match="no inner_slope"):
        al.twins.stellar_dark_model_from(
            witness_twin,
            source_model=witness_twin.tracer.galaxies.source,
            free_inner_slope=True,
        )


def test__stellar_dark_model__field_shear(masked_imaging_7x7):
    twin = al.twins.Twin.baseline(
        _reference_from(_field_shear_tracer(), masked_imaging_7x7)
    ).apply(
        al.twins.transform.ReplaceTotalMassWithStellarPlusDark(
            galaxy="lens",
            light="bulge",
            scale_radius_kpc=10.0,
            aperture_settings=COARSE_APERTURE,
        )
    )
    source = twin.tracer.galaxies.source
    model = al.twins.stellar_dark_model_from(twin, source_model=source)

    # The shear is a free ExternalShear of a MassField under `fields`, as in the
    # workspace's `fields=af.Model(al.MassField, redshift=..., shear=...)`.
    assert model.fields.cls is al.MassField
    assert model.fields.shear.cls is al.mp.ExternalShear
    assert ("fields", "shear", "gamma_1") in [paths[0] for paths in model.all_paths]
    assert model.prior_count == 1 + 2 + 2

    record = stellar_dark_record_from(twin)
    vector = _truth_vector_from(model, twin, record)
    instance = model.instance_from_vector(vector=vector)

    # The tracer AnalysisImaging builds from the instance matches the twin's.
    analysis = al.AnalysisImaging(dataset=masked_imaging_7x7, use_jax=False)
    tracer = analysis.tracer_via_instance_from(instance=instance)
    grid = _grid()
    assert (
        np.max(
            np.abs(
                tracer.deflections_yx_2d_from(grid=grid).array
                - twin.tracer.deflections_yx_2d_from(grid=grid).array
            )
        )
        < 1.0e-10
    )

    fixed = al.twins.stellar_dark_model_from(
        twin, source_model=source, free_shear=False
    )
    assert isinstance(fixed.fields.shear, al.mp.ExternalShear)
    assert fixed.prior_count == 1 + 2


def test__stellar_dark_model__source_model_dict_and_errors(witness_twin):
    source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.SersicSph)
    model = al.twins.stellar_dark_model_from(
        witness_twin, source_model={"source": source}
    )
    assert model.galaxies.source is source

    with pytest.raises(ValueError, match="not in the twin's tracer"):
        al.twins.stellar_dark_model_from(witness_twin, source_model={"src": source})


def test__stellar_dark_model__no_record_raises(witness_baseline):
    with pytest.raises(ValueError, match="no ReplaceTotalMassWithStellarPlusDark"):
        al.twins.stellar_dark_model_from(
            witness_baseline,
            source_model=witness_baseline.tracer.galaxies.source,
        )


def test__truth_paths_from(witness_twin):
    model = al.twins.stellar_dark_model_from(
        witness_twin, source_model=witness_twin.tracer.galaxies.source
    )
    assert al.twins.truth_paths_from(model, "lens") == {
        "mass_to_light_ratio": (
            "galaxies",
            "lens",
            "bulge",
            "profile_list",
            "0",
            "mass_to_light_ratio",
        ),
        "kappa_s": ("galaxies", "lens", "dark", "kappa_s"),
        "scale_radius_arcsec": ("galaxies", "lens", "dark", "scale_radius"),
    }
