"""
Tests for `Tracer(fields=...)` — the `MassField` slot on the tracer.

A `MassField` is the redshift-bearing container for the mass of everything *outside* the
modelled system (an external shear, a mass sheet, an external potential). It is not a
`Galaxy`: it has no light, and it never appears in `tracer.galaxies`.

The contract these tests pin down is a single sentence: **a field is a member of the
lensing system for every mass calculation and invisible to every galaxy-listing surface.**
So `tracer.fields=[MassField(shear=...)]` must give bit-for-bit the same deflections,
convergence, potential and traced grids as attaching that same shear to the lens galaxy
(the galaxy-attached form, which remains fully supported), while `galaxy_image_2d_dict_from`
keys and `extract_attributes_of_galaxies` stay galaxies-only.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import autofit as af
import autolens as al

from autonerves.dictable import from_json, output_to_json

from autolens.lens.tracer import MultiPlaneRedshiftWarning


@pytest.fixture(name="grid")
def make_grid():
    return al.Grid2D.uniform(shape_native=(10, 10), pixel_scales=0.1)


def _shear():
    return al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.06)


def _lens(with_shear: bool = False):
    kwargs = dict(redshift=0.5, mass=al.mp.Isothermal(einstein_radius=1.0))
    if with_shear:
        kwargs["shear"] = _shear()
    return al.Galaxy(**kwargs)


def _source():
    return al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(intensity=1.0))


# ======================================================================================
# storage + the galaxies list is untouched
# ======================================================================================


def test__fields_default_to_an_empty_list():
    tracer = al.Tracer(galaxies=[_lens(), _source()])

    assert tracer.fields == []


def test__fields_are_stored_and_galaxies_are_unchanged():
    field = al.MassField(redshift=0.5, shear=_shear())
    lens, source = _lens(), _source()

    tracer = al.Tracer(galaxies=[lens, source], fields=[field])

    assert tracer.fields == [field]
    assert list(tracer.galaxies) == [lens, source]
    assert all(not isinstance(galaxy, al.MassField) for galaxy in tracer.galaxies)


def test__fields_is_the_last_positional_argument__so_tracer_galaxies_cosmology_still_works():
    """`Tracer(galaxies, cosmology)` is a positional call in the wild; it must keep working."""
    cosmology = al.cosmo.Planck15()

    tracer = al.Tracer([_lens(), _source()], cosmology)

    assert tracer.cosmology is cosmology
    assert tracer.fields == []


def test__members_is_galaxies_then_fields__and_members_ascending_redshift_sorts():
    field = al.MassField(redshift=0.75, shear=_shear())
    lens, source = _lens(), _source()

    tracer = al.Tracer(galaxies=[source, lens], fields=[field])

    assert tracer.members == [source, lens, field]
    assert tracer.members_ascending_redshift == [lens, field, source]
    # `galaxies_ascending_redshift` keeps its galaxies-only meaning.
    assert tracer.galaxies_ascending_redshift == [lens, source]


# ======================================================================================
# lensing quantities: a field equals the same profile attached to the lens galaxy
# ======================================================================================


def test__traced_grid_2d_list_from__field_shear_equals_galaxy_attached_shear(grid):
    via_field = al.Tracer(
        galaxies=[_lens(), _source()],
        fields=[al.MassField(redshift=0.5, shear=_shear())],
    )
    via_galaxy = al.Tracer(galaxies=[_lens(with_shear=True), _source()])

    for grid_field, grid_galaxy in zip(
        via_field.traced_grid_2d_list_from(grid=grid),
        via_galaxy.traced_grid_2d_list_from(grid=grid),
    ):
        assert grid_field == pytest.approx(np.array(grid_galaxy), 1.0e-8)


@pytest.mark.parametrize(
    "method", ["deflections_yx_2d_from", "convergence_2d_from", "potential_2d_from"]
)
def test__lensing_quantities__field_shear_equals_galaxy_attached_shear(grid, method):
    via_field = al.Tracer(
        galaxies=[_lens(), _source()],
        fields=[al.MassField(redshift=0.5, shear=_shear())],
    )
    via_galaxy = al.Tracer(galaxies=[_lens(with_shear=True), _source()])

    assert np.array(getattr(via_field, method)(grid=grid)) == pytest.approx(
        np.array(getattr(via_galaxy, method)(grid=grid)), 1.0e-8
    )


def test__lensing_quantities__a_field_actually_changes_the_answer(grid):
    """Guards against the tests above passing because the field is being ignored."""
    without = al.Tracer(galaxies=[_lens(), _source()])
    with_field = al.Tracer(
        galaxies=[_lens(), _source()],
        fields=[al.MassField(redshift=0.5, shear=_shear())],
    )

    assert not np.allclose(
        np.array(without.deflections_yx_2d_from(grid=grid)),
        np.array(with_field.deflections_yx_2d_from(grid=grid)),
    )

    # A `MassSheet` has convergence where a shear does not, so this pins the convergence path too.
    with_sheet = al.Tracer(
        galaxies=[_lens(), _source()],
        fields=[al.MassField(redshift=0.5, mass_sheet=al.mp.MassSheet(kappa=0.1))],
    )

    assert not np.allclose(
        np.array(without.convergence_2d_from(grid=grid)),
        np.array(with_sheet.convergence_2d_from(grid=grid)),
    )


def test__potential_2d_from__external_potential_field_contributes(grid):
    field = al.MassField(
        redshift=0.5, potential=al.mp.ExternalPotential(gamma_1=0.05, tau_1=0.02)
    )

    without = al.Tracer(galaxies=[_lens(), _source()])
    with_field = al.Tracer(galaxies=[_lens(), _source()], fields=[field])

    assert not np.allclose(
        np.array(without.potential_2d_from(grid=grid)),
        np.array(with_field.potential_2d_from(grid=grid)),
    )


# ======================================================================================
# planes
# ======================================================================================


def test__a_field_at_a_galaxy_redshift_joins_that_plane():
    field = al.MassField(redshift=0.5, shear=_shear())
    lens, source = _lens(), _source()

    tracer = al.Tracer(galaxies=[lens, source], fields=[field])

    assert tracer.plane_redshifts == [0.5, 1.0]
    assert tracer.total_planes == 2
    assert list(tracer.planes[0]) == [lens, field]
    assert list(tracer.planes[1]) == [source]


def test__a_field_at_its_own_redshift_is_a_plane_of_its_own():
    field = al.MassField(redshift=0.75, mass_sheet=al.mp.MassSheet(kappa=0.1))
    lens, source = _lens(), _source()

    tracer = al.Tracer(galaxies=[lens, source], fields=[field])

    assert tracer.plane_redshifts == [0.5, 0.75, 1.0]
    assert tracer.total_planes == 3
    assert list(tracer.planes[1]) == [field]


def test__a_field_only_plane_is_ray_traced__grid_list_has_an_entry_for_it(grid):
    tracer = al.Tracer(
        galaxies=[_lens(), _source()],
        fields=[
            al.MassField(redshift=0.75, mass=al.mp.IsothermalSph(einstein_radius=0.5))
        ],
    )

    traced_grid_list = tracer.traced_grid_2d_list_from(grid=grid)

    assert len(traced_grid_list) == 3


def test__plane_index_via_redshift_from__finds_a_field_only_plane():
    tracer = al.Tracer(
        galaxies=[_lens(), _source()],
        fields=[al.MassField(redshift=0.75, shear=_shear())],
    )

    assert tracer.plane_index_via_redshift_from(redshift=0.75) == 1


# ======================================================================================
# has / cls_list_from / image lists
# ======================================================================================


def test__has__is_true_for_a_profile_held_only_by_a_field():
    tracer = al.Tracer(
        galaxies=[_lens(), _source()],
        fields=[al.MassField(redshift=0.5, shear=_shear())],
    )

    assert tracer.has(cls=al.mp.ExternalShear) is True
    assert (
        al.Tracer(galaxies=[_lens(), _source()]).has(cls=al.mp.ExternalShear) is False
    )


def test__cls_list_from__includes_a_fields_profiles():
    shear = _shear()
    tracer = al.Tracer(
        galaxies=[_lens(), _source()], fields=[al.MassField(redshift=0.5, shear=shear)]
    )

    assert shear in tracer.cls_list_from(cls=al.mp.ExternalShear)
    assert len(tracer.cls_list_from(cls=al.mp.MassProfile)) == 2


def test__image_2d_list_from__is_unaffected_by_a_field(grid):
    without = al.Tracer(galaxies=[_lens(), _source()])
    with_field = al.Tracer(
        galaxies=[_lens(), _source()],
        fields=[al.MassField(redshift=0.5, shear=_shear())],
    )

    # The image of a plane holding a field is the image of its galaxies alone: the field's
    # contribution is zeros, which is exactly what makes it safe to sum over plane members.
    assert len(with_field.image_2d_list_from(grid=grid)) == 2
    assert np.array(with_field.image_2d_list_from(grid=grid)[0]) == pytest.approx(
        np.array(without.image_2d_list_from(grid=grid)[0]), 1.0e-8
    )


def test__upper_plane_index_with_light_profile__a_field_only_top_plane_does_not_raise_it():
    tracer = al.Tracer(
        galaxies=[_lens(), _source()],
        fields=[al.MassField(redshift=2.0, mass_sheet=al.mp.MassSheet(kappa=0.1))],
    )

    assert tracer.upper_plane_index_with_light_profile == 1


# ======================================================================================
# per-galaxy surfaces stay galaxies-only
# ======================================================================================


def test__galaxy_image_2d_dict_from__keys_are_galaxies_only(grid):
    lens, source = _lens(), _source()
    field = al.MassField(redshift=0.5, shear=_shear())

    tracer = al.Tracer(galaxies=[lens, source], fields=[field])

    assert list(tracer.galaxy_image_2d_dict_from(grid=grid).keys()) == [lens, source]


def test__extract_attributes_of_galaxies__stays_galaxies_only():
    tracer = al.Tracer(
        galaxies=[_lens(), _source()],
        fields=[al.MassField(redshift=0.5, shear=_shear())],
    )

    centres = tracer.extract_attributes_of_galaxies(
        cls=al.mp.MassProfile, attr_name="centre"
    )

    assert len(centres) == 2


def test__extract_attribute__is_member_aware():
    """The whole-tracer attribute list is of *profiles*, so a field's profiles count."""
    tracer = al.Tracer(
        galaxies=[_lens(), _source()],
        fields=[al.MassField(redshift=0.5, shear=_shear())],
    )

    assert len(tracer.extract_attribute(cls=al.mp.MassProfile, attr_name="centre")) == 2


def test__extract_profile_and_plane_index__find_a_fields_component():
    field = al.MassField(redshift=0.5, shear=_shear())
    tracer = al.Tracer(galaxies=[_lens(), _source()], fields=[field])

    assert tracer.extract_profile(profile_name="shear") is field.shear
    assert tracer.extract_plane_index_of_profile(profile_name="shear") == 0


# ======================================================================================
# FitImaging
# ======================================================================================


def test__fit_imaging__galaxy_dict_and_plane_images_match_the_galaxy_attached_form(
    masked_imaging_7x7,
):
    lens, source = _lens(), _source()

    fit_field = al.FitImaging(
        dataset=masked_imaging_7x7,
        tracer=al.Tracer(
            galaxies=[lens, source], fields=[al.MassField(redshift=0.5, shear=_shear())]
        ),
    )
    fit_galaxy = al.FitImaging(
        dataset=masked_imaging_7x7,
        tracer=al.Tracer(galaxies=[_lens(with_shear=True), _source()]),
    )

    assert list(fit_field.galaxy_model_image_dict.keys()) == [lens, source]

    for image_field, image_galaxy in zip(
        fit_field.model_images_of_planes_list, fit_galaxy.model_images_of_planes_list
    ):
        assert np.array(image_field) == pytest.approx(np.array(image_galaxy), 1.0e-8)

    assert fit_field.log_likelihood == pytest.approx(fit_galaxy.log_likelihood, 1.0e-8)


def test__fit_imaging__a_field_only_plane_does_not_break_the_plane_image_list(
    masked_imaging_7x7,
):
    fit = al.FitImaging(
        dataset=masked_imaging_7x7,
        tracer=al.Tracer(
            galaxies=[_lens(), _source()],
            fields=[al.MassField(redshift=0.75, shear=_shear())],
        ),
    )

    assert len(fit.model_images_of_planes_list) == 3
    assert np.array(fit.model_images_of_planes_list[1]) == pytest.approx(0.0, 1.0e-8)


# ======================================================================================
# sliced_tracer_from
# ======================================================================================


def test__sliced_tracer_from__snaps_a_fields_redshift_to_the_nearest_plane():
    lens_g0 = al.Galaxy(redshift=0.5)
    source_g0 = al.Galaxy(redshift=2.0)
    field = al.MassField(redshift=0.42, shear=_shear())

    tracer = al.Tracer.sliced_tracer_from(
        lens_galaxies=[lens_g0],
        line_of_sight_galaxies=[],
        source_galaxies=[source_g0],
        planes_between_lenses=[1, 1],
        cosmology=al.cosmo.Planck15(),
        fields=[field],
    )

    assert field.redshift == 0.5
    assert tracer.fields == [field]
    assert field in list(tracer.planes[tracer.plane_index_via_redshift_from(0.5)])


# ======================================================================================
# json round trip
# ======================================================================================


def test__output_to_and_load_from_json__rebuilds_the_fields():
    json_file = Path(__file__).resolve().parent / "files" / "tracer_fields.json"

    tracer = al.Tracer(
        galaxies=[
            al.Galaxy(
                redshift=0.5, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0)
            ),
            al.Galaxy(redshift=1.0),
        ],
        cosmology=al.cosmo.Planck15(),
        fields=[al.MassField(redshift=0.5, shear=_shear())],
    )

    output_to_json(tracer, file_path=json_file)

    tracer_from_json = from_json(file_path=json_file)

    assert len(tracer_from_json.fields) == 1
    assert isinstance(tracer_from_json.fields[0], al.MassField)
    assert tracer_from_json.fields[0].redshift == 0.5
    assert tracer_from_json.fields[0].shear.gamma_1 == pytest.approx(0.05, 1.0e-12)
    assert tracer_from_json.fields[0].shear.gamma_2 == pytest.approx(0.06, 1.0e-12)


# ======================================================================================
# validation
# ======================================================================================


def test__fields_a_string_is_rejected_even_though_a_string_is_iterable():
    with pytest.raises(TypeError, match="MassField"):
        al.Tracer(galaxies=[_lens()], fields="not a list")


@pytest.mark.parametrize("fields", [42, {"a": 1}, 1.5, b"bytes"])
def test__fields_non_iterable_and_wrong_container_inputs_are_rejected(fields):
    with pytest.raises(TypeError, match="fields"):
        al.Tracer(galaxies=[_lens()], fields=fields)


def test__fields_a_list_entry_without_a_redshift_is_rejected_and_names_the_index():
    with pytest.raises(TypeError, match="index 1"):
        al.Tracer(galaxies=[_lens()], fields=[al.MassField(redshift=0.5), "nope"])


def test__fields_a_galaxy_is_rejected_with_a_message_saying_where_galaxies_go():
    with pytest.raises(TypeError, match="Galaxies go in the galaxies argument"):
        al.Tracer(galaxies=[_lens()], fields=[_source()])


def test__fields_a_model_instance_is_accepted():
    model = af.Collection(
        fields=af.Collection(
            field=af.Model(
                al.MassField, redshift=0.5, shear=af.Model(al.mp.ExternalShear)
            )
        )
    )
    instance = model.instance_from_prior_medians()

    tracer = al.Tracer(galaxies=[_lens(), _source()], fields=instance.fields)

    assert len(tracer.fields) == 1
    assert isinstance(tracer.fields[0], al.MassField)


# ======================================================================================
# MultiPlaneRedshiftWarning
# ======================================================================================


def test__warning__a_tracer_whose_only_mass_is_a_field_behind_all_light_warns():
    with pytest.warns(MultiPlaneRedshiftWarning):
        al.Tracer(
            galaxies=[al.Galaxy(redshift=0.5, bulge=al.lp.Sersic(intensity=1.0))],
            fields=[al.MassField(redshift=1.0, shear=_shear())],
        )


def test__warning__a_field_at_the_lens_redshift_with_light_behind_stays_quiet(recwarn):
    al.Tracer(
        galaxies=[al.Galaxy(redshift=0.5), _source()],
        fields=[al.MassField(redshift=0.5, shear=_shear())],
    )

    assert [w for w in recwarn if w.category is MultiPlaneRedshiftWarning] == []


# ======================================================================================
# identifier pin
# ======================================================================================


_IDENTIFIER_PIN_SNIPPET = """
import autofit as af
import autolens as al

model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy,
            redshift=0.5,
            mass=af.Model(al.mp.Isothermal),
            shear=af.Model(al.mp.ExternalShear),
        ),
        source=af.Model(al.Galaxy, redshift=1.0, bulge=af.Model(al.lp.Sersic)),
    )
)

print(model.identifier)
"""


def test__regression__the_model_identifier_is_unchanged_by_the_fields_slot():
    """
    The identifier of a lens/source model with no fields must not move when the `fields`
    slot is added to `Tracer`: a changed identifier silently orphans every existing
    results folder on disk.

    The model is the one in `_IDENTIFIER_PIN_SNIPPET` above — a lens `Galaxy` with an
    `Isothermal` and an `ExternalShear` at z=0.5 and a `Sersic` source at z=1.0 — and the
    expected value was captured on PyAutoLens `main` at 223132c, BEFORE any of this work
    was written. Never update it to match a new value.

    It runs in a subprocess, deliberately. A model identifier is a function of the model
    *and the active PyAutoFit configuration*, and `test_autolens/conftest.py` pushes the
    suite's own `test_autolens/config` onto `conf.instance` for every test. The pinned
    value was captured under the library's shipped configuration, so the only way to
    assert that exact constant is to evaluate it under that configuration — which is what
    a fresh interpreter with no conftest gives.
    """
    result = subprocess.run(
        [sys.executable, "-c", _IDENTIFIER_PIN_SNIPPET],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "PYAUTO_SKIP_WORKSPACE_VERSION_CHECK": "1"},
    )

    assert result.returncode == 0, result.stderr

    assert result.stdout.strip().splitlines()[-1] == "fef2697b5c32ba56bb18a7baecb7b0f6"
