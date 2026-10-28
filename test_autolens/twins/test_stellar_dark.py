import json

import numpy as np
import pytest

import autogalaxy as ag
import autolens as al
from autolens.twins._galaxies import galaxy_dict_from
from autolens.twins.transform import Transformation

from test_autolens.twins.conftest import COARSE_APERTURE, GNFW_APERTURE


def _canonical(obj) -> str:
    return json.dumps(al.to_dict(obj), sort_keys=True)


def _aperture_from(record, settings=COARSE_APERTURE):
    return ag.CircularAperture(
        radius=record.procedure["aperture"]["radius"],
        centre=tuple(record.procedure["aperture"]["centre"]),
        **settings,
    )


def _setup_grid(twin):
    mask = twin.setup.mask
    return al.Grid2D.uniform(
        shape_native=mask.shape_native,
        pixel_scales=mask.pixel_scales,
        origin=mask.origin,
        over_sample_size=1,
    )


def test__galaxies_are_addressed_by_tracer_json_keys(witness_reference):
    # The tracer reloaded from files/tracer.json keeps the model's galaxy names.
    assert list(galaxy_dict_from(witness_reference.tracer)) == ["lens", "source"]


def test__galaxies_of_a_list_tracer_are_addressed_by_position(tracer_x2_plane_7x7):
    assert list(galaxy_dict_from(tracer_x2_plane_7x7)) == [
        "galaxy_0",
        "galaxy_1",
        "galaxy_2",
    ]


def test__witness__50_50_budget_inside_parent_einstein_radius(
    witness_baseline, witness_twin
):
    assert isinstance(
        al.twins.transform.ReplaceTotalMassWithStellarPlusDark(
            galaxy="lens", light="bulge", scale_radius_kpc=10.0
        ),
        Transformation,
    )

    record = witness_twin.records[-1]
    parent_lens = witness_baseline.tracer.galaxies.lens
    lens = witness_twin.tracer.galaxies.lens

    # The SIE is removed; the light basis is now a stellar lmp.Gaussian basis; a dark halo is added.
    assert not hasattr(lens, "mass")
    assert all(
        isinstance(profile, al.lmp.Gaussian) for profile in lens.bulge.profile_list
    )
    assert isinstance(lens.dark, al.mp.NFWSph)

    grid = _setup_grid(witness_twin)
    aperture = _aperture_from(record)
    mass_reference = aperture.mass_from(mass_obj=parent_lens.mass, grid=grid)
    mass_stellar = aperture.mass_from(mass_obj=lens.bulge, grid=grid)
    mass_dark = aperture.mass_from(mass_obj=lens.dark, grid=grid)

    assert mass_reference == pytest.approx(
        record.procedure["derived"]["mass_reference"], rel=1e-12
    )
    assert mass_stellar / mass_reference == pytest.approx(0.5, abs=1.0e-3)
    assert mass_dark / mass_reference == pytest.approx(0.5, abs=1.0e-3)

    # theta_E,parent is the tracer-level Einstein radius of the whole parent.
    assert record.procedure["einstein_radius_parent"] == pytest.approx(
        al.LensCalc.from_tracer(witness_baseline.tracer).einstein_radius_from(
            grid=witness_baseline.setup.grid
        ),
        rel=1e-12,
    )
    assert record.procedure["aperture"]["centre"] == [0.0, 0.0]

    # Shear, light, and the source are unchanged.
    assert _canonical(lens.shear) == _canonical(parent_lens.shear)
    assert _canonical(witness_twin.tracer.galaxies.source) == _canonical(
        witness_baseline.tracer.galaxies.source
    )
    assert np.array_equal(
        lens.image_2d_from(grid=grid).array, parent_lens.image_2d_from(grid=grid).array
    )


def test__witness__record(witness_twin):
    record = witness_twin.records[-1]

    assert record.name == "ReplaceTotalMassWithStellarPlusDark"
    assert record.preserved == ["projected mass of lens inside theta_E,parent"]
    assert record.changed == [
        "galaxies.lens.mass",
        "galaxies.lens.bulge",
        "galaxies.lens.dark",
    ]
    assert any("external shear kept" in effect for effect in record.side_effects)
    assert any(
        "Einstein radius of the twin" in effect for effect in record.side_effects
    )

    assert record.requested["galaxy"] == "lens"
    assert record.requested["scale_radius_kpc"] == 10.0
    assert record.requested["fractions"] == {"stellar": 0.5, "dark": 0.5}
    assert record.requested["dark_cls"] == "autogalaxy.profiles.mass.dark.nfw.NFWSph"

    procedure = record.procedure
    assert "mass_budget.rst" in procedure["text"]
    for key in (
        "mass_reference",
        "mass_light",
        "mass_to_light_ratio",
        "kappa_s",
        "f_dark",
        "scale_radius",
    ):
        assert key in procedure["derived"]
    assert procedure["redshift"] == 0.5
    assert procedure["einstein_radius_twin"] > 0.0

    truth = record.truth
    lens = witness_twin.tracer.galaxies.lens
    assert (
        truth["mass_to_light_ratio"] == lens.bulge.profile_list[0].mass_to_light_ratio
    )
    assert truth["kappa_s"] == lens.dark.kappa_s
    assert truth["scale_radius_arcsec"] == lens.dark.scale_radius
    assert truth["inner_slope"] == 1.0
    assert truth["f_dark"] == 0.5

    def has_none(value):
        if value is None:
            return True
        if isinstance(value, dict):
            return any(has_none(item) for item in value.values())
        if isinstance(value, list):
            return any(has_none(item) for item in value)
        return False

    assert not has_none(record.requested)
    assert not has_none(record.procedure)


def test__galaxy_addressed_by_role(witness_baseline, witness_twin):
    by_role = witness_baseline.apply(
        al.twins.transform.ReplaceTotalMassWithStellarPlusDark(
            galaxy="main_deflector",
            light="bulge",
            scale_radius_kpc=10.0,
            aperture_settings=COARSE_APERTURE,
        )
    )
    assert _canonical(by_role.tracer) == _canonical(witness_twin.tracer)

    with pytest.raises(ValueError, match="No galaxy named"):
        witness_baseline.apply(
            al.twins.transform.ReplaceTotalMassWithStellarPlusDark(
                galaxy="dwarf", light="bulge", scale_radius_kpc=10.0
            )
        )


def test__does_not_mutate(witness_reference, witness_baseline, stellar_dark):
    tracer_before = _canonical(witness_baseline.tracer)
    reference_before = _canonical(witness_reference.tracer)
    setup_before = _canonical(witness_baseline.setup)

    first = witness_baseline.apply(stellar_dark)
    second = witness_baseline.apply(stellar_dark)

    assert _canonical(witness_baseline.tracer) == tracer_before
    assert _canonical(witness_reference.tracer) == reference_before
    assert _canonical(witness_baseline.setup) == setup_before
    assert witness_baseline.records == ()
    assert first.tracer is not witness_baseline.tracer
    assert first.setup is witness_baseline.setup
    assert _canonical(first.tracer) == _canonical(second.tracer)
    assert first.records == second.records


def test__record__round_trips_dict(witness_twin):
    record = witness_twin.records[-1]
    reloaded = al.from_dict(json.loads(json.dumps(al.to_dict(record))))
    assert reloaded == record
    assert reloaded.truth == record.truth

    reloaded_tracer = al.from_dict(json.loads(_canonical(witness_twin.tracer)))
    assert _canonical(reloaded_tracer) == _canonical(witness_twin.tracer)


def test__record__round_trips_simulated_twin_directory(witness_twin, tmp_path):
    simulated = witness_twin.simulate(seed=3)
    output = tmp_path / "simulated"
    simulated.output_to(output)

    assert (output / "reference_tracer.json").exists()

    with open(output / "twin.json") as f:
        twin_json = json.load(f)
    truth = twin_json["truth"]
    assert len(truth) == 1
    assert truth[0]["transformation"] == "ReplaceTotalMassWithStellarPlusDark"
    assert truth[0]["record_index"] == 0
    for key in (
        "mass_to_light_ratio",
        "kappa_s",
        "scale_radius_arcsec",
        "scale_radius_kpc",
        "inner_slope",
        "f_dark",
        "einstein_radius_parent",
    ):
        assert isinstance(truth[0][key], float)

    reloaded = al.twins.SimulatedTwin.from_directory(output)
    assert reloaded.twin.records == witness_twin.records
    assert _canonical(reloaded.truth) == _canonical(witness_twin.tracer)
    assert np.array_equal(
        reloaded.resimulate().dataset.data.native.array,
        simulated.dataset.data.native.array,
    )


def test__scale_radius_kpc_uses_setup_cosmology(witness_baseline):
    cosmology = al.cosmo.FlatLambdaCDM(H0=50.0, Om0=0.4)
    twin = witness_baseline.with_setup(
        witness_baseline.setup.with_(cosmology=cosmology)
    )

    transformed = twin.apply(
        al.twins.transform.ReplaceTotalMassWithStellarPlusDark(
            galaxy="lens",
            light="bulge",
            scale_radius_kpc=8.0,
            aperture_settings=COARSE_APERTURE,
        )
    )
    procedure = transformed.records[-1].procedure
    expected = 8.0 * cosmology.arcsec_per_kpc_from(redshift=0.5)

    assert procedure["cosmology_source"] == "setup"
    assert procedure["cosmology"].endswith("FlatLambdaCDM")
    assert procedure["scale_radius_arcsec"] == pytest.approx(expected, rel=1e-12)
    assert transformed.tracer.galaxies.lens.dark.scale_radius == pytest.approx(
        expected, rel=1e-12
    )

    default = witness_baseline.apply(
        al.twins.transform.ReplaceTotalMassWithStellarPlusDark(
            galaxy="lens",
            light="bulge",
            scale_radius_kpc=8.0,
            aperture_settings=COARSE_APERTURE,
        )
    )
    default_procedure = default.records[-1].procedure
    assert default_procedure["cosmology_source"] == "tracer"
    assert default_procedure["scale_radius_arcsec"] == pytest.approx(
        8.0 * al.cosmo.Planck15().arcsec_per_kpc_from(redshift=0.5), rel=1e-12
    )
    assert default_procedure["scale_radius_arcsec"] != pytest.approx(expected)


def test__gnfw_inner_slope(witness_baseline):
    transformed = witness_baseline.apply(
        al.twins.transform.ReplaceTotalMassWithStellarPlusDark(
            galaxy="lens",
            light="bulge",
            scale_radius_kpc=10.0,
            f_dark=0.3,
            dark_cls=al.mp.gNFWSph,
            aperture_settings=GNFW_APERTURE,
            inner_slope=1.5,
        )
    )
    record = transformed.records[-1]
    dark = transformed.tracer.galaxies.lens.dark

    assert isinstance(dark, al.mp.gNFWSph)
    assert dark.inner_slope == 1.5
    assert record.requested["dark_fixed"] == {"inner_slope": 1.5}
    assert record.procedure["derived"]["fixed"] == {"inner_slope": 1.5}
    assert record.truth["inner_slope"] == 1.5
    assert record.truth["f_dark"] == 0.3

    grid = _setup_grid(transformed)
    aperture = _aperture_from(record, settings=GNFW_APERTURE)
    mass_reference = record.procedure["derived"]["mass_reference"]
    assert aperture.mass_from(
        mass_obj=dark, grid=grid
    ) / mass_reference == pytest.approx(0.3, abs=1.0e-3)

    reloaded = al.from_dict(json.loads(json.dumps(al.to_dict(record))))
    assert reloaded == record
    reloaded_tracer = al.from_dict(json.loads(_canonical(transformed.tracer)))
    assert reloaded_tracer.galaxies.lens.dark.inner_slope == 1.5


def test__budget(witness_baseline):
    with pytest.raises(NotImplementedError):
        al.twins.transform.ReplaceTotalMassWithStellarPlusDark(
            galaxy="lens",
            light="bulge",
            scale_radius_kpc=10.0,
            budget={
                "lens": {"stellar": 0.5, "dark": 0.5},
                "dwarf": {"stellar": 0.5, "dark": 0.5},
            },
        )

    with pytest.raises(ValueError):
        al.twins.transform.ReplaceTotalMassWithStellarPlusDark(
            galaxy="lens", light="bulge", scale_radius_kpc=10.0, f_dark=1.5
        )

    transformed = witness_baseline.apply(
        al.twins.transform.ReplaceTotalMassWithStellarPlusDark(
            galaxy="lens",
            light="bulge",
            scale_radius_kpc=10.0,
            budget=ag.MassBudget(
                aperture=None, fractions={"lens": {"stellar": 0.8, "dark": 0.2}}
            ),
            aperture_settings=COARSE_APERTURE,
        )
    )
    assert transformed.records[-1].truth["f_dark"] == 0.2

    with pytest.raises(ValueError, match="budget names"):
        witness_baseline.apply(
            al.twins.transform.ReplaceTotalMassWithStellarPlusDark(
                galaxy="lens",
                light="bulge",
                scale_radius_kpc=10.0,
                budget={"source": {"stellar": 0.5, "dark": 0.5}},
            )
        )


def test__keep_shear_false_removes_shear(witness_baseline):
    transformed = witness_baseline.apply(
        al.twins.transform.ReplaceTotalMassWithStellarPlusDark(
            galaxy="lens",
            light="bulge",
            scale_radius_kpc=10.0,
            keep_shear=False,
            aperture_settings=COARSE_APERTURE,
        )
    )
    record = transformed.records[-1]
    assert not hasattr(transformed.tracer.galaxies.lens, "shear")
    assert "galaxies.lens.shear" in record.changed
    assert hasattr(witness_baseline.tracer.galaxies.lens, "shear")


def test__invalid_light_raises(witness_baseline):
    tracer = witness_baseline.tracer
    galaxies = galaxy_dict_from(tracer)
    sersic_basis = al.lp_basis.Basis(profile_list=[al.lp.SersicSph()])
    lens = al.Galaxy(redshift=0.5, bulge=sersic_basis, mass=galaxies["lens"].mass)
    twin = al.twins.Twin(
        reference=witness_baseline.reference,
        tracer=al.Tracer(galaxies=[lens, galaxies["source"]]),
    )

    with pytest.raises(ag.exc.ProfileException):
        twin.apply(
            al.twins.transform.ReplaceTotalMassWithStellarPlusDark(
                galaxy="galaxy_0", light="bulge", scale_radius_kpc=10.0
            )
        )

    with pytest.raises(ValueError, match="must be an MGE light Basis"):
        twin.apply(
            al.twins.transform.ReplaceTotalMassWithStellarPlusDark(
                galaxy="galaxy_0", light="mass", scale_radius_kpc=10.0
            )
        )
