import json

import numpy as np
import pytest

import autolens as al
from autolens.interop import coolest as interop_coolest

pytest.importorskip("coolest")


def grid():
    return al.Grid2D.uniform(shape_native=(10, 10), pixel_scales=0.12)


def lens_model_galaxies():
    """
    The Slack-thread parity model: PowerLaw + ExternalShear lens with Sersic
    light, Sersic source.
    """
    lens = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(
            centre=(0.05, -0.03),
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=70.0),
            intensity=1.2,
            effective_radius=0.9,
            sersic_index=3.5,
        ),
        mass=al.mp.PowerLaw(
            centre=(0.05, -0.03),
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=45.0),
            einstein_radius=1.3,
            slope=2.1,
        ),
        shear=al.mp.ExternalShear(gamma_1=0.02, gamma_2=-0.03),
    )
    source = al.Galaxy(
        redshift=1.5,
        bulge=al.lp.Sersic(
            centre=(0.1, 0.2),
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.6, angle=-30.0),
            intensity=0.7,
            effective_radius=0.3,
            sersic_index=1.2,
        ),
    )
    return [lens, source]


def test__to_coolest__writes_template_with_coolest_conventions(tmp_path):
    file_path = interop_coolest.to_coolest(
        galaxies=lens_model_galaxies(), file_path=str(tmp_path / "template")
    )

    with open(file_path) as f:
        template = json.load(f)

    entities = template["lensing_entities"]

    types = [entity["type"] for entity in entities]
    assert types.count("Galaxy") == 2
    assert types.count("MassField") == 1

    lens_entity = [
        e for e in entities if e["type"] == "Galaxy" and e["redshift"] == 0.5
    ][0]
    pemd = lens_entity["mass_model"][0]
    assert pemd["type"] == "PEMD"

    parameters = pemd["parameters"]
    assert parameters["gamma"]["point_estimate"]["value"] == pytest.approx(2.1)
    # theta_E is converted to the COOLEST intermediate-axis convention.
    assert parameters["theta_E"]["point_estimate"]["value"] == pytest.approx(
        1.3 * np.sqrt(0.7) * (2.0 / 1.7) ** (1.0 / 1.1)
    )
    assert parameters["q"]["point_estimate"]["value"] == pytest.approx(0.7)
    assert parameters["phi"]["point_estimate"]["value"] == pytest.approx(-45.0)

    # ag.cosmo.Planck15, the default cosmology, has H0 = 67.74.
    assert template["cosmology"]["H0"] == pytest.approx(67.74, rel=1e-3)


def test__round_trip__tracer_is_numerically_identical(tmp_path):
    galaxies = lens_model_galaxies()
    tracer = al.Tracer(galaxies=galaxies)

    file_path = interop_coolest.to_coolest(
        galaxies=tracer, file_path=str(tmp_path / "template")
    )
    tracer_back = interop_coolest.from_coolest(file_path=file_path)

    deflections = tracer.deflections_yx_2d_from(grid=grid())
    deflections_back = tracer_back.deflections_yx_2d_from(grid=grid())

    assert np.asarray(deflections_back) == pytest.approx(
        np.asarray(deflections), rel=1e-6, abs=1e-10
    )

    image = tracer.image_2d_from(grid=grid())
    image_back = tracer_back.image_2d_from(grid=grid())

    assert np.asarray(image_back) == pytest.approx(
        np.asarray(image), rel=1e-6, abs=1e-12
    )


def test__round_trip__nfw_uses_sigma_crit_from_cosmology(tmp_path):
    galaxies = [
        al.Galaxy(
            redshift=0.3,
            mass=al.mp.NFW(
                centre=(0.0, 0.1),
                ell_comps=al.convert.ell_comps_from(axis_ratio=0.85, angle=10.0),
                kappa_s=0.15,
                scale_radius=6.0,
            ),
        ),
        al.Galaxy(redshift=1.0, bulge=al.lp.SersicSph(intensity=0.5)),
    ]

    # A COOLEST template stores only H0 / Om0, so exact kappa_s recovery is
    # guaranteed only when the same cosmology is supplied to both directions
    # (the template's H0 / Om0 alone recover it to ~1e-5).
    cosmology = al.cosmo.Planck15()

    file_path = interop_coolest.to_coolest(
        galaxies=galaxies, file_path=str(tmp_path / "template"), cosmology=cosmology
    )
    tracer_back = interop_coolest.from_coolest(
        file_path=file_path, cosmology=cosmology
    )

    nfw_back = [
        galaxy for galaxy in tracer_back.galaxies if galaxy.redshift == 0.3
    ][0].mass_0

    assert nfw_back.kappa_s == pytest.approx(0.15, rel=1e-8)
    assert nfw_back.scale_radius == pytest.approx(6.0, rel=1e-8)

    tracer_back_default = interop_coolest.from_coolest(file_path=file_path)
    nfw_default = [
        galaxy for galaxy in tracer_back_default.galaxies if galaxy.redshift == 0.3
    ][0].mass_0

    assert nfw_default.kappa_s == pytest.approx(0.15, rel=1e-3)


def test__round_trip__relative_file_path(tmp_path, monkeypatch):
    # The coolest JSONSerializer rejects relative paths — to_coolest /
    # from_coolest must absolutize them.
    monkeypatch.chdir(tmp_path)

    file_path = interop_coolest.to_coolest(
        galaxies=lens_model_galaxies(), file_path="template"
    )

    assert file_path == str(tmp_path / "template.json")

    tracer_back = interop_coolest.from_coolest(file_path="template.json")

    assert len(tracer_back.galaxies) == 3


def test__from_coolest__missing_point_estimate_raises(tmp_path):
    pytest.importorskip("coolest")
    from coolest.template.lazy import (
        CoordinatesOrigin,
        Galaxy,
        Instrument,
        LensingEntityList,
        LightModel,
        MassModel,
        Observation,
    )
    from coolest.template.json import JSONSerializer
    from coolest.template.standard import COOLEST

    galaxy = Galaxy("lens", 0.5, mass_model=MassModel("SIE"))
    root = COOLEST(
        "MAP",
        CoordinatesOrigin(),
        LensingEntityList(galaxy),
        Observation(),
        Instrument(0.1),
    )
    path = str(tmp_path / "incomplete")
    JSONSerializer(path, obj=root, check_external_files=False).dump_simple()

    with pytest.raises(ValueError):
        interop_coolest.from_coolest(file_path=f"{path}.json")
