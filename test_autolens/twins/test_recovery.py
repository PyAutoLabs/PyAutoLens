import csv
import json

import numpy as np
import pytest

import autofit as af
import autolens as al
from autolens.twins.model import stellar_dark_record_from


def _samples_from(model, centres, widths, total=4000, seed=1):
    """
    Synthetic equal-weight samples: each parameter Gaussian about ``centres[name]``
    with standard deviation ``widths[name]``, by the last element of its path.
    """
    rng = np.random.default_rng(seed)
    columns = [
        rng.normal(centres[paths[0][-1]], widths[paths[0][-1]], size=total)
        for paths in model.all_paths
    ]
    parameter_lists = np.stack(columns, axis=1).tolist()
    sample_list = af.Sample.from_lists(
        model=model,
        parameter_lists=parameter_lists,
        log_likelihood_list=[0.0] * total,
        log_prior_list=[0.0] * total,
        weight_list=[1.0 / total] * total,
    )
    return af.SamplesPDF(model=model, sample_list=sample_list)


@pytest.fixture(name="model_and_truth")
def make_model_and_truth(witness_twin):
    model = al.twins.stellar_dark_model_from(
        witness_twin, source_model=witness_twin.tracer.galaxies.source
    )
    return model, stellar_dark_record_from(witness_twin).truth


def _centres(truth, offset=0.0):
    return {
        "mass_to_light_ratio": truth["mass_to_light_ratio"] * (1.0 + offset),
        "kappa_s": truth["kappa_s"] * (1.0 + offset),
        "scale_radius": truth["scale_radius_arcsec"] * (1.0 + offset),
        "gamma_1": 0.03,
        "gamma_2": -0.02,
    }


def _widths(truth):
    return {
        "mass_to_light_ratio": 0.01 * truth["mass_to_light_ratio"],
        "kappa_s": 0.01 * truth["kappa_s"],
        "scale_radius": 0.01 * truth["scale_radius_arcsec"],
        "gamma_1": 0.001,
        "gamma_2": 0.001,
    }


def test__recovery_table__truth_inside(model_and_truth, tmp_path):
    model, truth = model_and_truth
    samples = _samples_from(model, _centres(truth), _widths(truth))

    table = al.twins.recovery_table_from(samples, truth)

    assert [row.name for row in table] == [
        "mass_to_light_ratio",
        "kappa_s",
        "scale_radius_arcsec",
    ]
    assert table.all_inside
    row = table["kappa_s"]
    assert row.path == ("galaxies", "lens", "dark", "kappa_s")
    assert row.truth == truth["kappa_s"]
    assert row.median == pytest.approx(truth["kappa_s"], rel=2.0e-3)
    assert row.lower < row.median < row.upper
    # 3 sigma of a 1% Gaussian.
    assert row.upper - row.lower == pytest.approx(0.06 * truth["kappa_s"], rel=0.2)

    assert "kappa_s" in str(table)
    assert "yes" in str(table)

    table.to_csv(tmp_path / "recovery.csv")
    with open(tmp_path / "recovery.csv") as f:
        rows = list(csv.DictReader(f))
    assert [row["name"] for row in rows] == [row.name for row in table]
    assert rows[1]["path"] == "galaxies.lens.dark.kappa_s"
    assert float(rows[1]["truth"]) == truth["kappa_s"]
    assert rows[1]["inside"] == "True"


def test__recovery_table__truth_outside(model_and_truth):
    model, truth = model_and_truth
    samples = _samples_from(model, _centres(truth, offset=0.2), _widths(truth))

    table = al.twins.recovery_table_from(samples, truth, sigma=3.0)

    assert not table.all_inside
    assert not any(row.inside for row in table)
    assert "NO" in str(table)

    # The same offset is inside a posterior ten times wider.
    widths = {name: 10.0 * width for name, width in _widths(truth).items()}
    wide = _samples_from(model, _centres(truth, offset=0.2), widths)
    assert al.twins.recovery_table_from(wide, truth, sigma=3.0).all_inside


def test__recovery_table__shared_mass_to_light_ratio_any_member_path(
    model_and_truth,
):
    model, truth = model_and_truth
    samples = _samples_from(model, _centres(truth), _widths(truth))

    member_path = (
        "galaxies",
        "lens",
        "bulge",
        "profile_list",
        "2",
        "mass_to_light_ratio",
    )
    table = al.twins.recovery_table_from(
        samples, truth, paths={"mass_to_light_ratio": member_path}
    )
    assert len(table) == 1
    assert table["mass_to_light_ratio"].median == pytest.approx(
        truth["mass_to_light_ratio"], rel=2.0e-3
    )

    with pytest.raises(ValueError, match="not a free parameter"):
        al.twins.recovery_table_from(
            samples, truth, paths={"kappa_s": ("galaxies", "lens", "kappa_s")}
        )


def test__recovery_table__truth_from_twin_json_and_summary(model_and_truth, tmp_path):
    model, truth = model_and_truth
    samples = _samples_from(model, _centres(truth), _widths(truth))

    twin_json = {
        "truth": [
            {"transformation": "Identity", "record_index": 0, "galaxy": "other"},
            {
                "transformation": "ReplaceTotalMassWithStellarPlusDark",
                "record_index": 1,
                **truth,
            },
        ]
    }
    with open(tmp_path / "twin.json", "w") as f:
        json.dump(twin_json, f)

    from_dict = al.twins.recovery_table_from(samples, truth)
    from_file = al.twins.recovery_table_from(samples, tmp_path / "twin.json")
    from_directory = al.twins.recovery_table_from(samples, str(tmp_path))
    for table in (from_file, from_directory):
        assert [(row.name, row.truth, row.median) for row in table] == [
            (row.name, row.truth, row.median) for row in from_dict
        ]

    summary = samples.summary()
    from_summary = al.twins.recovery_table_from(summary, truth, sigma=3.0)
    for row, expected in zip(from_summary, from_dict):
        assert row.median == pytest.approx(expected.median, rel=1.0e-12)
        assert row.lower == pytest.approx(expected.lower, rel=1.0e-12)
        assert row.upper == pytest.approx(expected.upper, rel=1.0e-12)

    with pytest.raises(ValueError, match="sigma 1 and 3"):
        al.twins.recovery_table_from(summary, truth, sigma=2.0)


def test__end_to_end__simulate_model_nautilus_recovery(
    witness_twin, monkeypatch, tmp_path, plot_patch
):
    """
    simulate -> twin -> model -> Nautilus -> recovery table, with PYAUTO_TEST_MODE=1
    (the real Nautilus backend on a reduced budget), on the 7x7 witness twin. The
    budget is far too small to converge, so only the plumbing is asserted. Figures
    are not rendered (``plot_patch``), which is most of the run time otherwise.
    """
    monkeypatch.setenv("PYAUTO_TEST_MODE", "1")

    simulated = witness_twin.simulate(seed=1)
    simulated.output_to(tmp_path / "twin")

    model = al.twins.stellar_dark_model_from(
        simulated.twin, source_model=simulated.twin.tracer.galaxies.source
    )
    analysis = al.AnalysisImaging(dataset=simulated.dataset, use_jax=False)
    search = af.Nautilus(
        name="twin_end_to_end", path_prefix=str(tmp_path / "output"), n_live=50
    )
    result = search.fit(model=model, analysis=analysis)

    table = al.twins.recovery_table_from(result.samples, tmp_path / "twin")

    assert [row.name for row in table] == [
        "mass_to_light_ratio",
        "kappa_s",
        "scale_radius_arcsec",
    ]
    assert (
        table["kappa_s"].truth
        == stellar_dark_record_from(witness_twin).truth["kappa_s"]
    )
    for row in table:
        assert np.isfinite([row.median, row.lower, row.upper]).all()
        assert row.lower <= row.upper
    table.to_csv(tmp_path / "recovery.csv")
