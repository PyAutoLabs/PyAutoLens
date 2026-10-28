import json
import shutil

import numpy as np
import pytest
import yaml
from astropy.io import fits

import autofit as af
import autolens as al
from autolens.twins.transform import Transformation


def _canonical(obj) -> str:
    return json.dumps(al.to_dict(obj), sort_keys=True)


class _FitStub:
    """
    The part of a `PyAutoFit` aggregator `Fit` that `_imaging_from` reads: `value("dataset")` returning the
    `dataset.fits` HDU list written by `AnalysisImaging.save_attributes`.
    """

    def __init__(self, paths):
        self.paths = paths
        self.children = []

    def value(self, name):
        if name == "dataset":
            return fits.open(self.paths.image_path / "dataset.fits")
        return None


@pytest.fixture(name="fit_paths")
def make_fit_paths(tmp_path, analysis_imaging_7x7, tracer_x2_plane_7x7):
    """
    A model-fit output directory as `AnalysisImaging` writes it: `image/dataset.fits` (via
    `save_attributes`) and `files/tracer.json` (via `save_results`).
    """
    model = af.Collection(
        galaxies=af.Collection(
            lens=al.Galaxy(redshift=0.5), source=al.Galaxy(redshift=1.0)
        )
    )
    paths = af.DirectoryPaths(name="twin_fit", path_prefix=str(tmp_path / "fits"))

    analysis_imaging_7x7.save_attributes(paths=paths)
    analysis_imaging_7x7.save_results(
        paths=paths,
        result=al.m.MockResult(
            max_log_likelihood_tracer=tracer_x2_plane_7x7, model=model
        ),
    )

    return paths


@pytest.fixture(name="reference")
def make_reference(fit_paths):
    return al.twins.ReferenceModel.from_fit_directory(
        fit_paths.output_path,
        lens_id="lens_7x7",
        labels=("fixture", "x2_plane"),
        roles={"galaxy_0": "main_deflector"},
    )


@pytest.fixture(name="twin")
def make_twin(reference):
    return al.twins.Twin.baseline(reference)


def test__exports():
    for name in (
        "ObservationSetup",
        "ReferenceModel",
        "Twin",
        "SimulatedTwin",
        "TransformationRecord",
        "transform",
    ):
        assert hasattr(al.twins, name)
    assert isinstance(al.twins.transform.Identity(), Transformation)


def test__from_fit_directory__ingests_reference(
    reference, fit_paths, tracer_x2_plane_7x7
):
    assert isinstance(reference, al.twins.ReferenceModel)
    assert isinstance(reference.setup, al.twins.ObservationSetup)
    assert _canonical(reference.tracer) == _canonical(tracer_x2_plane_7x7)

    provenance = reference.provenance
    assert provenance["lens_id"] == "lens_7x7"
    assert provenance["labels"] == ["fixture", "x2_plane"]
    assert provenance["roles"] == {"galaxy_0": "main_deflector"}
    assert provenance["fit_path"] == str(fit_paths.output_path.resolve())
    assert len(provenance["dataset_sha256"]) == 64
    assert len(provenance["tracer_sha256"]) == 64
    assert provenance["ingested_at"].endswith("+00:00")
    for name in ("autofit", "autoarray", "autogalaxy", "autolens", "numpy", "python"):
        assert name in provenance["versions"]
    assert provenance["versions"]["numpy"] == np.__version__


def test__reference__is_immutable(reference):
    with pytest.raises(AttributeError):
        reference.tracer = None
    with pytest.raises(AttributeError):
        reference.setup = None
    with pytest.raises(AttributeError):
        del reference.tracer

    provenance = reference.provenance
    provenance["lens_id"] = "changed"
    assert reference.provenance["lens_id"] == "lens_7x7"


def test__setup_from_fit_directory__matches_aggregator_reload(fit_paths, reference):
    from autogalaxy.aggregator.imaging.imaging import _imaging_from

    aggregated = _imaging_from(fit=_FitStub(paths=fit_paths))[0]
    rebuilt = al.twins.ObservationSetup.dataset_from_fit_directory(
        fit_paths.output_path
    )
    setup = reference.setup

    assert np.array_equal(rebuilt.data.native.array, aggregated.data.native.array)
    assert np.array_equal(
        rebuilt.noise_map.native.array, aggregated.noise_map.native.array
    )
    assert np.array_equal(
        rebuilt.psf.kernel.native.array, aggregated.psf.kernel.native.array
    )
    assert np.array_equal(np.asarray(rebuilt.mask), np.asarray(aggregated.mask))

    assert np.array_equal(np.asarray(setup.mask), np.asarray(aggregated.mask))
    assert setup.pixel_scales == aggregated.mask.pixel_scales
    assert setup.origin == aggregated.mask.origin
    assert np.array_equal(
        setup.noise_map.native.array, aggregated.noise_map.native.array
    )
    assert np.array_equal(
        setup.psf.kernel.native.array, aggregated.psf.kernel.native.array
    )
    assert np.array_equal(
        np.asarray(setup.over_sample_size_lp),
        np.asarray(aggregated.grids.lp.over_sample_size),
    )


def test__setup_from_fit_directory__matches_fitted_dataset(
    reference, masked_imaging_7x7
):
    setup = reference.setup

    assert setup.mask.pixels_in_mask == masked_imaging_7x7.mask.pixels_in_mask
    assert np.array_equal(setup.noise_map.array, masked_imaging_7x7.noise_map.array)
    assert np.allclose(
        setup.psf.kernel.native.array, masked_imaging_7x7.psf.kernel.native.array
    )
    assert np.array_equal(setup.grid.array, masked_imaging_7x7.grids.lp.array)


def test__setup__with_returns_modified_copy(reference):
    setup = reference.setup
    modified = setup.with_(exposure_time=300.0, background_sky_level=0.1)

    assert modified.exposure_time == 300.0
    assert modified.background_sky_level == 0.1
    assert setup.exposure_time is None
    assert np.array_equal(modified.noise_map.array, setup.noise_map.array)

    with pytest.raises(TypeError):
        setup.with_(not_a_field=1)


def test__from_result(masked_imaging_7x7, tracer_x2_plane_7x7):
    result = al.m.MockResult(max_log_likelihood_tracer=tracer_x2_plane_7x7)

    reference = al.twins.ReferenceModel.from_result(
        result=result, dataset=masked_imaging_7x7, lens_id="from_result"
    )

    assert reference.tracer is tracer_x2_plane_7x7
    assert np.array_equal(
        reference.setup.noise_map.array, masked_imaging_7x7.noise_map.array
    )
    assert reference.provenance["fit_path"] is None
    assert len(reference.provenance["dataset_sha256"]) == 64
    assert len(reference.provenance["tracer_sha256"]) == 64


def test__baseline__noise_free_image_equals_simulator(twin, reference):
    image = twin.noise_free_image_2d()

    simulator = al.SimulatorImaging(
        exposure_time=1.0,
        psf=al.twins.setup._copy(reference.setup.psf),
        add_poisson_noise_to_data=False,
        include_poisson_noise_in_noise_map=False,
        noise_seed=1,
    )
    simulated = simulator.via_tracer_from(
        tracer=reference.tracer, grid=reference.setup.grid
    )
    simulated = al.Array2D(values=simulated.data.native, mask=reference.setup.mask)

    assert np.array_equal(image.array, simulated.array)

    dataset = reference.setup.dataset_from(data=reference.setup.noise_map)
    blurred = reference.tracer.blurred_image_2d_from(
        grid=dataset.grids.lp, blurring_grid=dataset.grids.blurring, psf=dataset.psf
    )
    assert np.array_equal(image.array, blurred.array)


def test__baseline__tracer_is_a_copy(twin, reference):
    assert twin.tracer is not reference.tracer
    assert _canonical(twin.tracer) == _canonical(reference.tracer)
    assert twin.records == ()


def test__simulate__seed_determinism_and_sensitivity(twin):
    simulated_1 = twin.simulate(seed=1)
    simulated_1_again = twin.simulate(seed=1)
    simulated_2 = twin.simulate(seed=2)

    assert np.array_equal(
        simulated_1.dataset.data.array, simulated_1_again.dataset.data.array
    )
    assert not np.array_equal(
        simulated_1.dataset.data.array, simulated_2.dataset.data.array
    )

    assert np.array_equal(
        simulated_1.dataset.noise_map.array, twin.setup.noise_map.array
    )
    assert simulated_1.seed == 1
    assert simulated_1.noise == "observed"
    assert simulated_1.simulator["route"] == "tracer.blurred_image_2d_from"
    assert simulated_1.truth is twin.tracer
    assert simulated_1.dataset.mask.pixels_in_mask == twin.setup.mask.pixels_in_mask

    residual = simulated_1.dataset.data.array - twin.noise_free_image_2d().array
    expected = np.random.default_rng(1).normal(0.0, twin.setup.noise_map.array)
    assert np.allclose(residual, expected)


def test__simulate__observed_route_leaves_global_random_state(twin):
    np.random.seed(5)
    expected = np.random.rand()

    np.random.seed(5)
    twin.simulate(seed=1)
    assert np.random.rand() == expected


def test__simulate__noise_route_errors(twin):
    with pytest.raises(ValueError, match="exposure_time"):
        twin.simulate(seed=1, noise="poisson")

    no_sky = twin.with_setup(twin.setup.with_(exposure_time=300.0))
    with pytest.raises(ValueError, match="background_sky_level"):
        no_sky.simulate(seed=1, noise="poisson")

    with pytest.raises(ValueError, match="noise"):
        twin.simulate(seed=1, noise="gaussian")

    with pytest.raises(ValueError, match="seed"):
        twin.simulate(seed=-1)


def test__simulate__poisson_route(twin):
    poisson_twin = twin.with_setup(
        twin.setup.with_(exposure_time=300.0, background_sky_level=0.1)
    )

    simulated = poisson_twin.simulate(seed=3, noise="poisson")
    simulated_again = poisson_twin.simulate(seed=3, noise="poisson")

    assert np.array_equal(
        simulated.dataset.data.array, simulated_again.dataset.data.array
    )
    assert simulated.simulator["route"] == "al.SimulatorImaging.via_tracer_from"
    assert simulated.simulator["exposure_time"] == 300.0
    assert not np.array_equal(
        simulated.dataset.noise_map.array, twin.setup.noise_map.array
    )
    assert poisson_twin.setup_overrides().keys() == {
        "exposure_time",
        "background_sky_level",
    }


def test__identity__record_and_apply(twin):
    applied = twin.apply(al.twins.transform.Identity())

    assert len(applied.records) == 1
    record = applied.records[0]
    assert record.name == "Identity"
    assert record.changed == []
    assert record.side_effects == []
    assert record.preserved == []

    assert twin.records == ()
    assert applied.reference is twin.reference
    assert np.array_equal(
        applied.noise_free_image_2d().array, twin.noise_free_image_2d().array
    )

    reloaded = al.from_dict(json.loads(json.dumps(al.to_dict(record))))
    assert reloaded == record


def test__apply__requires_a_record(twin):
    class Broken:
        def apply(self, twin):
            return twin.tracer, twin.setup, {"name": "not a record"}

    with pytest.raises(TypeError):
        twin.apply(Broken())


def test__to_dict_from_dict__round_trips(reference, twin):
    record = al.twins.TransformationRecord(
        name="Example",
        requested={"factor": 1.5},
        procedure={"text": "scaled", "derived": 2.0},
        changed=["galaxies.lens.mass"],
        side_effects=["einstein_radius"],
    )
    simulated = twin.apply(al.twins.transform.Identity()).simulate(seed=4)

    for obj in (reference.setup, reference, record, twin, simulated):
        dictionary = _canonical(obj)
        reloaded = al.from_dict(json.loads(dictionary))
        assert type(reloaded) is type(obj)
        assert _canonical(reloaded) == dictionary

    reloaded = al.from_dict(json.loads(_canonical(simulated)))
    assert np.array_equal(reloaded.dataset.data.array, simulated.dataset.data.array)
    assert reloaded.dataset.mask.pixels_in_mask == simulated.dataset.mask.pixels_in_mask
    assert np.array_equal(
        reloaded.resimulate().dataset.data.array, simulated.dataset.data.array
    )


def test__reference__output_to_and_from_directory(reference, tmp_path):
    reference.output_to(tmp_path / "reference")

    for name in ("tracer.json", "provenance.json", "labels.yaml"):
        assert (tmp_path / "reference" / name).exists()
    for name in ("mask.fits", "noise_map.fits", "psf.fits", "setup.json"):
        assert (tmp_path / "reference" / "setup" / name).exists()

    with open(tmp_path / "reference" / "labels.yaml") as f:
        labels = yaml.safe_load(f)
    assert labels["lens_id"] == "lens_7x7"
    assert labels["labels"] == ["fixture", "x2_plane"]

    reloaded = al.twins.ReferenceModel.from_directory(tmp_path / "reference")

    assert reloaded.provenance == reference.provenance
    assert _canonical(reloaded.tracer) == _canonical(reference.tracer)
    assert _canonical(reloaded.setup) == _canonical(reference.setup)
    assert isinstance(al.from_json(tmp_path / "reference" / "tracer.json"), al.Tracer)


def test__simulated_twin__output_from_directory_resimulate(twin, tmp_path):
    simulated = twin.simulate(seed=1)
    output = tmp_path / "simulated"
    simulated.output_to(output)

    for name in (
        "data.fits",
        "noise_map.fits",
        "psf.fits",
        "mask.fits",
        "truth_tracer.json",
        "twin.json",
        "setup.json",
    ):
        assert (output / name).exists()

    with open(output / "twin.json") as f:
        twin_json = json.load(f)
    assert twin_json["seed"] == 1
    assert twin_json["noise"] == "observed"
    assert twin_json["records"] == []
    assert twin_json["setup_overrides"] == {}
    assert twin_json["reference"]["lens_id"] == "lens_7x7"
    assert (
        twin_json["reference"]["dataset_sha256"]
        == twin.reference.provenance["dataset_sha256"]
    )
    assert "autolens" in twin_json["versions"]
    assert isinstance(al.from_json(output / "truth_tracer.json"), al.Tracer)

    stored = al.twins._util.ndarray_from_fits(output / "data.fits")

    reloaded = al.twins.SimulatedTwin.from_directory(output)
    assert reloaded.seed == 1
    assert reloaded.noise == "observed"
    assert reloaded.twin.reference.provenance == twin.reference.provenance
    assert np.array_equal(reloaded.dataset.data.native.array, stored)

    resimulated = reloaded.resimulate()
    assert np.array_equal(resimulated.dataset.data.native.array, stored)


def test__simulated_twin__from_directory_without_fit_path(twin, fit_paths, tmp_path):
    transformed = twin.apply(al.twins.transform.Identity()).with_setup(
        twin.setup.with_(exposure_time=300.0, background_sky_level=0.1)
    )
    simulated = transformed.simulate(seed=7)
    output = tmp_path / "simulated"
    simulated.output_to(output)

    assert (output / "reference_setup.json").exists()
    assert not (output / "reference_tracer.json").exists()

    with open(output / "twin.json") as f:
        twin_json = json.load(f)
    assert [record["arguments"]["name"] for record in twin_json["records"]] == [
        "Identity"
    ]
    assert set(twin_json["setup_overrides"]) == {
        "exposure_time",
        "background_sky_level",
    }

    shutil.rmtree(fit_paths.output_path)

    reloaded = al.twins.SimulatedTwin.from_directory(output)
    stored = al.twins._util.ndarray_from_fits(output / "data.fits")

    assert reloaded.twin.reference.setup.exposure_time is None
    assert reloaded.twin.setup.exposure_time == 300.0
    assert len(reloaded.twin.records) == 1
    assert np.array_equal(reloaded.resimulate().dataset.data.native.array, stored)
