"""
``SimulatedTwin``: one noisy observation of a twin together with its ground truth,
seed, noise route, simulator settings and the twin (reference provenance plus
transformation chain) it came from.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Union


import autoarray as aa
from autonerves.class_path import get_class_path
from autonerves.dictable import from_dict, from_json, output_to_json, to_dict

from autolens.twins import _util
from autolens.twins._util import Frozen
from autolens.twins.reference import ReferenceModel
from autolens.twins.setup import ObservationSetup
from autolens.twins.twin import Twin

logger = logging.getLogger(__name__)


class SimulatedTwin(Frozen):
    def __init__(
        self,
        dataset: aa.Imaging,
        truth,
        seed: int,
        noise: str,
        simulator: Dict,
        twin: Twin,
    ):
        """
        A simulated observation of ``twin``.

        Parameters
        ----------
        dataset
            The simulated ``al.Imaging``, masked and over-sampled like the twin's setup.
        truth
            The ground-truth tracer that was observed (``twin.tracer``).
        seed
            The noise seed.
        noise
            ``"observed"`` or ``"poisson"``.
        simulator
            The route and settings used (``route``, ``exposure_time``, ...).
        twin
            The twin that was simulated.
        """
        self._freeze_set("dataset", dataset)
        self._freeze_set("truth", truth)
        self._freeze_set("seed", int(seed))
        self._freeze_set("noise", str(noise))
        simulator = dict(simulator)
        # `al.from_dict` drops None-valued dict entries (PyAutoFit's dict parser);
        # restore the standard keys so a reloaded twin compares equal.
        for key in ("route", "noise", "exposure_time", "background_sky_level"):
            simulator.setdefault(key, None)
        self._freeze_set("simulator", simulator)
        self._freeze_set("twin", twin)

    def __repr__(self):
        return (
            f"SimulatedTwin(lens_id={self.twin.reference.lens_id!r}, "
            f"seed={self.seed}, noise={self.noise!r})"
        )

    def dict(self) -> dict:
        """
        The ``autonerves.dictable`` dictionary.

        ``al.Imaging`` does not round-trip through the dictable contract, so the
        ``dataset`` argument is written as a call of ``dataset_via_setup_from`` (a
        function ``al.from_dict`` resolves by class path and calls with its
        arguments) holding the setup, data and noise map; reloading rebuilds the
        masked, over-sampled dataset with ``ObservationSetup.dataset_from``.
        """
        return {
            "type": "instance",
            "class_path": get_class_path(type(self)),
            "arguments": {
                "dataset": {
                    "type": "instance",
                    "class_path": f"{dataset_via_setup_from.__module__}."
                    f"{dataset_via_setup_from.__qualname__}",
                    "arguments": {
                        "setup": to_dict(self.twin.setup),
                        "data": to_dict(self.dataset.data),
                        "noise_map": to_dict(self.dataset.noise_map),
                    },
                },
                "truth": to_dict(self.truth),
                "seed": self.seed,
                "noise": self.noise,
                "simulator": to_dict(self.simulator),
                "twin": to_dict(self.twin),
            },
        }

    def resimulate(self) -> "SimulatedTwin":
        """Simulate the same twin again with the stored seed and noise route."""
        return self.twin.simulate(seed=self.seed, noise=self.noise)

    def twin_json(self) -> dict:
        """
        The provenance written to ``twin.json``: the reference provenance (lens id,
        fit path, hashes, ingest time, labels, roles, library versions at ingest),
        the library versions now, the transformation records (dictable
        dictionaries), the setup fields that differ from the reference setup, and
        the seed, noise route and simulator settings.
        """
        return {
            "reference": self.twin.reference.provenance,
            "versions": _util.library_versions(),
            "records": [to_dict(record) for record in self.twin.records],
            "setup_overrides": self.twin.setup_overrides(),
            "seed": self.seed,
            "noise": self.noise,
            "simulator": self.simulator,
        }

    def output_to(self, directory: Union[str, Path]):
        """
        Write the observation and everything needed to reproduce it:

        - ``data.fits``, ``noise_map.fits``, ``psf.fits``, ``mask.fits``: native
          arrays (no visualisation zoom) with the mask's pixel scales in the header.
        - ``truth_tracer.json``: the ground truth.
        - ``twin.json``: see ``twin_json``.
        - ``setup.json``: the twin's complete setup. The FITS files alone cannot
          rebuild it (over-sampling, cosmology, exposure time and sky level are not
          in them, and a Poisson noise map differs from the setup's).
        - ``reference_setup.json`` / ``reference_tracer.json``: only when the twin's
          setup / tracer differ from the reference's, so the reference can be
          rebuilt without the original fit directory.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        mask = self.dataset.mask

        _util.output_array_to_fits(
            _util.native_values_from(self.dataset.data),
            mask=mask,
            file_path=directory / "data.fits",
            ext_name="data",
        )
        _util.output_array_to_fits(
            _util.native_values_from(self.dataset.noise_map),
            mask=mask,
            file_path=directory / "noise_map.fits",
            ext_name="noise_map",
        )
        if self.dataset.psf is not None:
            _util.output_array_to_fits(
                _util.native_values_from(self.dataset.psf.kernel),
                mask=self.dataset.psf.kernel.mask,
                file_path=directory / "psf.fits",
                ext_name="psf",
            )
        _util.output_mask_to_fits(mask, directory / "mask.fits")

        output_to_json(self.truth, directory / "truth_tracer.json")
        output_to_json(self.twin.setup, directory / "setup.json")

        reference = self.twin.reference
        if _canonical(reference.setup) != _canonical(self.twin.setup):
            output_to_json(reference.setup, directory / "reference_setup.json")
        if _canonical(reference.tracer) != _canonical(self.truth):
            output_to_json(reference.tracer, directory / "reference_tracer.json")

        with open(directory / "twin.json", "w") as f:
            json.dump(self.twin_json(), f, indent=4)

    @classmethod
    def from_directory(cls, directory: Union[str, Path]) -> "SimulatedTwin":
        """
        Reload a simulated twin written by ``output_to``.

        The reference is resolved from its ``fit_path`` when that directory still
        holds a ``dataset.fits`` and ``tracer.json`` whose SHA-256 match the stored
        provenance (setup fields not in the FITS are taken from the stored
        reference setup); otherwise it is rebuilt from the embedded provenance, the
        stored reference tracer (or the truth tracer, when they are the same) and
        the stored reference setup (or the twin setup).
        """
        directory = Path(directory)

        with open(directory / "twin.json") as f:
            twin_json = json.load(f)

        setup = from_json(file_path=directory / "setup.json")
        truth = from_json(file_path=directory / "truth_tracer.json")

        reference = _reference_from(
            directory=directory,
            provenance=twin_json["reference"],
            twin_setup=setup,
            truth=truth,
        )

        twin = Twin(
            reference=reference,
            tracer=truth,
            setup=setup,
            records=[from_dict(record) for record in twin_json["records"]],
        )

        data = _util.ndarray_from_fits(directory / "data.fits")
        noise_map = _util.ndarray_from_fits(directory / "noise_map.fits")

        return cls(
            dataset=setup.dataset_from(data=data, noise_map=noise_map),
            truth=truth,
            seed=twin_json["seed"],
            noise=twin_json["noise"],
            simulator=twin_json["simulator"],
            twin=twin,
        )


def dataset_via_setup_from(setup: ObservationSetup, data, noise_map) -> aa.Imaging:
    """
    ``setup.dataset_from(data, noise_map)``; the reload hook ``SimulatedTwin.dict``
    points ``al.from_dict`` at.
    """
    return setup.dataset_from(data=data, noise_map=noise_map)


def _canonical(obj) -> str:
    return json.dumps(to_dict(obj), sort_keys=True)


def _reference_from(
    directory: Path, provenance: dict, twin_setup: ObservationSetup, truth
) -> ReferenceModel:
    reference_setup_path = directory / "reference_setup.json"
    reference_tracer_path = directory / "reference_tracer.json"

    stored_setup = (
        from_json(file_path=reference_setup_path)
        if reference_setup_path.exists()
        else twin_setup
    )

    fit_path = provenance.get("fit_path")
    if fit_path is not None:
        fit_path = Path(fit_path)
        tracer_path = fit_path / "files" / "tracer.json"
        dataset_path = fit_path / "image" / "dataset.fits"
        if (
            tracer_path.exists()
            and dataset_path.exists()
            and _util.sha256_of_file(tracer_path) == provenance.get("tracer_sha256")
            and _util.sha256_of_file(dataset_path) == provenance.get("dataset_sha256")
        ):
            setup = ObservationSetup.from_fit_directory(fit_path).with_(
                cosmology=stored_setup.cosmology,
                exposure_time=stored_setup.exposure_time,
                background_sky_level=stored_setup.background_sky_level,
            )
            return ReferenceModel(
                tracer=from_json(file_path=tracer_path),
                setup=setup,
                provenance=provenance,
            )
        logger.info(
            f"Reference fit_path {fit_path} is missing or its files changed; "
            f"rebuilding the reference from the stored provenance."
        )

    tracer = (
        from_json(file_path=reference_tracer_path)
        if reference_tracer_path.exists()
        else truth
    )
    return ReferenceModel(tracer=tracer, setup=stored_setup, provenance=provenance)
