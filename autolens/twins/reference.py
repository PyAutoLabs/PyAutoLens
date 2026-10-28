"""
``ReferenceModel``: an immutable snapshot of a fitted lens (the tracer with its
solved intensities, the observation setup and a provenance record) from which any
number of twins are made.
"""

import copy
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import yaml

from autonerves.dictable import from_json, output_to_json

from autolens.twins import _util
from autolens.twins._util import Frozen
from autolens.twins.setup import ObservationSetup

PROVENANCE_DEFAULTS = {
    "lens_id": None,
    "labels": [],
    "roles": {},
    "fit_path": None,
    "dataset_sha256": None,
    "tracer_sha256": None,
    "ingested_at": None,
    "versions": {},
}


def provenance_from(
    lens_id: str,
    labels: Iterable[str] = (),
    roles: Optional[Dict[str, str]] = None,
    fit_path: Optional[str] = None,
    dataset_sha256: Optional[str] = None,
    tracer_sha256: Optional[str] = None,
) -> Dict:
    """The provenance dictionary of a reference model, stamped now."""
    return {
        "lens_id": str(lens_id),
        "labels": [str(label) for label in labels],
        "roles": dict(roles or {}),
        "fit_path": None if fit_path is None else str(fit_path),
        "dataset_sha256": dataset_sha256,
        "tracer_sha256": tracer_sha256,
        "ingested_at": _util.utc_now_iso(),
        "versions": _util.library_versions(),
    }


class ReferenceModel(Frozen):
    def __init__(self, tracer, setup: ObservationSetup, provenance: Dict):
        """
        A fitted lens frozen as the parent of twins.

        Attribute assignment raises. ``provenance`` is returned as a copy, so the
        stored record cannot be edited through it. The tracer and setup are shared
        objects and must be treated as read-only; ``Twin.baseline`` takes its own copy
        of the tracer.

        Parameters
        ----------
        tracer
            The maximum-likelihood ``al.Tracer`` with solved linear intensities.
        setup
            How the lens was observed.
        provenance
            ``lens_id``, ``labels``, ``roles`` (galaxy name -> role), ``fit_path``,
            ``dataset_sha256``, ``tracer_sha256``, ``ingested_at`` (ISO-8601 UTC) and
            ``versions`` (library and Python versions); see ``provenance_from``.
        """
        self._freeze_set("tracer", tracer)
        self._freeze_set("setup", setup)
        provenance = copy.deepcopy(dict(provenance))
        # `al.from_dict` drops None-valued dict entries (PyAutoFit's dict parser);
        # restore the standard keys so a reloaded reference compares equal.
        for key, default in PROVENANCE_DEFAULTS.items():
            provenance.setdefault(key, copy.deepcopy(default))
        if isinstance(provenance["versions"], dict):
            for name in _util.VERSIONED_DISTRIBUTIONS + ("python",):
                provenance["versions"].setdefault(name, None)
        provenance["labels"] = list(provenance["labels"])
        self._freeze_set("_provenance", provenance)

    @property
    def provenance(self) -> Dict:
        return copy.deepcopy(self._provenance)

    @property
    def lens_id(self) -> str:
        return self._provenance.get("lens_id")

    def __repr__(self):
        return (
            f"ReferenceModel(lens_id={self.lens_id!r}, "
            f"fit_path={self._provenance.get('fit_path')!r})"
        )

    @classmethod
    def from_fit_directory(
        cls,
        path: Union[str, Path],
        lens_id: str,
        labels: Iterable[str] = (),
        roles: Optional[Dict[str, str]] = None,
    ) -> "ReferenceModel":
        """
        Ingest a model-fit output directory (the folder holding ``files/tracer.json``
        and ``image/dataset.fits``). The tracer is loaded from ``tracer.json`` and
        therefore carries the solved linear intensities; the hashes are of the two
        files' bytes.
        """
        path = Path(path).resolve()
        tracer_path = path / "files" / "tracer.json"
        dataset_path = path / "image" / "dataset.fits"

        return cls(
            tracer=from_json(file_path=tracer_path),
            setup=ObservationSetup.from_fit_directory(path),
            provenance=provenance_from(
                lens_id=lens_id,
                labels=labels,
                roles=roles,
                fit_path=str(path),
                dataset_sha256=_util.sha256_of_file(dataset_path),
                tracer_sha256=_util.sha256_of_file(tracer_path),
            ),
        )

    @classmethod
    def from_result(
        cls,
        result,
        dataset,
        lens_id: str,
        labels: Iterable[str] = (),
        roles: Optional[Dict[str, str]] = None,
        fit_path: Optional[Union[str, Path]] = None,
    ) -> "ReferenceModel":
        """
        Ingest an in-memory result and the dataset it fitted. With no files to hash,
        ``dataset_sha256`` hashes the native data, noise map, PSF kernel and mask
        arrays and ``tracer_sha256`` hashes the tracer's canonical dictable JSON.
        """
        tracer = result.max_log_likelihood_tracer
        psf_kernel = [] if dataset.psf is None else [dataset.psf.kernel.native.array]
        return cls(
            tracer=tracer,
            setup=ObservationSetup.from_dataset(dataset),
            provenance=provenance_from(
                lens_id=lens_id,
                labels=labels,
                roles=roles,
                fit_path=None if fit_path is None else str(fit_path),
                dataset_sha256=_util.sha256_of_arrays(
                    dataset.data.native.array,
                    dataset.noise_map.native.array,
                    dataset.mask,
                    *psf_kernel,
                ),
                tracer_sha256=_util.sha256_of_object(tracer),
            ),
        )

    def output_to(self, directory: Union[str, Path]):
        """
        Write ``tracer.json``, ``setup/`` (``mask.fits``, ``noise_map.fits``,
        ``psf.fits`` for inspection and ``setup.json``, the complete setup that
        ``from_directory`` reads), ``provenance.json`` and ``labels.yaml``.
        """
        directory = Path(directory)
        setup_directory = directory / "setup"
        setup_directory.mkdir(parents=True, exist_ok=True)

        output_to_json(self.tracer, directory / "tracer.json")
        write_setup(self.setup, setup_directory)

        with open(directory / "provenance.json", "w") as f:
            json.dump(self._provenance, f, indent=4)

        with open(directory / "labels.yaml", "w") as f:
            yaml.safe_dump(
                {
                    "lens_id": self._provenance.get("lens_id"),
                    "labels": list(self._provenance.get("labels", [])),
                    "roles": dict(self._provenance.get("roles", {})),
                },
                f,
                sort_keys=False,
            )

    @classmethod
    def from_directory(cls, directory: Union[str, Path]) -> "ReferenceModel":
        """Reload a reference written by ``output_to``."""
        directory = Path(directory)
        with open(directory / "provenance.json") as f:
            provenance = json.load(f)
        return cls(
            tracer=from_json(file_path=directory / "tracer.json"),
            setup=from_json(file_path=directory / "setup" / "setup.json"),
            provenance=provenance,
        )


def write_setup(setup: ObservationSetup, directory: Path):
    """``mask.fits``, ``noise_map.fits``, ``psf.fits`` and ``setup.json`` of a setup."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _util.output_mask_to_fits(setup.mask, directory / "mask.fits")
    _util.output_array_to_fits(
        _util.native_values_from(setup.noise_map),
        mask=setup.mask,
        file_path=directory / "noise_map.fits",
        ext_name="noise_map",
    )
    if setup.psf is not None:
        _util.output_array_to_fits(
            _util.native_values_from(setup.psf.kernel),
            mask=setup.psf.kernel.mask,
            file_path=directory / "psf.fits",
            ext_name="psf",
        )
    output_to_json(setup, directory / "setup.json")
