"""
Small helpers shared by the ``al.twins`` modules: an immutability guard, library
versions for provenance, SHA-256 hashing and the FITS read/write recipe used for
masks and arrays.
"""

import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

import autoarray as aa
from autoarray.mask.mask_2d import Mask2DKeys
from autonerves.dictable import to_dict
from autonerves.fitsable import ndarray_via_hdu_from, output_to_fits

VERSIONED_DISTRIBUTIONS = (
    "autonerves",
    "autofit",
    "autoarray",
    "autogalaxy",
    "autolens",
    "numpy",
)


class Frozen:
    """
    Mixin making instances immutable after ``__init__``: attribute assignment and
    deletion raise ``AttributeError``. ``__init__`` sets attributes through
    ``self._freeze_set``.
    """

    def _freeze_set(self, name, value):
        object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        raise AttributeError(
            f"{type(self).__name__} is immutable; cannot set attribute '{name}'. "
            f"Build a modified copy instead."
        )

    def __delattr__(self, name):
        raise AttributeError(
            f"{type(self).__name__} is immutable; cannot delete attribute '{name}'."
        )


def library_versions() -> Dict[str, Optional[str]]:
    """
    The installed versions of the PyAuto stack and NumPy, plus the Python version.
    A distribution that is not installed (e.g. a source checkout on ``sys.path``
    without metadata) is recorded as ``None`` rather than raising.
    """
    versions = {}
    for name in VERSIONED_DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    versions["python"] = platform.python_version()
    return versions


def utc_now_iso() -> str:
    """The current time as an ISO-8601 UTC string (seconds precision)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_of_file(file_path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_of_object(obj) -> str:
    """
    SHA-256 of the canonical (key-sorted) JSON of an object's ``autonerves.dictable``
    dictionary. Used when there is no file on disk to hash (e.g. ``from_result``).
    """
    payload = json.dumps(to_dict(obj), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_of_arrays(*arrays) -> str:
    """SHA-256 over the raw bytes (and shapes) of several arrays."""
    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(np.asarray(array, dtype="float64"))
        digest.update(str(values.shape).encode("utf-8"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def native_values_from(array) -> np.ndarray:
    """The native 2D NumPy values of an ``aa.Array2D`` (masked entries are zero)."""
    return np.asarray(array.native.array)


def mask_from_hdu(hdu) -> aa.Mask2D:
    """
    Rebuild a ``Mask2D`` from a FITS HDU whose header carries the pixel scales and
    origin, as written by ``Mask2D.header_dict``.

    Replicates ``autogalaxy.aggregator.agg_util.mask_header_from`` (PyAutoGalaxy
    ``autogalaxy/aggregator/agg_util.py``, function ``mask_header_from``): a file
    written without a ``PIXSCAX`` card is assumed to have square pixels. Kept as a
    copy so ``al.twins`` does not depend on the aggregator's ``af.Fit`` interface;
    if that function changes, change this one too.
    """
    header = hdu.header
    pixel_scales_y = header[Mask2DKeys.PIXSCAY.value]
    pixel_scales = (
        pixel_scales_y,
        header.get(Mask2DKeys.PIXSCAX.value, pixel_scales_y),
    )
    origin = (
        header[Mask2DKeys.ORIGINY.value],
        header[Mask2DKeys.ORIGINX.value],
    )
    return aa.Mask2D(
        mask=ndarray_via_hdu_from(hdu),
        pixel_scales=pixel_scales,
        origin=origin,
    )


def output_mask_to_fits(mask: aa.Mask2D, file_path: Union[str, Path]):
    output_to_fits(
        values=np.asarray(mask).astype("float"),
        file_path=file_path,
        overwrite=True,
        header_dict=mask.header_dict,
        ext_name="mask",
    )


def output_array_to_fits(
    values: np.ndarray, mask: aa.Mask2D, file_path: Union[str, Path], ext_name: str
):
    """
    Write native values without the visualisation zoom (``native_for_fits``), so a
    reload is bit-identical and shaped like the mask.
    """
    output_to_fits(
        values=np.asarray(values, dtype="float64"),
        file_path=file_path,
        overwrite=True,
        header_dict=mask.header_dict,
        ext_name=ext_name,
    )


def ndarray_from_fits(file_path: Union[str, Path], hdu: int = 0) -> np.ndarray:
    from astropy.io import fits

    with fits.open(file_path) as hdu_list:
        return ndarray_via_hdu_from(hdu_list[hdu])


def mask_from_fits(file_path: Union[str, Path], hdu: int = 0) -> aa.Mask2D:
    from astropy.io import fits

    with fits.open(file_path) as hdu_list:
        return mask_from_hdu(hdu_list[hdu])
