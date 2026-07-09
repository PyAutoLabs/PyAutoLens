"""
Data structure for weak gravitational lensing observations.

Weak lensing measures the small, statistical distortion of background source-galaxy shapes induced by foreground
mass. The observable is a *shear catalogue*: a set of complex shear components ``(gamma_2, gamma_1)`` measured at
the (y, x) sky positions of a population of background galaxies, together with a per-galaxy noise estimate
(typically dominated by intrinsic shape noise — each galaxy has a random unlensed ellipticity that adds to its
measured shear).

``WeakDataset`` holds those three quantities together. It is the weak-lensing analogue of
:class:`autolens.point.dataset.PointDataset` and is the input to a :class:`autolens.weak.fit.FitWeak` (added in a
follow-up step).

The shear catalogue is stored as a :class:`autogalaxy.util.shear_field.ShearYX2DIrregular` so the convention is
the same one pinned by ``PyAutoGalaxy`` PR #366: column 0 is :math:`\\gamma_2`, column 1 is :math:`\\gamma_1`,
and the (y, x) galaxy positions are accessible via ``shear_yx.grid``.
"""
import csv
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

import autoarray as aa

from autogalaxy.util.shear_field import ShearYX2DIrregular


class WeakDataset:
    def __init__(
        self,
        shear_yx: ShearYX2DIrregular,
        noise_map: Union[float, aa.ArrayIrregular, List[float]],
        name: str = "",
        redshifts: Optional[Union[aa.ArrayIrregular, List[float]]] = None,
        is_reduced: bool = False,
    ):
        """
        A weak-lensing shear catalogue: a ``ShearYX2DIrregular`` shear field plus a per-galaxy noise map.

        Parameters
        ----------
        shear_yx
            The measured (or simulated) shear at each background source-galaxy position. Shape
            ``[total_galaxies, 2]`` with column 0 = :math:`\\gamma_2`, column 1 = :math:`\\gamma_1`. The (y, x)
            positions of the galaxies are carried by ``shear_yx.grid``.
        noise_map
            The per-galaxy shear noise standard deviation (one value per galaxy). For weak lensing this is
            dominated by intrinsic shape noise, typically in the range 0.2 - 0.4 per shear component. A scalar
            broadcasts to a constant noise level across all galaxies.
        name
            Optional label, mirroring ``PointDataset.name``. Used by downstream fitting code to pair this
            dataset with a corresponding model component when multiple datasets are fitted simultaneously.
        redshifts
            Optional per-galaxy source redshifts (one value per galaxy). Stored for provenance and for
            future per-galaxy lensing-efficiency (sigma_crit) scaling; the current fit assumes a single
            effective source plane (the tracer's source redshift).
        is_reduced
            Whether the stored values are *reduced* shears ``g = gamma / (1 - kappa)`` — what real surveys
            measure from galaxy ellipticities — rather than the shear ``gamma`` itself. ``FitWeak`` computes
            the matching model quantity, so set this to match how the catalogue was measured. Simulated
            datasets from ``SimulatorShearYX`` default to plain shear.
        """
        self.name = name

        if not isinstance(shear_yx, ShearYX2DIrregular):
            raise TypeError(
                "WeakDataset.shear_yx must be a ShearYX2DIrregular instance; "
                f"got {type(shear_yx).__name__}."
            )

        self.shear_yx = shear_yx

        n_galaxies = len(shear_yx)

        if isinstance(noise_map, (float, int)):
            noise_map = [float(noise_map)] * n_galaxies

        if not isinstance(noise_map, aa.ArrayIrregular):
            noise_map = aa.ArrayIrregular(values=list(noise_map))

        if len(noise_map) != n_galaxies:
            raise ValueError(
                f"WeakDataset.noise_map has length {len(noise_map)} but shear_yx has "
                f"{n_galaxies} entries; the two must match."
            )

        self.noise_map = noise_map

        if redshifts is not None and not isinstance(redshifts, aa.ArrayIrregular):
            redshifts = aa.ArrayIrregular(values=list(redshifts))

        if redshifts is not None and len(redshifts) != n_galaxies:
            raise ValueError(
                f"WeakDataset.redshifts has length {len(redshifts)} but shear_yx has "
                f"{n_galaxies} entries; the two must match."
            )

        self.redshifts = redshifts
        self.is_reduced = is_reduced

    @property
    def positions(self) -> aa.Grid2DIrregular:
        """The (y, x) sky positions of the source galaxies the shear is measured at."""
        return self.shear_yx.grid

    @property
    def n_galaxies(self) -> int:
        """Number of source galaxies in the catalogue."""
        return len(self.shear_yx)

    @property
    def info(self) -> str:
        """A short human-readable summary of the dataset, mirroring ``PointDataset.info``."""
        return (
            f"name : {self.name}\n"
            f"n_galaxies : {self.n_galaxies}\n"
            f"shear_yx : {self.shear_yx}\n"
            f"noise_map : {self.noise_map}\n"
            f"redshifts : {self.redshifts}\n"
            f"is_reduced : {self.is_reduced}\n"
        )

    @classmethod
    def from_arrays(
        cls,
        positions,
        gamma_1,
        gamma_2,
        noise_map: Optional[Union[float, List[float]]] = None,
        weights: Optional[List[float]] = None,
        redshifts: Optional[List[float]] = None,
        is_reduced: bool = True,
        name: str = "",
    ) -> "WeakDataset":
        """
        Build a ``WeakDataset`` from plain per-galaxy arrays — the shared entry point of the
        catalogue loaders (:meth:`from_csv`, :meth:`from_fits`).

        Exactly one of ``noise_map`` / ``weights`` must be given: real catalogues quote either a
        per-galaxy shear standard deviation or an inverse-variance weight, converted here via
        ``sigma = weights**-0.5``.

        Parameters
        ----------
        positions
            The ``(N, 2)`` ``(y, x)`` galaxy positions in arc-seconds.
        gamma_1
            The ``(N,)`` gamma_1 (axis-aligned) shear components.
        gamma_2
            The ``(N,)`` gamma_2 (diagonal) shear components.
        noise_map
            Per-galaxy shear standard deviation (scalar broadcasts).
        weights
            Per-galaxy inverse-variance weights, the common alternative in lensing catalogues.
        redshifts
            Optional per-galaxy source redshifts.
        is_reduced
            Whether the components are reduced shears ``g`` (the default here — real catalogues
            measure ellipticities, i.e. reduced shear) or plain shear ``gamma``.
        """
        if (noise_map is None) == (weights is None):
            raise ValueError(
                "WeakDataset.from_arrays requires exactly one of noise_map or weights."
            )

        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            if np.any(weights <= 0.0):
                raise ValueError(
                    "WeakDataset.from_arrays weights must all be positive to convert "
                    "to noise via sigma = weights**-0.5."
                )
            noise_map = list(weights**-0.5)

        grid = aa.Grid2DIrregular(values=np.asarray(positions, dtype=float))
        values = np.stack(
            [np.asarray(gamma_2, dtype=float), np.asarray(gamma_1, dtype=float)],
            axis=1,
        )
        shear_yx = ShearYX2DIrregular(values=values, grid=grid)

        return cls(
            shear_yx=shear_yx,
            noise_map=noise_map,
            name=name,
            redshifts=redshifts,
            is_reduced=is_reduced,
        )

    def to_csv(self, file_path: Union[str, Path]):
        """
        Write this dataset to ``file_path`` as a CSV with one row per galaxy.

        Columns: ``name, y, x, gamma_2, gamma_1, noise`` plus ``redshift`` when this dataset
        carries per-galaxy redshifts. Whether the components are reduced shear is not a column —
        pass ``is_reduced`` when loading (:meth:`from_csv` defaults to this dataset's convention
        only if you tell it to).
        """
        fieldnames = ["name", "y", "x", "gamma_2", "gamma_1", "noise"]
        if self.redshifts is not None:
            fieldnames.append("redshift")

        values = np.asarray(self.shear_yx)
        positions = np.asarray(self.positions)
        noise = np.asarray(self.noise_map)

        with open(file_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(self.n_galaxies):
                row = {
                    "name": self.name,
                    "y": positions[i, 0],
                    "x": positions[i, 1],
                    "gamma_2": values[i, 0],
                    "gamma_1": values[i, 1],
                    "noise": noise[i],
                }
                if self.redshifts is not None:
                    row["redshift"] = np.asarray(self.redshifts)[i]
                writer.writerow(row)

    @classmethod
    def from_csv(
        cls,
        file_path: Union[str, Path],
        is_reduced: bool = True,
        name: Optional[str] = None,
    ) -> "WeakDataset":
        """
        Load a ``WeakDataset`` from a CSV with columns ``name, y, x, gamma_2, gamma_1, noise``
        (optionally ``redshift``), as written by :meth:`to_csv`.

        Parameters
        ----------
        file_path
            Path to the CSV file.
        is_reduced
            Whether the catalogue's components are reduced shears ``g`` (the default — real
            catalogues measure ellipticities) or plain shear ``gamma``.
        name
            When given, only rows whose ``name`` column matches are loaded.
        """
        positions, gamma_1, gamma_2, noise, redshifts = [], [], [], [], []
        row_name = name or ""

        with open(file_path, newline="") as f:
            reader = csv.DictReader(f)
            has_redshift = "redshift" in (reader.fieldnames or [])
            for row in reader:
                if name is not None and row["name"] != name:
                    continue
                row_name = row["name"]
                positions.append((float(row["y"]), float(row["x"])))
                gamma_1.append(float(row["gamma_1"]))
                gamma_2.append(float(row["gamma_2"]))
                noise.append(float(row["noise"]))
                if has_redshift:
                    redshifts.append(float(row["redshift"]))

        if len(positions) == 0:
            raise ValueError(
                f"CSV file {str(file_path)!r} contained no WeakDataset rows"
                + (f" with name {name!r}." if name is not None else ".")
            )

        return cls.from_arrays(
            positions=positions,
            gamma_1=gamma_1,
            gamma_2=gamma_2,
            noise_map=noise,
            redshifts=redshifts if redshifts else None,
            is_reduced=is_reduced,
            name=row_name,
        )

    @classmethod
    def from_fits(
        cls,
        file_path: Union[str, Path],
        y_col: str,
        x_col: str,
        gamma_1_col: str,
        gamma_2_col: str,
        noise_col: Optional[str] = None,
        weight_col: Optional[str] = None,
        redshift_col: Optional[str] = None,
        hdu: int = 1,
        is_reduced: bool = True,
        name: str = "",
    ) -> "WeakDataset":
        """
        Load a ``WeakDataset`` from a FITS binary table — the standard distribution format of
        real shear catalogues — with explicit column-name mapping.

        Exactly one of ``noise_col`` / ``weight_col`` must be given (inverse-variance weights are
        converted via ``sigma = weight**-0.5``). Positions are read as-is: convert RA/Dec to
        projected arc-second offsets about the cluster centre *before* writing the table, or use
        a catalogue whose columns are already tangent-plane coordinates.

        Parameters
        ----------
        file_path
            Path to the FITS file.
        y_col, x_col
            Names of the position columns (arc-seconds, (y, x) convention).
        gamma_1_col, gamma_2_col
            Names of the shear-component columns.
        noise_col
            Name of the per-galaxy shear standard-deviation column.
        weight_col
            Name of the per-galaxy inverse-variance weight column.
        redshift_col
            Optional name of a per-galaxy source-redshift column.
        hdu
            The FITS extension holding the binary table.
        is_reduced
            Whether the components are reduced shears ``g`` (the default — real catalogues
            measure ellipticities) or plain shear ``gamma``.
        """
        from astropy.io import fits as astropy_fits

        if (noise_col is None) == (weight_col is None):
            raise ValueError(
                "WeakDataset.from_fits requires exactly one of noise_col or weight_col."
            )

        with astropy_fits.open(file_path) as hdul:
            table = hdul[hdu].data

            positions = np.stack(
                [np.asarray(table[y_col], dtype=float), np.asarray(table[x_col], dtype=float)],
                axis=1,
            )

            return cls.from_arrays(
                positions=positions,
                gamma_1=np.asarray(table[gamma_1_col], dtype=float),
                gamma_2=np.asarray(table[gamma_2_col], dtype=float),
                noise_map=(
                    list(np.asarray(table[noise_col], dtype=float))
                    if noise_col is not None
                    else None
                ),
                weights=(
                    list(np.asarray(table[weight_col], dtype=float))
                    if weight_col is not None
                    else None
                ),
                redshifts=(
                    list(np.asarray(table[redshift_col], dtype=float))
                    if redshift_col is not None
                    else None
                ),
                is_reduced=is_reduced,
                name=name,
            )

    def extent_from(self, buffer: float = 0.1) -> List[float]:
        """The axis-aligned bounding box of the source-galaxy positions, padded by ``buffer`` on each side."""
        positions = self.positions
        y_max = max(positions[:, 0]) + buffer
        y_min = min(positions[:, 0]) - buffer
        x_max = max(positions[:, 1]) + buffer
        x_min = min(positions[:, 1]) - buffer
        return [y_min, y_max, x_min, x_max]
