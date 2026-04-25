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
from typing import List, Optional, Union

import autoarray as aa

from autogalaxy.util.shear_field import ShearYX2DIrregular


class WeakDataset:
    def __init__(
        self,
        shear_yx: ShearYX2DIrregular,
        noise_map: Union[float, aa.ArrayIrregular, List[float]],
        name: str = "",
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
        )

    def extent_from(self, buffer: float = 0.1) -> List[float]:
        """The axis-aligned bounding box of the source-galaxy positions, padded by ``buffer`` on each side."""
        positions = self.positions
        y_max = max(positions[:, 0]) + buffer
        y_min = min(positions[:, 0]) - buffer
        x_max = max(positions[:, 1]) + buffer
        x_min = min(positions[:, 1]) - buffer
        return [y_min, y_max, x_min, x_max]
