"""
Data structures for point-source strong lens observations.

Point-source lensing arises when the background source is compact enough to be treated
as a point (e.g. a quasar, supernova, or compact radio source).  Gravitational lensing
splits the source into multiple images whose positions, fluxes, and time delays constrain
the lens mass distribution.

``PointDataset`` holds the image-plane positions, fluxes, and time delays of a single
named point source together with their noise maps.  The ``name`` attribute is used to
pair this dataset with the corresponding ``Point`` model component during fitting.
When multiple point sources are fitted simultaneously (for example many multiply-imaged
background sources in a strong-lens cluster) they are collected in a plain Python
``list`` of ``PointDataset`` objects.

Two I/O surfaces are supported:

- JSON (via :func:`autoconf.output_to_json` / :func:`autoconf.from_json`) — exact
  round-trip, one file per ``PointDataset``; the canonical modeling input.
- CSV (via :meth:`PointDataset.to_csv` / :meth:`PointDataset.from_csv` and the
  module-level :func:`output_to_csv` / :func:`list_from_csv`) — one row per observed
  image, grouped by ``name``, so that tens or hundreds of cluster-scale point sources
  can be edited in a single spreadsheet.
"""
from autoconf import csvable
from typing import List, Tuple, Optional, Union

import autoarray as aa


_BASE_HEADERS = ["name", "y", "x", "positions_noise"]
_FLUX_HEADERS = ["flux", "flux_noise"]
_TIME_DELAY_HEADERS = ["time_delay", "time_delay_noise"]


class PointDataset:
    def __init__(
        self,
        name: str,
        positions: Union[aa.Grid2DIrregular, List[List], List[Tuple]],
        positions_noise_map: Union[float, aa.ArrayIrregular, List[float]],
        fluxes: Optional[Union[aa.ArrayIrregular, List[float]]] = None,
        fluxes_noise_map: Optional[Union[float, aa.ArrayIrregular, List[float]]] = None,
        time_delays: Optional[Union[aa.ArrayIrregular, List[float]]] = None,
        time_delays_noise_map: Optional[
            Union[float, aa.ArrayIrregular, List[float]]
        ] = None,
    ):
        """
        A collection of the data component that can be used for point-source model-fitting, for example fitting the
        observed positions of a a strongly lensed quasar or supernovae or in strong lens cluster modeling, where
        there may be many tens or hundreds of individual source galaxies each of which are modeled as a point source.

        The name of the dataset is required for point-source model-fitting, as it pairs a point-source dataset with
        its corresponding point-source in the model-fit. For example, if a dataset has the name `source_1`, it will
        be paired with the `Point` model-component which has the name `source_1`. If a dataset component is not
        successfully paired with a model-component, an error is raised.

        Parameters
        ----------
        name
            The name of the point source dataset which is paired to a `Point` in the `Model`.
        positions
            The image-plane (y,x) positions of the point-source.
        positions_noise_map
            The noise-value of every (y,x) position, which is typically the pixel-scale of the data.
        fluxes
            The image-plane flux of each observed point-source of light.
        fluxes_noise_map
            The noise-value of every observed flux, which is typically measured from the pixel values of the pixel
            containing the point source after convolution with the PSF.
        time_delays
            The time delays of each observed point-source of light in days.
        time_delays_noise_map
            The noise-value of every observed time delay, which is typically measured from the time delay analysis.
        """

        self.name = name

        # Ensure positions is a Grid2DIrregular
        self.positions = (
            positions
            if isinstance(positions, aa.Grid2DIrregular)
            else aa.Grid2DIrregular(values=positions)
        )

        # Ensure positions_noise_map is an ArrayIrregular
        if isinstance(positions_noise_map, float):
            positions_noise_map = [positions_noise_map] * len(self.positions)

        self.positions_noise_map = (
            positions_noise_map
            if isinstance(positions_noise_map, aa.ArrayIrregular)
            else aa.ArrayIrregular(values=positions_noise_map)
        )

        def convert_to_array_irregular(values):
            """
            Convert data to ArrayIrregular if it is not already.
            """
            return (
                aa.ArrayIrregular(values=values)
                if values is not None and not isinstance(values, aa.ArrayIrregular)
                else values
            )

        # Convert fluxes, time delays and their noise maps to ArrayIrregular if provided as values and not already this type

        self.fluxes = convert_to_array_irregular(fluxes)
        self.fluxes_noise_map = convert_to_array_irregular(fluxes_noise_map)
        self.time_delays = convert_to_array_irregular(time_delays)
        self.time_delays_noise_map = convert_to_array_irregular(time_delays_noise_map)

    @property
    def info(self) -> str:
        """
        A dictionary representation of this instance.

        Arrays are represented as lists or lists of lists.
        """
        info = f"name : {self.name}\n"
        info += f"positions : {self.positions}\n"
        info += f"positions_noise_map : {self.positions_noise_map}\n"
        info += f"fluxes : {self.fluxes}\n"
        info += f"fluxes_noise_map : {self.fluxes_noise_map}\n"
        info += f"time_delays : {self.time_delays}\n"
        info += f"time_delays_noise_map : {self.time_delays_noise_map}\n"
        return info

    def extent_from(self, buffer: float = 0.1):
        y_max = max(self.positions[:, 0]) + buffer
        y_min = min(self.positions[:, 0]) - buffer
        x_max = max(self.positions[:, 1]) + buffer
        x_min = min(self.positions[:, 1]) - buffer

        return [y_min, y_max, x_min, x_max]

    def to_csv(self, file_path: str):
        """
        Write this dataset to ``file_path`` as a CSV with one row per image.

        Optional flux / time-delay columns are included only when this dataset carries
        the corresponding values.  For multi-dataset output use :func:`output_to_csv`.
        """
        output_to_csv([self], file_path)

    @classmethod
    def from_csv(
        cls, file_path: str, name: Optional[str] = None
    ) -> "PointDataset":
        """
        Load a single ``PointDataset`` from a CSV written by :meth:`to_csv` or
        :func:`output_to_csv`.

        Parameters
        ----------
        file_path
            Path to a CSV file with at minimum the columns
            ``name, y, x, positions_noise``.
        name
            The ``name`` group to load.  Must be provided when the CSV contains more
            than one ``name``; when the CSV contains exactly one group it is picked
            automatically.
        """
        datasets = list_from_csv(file_path)

        if not datasets:
            raise ValueError(
                f"CSV file {file_path!r} contained no PointDataset rows."
            )

        if name is None:
            if len(datasets) > 1:
                available = [d.name for d in datasets]
                raise ValueError(
                    f"CSV file {file_path!r} contains {len(datasets)} groups "
                    f"({available!r}); pass name= to select one."
                )
            return datasets[0]

        for dataset in datasets:
            if dataset.name == name:
                return dataset

        available = [d.name for d in datasets]
        raise ValueError(
            f"CSV file {file_path!r} has no group named {name!r}. "
            f"Available groups: {available!r}."
        )


def _optional_values(dataset: PointDataset, attr: str) -> Optional[List[float]]:
    values = getattr(dataset, attr)
    if values is None:
        return None
    return [float(v) for v in values]


def output_to_csv(datasets: List[PointDataset], file_path: str):
    """
    Write a list of ``PointDataset`` objects to a single CSV with one row per observed
    image.

    The base columns (``name, y, x, positions_noise``) are always written.  The
    optional ``flux``/``flux_noise`` and ``time_delay``/``time_delay_noise`` columns
    are included when *any* dataset in ``datasets`` carries those values; datasets
    that do not carry them leave those cells blank.

    This is the hand-editable / spreadsheet form preferred for strong-lens cluster
    workflows with tens or hundreds of multiply-imaged sources.  For exact
    round-trip serialisation use ``output_to_json`` / ``from_json``.
    """
    include_flux = any(d.fluxes is not None for d in datasets)
    include_time_delay = any(d.time_delays is not None for d in datasets)

    headers = list(_BASE_HEADERS)
    if include_flux:
        headers += _FLUX_HEADERS
    if include_time_delay:
        headers += _TIME_DELAY_HEADERS

    rows = []
    for dataset in datasets:
        positions = dataset.positions
        positions_noise = _optional_values(dataset, "positions_noise_map")
        fluxes = _optional_values(dataset, "fluxes")
        fluxes_noise = _optional_values(dataset, "fluxes_noise_map")
        time_delays = _optional_values(dataset, "time_delays")
        time_delays_noise = _optional_values(dataset, "time_delays_noise_map")

        for i in range(len(positions)):
            row = {
                "name": dataset.name,
                "y": float(positions[i][0]),
                "x": float(positions[i][1]),
                "positions_noise": positions_noise[i],
            }
            if include_flux:
                row["flux"] = "" if fluxes is None else fluxes[i]
                row["flux_noise"] = (
                    "" if fluxes_noise is None else fluxes_noise[i]
                )
            if include_time_delay:
                row["time_delay"] = (
                    "" if time_delays is None else time_delays[i]
                )
                row["time_delay_noise"] = (
                    "" if time_delays_noise is None else time_delays_noise[i]
                )
            rows.append(row)

    csvable.output_to_csv(rows, file_path, headers=headers)


def _float_column(
    group_rows: List[dict], column: str, group_name: str
) -> Optional[List[float]]:
    raw = [row.get(column, "") for row in group_rows]
    populated = [v for v in raw if v not in ("", None)]

    if not populated:
        return None

    if len(populated) != len(raw):
        raise ValueError(
            f"CSV group {group_name!r} has partially populated column "
            f"{column!r}; every row in the group must have a value or all be blank."
        )

    return [float(v) for v in raw]


def list_from_csv(file_path: str) -> List[PointDataset]:
    """
    Load a list of ``PointDataset`` objects from a CSV written by
    :func:`output_to_csv` (or :meth:`PointDataset.to_csv`).

    Rows are grouped by their ``name`` column — one ``PointDataset`` per distinct
    name, preserving the order of first appearance.  Optional columns
    (``flux``/``flux_noise``, ``time_delay``/``time_delay_noise``) are carried through
    per-group: if every row in a group populates the column the values are loaded,
    if every row leaves it blank the corresponding attribute is set to ``None``, and
    any partial-population is rejected with a ``ValueError``.
    """
    rows = csvable.list_from_csv(file_path)

    if not rows:
        return []

    headers = list(rows[0].keys())

    for required in _BASE_HEADERS:
        if required not in headers:
            raise ValueError(
                f"CSV file {file_path!r} is missing required column {required!r}; "
                f"expected headers starting with {_BASE_HEADERS!r}."
            )

    groups: "dict[str, List[dict]]" = {}
    for row in rows:
        groups.setdefault(row["name"], []).append(row)

    has_flux_column = "flux" in headers
    has_flux_noise_column = "flux_noise" in headers
    has_time_delay_column = "time_delay" in headers
    has_time_delay_noise_column = "time_delay_noise" in headers

    datasets: List[PointDataset] = []
    for name, group_rows in groups.items():
        positions = [(float(r["y"]), float(r["x"])) for r in group_rows]
        positions_noise_map = [
            float(r["positions_noise"]) for r in group_rows
        ]

        fluxes = (
            _float_column(group_rows, "flux", name)
            if has_flux_column
            else None
        )
        fluxes_noise_map = (
            _float_column(group_rows, "flux_noise", name)
            if has_flux_noise_column
            else None
        )
        time_delays = (
            _float_column(group_rows, "time_delay", name)
            if has_time_delay_column
            else None
        )
        time_delays_noise_map = (
            _float_column(group_rows, "time_delay_noise", name)
            if has_time_delay_noise_column
            else None
        )

        datasets.append(
            PointDataset(
                name=name,
                positions=positions,
                positions_noise_map=positions_noise_map,
                fluxes=fluxes,
                fluxes_noise_map=fluxes_noise_map,
                time_delays=time_delays,
                time_delays_noise_map=time_delays_noise_map,
            )
        )

    return datasets
