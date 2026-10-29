"""
``recovery_table_from``: compare a fit's posterior with the ground truth a twin was
simulated with.

For each recoverable ``truth`` key (``model.TRUTH_PARAMETER_NAMES``) that is a free
parameter of the fit, a ``RecoveryTable`` row lists the truth, the median of the 1D
marginalized PDF, its lower and upper values at ``sigma`` and whether the truth lies
between them. The truth keys are mapped onto model paths by
``model.truth_paths_from``, the same map the model builder publishes.

NumPy only: nothing here supports JAX.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from autolens.twins.model import truth_paths_from
from autolens.twins.transform.stellar_dark import NAME as STELLAR_DARK_NAME

CSV_COLUMNS = ("name", "path", "truth", "median", "lower", "upper", "inside")


@dataclass(frozen=True)
class RecoveryRow:
    """One parameter: its truth, PDF median and the bounds at the table's sigma."""

    name: str
    path: Tuple[str, ...]
    truth: float
    median: float
    lower: float
    upper: float

    @property
    def inside(self) -> bool:
        """Whether the truth lies within ``[lower, upper]``."""
        return self.lower <= self.truth <= self.upper


class RecoveryTable:
    def __init__(self, rows: List[RecoveryRow], sigma: float):
        """
        The recovery of a twin's truth by a fit, one ``RecoveryRow`` per parameter.

        Parameters
        ----------
        rows
            The rows, in ``truth`` key order.
        sigma
            The sigma at which ``lower`` and ``upper`` are taken.
        """
        self.rows = list(rows)
        self.sigma = float(sigma)

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, name: str) -> RecoveryRow:
        """The row of truth key ``name``."""
        for row in self.rows:
            if row.name == name:
                return row
        raise KeyError(name)

    @property
    def all_inside(self) -> bool:
        """Whether every truth lies within its bounds."""
        return all(row.inside for row in self.rows)

    def to_csv(self, file_path: Union[str, Path]):
        """Write the rows to ``file_path`` (columns ``CSV_COLUMNS``, path dot-joined)."""
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)
            for row in self.rows:
                writer.writerow(
                    [
                        row.name,
                        ".".join(row.path),
                        repr(row.truth),
                        repr(row.median),
                        repr(row.lower),
                        repr(row.upper),
                        row.inside,
                    ]
                )

    def __str__(self):
        header = (
            f"{'name':<22}{'truth':>13}{'median':>13}{'lower':>13}{'upper':>13}  "
            f"inside ({self.sigma:g} sigma)"
        )
        lines = [header]
        for row in self.rows:
            lines.append(
                f"{row.name:<22}{row.truth:>13.5g}{row.median:>13.5g}"
                f"{row.lower:>13.5g}{row.upper:>13.5g}  {'yes' if row.inside else 'NO'}"
            )
        return "\n".join(lines)

    def __repr__(self):
        return (
            f"RecoveryTable(sigma={self.sigma:g}, names={[row.name for row in self.rows]}, "
            f"all_inside={self.all_inside})"
        )


def truth_from(truth: Union[Dict, str, Path]) -> Dict:
    """
    The flat truth dictionary: ``truth`` itself, or the ``"truth"`` block of a
    ``twin.json`` file (or of the ``twin.json`` in a directory written by
    ``SimulatedTwin.output_to``). That block lists one entry per transformation; the
    last ``ReplaceTotalMassWithStellarPlusDark`` entry is returned (else the last).
    """
    if isinstance(truth, dict):
        return dict(truth)

    file_path = Path(truth)
    if file_path.is_dir():
        file_path = file_path / "twin.json"
    with open(file_path) as f:
        entries = json.load(f)["truth"]
    if not entries:
        raise ValueError(f"{file_path} has an empty truth block.")
    for entry in reversed(entries):
        if entry.get("transformation") == STELLAR_DARK_NAME:
            return dict(entry)
    return dict(entries[-1])


def _column_from(samples, path: Tuple[str, ...]) -> int:
    for index, prior_paths in enumerate(samples.paths):
        if tuple(path) in (tuple(prior_path) for prior_path in prior_paths):
            return index
    raise ValueError(f"{'.'.join(path)} is not a free parameter of the samples' model.")


def _values_at_sigma_from(samples, sigma: float) -> List[Tuple[float, float]]:
    if hasattr(samples, "values_at_sigma"):
        return samples.values_at_sigma(sigma=sigma, as_instance=False)

    # A SamplesSummary stores the values at 1 and 3 sigma only, keyed by path.
    stored = {1.0: "values_at_sigma_1", 3.0: "values_at_sigma_3"}.get(float(sigma))
    values = None if stored is None else getattr(samples, stored, None)
    if values is None:
        raise ValueError(
            f"A {type(samples).__name__} stores values at sigma 1 and 3 only; got "
            f"sigma={sigma}. Pass the full samples instead."
        )
    return [tuple(values[".".join(prior_paths[0])]) for prior_paths in samples.paths]


def recovery_table_from(
    samples,
    truth: Union[Dict, str, Path],
    paths: Optional[Dict[str, Tuple[str, ...]]] = None,
    sigma: float = 3.0,
) -> RecoveryTable:
    """
    Compare the posterior in ``samples`` with ``truth``.

    The median is ``samples.median_pdf(as_instance=False)`` and the bounds are
    ``samples.values_at_sigma(sigma, as_instance=False)`` (the 1D marginalized
    quantiles; parameter covariance is ignored). Sample columns are found through
    ``samples.paths``, which groups every path of a tied parameter, so any member's
    path of the shared ``mass_to_light_ratio`` finds its column. A ``SamplesSummary``
    is accepted for ``sigma`` 1 or 3.

    Parameters
    ----------
    samples
        The ``Samples`` (e.g. ``result.samples``) or ``SamplesSummary`` of a fit of a
        model made by ``stellar_dark_model_from``.
    truth
        A flat truth dictionary (``TransformationRecord.truth``) or the path of a
        ``twin.json`` (or of its directory), whose ``"truth"`` block is read.
    paths
        ``{truth key: model path}``; defaults to
        ``truth_paths_from(samples.model, truth["galaxy"])``. Only truth keys with a
        path become rows.
    sigma
        The sigma of the bounds (default 3).
    """
    truth = truth_from(truth)
    if paths is None:
        if samples.model is None:
            raise ValueError("The samples have no model; pass paths explicitly.")
        paths = truth_paths_from(samples.model, truth["galaxy"])

    median = samples.median_pdf(as_instance=False)
    bounds = _values_at_sigma_from(samples, sigma)

    rows = []
    for name, path in paths.items():
        if name not in truth:
            continue
        column = _column_from(samples, tuple(path))
        lower, upper = bounds[column]
        rows.append(
            RecoveryRow(
                name=name,
                path=tuple(path),
                truth=float(truth[name]),
                median=float(median[column]),
                lower=float(lower),
                upper=float(upper),
            )
        )
    return RecoveryTable(rows=rows, sigma=sigma)
