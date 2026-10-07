import os
from pathlib import Path
import numpy as np
import pytest

from autonerves.dictable import output_to_json, from_json, from_dict
from autofit.tools.util import open_

import autolens as al
from autolens import exc


def test__check_positions_on_instantiation():
    al.PositionsLH(
        positions=al.Grid2DIrregular([(1.0, 2.0), (3.0, 4.0)]), threshold=0.1
    )

    # Positions input with threshold but positions are length 1.

    with pytest.raises(exc.PositionsException):
        al.PositionsLH(positions=al.Grid2DIrregular([(1.0, 2.0)]), threshold=0.1)


def _penalty(positions_lh, einstein_radius):
    """The penalty for an isothermal lens of the given Einstein radius."""
    lens = al.Galaxy(
        redshift=0.5, mass=al.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=einstein_radius)
    )
    tracer = al.Tracer(galaxies=[lens, al.Galaxy(redshift=1.0)])

    class MockAnalysis:
        def tracer_via_instance_from(self, instance):
            return tracer

    return float(
        positions_lh.log_likelihood_penalty_from(instance=None, analysis=MockAnalysis())
    )


def test__non_finite_positions_are_dropped_and_the_penalty_still_fires():
    # `Result.positions_likelihood_from` pads the point solver's images with
    # (inf, inf). One such row used to make the max separation nan, so the
    # penalty was zero for every model.
    inf = np.inf
    positions = al.Grid2DIrregular([(0.0, 1.0), (inf, inf), (0.0, -1.0)] + [(inf, inf)] * 5)

    positions_lh = al.PositionsLH(positions=positions, threshold=0.1)

    assert len(positions_lh.positions) == 2
    # Images at +/-1" trace together only for theta_E = 1"; theta_E = 0.3" splits them.
    assert _penalty(positions_lh, einstein_radius=1.0) == 0.0
    assert _penalty(positions_lh, einstein_radius=0.3) > 0.0


def test__fewer_than_two_finite_positions_raises():
    inf = np.inf

    with pytest.raises(exc.PositionsException):
        al.PositionsLH(
            positions=al.Grid2DIrregular([(0.0, 1.0), (inf, inf), (inf, inf)]), threshold=0.1
        )


def test__output_positions_info():
    output_path = Path(__file__).resolve().parent / "files"

    positions_likelihood = al.PositionsLH(
        positions=al.Grid2DIrregular([(1.0, 2.0), (3.0, 4.0)]), threshold=0.1
    )

    tracer = al.m.MockTracer(
        traced_grid_2d_list_from=al.Grid2DIrregular(values=[[(0.5, 1.5), (2.5, 3.5)]])
    )

    positions_likelihood.output_positions_info(output_path=output_path, tracer=tracer)

    positions_file = output_path / "positions.info"

    with open_(positions_file, "r") as f:
        output = f.readlines()

    assert "Plane Index: -1" in output[0]
    assert "Positions" in output[1]

    os.remove(positions_file)


@pytest.fixture(name="settings_dict")
def make_settings_dict():
    return {
        "type": "instance",
        "class_path": "autolens.analysis.positions.PositionsLH",
        "arguments": {
            "positions": {
                "type": "ndarray",
                "array": [[1.0, 2.0], [3.0, 4.0]],
                "dtype": "float64",
            },
            "threshold": 0.1,
            "log_likelihood_penalty_factor": 100000000.0,
        },
    }


def test_settings_from_dict(settings_dict):
    assert isinstance(from_dict(settings_dict), al.PositionsLH)


def test_file():
    filename = "/tmp/temp.json"

    output_to_json(
        al.PositionsLH(
            positions=al.Grid2DIrregular([(1.0, 2.0), (3.0, 4.0)]), threshold=0.1
        ),
        filename,
    )

    try:
        assert isinstance(from_json(filename), al.PositionsLH)
    finally:
        os.remove(filename)
