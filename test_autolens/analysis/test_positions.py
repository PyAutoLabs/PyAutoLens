import os
from pathlib import Path
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
