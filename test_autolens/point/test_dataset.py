import os

import pytest

import autolens as al


def _assert_grid_close(actual, expected):
    actual_pairs = [(float(p[0]), float(p[1])) for p in actual]
    expected_pairs = [(float(p[0]), float(p[1])) for p in expected]
    assert actual_pairs == pytest.approx(expected_pairs)


def _assert_array_close(actual, expected):
    if expected is None:
        assert actual is None
        return
    assert actual is not None
    assert [float(v) for v in actual] == pytest.approx(
        [float(v) for v in expected]
    )


def _assert_dataset_equal(actual: al.PointDataset, expected: al.PointDataset):
    assert actual.name == expected.name
    _assert_grid_close(actual.positions, expected.positions)
    _assert_array_close(actual.positions_noise_map, expected.positions_noise_map)
    _assert_array_close(actual.fluxes, expected.fluxes)
    _assert_array_close(actual.fluxes_noise_map, expected.fluxes_noise_map)
    _assert_array_close(actual.time_delays, expected.time_delays)
    _assert_array_close(actual.time_delays_noise_map, expected.time_delays_noise_map)
    if expected.redshift is None:
        assert actual.redshift is None
    else:
        assert actual.redshift == pytest.approx(expected.redshift)


def test__csv_round_trip__positions_only(tmp_path):
    dataset = al.PointDataset(
        name="source_0",
        positions=[(0.5, 1.0), (-0.25, 2.0), (1.5, -1.0)],
        positions_noise_map=[0.05, 0.05, 0.1],
    )

    file_path = os.path.join(tmp_path, "point_dataset.csv")
    dataset.to_csv(file_path)

    loaded = al.PointDataset.from_csv(file_path)

    _assert_dataset_equal(loaded, dataset)
    assert loaded.fluxes is None
    assert loaded.fluxes_noise_map is None
    assert loaded.time_delays is None
    assert loaded.time_delays_noise_map is None


def test__csv_round_trip__positions_and_fluxes(tmp_path):
    dataset = al.PointDataset(
        name="source_0",
        positions=[(0.0, 0.0), (1.0, 1.0)],
        positions_noise_map=[0.05, 0.05],
        fluxes=[1.2, 0.8],
        fluxes_noise_map=[0.1, 0.15],
    )

    file_path = os.path.join(tmp_path, "point_dataset.csv")
    dataset.to_csv(file_path)

    loaded = al.PointDataset.from_csv(file_path)

    _assert_dataset_equal(loaded, dataset)
    assert loaded.time_delays is None
    assert loaded.time_delays_noise_map is None


def test__csv_round_trip__positions_and_time_delays(tmp_path):
    dataset = al.PointDataset(
        name="source_0",
        positions=[(0.0, 0.0), (1.0, 1.0)],
        positions_noise_map=[0.05, 0.05],
        time_delays=[10.0, 25.0],
        time_delays_noise_map=[1.0, 2.5],
    )

    file_path = os.path.join(tmp_path, "point_dataset.csv")
    dataset.to_csv(file_path)

    loaded = al.PointDataset.from_csv(file_path)

    _assert_dataset_equal(loaded, dataset)
    assert loaded.fluxes is None
    assert loaded.fluxes_noise_map is None


def test__csv_round_trip__positions_fluxes_and_time_delays(tmp_path):
    dataset = al.PointDataset(
        name="source_0",
        positions=[(0.0, 0.0), (1.0, 1.0), (-1.5, 0.5)],
        positions_noise_map=[0.05, 0.05, 0.08],
        fluxes=[1.2, 0.8, 0.5],
        fluxes_noise_map=[0.1, 0.15, 0.12],
        time_delays=[10.0, 25.0, 40.0],
        time_delays_noise_map=[1.0, 2.5, 3.0],
    )

    file_path = os.path.join(tmp_path, "point_dataset.csv")
    dataset.to_csv(file_path)

    loaded = al.PointDataset.from_csv(file_path)

    _assert_dataset_equal(loaded, dataset)


def test__csv_list_round_trip__heterogeneous_optional_columns(tmp_path):
    with_fluxes = al.PointDataset(
        name="source_0",
        positions=[(0.0, 0.0), (1.0, 1.0)],
        positions_noise_map=[0.05, 0.05],
        fluxes=[1.2, 0.8],
        fluxes_noise_map=[0.1, 0.15],
    )
    positions_only = al.PointDataset(
        name="source_1",
        positions=[(2.0, 0.5), (-1.0, 0.5)],
        positions_noise_map=[0.1, 0.1],
    )

    file_path = os.path.join(tmp_path, "point_datasets.csv")
    al.output_to_csv([with_fluxes, positions_only], file_path)

    loaded = al.list_from_csv(file_path)

    assert [d.name for d in loaded] == ["source_0", "source_1"]
    _assert_dataset_equal(loaded[0], with_fluxes)
    _assert_dataset_equal(loaded[1], positions_only)
    assert loaded[1].fluxes is None
    assert loaded[1].fluxes_noise_map is None


def test__csv_round_trip__redshift(tmp_path):
    dataset = al.PointDataset(
        name="source_0",
        positions=[(0.5, 1.0), (-0.25, 2.0), (1.5, -1.0)],
        positions_noise_map=[0.05, 0.05, 0.1],
        redshift=2.5,
    )

    file_path = os.path.join(tmp_path, "point_dataset.csv")
    dataset.to_csv(file_path)

    loaded = al.PointDataset.from_csv(file_path)

    _assert_dataset_equal(loaded, dataset)
    assert loaded.redshift == pytest.approx(2.5)


def test__csv_list_round_trip__mixed_redshift_presence(tmp_path):
    with_redshift = al.PointDataset(
        name="source_0",
        positions=[(0.0, 0.0), (1.0, 1.0)],
        positions_noise_map=[0.05, 0.05],
        redshift=1.8,
    )
    without_redshift = al.PointDataset(
        name="source_1",
        positions=[(2.0, 0.5), (-1.0, 0.5)],
        positions_noise_map=[0.1, 0.1],
    )

    file_path = os.path.join(tmp_path, "point_datasets.csv")
    al.output_to_csv([with_redshift, without_redshift], file_path)

    loaded = al.list_from_csv(file_path)

    assert [d.name for d in loaded] == ["source_0", "source_1"]
    _assert_dataset_equal(loaded[0], with_redshift)
    _assert_dataset_equal(loaded[1], without_redshift)
    assert loaded[0].redshift == pytest.approx(1.8)
    assert loaded[1].redshift is None


def test__list_from_csv__inconsistent_redshift_raises(tmp_path):
    file_path = os.path.join(tmp_path, "point_datasets.csv")
    with open(file_path, "w") as f:
        f.write("name,y,x,positions_noise,redshift\n")
        f.write("source_0,0.0,0.0,0.05,1.5\n")
        f.write("source_0,1.0,1.0,0.05,2.0\n")

    with pytest.raises(ValueError, match="inconsistent 'redshift'"):
        al.list_from_csv(file_path)


def test__list_from_csv__partial_redshift_raises(tmp_path):
    file_path = os.path.join(tmp_path, "point_datasets.csv")
    with open(file_path, "w") as f:
        f.write("name,y,x,positions_noise,redshift\n")
        f.write("source_0,0.0,0.0,0.05,1.5\n")
        f.write("source_0,1.0,1.0,0.05,\n")

    with pytest.raises(ValueError, match="partially populated column 'redshift'"):
        al.list_from_csv(file_path)


def test__from_csv__multiple_groups_requires_name(tmp_path):
    datasets = [
        al.PointDataset(
            name="source_0",
            positions=[(0.0, 0.0)],
            positions_noise_map=[0.05],
        ),
        al.PointDataset(
            name="source_1",
            positions=[(1.0, 1.0)],
            positions_noise_map=[0.05],
        ),
    ]

    file_path = os.path.join(tmp_path, "point_datasets.csv")
    al.output_to_csv(datasets, file_path)

    with pytest.raises(ValueError):
        al.PointDataset.from_csv(file_path)

    picked = al.PointDataset.from_csv(file_path, name="source_1")
    assert picked.name == "source_1"
