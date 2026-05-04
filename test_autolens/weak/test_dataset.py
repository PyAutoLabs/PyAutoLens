import numpy as np
import pytest

import autoarray as aa
import autolens as al

from autogalaxy.util.shear_field import ShearYX2DIrregular


def _make_shear_yx(positions, values):
    grid = aa.Grid2DIrregular(values=positions)
    return ShearYX2DIrregular(values=np.asarray(values), grid=grid)


def test__weak_dataset__construct_with_shear_yx_irregular_and_scalar_noise():
    shear_yx = _make_shear_yx(
        positions=[(0.5, 0.5), (1.0, -0.3), (-0.7, 0.2)],
        values=[(0.01, 0.02), (-0.03, 0.04), (0.05, -0.06)],
    )

    dataset = al.WeakDataset(shear_yx=shear_yx, noise_map=0.3, name="weak")

    assert dataset.name == "weak"
    assert dataset.n_galaxies == 3
    assert isinstance(dataset.noise_map, aa.ArrayIrregular)
    assert list(dataset.noise_map) == pytest.approx([0.3, 0.3, 0.3])
    assert dataset.shear_yx is shear_yx


def test__weak_dataset__construct_with_per_galaxy_noise_list():
    shear_yx = _make_shear_yx(
        positions=[(0.0, 0.0), (1.0, 1.0)], values=[(0.0, 0.0), (0.1, 0.1)]
    )

    dataset = al.WeakDataset(shear_yx=shear_yx, noise_map=[0.2, 0.4])

    assert list(dataset.noise_map) == pytest.approx([0.2, 0.4])


def test__weak_dataset__rejects_non_shear_yx_input():
    grid = aa.Grid2DIrregular(values=[(0.0, 0.0)])
    bad_shear = aa.VectorYX2DIrregular(values=np.array([[0.0, 0.0]]), grid=grid)

    with pytest.raises(TypeError):
        al.WeakDataset(shear_yx=bad_shear, noise_map=0.3)


def test__weak_dataset__rejects_mismatched_noise_length():
    shear_yx = _make_shear_yx(
        positions=[(0.0, 0.0), (1.0, 1.0)], values=[(0.0, 0.0), (0.1, 0.1)]
    )

    with pytest.raises(ValueError):
        al.WeakDataset(shear_yx=shear_yx, noise_map=[0.2, 0.4, 0.6])


def test__weak_dataset__positions_match_shear_grid():
    shear_yx = _make_shear_yx(
        positions=[(0.5, 0.5), (1.0, -0.3)], values=[(0.0, 0.0), (0.0, 0.0)]
    )

    dataset = al.WeakDataset(shear_yx=shear_yx, noise_map=0.3)

    assert dataset.positions is shear_yx.grid


def test__weak_dataset__extent_from_pads_position_bounding_box():
    shear_yx = _make_shear_yx(
        positions=[(0.0, 0.0), (1.0, 2.0), (-0.5, -0.4)],
        values=[(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
    )

    dataset = al.WeakDataset(shear_yx=shear_yx, noise_map=0.3)

    y_min, y_max, x_min, x_max = dataset.extent_from(buffer=0.1)
    assert y_min == pytest.approx(-0.6)
    assert y_max == pytest.approx(1.1)
    assert x_min == pytest.approx(-0.5)
    assert x_max == pytest.approx(2.1)
