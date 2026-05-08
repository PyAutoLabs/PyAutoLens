import numpy as np
import pytest

import autolens as al


def test__traced_grid_2d_list_from(grid_2d_7x7_simple):
    g0 = al.Galaxy(redshift=2.0, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0))
    g1 = al.Galaxy(redshift=2.0, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0))
    g2 = al.Galaxy(redshift=0.1, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0))
    g3 = al.Galaxy(redshift=3.0, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0))
    g4 = al.Galaxy(redshift=1.0, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0))
    g5 = al.Galaxy(redshift=3.0, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0))

    galaxies = [g0, g1, g2, g3, g4, g5]

    planes = al.util.tracer.planes_from(galaxies=galaxies)

    traced_grid_list = al.util.tracer.traced_grid_2d_list_from(
        planes=planes, grid=grid_2d_7x7_simple, cosmology=al.cosmo.Planck15()
    )

    # The scaling factors are as follows and were computed independently from the test_autoarray.
    beta_01 = 0.9348
    beta_02 = 0.9839601
    beta_12 = 0.7539734

    val = np.sqrt(2) / 2.0

    assert traced_grid_list[0][0] == pytest.approx(np.array([1.0, 1.0]), 1e-4)
    assert traced_grid_list[0][1] == pytest.approx(np.array([1.0, 0.0]), 1e-4)

    assert traced_grid_list[1][0] == pytest.approx(
        np.array([(1.0 - beta_01 * val), (1.0 - beta_01 * val)]), 1e-4
    )
    assert traced_grid_list[1][1] == pytest.approx(
        np.array([(1.0 - beta_01 * 1.0), 0.0]), 1e-4
    )

    defl11 = g0.deflections_yx_2d_from(
        grid=al.Grid2DIrregular([[(1.0 - beta_01 * val), (1.0 - beta_01 * val)]])
    )
    defl12 = g0.deflections_yx_2d_from(
        grid=al.Grid2DIrregular([[(1.0 - beta_01 * 1.0), 0.0]])
    )

    assert traced_grid_list[2][0] == pytest.approx(
        np.array(
            [
                (1.0 - beta_02 * val - beta_12 * defl11.array[0, 0]),
                (1.0 - beta_02 * val - beta_12 * defl11.array[0, 1]),
            ]
        ),
        1e-4,
    )
    assert traced_grid_list[2][1] == pytest.approx(
        np.array([(1.0 - beta_02 * 1.0 - beta_12 * defl12.array[0, 0]), 0.0]), 1e-4
    )

    assert traced_grid_list[3][1] == pytest.approx(np.array([1.0, 0.0]), 1e-4)

    traced_grid_list = al.util.tracer.traced_grid_2d_list_from(
        planes=planes,
        grid=grid_2d_7x7_simple,
        plane_index_limit=1,
        cosmology=al.cosmo.Planck15(),
    )

    # The scaling factors are as follows and were computed independently from the test_autoarray.
    beta_01 = 0.9348

    val = np.sqrt(2) / 2.0

    assert traced_grid_list[0][0] == pytest.approx(np.array([1.0, 1.0]), 1e-4)
    assert traced_grid_list[0][1] == pytest.approx(np.array([1.0, 0.0]), 1e-4)

    assert traced_grid_list[1][0] == pytest.approx(
        np.array([(1.0 - beta_01 * val), (1.0 - beta_01 * val)]), 1e-4
    )
    assert traced_grid_list[1][1] == pytest.approx(
        np.array([(1.0 - beta_01 * 1.0), 0.0]), 1e-4
    )

    assert len(traced_grid_list) == 2


def test__grid_2d_at_redshift_from(grid_2d_7x7):
    g0 = al.Galaxy(
        redshift=0.5,
        mass_profile=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.0),
    )
    g1 = al.Galaxy(
        redshift=0.75,
        mass_profile=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=2.0),
    )
    g2 = al.Galaxy(
        redshift=1.5,
        mass_profile=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=3.0),
    )
    g3 = al.Galaxy(
        redshift=1.0,
        mass_profile=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=4.0),
    )
    g4 = al.Galaxy(redshift=2.0)

    galaxies = [g0, g1, g2, g3, g4]
    planes = al.util.tracer.planes_from(galaxies=galaxies)

    traced_grid_list = al.util.tracer.traced_grid_2d_list_from(
        planes=planes, grid=grid_2d_7x7
    )

    grid_at_redshift = al.util.tracer.grid_2d_at_redshift_from(
        galaxies=galaxies, grid=grid_2d_7x7, redshift=0.5
    )

    assert grid_at_redshift == pytest.approx(traced_grid_list[0], 1.0e-4)

    grid_at_redshift = al.util.tracer.grid_2d_at_redshift_from(
        galaxies=galaxies, grid=grid_2d_7x7, redshift=0.75
    )

    assert grid_at_redshift == pytest.approx(traced_grid_list[1].array, 1.0e-4)

    grid_at_redshift = al.util.tracer.grid_2d_at_redshift_from(
        galaxies=galaxies, grid=grid_2d_7x7, redshift=1.0
    )

    assert grid_at_redshift == pytest.approx(traced_grid_list[2].array, 1.0e-4)

    grid_at_redshift = al.util.tracer.grid_2d_at_redshift_from(
        galaxies=galaxies, grid=grid_2d_7x7, redshift=1.5
    )

    assert grid_at_redshift == pytest.approx(traced_grid_list[3].array, 1.0e-4)

    grid_at_redshift = al.util.tracer.grid_2d_at_redshift_from(
        galaxies=galaxies, grid=grid_2d_7x7, redshift=2.0
    )

    assert grid_at_redshift == pytest.approx(traced_grid_list[4].array, 1.0e-4)


def test__grid_2d_at_redshift_from__redshift_between_planes(grid_2d_7x7):
    grid_2d_7x7[0] = al.Grid2DIrregular([[1.0, -1.0]])
    grid_2d_7x7[1] = al.Grid2DIrregular([[1.0, 0.0]])

    g0 = al.Galaxy(
        redshift=0.5,
        mass_profile=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.0),
    )
    g1 = al.Galaxy(
        redshift=0.75,
        mass_profile=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=2.0),
    )
    g2 = al.Galaxy(redshift=2.0)

    galaxies = [g0, g1, g2]

    grid_at_redshift = al.util.tracer.grid_2d_at_redshift_from(
        galaxies=galaxies, grid=grid_2d_7x7, redshift=1.9
    )

    assert grid_at_redshift[0][0] == pytest.approx(-1.06587, 1.0e-1)
    assert grid_at_redshift[0][1] == pytest.approx(1.06587, 1.0e-1)
    assert grid_at_redshift[1][0] == pytest.approx(-1.921583, 1.0e-1)
    assert grid_at_redshift[1][1] == pytest.approx(0.0, 1.0e-1)

    grid_at_redshift = al.util.tracer.grid_2d_at_redshift_from(
        galaxies=galaxies,
        grid=grid_2d_7x7.mask.derive_grid.all_false,
        redshift=0.3,
    )

    assert (grid_at_redshift == grid_2d_7x7.mask.derive_grid.all_false).all()


class _FakeTracedRedshift:
    """A redshift-like object that mimics a JAX traced scalar — calling ``float()``
    raises, so ``tracer_util._redshift_is_traced`` should return True. Used to
    exercise the JAX partition-and-splice path without importing ``jax`` (library
    unit tests stay numpy-only — see ``feedback_no_jax_in_unit_tests``)."""

    def __init__(self, name: str):
        self.name = name

    def __float__(self):
        raise TypeError("traced redshift cannot be coerced to float")

    def __repr__(self):
        return f"<TracedRedshift {self.name}>"


def test__redshift_is_traced__detects_traced_and_concrete():
    from autolens.lens import tracer_util

    assert tracer_util._redshift_is_traced(_FakeTracedRedshift("subhalo")) is True

    assert tracer_util._redshift_is_traced(0.5) is False
    assert tracer_util._redshift_is_traced(1) is False
    assert tracer_util._redshift_is_traced(np.float64(1.5)) is False
    assert tracer_util._redshift_is_traced(np.array(0.7)) is False


def test__plane_redshifts_from__partition_path__preserves_input_order():
    from autolens.lens import tracer_util

    lens = al.Galaxy(redshift=0.5)
    subhalo = al.Galaxy(redshift=_FakeTracedRedshift("subhalo"))
    source = al.Galaxy(redshift=1.0)

    plane_redshifts = tracer_util.plane_redshifts_from(
        galaxies=[lens, subhalo, source]
    )

    assert plane_redshifts[0] == 0.5
    assert isinstance(plane_redshifts[1], _FakeTracedRedshift)
    assert plane_redshifts[2] == 1.0


def test__plane_redshifts_from__partition_path__dedupes_concrete_only():
    from autolens.lens import tracer_util

    g0 = al.Galaxy(redshift=0.5)
    g1 = al.Galaxy(redshift=0.5)  # duplicate concrete redshift — should collapse
    subhalo = al.Galaxy(redshift=_FakeTracedRedshift("subhalo"))
    source = al.Galaxy(redshift=1.0)

    plane_redshifts = tracer_util.plane_redshifts_from(
        galaxies=[g0, g1, subhalo, source]
    )

    assert plane_redshifts[0] == 0.5
    assert isinstance(plane_redshifts[1], _FakeTracedRedshift)
    assert plane_redshifts[2] == 1.0
    assert len(plane_redshifts) == 3


def test__planes_from__partition_path__traced_galaxy_gets_dedicated_plane():
    from autolens.lens import tracer_util

    lens_a = al.Galaxy(redshift=0.5)
    lens_b = al.Galaxy(redshift=0.5)  # same plane as lens_a
    subhalo = al.Galaxy(redshift=_FakeTracedRedshift("subhalo"))
    source = al.Galaxy(redshift=1.0)

    planes = tracer_util.planes_from(galaxies=[lens_a, lens_b, subhalo, source])

    assert len(planes) == 3
    assert list(planes[0]) == [lens_a, lens_b]
    assert list(planes[1]) == [subhalo]
    assert list(planes[2]) == [source]


def test__time_delays_from():

    grid = al.Grid2DIrregular(values=[(0.7, 0.5), (1.0, 1.0)])

    mp = al.mp.Isothermal(
        centre=(0.0, 0.0), ell_comps=(0.0, -0.111111), einstein_radius=2.0
    )

    lens = al.Galaxy(redshift=0.2, mass=mp)
    source = al.Galaxy(redshift=0.7)

    time_delay = al.util.tracer.time_delays_from(
        galaxies=al.Galaxies([lens, source]),
        grid=grid,
        cosmology=al.cosmo.Planck15(),
    )

    assert time_delay == pytest.approx(np.array([8.52966247, -29.0176387]), 1.0e-4)
