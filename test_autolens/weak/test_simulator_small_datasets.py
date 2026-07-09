import autolens as al


def _tracer():
    lens = al.Galaxy(
        redshift=0.5,
        mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.6),
    )
    return al.Tracer(galaxies=[lens, al.Galaxy(redshift=1.0)])


def test__random_positions__capped_under_small_datasets(monkeypatch):
    monkeypatch.setenv("PYAUTO_SMALL_DATASETS", "1")

    dataset = al.SimulatorShearYX(noise_sigma=0.3, seed=1).via_tracer_random_positions_from(
        tracer=_tracer(), n_galaxies=200, grid_extent=3.0
    )

    assert dataset.n_galaxies == 25


def test__random_positions__uncapped_without_env_var(monkeypatch):
    monkeypatch.delenv("PYAUTO_SMALL_DATASETS", raising=False)

    dataset = al.SimulatorShearYX(noise_sigma=0.3, seed=1).via_tracer_random_positions_from(
        tracer=_tracer(), n_galaxies=40, grid_extent=3.0
    )

    assert dataset.n_galaxies == 40


def test__explicit_grid__never_capped(monkeypatch):
    import autoarray as aa
    import numpy as np

    monkeypatch.setenv("PYAUTO_SMALL_DATASETS", "1")

    grid = aa.Grid2DIrregular(values=np.random.default_rng(1).uniform(-3, 3, (60, 2)))
    dataset = al.SimulatorShearYX(noise_sigma=0.3, seed=1).via_tracer_from(
        tracer=_tracer(), grid=grid
    )

    assert dataset.n_galaxies == 60
