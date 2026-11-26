from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import autoarray as aa
import autofit as af
import autolens as al

from autogalaxy.util.shear_field import ShearYX2DIrregular

from autolens.weak.model.visualizer import VisualizerWeak

directory = Path(__file__).resolve().parent


def _lens_model(einstein_radius=1.6):
    return af.Collection(
        galaxies=af.Collection(
            lens=al.Galaxy(
                redshift=0.5,
                mass=al.mp.Isothermal(
                    centre=(0.0, 0.0),
                    ell_comps=(0.0, 0.05),
                    einstein_radius=einstein_radius,
                ),
            ),
            source=al.Galaxy(redshift=1.0),
        )
    )


def _make_dataset():
    grid = aa.Grid2DIrregular(
        values=[(0.7, 0.5), (1.0, 1.0), (-0.3, 0.6), (-1.1, -0.8)]
    )
    instance = _lens_model().instance_from_unit_vector([])
    tracer = al.Tracer(galaxies=[instance.galaxies.lens, instance.galaxies.source])
    return al.SimulatorShearYX(noise_sigma=0.1, seed=1).via_tracer_from(
        tracer=tracer, grid=grid, name="test"
    )


def test__fit_for_visualization__jax_analysis__returns_numpy_structured_fit():
    """
    PyAutoLens#614: on the JAX path `FitWeak.model_shear` is a raw `(N, 2)` array, which the
    shear-field plotters cannot draw (`shear_yx.grid`). The visualizer's fit must be the NumPy fit,
    whose model shear is a `ShearYX2DIrregular`, with the same log likelihood as the JAX fit.
    """
    pytest.importorskip("jax")

    dataset = _make_dataset()
    analysis = al.AnalysisWeak(dataset=dataset, use_jax=True)
    instance = _lens_model(einstein_radius=1.5).instance_from_unit_vector([])

    fit = analysis.fit_for_visualization(instance=instance)

    assert isinstance(fit.model_shear, ShearYX2DIrregular)
    np.testing.assert_allclose(
        fit.log_likelihood,
        float(analysis.fit_from(instance=instance).log_likelihood),
        rtol=1e-8,
    )


def test__visualize__jax_analysis__writes_fit_subplot(plot_patch):
    """
    End-to-end form of PyAutoLens#614: visualizing the max-likelihood fit of a JAX `AnalysisWeak`
    raised `AttributeError: 'ArrayImpl' object has no attribute 'grid'` before the fix.
    """
    pytest.importorskip("jax")

    dataset = _make_dataset()
    analysis = al.AnalysisWeak(dataset=dataset, use_jax=True)
    instance = _lens_model(einstein_radius=1.5).instance_from_unit_vector([])

    paths = SimpleNamespace(image_path=directory / "files")

    VisualizerWeak.visualize(
        analysis=analysis,
        paths=paths,
        instance=instance,
        during_analysis=False,
        quick_update=True,
    )

    assert str(directory / "files" / "subplot_fit_weak.png") in plot_patch.paths
