import os

import numpy as np

import autoarray as aa
import autofit as af
import autolens as al

from autolens.weak.fit import FitWeak
from autolens.weak.model.result import ResultWeak


def _lens_model(einstein_radius=1.6, ell_comps=(0.0, 0.05)):
    return af.Collection(
        galaxies=af.Collection(
            lens=al.Galaxy(
                redshift=0.5,
                mass=al.mp.Isothermal(
                    centre=(0.0, 0.0),
                    ell_comps=ell_comps,
                    einstein_radius=einstein_radius,
                ),
            ),
            source=al.Galaxy(redshift=1.0),
        )
    )


def _make_dataset(noise_sigma=0.3, seed=1, einstein_radius=1.6):
    grid = aa.Grid2DIrregular(
        values=[(0.7, 0.5), (1.0, 1.0), (-0.3, 0.6), (-1.1, -0.8)]
    )
    instance = _lens_model(einstein_radius=einstein_radius).instance_from_unit_vector(
        []
    )
    tracer = al.Tracer(galaxies=[instance.galaxies.lens, instance.galaxies.source])
    simulator = al.SimulatorShearYX(noise_sigma=noise_sigma, seed=seed)
    return simulator.via_tracer_from(tracer=tracer, grid=grid, name="test")


def test__log_likelihood__matches_fit_weak():
    dataset = _make_dataset()

    model = _lens_model()
    instance = model.instance_from_unit_vector([])

    analysis = al.AnalysisWeak(dataset=dataset)

    analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

    tracer = analysis.tracer_via_instance_from(instance=instance)

    fit = FitWeak(dataset=dataset, tracer=tracer)

    assert fit.log_likelihood == analysis_log_likelihood


def test__log_likelihood__noise_free_round_trip_gives_zero_chi_squared():
    """
    `FitWeak.model_shear` uses the same `LensCalc.shear_yx_2d_via_hessian_from` primitive as
    `SimulatorShearYX`, so fitting a noise-free dataset with its own truth tracer must give
    chi_squared == 0 through the analysis path too.
    """
    dataset = _make_dataset(noise_sigma=0.0)
    dataset.noise_map = aa.ArrayIrregular(values=[0.3, 0.3, 0.3, 0.3])

    analysis = al.AnalysisWeak(dataset=dataset)

    instance = _lens_model().instance_from_unit_vector([])

    fit = analysis.fit_from(instance=instance)

    assert fit.chi_squared == 0.0
    np.testing.assert_allclose(fit.residual_map, 0.0, atol=1e-12)


def test__log_likelihood__changes_when_model_wrong():
    dataset = _make_dataset(noise_sigma=0.0)
    dataset.noise_map = aa.ArrayIrregular(values=[0.3, 0.3, 0.3, 0.3])

    analysis = al.AnalysisWeak(dataset=dataset)

    truth = _lens_model(einstein_radius=1.6).instance_from_unit_vector([])
    wrong = _lens_model(einstein_radius=1.0).instance_from_unit_vector([])

    assert analysis.log_likelihood_function(
        instance=wrong
    ) < analysis.log_likelihood_function(instance=truth)


def test__result_class_is_result_weak():
    assert al.AnalysisWeak.Result is ResultWeak


def test__save_attributes__dataset_json_round_trips():
    dataset = _make_dataset()

    analysis = al.AnalysisWeak(dataset=dataset)

    paths = af.DirectoryPaths()

    analysis.save_attributes(paths=paths)

    loaded = al.from_json(file_path=paths._files_path / "dataset.json")

    assert isinstance(loaded, al.WeakDataset)
    assert loaded.name == "test"
    np.testing.assert_allclose(
        np.asarray(loaded.shear_yx), np.asarray(dataset.shear_yx)
    )

    os.remove(paths._files_path / "dataset.json")
