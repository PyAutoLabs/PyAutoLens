from pathlib import Path
import pytest

import autofit as af
import autoarray as aa
import autolens as al
from autolens import exc

from autolens.interferometer.model.result import ResultInterferometer

directory = Path(__file__).resolve().parent


def test__make_result__result_interferometer_is_returned(interferometer_7):
    model = af.Collection(galaxies=af.Collection(galaxy_0=al.Galaxy(redshift=0.5)))

    analysis = al.AnalysisInterferometer(dataset=interferometer_7, use_jax=False)

    search = al.m.MockSearch(name="test_search")

    result = search.fit(model=model, analysis=analysis)

    assert isinstance(result, ResultInterferometer)


def test__figure_of_merit__matches_correct_fit_given_galaxy_profiles(interferometer_7):
    lens_galaxy = al.Galaxy(redshift=0.5, light=al.lp.Sersic(intensity=0.1))

    model = af.Collection(galaxies=af.Collection(lens=lens_galaxy))

    analysis = al.AnalysisInterferometer(dataset=interferometer_7, use_jax=False)

    instance = model.instance_from_unit_vector([])
    analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

    tracer = analysis.tracer_via_instance_from(instance=instance)

    fit = al.FitInterferometer(dataset=interferometer_7, tracer=tracer)

    assert fit.log_likelihood == analysis_log_likelihood


def test__positions__likelihood_overwrite__changes_likelihood(
    interferometer_7, mask_2d_7x7
):
    lens = al.Galaxy(redshift=0.5, mass=al.mp.IsothermalSph(centre=(0.05, 0.05)))
    source = al.Galaxy(redshift=1.0, light=al.lp.SersicSph(centre=(0.05, 0.05)))

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    analysis = al.AnalysisInterferometer(dataset=interferometer_7, use_jax=False)

    instance = model.instance_from_unit_vector([])
    analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

    tracer = analysis.tracer_via_instance_from(instance=instance)

    fit = al.FitInterferometer(dataset=interferometer_7, tracer=tracer)

    assert fit.log_likelihood == analysis_log_likelihood
    assert analysis_log_likelihood == pytest.approx(-62.463179940, 1.0e-4)

    positions_likelihood = al.PositionsLH(
        positions=al.Grid2DIrregular([(1.0, 100.0), (200.0, 2.0)]), threshold=0.01
    )

    analysis = al.AnalysisInterferometer(
        dataset=interferometer_7,
        positions_likelihood_list=[positions_likelihood],
        use_jax=False,
    )
    analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

    assert analysis_log_likelihood == pytest.approx(-44097289569.2342, 1.0e-4)


def _pixelization_model():
    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=0.01),
    )
    return af.Collection(
        galaxies=af.Collection(
            galaxy_0=al.Galaxy(redshift=0.5),
            galaxy_1=al.Galaxy(redshift=0.5, pixelization=pixelization),
        )
    )


def test__shared_state_from__preloads_curvature_reused__figure_of_merit_unchanged(
    interferometer_7,
):
    dataset = interferometer_7.apply_sparse_operator(use_jax=False)

    model = _pixelization_model()
    instance = model.instance_from_unit_vector([])

    analysis = al.AnalysisInterferometer(
        dataset=dataset, use_jax=False, shared_preloads=True
    )

    # `shared_state_from` builds a `PreloadsInterferometer` carrying the curvature matrix `F` and
    # the mapper (the channel-invariant inversion-setup quantities).
    shared = analysis.shared_state_from(instance=instance)
    assert isinstance(shared, aa.PreloadsInterferometer)
    assert shared.curvature_matrix is not None
    assert shared.mapper_galaxy_dict is not None

    # The preloaded `F` and mapper are reused by the fit (identity) and leave the figure of merit
    # unchanged. Reusing the mapper means the Delaunay triangulation is not rebuilt per channel.
    fit_unshared = analysis.fit_from(instance=instance)
    fit_shared = analysis.fit_from(instance=instance, preloads=shared)

    assert fit_shared.inversion.curvature_matrix is shared.curvature_matrix
    assert fit_shared.tracer_to_inversion.mapper_galaxy_dict is shared.mapper_galaxy_dict
    assert fit_shared.figure_of_merit == pytest.approx(fit_unshared.figure_of_merit)

    # The full `log_likelihood_function` with the shared object matches the unshared call.
    assert analysis.log_likelihood_function(
        instance=instance, shared=shared
    ) == pytest.approx(analysis.log_likelihood_function(instance=instance))


def test__shared_state_from__returns_none_when_not_opted_in(interferometer_7):
    dataset = interferometer_7.apply_sparse_operator(use_jax=False)

    model = _pixelization_model()
    instance = model.instance_from_unit_vector([])

    analysis = al.AnalysisInterferometer(dataset=dataset, use_jax=False)

    assert analysis.shared_state_from(instance=instance) is None
