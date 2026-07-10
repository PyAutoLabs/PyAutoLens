from pathlib import Path
import pytest

import autofit as af

import autolens as al

from autolens.imaging.model.result import ResultImaging

from autolens import exc


directory = Path(__file__).resolve().parent


def test__make_result__result_imaging_is_returned(masked_imaging_7x7):

    model = af.Collection(galaxies=af.Collection(galaxy_0=al.Galaxy(redshift=0.5)))

    analysis = al.AnalysisImaging(dataset=masked_imaging_7x7, use_jax=False)

    search = al.m.MockSearch(name="test_search")

    result = search.fit(model=model, analysis=analysis)

    assert isinstance(result, ResultImaging)


def test__figure_of_merit__matches_correct_fit_given_galaxy_profiles(
    masked_imaging_7x7,
):
    lens = al.Galaxy(redshift=0.5, light=al.lp.Sersic(intensity=0.1))

    model = af.Collection(galaxies=af.Collection(lens=lens))

    analysis = al.AnalysisImaging(dataset=masked_imaging_7x7, use_jax=False)
    instance = model.instance_from_unit_vector([])
    analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

    tracer = analysis.tracer_via_instance_from(instance=instance)

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.log_likelihood == analysis_log_likelihood


def test__log_likelihood_function__returns_figure_of_merit_for_pixelization(
    masked_imaging_7x7,
):
    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )

    lens = al.Galaxy(
        redshift=0.5,
        mass=al.mp.IsothermalSph(einstein_radius=1.0),
    )
    source = al.Galaxy(redshift=1.0, pixelization=pixelization)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))
    instance = model.instance_from_unit_vector([])

    analysis = al.AnalysisImaging(dataset=masked_imaging_7x7, use_jax=False)
    analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

    tracer = analysis.tracer_via_instance_from(instance=instance)
    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert analysis_log_likelihood == pytest.approx(fit.figure_of_merit)
    assert fit.figure_of_merit != pytest.approx(fit.log_likelihood, rel=1e-6)


def test__positions__likelihood_overwrites__changes_likelihood(masked_imaging_7x7):
    lens = al.Galaxy(redshift=0.5, mass=al.mp.IsothermalSph(centre=(0.05, 0.05)))
    source = al.Galaxy(redshift=1.0, light=al.lp.SersicSph(centre=(0.05, 0.05)))

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    instance = model.instance_from_unit_vector([])

    analysis = al.AnalysisImaging(dataset=masked_imaging_7x7, use_jax=False)
    analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

    tracer = analysis.tracer_via_instance_from(instance=instance)

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.log_likelihood == pytest.approx(analysis_log_likelihood, 1.0e-4)
    assert analysis_log_likelihood == pytest.approx(-14.79034680979, 1.0e-4)

    positions_likelihood = al.PositionsLH(
        positions=al.Grid2DIrregular([(1.0, 100.0), (200.0, 2.0)]), threshold=0.01
    )

    analysis = al.AnalysisImaging(
        dataset=masked_imaging_7x7, positions_likelihood_list=[positions_likelihood], use_jax=False
    )
    analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

    assert analysis_log_likelihood == pytest.approx(-44097289521.734665, 1.0e-4)


def test__positions__likelihood_overwrites__changes_likelihood__double_source_plane_example(masked_imaging_7x7):

    lens = al.Galaxy(redshift=0.5, mass=al.mp.IsothermalSph(centre=(0.05, 0.05)))
    source_0 = al.Galaxy(redshift=1.0, light=al.lp.SersicSph(centre=(0.05, 0.05)))
    source_1 = al.Galaxy(redshift=2.0, light=al.lp.SersicSph(centre=(0.05, 0.05)))

    model = af.Collection(galaxies=af.Collection(lens=lens, source_0=source_0, source_1=source_1))

    instance = model.instance_from_unit_vector([])

    positions_likelihood_0 = al.PositionsLH(
        plane_redshift=1.0, positions=al.Grid2DIrregular([(1.0, 100.0), (200.0, 2.0)]), threshold=0.01
    )
    positions_likelihood_1 = al.PositionsLH(
        plane_redshift=2.0, positions=al.Grid2DIrregular([(1.0, 100.0), (200.0, 2.0)]), threshold=0.01
    )

    analysis = al.AnalysisImaging(
        dataset=masked_imaging_7x7, positions_likelihood_list=[positions_likelihood_0, positions_likelihood_1], use_jax=False
    )
    analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

    assert analysis_log_likelihood == pytest.approx(-44097289521.734665, 1.0e-4)



def _shared_mesh_analysis(masked_imaging_7x7, shared_preloads):
    """
    An `AnalysisImaging` with an image-mesh (`Overlay`) + `Delaunay` pixelization, the regime the
    multi-exposure shared-state path applies to (the source-plane mesh is traced from image-plane
    mesh centres, so it can be shared across exposures).
    """
    import autoarray as aa

    lens = al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=1.0),
    )

    pixelization = al.Pixelization(
        mesh=al.mesh.Delaunay(pixels=9, zeroed_pixels=0),
        regularization=al.reg.Constant(coefficient=0.01),
    )

    source = al.Galaxy(redshift=1.0, pixelization=pixelization)

    image_mesh = al.image_mesh.Overlay(shape=(3, 3))
    image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
        mask=masked_imaging_7x7.mask,
    )

    adapt_images = al.AdaptImages(
        galaxy_name_image_dict={
            str(("galaxies", "source")): masked_imaging_7x7.data,
        },
        galaxy_name_image_plane_mesh_grid_dict={
            str(("galaxies", "source")): image_plane_mesh_grid,
        },
    )

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    analysis = al.AnalysisImaging(
        dataset=masked_imaging_7x7,
        adapt_images=adapt_images,
        use_jax=False,
        shared_preloads=shared_preloads,
        raise_inversion_positions_likelihood_exception=False,
    )

    return model, analysis


def test__shared_state_from__mesh_reused__figure_of_merit_unchanged(
    masked_imaging_7x7,
):
    import autoarray as aa

    model, analysis = _shared_mesh_analysis(masked_imaging_7x7, shared_preloads=True)
    instance = model.instance_from_unit_vector([])

    # `shared_state_from` builds a `PreloadsImaging` carrying the source-plane mesh geometry (the
    # exposure-invariant quantity) — NOT the mapper / curvature matrix / regularization matrix,
    # which are per-exposure for imaging (PSFs, offsets, adaptive regularization).
    shared = analysis.shared_state_from(instance=instance)
    assert isinstance(shared, aa.PreloadsImaging)
    assert shared.source_plane_mesh_grid is not None
    assert shared.image_plane_mesh_grid is not None
    assert shared.curvature_matrix is None
    assert shared.mapper_galaxy_dict is None

    # The preloaded mesh is reused by the fit (identity) and leaves the figure of merit unchanged.
    fit_unshared = analysis.fit_from(instance=instance)
    fom_unshared = fit_unshared.figure_of_merit

    fit_shared = analysis.fit_from(instance=instance, preloads=shared)

    assert (
        fit_shared.tracer_to_inversion.traced_mesh_grid_pg_list
        is shared.source_plane_mesh_grid
    )
    assert fit_shared.figure_of_merit == pytest.approx(fom_unshared)

    # The full `log_likelihood_function` with the shared object matches the unshared call.
    assert analysis.log_likelihood_function(
        instance=instance, shared=shared
    ) == pytest.approx(analysis.log_likelihood_function(instance=instance))


def test__shared_state_from__returns_none_when_not_opted_in(masked_imaging_7x7):
    model, analysis = _shared_mesh_analysis(masked_imaging_7x7, shared_preloads=False)
    instance = model.instance_from_unit_vector([])

    assert analysis.shared_state_from(instance=instance) is None


def test__shared_state_from__returns_none_when_no_inversion(masked_imaging_7x7):
    lens = al.Galaxy(redshift=0.5, light=al.lp.Sersic(intensity=0.1))

    model = af.Collection(galaxies=af.Collection(lens=lens))
    instance = model.instance_from_unit_vector([])

    analysis = al.AnalysisImaging(
        dataset=masked_imaging_7x7, use_jax=False, shared_preloads=True
    )

    assert analysis.shared_state_from(instance=instance) is None


def _factor_graph_log_likelihood(masked_imaging_7x7, shared_preloads):
    factors = []
    model = None
    for _ in range(2):
        model, analysis = _shared_mesh_analysis(masked_imaging_7x7, shared_preloads)
        factors.append(af.AnalysisFactor(prior_model=model.copy(), analysis=analysis))

    factor_graph = af.FactorGraphModel(*factors)
    instance = factor_graph.global_prior_model.instance_from_unit_vector([])
    return factor_graph.log_likelihood_function(instance)


def test__factor_graph__shared_vs_unshared_equal(masked_imaging_7x7):
    ll_unshared = _factor_graph_log_likelihood(masked_imaging_7x7, shared_preloads=False)
    ll_shared = _factor_graph_log_likelihood(masked_imaging_7x7, shared_preloads=True)

    print(f"\nunshared={ll_unshared} shared={ll_shared}")
    assert ll_shared == pytest.approx(ll_unshared, rel=1e-10)


def test__factor_graph__shared_state_computed_once(masked_imaging_7x7, monkeypatch):
    calls = {"n": 0}

    original = al.AnalysisImaging.shared_state_from

    def counting(self, instance):
        result = original(self, instance)
        if result is not None:
            calls["n"] += 1
        return result

    monkeypatch.setattr(al.AnalysisImaging, "shared_state_from", counting)

    _factor_graph_log_likelihood(masked_imaging_7x7, shared_preloads=True)

    assert calls["n"] == 1


def test__preloads_scoped__cross_type_preloads_reduced_to_mesh_view(masked_imaging_7x7):
    import autoarray as aa

    lens = al.Galaxy(redshift=0.5, light=al.lp.Sersic(intensity=0.1))
    tracer = al.Tracer(galaxies=[lens])

    # Cross-dataset-type preloads (e.g. from an interferometer lead factor in a joint graph):
    # the mapper / curvature matrix embed the other dataset's grids and must NOT be consumed
    # by an imaging fit — only the mesh-geometry view survives the scoping.
    cross_type = aa.PreloadsInterferometer(
        curvature_matrix="other-datasets-F",
        mapper_galaxy_dict="other-datasets-mapper",
        source_plane_mesh_grid=[["mesh"]],
        image_plane_mesh_grid=[["image-mesh"]],
    )

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer, preloads=cross_type)

    scoped = fit._preloads_scoped
    assert isinstance(scoped, aa.PreloadsImaging)
    assert scoped.source_plane_mesh_grid == [["mesh"]]
    assert scoped.image_plane_mesh_grid == [["image-mesh"]]
    assert scoped.curvature_matrix is None
    assert scoped.mapper_galaxy_dict is None

    # Same-type preloads pass through untouched.
    same_type = aa.PreloadsImaging(source_plane_mesh_grid=[["mesh"]])
    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer, preloads=same_type)
    assert fit._preloads_scoped is same_type
