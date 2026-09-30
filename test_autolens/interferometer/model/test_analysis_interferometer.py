import importlib.util
from pathlib import Path
import pytest

import autofit as af
import autoarray as aa
import autolens as al
from autolens import exc

# The interferometer shared-state preload builds the jax-backed NUFFT sparse
# operator, so these tests need jax installed to run (it ships via the
# `[optional]` extras). The NumPy-only Python-version matrix has no jax, so skip
# there rather than fail.
requires_jax = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="requires jax (installed via the [optional] extras; absent on the NumPy-only matrix env)",
)

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

    assert analysis_log_likelihood == pytest.approx(-22048644815.84869, 1.0e-4)


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


@requires_jax
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


@requires_jax
def test__shared_state_from__returns_none_when_not_opted_in(interferometer_7):
    dataset = interferometer_7.apply_sparse_operator(use_jax=False)

    model = _pixelization_model()
    instance = model.instance_from_unit_vector([])

    analysis = al.AnalysisInterferometer(dataset=dataset, use_jax=False)

    assert analysis.shared_state_from(instance=instance) is None


def test__preloads_scoped__cross_type_preloads_reduced_to_mesh_view(interferometer_7):
    lens = al.Galaxy(redshift=0.5, light=al.lp.Sersic(intensity=0.1))
    tracer = al.Tracer(galaxies=[lens])

    # Cross-dataset-type preloads (e.g. from an imaging lead factor in a joint graph): only
    # the mesh-geometry view is valid for an interferometer fit.
    cross_type = aa.PreloadsImaging(
        source_plane_mesh_grid=[["mesh"]], image_plane_mesh_grid=[["image-mesh"]]
    )

    fit = al.FitInterferometer(
        dataset=interferometer_7, tracer=tracer, preloads=cross_type
    )

    scoped = fit._preloads_scoped
    assert isinstance(scoped, aa.PreloadsInterferometer)
    assert scoped.source_plane_mesh_grid == [["mesh"]]
    assert scoped.curvature_matrix is None
    assert scoped.mapper_galaxy_dict is None

    same_type = aa.PreloadsInterferometer(curvature_matrix="F")
    fit = al.FitInterferometer(
        dataset=interferometer_7, tracer=tracer, preloads=same_type
    )
    assert fit._preloads_scoped is same_type


@requires_jax
def test__shared_state_from__populates_mesh_geometry_fields(interferometer_7):
    dataset = interferometer_7.apply_sparse_operator(use_jax=False)

    model = _pixelization_model()
    instance = model.instance_from_unit_vector([])

    analysis = al.AnalysisInterferometer(
        dataset=dataset, use_jax=False, shared_preloads=True
    )

    # The mesh-geometry fields ride alongside the curvature matrix + mapper so that
    # cross-dataset-type factors of a joint graph can consume the shared mesh.
    shared = analysis.shared_state_from(instance=instance)
    assert shared.source_plane_mesh_grid is not None
    assert shared.image_plane_mesh_grid is not None


class _FitStub:
    """
    The part of a `PyAutoFit` aggregator `Fit` the interferometer loader reads: `value(name)` returning the
    `dataset.fits` HDU list written by `save_attributes` and the saved `transformer_class` json (if any).
    """

    def __init__(self, paths):
        self.paths = paths
        self.children = []

    def value(self, name):
        from astropy.io import fits
        from autonerves.dictable import from_dict

        if name == "dataset":
            return fits.open(self.paths.image_path / "dataset.fits")

        if name == "transformer_class":
            path = self.paths._files_path / "transformer_class.json"

            if not path.exists():
                return None

            return from_dict(self.paths.load_json("transformer_class"))

        return None


def test__save_attributes__array_free_dataset__aggregator_round_trip(
    interferometer_7, tmp_path
):
    """
    `save_attributes` writes an array-free dataset's `SparseTerms` to `dataset.fits` and the aggregator loader
    (shared with autogalaxy) rebuilds it via `Interferometer.from_sparse_terms`, so a fit of the reloaded
    dataset with a tracer reproduces the original `log_evidence`.
    """
    from astropy.io import fits

    from autolens.aggregator import _interferometer_from

    dataset = aa.Interferometer.from_stream(
        [
            (
                interferometer_7.uv_wavelengths,
                interferometer_7.data,
                interferometer_7.noise_map,
            )
        ],
        real_space_mask=interferometer_7.real_space_mask,
        transformer_class=type(interferometer_7.transformer),
    )

    paths = af.DirectoryPaths(name="array_free_round_trip", path_prefix=str(tmp_path))

    analysis = al.AnalysisInterferometer(dataset=dataset, use_jax=False)
    analysis.save_attributes(paths=paths)

    with fits.open(paths.image_path / "dataset.fits") as hdu_list:
        assert [hdu.name for hdu in hdu_list] == [
            "MASK",
            "NUFFT_PRECISION_OPERATOR",
            "DIRTY_IMAGE",
            "DIRTY_BEAM",
            "SPARSE_TERMS_SCALARS",
        ]

    assert not (paths._files_path / "transformer_class.json").exists()

    dataset_reloaded = _interferometer_from(fit=_FitStub(paths=paths))[0]

    assert dataset_reloaded.is_array_free
    assert (
        dataset_reloaded.sparse_operator.data_term
        == dataset.sparse_operator.data_term
    )
    assert (
        dataset_reloaded.sparse_operator.noise_normalization
        == dataset.sparse_operator.noise_normalization
    )

    lens = al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.0),
    )
    source = al.Galaxy(
        redshift=1.0,
        pixelization=al.Pixelization(
            mesh=al.mesh.RectangularUniform(shape=(3, 3)),
            regularization=al.reg.Constant(coefficient=1.0),
        ),
    )
    tracer = al.Tracer(galaxies=[lens, source])

    log_evidence = al.FitInterferometer(dataset=dataset, tracer=tracer).log_evidence
    log_evidence_reloaded = al.FitInterferometer(
        dataset=dataset_reloaded, tracer=tracer
    ).log_evidence

    assert log_evidence_reloaded == pytest.approx(log_evidence, rel=1.0e-8)
