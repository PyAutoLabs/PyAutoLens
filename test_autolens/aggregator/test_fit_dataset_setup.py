"""
A lens `FitImaging` / `FitInterferometer` reloaded by the aggregator reapplies the compute setup the search
wrote to `files/setup.json`: the lens aggregator loads its datasets through PyAutoGalaxy's `_imaging_from` /
`_interferometer_from`, which call `agg_util.dataset_with_setup_from`.
"""
from types import SimpleNamespace

import pytest

import autofit as af
import autoarray as aa
import autogalaxy as ag
import autolens as al

from autogalaxy.analysis import setup_util
from autogalaxy.analysis.analysis.dataset import AnalysisDataset
from autogalaxy.interferometer.model.analysis import interferometer_hdu_list_from

from autolens.aggregator.fit_imaging import _fit_imaging_from
from autolens.aggregator.fit_interferometer import _fit_interferometer_from


def _imaging_hdu_list_from(dataset, monkeypatch):
    """The `dataset.fits` HDU list `AnalysisImaging.save_attributes` writes for `dataset`."""
    captured = []

    monkeypatch.setattr(
        AnalysisDataset,
        "_save_dataset_fits",
        staticmethod(lambda paths, hdu_list: captured.append(hdu_list)),
    )

    analysis = ag.AnalysisImaging(dataset=dataset, use_jax=False)
    analysis.save_attributes(paths=SimpleNamespace(save_json=lambda **kwargs: None))

    monkeypatch.undo()

    return captured[0]


def _lens_instance():
    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=al.Galaxy(
                redshift=0.5,
                bulge=al.lp.Sersic(intensity=0.1, centre=(0.05, 0.05)),
                mass=al.mp.IsothermalSph(einstein_radius=0.2),
            ),
            source=al.Galaxy(redshift=1.0, pixelization=pixelization),
        ),
    )

    return model.instance_from_prior_medians()


def _fake_fit(values):
    """A `Fit` stand-in whose `value(name)` returns `values.get(name)` (None for a missing file)."""
    return SimpleNamespace(
        children=[], instance=_lens_instance(), value=lambda name: values.get(name)
    )


def _setup_dict(setup_id):
    return setup_util.setup_dict_from(
        setup=setup_util.setup_input_from(setup_id),
        chosen_by=setup_util.CHOSEN_BY_EXPLICIT,
    )


def test__fit_imaging_from__setup_json__reapplied(masked_imaging_7x7, monkeypatch):
    pytest.importorskip("numba")

    hdu_list = _imaging_hdu_list_from(masked_imaging_7x7, monkeypatch)

    fit_reloaded_no_setup = _fit_imaging_from(fit=_fake_fit({"dataset": hdu_list}))[0]

    assert fit_reloaded_no_setup.dataset.setup is None
    assert fit_reloaded_no_setup.sparse_operator is None

    fit_reloaded = _fit_imaging_from(
        fit=_fake_fit(
            {"dataset": hdu_list, "setup": _setup_dict("numba_cpu_sparse_fp64")}
        )
    )[0]

    assert fit_reloaded.dataset.setup.setup_id == "numba_cpu_sparse_fp64"
    assert fit_reloaded.dataset.sparse_operator is not None

    # The reloaded lens fit takes the sparse route and reproduces the search's (dense) figure of merit.
    assert fit_reloaded.inversion.dataset.sparse_operator is (
        fit_reloaded.dataset.sparse_operator
    )
    assert fit_reloaded.figure_of_merit == pytest.approx(
        fit_reloaded_no_setup.figure_of_merit, rel=1.0e-6
    )


def test__fit_interferometer_from__setup_json__reapplied(interferometer_7):
    pytest.importorskip("finufft")

    values = {
        "dataset": interferometer_hdu_list_from(dataset=interferometer_7),
        "transformer_class": aa.TransformerDFT,
    }

    fit_reloaded_no_setup = _fit_interferometer_from(fit=_fake_fit(values))[0]

    assert fit_reloaded_no_setup.dataset.setup is None
    assert fit_reloaded_no_setup.sparse_operator is None

    values["setup"] = _setup_dict("numba_cpu_sparse_fp64_wtilde")

    fit_reloaded = _fit_interferometer_from(fit=_fake_fit(values))[0]

    assert fit_reloaded.dataset.setup.setup_id == "numba_cpu_sparse_fp64_wtilde"
    assert fit_reloaded.dataset.sparse_operator is not None
    assert fit_reloaded.inversion.dataset.sparse_operator is (
        fit_reloaded.dataset.sparse_operator
    )
    assert fit_reloaded.log_likelihood == pytest.approx(
        fit_reloaded_no_setup.log_likelihood, rel=1.0e-8
    )
