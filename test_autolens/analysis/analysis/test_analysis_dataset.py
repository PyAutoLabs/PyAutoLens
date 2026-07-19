from pathlib import Path
import os
import pytest

from autonerves import conf
from autonerves.dictable import from_json

import autofit as af
import autolens as al
from autolens import exc

directory = Path(__file__).resolve().parent


def test__modify_before_fit__inversion_no_positions_likelihood__raises_exception(
    masked_imaging_7x7,
):
    lens = al.Galaxy(redshift=0.5, mass=al.mp.IsothermalSph())

    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(), regularization=al.reg.Constant()
    )

    source = al.Galaxy(redshift=1.0, pixelization=pixelization)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    analysis = al.AnalysisImaging(dataset=masked_imaging_7x7, use_jax=False)

    with pytest.raises(exc.AnalysisException):
        analysis.modify_before_fit(paths=af.DirectoryPaths(), model=model)

    positions_likelihood = al.PositionsLH(
        positions=al.Grid2DIrregular([(1.0, 100.0), (200.0, 2.0)]), threshold=0.01
    )

    analysis = al.AnalysisImaging(
        dataset=masked_imaging_7x7,
        positions_likelihood_list=[positions_likelihood],
        use_jax=False,
    )
    analysis.modify_before_fit(paths=af.DirectoryPaths(), model=model)


def test__save_results__tracer_output_to_json(analysis_imaging_7x7):
    lens = al.Galaxy(redshift=0.5)
    source = al.Galaxy(redshift=1.0)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    tracer = al.Tracer(galaxies=[lens, source])

    paths = af.DirectoryPaths()

    analysis_imaging_7x7.save_results(
        paths=paths,
        result=al.m.MockResult(max_log_likelihood_tracer=tracer, model=model),
    )

    tracer = from_json(file_path=paths._files_path / "tracer.json")

    assert tracer.galaxies[0].redshift == 0.5
    assert tracer.galaxies[1].redshift == 1.0

    os.remove(paths._files_path / "tracer.json")


def test__save_attributes__dataset_fits_output_for_aggregator(analysis_imaging_7x7):
    # Regression guard: `save_attributes` must always write `dataset.fits` to the
    # `files` folder so the aggregator loaders (`ImagingAgg`,
    # `agg_util.mask_header_from`) can reload the dataset via
    # `fit.value(name="dataset")`, independently of whether visualization ran.
    from astropy.io import fits

    paths = af.DirectoryPaths()

    analysis_imaging_7x7.save_attributes(paths=paths)

    dataset_fits_path = paths._files_path / "dataset.fits"

    assert dataset_fits_path.exists()

    with fits.open(dataset_fits_path) as hdu_list:
        ext_names = [hdu.name for hdu in hdu_list]

    assert ext_names[:4] == ["MASK", "DATA", "NOISE_MAP", "PSF"]

    os.remove(dataset_fits_path)
