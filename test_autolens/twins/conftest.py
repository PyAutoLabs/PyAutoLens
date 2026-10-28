import pytest

import autofit as af
import autolens as al

# Coarse aperture integration keeps the 7x7 tests fast; the witness compares masses
# measured with the same settings, so its tolerance does not depend on them.
COARSE_APERTURE = {
    "sub_size_interior": 4,
    "sub_size_edge": 8,
    "sub_size_centre": 16,
    "centre_refine_pixels": 2.0,
}

GNFW_APERTURE = {
    "sub_size_interior": 2,
    "sub_size_edge": 2,
    "sub_size_centre": 4,
    "centre_refine_pixels": 1.0,
}


def make_witness_tracer():
    """
    An SIE + external shear lens with an MGE (Basis of lp.Gaussian) light, and a
    parametric source, keyed "lens" / "source" like a model-fit's tracer.json.
    """
    mge = al.lp_basis.Basis(
        profile_list=[
            al.lp.Gaussian(
                centre=(0.0, 0.0),
                ell_comps=(0.05, 0.0),
                intensity=intensity,
                sigma=sigma,
            )
            for intensity, sigma in ((2.0, 0.3), (1.0, 0.8), (0.3, 1.6))
        ]
    )
    lens = al.Galaxy(
        redshift=0.5,
        bulge=mge,
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0), ell_comps=(0.05, 0.0), einstein_radius=1.0
        ),
        shear=al.mp.ExternalShear(gamma_1=0.03, gamma_2=-0.02),
    )
    source = al.Galaxy(
        redshift=1.0,
        bulge=al.lp.SersicSph(
            centre=(0.1, 0.1), intensity=1.0, effective_radius=0.5, sersic_index=1.5
        ),
    )
    return al.Tracer(galaxies=af.ModelInstance({"lens": lens, "source": source}))


@pytest.fixture(name="witness_fit_paths")
def make_witness_fit_paths(tmp_path, analysis_imaging_7x7):
    """A model-fit output directory (image/dataset.fits, files/tracer.json) of the witness lens."""
    tracer = make_witness_tracer()
    model = af.Collection(
        galaxies=af.Collection(
            lens=al.Galaxy(redshift=0.5), source=al.Galaxy(redshift=1.0)
        )
    )
    paths = af.DirectoryPaths(name="witness_fit", path_prefix=str(tmp_path / "fits"))
    analysis_imaging_7x7.save_attributes(paths=paths)
    analysis_imaging_7x7.save_results(
        paths=paths,
        result=al.m.MockResult(max_log_likelihood_tracer=tracer, model=model),
    )
    return paths


@pytest.fixture(name="witness_reference")
def make_witness_reference(witness_fit_paths):
    return al.twins.ReferenceModel.from_fit_directory(
        witness_fit_paths.output_path,
        lens_id="witness_7x7",
        labels=("fixture", "sie_shear_mge"),
        roles={"lens": "main_deflector", "source": "source"},
    )


@pytest.fixture(name="witness_baseline")
def make_witness_baseline(witness_reference):
    return al.twins.Twin.baseline(witness_reference)


@pytest.fixture(name="stellar_dark")
def make_stellar_dark():
    return al.twins.transform.ReplaceTotalMassWithStellarPlusDark(
        galaxy="lens",
        light="bulge",
        scale_radius_kpc=10.0,
        f_dark=0.5,
        aperture_settings=COARSE_APERTURE,
    )


@pytest.fixture(name="witness_twin")
def make_witness_twin(witness_baseline, stellar_dark):
    return witness_baseline.apply(stellar_dark)
