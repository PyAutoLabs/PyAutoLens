import autofit as af
import autolens as al

from autogalaxy.analysis.model_util import mge_model_from, mge_model_V2_from
from autogalaxy.analysis.model_util import mge_point_model_from
from autogalaxy.analysis.model_util import hilbert_pixels_from_pixel_scale

def simulator_start_here_model_from(
    include_lens_light: bool = True, use_point_source: bool = False
):

    if include_lens_light:
        bulge = af.Model(al.lp_snr.Sersic)

        bulge.centre = (0.0, 0.0)
        bulge.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
            mean=0.0, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
        )
        bulge.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
            mean=0.0, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
        )
        bulge.signal_to_noise_ratio = af.UniformPrior(
            lower_limit=20.0, upper_limit=60.0
        )
        bulge.effective_radius = af.UniformPrior(lower_limit=1.0, upper_limit=5.0)
        bulge.sersic_index = af.TruncatedGaussianPrior(
            mean=4.0, sigma=0.5, lower_limit=0.8, upper_limit=5.0
        )
    else:
        bulge = None

    mass = af.Model(al.mp.Isothermal)

    mass.centre = (0.0, 0.0)
    mass.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
    )
    mass.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
    )
    mass.einstein_radius = af.UniformPrior(lower_limit=0.2, upper_limit=1.8)

    shear = af.Model(al.mp.ExternalShear)

    shear.gamma_1 = af.GaussianPrior(mean=0.0, sigma=0.05)
    shear.gamma_2 = af.GaussianPrior(mean=0.0, sigma=0.05)

    lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass, shear=shear)

    if use_point_source:

        point_0 = af.Model(al.ps.PointFlux)

        point_0.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
        point_0.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
        point_0.flux = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)

        source = af.Model(al.Galaxy, redshift=1.0, point_0=point_0)

    else:

        bulge = af.Model(al.lp_snr.Sersic)

        bulge.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
        bulge.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
        bulge.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
            mean=0.0, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
        )
        bulge.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
            mean=0.0, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
        )
        bulge.signal_to_noise_ratio = af.UniformPrior(
            lower_limit=10.0, upper_limit=30.0
        )
        bulge.effective_radius = af.UniformPrior(lower_limit=0.01, upper_limit=3.0)
        bulge.sersic_index = af.TruncatedGaussianPrior(
            mean=2.0, sigma=0.5, lower_limit=0.8, upper_limit=5.0
        )

        source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

    return lens, source
