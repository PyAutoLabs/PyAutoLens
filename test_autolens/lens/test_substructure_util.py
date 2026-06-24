import numpy as np
import pytest

import autolens as al
from autolens.lens import substructure_util


requires_kaplinghat = pytest.mark.skipif(
    not hasattr(al.mp, "KaplinghatCoredNFWSph"),
    reason="Kaplinghat SIDM profiles are provided by the pending PyAutoGalaxy release.",
)


@requires_kaplinghat
def test__autolens_exposes_kaplinghat_profiles_from_autogalaxy():
    assert hasattr(al.mp, "KaplinghatCoredNFWSph")
    assert hasattr(al.mp, "KaplinghatCoredNFWMCRLudlowSph")


@requires_kaplinghat
def test__galaxies_to_halo_arrays__packs_kaplinghat_profile_parameters():
    profile = al.mp.KaplinghatCoredNFWSph(
        centre=(0.1, -0.2),
        kappa_s=0.03,
        scale_radius=1.7,
        sigma_over_m=2.0,
        t_age=8.0,
        interaction_radius=0.25,
    )
    galaxies = [
        al.Galaxy(redshift=0.5, mass=profile),
        al.Galaxy(redshift=1.0, mass_sheet=al.mp.MassSheet(kappa=-0.01)),
    ]

    params, mask, sheet_kappas = substructure_util.galaxies_to_halo_arrays(
        galaxies=galaxies,
        plane_redshifts=[0.5, 1.0],
        max_n=2,
        profile_cls=al.mp.KaplinghatCoredNFWSph,
    )

    assert params.shape == (2, 2, 7)
    assert mask.tolist() == [[True, False], [False, False]]
    assert sheet_kappas.tolist() == [0.0, -0.01]

    np.testing.assert_allclose(
        np.asarray(params[0, 0]),
        np.array(
            [
                0.1,
                -0.2,
                profile.kappa_s,
                profile.scale_radius,
                profile.interaction_radius,
                profile.central_density,
                profile.isothermal_radius,
            ]
        ),
    )


def test__galaxies_to_halo_arrays__raises_for_unsupported_profile_class():
    with pytest.raises(ValueError):
        substructure_util.galaxies_to_halo_arrays(
            galaxies=[],
            plane_redshifts=[0.5],
            max_n=1,
            profile_cls=al.mp.IsothermalSph,
        )
