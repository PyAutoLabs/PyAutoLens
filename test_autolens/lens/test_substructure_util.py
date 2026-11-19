import importlib.util

import numpy as np
import pytest

import autolens as al
from autolens.lens import substructure_util


requires_kaplinghat = pytest.mark.skipif(
    not hasattr(al.mp, "KaplinghatCoredNFWSph"),
    reason="Kaplinghat SIDM profiles are provided by the pending PyAutoGalaxy release.",
)

# `galaxies_to_halo_arrays` is jax-backed; these tests need jax installed to run
# (it ships via the `[optional]` extras). The NumPy-only Python-version matrix
# has no jax, so skip there rather than fail.
requires_jax = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="requires jax (installed via the [optional] extras; absent on the NumPy-only matrix env)",
)


@requires_kaplinghat
def test__autolens_exposes_kaplinghat_profiles_from_autogalaxy():
    assert hasattr(al.mp, "KaplinghatCoredNFWSph")
    assert hasattr(al.mp, "KaplinghatCoredNFWMCRLudlowSph")


@requires_jax
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


@requires_jax
def test__galaxies_to_halo_arrays__raises_for_unsupported_profile_class():
    with pytest.raises(ValueError):
        substructure_util.galaxies_to_halo_arrays(
            galaxies=[],
            plane_redshifts=[0.5],
            max_n=1,
            profile_cls=al.mp.IsothermalSph,
        )


requires_yang = pytest.mark.skipif(
    not hasattr(al.mp, "YangSIDMSph"),
    reason="Yang SIDM profiles are provided by the pending PyAutoGalaxy release.",
)


@requires_yang
def test__autolens_exposes_yang_profiles_from_autogalaxy():
    assert hasattr(al.mp, "YangSIDMSph")
    assert hasattr(al.mp, "YangSIDMMCRLudlowSph")


@requires_jax
@requires_yang
def test__galaxies_to_halo_arrays__packs_yang_profile_parameters():
    profile = al.mp.YangSIDMSph(
        centre=(0.1, -0.2),
        kappa_s=0.03,
        scale_radius=1.7,
        tau=0.5,
    )
    galaxies = [
        al.Galaxy(redshift=0.5, mass=profile),
        al.Galaxy(redshift=1.0, mass_sheet=al.mp.MassSheet(kappa=-0.01)),
    ]

    params, mask, sheet_kappas = substructure_util.galaxies_to_halo_arrays(
        galaxies=galaxies,
        plane_redshifts=[0.5, 1.0],
        max_n=2,
        profile_cls=al.mp.YangSIDMSph,
    )

    assert params.shape == (2, 2, 5)
    assert mask.tolist() == [[True, False], [False, False]]
    assert sheet_kappas.tolist() == [0.0, -0.01]

    np.testing.assert_allclose(
        np.asarray(params[0, 0]),
        np.array(
            [
                0.1,
                -0.2,
                profile.rho_s_evolved,
                profile.scale_radius_evolved,
                profile.core_radius_evolved,
            ]
        ),
    )


def _nfw_truncated_halos_on_plane(redshift, total):
    return [
        al.Galaxy(
            redshift=redshift,
            mass=al.mp.NFWTruncatedSph(
                centre=(0.01 * i, 0.0),
                kappa_s=0.01,
                scale_radius=1.0,
                truncation_radius=2.0,
            ),
        )
        for i in range(total)
    ]


@requires_jax
def test__galaxies_to_halo_arrays__raises_when_a_plane_overflows_max_n():
    galaxies = (
        _nfw_truncated_halos_on_plane(redshift=0.3, total=2)
        + _nfw_truncated_halos_on_plane(redshift=0.7, total=5)
        + [al.Galaxy(redshift=0.7, mass_sheet=al.mp.MassSheet(kappa=-0.01))]
    )

    with pytest.raises(ValueError) as error:
        substructure_util.galaxies_to_halo_arrays(
            galaxies=galaxies,
            plane_redshifts=[0.3, 0.7],
            max_n=3,
            profile_cls=al.mp.NFWTruncatedSph,
        )

    message = str(error.value)
    assert "plane 1 (z=0.7000) holds 5 halos" in message
    assert "plane 0" not in message
    assert "max_n=5" in message


@requires_jax
def test__galaxies_to_halo_arrays__max_n_none_sizes_to_fullest_plane_and_keeps_every_halo():
    galaxies = (
        _nfw_truncated_halos_on_plane(redshift=0.3, total=2)
        + _nfw_truncated_halos_on_plane(redshift=0.7, total=5)
        + [al.Galaxy(redshift=0.7, mass_sheet=al.mp.MassSheet(kappa=-0.01))]
    )

    params, mask, sheet_kappas = substructure_util.galaxies_to_halo_arrays(
        galaxies=galaxies,
        plane_redshifts=[0.3, 0.7],
        max_n=None,
        profile_cls=al.mp.NFWTruncatedSph,
    )

    assert params.shape[:2] == (2, 5)
    assert mask.sum(axis=1).tolist() == [2, 5]
    assert int(mask.sum()) == 7
    np.testing.assert_allclose(
        np.asarray(params[1, :, 0]), [0.0, 0.01, 0.02, 0.03, 0.04]
    )
    assert sheet_kappas.tolist() == [0.0, -0.01]


@requires_jax
def test__galaxies_to_halo_arrays__exact_max_n_is_accepted():
    params, mask, _ = substructure_util.galaxies_to_halo_arrays(
        galaxies=_nfw_truncated_halos_on_plane(redshift=0.7, total=3),
        plane_redshifts=[0.7],
        max_n=3,
        profile_cls=al.mp.NFWTruncatedSph,
    )

    assert mask.tolist() == [[True, True, True]]


@requires_jax
def test__los_realizations_to_arrays__max_n_none_sizes_to_fullest_plane_across_batch():
    realization_galaxies = [
        _nfw_truncated_halos_on_plane(redshift=0.7, total=2),
        _nfw_truncated_halos_on_plane(redshift=0.3, total=4),
    ]

    params, mask, _ = substructure_util.los_realizations_to_arrays(
        realization_galaxies=realization_galaxies,
        plane_redshifts=[0.3, 0.7],
        max_n=None,
        profile_cls=al.mp.NFWTruncatedSph,
    )

    assert params.shape[:3] == (2, 2, 4)
    assert mask.sum(axis=(1, 2)).tolist() == [2, 4]


@requires_jax
def test__los_realizations_to_arrays__overflow_names_the_realisation():
    realization_galaxies = [
        _nfw_truncated_halos_on_plane(redshift=0.7, total=2),
        _nfw_truncated_halos_on_plane(redshift=0.3, total=4),
    ]

    with pytest.raises(ValueError, match=r"Realisation 1: .*plane 0 \(z=0.3000\) holds 4 halos"):
        substructure_util.los_realizations_to_arrays(
            realization_galaxies=realization_galaxies,
            plane_redshifts=[0.3, 0.7],
            max_n=3,
            profile_cls=al.mp.NFWTruncatedSph,
        )
