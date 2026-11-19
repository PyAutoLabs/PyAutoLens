import numpy as np

import autogalaxy as ag


def _halo_parameter_extractor_from(profile_cls):
    kaplinghat_profile_classes = tuple(
        cls
        for cls in (
            getattr(ag.mp, "KaplinghatCoredNFWSph", None),
            getattr(ag.mp, "KaplinghatCoredNFWMCRLudlowSph", None),
        )
        if cls is not None
    )
    yang_profile_classes = tuple(
        cls
        for cls in (
            getattr(ag.mp, "YangSIDMSph", None),
            getattr(ag.mp, "YangSIDMMCRLudlowSph", None),
        )
        if cls is not None
    )

    if profile_cls is ag.mp.cNFWSph:

        def extract(prof):
            return [
                prof.centre[0],
                prof.centre[1],
                prof.kappa_s,
                prof.scale_radius,
                prof.core_radius,
            ]

        return 5, extract

    if profile_cls in kaplinghat_profile_classes:

        def extract(prof):
            return [
                prof.centre[0],
                prof.centre[1],
                prof.kappa_s,
                prof.scale_radius,
                prof.interaction_radius,
                prof.central_density,
                prof.isothermal_radius,
            ]

        return 7, extract

    if profile_cls in yang_profile_classes:

        def extract(prof):
            return [
                prof.centre[0],
                prof.centre[1],
                prof.rho_s_evolved,
                prof.scale_radius_evolved,
                prof.core_radius_evolved,
            ]

        return 5, extract

    if profile_cls is ag.mp.NFWTruncatedSph:

        def extract(prof):
            return [
                prof.centre[0],
                prof.centre[1],
                prof.kappa_s,
                prof.scale_radius,
                prof.truncation_radius,
            ]

        return 5, extract

    raise ValueError(f"Unsupported halo profile class: {profile_cls}")


def _padding_params_from(profile_cls, n_params):
    """
    The parameter row put into an empty halo slot: a zero centre followed by the profile's
    ``vmapped_padding_params`` (ones when it has none).

    The mask zeroes an empty slot's deflection, but zero padding would put a zero scale
    radius into the profile, whose infinite partial derivatives turn every ``jax.grad``
    through the plane into NaN (zero cotangent times infinity). A row inside the profile's
    domain keeps the gradients finite. ``vmapped_deflections_from`` swaps in the same row
    itself; padding here as well keeps the gradients finite with an autogalaxy whose hook
    predates that swap.
    """
    radial = getattr(profile_cls, "vmapped_padding_params", None)
    if radial is None:
        radial = np.ones(n_params - 2)
    return np.concatenate([np.zeros(2), np.asarray(radial, dtype=float)])


def precompute_scaling_matrix(plane_redshifts, cosmology=None):
    import jax.numpy as jnp

    cosmology = cosmology or ag.cosmo.Planck15()
    n = len(plane_redshifts)
    z_final = plane_redshifts[-1]
    mat = np.zeros((n, n))

    for i in range(n):
        for j in range(i):
            mat[i, j] = float(
                cosmology.scaling_factor_between_redshifts_from(
                    redshift_0=plane_redshifts[j],
                    redshift_1=plane_redshifts[i],
                    redshift_final=z_final,
                )
            )

    return jnp.array(mat)


def _plane_index_of_halos(galaxies, plane_redshifts, profile_cls):
    """
    Return, for each halo of type ``profile_cls`` that sits on one of ``plane_redshifts``,
    the index of its plane, in the order the halos appear in ``galaxies``.
    """
    z_to_plane = {round(float(z), 8): i for i, z in enumerate(plane_redshifts)}

    plane_indexes = []
    for g in galaxies:
        plane_i = z_to_plane.get(round(float(g.redshift), 8))
        if plane_i is None or hasattr(g, "mass_sheet"):
            continue
        if hasattr(g, "mass") and isinstance(g.mass, profile_cls):
            plane_indexes.append(plane_i)

    return plane_indexes


def halo_counts_per_plane_from(galaxies, plane_redshifts, profile_cls):
    """
    The number of halos of type ``profile_cls`` on each plane of ``plane_redshifts``.

    The largest entry is the smallest ``max_n`` that ``galaxies_to_halo_arrays`` accepts for
    these galaxies.
    """
    counts = np.zeros(len(plane_redshifts), dtype=int)
    for plane_i in _plane_index_of_halos(galaxies, plane_redshifts, profile_cls):
        counts[plane_i] += 1
    return counts


def _check_max_n(counts, plane_redshifts, max_n):
    overflowing = [
        f"plane {i} (z={float(plane_redshifts[i]):.4f}) holds {int(count)} halos"
        for i, count in enumerate(counts)
        if count > max_n
    ]
    if overflowing:
        raise ValueError(
            f"max_n={max_n} is too small: "
            + "; ".join(overflowing)
            + f". Pass max_n={int(max(counts))} or more, or max_n=None to size the arrays "
            "to the fullest plane. Halos beyond max_n would be dropped while the "
            "negative-convergence sheets still remove their mass."
        )


def galaxies_to_halo_arrays(galaxies, plane_redshifts, max_n, profile_cls):
    """
    Pack the halos in ``galaxies`` into padded, fixed-shape arrays for the JAX ray-tracing scan.

    Parameters
    ----------
    galaxies
        The halo galaxies (mass profiles of type ``profile_cls``) and the sheet galaxies or
        fields (with a ``mass_sheet``) of one line of sight.
    plane_redshifts
        The redshift of every plane; galaxies at any other redshift are ignored.
    max_n
        The number of halo slots per plane. A plane holding more halos than this raises a
        ``ValueError`` naming the plane and the ``max_n`` needed, because dropping halos while
        keeping the sheets that compensate for their mass would break mass conservation.
        ``None`` sizes the arrays to the fullest plane.
    profile_cls
        The halo mass profile class, which sets which parameters are packed.

    Returns
    -------
    The halo parameters ``(n_planes, max_n, n_params)``, the slot mask ``(n_planes, max_n)``
    and the sheet convergence of each plane ``(n_planes,)``. Slots the mask leaves empty hold
    the profile's benign padding parameters (see ``_padding_params_from``), not zeros.
    """
    import jax.numpy as jnp

    n_planes = len(plane_redshifts)
    n_params, extract = _halo_parameter_extractor_from(profile_cls=profile_cls)

    counts = halo_counts_per_plane_from(
        galaxies=galaxies, plane_redshifts=plane_redshifts, profile_cls=profile_cls
    )
    if max_n is None:
        max_n = max(int(counts.max(initial=0)), 1)
    else:
        _check_max_n(counts=counts, plane_redshifts=plane_redshifts, max_n=max_n)

    params = np.tile(
        _padding_params_from(profile_cls=profile_cls, n_params=n_params),
        (n_planes, max_n, 1),
    )
    mask = np.zeros((n_planes, max_n), dtype=bool)
    sheet_kappas = np.zeros(n_planes)

    z_to_plane = {}
    for i, z in enumerate(plane_redshifts):
        z_to_plane[round(float(z), 8)] = i

    for g in galaxies:
        z_key = round(float(g.redshift), 8)
        plane_i = z_to_plane.get(z_key)
        if plane_i is None:
            continue

        if hasattr(g, "mass_sheet"):
            sheet_kappas[plane_i] = float(g.mass_sheet.kappa)
        elif hasattr(g, "mass") and isinstance(g.mass, profile_cls):
            slot = int(mask[plane_i].sum())
            params[plane_i, slot] = extract(g.mass)
            mask[plane_i, slot] = True

    return jnp.array(params), jnp.array(mask), jnp.array(sheet_kappas)


def traced_grids_via_scan(
    grid,
    halo_params,
    halo_mask,
    scaling_matrix,
    lens_mass_fn,
    lens_mass_params,
    lens_plane_mask,
    sheet_kappas,
    halo_profile_cls,
):
    import jax
    import jax.numpy as jnp

    n_planes = halo_params.shape[0]
    n_grid = grid.shape[0]

    init_defl_buffer = jnp.zeros((n_planes, n_grid, 2))

    def scan_step(carry, plane_inputs):
        grid_0, defl_buffer, plane_idx = carry
        halo_p, halo_m, scaling_row, is_lens, sheet_kappa = plane_inputs

        scaled = jnp.einsum("p,pmd->md", scaling_row, defl_buffer)
        current_grid = grid_0 - scaled

        halo_defl = halo_profile_cls.vmapped_deflections_from(
            current_grid, halo_p, halo_m
        )

        lens_defl = lens_mass_fn(current_grid, lens_mass_params)
        lens_defl = is_lens * lens_defl

        sheet_defl = sheet_kappa * current_grid

        total_defl = halo_defl + lens_defl + sheet_defl
        defl_buffer = defl_buffer.at[plane_idx].set(total_defl)

        return (grid_0, defl_buffer, plane_idx + 1), current_grid

    plane_stack = (
        halo_params,
        halo_mask,
        scaling_matrix,
        lens_plane_mask,
        sheet_kappas,
    )

    init_carry = (grid, init_defl_buffer, 0)
    _, traced_grids = jax.lax.scan(scan_step, init_carry, plane_stack)

    return traced_grids


def simulate_substructure(
    grid,
    image_shape,
    halo_params,
    halo_mask,
    scaling_matrix,
    lens_mass_fn,
    lens_mass_params,
    lens_plane_mask,
    sheet_kappas,
    source_light_fn,
    source_light_params,
    psf_kernel,
    exposure_time,
    background_sky_level,
    prng_key,
    halo_profile_cls,
    lens_light_fn=None,
    lens_light_params=None,
    lens_plane_idx=None,
):
    import jax
    import jax.numpy as jnp

    traced_grids = traced_grids_via_scan(
        grid=grid,
        halo_params=halo_params,
        halo_mask=halo_mask,
        scaling_matrix=scaling_matrix,
        lens_mass_fn=lens_mass_fn,
        lens_mass_params=lens_mass_params,
        lens_plane_mask=lens_plane_mask,
        sheet_kappas=sheet_kappas,
        halo_profile_cls=halo_profile_cls,
    )

    image_1d = source_light_fn(traced_grids[-1], source_light_params)

    if lens_light_fn is not None:
        lens_image = lens_light_fn(traced_grids[lens_plane_idx], lens_light_params)
        image_1d = image_1d + lens_image

    image_2d = image_1d.reshape(image_shape)

    image_2d = jax.scipy.signal.fftconvolve(image_2d, psf_kernel, mode="same")

    image_2d = image_2d + background_sky_level

    if prng_key is not None:
        image_counts = image_2d * exposure_time
        noisy_counts = jax.random.poisson(prng_key, image_counts)
        image_2d = noisy_counts / exposure_time - background_sky_level
    else:
        image_2d = image_2d - background_sky_level

    return image_2d


def los_realizations_to_arrays(
    realization_galaxies,
    plane_redshifts,
    max_n,
    profile_cls,
):
    """
    Run ``galaxies_to_halo_arrays`` on every line of sight in ``realization_galaxies`` and stack
    the results along a leading batch axis.

    ``max_n=None`` sizes the arrays to the fullest plane across the whole batch, so every
    realisation shares one shape. An integer ``max_n`` that any plane of any realisation
    overflows raises a ``ValueError``.
    """
    import jax.numpy as jnp

    if max_n is None:
        max_n = max(
            [
                int(
                    halo_counts_per_plane_from(
                        galaxies=galaxies,
                        plane_redshifts=plane_redshifts,
                        profile_cls=profile_cls,
                    ).max(initial=0)
                )
                for galaxies in realization_galaxies
            ]
            + [1]
        )

    all_params = []
    all_masks = []
    all_kappas = []

    for realization_i, galaxies in enumerate(realization_galaxies):
        try:
            params, mask, kappas = galaxies_to_halo_arrays(
                galaxies=galaxies,
                plane_redshifts=plane_redshifts,
                max_n=max_n,
                profile_cls=profile_cls,
            )
        except ValueError as error:
            if "max_n=" not in str(error):
                raise
            raise ValueError(f"Realisation {realization_i}: {error}") from error
        all_params.append(params)
        all_masks.append(mask)
        all_kappas.append(kappas)

    return jnp.stack(all_params), jnp.stack(all_masks), jnp.stack(all_kappas)


def batched_simulate_substructure(
    grid,
    image_shape,
    halo_params_batch,
    halo_mask_batch,
    scaling_matrix,
    lens_mass_fn,
    lens_mass_params_batch,
    lens_plane_mask,
    sheet_kappas_batch,
    source_light_fn,
    source_light_params_batch,
    psf_kernel,
    exposure_time,
    background_sky_level,
    prng_keys,
    halo_profile_cls,
    lens_light_fn=None,
    lens_light_params_batch=None,
    lens_plane_idx=None,
):
    import jax
    import functools

    single_fn = functools.partial(
        simulate_substructure,
        grid=grid,
        image_shape=image_shape,
        scaling_matrix=scaling_matrix,
        lens_mass_fn=lens_mass_fn,
        lens_plane_mask=lens_plane_mask,
        source_light_fn=source_light_fn,
        psf_kernel=psf_kernel,
        exposure_time=exposure_time,
        background_sky_level=background_sky_level,
        halo_profile_cls=halo_profile_cls,
        lens_light_fn=lens_light_fn,
        lens_plane_idx=lens_plane_idx,
    )

    def call(hp, hm, sk, lmp, slp, llp, key):
        return single_fn(
            halo_params=hp,
            halo_mask=hm,
            sheet_kappas=sk,
            lens_mass_params=lmp,
            source_light_params=slp,
            lens_light_params=llp,
            prng_key=key,
        )

    return jax.vmap(call)(
        halo_params_batch,
        halo_mask_batch,
        sheet_kappas_batch,
        lens_mass_params_batch,
        source_light_params_batch,
        lens_light_params_batch,
        prng_keys,
    )
