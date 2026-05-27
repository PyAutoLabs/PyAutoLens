import numpy as np

import autogalaxy as ag


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


def galaxies_to_halo_arrays(galaxies, plane_redshifts, max_n, profile_cls):
    import jax.numpy as jnp

    n_planes = len(plane_redshifts)

    if profile_cls is ag.mp.cNFWSph:
        n_params = 5
        def extract(prof):
            return [
                prof.centre[0], prof.centre[1],
                prof.kappa_s, prof.scale_radius, prof.core_radius,
            ]
    else:
        n_params = 5
        def extract(prof):
            return [
                prof.centre[0], prof.centre[1],
                prof.kappa_s, prof.scale_radius, prof.truncation_radius,
            ]

    params = np.zeros((n_planes, max_n, n_params))
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
            if slot < max_n:
                params[plane_i, slot] = extract(g.mass)
                mask[plane_i, slot] = True

    return jnp.array(params), jnp.array(mask), jnp.array(sheet_kappas)


def traced_grids_via_scan(
    grid,
    halo_params,
    halo_mask,
    scaling_matrix,
    macro_deflections_fn,
    macro_plane_mask,
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
        halo_p, halo_m, scaling_row, is_macro, sheet_kappa = plane_inputs

        scaled = jnp.einsum("p,pmd->md", scaling_row, defl_buffer)
        current_grid = grid_0 - scaled

        halo_defl = halo_profile_cls.vmapped_deflections_from(
            current_grid, halo_p, halo_m
        )

        macro_defl = macro_deflections_fn(current_grid)
        macro_defl = is_macro * macro_defl

        sheet_defl = sheet_kappa * current_grid

        total_defl = halo_defl + macro_defl + sheet_defl
        defl_buffer = defl_buffer.at[plane_idx].set(total_defl)

        return (grid_0, defl_buffer, plane_idx + 1), current_grid

    plane_stack = (
        halo_params,
        halo_mask,
        scaling_matrix,
        macro_plane_mask,
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
    macro_deflections_fn,
    macro_plane_mask,
    sheet_kappas,
    source_image_fn,
    psf_kernel,
    exposure_time,
    background_sky_level,
    prng_key,
    halo_profile_cls,
):
    import jax
    import jax.numpy as jnp

    traced_grids = traced_grids_via_scan(
        grid=grid,
        halo_params=halo_params,
        halo_mask=halo_mask,
        scaling_matrix=scaling_matrix,
        macro_deflections_fn=macro_deflections_fn,
        macro_plane_mask=macro_plane_mask,
        sheet_kappas=sheet_kappas,
        halo_profile_cls=halo_profile_cls,
    )

    source_grid = traced_grids[-1]
    image_1d = source_image_fn(source_grid)
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
    import jax.numpy as jnp

    all_params = []
    all_masks = []
    all_kappas = []

    for galaxies in realization_galaxies:
        params, mask, kappas = galaxies_to_halo_arrays(
            galaxies=galaxies,
            plane_redshifts=plane_redshifts,
            max_n=max_n,
            profile_cls=profile_cls,
        )
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
    macro_deflections_fn,
    macro_plane_mask,
    sheet_kappas_batch,
    source_image_fn,
    psf_kernel,
    exposure_time,
    background_sky_level,
    prng_keys,
    halo_profile_cls,
):
    import jax
    import functools

    single_fn = functools.partial(
        simulate_substructure,
        grid=grid,
        image_shape=image_shape,
        scaling_matrix=scaling_matrix,
        macro_deflections_fn=macro_deflections_fn,
        macro_plane_mask=macro_plane_mask,
        source_image_fn=source_image_fn,
        psf_kernel=psf_kernel,
        exposure_time=exposure_time,
        background_sky_level=background_sky_level,
        halo_profile_cls=halo_profile_cls,
    )

    def call(halo_params, halo_mask, sheet_kappas, prng_key):
        return single_fn(
            halo_params=halo_params,
            halo_mask=halo_mask,
            sheet_kappas=sheet_kappas,
            prng_key=prng_key,
        )

    return jax.vmap(call)(
        halo_params_batch, halo_mask_batch, sheet_kappas_batch, prng_keys,
    )
