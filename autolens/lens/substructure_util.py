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
