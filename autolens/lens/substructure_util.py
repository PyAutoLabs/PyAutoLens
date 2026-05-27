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
