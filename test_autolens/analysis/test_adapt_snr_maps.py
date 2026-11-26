import numpy as np
import pytest

import autofit as af
import autolens as al


def test__exports():
    assert al.galaxy_snr_maps_from is not None
    assert al.soft_snr_maps_from is not None
    assert hasattr(al.AdaptImages, "from_result")
    assert al.image_mesh.HilbertSNR is not None


def _simulated_lens():
    grid = al.Grid2D.uniform(shape_native=(31, 31), pixel_scales=0.1, over_sample_size=1)
    psf = al.Convolver.from_gaussian(shape_native=(3, 3), pixel_scales=0.1, sigma=0.1, normalize=True)

    lens = al.Galaxy(redshift=0.5, mass=al.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=0.8))
    source = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(intensity=0.5, effective_radius=0.2, sersic_index=1.0))
    tracer = al.Tracer(galaxies=[lens, source])

    simulator = al.SimulatorImaging(
        exposure_time=300.0, psf=psf, add_poisson_noise_to_data=False, noise_seed=1
    )
    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)
    dataset = al.Imaging(
        data=dataset.data,
        psf=dataset.psf,
        noise_map=al.Array2D.full(fill_value=0.05, shape_native=(31, 31), pixel_scales=0.1),
        over_sample_size_lp=1,
        over_sample_size_pixelization=1,
    )
    mask = al.Mask2D.circular(shape_native=(31, 31), pixel_scales=0.1, radius=1.3)
    return dataset.apply_mask(mask=mask)


def test__hilbert_snr_soft_maps_and_short_keys__fit_with_adapt_split():
    dataset = _simulated_lens()
    mask = dataset.mask

    # The lens has no light, so the source S/N map is the data over the noise map.
    snr = al.Array2D(values=np.asarray(dataset.data.array) / 0.05, mask=mask)
    full_key = "('galaxies', 'source')"
    snr_maps = {full_key: snr}

    image_mesh = al.image_mesh.HilbertSNR(pixels=60, edge_pixels=12)
    mesh_grid = image_mesh.image_plane_mesh_grid_from(mask=mask, adapt_data=snr)
    soft_maps = al.soft_snr_maps_from(snr_maps)

    assert np.asarray(mesh_grid).shape == (72, 2)

    galaxies = af.ModelInstance()
    galaxies.lens = al.Galaxy(redshift=0.5, mass=al.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=0.8))
    galaxies.source = al.Galaxy(
        redshift=1.0,
        pixelization=al.Pixelization(
            mesh=al.mesh.Delaunay(pixels=image_mesh.pixels, zeroed_pixels=image_mesh.edge_pixels),
            regularization=al.reg.AdaptSplit(
                inner_coefficient=0.01, outer_coefficient=1.0, signal_scale=1.0
            ),
        ),
    )
    instance = af.ModelInstance()
    instance.galaxies = galaxies

    log_likelihoods = []
    for key in (full_key, "source"):
        adapt_images = al.AdaptImages(
            galaxy_name_image_dict={key: soft_maps[full_key]},
            galaxy_name_image_plane_mesh_grid_dict={key: mesh_grid},
        )
        analysis = al.AnalysisImaging(dataset=dataset, adapt_images=adapt_images, use_jax=False)
        log_likelihoods.append(analysis.log_likelihood_function(instance=instance))

    assert np.isfinite(log_likelihoods[0])
    assert log_likelihoods[1] == pytest.approx(log_likelihoods[0], 1.0e-10)
