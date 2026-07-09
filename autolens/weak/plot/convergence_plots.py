"""
Kaiser-Squires convergence-map reconstruction from a shear catalogue.

Kaiser & Squires (1993) showed the convergence :math:`\\kappa` and shear
:math:`\\gamma` are related algebraically in Fourier space:

.. math::
    \\hat{\\kappa}(\\mathbf{k}) = D^*(\\mathbf{k}) \\, \\hat{\\gamma}(\\mathbf{k}),
    \\qquad
    D(\\mathbf{k}) = \\frac{k_x^2 - k_y^2 + 2 i k_x k_y}{k_x^2 + k_y^2}

so a mass map can be reconstructed directly from the measured shear field with two
FFTs — no mass model assumed. This is the classic "dark matter map" technique used
for merging clusters (e.g. the Bullet cluster) and survey mass maps.

Because the catalogue's galaxy positions are irregular, the shears are first binned
onto a regular grid (plain per-cell mean, empty cells zero) over the catalogue's
extent, then optionally smoothed with a Gaussian kernel — standard practice, since
the inversion is only defined on a grid and raw per-cell shears are shape-noise
dominated. The imaginary part of the reconstruction (the B-mode map) carries no
lensing signal and is returned alongside the E-mode map as a systematics check.

The reconstruction inherits Kaiser-Squires' well-known caveats: the
:math:`\\mathbf{k} = 0` mode is unconstrained (the mass-sheet degeneracy — maps are
zero-mean by construction) and FFT periodicity causes edge artefacts on small
fields. For quantitative masses, fit a mass model (``scripts/weak/modeling.py``);
the map is a visualization and model-independent cross-check.
"""
from typing import Optional, Tuple

import numpy as np

import autoarray as aa

from autoarray.plot.utils import save_figure, subplots

from autolens.weak.plot.weak_dataset_plots import _positions_yx
from autolens.weak.plot.shear_profile_plots import _gamma_1_2_from


def convergence_via_kaiser_squires_from(
    shear_yx,
    shape_native: Tuple[int, int] = (50, 50),
    smoothing_sigma_pixels: float = 1.0,
    extent: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[aa.Array2D, aa.Array2D]:
    """
    Reconstruct E-mode (convergence) and B-mode maps from a shear field.

    Parameters
    ----------
    shear_yx
        The shear field (e.g. ``dataset.shear_yx``).
    shape_native
        The ``(rows, cols)`` shape of the regular grid the shears are binned onto.
    smoothing_sigma_pixels
        The sigma (in pixels) of the Gaussian kernel applied to the binned shear
        maps before inversion; ``0.0`` disables smoothing.
    extent
        The ``(x_min, x_max, y_min, y_max)`` field extent; defaults to the
        catalogue's bounding box (with a small buffer, matching
        ``WeakDataset.extent_from``).

    Returns
    -------
    An ``(e_mode, b_mode)`` pair of ``aa.Array2D`` maps: the convergence
    reconstruction and its B-mode systematics check.
    """
    positions = _positions_yx(shear_yx)
    gamma_1, gamma_2 = _gamma_1_2_from(shear_yx)

    if extent is None:
        buffer = 0.1
        x_min, x_max = positions[:, 1].min() - buffer, positions[:, 1].max() + buffer
        y_min, y_max = positions[:, 0].min() - buffer, positions[:, 0].max() + buffer
    else:
        x_min, x_max, y_min, y_max = extent

    rows, cols = shape_native

    # Bin the irregular shears onto the grid: plain mean per cell, zero where empty.
    y_edges = np.linspace(y_min, y_max, rows + 1)
    x_edges = np.linspace(x_min, x_max, cols + 1)

    counts, _, _ = np.histogram2d(
        positions[:, 0], positions[:, 1], bins=[y_edges, x_edges]
    )
    sum_1, _, _ = np.histogram2d(
        positions[:, 0], positions[:, 1], bins=[y_edges, x_edges], weights=gamma_1
    )
    sum_2, _, _ = np.histogram2d(
        positions[:, 0], positions[:, 1], bins=[y_edges, x_edges], weights=gamma_2
    )

    with np.errstate(invalid="ignore"):
        grid_1 = np.where(counts > 0, sum_1 / np.maximum(counts, 1), 0.0)
        grid_2 = np.where(counts > 0, sum_2 / np.maximum(counts, 1), 0.0)

    if smoothing_sigma_pixels > 0.0:
        kernel = _gaussian_kernel_2d(sigma=smoothing_sigma_pixels)
        grid_1 = _convolve_2d_same(grid_1, kernel)
        grid_2 = _convolve_2d_same(grid_2, kernel)

    # Kaiser-Squires inversion: kappa_hat = D*(k) gamma_hat, D = (kx^2 - ky^2 + 2i kx ky) / k^2.
    ky = np.fft.fftfreq(rows)[:, None]
    kx = np.fft.fftfreq(cols)[None, :]
    k_squared = kx**2.0 + ky**2.0
    k_squared[0, 0] = 1.0  # k = 0 mode is unconstrained (mass-sheet degeneracy); set below.

    d_conj = ((kx**2.0 - ky**2.0) - 2.0j * kx * ky) / k_squared

    gamma_hat = np.fft.fft2(grid_1) + 1.0j * np.fft.fft2(grid_2)
    kappa_hat = d_conj * gamma_hat
    kappa_hat[0, 0] = 0.0

    kappa = np.fft.ifft2(kappa_hat)

    pixel_scales = ((y_max - y_min) / rows, (x_max - x_min) / cols)

    # Row 0 of the binned grids is y_min; autoarray native arrays put y_max at row 0.
    e_mode = aa.Array2D.no_mask(
        values=np.flipud(kappa.real), pixel_scales=pixel_scales
    )
    b_mode = aa.Array2D.no_mask(
        values=np.flipud(kappa.imag), pixel_scales=pixel_scales
    )

    return e_mode, b_mode


def _gaussian_kernel_2d(sigma: float) -> np.ndarray:
    """A normalised 2D Gaussian kernel truncated at 3 sigma."""
    half_width = max(int(np.ceil(3.0 * sigma)), 1)
    x = np.arange(-half_width, half_width + 1)
    kernel_1d = np.exp(-0.5 * (x / sigma) ** 2.0)
    kernel = np.outer(kernel_1d, kernel_1d)
    return kernel / kernel.sum()


def _convolve_2d_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Same-size 2D convolution via zero-padded FFT (avoids a scipy dependency)."""
    kh, kw = kernel.shape
    ih, iw = image.shape
    padded = np.zeros((ih + kh - 1, iw + kw - 1))
    padded[:ih, :iw] = image
    kernel_padded = np.zeros_like(padded)
    kernel_padded[:kh, :kw] = kernel
    result = np.fft.ifft2(np.fft.fft2(padded) * np.fft.fft2(kernel_padded)).real
    h0, w0 = (kh - 1) // 2, (kw - 1) // 2
    return result[h0 : h0 + ih, w0 : w0 + iw]


def plot_convergence_map(
    shear_yx,
    shape_native: Tuple[int, int] = (50, 50),
    smoothing_sigma_pixels: float = 1.0,
    show_positions: bool = True,
    ax=None,
    title: str = "Kaiser-Squires Convergence",
    output_path: Optional[str] = None,
    output_filename: str = "convergence_map",
    output_format: Optional[str] = None,
):
    """
    Plot the Kaiser-Squires E-mode convergence reconstruction of a shear field.

    Parameters
    ----------
    shear_yx
        The shear field (e.g. ``dataset.shear_yx``).
    shape_native
        The ``(rows, cols)`` reconstruction grid shape.
    smoothing_sigma_pixels
        Gaussian smoothing applied to the binned shears before inversion.
    show_positions
        Whether to overlay the catalogue's galaxy positions on the map.
    ax
        An existing matplotlib axes to draw on; a new figure is created if ``None``.
    """
    e_mode, _ = convergence_via_kaiser_squires_from(
        shear_yx=shear_yx,
        shape_native=shape_native,
        smoothing_sigma_pixels=smoothing_sigma_pixels,
    )

    positions = _positions_yx(shear_yx)
    extent = [
        positions[:, 1].min() - 0.1,
        positions[:, 1].max() + 0.1,
        positions[:, 0].min() - 0.1,
        positions[:, 0].max() + 0.1,
    ]

    fig = None
    if ax is None:
        fig, ax = subplots(1, 1)

    image = ax.imshow(
        e_mode.native,
        origin="upper",
        extent=extent,
        cmap="magma",
    )
    if show_positions:
        ax.scatter(
            positions[:, 1], positions[:, 0], s=2, c="cyan", alpha=0.5, linewidths=0
        )
    ax.set_xlabel("x (arcsec)")
    ax.set_ylabel("y (arcsec)")
    ax.set_title(title)
    if fig is not None:
        fig.colorbar(image, ax=ax, label=r"$\kappa$ (E-mode)")
        save_figure(
            fig,
            path=output_path,
            filename=output_filename,
            format=output_format,
        )
