"""
Tangential / cross shear radial profiles for a ``WeakDataset`` or ``FitWeak``.

The azimuthally averaged tangential shear profile :math:`\\gamma_t(r)` is the standard
observable of cluster weak lensing (e.g. Oguri et al. 2012's SGAS analysis; Medezinski
et al. 2016's A2744 analysis): a foreground mass distribution stretches background
galaxies tangentially around it, so binning the tangential component of the shear in
radius traces the projected mass profile. The cross component :math:`\\gamma_x` (the
45-degree rotated component) is not produced by gravitational lensing at leading order,
so its binned profile doubles as the standard B-mode null test — a systematic that
leaks power into :math:`\\gamma_x` would contaminate :math:`\\gamma_t` too.

Component conventions follow the rest of ``autolens/weak``: shears are accessed
exclusively through the public ``.ellipticities`` (:math:`|\\gamma|`) and ``.phis``
(position angle in **degrees**) accessors, from which
:math:`\\gamma_1 = |\\gamma| \\cos 2\\phi`, :math:`\\gamma_2 = |\\gamma| \\sin 2\\phi`
are reconstructed — the raw ``[gamma_2, gamma_1]`` storage is never indexed directly.

With :math:`\\gamma = \\gamma_1 + i\\gamma_2` and :math:`\\varphi` the position angle of
a galaxy about the profile centre:

.. math::
    \\gamma_t = -\\mathrm{Re}[\\gamma e^{-2i\\varphi}], \\qquad
    \\gamma_x = -\\mathrm{Im}[\\gamma e^{-2i\\varphi}]

so a tangentially aligned shear field has :math:`\\gamma_t > 0`.
"""
from typing import Optional, Tuple

import numpy as np

from autoarray.plot.utils import save_figure, subplots

from autolens.weak.plot.weak_dataset_plots import _positions_yx


def _gamma_1_2_from(shear_yx) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct ``(gamma_1, gamma_2)`` from the public accessors."""
    magnitudes = np.asarray(shear_yx.ellipticities)
    phis_rad = np.deg2rad(np.asarray(shear_yx.phis))
    return magnitudes * np.cos(2.0 * phis_rad), magnitudes * np.sin(2.0 * phis_rad)


def shear_tangential_cross_from(
    shear_yx,
    centre: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The per-galaxy tangential and cross shear components about ``centre``.

    Parameters
    ----------
    shear_yx
        The shear field (e.g. ``dataset.shear_yx`` or ``fit.model_shear``).
    centre
        The ``(y, x)`` centre about which tangential / cross components are defined,
        e.g. the lens-galaxy centre.

    Returns
    -------
    A ``(gamma_t, gamma_x, radii)`` tuple of ``(N,)`` arrays, where ``radii`` are the
    galaxy distances from ``centre``.
    """
    positions = _positions_yx(shear_yx)
    dy = positions[:, 0] - centre[0]
    dx = positions[:, 1] - centre[1]

    radii = np.sqrt(dy**2.0 + dx**2.0)
    varphi = np.arctan2(dy, dx)

    gamma_1, gamma_2 = _gamma_1_2_from(shear_yx)

    cos_2varphi = np.cos(2.0 * varphi)
    sin_2varphi = np.sin(2.0 * varphi)

    gamma_t = -(gamma_1 * cos_2varphi + gamma_2 * sin_2varphi)
    gamma_x = -(gamma_2 * cos_2varphi - gamma_1 * sin_2varphi)

    return gamma_t, gamma_x, radii


def shear_profile_from(
    shear_yx,
    centre: Tuple[float, float] = (0.0, 0.0),
    bins: int = 10,
    noise_map=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Azimuthally averaged tangential / cross shear profiles in radial bins.

    Bin edges are linearly spaced between the innermost and outermost galaxy radius.
    Each bin's value is the plain mean of its galaxies' components and its error the
    standard error on that mean — when ``noise_map`` is given, the per-galaxy
    measurement noise replaces the empirical scatter in bins with fewer than two
    galaxies (where a standard error is undefined).

    Returns
    -------
    A ``(bin_radii, gamma_t, gamma_t_err, gamma_x, gamma_x_err)`` tuple of ``(bins,)``
    arrays; bins containing no galaxies hold ``np.nan``.
    """
    gamma_t, gamma_x, radii = shear_tangential_cross_from(
        shear_yx=shear_yx, centre=centre
    )

    edges = np.linspace(radii.min(), radii.max(), bins + 1)
    bin_radii = 0.5 * (edges[:-1] + edges[1:])

    profile_t = np.full(bins, np.nan)
    error_t = np.full(bins, np.nan)
    profile_x = np.full(bins, np.nan)
    error_x = np.full(bins, np.nan)

    noise = np.asarray(noise_map) if noise_map is not None else None

    for i in range(bins):
        # Right-inclusive final bin so the outermost galaxy is not dropped.
        if i == bins - 1:
            in_bin = (radii >= edges[i]) & (radii <= edges[i + 1])
        else:
            in_bin = (radii >= edges[i]) & (radii < edges[i + 1])
        n = in_bin.sum()
        if n == 0:
            continue
        profile_t[i] = gamma_t[in_bin].mean()
        profile_x[i] = gamma_x[in_bin].mean()
        if n > 1:
            error_t[i] = gamma_t[in_bin].std(ddof=1) / np.sqrt(n)
            error_x[i] = gamma_x[in_bin].std(ddof=1) / np.sqrt(n)
        elif noise is not None:
            error_t[i] = error_x[i] = noise[in_bin][0]

    return bin_radii, profile_t, error_t, profile_x, error_x


def plot_shear_profile(
    obj,
    centre: Tuple[float, float] = (0.0, 0.0),
    bins: int = 10,
    ax=None,
    title: str = "Shear Profile",
    output_path: Optional[str] = None,
    output_filename: str = "shear_profile",
    output_format: Optional[str] = None,
):
    """
    Plot binned tangential and cross shear profiles of a ``WeakDataset`` or ``FitWeak``.

    The tangential profile is drawn with error bars; the cross profile (offset points)
    should scatter around zero — the B-mode null test. When ``obj`` is a ``FitWeak``
    the model shear's tangential profile is overlaid as a line, so data and model can
    be compared in the space where cluster weak-lensing results are usually shown.

    Parameters
    ----------
    obj
        A ``WeakDataset`` or ``FitWeak``.
    centre
        The ``(y, x)`` centre about which the profiles are computed.
    bins
        The number of linearly spaced radial bins.
    ax
        An existing matplotlib axes to draw on; a new figure is created if ``None``.
    """
    dataset = obj.dataset if hasattr(obj, "model_shear") else obj

    bin_radii, gamma_t, gamma_t_err, gamma_x, gamma_x_err = shear_profile_from(
        shear_yx=dataset.shear_yx,
        centre=centre,
        bins=bins,
        noise_map=dataset.noise_map,
    )

    fig = None
    if ax is None:
        fig, ax = subplots(1, 1)

    ax.errorbar(
        bin_radii,
        gamma_t,
        yerr=gamma_t_err,
        fmt="o",
        color="k",
        capsize=3,
        label=r"$\gamma_t$ (data)",
    )
    ax.errorbar(
        bin_radii,
        gamma_x,
        yerr=gamma_x_err,
        fmt="s",
        color="tab:blue",
        alpha=0.6,
        capsize=3,
        label=r"$\gamma_\times$ (B-mode null test)",
    )

    if hasattr(obj, "model_shear"):
        _, model_t, _, _, _ = shear_profile_from(
            shear_yx=obj.model_shear, centre=centre, bins=bins
        )
        ax.plot(bin_radii, model_t, color="r", alpha=0.8, label=r"$\gamma_t$ (model)")

    ax.axhline(0.0, color="0.6", lw=0.8, zorder=0)
    ax.set_xlabel("Radius from centre (arcsec)")
    ax.set_ylabel("Shear")
    ax.set_title(title)
    ax.legend()

    if fig is not None:
        save_figure(
            fig,
            path=output_path,
            filename=output_filename,
            format=output_format,
        )
