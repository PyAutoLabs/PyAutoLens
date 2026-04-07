"""Standalone subplot functions for subhalo detection visualisation."""
from typing import Optional

from autogalaxy.util.plot_utils import plot_array
from autoarray.plot.utils import subplots, save_figure, tight_layout
from autolens.imaging.plot.fit_imaging_plots import _plot_source_plane


def subplot_detection_imaging(
    result,
    fit_imaging_with_subhalo,
    output_path: Optional[str] = None,
    output_format: str = None,
    colormap: Optional[str] = None,
    use_log10: bool = False,
    use_log_evidences: bool = True,
    relative_to_value: float = 0.0,
    remove_zeros: bool = False,
):
    """
    Produce a 4-panel subplot summarising subhalo detection from imaging data.

    This function is the primary summary diagnostic for a subhalo
    detection analysis run on imaging data.  The four panels are:

    1. Imaging data (from the fit that includes the subhalo).
    2. Signal-to-noise map of that fit.
    3. Figure-of-merit (log-evidence or log-likelihood increase) grid,
       indicating where a subhalo improves the fit.
    4. Best-fit subhalo mass at each grid position.

    Parameters
    ----------
    result : SubhaloResult
        The subhalo detection result object exposing
        ``figure_of_merit_array`` and ``subhalo_mass_array``.
    fit_imaging_with_subhalo : FitImaging
        The best-fit imaging fit that includes the subhalo, used for the
        data and S/N panels.
    output_path : str, optional
        Directory in which to save the figure.  If ``None`` the figure is
        not saved to disk.
    output_format : str, optional
        Image format passed to :func:`~autoarray.plot.utils.save_figure`.
    colormap : str, optional
        Matplotlib colormap name.
    use_log10 : bool, optional
        If ``True`` a log10 stretch is applied to the data and S/N panels.
    use_log_evidences : bool, optional
        If ``True`` (default) log-evidence increases are shown in the
        figure-of-merit panel; otherwise log-likelihood increases are used.
    relative_to_value : float, optional
        Value subtracted from each figure-of-merit entry before plotting.
        Defaults to ``0.0`` (no subtraction).
    remove_zeros : bool, optional
        If ``True`` grid positions where the figure of merit is exactly
        zero are masked out before plotting.
    """
    fig, axes = subplots(1, 4, figsize=(28, 7))

    plot_array(
        array=fit_imaging_with_subhalo.data,
        ax=axes[0],
        title="Data",
        colormap=colormap,
        use_log10=use_log10,
    )
    plot_array(
        array=fit_imaging_with_subhalo.signal_to_noise_map,
        ax=axes[1],
        title="Signal-To-Noise Map",
        colormap=colormap,
        use_log10=use_log10,
    )

    fom_array = result.figure_of_merit_array(
        use_log_evidences=use_log_evidences,
        relative_to_value=relative_to_value,
        remove_zeros=remove_zeros,
    )
    plot_array(
        array=fom_array,
        ax=axes[2],
        title="Increase in Log Evidence",
        colormap=colormap,
    )

    mass_array = result.subhalo_mass_array
    plot_array(
        array=mass_array,
        ax=axes[3],
        title="Subhalo Mass",
        colormap=colormap,
    )

    tight_layout()
    save_figure(fig, path=output_path, filename="detection_imaging", format=output_format)


def subplot_detection_fits(
    fit_imaging_no_subhalo,
    fit_imaging_with_subhalo,
    output_path: Optional[str] = None,
    output_format: str = None,
    colormap: Optional[str] = None,
):
    """
    Produce a 6-panel subplot comparing imaging fits with and without a subhalo.

    Displays residual maps and source-plane images in a 2 × 3 grid,
    with the top row corresponding to the no-subhalo baseline and the
    bottom row to the fit that includes the subhalo:

    * Top row (no subhalo):

      1. Normalised residual map.
      2. Chi-squared map.
      3. Source-plane image (plane 1).

    * Bottom row (with subhalo):

      4. Normalised residual map.
      5. Chi-squared map.
      6. Source-plane image (plane 1).

    A visually improved source-plane reconstruction in the bottom row
    indicates that the subhalo is detected.

    Parameters
    ----------
    fit_imaging_no_subhalo : FitImaging
        The imaging fit from the model *without* a subhalo.
    fit_imaging_with_subhalo : FitImaging
        The imaging fit from the model *with* a subhalo included.
    output_path : str, optional
        Directory in which to save the figure.  If ``None`` the figure is
        not saved to disk.
    output_format : str, optional
        Image format passed to :func:`~autoarray.plot.utils.save_figure`.
    colormap : str, optional
        Matplotlib colormap name.
    """
    fig, axes = subplots(2, 3, figsize=(21, 14))

    plot_array(
        array=fit_imaging_no_subhalo.normalized_residual_map,
        ax=axes[0][0],
        title="Normalized Residual Map (No Subhalo)",
        colormap=colormap,
    )
    plot_array(
        array=fit_imaging_no_subhalo.chi_squared_map,
        ax=axes[0][1],
        title="Chi-Squared Map (No Subhalo)",
        colormap=colormap,
    )
    _plot_source_plane(fit_imaging_no_subhalo, axes[0][2], plane_index=1,
                       colormap=colormap)

    plot_array(
        array=fit_imaging_with_subhalo.normalized_residual_map,
        ax=axes[1][0],
        title="Normalized Residual Map (With Subhalo)",
        colormap=colormap,
    )
    plot_array(
        array=fit_imaging_with_subhalo.chi_squared_map,
        ax=axes[1][1],
        title="Chi-Squared Map (With Subhalo)",
        colormap=colormap,
    )
    _plot_source_plane(fit_imaging_with_subhalo, axes[1][2], plane_index=1,
                       colormap=colormap)

    tight_layout()
    save_figure(fig, path=output_path, filename="detection_fits", format=output_format)
