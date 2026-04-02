import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

from autoarray.plot.utils import save_figure, conf_subplot_figsize


def subplot_fit(
    fit,
    output_path: Optional[str] = None,
    output_format: str = "png",
    image_plane_lines=None,
    image_plane_line_colors=None,
    source_plane_lines=None,
    source_plane_line_colors=None,
):
    """
    Produce a subplot summarising a `FitPointDataset`.

    The subplot contains one or two panels depending on whether flux
    measurements are present in the dataset:

    * **Positions panel** (always shown): observed point-source positions
      plotted as a grid, with the model-predicted positions overlaid as
      red scatter points.
    * **Fluxes panel** (shown only when ``fit.dataset.fluxes`` is not
      ``None``): a bar/line plot of the observed flux values.

    Parameters
    ----------
    fit : FitPointDataset
        The point-source dataset fit to visualise.
    output_path : str, optional
        Directory in which to save the figure.  If ``None`` the figure is
        not saved to disk.
    output_format : str, optional
        Image format passed to :func:`~autoarray.plot.utils.save_figure`.
    """
    from autoarray.plot.grid import plot_grid
    from autoarray.plot.yx import plot_yx

    has_fluxes = fit.dataset.fluxes is not None
    n = 2 if has_fluxes else 1

    fig, axes = plt.subplots(1, n, figsize=conf_subplot_figsize(1, n))
    axes_flat = [axes] if n == 1 else list(np.array(axes).flatten())

    # Positions panel
    obs_grid = np.array(
        fit.dataset.positions.array
        if hasattr(fit.dataset.positions, "array")
        else fit.dataset.positions
    )
    model_grid = np.array(
        fit.positions.model_data.array
        if hasattr(fit.positions.model_data, "array")
        else fit.positions.model_data
    )

    plot_grid(
        grid=obs_grid,
        ax=axes_flat[0],
        title=f"{fit.dataset.name} Fit Positions",
        output_path=None,
        output_filename=None,
        output_format=output_format,
    )
    axes_flat[0].scatter(model_grid[:, 1], model_grid[:, 0], c="r", s=20, zorder=5)

    # Fluxes panel
    if has_fluxes and n > 1:
        y = np.array(fit.dataset.fluxes)
        x = np.arange(len(y))
        plot_yx(
            y=y,
            x=x,
            ax=axes_flat[1],
            title=f"{fit.dataset.name} Fit Fluxes",
            output_path=None,
            output_filename="fit_point_fluxes",
            output_format=output_format,
        )

    plt.tight_layout()
    save_figure(fig, path=output_path, filename="fit", format=output_format)
