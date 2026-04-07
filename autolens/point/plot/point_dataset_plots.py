import numpy as np
from typing import Optional

from autoarray.plot.utils import subplots, save_figure, conf_subplot_figsize, tight_layout


def subplot_dataset(
    dataset,
    output_path: Optional[str] = None,
    output_format: str = None,
    title_prefix: str = None,
):
    """
    Produce a subplot visualising a `PointDataset`.

    The subplot contains one or two panels depending on whether flux
    measurements are present:

    * **Positions panel** (always shown): the observed point-source
      positions rendered as a grid scatter plot.
    * **Fluxes panel** (shown only when ``dataset.fluxes`` is not
      ``None``): a line/bar plot of the observed flux values indexed by
      image position.

    Parameters
    ----------
    dataset : PointDataset
        The point-source dataset to visualise.
    output_path : str, optional
        Directory in which to save the figure.  If ``None`` the figure is
        not saved to disk.
    output_format : str, optional
        Image format passed to :func:`~autoarray.plot.utils.save_figure`.
    """
    from autogalaxy.util.plot_utils import plot_grid
    from autoarray.plot.yx import plot_yx

    has_fluxes = dataset.fluxes is not None
    n = 2 if has_fluxes else 1

    fig, axes = subplots(1, n, figsize=conf_subplot_figsize(1, n))
    axes_flat = [axes] if n == 1 else list(np.array(axes).flatten())

    grid = np.array(
        dataset.positions.array
        if hasattr(dataset.positions, "array")
        else dataset.positions
    )

    _prefix = title_prefix or ""
    plot_grid(
        grid=grid,
        ax=axes_flat[0],
        title=f"{_prefix}{dataset.name} Positions",
        output_path=None,
        output_filename=None,
        output_format=output_format,
    )

    if has_fluxes and n > 1:
        y = np.array(dataset.fluxes)
        x = np.arange(len(y))
        plot_yx(
            y=y,
            x=x,
            ax=axes_flat[1],
            title=f"{_prefix}{dataset.name} Fluxes",
            output_path=None,
            output_filename="point_dataset_fluxes",
            output_format=output_format,
        )

    tight_layout()
    save_figure(fig, path=output_path, filename="dataset_point", format=output_format)
