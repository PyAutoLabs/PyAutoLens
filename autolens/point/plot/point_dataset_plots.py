import numpy as np
from typing import Optional

from autoarray.plot.utils import subplots, save_figure, conf_subplot_figsize, tight_layout


def subplot_dataset(
    dataset,
    output_path: Optional[str] = None,
    output_format: str = None,
    title_prefix: str = None,
    solver=None,
    family_index=None,
):
    """
    Produce a subplot visualising a `PointDataset`.

    The subplot contains one or two panels depending on whether flux
    measurements are present:

    * **Positions panel** (always shown), titled ``Dataset Positions (<name>)``:
      the observed point-source positions as large filled markers, each with a
      circle at its noise-map radius (never inflated, so it may be sub-pixel)
      and the sigma stated in the legend, e.g. ``Observed (σ = 0.005″)``. The
      panel extent is the ``PointSolver`` grid extent when ``solver`` is given
      (where the model can find images), otherwise the critical-curve region of
      ``critical_curve_grid_from``, so no position is clipped. There are no axis
      labels, only arc-second tick labels. Images are numbered 1..N as on every
      point-source figure. This is the one-source case of
      :func:`~autolens.point.plot.multi_source_plots.plot_dataset_positions`,
      which draws every source of a cluster in one panel.
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
    title_prefix : str, optional
        Text prepended to every panel title.
    solver : PointSolver, optional
        The solver of the model-fit, whose grid sets the positions panel extent.
    """
    from autoarray.plot.yx import plot_yx
    from autolens.point.plot import multi_source_plots as msp
    from autolens.point.plot import point_sources as ps

    if dataset.fluxes is None:
        msp.plot_dataset_positions(
            [dataset],
            solver=solver,
            output_path=output_path,
            output_format=output_format,
            filename="dataset_point",
            title_prefix=title_prefix,
            family_index=family_index,
        )
        return

    records = ps.records_from_datasets([dataset])
    ps.assign_styles(records)
    extent = ps.union_extent_from(records, solver=solver)

    fig, axes = subplots(1, 2, figsize=conf_subplot_figsize(1, 2))
    axes_flat = list(np.array(axes).flatten())

    msp.draw_dataset_positions(
        axes_flat[0],
        records,
        extent,
        title=msp.dataset_title_from(records, title_prefix),
        family_index=family_index,
    )

    _prefix = f"{title_prefix.rstrip()} " if title_prefix else ""
    y = np.array(dataset.fluxes)
    x = np.arange(len(y))
    plot_yx(
        y=y,
        x=x,
        ax=axes_flat[1],
        title=f"{_prefix}Dataset Fluxes ({dataset.name})",
        output_path=None,
        output_filename="point_dataset_fluxes",
        output_format=output_format,
    )

    tight_layout()
    save_figure(fig, path=output_path, filename="dataset_point", format=output_format)
