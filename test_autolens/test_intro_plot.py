"""The introductory dataset compositor is available through the public lens API."""
import autolens.plot as aplt
from autoarray.dataset.plot.imaging_plots import subplot_imaging_dataset_intro


def test_intro_export():
    assert aplt.subplot_imaging_dataset_intro is subplot_imaging_dataset_intro
