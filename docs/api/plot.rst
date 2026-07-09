========
Plotting
========

**PyAutoLens** custom visualization library.

Step-by-step Juypter notebook guides illustrating all objects listed on this page are 
provided on the `autolens_workspace: plot tutorials <https://github.com/PyAutoLabs/autolens_workspace/tree/main/notebooks/guides/plot>`_ and
it is strongly recommended you use those to learn plot customization.

**Examples / Tutorials:**

- `autolens_workspace: plot tutorials <https://github.com/PyAutoLabs/autolens_workspace/tree/main/notebooks/guides/plot>`_

Plotters [aplt]
---------------

Create figures and subplots showing quantities of standard **PyAutoLens** objects.

.. currentmodule:: autolens.plot

**Basic Plot Functions:**

.. autosummary::
   :toctree: _autosummary

    plot_array
    plot_grid

**Tracer and Galaxies Subplots:**

.. autosummary::
   :toctree: _autosummary

    subplot_tracer
    subplot_lensed_images
    subplot_galaxies_images

**Imaging Fit Subplots:**

.. autosummary::
   :toctree: _autosummary

    subplot_fit_imaging
    subplot_fit_imaging_log10
    subplot_fit_imaging_x1_plane
    subplot_fit_imaging_log10_x1_plane
    subplot_fit_imaging_of_planes
    subplot_fit_imaging_tracer
    subplot_fit_combined
    subplot_fit_combined_log10

**Interferometer Fit Subplots:**

.. autosummary::
   :toctree: _autosummary

    subplot_fit_interferometer
    subplot_fit_interferometer_real_space

**Point Source Subplots:**

.. autosummary::
   :toctree: _autosummary

    subplot_fit_point
    subplot_point_dataset

**Subhalo Detection Subplots:**

.. autosummary::
   :toctree: _autosummary

    subplot_detection_imaging
    subplot_detection_fits

**Sensitivity Mapping Subplots:**

.. autosummary::
   :toctree: _autosummary

    subplot_sensitivity_tracer_images
    subplot_sensitivity
    subplot_sensitivity_figures_of_merit

Non-linear Search Plot Functions [aplt]
---------------------------------------

Module-level functions for visualizing non-linear search results.

.. currentmodule:: autofit.plot

.. autosummary::
   :toctree: _autosummary

   corner_cornerpy
   corner_anesthetic
   subplot_parameters
   log_likelihood_vs_iteration
