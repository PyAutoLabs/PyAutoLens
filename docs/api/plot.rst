========
Plotting
========

**PyAutoLens** custom visualization library.

Step-by-step Juypter notebook guides illustrating all objects listed on this page are 
provided on the `autolens_workspace: plot tutorials <https://github.com/Jammy2211/autolens_workspace/tree/release/notebooks/plot>`_ and
it is strongly recommended you use those to learn plot customization.

**Examples / Tutorials:**

- `autolens_workspace: plot tutorials <https://github.com/Jammy2211/autolens_workspace/tree/release/notebooks/plot>`_

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

Plot Customization [aplt]
-------------------------

Customize figures created via ``Plotter`` objects, including changing ``matplotlib`` settings and adding
visuals to figures.

.. currentmodule:: autogalaxy.plot

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   MatPlot1D
   MatPlot2D
   Visuals1D
   Visuals2D

Matplot Lib Wrappers [aplt]
---------------------------

Wrappers for every ``matplotlib`` function used by a ``Plotter``, allowing for detailed customization of
every figure and subplot.

.. currentmodule:: autogalaxy.plot

**Matplotlib Wrapper Base Objects:**

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   Units
   Figure
   Axis
   Cmap
   Colorbar
   ColorbarTickParams
   TickParams
   YTicks
   XTicks
   Title
   YLabel
   XLabel
   Legend
   Output

**Matplotlib Wrapper 1D Objects:**

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   YXPlot

**Matplotlib Wrapper 2D Objects:**

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   ArrayOverlay
   GridScatter
   GridPlot
   VectorYXQuiver
   PatchOverlay
   VoronoiDrawer
   OriginScatter
   MaskScatter
   BorderScatter
   PositionsScatter
   IndexScatter
   MeshGridScatter
   ParallelOverscanPlot
   SerialPrescanPlot
   SerialOverscanPlot