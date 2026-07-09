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

Plot Customization [aplt]
-------------------------

The plotting API is **functional**: customization is done by passing keyword
arguments directly to any ``aplt`` plotting function. There is no ``MatPlot2D``
object anymore (nor the old ``Cmap`` / ``Visuals`` / ``Units`` / matplotlib-wrapper
classes) — these were removed in favour of the keyword-argument interface below.

Every figure and subplot function accepts:

- ``title`` — the figure title.
- ``colormap`` — the matplotlib colormap name (e.g. ``"jet"``, ``"hot"``, ``"gray"``).
- ``use_log10`` — if ``True``, plot the colormap on a ``log10`` scale.
- ``output_path`` — directory to save the figure to (omit to display it interactively).
- ``output_filename`` — the saved file's name.
- ``output_format`` — the saved file's format, e.g. ``"png"`` or ``"pdf"``.

For example:

.. code-block:: python

    import autolens.plot as aplt

    # Customize the appearance:
    aplt.plot_array(array=image, title="Image", colormap="jet", use_log10=True)

    # Save to disk instead of displaying:
    aplt.plot_array(
        array=image, output_path="output", output_filename="image", output_format="png"
    )

Default plotting values (figure size, fonts, colormaps, ticks, labels, ...) are set
via the workspace ``config/visualize`` YAML files rather than in code, so most figures
need no customization at all.

Figure Output [aplt]
--------------------

The ``Output`` object gives lower-level control of how and where figures are written
to disk, and is accepted by the plotting functions in place of the individual
``output_*`` keyword arguments.

.. currentmodule:: autoarray.plot

.. autosummary::
   :toctree: _autosummary

   Output
