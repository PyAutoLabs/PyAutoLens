Plotting
========

Use ``autolens.plot`` to inspect datasets, models and fits. Functions produce individual figures or diagnostic subplots; pass plotting options directly to the function you call.

For a runnable introduction and customization examples, open the `plotting guide <https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/guides/plot/start_here.ipynb>`_.

Plot an image
-------------

Given an existing ``Array2D`` called ``image``:

.. code-block:: python

    import autolens.plot as aplt

    aplt.plot_array(array=image, title="Image", colormap="gray")

.. _plotters-aplt:

Choose a plot
-------------

Choose the group that matches the object or result you want to inspect. The linked references give each function’s arguments.

.. currentmodule:: autolens.plot

Basic plots
~~~~~~~~~~~

Display an array or a grid of coordinates.

.. autosummary::
   :toctree: _autosummary

    plot_array
    plot_grid

Tracer and Galaxies subplots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inspect the lens model, its lensed images and the contributions from its galaxies.

.. autosummary::
   :toctree: _autosummary

    subplot_tracer
    subplot_lensed_images
    subplot_galaxies_images

Imaging Fit subplots
~~~~~~~~~~~~~~~~~~~~

Compare the image data and fitted model, and inspect residuals and other fit diagnostics.

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

Interferometer subplots
~~~~~~~~~~~~~~~~~~~~~~~

Inspect visibilities, fits and reconstructed images. Dirty images are useful diagnostics; the interferometer fit is evaluated in visibility space.

.. autosummary::
   :toctree: _autosummary

    subplot_interferometer_dataset
    subplot_fit_interferometer
    subplot_fit_interferometer_real_space
    subplot_fit_interferometer_dirty_images
    subplot_fit_interferometer_combined

Point Source subplots
~~~~~~~~~~~~~~~~~~~~~

Inspect the observed point-source positions, fitted predictions and solver output.

.. autosummary::
   :toctree: _autosummary

    subplot_fit_point
    subplot_point_dataset
    plot_point_solver

Subhalo Detection subplots
~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare subhalo-search results and their associated fits.

.. autosummary::
   :toctree: _autosummary

    subplot_detection_imaging
    subplot_detection_fits

Sensitivity Mapping subplots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inspect sensitivity results and their figures of merit across the lens.

.. autosummary::
   :toctree: _autosummary

    subplot_sensitivity_tracer_images
    subplot_sensitivity
    subplot_sensitivity_figures_of_merit

.. _non-linear-search-plot-functions-aplt:

Search results
--------------

These functions visualize samples and search progress. They are implemented by **PyAutoFit** and re-exported through ``aplt``.

.. currentmodule:: autofit.plot

.. autosummary::
   :toctree: _autosummary

   corner_cornerpy
   corner_anesthetic
   subplot_parameters
   log_likelihood_vs_iteration

.. _plot-customization-aplt:

Customize a plot
----------------

Options depend on the function. For ``plot_array``, you can set:

- ``title``: the figure title.
- ``colormap``: a Matplotlib colormap, such as ``"gray"`` or ``"viridis"``.
- ``use_log10``: request logarithmic colour scaling. Use a linear or suitable signed scale when inspecting quantities that can be negative, such as residuals.
- ``output_path``, ``output_filename`` and ``output_format``: the output directory, filename and format.

Subplot functions may use different options. For example, ``subplot_tracer`` accepts ``title_prefix`` rather than ``title`` and does not expose ``output_filename``. Check the linked signature before transferring options between functions.

Defaults such as figure size, fonts and colours are configured through the workspace’s ``config/visualize`` YAML files. See `Configuration <https://pyautolens.readthedocs.io/en/latest/general/configs.html>`_ for the configuration workflow.

.. _figure-output-aplt:

Save figures
------------

For ``plot_array``, save the existing image to ``output/image.png`` with:

.. code-block:: python

    aplt.plot_array(
        array=image,
        output_path="output",
        output_filename="image",
        output_format="png",
    )

The functions above use the output keywords documented in their signatures. ``Output`` is a lower-level **PyAutoArray** helper; it is not a replacement argument for these public plotting functions. Its reference remains available for code that uses it directly.

.. currentmodule:: autoarray.plot

.. autosummary::
   :toctree: _autosummary

   Output
