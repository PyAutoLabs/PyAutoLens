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

Display an array or a grid of coordinates, or write an array to a ``.fits`` file.

.. autosummary::
   :toctree: _autosummary

    plot_array
    plot_grid
    fits_array

Dataset Subplots
~~~~~~~~~~~~~~~~

Inspect an imaging or interferometer dataset before fitting it, or write its components to ``.fits`` files.

.. autosummary::
   :toctree: _autosummary

    subplot_imaging_dataset
    subplot_imaging_dataset_intro
    subplot_imaging_dataset_list
    fits_imaging
    subplot_interferometer_dirty_images
    fits_interferometer

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
    subplot_fit_imaging_mappings
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
    subplot_fit_interferometer_tracer
    subplot_fit_interferometer_combined

Point Source subplots
~~~~~~~~~~~~~~~~~~~~~

Inspect the observed point-source positions, fitted predictions and solver output.

.. autosummary::
   :toctree: _autosummary

    subplot_fit_point
    subplot_point_dataset
    plot_point_solver
    plot_image_with_points

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

Galaxy and Light and Mass Profile subplots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inspect profile components, combined galaxy images, basis profiles and adapt images. These functions are implemented by **PyAutoGalaxy** and re-exported through ``aplt``.

.. autosummary::
   :toctree: _autosummary

    subplot_galaxy_light_profiles
    subplot_galaxy_mass_profiles
    subplot_basis_image
    subplot_galaxies
    subplot_galaxy_images
    subplot_adapt_images

Galaxy Fit subplots
~~~~~~~~~~~~~~~~~~~

Inspect a fit one galaxy at a time, or the real-space and dirty-image views of an interferometer fit. These are the **PyAutoGalaxy** fit subplots, re-exported through ``aplt``.

.. autosummary::
   :toctree: _autosummary

    subplot_fit_imaging_of_galaxy
    subplot_fit_dirty_images
    subplot_fit_real_space

Ellipse Fit subplots
~~~~~~~~~~~~~~~~~~~~

Inspect fitted isophotes and the errors associated with the ellipse fits.

.. autosummary::
   :toctree: _autosummary

    subplot_fit_ellipse
    subplot_ellipse_errors

Magnification maps
~~~~~~~~~~~~~~~~~~

Map the lensing magnification in the image plane and source plane, and its uncertainty over a set of posterior draws.

.. autosummary::
   :toctree: _autosummary

    plot_magnification_2d
    plot_magnification_source_plane
    plot_magnification_source_plane_grid
    plot_magnification_uncertainty_2d
    magnification_fractional_error_2d_from

Cluster plots
~~~~~~~~~~~~~

Inspect a cluster dataset, its multiple-image positions and the critical curves and caustics of the lens model.

.. autosummary::
   :toctree: _autosummary

    subplot_cluster_dataset
    plot_positions_overlay
    plot_image_group_zooms
    plot_critical_curves
    plot_caustics

Weak Lensing plots
~~~~~~~~~~~~~~~~~~

Inspect a weak-lensing shear dataset, its fit, the shear profile and the convergence map.

.. autosummary::
   :toctree: _autosummary

    subplot_weak_dataset
    plot_shear_yx_2d
    plot_ellipticities
    plot_phis
    plot_noise_map
    subplot_fit_weak
    plot_data_vs_model
    plot_residuals
    plot_chi_squared_map
    plot_shear_profile
    plot_convergence_map

Twin comparisons
~~~~~~~~~~~~~~~~

Compare a fitted lens (the parent) with a simulated twin made from it (``al.twins.compare``): images, residuals, convergence, enclosed mass and critical curves.

.. autosummary::
   :toctree: _autosummary

    subplot_twin_comparison

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
   output_figure

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
