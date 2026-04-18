.. _workspace:

Workspace Tour
==============

You should have downloaded and configured the `autolens workspace <https://github.com/Jammy2211/autolens_workspace>`_
when you installed **PyAutoLens**. If you didn't, checkout the
`installation instructions <https://pyautolens.readthedocs.io/en/latest/general/installation.html#installation-with-pip>`_
for how to downloaded and configure the workspace.

The ``README.rst`` files distributed throughout the workspace describe every folder and file, and specify if
examples are for beginner or advanced users.

New users should begin by checking out the following parts of the workspace.

Scripts / Notebooks
-------------------

There are numerous example describing how to perform lensing calculations, lens modeling, and many other
**PyAutoLens** features. All examples are provided as Python scripts and Jupyter notebooks.

Descriptions of every configuration file and their input parameters are provided in the ``README.rst`` in
the `config directory of the workspace <https://github.com/Jammy2211/autolens_workspace/tree/release/config>`_

Config
------

Here, you'll find the configuration files which customize:

    - The default settings used by every non-linear search.
    - Visualization, including the backend used by *matplotlib*.
    - The priors and notation configs associated with the light and mass profiles used for lens modeling.
    - The behaviour of different (y,x) Cartesian grids used to perform lens calculations.
    - The general.yaml config which customizes other aspects of **PyAutoLens**.

Checkout the `configuration <https://pyautolens.readthedocs.io/en/latest/general/installation.html#installation-with-pip>`_
section of the readthedocs for a complete description of every configuration file.

Dataset
-------

Contains the dataset's used to perform lens modeling. Example datasets using simulators included with the workspace
are included here by default.

Output
------

The folder where  modeling results are stored.

SLaM
----

Advanced lens modeling pipelines that use the Source, Light and Mass (SLaM) approach to lens modeling.

See `here <https://pyautolens.readthedocs.io/en/latest/advanced/slam.html>`_ for an overview.

HowToLens
---------

The **HowToLens** lecture series are a collection of Jupyter notebooks describing how to build a **PyAutoLens** model
fitting project and giving illustrations of different statistical methods and techniques.

Checkout the
`tutorials section <file:///Users/Jammy/Code/PyAutoLabs/PyAutoLens/docs/_build/tutorials/howtolens.html>`_ for a
full description of the lectures and online examples of every notebook.