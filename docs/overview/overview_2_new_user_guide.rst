.. _overview_2_new_user_guide:

New User Guide
==============

**PyAutoLens** can analyse strong lens systems across a range of physical scales (e.g. galaxy, group, and cluster) and for
different types of data (e.g. imaging, interferometer, and point-source observations). Depending on the scientific questions you are interested in, the analysis you perform may differ significantly.

The autolens_workspace contains a suite of example Jupyter Notebooks, organised by lens scale and dataset type.
To help you find the most appropriate starting point, we begin by answering two simple questions.

What Scale Lens?
----------------

What size and scale of strong lens system are you expecting to work with?

There are three scales to choose from:

- **Galaxy Scale**: Made up of a single lens galaxy lensing a single source galaxy, the simplest strong lens you can get!
  If you're interested in galaxy scale lenses, go to the question below called "What Data Type?".

- **Group Scale**: Strong Lens Groups contains 2-10 lens galaxies, normally with one main large galaxy responsible for the majority of lensing.
  They also typically lens just one source galaxy. If you are interested in groups, go to the `group/start_here.ipynb <https://github.com/Jammy2211/autolens_workspace/blob/release/notebooks/group/start_here.ipynb>`_ notebook.

- **Cluster Scale**: Strong Lens Galaxy clusters often contained 20-50, or more, lens galaxies, lensing 10, or more, sources galaxies.
  If you are interested in clusters, go to the `cluster/start_here.ipynb <https://github.com/Jammy2211/autolens_workspace/blob/release/notebooks/cluster/start_here.ipynb>`_ notebook.

What Dataset Type?
------------------

If you are interested in galaxy-scale strong lenses, you now need to decide what type of strong lens data you are
interested in:

- **CDD Imaging**: For image data from telescopes like Hubble and James Webb, go to `imaging/start_here.ipynb <https://github.com/Jammy2211/autolens_workspace/blob/release/notebooks/imaging/start_here.ipynb>`_.

- **Interferometer**: For radio / sub-mm interferometer from instruments like ALMA, go to `interferometer/start_here.ipynb <https://github.com/Jammy2211/autolens_workspace/blob/release/notebooks/interferometer/start_here.ipynb>`_.

- **Point Sources**: For strongly lensed point sources (e.g. lensed quasars, supernovae), go to `point_source/start_here.ipynb <hhttps://github.com/Jammy2211/autolens_workspace/blob/release/notebooks/point_source/start_here.ipynb>`_.

Google Colab
------------

The links above take you to the GitHub page of each notebook, and if you've cloned the workspace you can open them
locally on your machine.

However, you can also open and run each notebook directly in Google Colab, which provides a free cloud computing
environment with all the required dependencies already installed.

This is a great way to get started quickly without needing to install **PyAutoLens** on your own machine,
so you can check its the right software for you before going through the installation process:

 - `imaging/start_here.ipynb <https://colab.research.google.com/github/PyAutoLabs/autolens_workspace/blob/2026.4.13.6/notebooks/imaging/start_here.ipynb>`_: Galaxy scale strong lenses observed with CCD imaging (e.g. Hubble, James Webb).
 - `interferometer/start_here.ipynb <https://colab.research.google.com/github/PyAutoLabs/autolens_workspace/blob/2026.4.13.6/notebooks/interferometer/start_here.ipynb>`_: Galaxy scale strong lenses observed with interferometer data (e.g. ALMA).
 - `point_source/start_here.ipynb <https://colab.research.google.com/github/PyAutoLabs/autolens_workspace/blob/2026.4.13.6/notebooks/point_source/start_here.ipynb>`_: Galaxy scale strong lenses with a lensed point source (e.g. lensed quasars).
 - `group/start_here.ipynb <https://colab.research.google.com/github/PyAutoLabs/autolens_workspace/blob/2026.4.13.6/notebooks/group/start_here.ipynb>`_: Group scale strong lenses where there are 2-10 lens galaxies.
 - `cluster/start_here.ipynb <https://colab.research.google.com/github/PyAutoLabs/autolens_workspace/blob/2026.4.13.6/notebooks/cluster/start_here.ipynb>`_: Cluster scale strong lenses with 2+ lenses and 5+ source galaxies.

Still Unsure?
-------------

Each notebook is short and self-contained, and can be completed and adapted quickly to your particular task.
Therefore, if you're unsure exactly which scale of lensing applies to you, or quite what data you want to use, you
should just read through a few different notebooks and go from there.

HowToLens
---------

For experienced scientists, the **PyAutoLens** examples will be simple to follow. Concepts surrounding strong lensing may
already be familiar and the statistical techniques used for fitting and modeling already understood.

For those less familiar with these concepts (e.g. undergraduate students, new PhD students or interested members of the
public), things may have been less clear and a slower more detailed explanation of each concept would be beneficial.

The **HowToLens** Jupyter Notebook lectures provide exactly this. They are a 3+ chapter guide which thoroughly
take you through the core concepts of strong lensing, teach you the principles of the statistical techniques
used in modeling and ultimately will allow you to undertake scientific research like a professional astronomer.

To complete thoroughly, they'll probably take 2-4 days, so you may want try moving ahead to the examples but can
go back to these lectures if you find them hard to follow.

If this sounds like it suits you, checkout the `HowToLens <https://github.com/PyAutoLabs/HowToLens>`_ repository now.

Wrap Up
-------

After completing this guide, you should be able to use **PyAutoLens** for your science research.

The biggest decisions you'll need to make are what features and functionality your specific science case requires,
which the next readthedocs page gives an overview of to help you decide.