.. _overview:

Overview
========

**PyAutoLens** requires Python 3.12 - 3.13 and supports the Linux, MacOS and Windows operating systems.

**PyAutoLens** can be installed via the Python distribution `Anaconda <https://www.anaconda.com/>`_ or using
`Pypi <https://pypi.org/>`_ to ``pip install autolens`` into your Python distribution.

We recommend Anaconda as it manages the installation of many major libraries (e.g. numpy, scipy,
matplotlib, etc.) making installation more straight forward. Windows users must use Anaconda.

The installation guide for both approaches can be found at:

- `Anaconda installation guide <https://pyautolens.readthedocs.io/en/latest/installation/conda.html>`_

- `PyPI installation guide <https://pyautolens.readthedocs.io/en/latest/installation/pip.html>`_

Users who wish to build **PyAutoLens** from source (e.g. via a ``git clone``) should follow
our `building from source installation guide <https://pyautolens.readthedocs.io/en/latest/installation/source.html>`_.

JAX & GPU
---------

**PyAutoLens** runs significantly faster on GPUs — often **50x or more** compared to CPUs.

This acceleration is achieved through [**JAX**](https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html), which provides GPU and TPU support.

When you install **PyAutoLens** (see instructions below), JAX will also be installed. However, the default installation may not include GPU support.

To ensure GPU acceleration, it is recommended that you install JAX with GPU support **before** installing **PyAutoLens**, by following the official [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

If you install **PyAutoLens** without a proper GPU setup, a warning will be displayed.

Dependencies
------------

**PyAutoLens** uses the following parent packages:

**PyAutoConf** https://github.com/rhayes777/PyAutoConf

**PyAutoFit** https://github.com/rhayes777/PyAutoFit

**PyAutoArray** https://github.com/Jammy2211/PyAutoArray

**PyAutoGalaxy** https://github.com/Jammy2211/PyAutoGalaxy

