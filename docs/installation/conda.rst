.. _conda:

Installation with conda
=======================

JAX & GPU
---------

**PyAutoLens** runs significantly faster on GPUs — often **50x or more** compared to CPUs.

This acceleration is achieved through [**JAX**](https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html), which provides GPU and TPU support.

When you install **PyAutoLens** (see instructions below), JAX will also be installed. However, the default installation may not include GPU support.

To ensure GPU acceleration, it is recommended that you install JAX with GPU support **before** installing **PyAutoLens**, by following the official [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

If you install **PyAutoLens** without a proper GPU setup, a warning will be displayed.

Install
-------

Installation via a conda environment circumvents compatibility issues when installing certain libraries. This guide
assumes you have a working installation of `conda <https://conda.io/miniconda.html>`_.

First, update conda:

.. code-block:: bash

    conda update -n base -c defaults conda

Next, create a conda environment (we name this ``autolens`` to signify it is for the **PyAutoLens** install):

The command below creates this environment with Python 3.11, the most recent supported version of Python:

.. code-block:: bash

    conda create -n autolens python=3.11

Activate the conda environment (you will have to do this every time you want to run **PyAutoLens**):

.. code-block:: bash

    conda activate autolens

We upgrade pip to ensure certain libraries install:

.. code-block:: bash

    pip install --upgrade pip

The latest version of **PyAutoLens** is installed via pip as follows (the command ``--no-cache-dir`` prevents
caching issues impacting the installation):

.. code-block:: bash

    pip install autolens --no-cache-dir

You may get warnings which state something like:

.. code-block:: bash

    ERROR: autoarray 2026.4.5.3 has requirement numpy<=1.22.1, but you'll have numpy 1.22.2 which is incompatible.
    ERROR: numba 0.53.1 has requirement llvmlite<0.37,>=0.36.0rc1, but you'll have llvmlite 0.38.0 which is incompatible.

If you see these messages, they do not mean that the installation has failed and the instructions below will
identify clearly if the installation is a success.

If there are no errors (but only the warnings above) **PyAutoLens** is installed! 

If there is an error check out the `troubleshooting section <https://pyautolens.readthedocs.io/en/latest/installation/troubleshooting.html>`_.

Numba
-----

Numba (https://numba.pydata.org)  is an optional library which makes **PyAutoLens** run a lot faster, which we
strongly recommend users have installed.

You can install numba via the following command:

.. code-block:: bash

    pip install numba --no-cache-dir

Some users have experienced difficulties installing numba, which is why it is an optional library. If your
installation is not successful, you can use **PyAutoLens** without it installed for now, to familiarize yourself
with the software and determine if it is the right software for you.

If you decide that **PyAutoLens** is the right software, then I recommend you commit the time to getting a
successful numba install working, with more information provided `at this readthedocs page <https://pyautolens.readthedocs.io/en/latest/installation/numba.html>`_

Workspace
---------

Next, clone the ``autolens workspace`` (the line ``--depth 1`` clones only the most recent branch on
the ``autolens_workspace``, reducing the download size):

.. code-block:: bash

   cd /path/on/your/computer/you/want/to/put/the/autolens_workspace
   git clone https://github.com/Jammy2211/autolens_workspace --depth 1
   cd autolens_workspace

Run the ``welcome.py`` script to get started!

.. code-block:: bash

   python3 welcome.py

It should be clear that **PyAutoLens** runs without issue. 

If there is an error check out the `troubleshooting section <https://pyautolens.readthedocs.io/en/latest/installation/troubleshooting.html>`_.

Optional
--------

For interferometer analysis there are two optional dependencies that must be installed via the commands:

.. code-block:: bash

    pip install pynufft

**PyAutoLens** will run without these libraries and it is recommended that you only install them if you intend to
do interferometer analysis.

If you run interferometer code a message explaining that you need to install these libraries will be printed, therefore
it is safe not to install them initially.