.. _pip:

Installation with pip
=====================

.. note::
   **PyAutoLens** requires **Python 3.12 or later**. If you are on Python
   3.9, 3.10, or 3.11, ``pip install autolens`` will fail with a "no matching
   distribution" error. Upgrade Python to 3.12+ before installing.

JAX & GPU
---------

**PyAutoLens** runs significantly faster on GPUs — often **50x or more** compared to CPUs.

This acceleration is achieved through [**JAX**](https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html), which provides GPU and TPU support.

When you install **PyAutoLens** (see instructions below), JAX will also be installed. However, the default installation may not include GPU support.

To ensure GPU acceleration, it is recommended that you install JAX with GPU support **before** installing **PyAutoLens**, by following the official [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

If you install **PyAutoLens** without a proper GPU setup, a warning will be displayed.


Install
-------

We strongly recommend that you install **PyAutoLens** in a
`Python virtual environment <https://www.geeksforgeeks.org/python-virtual-environment/>`_, with the link attached
describing what a virtual environment is and how to create one.

We upgrade pip to ensure certain libraries install:

.. code-block:: bash

    pip install --upgrade pip

The latest version of **PyAutoLens** is installed via pip as follows (specifying the version as shown below ensures
the installation has clean dependencies):

.. code-block:: bash

    pip install autolens

If pip prints warnings about dependency version conflicts, these can usually be ignored — the instructions below
will identify clearly if the installation is a success.

If there are no errors **PyAutoLens** is installed!

If there is an error check out the `troubleshooting section <https://pyautolens.readthedocs.io/en/latest/installation/troubleshooting.html>`_.

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

Numba
-----

Numba (https://numba.pydata.org)  is an optional library which makes **PyAutoLens** run a lot faster, which we
strongly recommend users have installed.

You can install numba via the following command:

.. code-block:: bash

    pip install numba

Some users have experienced difficulties installing numba, which is why it is an optional library. If your
installation is not successful, you can use **PyAutoLens** without it installed for now, to familiarize yourself
with the software and determine if it is the right software for you.

If you decide that **PyAutoLens** is the right software, then I recommend you commit the time to getting a
successful numba install working, with more information provided `at this readthedocs page <https://pyautolens.readthedocs.io/en/latest/installation/numba.html>`_

Optional
--------

For interferometer analysis there are two optional dependencies that must be installed via the commands:

.. code-block:: bash

    pip install pynufft

**PyAutoLens** will run without these libraries and it is recommended that you only install them if you intend to
do interferometer analysis.

If you run interferometer code a message explaining that you need to install these libraries will be printed, therefore
it is safe not to install them initially.

Legacy Python versions
----------------------

We dropped support for Python 3.9, 3.10, and 3.11 in release ``2026.4.5.3``
(April 2026). Pre-``2026.4.5.3`` releases on PyPI have been yanked, so they
will not install via the standard ``pip install autolens`` command.

If you have an existing project that requires a pre-``2026.4.5.3`` version,
you can still install it explicitly by pinning the version, e.g.:

.. code-block:: bash

    pip install autolens==2025.10.6.1

Yanked releases remain available for explicit pins; only resolver-driven
fallback is blocked.