(source)=

# Building From Source

## JAX & GPU

**PyAutoLens** runs significantly faster on GPUs — often **50x or more** compared to CPUs.

This acceleration is achieved through \[**JAX**\](<https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html>), which provides GPU and TPU support.

When you install **PyAutoLens** (see instructions below), JAX will also be installed. However, the default installation may not include GPU support.

To ensure GPU acceleration, it is recommended that you install JAX with GPU support **before** installing **PyAutoLens**, by following the official \[JAX installation guide\](<https://jax.readthedocs.io/en/latest/installation.html>).

If you install **PyAutoLens** without a proper GPU setup, a warning will be displayed.

## Packages

Building from source means that you clone (or fork) the **PyAutoLens** GitHub repository and run **PyAutoLens** from
there. Unlike `conda` and `pip` this provides a build of the source code that you can edit and change, to
contribute the development **PyAutoLens** or experiment with yourself!

A large amount of **PyAutoLens** functionality is contained in its parent projects:

**PyAutoFit** <https://github.com/PyAutoLabs/PyAutoFit>

**PyAutoArray** <https://github.com/PyAutoLabs/PyAutoArray>

**PyAutoGalaxy** <https://github.com/PyAutoLabs/PyAutoGalaxy>

If you wish to build from source all code you may need to build from source these 3 additional
projects. We include below instructions for building just **PyAutoLens** from source or building all projects.

## Building Only PyAutoLens

We upgrade pip to ensure certain libraries install:

```bash
pip install --upgrade pip
```

First, clone (or fork) the **PyAutoLens** GitHub repository:

```bash
git clone https://github.com/PyAutoLabs/PyAutoLens
```

Next, install the **PyAuto** parent projects via pip:

```bash
pip install autoconf
pip install autofit
pip install autoarray
pip install autogalaxy
```

Next, install the **PyAutoLens** dependencies via pip:

```bash
pip install -r PyAutoLens/requirements.txt
```

Next, install the optional dependency numba via pip (see [this link](https://pyautolens.readthedocs.io/en/latest/installation/numba.html) for a description of numba):

```bash
pip install numba
```

For unit tests to pass you will also need the following optional requirements:

```bash
pip install pynufft
```

If you are using a `conda` environment, add the source repository as follows:

\[NOTE: Certain versions of conda use the command `conda develop` (without a dash) instead of those shown below.\]

```bash
conda-develop PyAutoLens
```

Alternatively, if you are using a Python environment include the **PyAutoLens** source repository in your PYTHONPATH
(noting that you must replace the text `/path/to` with the path to the **PyAutoLens** directory on your computer):

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/PyAutoLens
```

Finally, check the **PyAutoLens** unit tests run and pass (you may need to install pytest via `pip install pytest`):

```bash
cd /path/to/PyAutoLens
python3 -m pytest
```

## Building All Projects

We upgrade pip to ensure certain libraries install:

```bash
pip install --upgrade pip
```

First, clone (or fork) all 4 GitHub repositories:

```bash
git clone https://github.com/PyAutoLabs/PyAutoFit
git clone https://github.com/PyAutoLabs/PyAutoArray
git clone https://github.com/PyAutoLabs/PyAutoGalaxy
git clone https://github.com/PyAutoLabs/PyAutoLens
```

Next, install **PyAutoConf** via pip:

```bash
pip install autoconf
```

Next, install the source build dependencies of each project via pip:

```bash
pip install -r PyAutoFit/requirements.txt
pip install -r PyAutoArray/requirements.txt
pip install -r PyAutoGalaxy/requirements.txt
pip install -r PyAutoLens/requirements.txt
```

Next, install the optional dependency numba via pip (see [this link](https://pyautolens.readthedocs.io/en/latest/installation/numba.html) for a description of numba):

```bash
pip install numba
```

For unit tests to pass you will also need the following optional requirements:

```bash
pip install -r PyAutoArray/optional_requirements.txt
```

If you are using a `conda` environment, add each source repository as follows:

\[NOTE: Certain versions of conda use the command `conda develop` (without a dash) instead of those shown below.\]

```bash
conda-develop PyAutoFit
conda-develop PyAutoArray
conda-develop PyAutoGalaxy
conda-develop PyAutoLens
```

Alternatively, if you are using a Python environment include each source repository in your PYTHONPATH
(noting that you must replace the text `/path/to` with the path to the each directory on your computer):

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/PyAutoFit
export PYTHONPATH=$PYTHONPATH:/path/to/PyAutoArray
export PYTHONPATH=$PYTHONPATH:/path/to/PyAutoGalaxy
export PYTHONPATH=$PYTHONPATH:/path/to/PyAutoLens
```

Finally, check the unit tests run and pass for each project (you may need to install pytest via `pip install pytest`):

```bash
cd /path/to/PyAutoFit
python3 -m pytest
cd ../PyAutoArray
python3 -m pytest
cd ../PyAutoGalaxy
python3 -m pytest
cd ../PyAutoLens
python3 -m pytest
```
