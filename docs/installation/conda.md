(conda)=

<span id="installation-with-conda"></span>

# Install with conda

Use this route if you have [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html), or want it to manage your Python version. Open a terminal with conda available; on Windows, use Anaconda Prompt.

<span id="install"></span>

## Create an environment and install

```bash
conda create -n autolens python=3.12 pip
conda activate autolens
python -m pip install --upgrade pip
python -m pip install autolens
```

Run `conda activate autolens` each time you open a new terminal to use this environment.

For GPU support, follow [GPU setup](acceleration.md) in the same environment.

## Check the installation

```bash
python -m pip check
python -c "import autolens; print(autolens.__version__)"
```

`pip check` should report no broken requirements, and the second command should print the installed version. Resolve errors using [Troubleshooting](troubleshooting.md) before continuing.

Next, [download the workspace and run an example](workspace.md).

## Workspace

[Download the workspace and run your first example](workspace.md).

## Numba

See [Numba and optional CPU acceleration](acceleration.md#numba).

## Optional

See [interferometer packages](acceleration.md#interferometer-calculations).

## JAX & GPU

See [JAX and GPU setup](acceleration.md#jax-and-gpus).
