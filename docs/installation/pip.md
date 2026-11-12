(pip)=

<span id="installation-with-pip"></span>

# Install with pip

Use an isolated environment so this installation does not change packages used by your other projects.

## Create an environment

On Linux or macOS, with Python 3.12 installed:

```bash
python3.12 -m venv autolens-env
source autolens-env/bin/activate
```

On Windows, in Command Prompt:

```bat
py -3.12 -m venv autolens-env
autolens-env\Scriptsctivate.bat
```

If you use a different supported Python version, substitute it in the environment-creation command. See [Python’s environment guide](https://docs.python.org/3/library/venv.html) for other shells, including PowerShell.

## Install

```bash
python -m pip install --upgrade pip
python -m pip install autolens
```

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

## Legacy Python versions

See [older Python versions](troubleshooting.md#older-python-versions).
