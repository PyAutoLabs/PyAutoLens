(troubleshooting)=

# Troubleshooting

<span id="pip-version"></span>

<span id="pip-conda"></span>

## Check the active environment

Run these commands in the terminal where installation or execution failed:

```bash
python --version
python -m pip --version
python -m pip check
```

Use Python 3.12 or later. If pip reports conflicting dependencies, resolve them or retry in a fresh environment. Do not ignore the conflict and assume the installation is complete.

For a pip installation error, update pip in that environment and retry:

```bash
python -m pip install --upgrade pip
python -m pip install autolens
```

<span id="current-working-directory"></span>

## Missing files when running examples

Run scripts from the workspace root, not from inside a scripts subfolder:

```bash
cd path/to/autolens_workspace
python scripts/imaging/start_here.py
```

Replace `path/to/autolens_workspace` with your workspace location. Check that the dataset required by the example is present; a missing file is not always a working-directory problem.

## Workspace and package versions

A checkout of the workspace’s `main` branch can be newer than your installed library. Check your version:

```bash
python -c "import autolens; print(autolens.__version__)"
```

Use a workspace release tag matching that version where available, or update to a compatible published library/workspace pair. Do not change API calls just to work around an unexplained version mismatch.

<span id="matplotlib-backend"></span>

## Plot windows do not open

Check whether your Python session has access to a graphical desktop. On a remote or headless machine, save figures to files instead of opening windows.

The workspace setting is `config/visualize/general.yaml` → `general` → `backend`. See [Matplotlib’s backend guide](https://matplotlib.org/stable/users/explain/figure/backends.html) before choosing a backend; interactive backends require their corresponding GUI packages.

## GPU not detected

See [GPU setup](acceleration.md) and check the output of `jax.devices()`.

## Older Python versions

Use a new environment with Python 3.12 or later for current releases. If you need a historical version, pin that version explicitly and use its corresponding documentation.

<span id="community-support"></span>

## Get help

Post in [PyAutoLabs Discussions](https://github.com/orgs/PyAutoLabs/discussions) with your operating system, Python and package versions, the command you ran, and the complete traceback.
