(installation-workspace)=

# Run your first example

The [autolens_workspace](https://github.com/PyAutoLabs/autolens_workspace) contains example scripts, notebooks and configuration files. With your installation environment active, download it into a folder of your choice:

```bash
git clone --depth 1 https://github.com/PyAutoLabs/autolens_workspace.git
cd autolens_workspace
python welcome.py
```

This requires [Git](https://git-scm.com/downloads). The welcome script checks the setup and opens example plots; close each plot window when prompted to continue.

Run workspace scripts from this directory so they can find their configuration and data files.

## Choose your next step

- Read [start_here.ipynb](https://github.com/PyAutoLabs/autolens_workspace/blob/main/start_here.ipynb) for an introduction to the Python API.
- Run the same walkthrough as a Python script:

```bash
python start_here.py
```

- Use the [New User Guide](https://pyautolens.readthedocs.io/en/latest/overview/overview_2_new_user_guide.html) to choose an example for your observations.

To run notebooks locally, install and launch JupyterLab in this environment:

```bash
python -m pip install jupyterlab
python -m jupyter lab
```

If a workspace example reports a missing API, see [version mismatches](troubleshooting.md#workspace-and-package-versions).
