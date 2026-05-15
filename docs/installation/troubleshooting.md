(troubleshooting)=

# Troubleshooting

## Pip Version

If an error message appears after trying to run `pip install autolens` first make sure you are using
the latest version of pip.

```bash
pip install --upgrade pip
pip3 install --upgrade pip
```

## Pip / Conda

If you are trying to [install via pip](https://pyautolens.readthedocs.io/en/latest/installation/pip.html) but
still having issues, we recommend you try to [install via conda](https://pyautogalaxy.readthedocs.io/en/latest/installation/conda.html)
instead, or visa versa.

## Support

If you are still having issues with installation, please raise an issue on the
[autolens_workspace issues page](https://github.com/PyAutoLabs/autolens_workspace/issues) with a description of the
problem and your system setup (operating system, Python version, etc.).

## Current Working Directory

**PyAutoLens** scripts assume that the `autolens_workspace` directory is the Python working directory. This means
that, when you run an example script, you should run it from the `autolens_workspace` as follows:

```bash
cd path/to/autolens_workspace (if you are not already in the autolens_workspace).
python3 examples/model/beginner/mass_total__source_lp.py
```

The reasons for this are so that **PyAutoLens** can:

> - Load configuration settings from config files in the `autolens_workspace/config` folder.
> - Load example data from the `autolens_workspace/dataset` folder.
> - Output the results of models fits to your hard-disk to the `autolens/output` folder.

If you have any errors relating to importing modules, loading data or outputting results it is likely because you
are not running the script with the `autolens_workspace` as the working directory!

## Matplotlib Backend

Matplotlib uses the default backend on your computer, as set in the config file:

```bash
autolens_workspace/config/visualize/general.yaml
```

If unchanged, the backend is set to 'default', meaning it will use the backend automatically set up for Python on
your system.

```bash
general:
  backend: default
```

There have been reports that using the default backend causes crashes when running the test script below (either the
code crashes without a error or your computer restarts). If this happens, change the config's backend until the test
works (TKAgg has worked on Linux machines, Qt5Agg has worked on new MACs). For example:

```bash
general:
  backend: TKAgg
```
