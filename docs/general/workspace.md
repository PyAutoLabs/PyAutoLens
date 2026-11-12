(workspace)=
# Workspace tour

The [autolens workspace](https://github.com/PyAutoLabs/autolens_workspace) contains examples and datasets for modelling strong lenses. Use this page to find where things live; the [workspace README](https://github.com/PyAutoLabs/autolens_workspace#readme) has the full directory guide.

<span id="scripts--notebooks"></span>
<span id="scripts-notebooks"></span>


## Find an example

Start with the [New User Guide](https://pyautolens.readthedocs.io/en/latest/overview/overview_2_new_user_guide.html) to choose an example for your observations. Open its notebook to read and run it, or use the matching Python script where provided.

<span id="config"></span>

<span id="dataset"></span>

<span id="output"></span>

## Where files live

| Folder | What you use it for |
|---|---|
| [`notebooks/`](https://github.com/PyAutoLabs/autolens_workspace/tree/main/notebooks) | Worked examples with explanations, code and figures. |
| [`scripts/`](https://github.com/PyAutoLabs/autolens_workspace/tree/main/scripts) | Python versions to run and adapt for your analysis. |
| [`dataset/`](https://github.com/PyAutoLabs/autolens_workspace/tree/main/dataset) | Example observations and inputs used by the scripts. |
| [`config/`](https://github.com/PyAutoLabs/autolens_workspace/tree/main/config) | Overrides for priors, plots, output and other defaults. See [Configuration](configs.md). |
| `output/` | Results written when you run a fit. The search's name and path settings organize each analysis. |

Run local examples from the workspace root so their relative dataset and configuration paths resolve. If you need to set up the workspace, follow the existing [installation guide](https://pyautolens.readthedocs.io/en/latest/installation/overview.html).

<span id="slam"></span>

## SLaM pipelines

For analyses that fit the source, lens light and mass in stages, see the [SLaM examples](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/pixelization/slam.ipynb). Start with a basic imaging fit before adapting a pipeline.

<span id="howtolens"></span>

<span id="howtogalaxy"></span>

## Next steps

- Change plotting or output defaults in [Configuration](configs.md).
- Build and inspect your model with the [Model cookbook](model_cookbook.md).
- Work through [HowToLens](https://github.com/PyAutoLabs/HowToLens) for the science behind the examples.

If you use the [AI Assistant](https://github.com/PyAutoLabs/autolens_assistant), you can ask:

> I have [describe your observations] and want to model [describe the objects]. Find a matching workspace example, explain its inputs, and show me which parts I need to adapt.
