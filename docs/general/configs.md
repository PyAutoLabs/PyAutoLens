<span id="configs"></span>

# Configuration

PyAutoLens uses configuration files for plotting, default profile priors and saved output. You can keep the supplied settings for a first fit, then change the ones your analysis needs.

## Defaults and overrides

The libraries supply packaged defaults. A `config/` folder in your working directory overrides them; settings you do not override fall back to the packaged values. This is why workspace examples are normally run from the workspace root.

| Configuration | Controls |
|---|---|
| [`visualize/general.yaml`](https://github.com/PyAutoLabs/autolens_workspace/tree/main/config/visualize/general.yaml) | Plot backend, colours, units and display settings. |
| [`visualize/plots.yaml`](https://github.com/PyAutoLabs/autolens_workspace/tree/main/config/visualize/plots.yaml) | Which dataset, fit and reconstruction figures are saved. |
| [`priors/`](https://github.com/PyAutoLabs/autolens_workspace/tree/main/config/priors) | Default priors for profile parameters. |
| [`output.yaml`](https://github.com/PyAutoLabs/autolens_workspace/tree/main/config/output.yaml) | Saved samples, search internals and derived quantities. |
| [`notation.yaml`](https://github.com/PyAutoLabs/autolens_workspace/tree/main/config/notation.yaml) | Parameter labels and formatting. |
| [`general.yaml`](https://github.com/PyAutoLabs/autolens_workspace/tree/main/config/general.yaml) | Other analysis-wide settings. |

The [configuration README](https://github.com/PyAutoLabs/autolens_workspace/tree/main/config) describes the remaining files. Settings under `build/` are for maintaining the workspace rather than running a science analysis.

## Save fit figures in two formats

In `config/visualize/plots.yaml`, set:

```yaml
subplot_format: [png, pdf]
```

This changes the saved subplot formats. Keep the figures that help you assess the fit; for example:

```yaml
fit:
  subplot_fit: true
  subplot_fit_log10: true
```

The logarithmic view can make faint structure easier to inspect alongside the ordinary fit plot.

## Change the plotting defaults

In `config/visualize/general.yaml`, change the colour map:

```yaml
colormap: viridis
```

For a non-interactive machine, set the backend before importing the plotting module:

```yaml
general:
  backend: Agg
```

These are excerpts: edit the named keys in the existing file, keeping its other settings. Restart your Python session after editing configuration, especially after changing the plotting backend.

## Set a default prior

For linear Sérsic light profiles, `config/priors/light/linear/sersic.yaml` includes the default prior on the Sérsic index. For example:

```yaml
Sersic:
  sersic_index:
    type: Uniform
    lower_limit: 1.0
    upper_limit: 6.0
```

This changes the default for newly composed `lp_linear.Sersic` models using that configuration. Choose limits for your scientific model; the numbers here are an example, not a recommendation for every galaxy. Preserve the other entries in the file.

For one component in one analysis, setting its prior in Python is clearer than changing a shared default. The [Model cookbook](model_cookbook.md#prior-customization) shows how. A linear light profile's intensity is solved during the fit and does not have a sampled intensity prior.

## Keep samples for later analysis

In `config/output.yaml`:

```yaml
samples: true
samples_weight_threshold: null
```

This retains sampled parameter values, likelihoods and weights, without filtering low-weight samples from the saved file. It can produce larger files, but keeps the samples available for later posterior analysis. Changing this setting does not change the sampler's inference.

<span id="setup"></span>

## Set configuration and output paths

To use a project-specific configuration directory:

```python
from autonerves import conf

conf.instance.push(
    new_path="path/to/config",
    output_path="path/to/output",
)
```

Replace these paths with your directories. The configuration directory must exist and contain configuration files. Relative paths are resolved from your working directory. Push the path before creating models or searches that read those settings.

## Set search options in Python

Individual search defaults are defined by PyAutoFit. Pass options explicitly when creating a search:

```python
import autofit as af

search = af.Nautilus(name="example", n_live=3000)
```

`n_live` controls the number of live points. Its appropriate value depends on the model and the convergence checks for your analysis. The workspace's `non_linear/` folder contains overrides such as grid-search settings; it is not a complete list of sampler options.

## Next steps

Use the [Model cookbook](model_cookbook.md) to customize an individual model, or return to your workspace example and check the figures and files produced by a short run.
