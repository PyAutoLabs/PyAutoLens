(overview)=

<span id="id1"></span>

# Install PyAutoLens

Use Python 3.12 or later in a separate environment. The guides below use Python 3.12.

| Your setup | Start here |
| --- | --- |
| You use conda, or need it to manage Python | [Install with conda](conda.md) |
| You already have Python installed | [Install with pip](pip.md) |
| You want to edit the library code | [Install from source](source.md) |

Both the conda and pip routes install `autolens` from PyPI. A GPU is optional; start with the standard installation, or see [GPU setup](acceleration.md) if you plan to use one.

Prefer to try the software without installing it? Open an example in [Google Colab](https://pyautolens.readthedocs.io/en/latest/overview/overview_2_new_user_guide.html).

<details>
<summary>Install with help from an AI agent</summary>

Paste this into Claude Code, Codex or another AI coding agent:

> Help me install PyAutoLens and set up autolens_workspace. Use the current installation guide at https://pyautolens.readthedocs.io/en/latest/installation/overview.html. Check my operating system and Python version, create an isolated environment, verify the installation and help me run the first example. Ask before changing an existing environment or installing system software.

The [PyAutoLens AI Assistant](https://github.com/PyAutoLabs/autolens_assistant) can help with the examples once you are set up.

</details>

After installing, [run your first example](workspace.md).

## Dependencies

The installer resolves the required packages automatically. See [optional packages](acceleration.md) for additions needed by particular workflows.

## JAX & GPU

See [JAX and GPU setup](acceleration.md#jax-and-gpus).
