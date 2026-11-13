```{image} https://github.com/Jammy2211/PyAutoLogo/blob/main/gifs/pyautolens.gif?raw=true
:width: 900
```

# What is PyAutoLens?

When two or more galaxies are aligned perfectly down our line-of-sight, the background galaxy appears multiple times.

This is called strong gravitational lensing. **PyAutoLens** models strong gravitational lenses to measure the distribution of matter and reconstruct distant galaxies. Its JAX numerical backend provides automatic differentiation and GPU acceleration, while the PyAutoLens Assistant helps you develop and run analyses through conversation.

# Getting Started

## Explore It Yourself

- **[Start Here](https://pyautolens.readthedocs.io/en/latest/overview/overview_1_start_here.html)** — discover PyAutoLens and work through your first example.
- **[Try it in your browser](https://colab.research.google.com/github/PyAutoLabs/autolens_workspace/blob/2026.11.13.1/start_here.ipynb)** — run the introductory notebook without installing anything.
- **[Install PyAutoLens](https://pyautolens.readthedocs.io/en/latest/installation/overview.html)** — get set up on your computer.

Explore more examples in the [workspace](https://github.com/PyAutoLabs/autolens_workspace), or learn lensing from the ground up with [HowToLens](https://github.com/PyAutoLabs/HowToLens).

## Explore with AI

The [**PyAutoLens AI Assistant**](https://github.com/PyAutoLabs/autolens_assistant) lets you explore
gravitational lensing through conversation. Ask a question, explore an idea, or describe the analysis
you want to run. Both routes use the same software. To get started, paste this prompt into Claude Code, Codex
or another AI coding agent:

:::{container} nl-prompt
> I want to do strong lensing science with PyAutoLens (https://github.com/PyAutoLabs/PyAutoLens) and the
> autolens_assistant (https://github.com/PyAutoLabs/autolens_assistant). First clone that repository,
> cd into it and follow its AGENTS.md.
>
> Begin the "start here" guide for a new user.
:::

The assistant guides you through [Start Here](https://pyautolens.readthedocs.io/en/latest/overview/overview_1_start_here.html)
as a conversation, then helps you analyse your own lenses. See [AI Options](https://github.com/PyAutoLabs/PyAutoScientist/blob/main/AI_OPTIONS.md)
for recommended tools, costs and free alternatives.

# Community & Contributing

**PyAutoLens** is built in the open by its users: everyone is welcome to ask questions,
share what they have made with it, and contribute.

Questions, ideas and bug reports: the [PyAutoLabs Discussions](https://github.com/orgs/PyAutoLabs/discussions).
Chat with us on [Slack](https://join.slack.com/t/pyautolens/shared_invite/zt-2cufp4eyf-fXfgMxRGuvg~bMrI3uOAxg).

Community-built tools and tutorials, and how to contribute: the [**PyAutoLens** community page](https://pyautolens.readthedocs.io/en/latest/general/community.html).

# HowToLens

For users less familiar with gravitational lensing, Bayesian inference and scientific analysis
you may wish to read through the **HowToLens** lectures. These teach you the basic principles of gravitational lensing
and Bayesian inference, with the content pitched at undergraduate level and above.

A complete overview of the lectures [is provided on the HowToLens readthedocs page](https://pyautolens.readthedocs.io/en/latest/howtolens/howtolens.html), and the notebooks themselves live in the [PyAutoLabs/HowToLens](https://github.com/PyAutoLabs/HowToLens) repository.

# Citations

Information on how to cite **PyAutoLens** in publications can be found [on the citations page](https://github.com/PyAutoLabs/PyAutoLens/blob/main/CITATIONS.md).

```{toctree}
:caption: 'Overview:'
:hidden: true
:maxdepth: 1

overview/overview_1_start_here
overview/overview_2_new_user_guide
overview/overview_3_features
```

```{toctree}
:caption: 'Installation:'
:hidden: true
:maxdepth: 1

installation/overview
installation/conda
installation/pip
installation/workspace
installation/acceleration
installation/source
installation/troubleshooting
```

```{toctree}
:caption: 'General:'
:hidden: true
:maxdepth: 1

general/workspace
general/configs
general/model_cookbook
general/likelihood_function
general/demagnified_solutions
general/citations
general/papers
general/credits
general/community
```

```{toctree}
:caption: 'Tutorials:'
:hidden: true
:maxdepth: 1

howtolens/howtolens
```

```{toctree}
:caption: 'API Reference:'
:hidden: true
:maxdepth: 1

api/data
api/light
api/mass
api/galaxy
api/fitting
api/modeling
api/pixelization
api/point
api/plot
api/potential_correction
api/twins
api/source
```
