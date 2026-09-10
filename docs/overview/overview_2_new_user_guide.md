(overview-2-new-user-guide)=

# New User Guide

## Human-Readable Guide

**PyAutoLens** can analyse strong lens systems across a range of physical scales (e.g. galaxy, multi-galaxy, group, and cluster) and for
different types of data (e.g. imaging, interferometer, and point-source observations). Depending on the scientific questions you are interested in, the analysis you perform may differ significantly.

The autolens_workspace contains a suite of example Jupyter Notebooks, organised by lens scale and dataset type.

This guide begins with two simple questions to help you find the most appropriate example notebook for your science case.

## PyAutoLens AI Assistant

The [**PyAutoLens AI Assistant**](https://github.com/PyAutoLabs/autolens_assistant) lets you do gravitational lensing science in natural language from inside an AI coding agent. You can get started simply by asking it a question about gravitational lensing or describing the task you would like to perform with **PyAutoLens**. See the [autolens_assistant GitHub page](https://github.com/PyAutoLabs/autolens_assistant) for its full scope and instructions.

**The assistant runs inside an AI coding agent: Claude Code or Codex are recommended. Sustained scientific use normally needs paid access to one of them (a personal subscription, institutional access or API billing). OpenCode is an experimental alternative whose client is free but whose model access, cost and capability depend on the provider. Browser chat routes (ChatGPT or Claude with a GitHub connector) are no longer supported.**

## What Scale Lens?

What size and scale of strong lens system are you expecting to work with?

There are four scales to choose from, which form a ladder: every group and cluster is a multi-galaxy system,
but not vice versa. What changes as you climb is first the mass model, then the entire analysis strategy:

- **Galaxy Scale**: Made up of a single lens galaxy lensing a single source galaxy, the simplest strong lens you can get!
  If you're interested in galaxy scale lenses, go to the question below called "What Dataset Type?".
- **Multi Galaxy**: Two or more galaxies of comparable mass all contribute significantly to the lensing, with no
  dominant shared dark-matter halo (systems with one belong on the group or cluster rungs below) — each co-dominant
  deflector gets its own free light and mass model, and the source is a single
  extended galaxy fitted at pixel level exactly as at galaxy scale. If this is your system, go to the
  [multi_galaxy/start_here.ipynb](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/start_here.ipynb) notebook.
- **Group Scale**: Strong lens groups contain 2-10 lens galaxies and typically lens just one source galaxy. A dominant
  group-scale dark-matter halo may enter the model as an explicit choice, with fainter members organised into tiers
  (scaling relations, tidally truncated profiles). The source modelling is unchanged — one extended source fitted at
  pixel level. If you are interested in groups, go to the [group/start_here.ipynb](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/start_here.ipynb) notebook.
- **Cluster Scale**: Strong lens galaxy clusters often contain 20-50, or more, lens galaxies, lensing 10, or more, source
  galaxies across a wide range of redshifts. The mass framework is the same as a group's, but the **analysis changes**:
  the default workflow fits point-source multiple-image positions (multi-plane, per-source redshifts) rather than
  reconstructing extended sources, and the lens galaxies' light is not modeled. If you are interested in clusters, go
  to the [cluster/start_here.ipynb](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/cluster/start_here.ipynb) notebook.

For a sense of the real science each rung anchors to: multi-galaxy lenses include merging-pair systems like
SDSS J1011+0143 (two SIEs + shear; Shu et al. 2016) and the famous time-delay lens B1608+656; group-scale
lenses include CSWA 19 (Ding et al. 2025, modeled with **PyAutoLens**) and the SL2S group sample; cluster-scale
lenses include the Hubble Frontier Fields clusters, most notably Abell 2744 (Bergamini et al. 2023) — the
system the workspace's cluster `start_here` models.

## What Dataset Type?

If you are interested in galaxy-scale strong lenses, you now need to decide what type of strong lens data you are
interested in:

- **CDD Imaging**: For image data from telescopes like Hubble and James Webb, go to [imaging/start_here.ipynb](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/start_here.ipynb).
- **Interferometer**: For radio / sub-mm interferometer from instruments like ALMA, go to [interferometer/start_here.ipynb](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/start_here.ipynb).
- **Data Cubes**: For spectral-line data cubes (e.g. ALMA CO cubes), where many frequency channels are fitted simultaneously with a shared lens model, go to [interferometer/features/datacube/start_here.ipynb](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/datacube/start_here.ipynb).
- **Point Sources**: For strongly lensed point sources (e.g. lensed quasars, supernovae), go to [point_source/start_here.ipynb](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/point_source/start_here.ipynb).

## Google Colab

The links above take you to the GitHub page of each notebook, and if you've cloned the workspace you can open them
locally on your machine.

However, you can also open and run each notebook directly in Google Colab, which provides a free cloud computing
environment with all the required dependencies already installed.

This is a great way to get started quickly without needing to install **PyAutoLens** on your own machine,
so you can check its the right software for you before going through the installation process:

> - [imaging/start_here.ipynb](https://colab.research.google.com/github/PyAutoLabs/autolens_workspace/blob/2026.9.8.1/notebooks/imaging/start_here.ipynb): Galaxy scale strong lenses observed with CCD imaging (e.g. Hubble, James Webb).
> - [interferometer/start_here.ipynb](https://colab.research.google.com/github/PyAutoLabs/autolens_workspace/blob/2026.9.8.1/notebooks/interferometer/start_here.ipynb): Galaxy scale strong lenses observed with interferometer data (e.g. ALMA).
> - [point_source/start_here.ipynb](https://colab.research.google.com/github/PyAutoLabs/autolens_workspace/blob/2026.9.8.1/notebooks/point_source/start_here.ipynb): Galaxy scale strong lenses with a lensed point source (e.g. lensed quasars).
> - [multi_galaxy/start_here.ipynb](https://colab.research.google.com/github/PyAutoLabs/autolens_workspace/blob/2026.9.8.1/notebooks/multi_galaxy/start_here.ipynb): Multi-galaxy strong lenses with 2+ co-dominant lens galaxies and no host halo.
> - [group/start_here.ipynb](https://colab.research.google.com/github/PyAutoLabs/autolens_workspace/blob/2026.9.8.1/notebooks/group/start_here.ipynb): Group scale strong lenses where there are 2-10 lens galaxies.
> - [cluster/start_here.ipynb](https://colab.research.google.com/github/PyAutoLabs/autolens_workspace/blob/2026.9.8.1/notebooks/cluster/start_here.ipynb): Cluster scale strong lenses with 20+ lens galaxies and 10+ source galaxies.

## Still Unsure?

Each notebook is short and self-contained, and can be completed and adapted quickly to your particular task.
Therefore, if you're unsure exactly which scale of lensing applies to you, or quite what data you want to use, you
should just read through a few different notebooks and go from there.

## HowToLens

For experienced scientists, the **PyAutoLens** examples will be simple to follow. Concepts surrounding strong lensing may
already be familiar and the statistical techniques used for fitting and modeling already understood.

For those less familiar with these concepts (e.g. undergraduate students, new PhD students or interested members of the
public), things may have been less clear and a slower more detailed explanation of each concept would be beneficial.

The **HowToLens** Jupyter Notebook lectures provide exactly this. They are a four-chapter guide which thoroughly
take you through the core concepts of strong lensing, teach you the principles of the statistical techniques
used in modeling and ultimately will allow you to undertake scientific research like a professional astronomer.

To complete thoroughly, they'll probably take 2-4 days, so you may want try moving ahead to the examples but can
go back to these lectures if you find them hard to follow.

If this sounds like it suits you, checkout the [HowToLens](https://github.com/PyAutoLabs/HowToLens) repository now.

## Wrap Up

After completing this guide, you should be able to use **PyAutoLens** for your science research.

The biggest decisions you'll need to make are what features and functionality your specific science case requires,
which the next readthedocs page gives an overview of to help you decide.
