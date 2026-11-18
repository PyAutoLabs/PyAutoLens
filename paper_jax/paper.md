---
title: "PyAutoLens II: Automated, Autodifferentiable and Autonomous Gravitational Lens Modelling"
tags:
  - Python
  - astronomy
  - gravitational lensing
  - automatic differentiation
  - GPU acceleration
authors:
  - name: James W. Nightingale
    orcid: 0000-0002-8987-7401
    affiliation: 1
    corresponding: true
affiliations:
  - name: Institute for Computational Cosmology, Durham University, United Kingdom
    index: 1
date: 14 July 2026
bibliography: paper.bib
---

# Summary

Gravitational lensing occurs when the gravitational field of intervening matter bends light from a distant
source. It can produce multiple images and extended arcs around individual galaxies, groups and clusters,
as well as weaker distortions in the shapes of background galaxies. These effects allow us to measure the
distribution of luminous and dark matter, reconstruct distant sources and constrain cosmological parameters.
Lensing datasets are growing rapidly in size: Stage IV surveys such as Euclid [@EuclidCollaboration2025] and the Vera C. Rubin
Observatory [@LSSTDarkEnergyScienceCollaboration2012] will measure billions of galaxies across large fractions of the 
sky. They are also growing in diversity: a single system may include strong- and weak-lensing constraints, multi-band 
optical and infrared imaging, radio-interferometric visibilities, and point-source measurements of lensed quasars or
supernovae. Fully exploiting these observations requires joint probabilistic modelling across lensing scales and data types. 
However, the increasing volume of data and complexity of lens models make these analyses increasingly computationally expensive.

PyAutoLens II brings together “the three autos”—automated modelling, automatic differentiation and autonomous
workflows—within a unified framework for gravitational lens modelling. Automated modelling extends the
established PyAutoLens approach to galaxy-, group- and cluster-scale systems, combining imaging,
interferometric visibilities, point-source observables and weak-lensing measurements within reproducible
analysis pipelines. Automatic differentiation, implemented through JAX alongside just-in-time compilation and
GPU acceleration, supports efficient likelihood evaluation and gradient-based inference for increasingly
complex models. Autonomous workflows, guided through the [PyAutoLens AI
assistant](https://github.com/PyAutoLabs/autolens_assistant), help scientists construct analyses, plan
computational campaigns and interpret their outputs, connecting individual model fits to the management of
complete lens samples. Together, the three autos support increasingly scalable and autonomous analyses while
scientists retain control over physical assumptions and scientific interpretation.

# Statement of need

Modern gravitational-lensing datasets offer the opportunity to study luminous and dark matter through richer
models of individual systems and statistical analyses of large lens samples. Strong lenses observed across
optical, infrared and submillimetre wavelengths can reveal complementary views of the source and lens mass
distribution, but exploiting these observations requires joint multiwavelength modelling. Group- and
cluster-scale systems offer constraints across a wider range of physical scales, yet incorporating many
deflecting galaxies alongside both strong- and weak-lensing measurements introduces substantial model
complexity. Lensed quasars and supernovae provide opportunities to measure cosmological distances, with
transient systems also requiring rapid mass modelling and time-delay predictions to guide follow-up
observations. Euclid, Rubin and other wide-field surveys will expand these opportunities to more than 100,000
galaxy-scale lenses and thousands of group- and cluster-scale systems [@Collett2015; @Bergamini2025].
Realising this potential requires overcoming both the computational cost of inference and the human effort
needed to construct, execute and assess increasingly complex analyses across large samples.

PyAutoLens II provides the solution through three complementary capabilities: the three autos. **Automated
modelling** builds on the original PyAutoLens framework, extending its approach from galaxy-scale strong
lenses to groups, clusters and weak lensing, with pipelines that combine diverse observations and carry
information between successive model fits. **Automatic differentiation** enables gradient-based optimisation
and sampling, while JAX compilation and GPU acceleration reduce the computational cost of evaluating complex
models. **Autonomous workflows**, supported by the [PyAutoLens AI
assistant](https://github.com/PyAutoLabs/autolens_assistant), help scientists develop bespoke analyses, plan
and manage computational campaigns, inspect model fits and assemble reproducible results libraries. These
capabilities are connected through a composable Python API: users construct galaxies from light and mass
profiles, arrange them across lens planes and combine the likelihoods of their chosen datasets into a single
probabilistic model. For example, a cluster model can jointly fit extended arcs, multiple-image positions and
a weak-lensing catalogue, while a galaxy-scale model can combine optical imaging with interferometric
observations of the same source. This allows scientists to build bespoke likelihood functions around the
observations and physical complexity of each lensing system, then apply the same tools for automation,
differentiation and autonomous execution to the resulting analysis.

# Case studies

We develop three case studies that connect the three autos to scientific analyses: combining
complementary views of one galaxy-scale lens, modelling a complex cluster, and turning a sample of
survey lenses into a reproducible catalogue. The construction of the models, likelihoods and inference
workflows is described within each example.

<!-- DRAFT STATUS: These are planned analyses, not completed scientific results.
Replace prospective descriptions with measured outcomes only after validation.
Retain versioned data, configurations and run records behind every figure and performance claim.
-->

## One lens, two views: joint JWST and ALMA modelling

JWST imaging and ALMA interferometry reveal different components of a distant galaxy. A joint lens
model can use both views to constrain the foreground mass distribution while reconstructing the
near-infrared and submillimetre source emission independently. Our proposed example is SPT0418−47,
a strong lens observed by both facilities.

The analysis will combine an imaging likelihood, including the JWST point-spread function and lens
light, with a likelihood evaluated directly from ALMA visibilities. The two likelihoods will share
lens mass parameters while retaining separate source models and accounting for relative astrometric
alignment. This provides a concrete introduction to the composable API: each observation contributes
its own likelihood to one physical lens model.

We will use staged Source-Light-Mass (SLaM) modelling to initialise and refine the analysis, then
compare Nautilus posterior sampling with a gradient-based fit of the same final model. The example
will explain how automation carries information between modelling stages and how automatic
differentiation supports the search for a solution. Optimisation and posterior sampling will be
reported separately, with fit quality and uncertainty checks supporting the interpretation.

**Figure placeholder:** JWST imaging alongside the ALMA continuum view, the corresponding model
residuals, and independently reconstructed near-infrared and submillimetre source emission. ALMA
image-plane panels are visualisations; the fit uses the visibilities.

<!-- TO COMPLETE
Verify science/error/PSF products, calibrated visibilities and weights; record archive identifiers,
units, redshifts, masks, coordinate conventions and astrometric offsets. Include the source companion
or document its masking. Select and validate the SLaM stages and gradient method. Report measured
runtime in the narrative, linked to autolens_profiling / autolens_inference evidence where applicable.
Target choice remains provisional until the data audit is complete.
-->

## A complex cluster: strong lensing in MACS J0416

Cluster lenses combine large-scale dark matter haloes with many member galaxies and background
sources at different redshifts. MACS J0416, a Hubble Frontier Fields cluster with a published catalogue
of 237 spectroscopically confirmed multiple images, is our proposed demanding example.

We will construct a mass model from cluster haloes and member galaxies, using scaling relations to
connect member masses to their observed properties. Multiple-image positions will constrain the
shared mass distribution through source-plane and image-plane chi-squared analyses. A source-plane
fit can provide an initial solution for image-plane refinement; these are alternative treatments of
the same positional measurements and will not be summed as independent constraints.

The case study will explain how the API assembles many mass components and source systems, and test
whether GPU execution and gradients make a complex analysis practical on short timescales. Success
requires an acceptable predicted image configuration and stable inference, rather than runtime alone.

**Figure placeholder:** The cluster field with observed and predicted image positions, critical
curves for specified source redshifts, and selected reconstructed sources or magnification maps.

<!-- TO COMPLETE
Cite Bergamini et al. (2023), DOI 10.1051/0004-6361/202244834, and the selected catalogue version.
Audit member photometry, image families, redshifts, positional errors, priors and mass components.
Validate source-plane approximation, image matching and gradients. Distinguish cold-start and
initialised runs, optimisation and posterior sampling. Under one hour on GPU is a target to test,
not a result or a requirement that justifies simplifying the model without disclosure.
-->

## From a short instruction to a Euclid lens catalogue

The third case study will follow an AI-assisted analysis of 100 high-confidence lenses selected
from the public Euclid Q1 discovery catalogue. Starting with a short scientific brief, a fixed sample
and explicit modelling choices, the assistant will prepare and operate a campaign whose outputs
form a traceable lens-model catalogue.

The narrative will follow the analysis from dataset preparation through automated modelling,
resource planning, execution, fit inspection and results assembly. It will show the scientist's
inputs alongside the actions completed by the assistant, including requests for intervention,
retries and unsuccessful fits. The resulting catalogue will connect inferred parameters and
uncertainties to diagnostic images, input data and run provenance.

**Figure placeholder:** A mosaic of observed lenses and reconstructed sources, with a compact
illustration of the progression from scientific brief to catalogue. Retain all selected systems
in the outcome accounting, including those without an accepted fit.

<!-- GATED CASE STUDY
Freeze 100 catalogue IDs using published grades and explicit pre-fit quality cuts; this is a
selected demonstration sample, not an unbiased population sample. Verify science images, PSFs and
noise maps. Run a small pilot before the full campaign. Full execution depends on validated many-lens
HPC planning, recovery and results-library functionality. Record assistant/model versions, prompts,
human interventions, completion criteria and compute use. Do not describe the workflow as demonstrated
until end-to-end evidence exists. Public catalogue: https://zenodo.org/records/15025832.
-->

# Workspace and HowToLens tutorials

The [PyAutoLens workspace](https://github.com/PyAutoLabs/autolens_workspace) provides runnable examples
of model construction, dataset likelihoods and inference workflows. Readers can use these examples
to develop analyses tailored to their observations, progressing from individual datasets to joint
models and automated pipelines.

[HowToLens](https://github.com/PyAutoLabs/HowToLens) connects these tools to the underlying lensing
physics and Bayesian inference. The new Chapter 4, *Scaling Up Lensing*, extends the tutorials beyond
an isolated lens galaxy to multiple deflectors, scaling relations, group and cluster lenses, and weak
lensing. These tutorials explain how model complexity and observational constraints change across
physical scales, and how to assess the resulting fits and posterior uncertainties.

<!-- PLANNED TEACHING EXTENSION — NOT YET VERIFIED AS IMPLEMENTED
Introduce gradients in the context of Bayesian inference: differentiating a likelihood/posterior,
using gradients for optimisation and sampling, and distinguishing a MAP estimate from posterior
uncertainty. Current Chapter 4 scripts use Nautilus and do not yet provide this gradient teaching.
Link each completed paper case study to a runnable workspace example and the relevant tutorials.
Do not claim the case-study examples are published in the user workspace before their release.
-->

# Research impact statement

<!--
Give specific evidence of research enabled by PyAutoLens: published applications,
external adoption and survey-scale analyses. Use existing large-sample studies to
establish the automation foundation, then connect the additional capabilities of
PyAutoLens II to supported scientific applications. Distinguish demonstrated impact
from prospective uses; avoid repeating the benchmark results.
-->

# AI usage disclosure

Generative AI tools were used to scaffold this manuscript template and may be used to assist drafting. All scientific claims, citations, and prose are reviewed and verified by the authors.

# Acknowledgements

<!-- List funding, facilities, software contributors, and other support. -->

# References
