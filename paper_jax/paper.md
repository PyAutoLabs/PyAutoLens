---
title: "PyAutoLens-JAX: Differentiable GPU-accelerated strong and weak lensing from galaxies to clusters"
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

Gravitational lensing probes luminous and dark matter in galaxy-, group-, and cluster-scale systems. Lensing datasets 
are growing rapidly in size: Stage IV surveys such as Euclid [@EuclidCollaboration2025] and the Vera C. Rubin 
Observatory [@LSSTDarkEnergyScienceCollaboration2012] will measure billions of galaxies across large fractions of the 
sky. They are also growing in diversity: a single system may include strong- and weak-lensing constraints, multi-band 
optical and infrared imaging, radio-interferometric visibilities, and point-source measurements of lensed quasars or
supernovae. Fully exploiting these observations requires joint probabilistic modelling across lensing scales and data types. 
However, the increasing volume of data and complexity of lens models make these analyses increasingly computationally expensive.

PyAutoLens is now implemented using JAX throughout its core modelling framework, providing just-in-time compilation, 
GPU acceleration, and automatic differentiation without introducing a separate package or replacing its established 
object-oriented API. Galaxy-, group-, and cluster-scale lens models can be constrained using CCD imaging, 
interferometer visibilities, point-source observables, and weak-lensing catalogues fully in JAX. Crucially, 
these are not isolated capabilities: users can combine multiple datasets and strong- and weak-lensing constraints 
within a single differentiable, GPU-accelerated probabilistic model. In doing so, PyAutoLens-JAX allows 
gravitational-lensing analyses to scale with the size and complexity of next-generation datasets.

# Statement of need

Gravitational lensing analysis of modern lensing datasets is limited by computational run times and an ability to fix
more complex models with more parameters. For example, strong lens are often observed across multiple optical and 
submm wavelengths, however joint multiwavelength modeling is rarely performed. The most complex lensing clusters, 
such as the Hubble Frontier Fields, requires months of CPU time to analyse, even though the majority of galaxies are
tired to some form of scaling relatiojn. This resticts model complexity and makes inclusion of complementary data, notably
weak lensing catalogues unfeasibles. Measuring the Hubble constant via time delay quasars also takes thousands of human hours partly driven by
computational overheads, thus studies of lensed supernovae, which require rapid mass models and time-delay 
predictions to guide time-sensitive follow-up [@Peng2023; @Lange2025; @Schaefer2020], are also infeasible.
Euclid, Rubin and other wide-field surveys are poised to discover more than 100,000 galaxy-scale lenses and thousands of 
group- and cluster-scale systems [@Collett2015; @Bergamini2025]. This influx of lensing data combined with a critical assessment of
existing gravitational lensing studies shows new software and approaches are required to fully scale-up and exploit
the data for its fully scientific potential.

PyAutoLens-JAX provides the solution. 
extends the established automation of PyAutoLens beyond this computational boundary by making its 
complete modelling framework compatible with just-in-time compilation, GPU execution, and automatic differentiation. 
The same accelerated framework supports galaxy-, group-, and cluster-scale models constrained by imaging, 
interferometric visibilities, point-source observables, and weak-lensing catalogues, including joint analyses across 
these data types. Faster likelihood evaluation makes richer models and larger samples practical, while automatic 
differentiation enables gradient-based optimisation and sampling methods that can scale to significant more free parameters. 
PyAutoLens-JAX therefore provides the computational foundation required to combine the richest available datasets, 
model next-generation lens samples at scale, accelerate complex cluster analyses, and deliver rapid inference for time-critical transient lensing.

The pre-JAX implementation of PyAutoLens has already demonstrated that automated lens modelling can scale to large 
samples. COWLS I modelled 419 JWST-selected candidates across four NIRCam bands, while the Euclid Q1 analysis 
successfully modelled more than 300 additional systems [@Nightingale2025; @Lines2025]. Existing lens analysis is therefore 
prohibited by analysis run time and complexity, massive speed up is required for the orders of magnitude increase of lens 
numbers now being found. 



This rapid advance in lens analysis run time is paired with PyAutoLens-Assistant, which allows a scientist to describe
lens modeling using natural language, such that agentic AI then performs it. In doing so, this makes performing bespoke
and complex lens modeling of individual lenses or large lens samples feasibles. PyAutoLens-JAX is therefore
vital in making the actual computartional run times paired with this make the science possible. 

# State of the field

<!--
Position PyAutoLens-JAX relative to the published PyAutoLens software
[@Nightingale2021] and other current strong- and weak-lensing packages. Focus on
the combination of established modelling abstractions, joint multi-dataset
inference, automatic differentiation, and GPU execution.
-->

# Software design

## Differentiable modelling

<!--
Describe the JAX-compatible numerical backend, just-in-time compilation,
automatic differentiation, and preservation of the public PyAutoLens API. Cite
JAX [@jax2018github] and explain which model components are differentiable.
-->

## Joint datasets and lensing regimes

<!--
Describe how analyses combine imaging, interferometer, point-source, and weak-
lensing observations across lens planes and physical scales in one likelihood.
Include a concise representative example or schematic.
-->

## End-to-end modelling benchmarks

All benchmarks use gradient-based inference with JAX automatic differentiation and are executed on an NVIDIA A100 GPU. Unless otherwise stated, the lens model comprises a singular isothermal ellipsoid with external shear, a multi-Gaussian expansion for the lens light, and a pixelized reconstruction of the lensed source. For every analysis, we report the total wall-clock time, JAX compilation time, number of likelihood evaluations, and post-compilation runtime.

### Single-dataset and single-regime benchmarks

These benchmarks establish GPU acceleration and automatic differentiation across the individual lensing scales and observational data types supported by PyAutoLens-JAX.

- **Galaxy-scale CCD imaging:** Model JWST COSMOS-Web Ring F150W imaging, including lens-light subtraction and a pixelized source reconstruction, in approximately five minutes.
- **Interferometry:** Model a real ALMA strong-lensing dataset containing more than one million interferometer visibilities in approximately five minutes.
- **Point-source lensing:** Model a real multiply imaged quasar or supernova using point-source observables, including image positions and, where available, time delays or flux information, in under five minutes.
- **Group-scale strong lensing:** Model a real group-scale lens containing multiple deflecting galaxies in under five minutes, demonstrating that PyAutoLens-JAX is not restricted to isolated galaxy-scale lenses.
- **Cluster-scale strong lensing:** Model a real cluster lens with multiple mass components, multiple images, and potentially multiple source planes in under five minutes.
- **Weak lensing:** Fit a weak-lensing shear catalogue using a differentiable JAX likelihood in under five minutes, demonstrating that PyAutoLens-JAX is not restricted to strong-lensing data.

### Joint and multi-dataset benchmarks

These benchmarks demonstrate the central capability of PyAutoLens-JAX: different datasets, lensing regimes, and physical scales can be combined within a single differentiable, GPU-accelerated probabilistic model.

- **Multi-band imaging:** Jointly model the four available JWST COSMOS-Web Ring bands, constraining a common lens mass model while fitting the wavelength-dependent lens and source emission in each dataset.
- **Joint strong and weak lensing:** Constrain a single group- or cluster-scale mass model using both strong-lensing and weak-lensing observables.
- **Imaging and point-source lensing:** Jointly model extended arcs and point-source constraints from a lensed quasar or supernova within the same lens model.
- **Imaging and interferometry:** Jointly fit optical or infrared imaging and radio or submillimetre interferometer visibilities, constraining a common mass model using complementary observations of the lensed source.

The single examples verify that each major PyAutoLens likelihood and modelling regime is JAX compatible. The combined examples then demonstrate that these capabilities are not implemented as isolated workflows: they can be composed while retaining a common object-oriented API, automatic differentiation, and GPU execution.

# Research impact statement

<!--
Give specific evidence of research enabled by PyAutoLens and the JAX framework:
published or ongoing projects, external adoption, survey-scale applications,
and reproducible examples. Keep this about software impact rather than new
scientific results.
-->

# AI usage disclosure

Generative AI tools were used to scaffold this manuscript template and may be used to assist drafting. All scientific claims, citations, and prose are reviewed and verified by the authors.

# Acknowledgements

<!-- List funding, facilities, software contributors, and other support. -->

# References
