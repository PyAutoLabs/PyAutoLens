(overview-2-new-user-guide)=

# New User Guide

This guide helps you choose a **PyAutoLens example notebook** for your observations. Start with the scale of your lens, then choose your data type. Each example includes Python code you can read, run and adapt.

If you use AI, describe your lens and observations to the [PyAutoLens AI Assistant](https://github.com/PyAutoLabs/autolens_assistant). It can help you choose an example and work through it interactively.

(what-scale-lens)=

## What scale lens?

Choose the example that matches the galaxies contributing to the lensing. The distinction is the mass model and fitting approach, not a strict galaxy-count threshold.

![Four conceptual lensing scenes: one dominant galaxy, a comparable-mass galaxy pair, a small group and a rich cluster. Gold marks foreground galaxies and blue marks lensed background light. Indicative Einstein radii: galaxy approximately 1 arcsecond; multi-galaxy 1–3 arcseconds; group 3–10 arcseconds; cluster 10–50 arcseconds.](images/overview_2/lens_scales.png)

*AI-generated conceptual illustration, not observations or a calculated lens model. Panels are not shown at the same angular scale.*

::::{grid} 1 2 2 4
:gutter: 3


:::{grid-item-card} Galaxy
Indicative θ_E_: ~1″

One galaxy dominates the lensing. Start with your data type below.

[Choose your data type below](#what-dataset-type)
:::

:::{grid-item-card} Multi-galaxy
Indicative θ_E_: ~1–3″

Several galaxies contribute significantly to the lensing, without a dominant shared halo. Model their light and mass together and reconstruct an extended background source.

[Notebook ](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/start_here.ipynb)·[Google Colab ](https://colab.research.google.com/github/PyAutoLabs/autolens_workspace/blob/2026.11.12.1/notebooks/multi_galaxy/start_here.ipynb)
:::

:::{grid-item-card} Group
Indicative θ_E_: ~3–10″

A group of galaxies lenses an extended background source. The example introduces an optional shared dark-matter halo and ways to model fainter members.

[Notebook ](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/start_here.ipynb)·[Google Colab ](https://colab.research.google.com/github/PyAutoLabs/autolens_workspace/blob/2026.11.12.1/notebooks/group/start_here.ipynb)
:::

:::{grid-item-card} Cluster
Indicative θ_E_: ~10–50″

Many member galaxies lens multiple background sources at different redshifts. This example fits their multiple-image positions; it does not fit the lens galaxies’ light.

[Notebook ](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/cluster/start_here.ipynb)·[Google Colab ](https://colab.research.google.com/github/PyAutoLabs/autolens_workspace/blob/2026.11.12.1/notebooks/cluster/start_here.ipynb)
:::

::::

**Einstein radius (θE)** is a characteristic angular scale of strong lensing, approximately the ring radius for an aligned source and a symmetric lens. These values are rough guides in arcseconds (″), not category boundaries; they depend on the mass distribution and lens/source distances. Complex lenses can have several critical curves.

<details class="sources"><summary>About these scale estimates</summary><p>These are illustrative ranges, not survey limits or predicted radii for the artwork. Multi-galaxy systems overlap galaxy and group scales. The workspace’s lensing guide uses order-one arcsecond galaxy lenses, few-arcsecond multi-galaxy/group lenses and tens-of-arcseconds cluster lenses.</p><p>Examples grounding the scale: <a href="https://www.aanda.org/articles/aa/pdf/2025/05/aa51868-24.pdf">Euclid galaxy-scale lenses</a>; <a href="https://doi.org/10.1051/0004-6361/200811473">SL2S groups with 3–8″ Einstein radii</a>; <a href="https://doi.org/10.1051/0004-6361/202349060">a cluster lens with an effective Einstein radius around 43″</a>. For a complex lens, an effective Einstein radius is often defined from the area enclosed by its tangential critical curve for a specified source redshift.</p></details>

(what-dataset-type)=

## What dataset type?

For galaxy-scale lenses, choose the example for your observations. For multi-galaxy, group or cluster systems, begin with the scale-specific guide above.

![Five conceptual dataset representations: a pixelated lensed image, samples in the interferometer u–v plane, the same lens in three wavebands, stacked spectral-channel images, and four compact quasar images around a lens galaxy.](images/overview_2/dataset_types.png)

*AI-generated schematic, not measured data. The interferometer panel shows sampling locations; multi-band images show separate wavebands, while cube slices represent spectral channels.*

::::{grid} 1 2 2 2
:gutter: 3


:::{grid-item-card} Imaging
Images from telescopes such as Hubble or JWST. Fit the lens light, mass and extended source.

[Notebook ](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/start_here.ipynb)·[Google Colab ](https://colab.research.google.com/github/PyAutoLabs/autolens_workspace/blob/2026.11.12.1/notebooks/imaging/start_here.ipynb)
:::

:::{grid-item-card} Interferometer
Radio or submillimetre observations, such as ALMA data. Fit the measured visibilities directly.

[Notebook ](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/start_here.ipynb)·[Google Colab ](https://colab.research.google.com/github/PyAutoLabs/autolens_workspace/blob/2026.11.12.1/notebooks/interferometer/start_here.ipynb)
:::

:::{grid-item-card} Multi-band imaging
The same lens observed in several wavebands. Fit the images together with a shared lens model using `multi_dataset/`.

[Notebook ](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/start_here.ipynb)·[Google Colab ](https://colab.research.google.com/github/PyAutoLabs/autolens_workspace/blob/2026.11.12.1/notebooks/multi_dataset/start_here.ipynb)
:::

:::{grid-item-card} Data cubes
Spectral-line channels, such as an ALMA CO cube. Fit the channels together with a shared lens model.

[Notebook ](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/datacube/start_here.ipynb)·[Google Colab ](https://colab.research.google.com/github/PyAutoLabs/autolens_workspace/blob/2026.11.12.1/notebooks/interferometer/features/datacube/start_here.ipynb)
:::

:::{grid-item-card} Point sources
Multiple images of a compact source, such as a quasar or supernova. Start with the point-source example.

[Notebook ](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/point_source/start_here.ipynb)·[Google Colab ](https://colab.research.google.com/github/PyAutoLabs/autolens_workspace/blob/2026.11.12.1/notebooks/point_source/start_here.ipynb)
:::

::::

(google-colab)=

Read each notebook on GitHub, open it locally, or use its Google Colab link to run it in your browser. In Colab, run the setup cell first to install the software and prepare the example.

(weak-lensing)=

## Weak lensing

To model the subtle distortions of many background galaxies, start with [weak/](https://github.com/PyAutoLabs/autolens_workspace/tree/main/scripts/weak). You can also combine weak and strong lensing to study a cluster across both scales.

![Strong-lensing arcs near a cluster centre and weakly distorted background galaxies across a wider field.](images/overview_3/strong_weak_lensing.png)

*AI-generated conceptual illustration. The panels cover different angular scales; dashed contours represent a mass model, not a direct measurement from individual galaxy shapes.*

[Notebook ](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/weak/start_here.ipynb)·[Google Colab ](https://colab.research.google.com/github/PyAutoLabs/autolens_workspace/blob/2026.11.12.1/notebooks/weak/start_here.ipynb)

## Still unsure?

Tell the [AI Assistant](https://github.com/PyAutoLabs/autolens_assistant) what objects are in your field and what observations you have. It can help you identify a starting example.

(howtolens)=

## Learn the science

[HowToLens](https://github.com/PyAutoLabs/HowToLens) teaches gravitational lensing and Bayesian modelling through worked notebooks. Use it for a slower explanation of the concepts behind these examples.

(wrap-up)=

## Next steps

Once you have chosen an example, explore the [Features guide](https://pyautolens.readthedocs.io/en/latest/overview/overview_3_features.html) to find the modelling tools relevant to your science.
