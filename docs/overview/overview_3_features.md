(overview-3-features)=

# Features

This page provides an overview of the advanced features of **PyAutoLens**.

Firstly, brief one sentence descriptions of each feature are given, with more detailed descriptions below including
links to the relevant workspace examples.

**Pixelizations**: Reconstructing the source galaxy on a mesh of pixels, to capture extremely irregular structures like spiral arms.

**Point Sources**: Modeling point sources (e.g. quasars) observed in the strong lens imaging data.

**Interferometry**: Modeling of interferometer data (e.g. ALMA, LOFAR) directly in the uv-plane.

**Multi Gaussian Expansion (MGE)**: Decomposing the lens galaxy into hundreds of Gaussians, for a clean lens subtraction.

**Groups**: Modeling group-scale strong lenses with multiple lens galaxies and multiple source galaxies.

**Multi-Wavelength**: Simultaneous analysis of imaging and / or interferometer datasets observed at different wavelengths.

**Ellipse Fitting**: Fitting ellipses to determine a lens galaxy's ellipticity, position angle and centre.

**Shapelets**: Decomposing a galaxy into a set of shapelet orthogonal basis functions, capturing more complex structures than simple light profiles.

**Operated Light Profiles**: Assuming a light profile has already been convolved with the PSF, for when the PSF is a significant effect.

**Sky Background**: Including the background sky in the model to ensure robust fits to the outskirts of galaxies.

## Pixelizations

Pixelizations reconstruct the source galaxy's light on a pixel-grid. Unlike `LightProfile`'s, they are able to
reconstruct the light of non-symmetric, irregular and clumpy sources.

The image below shows a pixelized source reconstruction of the strong lens SLACS1430+4105, where the source is
reconstructed on a Voronoi mesh adapted to the source morphology, revealing it to be a grand-design face on spiral
galaxy:

```{image} https://github.com/PyAutoLabs/PyAutoLens/blob/main/files/imageaxis.png?raw=true
:alt: Alternative text
:width: 600
```

A complete overview of pixelized source reconstructions can be found
at `notebooks/overview/overview_5_pixelizations.ipynb`.

Chapter 4 of lectures describes pixelizations in detail and teaches users how they can be used to
perform lens modeling.

## Point Sources

There are many lenses where the background source is not extended but is instead a point-source, for example strongly
lensed quasars and supernovae.

For these objects, we do not want to model the source using a light profile, which implicitly assumes an extended
surface brightness distribution.

Instead, we assume that our source is a point source with a centre (y,x), and ray-trace triangles at iteratively
higher resolutions to determine the source's exact locations in the image-plane:

```{image} https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_3/point_0.png
:alt: Alternative text
:width: 400
```

```{image} https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_3/point_1.png
:alt: Alternative text
:width: 400
```

```{image} https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_3/point_2.png
:alt: Alternative text
:width: 400
```

```{image} https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_3/point_3.png
:alt: Alternative text
:width: 400
```

```{image} https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_3/point_4.png
:alt: Alternative text
:width: 400
```

Note that the image positions above include the fifth central image of the strong lens, which is often not seen in
strong lens imaging data. It is easy to disable this image in the point source modeling.

Checkout the `autolens_workspace/*/point_source` package to get started.

## Interferometry

Modeling of interferometer data from submillimeter (e.g. ALMA) and radio (e.g. LOFAR) observatories:

```{image} https://raw.githubusercontent.com/PyAutoLabs/PyAutoGalaxy/main/paper/almacombined.png
:alt: Alternative text
:width: 600
```

Visibilities data is fitted directly in the uv-plane, circumventing issues that arise when fitting a dirty image
such as correlated noise. This uses the non-uniform fast fourier transform algorithm
\[PyNUFFT\](<https://github.com/jyhmiinlin/pynufft>) to efficiently map the galaxy model images to the uv-plane.

Checkout the `autolens_workspace/*/interferometer` package to get started.

## Multi Gaussian Expansion (MGE)

An MGE decomposes the light of a galaxy into tens or hundreds of two dimensional Gaussians:

```{image} https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_3/mge.png
:alt: Alternative text
:width: 600
```

In the image above, 30 Gaussians are shown, where their sizes go from below the pixel scale (in order to resolve
point emission) to beyond the size of the galaxy (to capture its extended emission).

An MGE is an extremely powerful way to model and subtract the light of the foreground lens galaxy in strong lens imaging,
and makes it possible to model the stellar mass of the lens galaxy in a way that is tied to its light.

Scientific Applications include capturing departures from elliptical symmetry in the light of galaxies, providing a
flexible model to deblend the emission of point sources (e.g. quasars) from the emission of their host galaxy and
deprojecting the light of a galaxy from 2D to 3D.

The following paper gives a detailed overview of MGEs and their applications in strong lensing: <https://arxiv.org/abs/2403.16253>

Checkout `autolens_workspace/notebooks/features/multi_gaussian_expansion.ipynb` to learn how to use an MGE.

## Groups

The strong lenses we've discussed so far have just a single lens galaxy responsible for the lensing. Group-scale
strong lenses are systems where there two or more lens galaxies deflecting one or more background sources:

```{image} https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_3/group.png
:alt: Alternative text
:width: 600
```

**PyAutoLens** has built in tools for modeling group-scale lenses, with no limit on the number of
lens and source galaxies!

Overviews of group and analysis are given in `notebooks/overview/overview_9_groups.ipynb`
The `autolens_workspace/*/group` package has example scripts for simulating datasets and lens modeling.

## Multi-Wavelength

Modeling imaging datasets observed at different wavelengths (e.g. HST F814W and F150W) simultaneously or simultaneously
analysing imaging and interferometer data:

```{image} https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_3/g_image.png
:alt: Alternative text
:width: 600
```

```{image} https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_3/r_image.png
:alt: Alternative text
:width: 600
```

The appearance of the strong changes as a function of wavelength, therefore multi-wavelength analysis means we can learn
more about the different components in a galaxy (e.g a redder bulge and bluer disk) or when imaging and interferometer
data are combined, we can compare the emission from stars and dust.

Checkout the `autolens_workspace/*/multi` package to get started, however combining datasets is a more advanced
feature and it is recommended you first get to grips with the core API.

### Ellipse Fitting

Ellipse fitting is a technique which fits many ellipses to a galaxy's emission to determine its ellipticity, position
angle and centre, without assuming a parametric form for its light (e.g. like a Seisc profile):

```{image} https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_3/ellipse.png
:alt: Alternative text
:width: 600
```

This provides complementary information to parametric light profile fitting, for example giving insights on whether
the ellipticity and position angle are constant with radius or if the galaxy's emission is lopsided.

There are also multipole moment extensions to ellipse fitting, which determine higher order deviations from elliptical
symmetry providing even more information on the galaxy's structure.

The following paper describes the technique in detail: <https://arxiv.org/html/2407.12983v1>

Checkout `autolens_workspace/notebooks/features/ellipse_fitting.ipynb` to learn how to use ellipse fitting.

## Shapelets

Shapelets are a set of orthogonal basis functions that can be combined the represent galaxy structures:

```{image} https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_3/shapelets.png
:alt: Alternative text
:width: 600
```

Scientific Applications include capturing symmetric structures in a galaxy which are more complex than a Sersic profile,
irregular and asymmetric structures in a galaxy like spiral arms and providing a flexible model to deblend the emission
of point sources (e.g. quasars) from the emission of their host galaxy.

Checkout `autolens_workspace/notebooks/features/shapelets.ipynb` to learn how to use shapelets.

## Operated Light Profiles

An operated light profile is one where it is assumed to already be convolved with the PSF of the data, with the
`Moffat` and `Gaussian` profiles common choices:

They are used for certain scientific applications where the PSF convolution is known to be a significant effect and
the knowledge of the PSF allows for detailed modeling abd deblending of the galaxy's light.

Checkout `autogalaxy_workspace/notebooks/features/operated_light_profiles.ipynb` to learn how to use operated profiles.

## Sky Background

When an image of a galaxy is observed, the background sky contributes light to the image and adds noise:

For detailed studies of the outskirts of galaxies (e.g. stellar halos, faint extended disks), the sky background must be
accounted for in the model to ensure robust and accurate fits.

Checkout `autogalaxy_workspace/notebooks/features/sky_background.ipynb` to learn how to use include the sky
background in your model.

## Other

- mass models (aris paper)
- Automated pipelines / SLaM.
- Dark matter subhalos.
- Graphical models.
