(model-cookbook)=

# Model Cookbook

The model cookbook provides a concise reference to lens model composition tools, specifically the `Model`
and `Collection` objects.

Examples using different **PyAutoLens** API’s for model composition are provided, which produce more concise and
readable code for different use-cases.

## Simple Lens Model

A simple lens model has a lens galaxy with a Sersic light profile, Isothermal mass profile and
source galaxy with a Sersic light profile:

```python
# Lens:

bulge = af.Model(al.lp.Sersic)
mass = af.Model(al.mp.Isothermal)

lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=bulge,
    mass=mass,
)

# Source:

bulge = af.Model(al.lp.SersicCore)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))
```

The redshifts in the above model are used to determine which galaxy is the lens and which is the source.

The model `total_free_parameters` attribute which tells us the total number of free parameters (which are fitted for
via a non-linear search), which in this case is 19 (7 from the lens Sersic, 5 from the lens Isothermal and 7 from the
source Sersic).

```python
print(f"Model Total Free Parameters = {model.total_free_parameters}")
```

If we print the `info` attribute of the model we get information on all of the parameters and their priors.

```python
print(model.info)
```

This gives the following output:

```bash
galaxies
    lens
        redshift                                 0.5
        bulge
            centre
                centre_0                         GaussianPrior, mean = 0.0, sigma = 0.3
                centre_1                         GaussianPrior, mean = 0.0, sigma = 0.3
            ell_comps
                ell_comps_0                      GaussianPrior, mean = 0.0, sigma = 0.5
                ell_comps_1                      GaussianPrior, mean = 0.0, sigma = 0.5
            intensity                            LogUniformPrior, lower_limit = 1e-06, upper_limit = 1000000.0
            effective_radius                     UniformPrior, lower_limit = 0.0, upper_limit = 30.0
        mass
            centre
                centre_0                         GaussianPrior, mean = 0.0, sigma = 0.1
                centre_1                         GaussianPrior, mean = 0.0, sigma = 0.1
            ell_comps
                ell_comps_0                      GaussianPrior, mean = 0.0, sigma = 0.3
                ell_comps_1                      GaussianPrior, mean = 0.0, sigma = 0.3
            einstein_radius                      UniformPrior, lower_limit = 0.0, upper_limit = 8.0
    source
        redshift                                 1.0
        disk
            centre
                centre_0                         GaussianPrior, mean = 0.0, sigma = 0.3
                centre_1                         GaussianPrior, mean = 0.0, sigma = 0.3
            ell_comps
                ell_comps_0                      GaussianPrior, mean = 0.0, sigma = 0.5
                ell_comps_1                      GaussianPrior, mean = 0.0, sigma = 0.5
            intensity                            LogUniformPrior, lower_limit = 1e-06, upper_limit = 1000000.0
            effective_radius                     UniformPrior, lower_limit = 0.0, upper_limit = 30.0
```

## More Complex Lens Models

The API above can be easily extended to compose lens models where each galaxy has multiple light or mass profiles:

```python
# Lens:

bulge = af.Model(al.lp.Sersic)
disk = af.Model(al.lp.Exponential)

mass = af.Model(al.mp.Isothermal)
shear = af.Model(al.mp.ExternalShear)

lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=bulge,
    disk=disk,
    mass=mass,
    shear=shear,
)

# Source:

bulge = af.Model(al.lp.SersicCore)
disk = af.Model(al.lp.Exponential)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge, disk=disk)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))
```

The use of the words `bulge`, `disk`, `mass` and `shear` above are arbitrary. They can be replaced with any name you
like, e.g. `bulge_0`, `bulge_1`, `mass_0`, `mass_1`, and the model will still behave in the same way.

The API can also be extended to compose lens models where there are multiple galaxies:

```python
bulge = af.Model(al.lp.Sersic)
mass = af.Model(al.mp.Isothermal)

lens_0 = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=bulge,
    mass=mass,
)

bulge = af.Model(al.lp.Sersic)
mass = af.Model(al.mp.Isothermal)

lens_1 = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=bulge,
    mass=mass,
)

# Source 0:

bulge = af.Model(al.lp.SersicCore)

source_0 = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

# Source 1 :

bulge = af.Model(al.lp.SersicCore)

source_1 = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

# Overall Lens Model:

model = af.Collection(
    galaxies=af.Collection(
        lens_0=lens_0,
        lens_1=lens_1,
        source_0=source_0,
        source_1=source_1
    )
)
```

The above lens model consists of only two planes (an image-plane and source-plane), but has four galaxies in total.
This is because the lens galaxies have the same redshift and the souece galaxies have the same redshift.

If we gave one of the lens galaxies a different redshift, it would be included in a third plane, and the model would
perform multi-plane ray tracing when the model-fit is performed.

## Concise API

If a light or mass profile is passed directly to the `af.Model` of a galaxy, it is automatically assigned to be a
`af.Model` component of the galaxy.

This means we can write the model above comprising multiple light and mass profiles more concisely as follows (also
removing the comments reading Lens / Source / Overall Lens Model to make the code more readable):

```python
lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=al.lp.Sersic,
    disk=al.lp.Sersic,
    mass=al.mp.Isothermal,
    shear=al.mp.ExternalShear,
)

source = af.Model(
    al.Galaxy,
    redshift=1.0,
    bulge=al.lp.SersicCore,
    disk=al.lp.Exponential
)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))
```

## Prior Customization

We can customize the priors of the lens model component individual parameters as follows:

```python
# Lens:

bulge = af.Model(al.lp.Sersic)
bulge.intensity = af.LogUniformPrior(lower_limit=1e-4, upper_limit=1e4)
bulge.sersic_index = af.GaussianPrior(mean=4.0, sigma=1.0, lower_limit=1.0, upper_limit=8.0)

mass = af.Model(al.mp.Isothermal)
mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1, lower_limit=-0.5, upper_limit=0.5)
mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1, lower_limit=-0.5, upper_limit=0.5)
mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=8.0)

lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=bulge,
    mass=mass,
)

# Source

bulge = af.Model(al.lp.Sersic)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)
source.effective_radius = af.GaussianPrior(mean=0.1, sigma=0.05, lower_limit=0.0, upper_limit=1.0)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))
```

## Model Customization

We can customize the lens model parameters in a number of different ways, as shown below:

```python
# Lens:

bulge = af.Model(al.lp.Sersic)
disk = af.Model(al.lp.Exponential)

# Parameter Pairing: Pair the centre of the bulge and disk together, reducing
# the complexity of non-linear parameter space by N = 2

bulge.centre = disk.centre

# Parameter Fixing: Fix the sersic_index of the bulge to a value of 4, reducing
# the complexity of non-linear parameter space by N = 1

bulge.sersic_index = 4.0

mass = af.Model(al.mp.Isothermal)

# Parameter Offsets: Make the mass model centre parameters the same value as
# the bulge / disk but with an offset.

mass.centre.centre_0 = bulge.centre.centre_0 + 0.1
mass.centre.centre_1 = bulge.centre.centre_1 + 0.1

shear = af.Model(al.mp.ExternalShear)

lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=bulge,
    disk=disk,
    mass=mass,
    shear=shear,
)

# Source:

bulge = af.Model(al.lp.SersicCore)
disk = af.Model(al.lp.Exponential)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge, disk=disk)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

# Assert that the effective radius of the bulge is larger than that of the disk.
# (Assertions can only be added at the end of model composition, after all components
# have been bright together in a `Collection`.
model.add_assertion(model.galaxies.lens.bulge.effective_radius > model.galaxies.lens.disk.effective_radius)

# Assert that the Einstein Radius is below 3.0":
model.add_assertion(model.galaxies.lens.mass.einstein_radius < 3.0)
```

## Redshift Free

The redshift of a galaxy can be treated as a free parameter in the model-fit by using the following API:

```python
redshift = af.Model(al.Redshift)
redshift.redshift = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)

lens = af.Model(
    al.Galaxy,
    redshift=redshift,
    mass=al.mp.Isothermal
)
```

The model-fit will automatically enable multi-plane ray tracing and alter the ordering of the planes depending on the
redshifts of the galaxies.

NOTE: For strong lenses with just two planes (an image-plane and source-plane) the redshifts of the galaxies do not
impact the model-fit. You should therefore never make the redshifts free if you are only modeling a two-plane lens
system. This is because lensing calculations can be defined in arc-second coordinates, which do not change as a
function of redshift.

Redshifts should be made free when modeling three or more planes, as the mulit-plane ray-tracing calculations have an
obvious dependence on the redshifts of the galaxies which could be inferred by the model-fit.

## Available Model Components

The light profiles, mass profiles and other components that can be used for lens modeling are given at the following
API documentation pages:

> - <https://pyautolens.readthedocs.io/en/latest/api/light.html>
> - <https://pyautolens.readthedocs.io/en/latest/api/mass.html>
> - <https://pyautolens.readthedocs.io/en/latest/api/pixelization.html>

## JSon Outputs

After a model is composed, it can easily be output to a .json file on hard-disk in a readable structure:

```python
import os
import json

model_path = path.join("path", "to", "model", "json")

os.makedirs(model_path, exist_ok=True)

model_file = path.join(model_path, "model.json")

with open(model_file, "w+") as f:
    json.dump(model.dict(), f, indent=4)
```

We can load the model from its `.json` file.

```python
model = af.Model.from_json(file=model_file)
```

This means in **PyAutoLens** one can write a model in a script, save it to hard disk and load it elsewhere, as well
as manually customize it in the .json file directory.

This is used for composing complex models of group scale lenses.

## Many Profile Models (Advanced)

Features such as the Multi Gaussian Expansion (MGE) and shapelets compose models consisting of 50 - 500+ light
profiles.

The following example notebooks show how to compose and fit these models:

<https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/multi_gaussian_expansion/modeling.ipynb>
<https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/shapelets/modeling.ipynb>

## Model Linking (Advanced)

When performing non-linear search chaining, the inferred model of one phase can be linked to the model.

The following example notebooks show how to compose and fit these models:

<https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/advanced/chaining/start_here.ipynb>

## Across Datasets (Advanced)

When fitting multiple datasets, model can be composed where the same model component are used across the datasets
but certain parameters are free to vary across the datasets.

The following example notebooks show how to compose and fit these models:

<https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi/start_here.ipynb>

## Relations (Advanced)

We can compose models where the free parameter(s) vary according to a user-specified function
(e.g. y = mx +c -> intensity = (m * wavelength) + c across the datasets.

The following example notebooks show how to compose and fit these models:

<https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi/features/wavelength_dependence/modeling.ipynb>

## PyAutoFit API

**PyAutoFit** is a general model composition library which offers even more ways to compose lens models not
detailed in this cookbook.

The **PyAutoFit** model composition cookbooks detail this API in more detail:

<https://pyautofit.readthedocs.io/en/latest/cookbooks/model.html>
<https://pyautofit.readthedocs.io/en/latest/cookbooks/multi_level_model.html>

## Wrap Up

This cookbook shows how to compose simple lens models using the `af.Model()` and `af.Collection()` objects.
