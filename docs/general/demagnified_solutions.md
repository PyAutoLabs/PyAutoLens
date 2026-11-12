(demagnified-solutions)=
# Demagnified solutions

A flexible pixelized source can fit an image well even when the lens mass model is wrong. A positions penalty helps reject models in which identified multiple images fail to trace back to the same source region.

<span id="overview"></span>

## Recognize the failure

With too little lens mass, little deflection occurs and the reconstructed source can resemble the observed arcs or ring rather than a compact unlensed galaxy. A second pathological branch can occur with too much mass, producing an inverted reconstruction. Inspect the reconstructed source as well as the image-plane residuals.

The diagrams and reconstructions in [Maresca et al.](https://arxiv.org/abs/2012.04665) illustrate these branches:

```{image} images/maresca_fig1.png
:alt: Schematic comparison of physically plausible and pathological source reconstructions from different lens mass models.
:width: 650
```

```{image} images/maresca_fig2.png
:alt: Image fits and source reconstructions showing how pathological lens models can reproduce the observed image.
:width: 650
```

*Figures from Maresca et al.; see the paper for the model definitions and interpretation.*

A pathological solution need not have the highest likelihood to trap a search: it can occupy a large region of parameter space that is easier to find than the physical solution. The positions penalty supplies additional information about which image features correspond to the same source emission.

<span id="solution"></span>

## Add a positions penalty

Identify two or more image-plane positions belonging to the same source feature. The coordinates are `(y, x)` in arcseconds. These illustrative positions must be replaced with measurements from your own image:

```python
import autolens as al

positions = al.Grid2DIrregular(
    values=[(0.4, 1.6), (1.58, -0.35), (-0.43, -1.59), (-1.45, 0.2)]
)

positions_likelihood = al.PositionsLH(positions=positions, threshold=0.3)

# dataset is the imaging dataset prepared for your fit.
analysis = al.AnalysisImaging(
    dataset=dataset,
    positions_likelihood_list=[positions_likelihood],
)
```

The [positions GUI](https://github.com/PyAutoLabs/autolens_workspace/blob/main/scripts/imaging/data_preparation/gui/positions.py) lets you mark positions on an image. Choose corresponding features, not arbitrary bright peaks that may come from different source regions.

For each candidate mass model, `PositionsLH` ray-traces these coordinates to the source plane and measures their maximum separation. Above the threshold, the model receives a penalty that becomes more severe as the separation increases. At or below the threshold, no positions penalty is applied. Ray-tracing a few coordinates is inexpensive compared with reconstructing a pixelized source.

## Choose a threshold

The example uses **0.3 arcseconds in the source plane**. This is a tolerance for rejecting inconsistent lens models, not a measurement error or a universal setting. A threshold that is too small can exclude plausible models when the selected image features do not correspond exactly; one that is too large may fail to reject unwanted solutions.

An accurate model may trace the positions much closer together, sometimes below 0.01 arcseconds. The initial threshold does not need to demand that precision. Check the inferred source, the traced positions and the fit residuals; a penalty alone does not establish that the mass model is correct.

<span id="auto-position-updates"></span>

## Update positions from an earlier fit

Manual selection is time-consuming for large samples, and corresponding features can be ambiguous for complex sources. After obtaining a useful mass and parametric source model, use it to predict multiple images of the source centre:

```python
positions = result.image_plane_multiple_image_positions()
```

The simpler source model used in an initial search is less able to imitate the observed arcs than a flexible pixelized reconstruction. Inspect that initial fit before using it to initialize later stages.

You can set a threshold from the source-plane separation of the predicted images:

```python
threshold = result.positions_threshold_from(
    factor=2.0,
    minimum_threshold=0.3,
)
```

`factor` multiplies the inferred separation, and `minimum_threshold` sets a floor. For example, a separation of 0.1 arcseconds multiplied by 2 gives 0.2 arcseconds, so the floor makes the returned threshold 0.3 arcseconds. These controls stop later stages from inheriting an unnecessarily restrictive threshold.

The convenience method creates the positions penalty directly:

```python
positions_likelihood = result.positions_likelihood_from(
    factor=3.0,
    minimum_threshold=0.2,
)

analysis = al.AnalysisImaging(
    dataset=dataset,
    positions_likelihood_list=[positions_likelihood],
)
```

<span id="multiple-source-plane-systems"></span>

## Multiple source planes

Use a separate position set for each source. `plane_redshift` specifies the source plane to which that set is traced. A system with two distinct source planes behind a foreground lens has at least three galaxy planes.

```python
# positions_source_1 and positions_source_2 contain measured coordinates
# for corresponding images of each separate source.
positions_likelihood_1 = al.PositionsLH(
    positions=positions_source_1, threshold=0.3, plane_redshift=1.0
)
positions_likelihood_2 = al.PositionsLH(
    positions=positions_source_2, threshold=0.3, plane_redshift=2.0
)

analysis = al.AnalysisImaging(
    dataset=dataset,
    positions_likelihood_list=[positions_likelihood_1, positions_likelihood_2],
)
```

The example redshifts and thresholds must match your source models. Each position set is tested independently at its own source redshift.

From a previous fit containing both sources:

```python
analysis = al.AnalysisImaging(
    dataset=dataset,
    positions_likelihood_list=[
        result.positions_likelihood_from(
            factor=3.0,
            minimum_threshold=0.2,
            plane_redshift=result.instance.galaxies.source_1.redshift,
        ),
        result.positions_likelihood_from(
            factor=3.0,
            minimum_threshold=0.2,
            plane_redshift=result.instance.galaxies.source_2.redshift,
        ),
    ],
)
```

## Next steps

Follow the [search-chaining guide](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/guides/modeling/chaining.ipynb) to carry information between fits, or the [double-source-plane example](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/double_source_plane_lens/start_here.ipynb) for multiple sources.
