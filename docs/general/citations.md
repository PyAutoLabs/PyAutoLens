(references)=

# Citations & References

The bibtex entries for **PyAutoLens** and its affiliated software packages can be found
[here](https://github.com/PyAutoLabs/PyAutoLens/blob/main/files/citations.bib), with example text for citing **PyAutoLens**
in [.tex format here](https://github.com/PyAutoLabs/PyAutoLens/blob/main/files/citations.tex) format here and
[.md format here](https://github.com/PyAutoLabs/PyAutoLens/blob/main/files/citations.md). As shown in the examples, we
would greatly appreciate it if you mention **PyAutoLens** by name and include a link to our GitHub page!

**PyAutoLens** is published in the [Journal of Open Source Software](https://joss.theoj.org/papers/10.21105/joss.02825#) and its
entry in the above .bib file is under the citation key `pyautolens`. Please also cite the MNRAS AutoLens
papers (<https://academic.oup.com/mnras/article/452/3/2940/1749640> and <https://academic.oup.com/mnras/article-abstract/478/4/4738/5001434?redirectedFrom=fulltext>) which are included
under the citation keys `Nightingale2015` and `Nightingale2018`.

You should also specify the non-linear search(es) you use in your analysis (e.g. Nautilus, Dynesty, Emcee, Zeus, etc) in
the main body of text, and delete as appropriate any packages your analysis did not use. The citations.bib file includes
the citation key for all of these projects.

## JAX

**PyAutoLens** runs on a NumPy backend by default and an optional
[JAX](https://github.com/jax-ml/jax) backend for just-in-time compilation, automatic
differentiation, and GPU/TPU execution. If you run any analysis on the JAX path, please
cite JAX under the citation key `jax`. JAX-specific components that are also cited under
their own keys when used are `optax` (gradient-based optimizers, key `optax`), the
interferometer non-uniform FFT (`nufftax` and its FINUFFT algorithm, keys `nufftax` and
`finufft`, see below), and the critical-curve/caustic solver (`Jax-Zero-Contour`, key
`jax_zero_contour`, see below).

## Jax-Zero-Contour

If you use the zero-contour method for critical curve and caustic computation (the default in
`visualize/general.yaml` via `critical_curves_method: zero_contour`), please cite the
`Jax-Zero-Contour` package by Coleman Krawczyk:

```bibtex
@software{coleman_krawczyk_2025_15730415,
  author       = {Coleman Krawczyk},
  title        = {CKrawczyk/Jax-Zero-Contour: Version 2.0.0},
  month        = jun,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {v2.0.0},
  doi          = {10.5281/zenodo.15730415},
  url          = {https://doi.org/10.5281/zenodo.15730415},
}
```

The package is available at <https://github.com/CKrawczyk/Jax-Zero-Contour> and archived at
<https://doi.org/10.5281/zenodo.15730415>.

## NUFFTax

If you fit interferometer datasets on the JAX path, the non-uniform FFT is performed by
`nufftax`, a pure-JAX NUFFT implementation by the GragasLab team. Please cite the
package:

```bibtex
@software{nufftax,
  author = {Gragas and Oudoumanessah, Geoffroy and Iollo, Jacopo},
  title  = {nufftax: Pure JAX implementation of the Non-Uniform Fast Fourier Transform},
  url    = {https://github.com/GragasLab/nufftax},
  year   = {2026},
}
```

`nufftax`'s algorithm is based on FINUFFT (Flatiron Institute); the upstream paper should
also be cited:

```bibtex
@article{finufft,
  author  = {Barnett, Alexander H. and Magland, Jeremy F. and af Klinteberg, Ludvig},
  title   = {A parallel non-uniform fast Fourier transform library based on an
             'exponential of semicircle' kernel},
  journal = {SIAM J. Sci. Comput.},
  volume  = {41},
  number  = {5},
  pages   = {C479--C504},
  year    = {2019},
}
```

The package is available at <https://github.com/GragasLab/nufftax>.

## Rectangular Mesh (Pixelized Source Reconstructions)

If you reconstruct a source using the adaptive rectangular meshes (`RectangularAdaptDensity` or
`RectangularAdaptImage`), you **must cite** the following paper <https://arxiv.org/abs/2606.30620> under citation
key `Enzi2026`. The mesh's adaptive coordinate transform implements the ray-guided transformed uniform (RTU)
grid formulation this paper introduces, which is what makes the pixelization adaptive and fully
auto-differentiable:

```bibtex
@article{Enzi2026,
  author        = {Enzi, Wolfgang J. R. and Krawczyk, Coleman M. and Li, Tian and Collett, Thomas E.},
  title         = {Gaussian processes on ray-guided transformed uniform grids for fast, flexible, and auto-differentiable adaptive source reconstruction in lens modelling},
  journal       = {MNRAS, submitted},
  eprint        = {2606.30620},
  archivePrefix = {arXiv},
  primaryClass  = {astro-ph.GA},
  year          = {2026},
  url           = {https://arxiv.org/abs/2606.30620},
}
```

Note that **PyAutoLens** pairs the RTU grid with its own regularization schemes (e.g. `reg.Constant`,
`reg.Adapt`) rather than the Gaussian-process source prior used in the paper, so quantitative results are not
directly comparable between the two implementations. The `RectangularUniform` mesh performs no RTU transform,
so this citation is not required when only the uniform mesh is used.

## Dynesty

If you used the nested sampling algorithm Dynesty, please follow the citation instructions [on the dynesty readthedocs](https://dynesty.readthedocs.io/en/latest/references.html).

## Mass Models

If you use decomposed mass models (e.g. stellar mass models like an `Sersic` or dark matter models like
an `NFW`) please cite the following paper <https://arxiv.org/abs/2106.11464> under
citation key `Oguri2021`. Our deflection angle calculations are based on this method.

If you specifically use a decomposed mass model with the `gNFW` please cite the following paper <https://academic.oup.com/mnras/article/488/1/1387/5526256> under
citation key `Anowar2019`.

## Science Papers

The citations.bib file above also includes my work on [using strong lensing to study galaxy structure](https://ui.adsabs.harvard.edu/abs/2019MNRAS.489.2049N/abstract). If you're feeling kind, please go ahead and stick
a citation in your introduction using citep\{Nightingale2019} or [@Nightingale2019] ;).
