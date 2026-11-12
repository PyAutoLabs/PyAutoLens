(references)=
<span id="citations--references"></span>
<span id="citations-references"></span>


# Citations and references

Cite PyAutoLens and the methods used in your analysis. State the software version, name the sampler, and include only the dependencies and methods relevant to the calculation you performed.

## Cite the software

Use the [PyAutoLens software paper](https://doi.org/10.21105/joss.02825) with bibliography key `pyautolens`. Model composition and fitting use [PyAutoFit](https://doi.org/10.21105/joss.02550), key `pyautofit`.

For the AutoLens modelling and source-reconstruction methods, also cite [Nightingale & Dye (2015)](https://doi.org/10.1093/mnras/stv1455), key `Nightingale2015`, and [Nightingale, Dye & Massey (2018)](https://doi.org/10.1093/mnras/sty1264), key `Nightingale2018`.

Download the [BibTeX bibliography](https://github.com/PyAutoLabs/PyAutoLens/blob/main/files/citations.bib), or adapt the example citation text in [LaTeX](https://github.com/PyAutoLabs/PyAutoLens/blob/main/files/citations.tex) or [Markdown](https://github.com/PyAutoLabs/PyAutoLens/blob/main/files/citations.md). Check the example against the packages and versions you actually used.

<span id="dynesty"></span>

## Cite your sampler or optimizer

| Used in your analysis | Reference / bibliography key |
|---|---|
| **Nautilus** | [Lange (2023)](https://doi.org/10.1093/mnras/stad2441), `nautilus` |
| Dynesty | `dynesty`; follow the [version-specific citation guidance](https://dynesty.readthedocs.io/en/latest/references.html) for additional references. |
| Emcee | `emcee` |
| Zeus | `zeus1`, `zeus2` |
| BlackJAX-based searches | [BlackJAX](https://arxiv.org/abs/2402.10797), `blackjax` |
| SciPy optimizers | `scipy` |
| Optax optimizers | `optax` |
| MultiStartProdigy | [Prodigy](https://proceedings.mlr.press/v235/mishchenko24a.html), `prodigy`, alongside `optax` |

For other search packages or older analyses, use the package's own citation instructions. The bibliography also retains references for previously supported searches so older work remains citable.

<span id="jax"></span>

<span id="jax-zero-contour"></span>

<span id="nufftax"></span>

## Cite the numerical methods you used

| Feature | Citation |
|---|---|
| JAX compilation, differentiation or execution | `jax` |
| Non-uniform FFT with FINUFFT | `finufft` |
| Non-uniform FFT with NUFFTax | `nufftax` and `finufft` |
| Jax-Zero-Contour critical-curve calculation | `jax_zero_contour`, when this method is selected |
| Smooth RTU adaptive rectangular mesh | [Enzi et al.](https://arxiv.org/abs/2606.30620), `Enzi2026` |
| Multi-Gaussian light modelling and joint linear inversion | [He et al. (2024)](https://arxiv.org/abs/2403.16253), key `He2024`, alongside the underlying MGE method used |

The NumPy `TransformerNUFFT` path uses FINUFFT where supported; the JAX path uses NUFFTax. Cite the backend that performed your calculation. A direct Fourier transform does not use either NUFFT implementation.

Jax-Zero-Contour is optional. Select it through `critical_curves_method: zero_contour`; do not infer its use simply because a figure contains critical curves. The shared packaged plotting configuration currently selects `marching_squares`.

<span id="rectangular-mesh-pixelized-source-reconstructions"></span>

<span id="rectangular-mesh-pixelized-reconstructions"></span>

## Adaptive rectangular reconstructions

`RectangularRTUAdaptDensity` and `RectangularRTUAdaptImage` use a smooth adaptive coordinate transform based on the ray-guided transformed uniform grid method of [Enzi et al.](https://arxiv.org/abs/2606.30620). Cite that method when using these meshes.

The `RectangularBilinearAdaptDensity` and `RectangularBilinearAdaptImage` variants use an empirical rank-CDF transform. They are not interchangeable with the smooth RTU transform when describing differentiability or numerical implementation. State the mesh class used in your analysis. `RectangularUniform` has no adaptive RTU transform.

PyAutoLens combines its meshes with its own regularization schemes, such as `reg.Constant` and `reg.Adapt`, rather than automatically adopting the Gaussian-process source prior in the RTU paper. Preserve that distinction when describing the likelihood or comparing results.

<span id="mass-models"></span>

## Mass-profile calculations

For the CSE-based decomposed stellar/dark-matter profile calculations, cite [Oguri (2021)](https://arxiv.org/abs/2106.11464), key `Oguri2021`. For the generalized NFW calculation described by [Shajib (2019)](https://academic.oup.com/mnras/article/488/1/1387/5526256), use `Anowar2019` when that implementation is used. Not every mass profile or numerical route uses the same method.

<span id="science-papers"></span>

## Next steps

Use the [likelihood guide](likelihood_function.md) to identify the calculation behind your fit, and [Papers](papers.md) for applications of the software.
