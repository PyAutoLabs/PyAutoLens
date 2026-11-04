# Chapter 3: Pixelizations

In chapter 3, we reconstruct complex source light on pixelized grids. We use Bayesian evidence to choose the smoothing strength, investigate fits that remain inadequate, and adapt the mesh and regularization to the light distribution.

The chapter contains the following tutorials:

[Tutorial 1: Pixelizations, Mappers and Inversions](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/main/notebooks/chapter_3_pixelizations/tutorial_1_pixelizations.ipynb)
\- Connect mesh points to image pixels and reconstruct complex source light with a linear inversion.

[Tutorial 2: Bayesian Regularization](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/main/notebooks/chapter_3_pixelizations/tutorial_2_bayesian_regularization.ipynb)
\- Compare log evidences to choose the regularization strength and examine its dependence on mesh resolution.

[Tutorial 3: Borders](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/main/notebooks/chapter_3_pixelizations/tutorial_3_borders.ipynb)
\- Explore how masks and border relocation affect the extent and resolution of the source mesh.

[Tutorial 4: Lens Modeling](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/main/notebooks/chapter_3_pixelizations/tutorial_4_lens_modeling.ipynb)
\- Fit lens mass and foreground light alongside a pixelized source, using search chaining and the positions likelihood penalty introduced in chapter 2.

[Tutorial 5: Fit Problems](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/main/notebooks/chapter_3_pixelizations/tutorial_5_fit_problems.ipynb)
\- See why compact source light challenges mesh resolution and constant regularization, even after choosing the coefficient using evidence.

[Tutorial 6: Adaptive Reconstruction](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/main/notebooks/chapter_3_pixelizations/tutorial_6_adaptive_reconstruction.ipynb)
\- Combine Delaunay meshes, Hilbert brightness adaptation and adaptive Split regularization in a chained lens-model fit.
