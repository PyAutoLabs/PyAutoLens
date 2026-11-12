(howtolens)=

# HowToLens

Learn gravitational lensing and Bayesian modelling through worked Python notebooks with **PyAutoLens**. The course starts with simple models and builds towards fitting observations and analysing more complex systems.

Basic Python familiarity is useful. You do not need previous experience with lensing or Bayesian statistics.

## Start here

**Tutorial 0: Visualization** — [GitHub](https://github.com/PyAutoLabs/HowToLens/blob/main/notebooks/chapter_1_introduction/tutorial_0_visualization.ipynb) / [Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/main/notebooks/chapter_1_introduction/tutorial_0_visualization.ipynb)

Check your setup and learn how to display figures. In Colab, run the setup cell first, then work through the notebook in order.

You can also run the notebooks locally or use the [Python scripts](https://github.com/PyAutoLabs/HowToLens/tree/main/scripts).

## Choose a chapter

GitHub opens the chapter’s notebook folder. Colab opens its first tutorial; the optional section opens Alternative Searches.

| Chapter | What you’ll learn | Before you start | Open |
| --- | --- | --- | --- |
| <span id="chapter_1_introduction"></span>1. Strong Lensing | Build galaxies, trace light, simulate observations and compare model images with data. | Basic Python; lensing concepts are introduced as you go. | [GitHub](https://github.com/PyAutoLabs/HowToLens/tree/main/notebooks/chapter_1_introduction) / [Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/main/notebooks/chapter_1_introduction/tutorial_0_visualization.ipynb) |
| <span id="chapter_2_lens_modeling"></span>2. Lens Modeling | Fit lens models with Bayesian searches, inspect results and fit components in stages. | Chapter 1: ray tracing, data, residuals and likelihoods. | [GitHub](https://github.com/PyAutoLabs/HowToLens/tree/main/notebooks/chapter_2_lens_modeling) / [Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/main/notebooks/chapter_2_lens_modeling/tutorial_1_non_linear_search.ipynb) |
| <span id="chapter_3_pixelizations"></span>3. Pixelizations | Reconstruct background-source light on a mesh; choose smoothing using Bayesian evidence and improve the reconstruction. | Chapter 2, including search chaining and positions constraints. | [GitHub](https://github.com/PyAutoLabs/HowToLens/tree/main/notebooks/chapter_3_pixelizations) / [Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/main/notebooks/chapter_3_pixelizations/tutorial_1_pixelizations.ipynb) |
| <span id="chapter_4_scaling_up_lensing"></span>4. Scaling Up Lensing | Model multiple deflectors, groups and clusters; progress from extended images to image positions and weak-lensing shear. | Chapter 2 and the pixelized-source methods introduced in Chapter 3. | [GitHub](https://github.com/PyAutoLabs/HowToLens/tree/main/notebooks/chapter_4_scaling_up_lensing) / [Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/main/notebooks/chapter_4_scaling_up_lensing/tutorial_1_extra_galaxies.ipynb) |
| <span id="chapter_optional"></span>Optional tutorials | Explore alternative searches and runtime. Prior passing is historical background. | Chapter 2. | [GitHub](https://github.com/PyAutoLabs/HowToLens/tree/main/notebooks/chapter_optional) / [Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/main/notebooks/chapter_optional/tutorial_searches.ipynb) |

Work through Chapters 1 and 2, then practise with the [workspace examples](https://github.com/PyAutoLabs/autolens_workspace#find-your-examples). Return to Chapters 3 and 4 when you need their methods.

These links follow the current course on `main`. For a particular release, choose its tag in the [course repository](https://github.com/PyAutoLabs/HowToLens/tags).

## Help as you learn

If you want AI-guided help, the [PyAutoLens AI Assistant](https://github.com/PyAutoLabs/autolens_assistant) can discuss the concepts, code and figures as you work through the notebooks.

For additional theory, refer to [Massimo Meneghetti’s gravitational-lensing lectures](http://www.ita.uni-heidelberg.de/~massimo/sub/Lectures/gl_all.pdf).
