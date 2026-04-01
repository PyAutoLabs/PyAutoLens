.. _references:

Citations & References
======================

The bibtex entries for **PyAutoLens** and its affiliated software packages can be found
`here <https://github.com/Jammy2211/PyAutoLens/blob/main/files/citations.bib>`_, with example text for citing **PyAutoLens**
in `.tex format here <https://github.com/Jammy2211/PyAutoLens/blob/main/files/citations.tex>`_ format here and
`.md format here <https://github.com/Jammy2211/PyAutoLens/blob/main/files/citations.md>`_. As shown in the examples, we
would greatly appreciate it if you mention **PyAutoLens** by name and include a link to our GitHub page!

**PyAutoLens** is published in the `Journal of Open Source Software <https://joss.theoj.org/papers/10.21105/joss.02825#>`_ and its
entry in the above .bib file is under the citation key ``pyautolens``. Please also cite the MNRAS AutoLens
papers (https://academic.oup.com/mnras/article/452/3/2940/1749640 and https://academic.oup.com/mnras/article-abstract/478/4/4738/5001434?redirectedFrom=fulltext) which are included
under the citation keys ``Nightingale2015`` and ``Nightingale2018``.

You should also specify the non-linear search(es) you use in your analysis (e.g. Dynesty, Emcee, PySwarms, etc) in
the main body of text, and delete as appropriate any packages your analysis did not use. The citations.bib file includes
the citation key for all of these projects.

Jax-Zero-Contour
----------------

If you use the zero-contour method for critical curve and caustic computation (the default in
``visualize/general.yaml`` via ``critical_curves_method: zero_contour``), please cite the
``Jax-Zero-Contour`` package by Coleman Krawczyk:

.. code-block:: bibtex

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

The package is available at https://github.com/CKrawczyk/Jax-Zero-Contour and archived at
https://doi.org/10.5281/zenodo.15730415.

Dynesty
-------

If you used the nested sampling algorithm Dynesty, please follow the citation instructions `on the dynesty readthedocs <https://dynesty.readthedocs.io/en/latest/references.html>`_.

Mass Models
-----------

If you use decomposed mass models (e.g. stellar mass models like an ``Sersic`` or dark matter models like
an ``NFW``) please cite the following paper https://arxiv.org/abs/2106.11464 under
citation key ``Oguri2021``. Our deflection angle calculations are based on this method.

If you specifically use a decomposed mass model with the ``gNFW`` please cite the following paper https://academic.oup.com/mnras/article/488/1/1387/5526256 under
citation key ``Anowar2019``.

Science Papers
--------------

The citations.bib file above also includes my work on `using strong lensing to study galaxy structure
<https://ui.adsabs.harvard.edu/abs/2019MNRAS.489.2049N/abstract>`_. If you're feeling kind, please go ahead and stick
a citation in your introduction using \citep{Nightingale2019} or [@Nightingale2019] ;).

