# Citations & References

The bibtex entries for **PyAutoLens** and its affiliated software packages can be found
[here](https://github.com/Jammy2211/PyAutoLens/blob/main/files/citations.bib), with example text for citing **PyAutoLens**
in [.tex format here](https://github.com/Jammy2211/PyAutoLens/blob/main/files/citations.tex) format here and
[.md format here](https://github.com/Jammy2211/PyAutoLens/blob/main/files/citations.md). As shown in the examples, we
would greatly appreciate it if you mention **PyAutoLens** by name and include a link to our GitHub page!

**PyAutoLens** is published in the [Journal of Open Source Software](https://joss.theoj.org/papers/10.21105/joss.02825#) and its
entry in the above .bib file is under the citation key `pyautolens`. Please also cite the MNRAS AutoLens
papers (<https://academic.oup.com/mnras/article/452/3/2940/1749640> and <https://academic.oup.com/mnras/article-abstract/478/4/4738/5001434?redirectedFrom=fulltext>) which are included
under the citation keys `Nightingale2015` and `Nightingale2018`.

You should also specify the non-linear search(es) you use in your analysis (e.g. Dynesty, Emcee, PySwarms, etc) in
the main body of text, and delete as appropriate any packages your analysis did not use. The citations.bib file includes
the citation key for all of these projects.

If you use decomposed mass models (e.g. stellar mass models like an `Sersic` or dark matter models like
an `NFW`) please cite the following paper <https://arxiv.org/abs/2106.11464> under
citation key `Oguri2021`. Our deflection angle calculations are based on this method.

If you specifically use a decomposed mass model with the `gNFW` please cite the following paper <https://academic.oup.com/mnras/article/488/1/1387/5526256> under
citation key `Anowar2019`.

The citations.bib file above also includes my work on [using strong lensing to study galaxy structure](https://ui.adsabs.harvard.edu/abs/2019MNRAS.489.2049N/abstract). If you're feeling kind, please go ahead and stick
a citation in your introduction using citep\{Nightingale2019} or [@Nightingale2019] ;).
