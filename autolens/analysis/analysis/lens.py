"""
Lensing-specific mixin for all **PyAutoLens** ``Analysis`` classes.

``AnalysisLens`` is a mixin that adds lensing-specific behaviour to any analysis class.
It is inherited (alongside dataset-specific base classes) by ``AnalysisDataset``
and ``AnalysisPoint``.

Key responsibilities:

- ``tracer_via_instance_from`` — constructs a ``Tracer`` from a ``PyAutoFit`` model
  instance, including automatic multi-plane ordering of galaxies by redshift.
- Position likelihood application — evaluates ``PositionsLH`` objects against the
  current tracer and adds any penalty to the log likelihood.
- ``use_jax`` flag — forwarded to ``FitImaging`` / ``FitInterferometer`` to enable JAX
  acceleration of the likelihood evaluation.
"""
import logging
import numpy as np
from typing import List, Optional

import autofit as af
import autoarray as aa
import autogalaxy as ag

from autolens.analysis.positions import PositionsLH
from autolens.lens.tracer import Tracer

from autolens.lens import tracer_util

from autolens import exc

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


class AnalysisLens:
    def __init__(
        self,
        positions_likelihood_list: Optional[List[PositionsLH]] = None,
        cosmology: ag.cosmo.LensingCosmology = None,
        use_jax: bool = True,
    ):
        """
        Analysis classes are used by PyAutoFit to fit a model to a dataset via a non-linear search.

        This abstract Analysis class has attributes and methods for all model-fits which include lensing calculations,
        but does not perform a model-fit by itself (and is therefore only inherited from).

        This class stores the Cosmology used for the analysis and settings that control specific aspects of the lensing
        calculation, for example how close the brightest pixels in the lensed source have to trace within one another
        in the source plane for the model to not be discarded.

        Parameters
        ----------
        cosmology
            The Cosmology assumed for this analysis.
        """
        from autogalaxy.cosmology.model import Planck15

        self.cosmology = cosmology or Planck15()
        self.positions_likelihood_list = positions_likelihood_list

        # Mirror the autofit Analysis fallback: if jax isn't installed,
        # downgrade silently here too (the parent Analysis.__init__ already
        # emitted the loud banner — no need to repeat it).
        import importlib.util
        if use_jax and importlib.util.find_spec("jax") is None:
            use_jax = False

        self._use_jax = use_jax

    @property
    def _xp(self):
        if self._use_jax:
            import jax.numpy as jnp

            return jnp
        return np

    def tracer_via_instance_from(
        self,
        instance: af.ModelInstance,
    ) -> Tracer:
        """
        Create a `Tracer` from the galaxies contained in a model instance.

        Parameters
        ----------
        instance
            An instance of the model that is fitted to the data by this analysis (whose parameters may have been set
            via a non-linear search).

        Returns
        -------
        Tracer
            An instance of the Tracer class that is used to then fit the dataset.
        """
        if hasattr(instance, "perturb"):
            instance.galaxies.subhalo = instance.perturb

        # TODO : Need to think about how we do this without building it into the model attribute names.
        # TODO : A Subhalo class that extends the Galaxy class maybe?

        if hasattr(instance.galaxies, "subhalo"):
            subhalo_centre = tracer_util.grid_2d_at_redshift_from(
                galaxies=instance.galaxies,
                redshift=instance.galaxies.subhalo.redshift,
                grid=aa.Grid2DIrregular(
                    values=[instance.galaxies.subhalo.mass.centre], xp=self._xp
                ),
                cosmology=self.cosmology,
                xp=self._xp,
            )

            instance.galaxies.subhalo.mass.centre = tuple(subhalo_centre.in_list[0])

        if hasattr(instance, "cosmology"):
            cosmology = instance.cosmology
        else:
            cosmology = self.cosmology

        galaxy_list = list(instance.galaxies)

        if hasattr(instance, "extra_galaxies"):
            if getattr(instance, "extra_galaxies", None) is not None:
                galaxy_list += list(instance.extra_galaxies)

        if hasattr(instance, "scaling_galaxies"):
            if getattr(instance, "scaling_galaxies", None) is not None:
                galaxy_list += list(instance.scaling_galaxies)

        return Tracer(
            galaxies=galaxy_list,
            cosmology=cosmology,
        )

    def log_likelihood_penalty_from(
        self,
        instance: af.ModelInstance,
    ) -> Optional[float]:
        """
        Call the positions overwrite log likelihood function, which add a penalty term to the likelihood if the
        positions of the multiple images of the lensed source do not trace close to one another in the
        source plane.

        This function handles a number of exceptions which may occur when calling the overwrite function via the
        `PositionsLikelihood` class, so that they do not need to be handled individually for each `Analysis` class.

        Parameters
        ----------
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).

        Returns
        -------
        The penalty value of the positions log likelihood, if the positions do not trace close in the source plane,
        else a None is returned to indicate there is no penalty.
        """
        log_likelihood_penalty = self._xp.array(0.0)

        if self.positions_likelihood_list is not None:

            for positions_likelihood in self.positions_likelihood_list:

                if positions_likelihood is not None:

                    log_likelihood_penalty = (
                        positions_likelihood.log_likelihood_penalty_from(
                            instance=instance, analysis=self, xp=self._xp
                        )
                    )

                    log_likelihood_penalty += log_likelihood_penalty

            return log_likelihood_penalty

        return log_likelihood_penalty
