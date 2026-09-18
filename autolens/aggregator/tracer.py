"""
Aggregator interface for loading ``Tracer`` objects from model-fit results.

After a **PyAutoLens** model-fit the best-fit galaxy model parameters are stored in the
output directory or SQLite database as part of the model JSON file.  This module
reconstructs ``Tracer`` instances from those stored parameters so that the full
ray-tracing model can be re-evaluated and inspected without re-running the fit.

Two public objects are provided:

- ``_tracer_from`` — a free function that accepts a single ``PyAutoFit`` ``Fit`` entry
  and returns the list of ``Tracer`` objects (one per summed ``Analysis``) for the
  requested model instance.
- ``TracerAgg`` — a ``PyAutoFit`` ``AggBase`` subclass wrapping an ``Aggregator`` that
  exposes a generator of tracer lists, one entry per stored model-fit.
"""
import logging
from typing import List, Optional

import autofit as af
import autoarray as aa
import autogalaxy as ag

from autolens.lens.tracer import Tracer

from autogalaxy.aggregator import agg_util
from autolens.lens import tracer_util

logger = logging.getLogger(__name__)


def _tracer_from(
    fit: af.Fit, instance: Optional[af.ModelInstance] = None
) -> List[Tracer]:
    """
    Returns a list of `Tracer` objects from a `PyAutoFit` loaded directory `Fit` or sqlite database `Fit` object.

    The results of a model-fit can be loaded from hard-disk or stored in a sqlite database, including the following
    attributes of the fit:

    - The model and its best fit parameters (e.g. `model.json`).

    Each individual attribute can be loaded from the database via the `fit.value()` method.

    This method combines this attributesand returns a `Tracer` object for a given non-linear search sample
    (e.g. the maximum likelihood model).

    If multiple `Tracer` objects were fitted simultaneously via multiple analysis, the instance is iterated over as
    a list such that a list of `Tracer` objects with parameters updated for each analysis are returned. This means
    fits using a single analysis are wrapped in a list to prodcue a consistent API.

    Parameters
    ----------
    fit
        A `PyAutoFit` `Fit` object which contains the results of a model-fit as an entry which has been loaded from
        an output directory or from an sqlite database..
    instance
        A manual instance that overwrites the max log likelihood instance in fit (e.g. for drawing the instance
        randomly from the PDF).
    """

    instance_list = agg_util.instance_list_from(fit=fit, instance=instance)

    tracer_list = []

    for instance in instance_list:

        try:
            cosmology = instance.cosmology
        except AttributeError:
            cosmology = fit.value(name="cosmology")

        if cosmology is None:
            cosmology = ag.cosmo.Planck15()

        # TODO : These are ugly as hell (>_<)

        if hasattr(instance, "perturb"):
            instance.galaxies.subhalo = instance.perturb

        if hasattr(instance.galaxies, "subhalo"):
            subhalo_centre = tracer_util.grid_2d_at_redshift_from(
                galaxies=instance.galaxies,
                redshift=instance.galaxies.subhalo.redshift,
                grid=aa.Grid2DIrregular(values=[instance.galaxies.subhalo.mass.centre]),
                cosmology=cosmology,
            )

            instance.galaxies.subhalo.mass.centre = tuple(subhalo_centre.in_list[0])

        galaxy_list = list(instance.galaxies)

        if hasattr(instance, "extra_galaxies"):
            if instance.extra_galaxies is not None:
                galaxy_list += list(instance.extra_galaxies)

        if hasattr(instance, "scaling_galaxies"):
            if instance.scaling_galaxies is not None:
                galaxy_list += list(instance.scaling_galaxies)

        # Mirrors the `fields` fold in `AnalysisLens.tracer_via_instance_from`: a model may declare an
        # external field alongside its galaxies, and a `MassField` is not a galaxy.
        fields = tracer_util.fields_list_from(getattr(instance, "fields", None))

        tracer = Tracer(galaxies=galaxy_list, cosmology=cosmology, fields=fields)

        tracer_list.append(tracer)

    return tracer_list


class TracerAgg(af.AggBase):
    """
    Interfaces with an `PyAutoFit` aggregator object to create instances of `Tracer` objects from the results
    of a model-fit.

    The results of a model-fit can be loaded from hard-disk or stored in a sqlite database, including the following
    attributes of the fit:

    - The model and its best fit parameters (e.g. `model.json`).
    - The adapt images associated with adaptive galaxy features (`adapt` folder).

    The `aggregator` contains the path to each of these files, and they can be loaded individually. This class
    can load them all at once and create an `Tracer` object via the `_tracer_from` method.

    This class's methods returns generators which create the instances of the `Tracer` objects. This ensures
    that large sets of results can be efficiently loaded from the hard-disk and do not require storing all
    `Tracer` instances in the memory at once.

    For example, if the `aggregator` contains 3 model-fits, this class can be used to create a generator which
    creates instances of the corresponding 3 `Tracer` objects.

    If multiple `Tracer` objects were fitted simultaneously via analysis summing, the `fit.child_values()` method
    is instead used to load lists of Tracers. This is necessary if each Tracer has different galaxies (e.g. certain
    parameters vary across each dataset and `Analysis` object).

    This can be done manually, but this object provides a more concise API.

    Parameters
    ----------
    aggregator
        A `PyAutoFit` aggregator object which can load the results of model-fits.
    """

    def object_via_gen_from(
        self, fit, instance: Optional[af.ModelInstance] = None
    ) -> List[Tracer]:
        """
        Returns a generator of `Tracer` objects from an input aggregator.

        See `__init__` for a description of how the `Tracer` objects are created by this method.

        Parameters
        ----------
        fit
            A `PyAutoFit` `Fit` object which contains the results of a model-fit as an entry which has been loaded from
        an output directory or from an sqlite database..
        galaxies
            A list of galaxies corresponding to a sample of a non-linear search and model-fit.
        """
        return _tracer_from(fit=fit, instance=instance)
