"""
Analysis class for fitting a ``Tracer`` model to a point-source dataset.

``AnalysisPoint`` implements the ``log_likelihood_function`` called by a ``PyAutoFit``
non-linear search at each iteration.  It:

1. Constructs a ``Tracer`` from the current model instance.
2. Calls ``FitPointDataset`` to fit the point-source positions (and optionally fluxes
   and time delays) using the ``PointSolver`` to find predicted image positions.
3. Optionally adds a position-based prior via ``PositionsLH`` that penalises models
   where image positions are not self-consistent.
4. Returns the total log likelihood as the figure of merit.

It also manages result output (``ResultPoint``) and on-the-fly visualisation
(``VisualizerPoint``).
"""

import logging
from typing import Optional
import os

import numpy as np
from autonerves.test_mode import is_test_mode

import autofit as af
import autogalaxy as ag

from autogalaxy.analysis.analysis.analysis import Analysis as AgAnalysis

from autolens.analysis.analysis.lens import AnalysisLens
from autogalaxy.analysis.setup_util import SETUP_DEFAULT, setup_and_use_jax_from

from autolens.analysis.analysis.setup import (
    AnalysisSetupNoOperator,
    raise_if_setup_unresolved,
)
from autolens.analysis.exceptions import raise_fit_exception
from autolens.analysis.runtime_estimate import AnalysisRuntimeEstimate
from autolens.point.fit.positions.image.pair_all import FitPositionsImagePairAllSolved
from autolens.point.fit.dataset import FitPointDataset
from autolens.point.fit.fluxes import FitFluxes
from autolens.point.fit.times_delays import FitTimeDelays
from autolens.point.dataset import PointDataset
from autolens.point.model.result import ResultPoint
from autolens.point.model.visualizer import VisualizerPoint
from autolens.point.solver import PointSolver

logger = logging.getLogger(__name__)


class AnalysisPoint(
    AnalysisRuntimeEstimate, AnalysisSetupNoOperator, AgAnalysis, AnalysisLens
):
    Visualizer = VisualizerPoint
    Result = ResultPoint

    # Gradient-based searches (`Fitness.grad`, `af.MultiStartAdam` & co.)
    # differentiate this likelihood in forward mode (`jax.jacfwd` over the flat
    # parameter vector). The point-source likelihood contains an inner
    # forward-mode lensing Hessian, so reverse mode would run
    # reverse-over-forward through every mass profile; forward mode measured
    # 2-4.5x faster and up to 8x faster to compile through 24 free parameters
    # (autolens_profiling #327/#331). Forward mode carries one tangent per free
    # parameter, so for a very large model (or memory-bound vmapped starts)
    # override it per search: `af.MultiStartAdam(gradient_mode="reverse")`.
    gradient_mode = "forward"

    def __init__(
        self,
        dataset: PointDataset,
        solver: PointSolver,
        fit_positions_cls=FitPositionsImagePairAllSolved,
        image=None,
        cosmology: ag.cosmo.LensingCosmology = None,
        title_prefix: str = None,
        use_jax: Optional[bool] = None,
        fit_flux_cls=FitFluxes,
        fit_time_delays_cls=FitTimeDelays,
        setup=SETUP_DEFAULT,
        **kwargs,
    ):
        """
        Fits a lens model to a point source dataset (e.g. positions, fluxes, time delays) via a non-linear search.

        The `Analysis` class defines the `log_likelihood_function` which fits the model to the dataset and returns the
        log likelihood value defining how well the model fitted the data.

        It handles many other tasks, such as visualization, outputting results to hard-disk and storing results in
        a format that can be loaded after the model-fit is complete.

        This class is used for model-fits which fit lens models to point datasets, which may include some combination
        of positions, fluxes and time-delays.

        This class stores the settings used to perform the model-fit for certain components of the model (e.g. a
        pixelization or inversion), the Cosmology used for the analysis and adapt images used for certain model
        classes.

        Parameters
        ----------
        dataset
            The `PointDataset` that is fitted by the model, which contains a combination of positions, fluxes and
            time-delays.
        solver
            Solves the lens equation in order to determine the image-plane positions of a point source by ray-tracing
            triangles to and from the source-plane. At construction, a warning is logged if observed positions lie
            within two initial triangle scales plus three position sigmas of a grid edge, or outside it. This
            diagnostic does not establish image completeness across the model prior and never changes the grid.
        fit_positions_cls
            The class used to fit the positions of the point source dataset, which could be an image-plane or
            source-plane chi-squared.
        fit_flux_cls
            The class used to fit the fluxes of the point source dataset, which could be a free-flux
            (`FitFluxes`) or analytically-solved-flux (`FitFluxesSolved`) fit.
        fit_time_delays_cls
            The class used to fit the time delays of the point source dataset, which could be the
            min-subtraction (`FitTimeDelays`) or analytically-solved-reference-time (`FitTimeDelaysSolved`) fit.
        image
            An optional observed image of the lensed point source (e.g. a quasar), which is used only for
            visualization: the observed and model point-source positions are plotted over it. It does not enter
            the likelihood.
        cosmology
            The Cosmology assumed for this analysis.
        title_prefix
            A string that is added before the title of all figures output by visualization, for example to
            put the name of the dataset and galaxy in the title.
        use_jax
            Whether the likelihood is evaluated with JAX. Passing it with no `setup` opts out of `setup="auto"`:
            the backend is `use_jax`, as with `setup=None`. With a `setup` the resolved setup sets the backend.
        setup
            The compute setup of the likelihood: `"auto"` (the default when neither `setup` nor `use_jax` is
            passed), `None` (the manual opt-out: backend `use_jax`, default `True`, plus one log line
            saying what `"auto"` would choose), an `aa.Setup` or a setup_id string. It is resolved in
            `modify_before_fit`; with `"auto"` the rule key is `point_source_source/source_plane` when `fit_positions_cls` is a source-plane fit (`FitPositionsSource` or a subclass) and `point_source_image/image_plane` otherwise (see `setup_key_from`). A resolved setup sets the backend, pins the thread pools, is
            logged as a banner line and is written to `files/setup.json`. Evaluating the likelihood before
            an explicitly passed setup is resolved (by the fit or `prepare_for_model(model)`) raises an
            `exc.SetupException`; before the default `"auto"` is resolved it runs with the constructor backend.
        """
        setup, use_jax, setup_defaulted = setup_and_use_jax_from(
            setup=setup, use_jax=use_jax, default_use_jax=True
        )

        super().__init__(cosmology=cosmology, use_jax=use_jax, **kwargs)

        # `super().__init__` (af.Analysis) is the single reader of the
        # disable-jax env var + the jax-availability check; forward the
        # resolved `self._use_jax` so `AnalysisLens` does not overwrite it.
        AnalysisLens.__init__(self=self, cosmology=cosmology, use_jax=self._use_jax)

        self.dataset = dataset
        self.image = image

        self.solver = solver
        self.fit_positions_cls = fit_positions_cls
        self.fit_flux_cls = fit_flux_cls
        self.fit_time_delays_cls = fit_time_delays_cls
        self.title_prefix = title_prefix
        self._init_setup(setup, setup_defaulted=setup_defaulted)
        self._check_solver_extent()

    def setup_key_from(self, model: af.Collection = None):
        """
        The `(dataset_type, variant)` rule key of `aa.Setup.auto` for this point-source analysis.

        The key follows the positions fit, which decides where the likelihood is evaluated:

        - `("point_source_source", "source_plane")` when `fit_positions_cls` is a source-plane fit
          (`FitPositionsSource`, `FitPositionsSourceSolved` or any other subclass of `FitPositionsSource`),
          which traces the observed positions to the source plane and needs no `PointSolver`.
        - `("point_source_image", "image_plane")` otherwise: the image-plane pair fits
          (`AbstractFitPositionsImagePair` subclasses), which solve for the model image positions with the
          `PointSolver`, and any custom or absent positions fit.

        The model does not change the key (the rule table has one variant per point-source likelihood).
        """
        from autolens.point.fit.positions.source.separations import FitPositionsSource

        fit_positions_cls = self.fit_positions_cls

        if isinstance(fit_positions_cls, type) and issubclass(
            fit_positions_cls, FitPositionsSource
        ):
            return ("point_source_source", "source_plane")

        return ("point_source_image", "image_plane")

    def modify_before_fit(self, paths: af.DirectoryPaths, model: af.Collection):
        """
        Called immediately before the non-linear search begins: resolves and applies the compute `setup`
        (see `AnalysisSetupNoOperator.resolve_setup`), before any other preparation.
        """
        self.resolve_setup(model=model)

        return super().modify_before_fit(paths=paths, model=model)

    def _check_solver_extent(self):
        """Diagnose observed-position coverage once, outside likelihood tracing.

        The margin is two initial triangle scales plus three position sigmas.
        This is a data-based hint, not a completeness test over the model prior.
        Custom solvers without rectangular bounds and absent/invalid diagnostic
        inputs are left to their existing validation paths.
        """
        positions = getattr(self.dataset, "positions", None)
        noise = getattr(self.dataset, "positions_noise_map", None)
        if positions is None or noise is None or self.solver is None:
            return
        try:
            bounds = np.asarray(
                [
                    self.solver.y_min,
                    self.solver.y_max,
                    self.solver.x_min,
                    self.solver.x_max,
                ],
                dtype=float,
            )
            scale = float(self.solver.scale)
            positions = np.asarray(getattr(positions, "array", positions), dtype=float)
            noise = np.asarray(getattr(noise, "array", noise), dtype=float)
            if positions.ndim != 2 or positions.shape[1] != 2 or not len(positions):
                return
            noise = np.broadcast_to(noise, (len(positions),))
        except (AttributeError, TypeError, ValueError):
            return
        if (
            not np.all(np.isfinite(bounds))
            or not np.isfinite(scale)
            or scale <= 0
            or not np.all(np.isfinite(positions))
            or not np.all(np.isfinite(noise))
            or np.any(noise < 0)
        ):
            return
        lower, upper = bounds[[0, 2]], bounds[[1, 3]]
        if np.any(upper <= lower):
            return
        margin = (2 * scale + 3 * noise)[:, None]
        if np.any(positions - lower <= margin) or np.any(upper - positions <= margin):
            logger.warning(
                "PointSolver extent %s is too close to, or excludes, observed "
                "positions for dataset %r (margin: 2 * initial scale + 3 * "
                "position sigma). Images near or beyond the grid edge may be "
                "missed. Enlarge the solver grid and check the model prior: "
                "covering observed positions does not guarantee image "
                "completeness across the model prior.",
                tuple(bounds.tolist()),
                getattr(self.dataset, "name", None),
            )
        elif not is_test_mode() and os.environ.get("PYAUTO_SMALL_DATASETS") != "1":
            midpoint = (lower + upper) / 2
            required = np.max(np.abs(positions - midpoint) + margin, axis=0)
            if np.all((upper - lower) / 2 > 3 * required):
                logger.info(
                    "PointSolver extent for dataset %r is more than 3 times "
                    "the observed-position envelope on both axes, including "
                    "the safety margin. At fixed initial scale, initial grid "
                    "work grows with grid area. Consider a smaller grid only "
                    "after checking image completeness across the model prior.",
                    getattr(self.dataset, "name", None),
                )

    def log_likelihood_function(self, instance):
        """
        Given an instance of the model, where the model parameters are set via a non-linear search, fit the model
        instance to the point source dataset.

        This function returns a log likelihood which is used by the non-linear search to guide the model-fit.

        For this analysis class, this function performs the following steps:

        1) Extracts all galaxies from the model instance and set up a `Tracer`, which includes ordering the galaxies
           by redshift to set up each `Plane`.

        2) Use the `Tracer` and other attributes to create a `FitPointDataset` object, which performs the steps
           below to fit different components of the point source dataset.

        3) If the point source dataset has positions and model fits positions, perform this fit and compute the
           log likelihood. This calculation uses the `fit_positions_cls` object, which may be an image-plane or
           source-plane chi-squared.

        4) If the point source dataset has fluxes and model fits fluxes, perform this fit and compute the log likelihood.

        5) If the point source dataset has time-delays and model fits time-delays, perform this fit and compute the
           log likelihood [NOT SUPPORTED YET].

        6) Sum the log likelihoods of the positions, fluxes and time-delays (if they are fitted) to get the overall
           log likelihood of the model.

        Certain models will fail to fit the dataset and raise an exception. For example for ill defined mass models
        the `PointSolver` may find no solution. In such circumstances the model is discarded and its likelihood value
        is passed to the non-linear search in a way that it ignores it (for example, using a value of -1.0e99).

        Parameters
        ----------
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).

        Returns
        -------
        float
            The log likelihood indicating how well this model instance fitted the imaging data.
        """
        raise_if_setup_unresolved(analysis=self)

        if self._use_jax:
            return self.fit_from(instance=instance).log_likelihood

        try:
            return self.fit_from(instance=instance).log_likelihood
        except Exception as e:
            raise_fit_exception(e)

    def fit_from(
        self,
        instance,
    ) -> FitPointDataset:
        """
        Given a model instance create a `FitPointDataset` object.

        This function is used in the `log_likelihood_function` to fit the model to the imaging data and compute the
        log likelihood.

        Parameters
        ----------
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).

        Returns
        -------
        The fit of the lens model to the point source dataset.
        """

        if self._use_jax:
            self._register_fit_point_pytrees()

        tracer = self.tracer_via_instance_from(
            instance=instance,
        )

        return FitPointDataset(
            dataset=self.dataset,
            tracer=tracer,
            solver=self.solver,
            fit_positions_cls=self.fit_positions_cls,
            fit_flux_cls=self.fit_flux_cls,
            fit_time_delays_cls=self.fit_time_delays_cls,
            xp=self._xp,
        )

    @staticmethod
    def _register_fit_point_pytrees() -> None:
        """Register every type reachable from a ``FitPointDataset`` return value
        so ``jax.jit(fit_from)`` can flatten its output.

        ``dataset`` and ``solver`` are constants per analysis — ride as aux so
        JAX does not recurse into them. ``fit_positions_cls`` is a class reference
        (not a value) so must also ride as aux. ``tracer`` is dynamic per fit.
        """
        from autoarray.abstract_ndarray import register_instance_pytree
        from autolens.lens.tracer import Tracer
        from autolens.point.fit.positions.image.pair_all import (
            FitPositionsImagePairAll,
            FitPositionsImagePairAllSolved,
        )
        from autolens.point.fit.positions.image.pair_repeat import (
            FitPositionsImagePairRepeat,
            FitPositionsImagePairRepeatSolved,
        )
        from autolens.point.fit.positions.image.pair import FitPositionsImagePair
        from autolens.point.fit.positions.source.separations import (
            FitPositionsSource,
            FitPositionsSourceSolved,
        )
        from autolens.point.fit.fluxes import FitFluxesSolved
        from autolens.point.fit.times_delays import FitTimeDelaysSolved
        import autogalaxy as ag

        register_instance_pytree(
            FitPointDataset,
            no_flatten=(
                "dataset",
                "solver",
                "fit_positions_cls",
                "fit_flux_cls",
                "fit_time_delays_cls",
            ),
        )
        register_instance_pytree(Tracer, no_flatten=("cosmology",))
        # fit-point-pytree: observed data/noise are per-analysis constants; solver/name/use_jax are non-JAX
        register_instance_pytree(
            FitPositionsImagePairAll,
            no_flatten=("solver", "name", "use_jax", "_data", "_noise_map"),
        )
        # fit-point-pytree
        register_instance_pytree(
            FitPositionsImagePairRepeat,
            no_flatten=("solver", "name", "use_jax", "_data", "_noise_map"),
        )
        # fit-point-pytree
        register_instance_pytree(
            FitPositionsImagePair,
            no_flatten=("solver", "name", "use_jax", "_data", "_noise_map"),
        )
        # fit-point-pytree: no solver is used (source-plane / analytic fits), so only
        # name/use_jax/observed data+noise are non-JAX.
        register_instance_pytree(
            FitPositionsSource,
            no_flatten=("name", "use_jax", "_data", "_noise_map"),
        )
        register_instance_pytree(
            FitPositionsSourceSolved,
            no_flatten=("name", "use_jax", "_data", "_noise_map"),
        )
        # fit-point-pytree: solved image-plane variants still forward-solve via `solver`.
        register_instance_pytree(
            FitPositionsImagePairAllSolved,
            no_flatten=("solver", "name", "use_jax", "_data", "_noise_map"),
        )
        register_instance_pytree(
            FitPositionsImagePairRepeatSolved,
            no_flatten=("solver", "name", "use_jax", "_data", "_noise_map"),
        )
        # fit-point-pytree: flux/time-delay fits carry no solver, only observed positions.
        register_instance_pytree(
            FitFluxesSolved,
            no_flatten=("name", "use_jax", "_data", "_noise_map", "positions"),
        )
        register_instance_pytree(
            FitTimeDelaysSolved,
            no_flatten=("name", "use_jax", "_data", "_noise_map", "positions"),
        )
        # fit-point-pytree: ag.ps.Point / PointFlux / PointSolved are handled by
        # autofit.jax.pytrees.register_model before jit is called; skip here.

    def save_attributes(self, paths: af.DirectoryPaths):
        """
        Before the non-linear search begins, this routine saves attributes of the `Analysis` object to the `files`
        folder such that they can be loaded after the analysis using PyAutoFit's database and aggregator tools.

        For this analysis, it uses the `AnalysisDataset` object's method to output the following:

        - The dataset's point source dataset as a readable .json file.
        - The resolved compute setup (`setup.json`), when a `setup` was given.

        It is common for these attributes to be loaded by many of the template aggregator functions given in the
        `aggregator` modules. For example, when using the database tools to perform a fit, the default behaviour is for
        the dataset, settings and other attributes necessary to perform the fit to be loaded via the pickle files
        output by this function.

        Parameters
        ----------
        paths
            The paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization, and the pickled objects used by the aggregator output by this function.
        """
        ag.output_to_json(
            obj=self.dataset,
            file_path=paths._files_path / "dataset.json",
        )

        self.save_setup(paths=paths)
