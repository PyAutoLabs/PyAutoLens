"""
Analysis class for fitting a ``Tracer`` lens model to an imaging dataset.

``AnalysisImaging`` implements the ``log_likelihood_function`` that a ``PyAutoFit``
non-linear search calls on each iteration.  It:

1. Constructs a ``Tracer`` from the current model instance.
2. Optionally applies adaptive galaxy images to linear components.
3. Calls ``FitImaging`` to evaluate the log likelihood.
4. Returns the figure of merit (log likelihood or log evidence).

It also manages result output (``ResultImaging``), on-the-fly visualisation
(``VisualizerImaging``), and position-based priors via ``PositionLikelihood``.
"""
import logging

import autoarray as aa
import autofit as af
import autogalaxy as ag

from autoconf.fitsable import hdu_list_for_output_from

from autolens.analysis.analysis.dataset import AnalysisDataset
from autolens.analysis.latent import LatentLens
from autolens.imaging.model.result import ResultImaging
from autolens.imaging.model.visualizer import VisualizerImaging
from autolens.imaging.fit_imaging import FitImaging

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


_FIT_IMAGING_PYTREES_REGISTERED = False


class AnalysisImaging(AnalysisDataset):

    Result = ResultImaging
    Visualizer = VisualizerImaging
    Latent = LatentLens

    def __init__(
        self,
        dataset,
        positions_likelihood_list=None,
        adapt_images: ag.AdaptImages = None,
        cosmology: ag.cosmo.LensingCosmology = None,
        settings=None,
        raise_inversion_positions_likelihood_exception: bool = True,
        title_prefix: str = None,
        use_jax: bool = True,
        shared_preloads: bool = False,
        **kwargs,
    ):
        """
        Fits a lens model to an imaging dataset via a non-linear search (see `AnalysisDataset` for
        the full docstring of the shared parameters).

        Parameters
        ----------
        shared_preloads
            Opts this analysis into the cross-factor shared-state mechanism of a `FactorGraphModel`
            (see `shared_state_from`). Set this to `True` only when this analysis is one of many
            exposures of the same lens (e.g. multi-exposure imaging with per-exposure pixel offsets)
            sharing an identical lens model, so the exposure-invariant source-plane mesh geometry
            can be computed once and reused by every exposure. `False` by default, leaving the
            standard per-analysis behaviour unchanged.
        """
        super().__init__(
            dataset=dataset,
            positions_likelihood_list=positions_likelihood_list,
            adapt_images=adapt_images,
            cosmology=cosmology,
            settings=settings,
            raise_inversion_positions_likelihood_exception=raise_inversion_positions_likelihood_exception,
            title_prefix=title_prefix,
            use_jax=use_jax,
            **kwargs,
        )

        self.shared_preloads = shared_preloads

    def log_likelihood_function(self, instance: af.ModelInstance, shared=None) -> float:
        """
        Given an instance of the model, where the model parameters are set via a non-linear search, fit the model
        instance to the imaging dataset.

        This function returns a log likelihood which is used by the non-linear search to guide the model-fit.

        For this analysis class, this function performs the following steps:

        1) If the analysis has a adapt image, associated the model galaxy images of this dataset to the galaxies in
           the model instance.

        2) Extract attributes which model aspects of the data reductions, like the scaling the background sky
           and background noise.

        3) Extracts all galaxies from the model instance and set up a `Tracer`, which includes ordering the galaxies
           by redshift to set up each `Plane`.

        4) Use the `Tracer` and other attributes to create a `FitImaging` object, which performs steps such as creating
           model images of every galaxy in the tracer, blurring them with the imaging dataset's PSF and computing
           residuals, a chi-squared statistic and the log likelihood.

        Certain models will fail to fit the dataset and raise an exception. For example if an `Inversion` is used, the
        linear algebra calculation may be invalid and raise an Exception. In such circumstances the model is discarded
        and its likelihood value is passed to the non-linear search in a way that it ignores it (for example, using a
        value of -1.0e99).

        Parameters
        ----------
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).
        shared
            The cross-factor shared state of a `FactorGraphModel`, computed once per evaluation by the lead
            factor's `shared_state_from` (see that method). For this analysis it is a `PreloadsImaging`
            carrying the exposure-invariant source-plane mesh geometry; when provided it is reused by the fit
            instead of being recomputed. `None` (the default, e.g. a standalone fit) leaves behaviour unchanged.

        Returns
        -------
        float
            The log likelihood indicating how well this model instance fitted the imaging data.
        """

        log_likelihood_penalty = self.log_likelihood_penalty_from(
            instance=instance,
        )

        if self._use_jax:
            return (
                self.fit_from(instance=instance, preloads=shared).figure_of_merit
                - log_likelihood_penalty
            )

        try:
            return (
                self.fit_from(instance=instance, preloads=shared).figure_of_merit
                - log_likelihood_penalty
            )
        except Exception as e:
            raise af.exc.FitException

    def shared_state_from(self, instance: af.ModelInstance):
        """
        Compute the exposure-invariant source-plane mesh geometry once so it can be shared across the factors
        of a multi-exposure `FactorGraphModel` (see `autofit.Analysis.shared_state_from`).

        When `shared_preloads` is set, every factor of the graph is an exposure of the same lens sharing an
        identical lens model, so the source-plane mesh (the image-mesh centres of this lead exposure,
        ray-traced through the shared lens model) is built once here and returned inside a `PreloadsImaging`,
        which `FactorGraphModel` forwards as the `shared` argument to every factor's
        `log_likelihood_function`. Each exposure then maps its own (offset) data grid onto the shared mesh
        instead of computing its own image-mesh and mesh ray-trace, so every exposure reconstructs on an
        identical source-pixel grid.

        Unlike the interferometer datacube case, the mapper, mapping matrix, curvature matrix and
        regularization matrix are NOT shared — per-exposure PSFs and pixel offsets make the first three
        per-dataset, and regularization may adapt to per-exposure data.

        Returns `None` when the analysis has not opted in (`shared_preloads=False`) or when the model performs
        no inversion, in which case no state is shared and every factor fits as normal.

        The caller is responsible for the invariance contract: only enable `shared_preloads` when the factors
        genuinely share the lens model, so the source-plane mesh really is exposure-invariant. The lead
        factor's own `DatasetModel` offset (if any) is applied when the mesh is traced, so the mesh is defined
        in the lead exposure's frame.
        """
        if not self.shared_preloads:
            return None

        fit = self.fit_from(instance=instance)

        if not fit.perform_inversion:
            return None

        tracer_to_inversion = fit.tracer_to_inversion

        return aa.PreloadsImaging(
            source_plane_mesh_grid=tracer_to_inversion.traced_mesh_grid_pg_list,
            image_plane_mesh_grid=tracer_to_inversion.image_plane_mesh_grid_pg_list,
        )

    def fit_from(
        self,
        instance: af.ModelInstance,
        preloads=None,
    ) -> FitImaging:
        """
        Given a model instance create a `FitImaging` object.

        This function is used in the `log_likelihood_function` to fit the model to the imaging data and compute the
        log likelihood.

        Parameters
        ----------
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).
        preloads
            An optional `PreloadsImaging` carrying the exposure-invariant source-plane mesh geometry,
            computed once and reused by the fit instead of being rebuilt. Supplied by the multi-exposure
            shared-state path (see `shared_state_from`); `None` (the default) fits as normal.

        Returns
        -------
        FitImaging
            The fit of the plane to the imaging dataset, which includes the log likelihood.
        """

        if self._use_jax:
            self._register_fit_imaging_pytrees()

        tracer = self.tracer_via_instance_from(
            instance=instance,
        )

        dataset_model = self.dataset_model_via_instance_from(instance=instance)

        adapt_images = self.adapt_images_via_instance_from(
            instance=instance,
            dataset_model=dataset_model,
            galaxies=tracer.galaxies,
            xp=self._xp,
        )

        return FitImaging(
            dataset=self.dataset,
            tracer=tracer,
            dataset_model=dataset_model,
            adapt_images=adapt_images,
            settings=self.settings,
            xp=self._xp,
            preloads=preloads,
        )

    def save_attributes(self, paths: af.DirectoryPaths):
        """
        Before the non-linear search begins, output the imaging ``dataset.fits``
        to the ``files`` folder so the aggregator loaders (e.g. ``ImagingAgg``,
        ``agg_util.mask_header_from``) can always reload the dataset via
        ``fit.value(name="dataset")``, independently of whether the visualization
        ``fits_dataset`` output ran. The plotter interface also writes this file
        to the ``image`` folder for inspection, but that write is gated on
        visualization settings and is not guaranteed for every fit.
        """
        super().save_attributes(paths=paths)

        image_list = [
            self.dataset.data.native_for_fits,
            self.dataset.noise_map.native_for_fits,
            self.dataset.psf.kernel.native_for_fits,
            self.dataset.grids.lp.over_sample_size.native_for_fits.astype("float"),
            self.dataset.grids.pixelization.over_sample_size.native_for_fits.astype(
                "float"
            ),
        ]

        paths.save_fits(
            name="dataset",
            fits=hdu_list_for_output_from(
                values_list=[image_list[0].mask.astype("float")] + image_list,
                ext_name_list=[
                    "mask",
                    "data",
                    "noise_map",
                    "psf",
                    "over_sample_size_lp",
                    "over_sample_size_pixelization",
                ],
                header_dict=self.dataset.mask.header_dict,
            ),
        )

    @staticmethod
    def _register_fit_imaging_pytrees() -> None:
        """Register every type reachable from a ``FitImaging`` return value
        so ``jax.jit(fit_from)`` can flatten its output.

        ``dataset``, ``adapt_images`` and ``settings`` are constants per
        analysis — ride as aux so JAX does not recurse into them. Everything
        else (``tracer``, ``dataset_model`` and the autoarray wrappers they
        carry) is dynamic per fit.

        Idempotent — guarded by the module-level
        ``_FIT_IMAGING_PYTREES_REGISTERED`` flag. ``DatasetModel`` and
        ``Tracer`` may already be registered by
        ``autofit.jax.pytrees.register_model`` (its
        ``_REGISTERED_INSTANCE_CLASSES`` set is independent of autoarray's
        ``_pytree_registered_classes``); cross-populate so
        ``register_instance_pytree`` short-circuits. Mirrors the defense in
        ``autogalaxy/ellipse/model/analysis.py``.
        """
        global _FIT_IMAGING_PYTREES_REGISTERED
        if _FIT_IMAGING_PYTREES_REGISTERED:
            return

        from autoarray.abstract_ndarray import (
            register_instance_pytree,
            _pytree_registered_classes,
        )
        from autoarray.dataset.dataset_model import DatasetModel
        from autolens.lens.tracer import Tracer

        try:
            from autofit.jax.pytrees import (
                _REGISTERED_INSTANCE_CLASSES as _af_registered,
            )
        except ImportError:
            _af_registered = set()

        for cls in (DatasetModel, Tracer):
            if cls in _af_registered:
                _pytree_registered_classes.add(cls)

        register_instance_pytree(
            FitImaging,
            no_flatten=("dataset", "adapt_images", "settings", "preloads"),
        )
        register_instance_pytree(DatasetModel)
        # ``cosmology`` is a fixed physical constant per fit; ride as aux.
        register_instance_pytree(Tracer, no_flatten=("cosmology",))

        _FIT_IMAGING_PYTREES_REGISTERED = True

