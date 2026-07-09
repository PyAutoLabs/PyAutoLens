"""
Analysis class for fitting a ``Tracer`` model to a weak-lensing shear catalogue.

``AnalysisWeak`` implements the ``log_likelihood_function`` called by a ``PyAutoFit``
non-linear search at each iteration.  It:

1. Constructs a ``Tracer`` from the current model instance.
2. Calls ``FitWeak`` to compare the tracer's model shear field (evaluated at the
   catalogue's galaxy positions via ``LensCalc.shear_yx_2d_via_hessian_from``) against
   the observed ``WeakDataset``.
3. Returns the fit's log likelihood as the figure of merit.

It also manages result output (``ResultWeak``) and on-the-fly visualisation
(``VisualizerWeak``).
"""
import autofit as af
import autogalaxy as ag

from autogalaxy.analysis.analysis.analysis import Analysis as AgAnalysis

from autolens.analysis.analysis.lens import AnalysisLens
from autolens.weak.dataset import WeakDataset
from autolens.weak.fit import FitWeak
from autolens.weak.model.result import ResultWeak
from autolens.weak.model.visualizer import VisualizerWeak


class AnalysisWeak(AgAnalysis, AnalysisLens):
    Visualizer = VisualizerWeak
    Result = ResultWeak

    def __init__(
        self,
        dataset: WeakDataset,
        cosmology: ag.cosmo.LensingCosmology = None,
        title_prefix: str = None,
        use_jax: bool = False,
        **kwargs,
    ):
        """
        Fits a lens model to a weak-lensing shear catalogue via a non-linear search.

        The `Analysis` class defines the `log_likelihood_function` which fits the model to the dataset and returns the
        log likelihood value defining how well the model fitted the data.

        It handles many other tasks, such as visualization, outputting results to hard-disk and storing results in
        a format that can be loaded after the model-fit is complete.

        This class is used for model-fits which fit lens mass models to `WeakDataset` shear catalogues — the
        weak-lensing analogue of `AnalysisImaging` / `AnalysisPoint`. Each background galaxy in the catalogue
        contributes two independent shear measurements (gamma_1 and gamma_2), which `FitWeak` compares against
        the model shear field of the `Tracer`.

        `use_jax` defaults to `False`, the conservative choice for the newest Analysis class; pass
        `use_jax=True` to run the `xp`-threaded fit path with `FitWeak` pytree registration (validated by
        the `autolens_workspace_test` weak vmap-parity script).

        Parameters
        ----------
        dataset
            The `WeakDataset` that is fitted by the model, containing the observed per-galaxy shear
            measurements, their positions and the per-galaxy noise.
        cosmology
            The Cosmology assumed for this analysis.
        title_prefix
            A string that is added before the title of all figures output by visualization, for example to
            put the name of the dataset and galaxy in the title.
        """
        super().__init__(cosmology=cosmology, use_jax=use_jax, **kwargs)

        AnalysisLens.__init__(self=self, cosmology=cosmology, use_jax=use_jax)

        self.dataset = dataset

        self.title_prefix = title_prefix

    def log_likelihood_function(self, instance):
        """
        Given an instance of the model, where the model parameters are set via a non-linear search, fit the model
        instance to the weak-lensing shear catalogue.

        This function returns a log likelihood which is used by the non-linear search to guide the model-fit.

        For this analysis class, this function performs the following steps:

        1) Extracts all galaxies from the model instance and sets up a `Tracer`, which includes ordering the galaxies
           by redshift to set up each `Plane`.

        2) Uses the `Tracer` to create a `FitWeak` object, which evaluates the tracer's shear field at the
           catalogue's galaxy positions (via the same `LensCalc.shear_yx_2d_via_hessian_from` primitive the
           `SimulatorShearYX` uses) and compares it to the observed shears.

        3) Returns the fit's log likelihood — a Gaussian likelihood over the N x 2 independent shear components.

        Parameters
        ----------
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).

        Returns
        -------
        float
            The log likelihood indicating how well this model instance fitted the weak-lensing data.
        """
        return self.fit_from(instance=instance).log_likelihood

    def fit_from(self, instance) -> FitWeak:
        """
        Given a model instance create a `FitWeak` object.

        This function is used in the `log_likelihood_function` to fit the model to the weak-lensing data and
        compute the log likelihood.

        Parameters
        ----------
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).

        Returns
        -------
        The fit of the lens model to the weak-lensing shear catalogue.
        """
        if self._use_jax:
            self._register_fit_weak_pytrees()

        tracer = self.tracer_via_instance_from(
            instance=instance,
        )

        return FitWeak(
            dataset=self.dataset,
            tracer=tracer,
            xp=self._xp,
        )

    @staticmethod
    def _register_fit_weak_pytrees() -> None:
        """Register every type reachable from a ``FitWeak`` return value so
        ``jax.jit(fit_from)`` can flatten its output.

        ``dataset`` and ``_xp`` are constants per analysis — ride as aux so JAX does
        not recurse into them (the cached ``_redshift_scale_factors`` derives purely
        from the dataset and plane redshifts, so it stays concrete). ``tracer`` is
        dynamic per fit.
        """
        from autoarray.abstract_ndarray import register_instance_pytree
        from autolens.lens.tracer import Tracer

        register_instance_pytree(
            FitWeak,
            no_flatten=("dataset", "_xp", "_redshift_scale_factors"),
        )
        register_instance_pytree(Tracer, no_flatten=("cosmology",))

    def save_attributes(self, paths: af.DirectoryPaths):
        """
        Before the non-linear search begins, this routine saves attributes of the `Analysis` object to the `files`
        folder such that they can be loaded after the analysis using PyAutoFit's database and aggregator tools.

        For this analysis, it outputs the following:

        - The weak-lensing shear catalogue as a readable .json file.

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
