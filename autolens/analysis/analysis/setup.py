"""
The compute `setup` of the lens analyses that have no dataset operator (point-source and weak lensing),
and the guard that stops a setup being silently ignored inside a factor graph.

The imaging and interferometer analyses resolve their setup through the **PyAutoGalaxy** base class
(`autogalaxy.analysis.analysis.dataset.AnalysisDataset.resolve_setup`), which also applies it to the
dataset (attaching the sparse operator it names). `AnalysisPoint` and `AnalysisWeak` do not fit an
`aa.Imaging` / `aa.Interferometer`, so their setup has no operator to attach (operator `"na"` or
`"dense"`); `AnalysisSetupNoOperator` resolves and applies the rest of it (backend, threads, banner,
`files/setup.json`), reusing the helpers of `autogalaxy.analysis.setup_util`.

`raise_if_setup_unresolved` is called at the top of every lens `log_likelihood_function`. A setup is
resolved in `modify_before_fit`, which a `FactorGraphModel` does not call on its factors, so a setup
given to an analysis fitted inside a factor graph would otherwise never be applied.
"""

import logging
from typing import Optional, Tuple

import autofit as af
import autoarray as aa

from autogalaxy.analysis import setup_util

from autolens import exc

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


SETUP_UNRESOLVED_MESSAGE = """
The {cls} was given `setup={setup_input!r}`, but the setup was never resolved, because
`modify_before_fit` was not called before the likelihood was evaluated.

This happens when the analysis is fitted inside a factor graph (`af.FactorGraphModel` of
`af.AnalysisFactor`s): the factor graph does not call each factor's `modify_before_fit`, so the
setup would silently not be applied. Setups are not yet supported inside factor graphs (this is
planned for the PyAutoFit phase of the likelihood lifecycle, 5D).

Instead, apply the setup to each dataset yourself and choose the backend explicitly, e.g.

    dataset = dataset.apply_setup("numba_cpu_sparse_fp64")
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)

(`use_jax=True` for a `jax_*` setup), and do not pass `setup=` to the analysis.
"""


def raise_if_setup_unresolved(analysis):
    """
    Raise an `exc.SetupException` when an analysis was given a `setup` that was never resolved.

    A setup is resolved in `modify_before_fit`, which a non-linear search calls before sampling. A
    `FactorGraphModel` does not call it on its factors, so a factor given `setup=` would otherwise be
    fitted with its constructor backend and the setup silently ignored.

    This is a cheap Python attribute check, safe under JAX tracing. It is called at the top of each
    `log_likelihood_function`, outside the `raise_fit_exception` wrapper, so it stops the fit instead of
    being turned into a resample. It is not called by `fit_from`, so visualization, results and the
    aggregator are unaffected.
    """
    setup_input = getattr(analysis, "setup_input", None)

    if setup_input is None or getattr(analysis, "setup", None) is not None:
        return

    if isinstance(setup_input, aa.Setup):
        setup_input = setup_input.setup_id

    raise exc.SetupException(
        SETUP_UNRESOLVED_MESSAGE.format(
            cls=type(analysis).__name__, setup_input=setup_input
        )
    )


def setup_environment():
    """
    The detected hardware (`autonerves.environment.Environment`), or `None` if detection fails.
    """
    try:
        from autonerves.environment import Environment

        return Environment.detect()
    except Exception as e:
        logger.debug(f"Setup: hardware detection failed: {e}")
        return None


class AnalysisSetupNoOperator:
    """
    Mixin resolving the compute `setup` of an analysis whose dataset has no operator to attach
    (`AnalysisPoint`, `AnalysisWeak`).

    The subclass calls `_init_setup(setup)` in its constructor, `resolve_setup(model)` first in its
    `modify_before_fit`, `save_setup(paths)` in its `save_attributes`, and implements `setup_key_from`
    (the `(dataset_type, variant)` rule key of `aa.Setup.auto`).
    """

    def _init_setup(self, setup):
        """
        Store the normalised `setup` input (`None`, `"auto"` or an `aa.Setup`, see
        `setup_util.setup_input_from`); the resolved setup is set by `resolve_setup`.
        """
        self.setup_input = setup_util.setup_input_from(setup)
        self.setup = None
        self.setup_chosen_by = None

    def setup_key_from(self, model: af.Collection) -> Tuple[str, str]:
        """
        The `(dataset_type, variant)` rule key of `aa.Setup.auto` for this analysis and model.
        """
        raise NotImplementedError

    def resolve_setup(self, model: af.Collection):
        """
        Resolve the `setup` input of the analysis against the model and apply it.

        With no `setup` input nothing changes: one log line reports what `setup="auto"` would choose (any
        error working that out is ignored).

        With `setup="auto"` the setup is chosen by `aa.Setup.auto` from `setup_key_from(model)`;
        `PYAUTO_SETUP=<setup_id>` overrides it. An explicit `aa.Setup` or setup_id is used as given. The
        resolved setup then:

        - sets the analysis backend (`_use_jax = setup.backend == "jax"`, falling back to numpy when JAX is
          disabled or not installed, as `af.Analysis` does); `_xp` follows, as it derives from `_use_jax`;
        - pins the thread pools (`setup.apply_threads()`);
        - is logged as the setup banner line and stored as `self.setup`.

        There is no dataset operator to attach or check.
        """
        setup_input = getattr(self, "setup_input", None)

        if setup_input is None:
            self._log_setup_auto_would_choose(model=model)
            return

        environment = setup_environment()

        if setup_input == setup_util.SETUP_AUTO:
            setup = aa.Setup.auto(*self.setup_key_from(model), environment=environment)
            chosen_by = setup_util.chosen_by_from(setup=setup, auto=True)
        else:
            setup = setup_input
            chosen_by = setup_util.chosen_by_from(setup=setup, auto=False)

        self._use_jax = setup_util.use_jax_from(setup.backend == "jax")

        setup.apply_threads()

        self.setup = setup
        self.setup_chosen_by = chosen_by

        logger.info(
            setup_util.setup_banner_from(
                setup=setup,
                chosen_by=chosen_by,
                environment=environment,
            )
        )

    def _log_setup_auto_would_choose(self, model: af.Collection):
        """
        Log one line reporting the setup `setup="auto"` would choose, applying nothing.
        """
        try:
            setup = aa.Setup.auto(
                *self.setup_key_from(model), environment=setup_environment()
            )
        except Exception as e:
            logger.debug(
                f"Setup: could not work out what setup='auto' would choose: {e}"
            )
            return

        logger.info(
            f"Setup: auto would choose {setup.setup_id} (rule {setup.rule_id}); "
            f'pass setup="auto" to use it'
        )

    def save_setup(self, paths: af.DirectoryPaths):
        """
        Write the resolved setup to `files/setup.json`, when a setup was resolved.
        """
        setup: Optional[aa.Setup] = getattr(self, "setup", None)

        if setup is not None:
            paths.save_json(
                name="setup",
                object_dict=setup_util.setup_dict_from(
                    setup=setup, chosen_by=self.setup_chosen_by
                ),
            )
