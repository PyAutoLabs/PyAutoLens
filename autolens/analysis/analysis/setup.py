"""
The compute `setup` of the lens analyses that have no dataset operator (point-source and weak lensing),
and the guard that stops a likelihood being evaluated before its setup has been resolved.

The imaging and interferometer analyses resolve their setup through the **PyAutoGalaxy** base class
(`autogalaxy.analysis.analysis.dataset.AnalysisDataset.resolve_setup`), which also applies it to the
dataset (attaching the sparse operator it names). `AnalysisPoint` and `AnalysisWeak` do not fit an
`aa.Imaging` / `aa.Interferometer`, so their setup has no operator to attach (operator `"na"` or
`"dense"`); `AnalysisSetupNoOperator` resolves and applies the rest of it (backend, threads, banner,
`files/setup.json`) through PyAutoGalaxy's shared `AnalysisSetupMixin`
(`autogalaxy.analysis.analysis.setup`).

`raise_if_setup_unresolved` is called at the top of every lens `log_likelihood_function`. A setup
resolves when the fit starts (`modify_before_fit`, which a search calls on the analysis and a
`FactorGraphModel` forwards to each factor) or on a pre-fit call that receives a model
(`analysis.prepare_for_model(model)`, called by e.g. `print_vram_use(model=...)`). A likelihood evaluated
before either would otherwise run with the constructor backend and the setup silently ignored.
"""

import logging
from typing import Optional, Tuple

import autofit as af
import autoarray as aa

from autogalaxy.analysis.analysis.setup import AnalysisSetupMixin
from autogalaxy.analysis.setup_util import setup_environment  # noqa: F401 (kept importable from here)

from autolens import exc

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


SETUP_UNRESOLVED_MESSAGE = """
The {cls} was given `setup={setup_input!r}`, but its likelihood was evaluated before the setup was
resolved.

The setup resolves when the fit starts (`search.fit` calls `modify_before_fit`, which a factor graph
forwards to each factor's analysis) or on a pre-fit call that receives a model, e.g.
`analysis.print_vram_use(model=model, batch_size=...)`. The likelihood was evaluated before the fit
started and before either.

Call `analysis.prepare_for_model(model)` first (it resolves the setup from the model you will fit),
or run the fit. Alternatively, apply the setup to the dataset yourself and choose the backend
explicitly, e.g.

    dataset = dataset.apply_setup("numba_cpu_sparse_fp64")
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)

(`use_jax=True` for a `jax_*` setup), and do not pass `setup=` to the analysis.
"""


def raise_if_setup_unresolved(analysis):
    """
    Raise an `exc.SetupException` when an analysis was explicitly given a `setup` that was never resolved.

    The default `setup="auto"` (an analysis constructed with neither `setup` nor `use_jax`) does not raise:
    before it resolves, the likelihood runs with the constructor backend, so direct pre-fit calls such as
    `analysis.log_likelihood_function(instance)` keep working.

    A setup is resolved in `modify_before_fit`, which a non-linear search calls before sampling (and a
    `FactorGraphModel` forwards to each factor), or by `prepare_for_model(model)` on pre-fit calls that
    receive a model (e.g. `print_vram_use`). A likelihood evaluated before either would otherwise use the
    constructor backend with the setup silently ignored.

    This is a cheap Python attribute check, safe under JAX tracing. It is called at the top of each
    `log_likelihood_function`, outside the `raise_fit_exception` wrapper, so it stops the fit instead of
    being turned into a resample. It is not called by `fit_from`, so visualization, results and the
    aggregator are unaffected.
    """
    setup_input = getattr(analysis, "setup_input", None)

    if setup_input is None or getattr(analysis, "setup", None) is not None:
        return

    # The default `setup="auto"` (neither `setup` nor `use_jax` passed) does not stop a pre-fit evaluation:
    # it runs with the constructor backend, as the analyses did before `"auto"` became the default.
    if getattr(analysis, "setup_defaulted", False):
        return

    if isinstance(setup_input, aa.Setup):
        setup_input = setup_input.setup_id

    raise exc.SetupException(
        SETUP_UNRESOLVED_MESSAGE.format(
            cls=type(analysis).__name__, setup_input=setup_input
        )
    )


class AnalysisSetupNoOperator(AnalysisSetupMixin):
    """
    Mixin resolving the compute `setup` of an analysis whose dataset has no operator to attach
    (`AnalysisPoint`, `AnalysisWeak`).

    It is a thin subclass of PyAutoGalaxy's `AnalysisSetupMixin`, which holds the one shared resolution
    path (`_init_setup`, `resolve_setup`, `_log_setup_auto_would_choose`, `save_setup`). The subclass
    calls `_init_setup(setup)` in its constructor, `resolve_setup(model)` first in its
    `modify_before_fit`, `save_setup(paths)` in its `save_attributes`, and implements `setup_key_from`
    (the `(dataset_type, variant)` rule key of `aa.Setup.auto`). There is no dataset operator to attach
    or check, so `_apply_resolved_setup` keeps the mixin's do-nothing default and the banner has no note.
    """

    def setup_key_from(self, model: af.Collection) -> Tuple[str, str]:
        """
        The `(dataset_type, variant)` rule key of `aa.Setup.auto` for this analysis and model.
        """
        raise NotImplementedError

    def setup_key_and_note_from(
        self, model: af.Collection
    ) -> Tuple[Tuple[str, str], Optional[str]]:
        """
        The rule key of `setup_key_from(model)`, with no banner note.
        """
        return self.setup_key_from(model), None
