import os
from typing import NoReturn

import autofit as af


def raise_fit_exception(exception: Exception) -> NoReturn:
    """
    Re-raise a numpy-path fit failure as an ``af.exc.FitException`` so the
    non-linear search discards and resamples the model.

    This is path-parity with the JAX branch: there, a pathological model (e.g. a
    non-positive-definite inversion) yields ``NaN`` which ``autofit`` maps to a
    resample, whereas the numpy branch raises (e.g. ``numpy.linalg.LinAlgError``).
    Wrapping the raise in ``FitException`` — which ``autofit`` catches and
    resamples — makes both branches behave identically for real fits.

    The original exception is preserved as the ``__cause__`` (via ``raise ...
    from``), so the true failure remains visible in the traceback rather than
    being masked behind a bare ``FitException``.

    Set the environment variable ``PYAUTO_RAISE_ANALYSIS_EXCEPTIONS=1`` to
    re-raise the original exception unchanged instead of wrapping it. This is a
    debugging aid for surfacing the real cause of a masked failure — for example
    under ``PYAUTO_TEST_MODE``, where a single likelihood evaluation is not
    absorbed by a sampler and the wrapped ``FitException`` would otherwise hide
    the underlying error. Off by default so production searches keep resampling.

    Parameters
    ----------
    exception
        The original exception raised while evaluating the numpy-path fit.
    """
    if os.environ.get("PYAUTO_RAISE_ANALYSIS_EXCEPTIONS", "0") == "1":
        raise exception
    raise af.exc.FitException from exception
