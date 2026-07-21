import pytest

import autofit as af
from autolens.analysis.exceptions import raise_fit_exception


class _Sentinel(Exception):
    """A distinctive error standing in for a real numpy-path fit failure."""


def test__wraps_as_fit_exception_and_preserves_cause():
    sentinel = _Sentinel("non-PD inversion")

    with pytest.raises(af.exc.FitException) as exc_info:
        raise_fit_exception(sentinel)

    # The search resamples on FitException, but the true failure must remain
    # visible in the traceback via __cause__ rather than being masked.
    assert exc_info.value.__cause__ is sentinel


def test__env_flag_reraises_original_unwrapped(monkeypatch):
    monkeypatch.setenv("PYAUTO_RAISE_ANALYSIS_EXCEPTIONS", "1")
    sentinel = _Sentinel("non-PD inversion")

    with pytest.raises(_Sentinel) as exc_info:
        raise_fit_exception(sentinel)

    assert exc_info.value is sentinel


def test__env_flag_zero_still_wraps(monkeypatch):
    # Only "1" enables raise-through; "0" must keep wrapping (guards against the
    # "0"-is-a-truthy-string trap).
    monkeypatch.setenv("PYAUTO_RAISE_ANALYSIS_EXCEPTIONS", "0")

    with pytest.raises(af.exc.FitException):
        raise_fit_exception(_Sentinel())
