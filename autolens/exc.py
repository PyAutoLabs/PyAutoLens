import autofit as af
from autofit.exc import *
from autoarray.exc import *
from autogalaxy.exc import *


class RayTracingException(af.exc.FitException):
    """
    Raises exceptions associated with the `lens/tracer.py` module and `Tracer` class.

    This exception inherits from a `FitException`. This means that if this exception is raised during a model-fit in
    the analysis class's `log_likelihood_function` that model is resampled and does not terminate the code.
    """

    pass


class PositionsException(af.exc.FitException):
    """
    Raises exceptions associated with the positions data in the  `point` module.

    For example if the multiple image positions do not meet certain format requirements.

    This exception inehrits from a `FitException`. This means that if this exception is raised during a model-fit in
    the analysis class's `log_likelihood_function` that model is resampled and does not terminate the code.
    """

    pass


class PixelizationException(af.exc.FitException):
    """
    Raises exceptions associated with the `inversion/pixelization` modules and `Pixelization` classes.

    For example if a `RectangularRTUAdaptDensity` mesh has dimensions below 3x3.

    This exception overwrites `autoarray.exc.PixelizationException` in order to add a `FitException`. This means that
    if this exception is raised during a model-fit in the analysis class's `log_likelihood_function` that model
    is resampled and does not terminate the code.
    """

    pass


class PointExtractionException(Exception):
    """
    Raises exceptions associated with the extraction of quantities in the  `point` module, where the name of a
    `PointSource` profile often relates to a model-component.

    For example if one tries to extract a profile `point_1` but there is no corresponding `PointSource` profile
    named `point_1`.
    """

    pass


class PointProfileMismatchException(Exception):
    """
    Raised when a point-source profile is paired with a fit class that cannot honestly use it — e.g. a
    centre-bearing `ps.Point` / `ps.PointFlux` with a `*Solved` fit (whose analytic solve would leave the centre
    or flux priors sampled but silently ignored), or a profile without the attribute a fit class requires.

    Deliberately NOT a subclass of `PointExtractionException`: `FitPointDataset` swallows that exception to skip
    absent dataset components (its long-standing name-pairing semantics), and profile/fit mismatches must never
    be silently skipped — they invalidate the composed model.
    """


class SetupException(AnalysisException):
    """
    Raised when the compute `setup` of an analysis cannot be honoured, for example when the likelihood of an
    analysis given `setup=` is evaluated before the setup was resolved (by `modify_before_fit` when the fit
    starts, or by `prepare_for_model(model)` before it).

    Deliberately NOT a subclass of `af.exc.FitException`: it is a configuration error, so it must stop the
    model-fit with a clear message rather than be swallowed into a resample on every likelihood call.
    """

    pass
