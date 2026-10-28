"""
``Twin``: a reference model's tracer and setup after a chain of recorded
transformations. A twin is a value: ``apply`` and ``with_setup`` return new twins
and nothing mutates the reference, the tracer or the setup.
"""

import copy
import json
from typing import Iterable, Optional

import numpy as np

from autonerves.dictable import to_dict

from autolens.twins._util import Frozen
from autolens.twins.record import TransformationRecord
from autolens.twins.reference import ReferenceModel
from autolens.twins.setup import ObservationSetup

NOISE_ROUTES = ("observed", "poisson")

OBSERVED_ROUTE = "tracer.blurred_image_2d_from"
POISSON_ROUTE = "al.SimulatorImaging.via_tracer_from"


def _check_seed(seed) -> int:
    if (
        isinstance(seed, bool)
        or not isinstance(seed, (int, np.integer))
        or int(seed) < 0
    ):
        raise ValueError(
            f"seed must be a non-negative integer (got {seed!r}); a twin is always "
            f"simulated with an explicit seed so it can be reproduced."
        )
    return int(seed)


class Twin(Frozen):
    def __init__(
        self,
        reference: ReferenceModel,
        tracer=None,
        setup: Optional[ObservationSetup] = None,
        records: Iterable[TransformationRecord] = (),
    ):
        """
        A ground-truth tracer and observation setup derived from ``reference``.

        Parameters
        ----------
        reference
            The immutable parent.
        tracer
            The ground truth; defaults to a copy of the reference tracer.
        setup
            Defaults to the reference setup.
        records
            The transformation chain, oldest first.
        """
        self._freeze_set("reference", reference)
        self._freeze_set(
            "tracer", copy.deepcopy(reference.tracer) if tracer is None else tracer
        )
        self._freeze_set("setup", reference.setup if setup is None else setup)
        self._freeze_set("records", tuple(records))

    def __repr__(self):
        return (
            f"Twin(lens_id={self.reference.lens_id!r}, "
            f"records={[record.name for record in self.records]})"
        )

    @classmethod
    def baseline(cls, reference: ReferenceModel) -> "Twin":
        """The untransformed twin: a copy of the reference tracer on its setup."""
        return cls(reference=reference)

    def apply(self, transformation) -> "Twin":
        """A new twin with ``transformation`` applied and its record appended."""
        tracer, setup, record = transformation.apply(self)
        if not isinstance(record, TransformationRecord):
            raise TypeError(
                f"{type(transformation).__name__}.apply must return a "
                f"TransformationRecord as its third value, got {type(record).__name__}."
            )
        return Twin(
            reference=self.reference,
            tracer=tracer,
            setup=setup,
            records=self.records + (record,),
        )

    def with_setup(self, setup: ObservationSetup) -> "Twin":
        """A new twin observed with ``setup`` (e.g. ``setup.with_(exposure_time=...)``)."""
        return Twin(
            reference=self.reference,
            tracer=self.tracer,
            setup=setup,
            records=self.records,
        )

    def setup_overrides(self) -> dict:
        """
        The setup fields that differ from the reference setup, as dictable
        dictionaries (empty for a twin observed exactly like its reference).
        """
        reference_fields = self.reference.setup.fields
        overrides = {}
        for name, value in self.setup.fields.items():
            value_dict = to_dict(value)
            if json.dumps(value_dict, sort_keys=True) != json.dumps(
                to_dict(reference_fields[name]), sort_keys=True
            ):
                overrides[name] = value_dict
        return overrides

    def noise_free_image_2d(self):
        """
        The PSF-convolved, noise-free image of the twin's tracer on the setup mask.

        Computed with ``tracer.blurred_image_2d_from`` on the light-profile and
        blurring grids of ``setup.dataset_from(...)``: the same forward model a
        ``FitImaging`` of that dataset uses, without any noise draw.
        """
        dataset = self.setup.dataset_from(data=self.setup.noise_map)
        return self.tracer.blurred_image_2d_from(
            grid=dataset.grids.lp,
            blurring_grid=dataset.grids.blurring,
            psf=dataset.psf,
        )

    def simulate(self, seed: int, noise: str = "observed"):
        """
        Simulate an observation of the twin.

        ``noise="observed"`` (default, ruling R2): ``noise_free_image_2d()`` plus
        Gaussian noise drawn from the setup's observed noise map with
        ``numpy.random.default_rng(seed)`` on unmasked pixels only. The global NumPy
        random state is neither read nor changed. The noise map is the setup's.

        ``noise="poisson"``: ``al.SimulatorImaging`` with the setup's
        ``exposure_time`` and ``background_sky_level`` and ``noise_seed=seed``
        (Poisson noise, noise map from the counts). Both fields must be set. This
        route seeds NumPy's global random state, as the simulator does.
        """
        from autolens.twins.simulate import SimulatedTwin

        seed = _check_seed(seed)

        if noise == "observed":
            image = self.noise_free_image_2d()
            noise_map = self.setup.noise_map
            rng = np.random.default_rng(seed)
            data = np.asarray(image.array) + rng.normal(
                loc=0.0, scale=np.asarray(noise_map.array)
            )
            dataset = self.setup.dataset_from(data=data, noise_map=noise_map)
            simulator = {
                "route": OBSERVED_ROUTE,
                "noise": "gaussian: numpy.random.default_rng(seed).normal(0, noise_map)",
                "exposure_time": None,
                "background_sky_level": None,
            }
        elif noise == "poisson":
            dataset, simulator = self._simulate_poisson(seed=seed)
        else:
            raise ValueError(f"noise must be one of {NOISE_ROUTES}, got {noise!r}.")

        return SimulatedTwin(
            dataset=dataset,
            truth=self.tracer,
            seed=seed,
            noise=noise,
            simulator=simulator,
            twin=self,
        )

    def _simulate_poisson(self, seed: int):
        from autolens.imaging.simulator import SimulatorImaging
        from autolens.twins.setup import _copy

        for field in ("exposure_time", "background_sky_level"):
            if getattr(self.setup, field) is None:
                raise ValueError(
                    f'noise="poisson" requires setup.{field} to be set; use '
                    f"twin.with_setup(twin.setup.with_({field}=...)) or the default "
                    f'noise="observed".'
                )

        simulator = SimulatorImaging(
            exposure_time=self.setup.exposure_time,
            background_sky_level=self.setup.background_sky_level,
            psf=_copy(self.setup.psf),
            noise_seed=seed,
        )
        # A copy, because `via_tracer_from` sets the intensity of any
        # `LightProfileSNR` in place; the twin's ground truth must not change.
        simulated = simulator.via_tracer_from(
            tracer=copy.deepcopy(self.tracer), grid=self.setup.grid
        )
        dataset = self.setup.dataset_from(
            data=simulated.data.native.array,
            noise_map=simulated.noise_map.native.array,
        )
        settings = {
            "route": POISSON_ROUTE,
            "noise": "poisson: SimulatorImaging(noise_seed=seed)",
            "exposure_time": self.setup.exposure_time,
            "background_sky_level": self.setup.background_sky_level,
        }
        return dataset, settings
