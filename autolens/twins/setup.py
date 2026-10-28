"""
``ObservationSetup``: everything about *how* a lens is observed, separate from
*what* is observed (the tracer).

A setup holds the mask, the PSF (an ``al.Convolver``), the observed noise map,
the pixel scales and origin, the light-profile and pixelization over-sampling,
an optional cosmology and, for the Poisson noise route only, an exposure time and
a background sky level. It is a value: ``with_(...)`` returns a modified copy and
nothing here mutates a setup in place.

Serialisation: ``aa.Mask2D``, ``aa.Array2D``, ``aa.Convolver``, the cosmology
objects and plain ints/floats/tuples round-trip through ``autonerves.dictable``
natively, *except* that an array loaded from FITS carries an ``aa.Header`` which
does not. The constructor therefore stores header-free copies of the mask, PSF,
noise map and over-sample arrays (``_copy``), and ``ObservationSetup`` then uses
the default constructor-argument contract, so ``al.to_dict`` / ``al.from_dict``
work unchanged.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

import autoarray as aa
from autonerves.dictable import from_dict, to_dict

from autolens.twins import _util

_FIELDS = (
    "mask",
    "psf",
    "noise_map",
    "pixel_scales",
    "origin",
    "over_sample_size_lp",
    "over_sample_size_pixelization",
    "cosmology",
    "exposure_time",
    "background_sky_level",
)


def _clean_mask(mask: aa.Mask2D) -> aa.Mask2D:
    return aa.Mask2D(
        mask=np.array(mask, dtype=bool),
        pixel_scales=tuple(mask.pixel_scales),
        origin=tuple(mask.origin),
    )


def _clean_array(array: aa.Array2D) -> aa.Array2D:
    return aa.Array2D(
        values=np.array(array.native.array),
        mask=_clean_mask(array.mask),
    )


def _copy(obj):
    """
    A fresh, header-free copy of a mask, array or ``Convolver``.

    Arrays loaded from FITS carry an ``aa.Header`` (astropy cards) that does not
    round-trip through ``autonerves.dictable``; ``Mask2D``, ``Array2D`` and
    ``Convolver`` built from plain values do. Every array a setup holds goes
    through this, so a setup always serialises and never shares (or has
    normalised in place by ``aa.Imaging``) an array with a dataset.
    """
    if obj is None or isinstance(obj, (bool, int, float, np.integer, np.floating)):
        return obj
    if isinstance(obj, aa.Convolver):
        return aa.Convolver(
            kernel=_clean_array(obj.kernel),
            use_fft=obj._use_fft,
            convolve_over_sample_size=obj.convolve_over_sample_size,
        )
    if isinstance(obj, aa.Mask2D):
        return _clean_mask(obj)
    if isinstance(obj, aa.Array2D):
        return _clean_array(obj)
    return from_dict(to_dict(obj))


class ObservationSetup:
    def __init__(
        self,
        mask: aa.Mask2D,
        psf: Optional[aa.Convolver],
        noise_map: aa.Array2D,
        pixel_scales: Optional[Tuple[float, float]] = None,
        origin: Optional[Tuple[float, float]] = None,
        over_sample_size_lp: Union[int, aa.Array2D] = 4,
        over_sample_size_pixelization: Union[int, aa.Array2D] = 4,
        cosmology=None,
        exposure_time: Optional[float] = None,
        background_sky_level: Optional[float] = None,
    ):
        """
        How a lens is observed: mask, PSF, observed noise map and sampling.

        Parameters
        ----------
        mask
            The mask of the observation; twins are simulated and stored on it.
        psf
            The PSF as an ``al.Convolver`` (already normalised, as held by the dataset).
        noise_map
            The observed noise map (standard deviation per pixel) on ``mask``. It is
            the default noise of every twin (ruling R2).
        pixel_scales, origin
            Default to the mask's values.
        over_sample_size_lp, over_sample_size_pixelization
            Over-sampling of the light-profile and pixelization grids (an int or an
            ``aa.Array2D`` on ``mask``), as used by the fit.
        cosmology
            The cosmology the twin is interpreted in; ``None`` means the tracer's own.
        exposure_time, background_sky_level
            Only used by the ``"poisson"`` noise route, which requires both.
        """
        mask = _copy(mask)
        self.mask = mask
        self.psf = _copy(psf)
        self.noise_map = _copy(noise_map)
        self.pixel_scales = tuple(
            mask.pixel_scales if pixel_scales is None else pixel_scales
        )
        self.origin = tuple(mask.origin if origin is None else origin)
        self.over_sample_size_lp = _copy(over_sample_size_lp)
        self.over_sample_size_pixelization = _copy(over_sample_size_pixelization)
        self.cosmology = cosmology
        self.exposure_time = exposure_time
        self.background_sky_level = background_sky_level

    def __repr__(self):
        return (
            f"ObservationSetup(shape_native={self.mask.shape_native}, "
            f"pixels_in_mask={self.mask.pixels_in_mask}, "
            f"pixel_scales={self.pixel_scales}, "
            f"exposure_time={self.exposure_time}, "
            f"background_sky_level={self.background_sky_level})"
        )

    @property
    def fields(self) -> dict:
        """The constructor arguments of this setup, by name."""
        return {name: getattr(self, name) for name in _FIELDS}

    @classmethod
    def from_dataset(
        cls,
        dataset: aa.Imaging,
        cosmology=None,
        exposure_time: Optional[float] = None,
        background_sky_level: Optional[float] = None,
    ) -> "ObservationSetup":
        """
        The setup of a masked ``al.Imaging`` dataset: its mask, PSF, noise map and
        over-sampling. The PSF and noise map are copied so later changes to the
        dataset do not leak into the setup.
        """
        return cls(
            mask=dataset.mask,
            psf=dataset.psf,
            noise_map=dataset.noise_map,
            over_sample_size_lp=dataset.grids.lp.over_sample_size,
            over_sample_size_pixelization=dataset.grids.pixelization.over_sample_size,
            cosmology=cosmology,
            exposure_time=exposure_time,
            background_sky_level=background_sky_level,
        )

    @staticmethod
    def dataset_from_fit_directory(path: Union[str, Path]) -> aa.Imaging:
        """
        Rebuild the masked, over-sampled ``Imaging`` a model-fit used from its
        ``image/dataset.fits`` (HDUs: 0 mask, 1 data, 2 noise map, 3 PSF,
        4 light-profile over-sampling, 5 pixelization over-sampling), as written by
        ``AnalysisImaging.save_attributes`` (``autolens/imaging/model/analysis.py``).

        This replicates PyAutoGalaxy's aggregator reload ``_imaging_from``
        (``autogalaxy/aggregator/imaging/imaging.py`` lines 29-115) and
        ``agg_util.mask_header_from`` line for line, for a single (non-summed)
        analysis, without needing an ``af.Fit``. If that recipe changes, change this.
        """
        from astropy.io import fits

        dataset_path = Path(path) / "image" / "dataset.fits"

        with fits.open(dataset_path) as hdu_list:
            mask = _util.mask_from_hdu(hdu_list[0])
            header = aa.Header(header_sci_obj=hdu_list[0].header)

            def values_from(hdu: int):
                return aa.Array2D.no_mask(
                    values=_util.ndarray_via_hdu_from(hdu_list[hdu]),
                    pixel_scales=mask.pixel_scales,
                    header=header,
                    origin=mask.origin,
                )

            data = values_from(hdu=1)
            noise_map = values_from(hdu=2)

            try:
                psf = aa.Convolver(kernel=values_from(hdu=3))
            except (TypeError, IndexError):
                psf = None

            dataset = aa.Imaging(
                data=data,
                noise_map=noise_map,
                psf=psf,
                check_noise_map=False,
            )

            dataset = dataset.apply_mask(mask=mask)

            try:
                over_sample_size_lp = values_from(hdu=4).native
                over_sample_size_lp = over_sample_size_lp.apply_mask(mask=mask)
            except (TypeError, IndexError):
                over_sample_size_lp = 1

            try:
                over_sample_size_pixelization = values_from(hdu=5).native
                over_sample_size_pixelization = (
                    over_sample_size_pixelization.apply_mask(mask=mask)
                )
            except (TypeError, IndexError):
                over_sample_size_pixelization = 1

        return dataset.apply_over_sampling(
            over_sample_size_lp=over_sample_size_lp,
            over_sample_size_pixelization=over_sample_size_pixelization,
        )

    @classmethod
    def from_fit_directory(
        cls,
        path: Union[str, Path],
        cosmology=None,
        exposure_time: Optional[float] = None,
        background_sky_level: Optional[float] = None,
    ) -> "ObservationSetup":
        """
        The setup of a model-fit output directory (the folder holding ``image/`` and
        ``files/``), read from ``image/dataset.fits`` exactly as the aggregator does
        (see ``dataset_from_fit_directory``).
        """
        return cls.from_dataset(
            dataset=cls.dataset_from_fit_directory(path),
            cosmology=cosmology,
            exposure_time=exposure_time,
            background_sky_level=background_sky_level,
        )

    def with_(self, **overrides) -> "ObservationSetup":
        """
        A copy of this setup with the given fields replaced, e.g.
        ``setup.with_(exposure_time=2000.0, background_sky_level=1.0)``.

        Fields that depend on the mask (noise map, array over-sampling) are not
        re-derived: replacing the mask requires replacing them too.
        """
        unknown = set(overrides) - set(_FIELDS)
        if unknown:
            raise TypeError(
                f"ObservationSetup.with_ got unknown field(s) {sorted(unknown)}; "
                f"valid fields are {list(_FIELDS)}."
            )
        fields = self.fields
        if "mask" in overrides:
            fields.pop("pixel_scales")
            fields.pop("origin")
        fields.update(overrides)
        return ObservationSetup(**fields)

    @property
    def grid(self) -> aa.Grid2D:
        """The masked lens-plane grid, over-sampled with ``over_sample_size_lp``."""
        return aa.Grid2D.from_mask(
            mask=self.mask, over_sample_size=self.over_sample_size_lp
        )

    def _array_2d_from(self, values) -> aa.Array2D:
        """Unmasked native values on the setup's pixel grid (masked entries zero)."""
        if isinstance(values, aa.Array2D):
            native = _util.native_values_from(values)
        else:
            values = np.asarray(values, dtype="float64")
            if values.shape == self.mask.shape_native:
                native = values
            else:
                native = _util.native_values_from(
                    aa.Array2D(values=values, mask=self.mask)
                )
        if native.shape != self.mask.shape_native:
            raise ValueError(
                f"Array of native shape {native.shape} does not match the setup mask "
                f"shape {self.mask.shape_native}."
            )
        native = np.where(np.asarray(self.mask), 0.0, native)
        return aa.Array2D.no_mask(
            values=native,
            pixel_scales=self.mask.pixel_scales,
            origin=self.mask.origin,
        )

    def dataset_from(self, data, noise_map=None) -> aa.Imaging:
        """
        An ``al.Imaging`` of ``data`` observed with this setup: masked with the setup
        mask and over-sampled with its over-sample sizes, following the aggregator's
        reload order (build unmasked, ``apply_mask`` once, ``apply_over_sampling``).

        Parameters
        ----------
        data
            An ``aa.Array2D`` (masked or native) or an ndarray (native or slim).
        noise_map
            Defaults to the setup's observed noise map.
        """
        noise_map = self.noise_map if noise_map is None else noise_map
        dataset = aa.Imaging(
            data=self._array_2d_from(data),
            noise_map=self._array_2d_from(noise_map),
            psf=_copy(self.psf),
            check_noise_map=False,
            use_normalized_psf=False,
        )
        dataset = dataset.apply_mask(mask=self.mask)
        return dataset.apply_over_sampling(
            over_sample_size_lp=_copy(self.over_sample_size_lp),
            over_sample_size_pixelization=_copy(self.over_sample_size_pixelization),
        )
