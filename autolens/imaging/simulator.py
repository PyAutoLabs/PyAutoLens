"""
Imaging simulator for strong gravitational lens observations.

``SimulatorImaging`` extends ``aa.SimulatorImaging`` with a ``via_tracer_from`` method
that accepts a ``Tracer`` object and:

1. Uses the tracer's mass profiles to ray-trace the image-plane grid to the source plane.
2. Evaluates the light profiles of all galaxies on the (traced) grid.
3. Passes the result through the standard ``SimulatorImaging`` pipeline: PSF convolution,
   background sky addition, and Poisson noise realisation.

This is the primary entry point for generating synthetic lens datasets for testing,
validation, and mock-data studies.
"""
import numpy as np
from typing import List

import autoarray as aa
import autogalaxy as ag

from autolens.lens.tracer import Tracer

class SimulatorImaging(aa.SimulatorImaging):

    def via_tracer_from(self, tracer : Tracer, grid : aa.type.Grid2DLike, xp=None) -> aa.Imaging:
        """
        Simulate an `Imaging` dataset from an input `Tracer` object and a 2D grid of (y,x) coordinates.

        The mass profiles of each galaxy in the tracer are used to perform ray-tracing of the input 2D grid and
        their light profiles are used to generate the image of the galaxies which are simulated.

        The steps of the `SimulatorImaging` simulation process (e.g. PSF convolution, noise addition) are
        described in the `SimulatorImaging` `__init__` method docstring, found in the PyAutoArray project.

        If one of more galaxy light profiles are a `LightProfileSNR` object, the `intensity` of the light profile is
        automatically set such that the signal-to-noise ratio of the light profile is equal to its input
        `signal_to_noise_ratio` value.

        For example, if a `LightProfileSNR` object has a `signal_to_noise_ratio` of 5.0, the intensity of the light
        profile is set such that the peak surface brightness of the profile is 5.0 times the background noise level of
        the image.

        Parameters
        ----------
        tracer
            The tracer, which describes the ray-tracing and strong lens configuration used to simulate the imaging
            dataset as well as the light profiles of the galaxies used to simulate the image of the galaxies.
        grid
            The 2D grid of (y,x) coordinates which the mass profiles of the galaxies in the tracer are ray-traced using
            in order to generate the image of the galaxies via their light profiles.
        """

        if xp is None:
            xp = self._xp

        tracer.set_snr_of_snr_light_profiles(
            grid=grid,
            exposure_time=self.exposure_time,
            background_sky_level=self.background_sky_level,
        )

        image = tracer.padded_image_2d_from(
            grid=grid, psf_shape_2d=self.psf.kernel.shape_native, xp=xp
        )

        over_sample_size = grid.over_sample_size.resized_from(
            new_shape=image.shape_native, mask_pad_value=1
        )

        dataset = self.via_image_from(image=image, over_sample_size=over_sample_size, xp=xp)

        return dataset.trimmed_after_convolution_from(
            kernel_shape=self.psf.kernel.shape_native
        )

    def via_galaxies_from(self, galaxies : List[ag.Galaxy], grid : aa.type.Grid2DLike) -> aa.Imaging:
        """
        Simulate an `Imaging` dataset from an input list of `Galaxy` objects and a 2D grid of (y,x) coordinates.

        The galaxies are used to create a `Tracer`. The mass profiles of each galaxy in the tracer are used to
        perform ray-tracing of the input 2D grid and their light profiles are used to generate the image of the
        galaxies which are simulated.

        The steps of the `SimulatorImaging` simulation process (e.g. PSF convolution, noise addition) are
        described in the `SimulatorImaging` `__init__` method docstring.

        If one of more galaxy light profiles are a `LightProfileSNR` object, the `intensity` of the light profile is
        automatically set such that the signal-to-noise ratio of the light profile is equal to its input
        `signal_to_noise_ratio` value.

        For example, if a `LightProfileSNR` object has a `signal_to_noise_ratio` of 5.0, the intensity of the light
        profile is set such that the peak surface brightness of the profile is 5.0 times the background noise level of
        the image.

        Parameters
        ----------
        galaxies
            The galaxies used to create the tracer, which describes the ray-tracing and strong lens configuration
            used to simulate the imaging dataset.
        grid
            The image-plane 2D grid of (y,x) coordinates grid which the image of the strong lens is generated on.
        """

        tracer = Tracer(galaxies=galaxies)

        return self.via_tracer_from(tracer=tracer, grid=grid)

    def via_deflections_and_galaxies_from(self, deflections : aa.VectorYX2D, galaxies : List[ag.Galaxy]) -> aa.Imaging:
        """
        Simulate an `Imaging` dataset from an input deflection angle map and list of galaxies.

        The input deflection angle map ray-traces the image-plane coordinates from the image-plane to source-plane,
        via the lens equation.

        This traced grid is then used to evaluate the light of the list of galaxies, which therefore simulate the
        image of the strong lens.

        This function is used in situations where one has access to a deflection angle map which does not suit being
        ray-traced using a `Tracer` object (e.g. deflection angles from a cosmological simulation of a galaxy).

        The steps of the `SimulatorImaging` simulation process (e.g. PSF convolution, noise addition) are
        described in the `SimulatorImaging` `__init__` method docstring.

        Parameters
        ----------
        galaxies
            The galaxies used to create the tracer, which describes the ray-tracing and strong lens configuration
            used to simulate the imaging dataset.
        grid
            The image-plane 2D grid of (y,x) coordinates grid which the image of the strong lens is generated on.
        """
        grid = aa.Grid2D.uniform(
            shape_native=deflections.shape_native,
            pixel_scales=deflections.pixel_scales,
            over_sample_size=1
        )

        deflected_grid = aa.Grid2D(
            values=grid - deflections,
            mask=grid.mask,
            over_sample_size=1,
            over_sampled=grid - deflections,
            over_sampler=grid.over_sampler
        )

        image = sum(map(lambda g: g.image_2d_from(grid=deflected_grid), galaxies))

        return self.via_image_from(image=image)

    def via_source_image_from(self, tracer : Tracer, grid : aa.type.Grid2DLike, source_image : aa.Array2D) -> aa.Imaging:
        """
        Simulate an `Imaging` dataset from an input image of a source galaxy.

        This input image is on a uniform and regular 2D array, meaning it can simulate the source's irregular
        and asymmetric source galaxy morphological features.

        The typical use case is inputting the image of an irregular galaxy in the source-plane (whose values are
        on a uniform array) and using this function to compute the lensed image of this source galaxy.

        The tracer is used to perform ray-tracing and generate the image of the strong lens galaxies (e.g.
        the lens light, lensed source light, etc) which is simulated.

        The source galaxy light profiles are ignored in favour of the input source image, but the emission of
        other galaxies (e.g. the lems galaxy's light) are included.

        The steps of the `SimulatorImaging` simulation process (e.g. PSF convolution, noise addition) are
        described in the `SimulatorImaging` `__init__` method docstring.

        Parameters
        ----------
        tracer
            The tracer, which describes the ray-tracing and strong lens configuration used to simulate the imaging
            dataset.
        grid
            The image-plane 2D grid of (y,x) coordinates grid which the image of the strong lens is generated on.
        source_image
            The image of the source-plane and source galaxy which is interpolated to compute the lensed image.
        """

        image = tracer.image_2d_via_input_plane_image_from(
            grid=grid,
            plane_image=source_image
        )

        padded_image = image.padded_before_convolution_from(
            kernel_shape=self.psf.kernel.shape_native
        )
        dataset = self.via_image_from(image=padded_image)

        return dataset.trimmed_after_convolution_from(
            kernel_shape=self.psf.kernel.shape_native
        )