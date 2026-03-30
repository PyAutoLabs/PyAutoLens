from enum import Enum


class FITSGalaxyImages(Enum):
    """
    The HDUs that can be extracted from the fit.fits file.
    """

    lens_light_image = "GALAXY_0"
    lensed_source_image = "GALAXY_1"


class FITSModelGalaxyImages(Enum):
    """
    The HDUs that can be extracted from the fit.fits file.
    """

    lens_light_image = "GALAXY_0"
    lensed_source_image = "GALAXY_1"


class FITSTracer(Enum):
    """
    The HDUs that can be extracted from the fit.fits file.
    """

    convergence = "CONVERGENCE"
    potential = "POTENTIAL"
    deflections_y = "DEFLECTIONS_Y"
    deflections_x = "DEFLECTIONS_X"


class FITSFit(Enum):
    """
    The HDUs that can be extracted from the fit.fits file.
    """

    model_data = "MODEL_DATA"
    residual_map = "RESIDUAL_MAP"
    normalized_residual_map = "NORMALIZED_RESIDUAL_MAP"
    chi_squared_map = "CHI_SQUARED_MAP"


class Dataset(Enum):
    """
    The subplots that can be extracted from the fit image.

    The values correspond to the position of the subplot in the 4x3 grid.
    """

    data = (0, 0)
    data_log_10 = (1, 0)
    noise_map = (2, 0)
    psf = (0, 1)
    psf_log_10 = (1, 1)
    signal_to_noise_map = (2, 1)
    over_sample_size_lp = (0, 2)
    over_sample_size_pixelization = (1, 2)


class Tracer(Enum):
    """
    The subplots that can be extracted from the tracer image.

    The values correspond to the position of the subplot in the 3x3 grid.
    """

    image = (0, 0)
    source_image = (1, 0)
    source_plane_image = (2, 0)
    lens_light_image = (0, 1)
    convergence = (1, 1)
    potential = (2, 1)
    magnification = (0, 2)
    deflections_y = (1, 2)
    deflections_x = (2, 2)


class FitX1Plane(Enum):
    """
    The subplots that can be extracted from the fit image.

    The values correspond to the position of the subplot in the 4x3 grid.
    """

    data = (0, 0)
    signal_to_noise_map = (1, 0)
    model_data = (2, 0)
    lens_light_subtracted_image = (0, 1)
    lens_light_subtracted_image_zero = (1, 1)
    normalized_residual_map = (2, 1)


class Fit(Enum):
    """
    The subplots that can be extracted from the fit image.

    The values correspond to the position of the subplot in the 4x3 grid.
    """

    data = (0, 0)
    data_source_scale = (1, 0)
    signal_to_noise_map = (2, 0)
    model_data = (3, 0)
    lens_light_model = (0, 1)
    lens_light_subtracted_image = (1, 1)
    source_model_image = (2, 1)
    source_plane_image_zoom = (3, 1)
    normalized_residual_map = (0, 2)
    normalized_residual_map_one_sigma = (1, 2)
    chi_squared_map = (2, 2)
    source_plane_image = (3, 2)


class FitLog10(Enum):
    """
    The subplots that can be extracted from the fit image.

    The values correspond to the position of the subplot in the 4x3 grid.
    """

    data = (0, 0)
    data_source_scale = (1, 0)
    signal_to_noise_map = (2, 0)
    model_data = (3, 0)
    lens_light_model = (0, 1)
    lens_light_subtracted_image = (1, 1)
    source_model_image = (2, 1)
    source_plane_image_zoom = (3, 1)
    normalized_residual_map = (0, 2)
    normalized_residual_map_one_sigma = (1, 2)
    chi_squared_map = (2, 2)
    source_plane_image = (3, 2)
