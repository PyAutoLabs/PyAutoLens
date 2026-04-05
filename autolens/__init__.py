from autoconf import jax_wrapper
from autoconf.dictable import from_dict, from_json, output_to_json, to_dict
from autoarray import preprocess

from autoarray.dataset.imaging.dataset import Imaging
from autoarray.dataset.interferometer.dataset import (
    Interferometer,
)
from autoarray.dataset.grids import GridsInterface
from autoarray.dataset.dataset_model import DatasetModel
from autoarray.mask.mask_1d import Mask1D
from autoarray.mask.mask_2d import Mask2D
from autoarray.mask.derive.zoom_2d import Zoom2D
from autoarray.operators.over_sampling.over_sampler import OverSampler  # noqa
from autoarray.inversion.inversion.dataset_interface import DatasetInterface
from autoarray.inversion.mesh import image_mesh
from autoarray.inversion.mesh import mesh
from autoarray.inversion import regularization as reg
from autoarray.inversion.mesh.image_mesh.abstract import AbstractImageMesh
from autoarray.inversion.mesh.mesh.abstract import AbstractMesh
from autoarray.inversion.regularization.abstract import AbstractRegularization
from autoarray.inversion.pixelization import Pixelization
from autoarray.settings import Settings
from autoarray.inversion.inversion.factory import inversion_from as Inversion
from autoarray.inversion.mappers.abstract import Mapper
from autoarray.inversion.mesh.border_relocator import BorderRelocator
from autoarray.operators.convolver import Convolver
from autoarray.operators.transformer import TransformerDFT
from autoarray.operators.transformer import TransformerNUFFT
from autoarray.structures.arrays.uniform_1d import Array1D
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.arrays.rgb import Array2DRGB
from autoarray.structures.arrays.irregular import ArrayIrregular
from autoarray.structures.grids.uniform_1d import Grid1D
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.inversion.mesh.interpolator.rectangular import (
    InterpolatorRectangular,
)
from autoarray.inversion.mesh.interpolator.delaunay import InterpolatorDelaunay
from autoarray.structures.triangles.shape import Circle
from autoarray.structures.triangles.shape import Triangle
from autoarray.structures.triangles.shape import Square
from autoarray.structures.triangles.shape import Polygon
from autoarray.structures.vectors.uniform import VectorYX2D
from autoarray.structures.vectors.irregular import VectorYX2DIrregular
from autoarray.structures.visibilities import Visibilities
from autoarray.structures.visibilities import VisibilitiesNoiseMap
from autoarray.inversion.mesh.mesh_geometry.rectangular import (
    rectangular_edge_pixel_list_from,
)

from autogalaxy import cosmology as cosmo
from autogalaxy.analysis.adapt_images.adapt_images import AdaptImages
from autogalaxy.analysis.adapt_images.adapt_images import (
    galaxy_name_image_dict_via_result_from,
)
from autogalaxy.gui.clicker import Clicker
from autogalaxy.gui.scribbler import Scribbler
from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.galaxy.galaxies import Galaxies
from autogalaxy.galaxy.redshift import Redshift

from autogalaxy.quantity.dataset_quantity import DatasetQuantity
from autogalaxy.profiles.geometry_profiles import EllProfile
from autogalaxy.profiles import (
    point_sources as ps,
    mass as mp,
    light_and_mass_profiles as lmp,
    light_linear_and_mass_profiles as lmp_linear,
    scaling_relations as sr,
)
from autogalaxy.profiles.light.abstract import LightProfile
from autogalaxy.profiles.light import standard as lp
from autogalaxy.profiles.light import snr as lp_snr
from autogalaxy.profiles.light import linear as lp_linear
from autogalaxy.profiles.light import operated as lp_operated
from autogalaxy.profiles import basis as lp_basis
from autogalaxy.profiles.light import (
    linear_operated as lp_linear_operated,
)
from autogalaxy.profiles.light.linear import LightProfileLinearObjFuncList
from autogalaxy.operate.image import OperateImage
from autogalaxy.operate.lens_calc import LensCalc
from autogalaxy.quantity.dataset_quantity import DatasetQuantity
from autogalaxy import convert

from . import plot
from . import aggregator as agg
from .analysis import model_util
from .lens import subhalo
from .lens.tracer import Tracer
from .lens.sensitivity import SubhaloSensitivityResult
from .lens.to_inversion import TracerToInversion
from .analysis.positions import PositionsLH
from .imaging.simulator import SimulatorImaging
from .imaging.fit_imaging import FitImaging
from .imaging.model.analysis import AnalysisImaging
from .imaging.model.visualizer import VisualizerImaging
from .interferometer.simulator import SimulatorInterferometer
from .interferometer.fit_interferometer import FitInterferometer
from .interferometer.model.analysis import AnalysisInterferometer
from .interferometer.model.visualizer import VisualizerInterferometer
from .point.dataset import PointDataset
from .point.fit.dataset import FitPointDataset
from .point.fit.fluxes import FitFluxes
from .point.fit.times_delays import FitTimeDelays
from .point.fit.positions.image.abstract import AbstractFitPositionsImagePair
from .point.fit.positions.image.pair import FitPositionsImagePair
from .point.fit.positions.image.pair_all import FitPositionsImagePairAll
from .point.fit.positions.image.pair_repeat import FitPositionsImagePairRepeat
from .point.fit.positions.source.separations import FitPositionsSource
from .point.max_separation import SourceMaxSeparation
from .point.model.analysis import AnalysisPoint
from .point.solver import PointSolver
from .point.solver.shape_solver import ShapeSolver
from .quantity.fit_quantity import FitQuantity
from .quantity.model.analysis import AnalysisQuantity

from . import exc
from . import mock as m
from . import util

from autoconf import conf
from autoconf.fitsable import ndarray_via_hdu_from
from autoconf.fitsable import ndarray_via_fits_from
from autoconf.fitsable import header_obj_from
from autoconf.fitsable import output_to_fits
from autoconf.fitsable import hdu_list_for_output_from

conf.instance.register(__file__)

__version__ = "2026.4.5.3"
