import numpy as np

from autofit.mock import *  # noqa
from autoarray.mock import *  # noqa
from autogalaxy.mock import *  # noqa
from autolens import Tracer

from autolens.imaging.mock.mock_fit_imaging import MockFitImaging  # noqa
from autolens.lens.mock.mock_tracer import MockTracer  # noqa
from autolens.lens.mock.mock_tracer import MockTracerPoint  # noqa
from autolens.point.mock.mock_solver import MockPointSolver  # noqa


class NullTracer(Tracer):
    def __init__(self):
        super().__init__([])

    def deflections_yx_2d_from(self, grid, xp=np):
        return xp.zeros_like(grid.array)

    def deflections_between_planes_from(self, grid, xp=np, plane_i=0, plane_j=-1):
        return xp.zeros_like(grid.array)
