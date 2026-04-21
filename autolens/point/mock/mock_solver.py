from typing import Optional

import numpy as np


class MockPointSolver:
    def __init__(self, model_positions):
        self.model_positions = model_positions

    def solve(
        self,
        tracer,
        source_plane_coordinate,
        xp=np,
        plane_redshift: Optional[float] = None,
        remove_infinities: bool = True,
    ):
        return self.model_positions
