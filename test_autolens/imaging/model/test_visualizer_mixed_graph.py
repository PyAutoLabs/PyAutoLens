import autoarray as aa
import autolens as al

from autolens.imaging.model.visualizer import VisualizerImaging


class _RaisingPaths:
    """visualize_combined must return before touching paths when no imaging analysis is present."""

    @property
    def image_path(self):
        raise AssertionError("plotting was attempted for a graph with no imaging analyses")


def test__visualize_combined__skips_non_imaging_analyses():
    """
    A mixed-dataset factor graph (e.g. imaging + weak) routes every factor's analysis into the
    lead factor's Visualizer.visualize_combined; non-imaging analyses must be filtered out
    rather than treated as FitImaging producers (which raised AttributeError before the fix).
    """
    grid = aa.Grid2DIrregular(values=[(1.0, 1.0), (-1.0, 1.0)])
    tracer = al.Tracer(
        galaxies=[
            al.Galaxy(
                redshift=0.5,
                mass=al.mp.IsothermalSph(einstein_radius=1.0),
            ),
            al.Galaxy(redshift=1.0),
        ]
    )
    dataset = al.SimulatorShearYX(noise_sigma=0.1, seed=1).via_tracer_from(
        tracer=tracer, grid=grid
    )
    analysis_weak = al.AnalysisWeak(dataset=dataset)

    VisualizerImaging.visualize_combined(
        analyses=[analysis_weak],
        paths=_RaisingPaths(),
        instance=[None],
        during_analysis=True,
    )
