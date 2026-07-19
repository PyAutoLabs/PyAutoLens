from autonerves.dictable import from_dict
import autolens as al
from autolens.point.solver import PointSolver


instance_dict = {
    "type": "instance",
    "class_path": "autofit.mapper.model.ModelInstance",
    "arguments": {
        "child_items": {
            "type": "dict",
            "arguments": {
                "source_galaxy": {
                    "type": "instance",
                    "class_path": "autogalaxy.galaxy.galaxy.Galaxy",
                    "arguments": {
                        "redshift": 1.0,
                        "label": "cls6",
                        "light": {
                            "type": "instance",
                            "class_path": "autogalaxy.profiles.light.standard.exponential.Exponential",
                            "arguments": {
                                "effective_radius": 0.1,
                                "ell_comps": [0.4731722153284571, -0.27306667016189645],
                                "centre": [-0.04829335038475, 0.02350935356045],
                                "intensity": 0.1,
                            },
                        },
                        "point_0": {
                            "type": "instance",
                            "class_path": "autogalaxy.profiles.point_source.Point",
                            "arguments": {
                                "centre": [-0.04829335038475, 0.02350935356045]
                            },
                        },
                    },
                },
                "lens_galaxy": {
                    "type": "instance",
                    "class_path": "autogalaxy.galaxy.galaxy.Galaxy",
                    "arguments": {
                        "redshift": 0.5,
                        "label": "cls6",
                        "mass": {
                            "type": "instance",
                            "class_path": "autogalaxy.profiles.mass.total.isothermal.Isothermal",
                            "arguments": {
                                "ell_comps": [
                                    0.05263157894736841,
                                    3.2227547345982974e-18,
                                ],
                                "einstein_radius": 1.6,
                                "centre": [0.0, 0.0],
                            },
                        },
                    },
                },
            },
        }
    },
}


def test_missing_multiple_image(grid):
    instance = from_dict(instance_dict)

    tracer = al.Tracer(galaxies=[instance.lens_galaxy, instance.source_galaxy])

    solver = PointSolver.for_grid(
        grid=grid,
        pixel_scale_precision=0.001,
    )

    triangle_positions = solver.solve(
        tracer=tracer, source_plane_coordinate=instance.source_galaxy.point_0.centre
    )
