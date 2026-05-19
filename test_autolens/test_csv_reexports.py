"""Sanity check that the new galaxy_model_csv helpers are re-exported under al.*."""
import autolens as al


def test__al_namespace_exposes_galaxy_model_csv_helpers():
    assert callable(al.galaxy_models_from_csv)
    assert callable(al.galaxy_models_to_csv)
    assert callable(al.galaxies_from_csv_tables)
    assert callable(al.galaxy_af_models_from_csv_tables)
    assert al.GalaxyModelRow.__name__ == "GalaxyModelRow"
    assert al.GalaxyModelTable.__name__ == "GalaxyModelTable"
