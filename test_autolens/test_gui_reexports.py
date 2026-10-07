"""Sanity check that autogalaxy GUI helpers are re-exported under al.*."""
import autolens as al


def test__mask_2d_regridded_from_is_exported_at_package_level():
    from autogalaxy.gui import display_util as du

    assert al.mask_2d_regridded_from is du.mask_2d_regridded_from
