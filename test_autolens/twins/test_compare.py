import numpy as np
import pytest

import autogalaxy as ag
import autolens as al
import autolens.plot as aplt

from test_autolens.twins.conftest import COARSE_APERTURE


def test__exports():
    for name in (
        "compare",
        "compare_to_data",
        "TwinComparison",
        "einstein_radius_aperture_from",
    ):
        assert hasattr(al.twins, name)
    assert hasattr(al.twins.transform, "ReplaceTotalMassWithStellarPlusDark")
    assert hasattr(aplt, "subplot_twin_comparison")


def test__einstein_radius_aperture(witness_baseline):
    aperture = al.twins.einstein_radius_aperture_from(
        tracer=witness_baseline.tracer,
        grid=witness_baseline.setup.grid,
        centre=(0.1, -0.1),
        sub_size_centre=8,
    )
    assert isinstance(aperture, ag.CircularAperture)
    assert aperture.centre == (0.1, -0.1)
    assert aperture.sub_size_centre == 8
    assert aperture.radius == pytest.approx(1.0, abs=0.05)


def test__compare__twin_with_itself_is_zero(witness_baseline):
    comparison = al.twins.compare(
        witness_baseline, witness_baseline, aperture_settings=COARSE_APERTURE
    )

    assert comparison.delta_chi_squared_image == 0.0
    assert np.all(comparison.convergence_difference.array == 0.0)
    assert np.all(comparison.deflections_difference.array == 0.0)
    assert np.all(comparison.mass_enclosed_difference == 0.0)
    assert np.all(comparison.normalized_residual_map.array == 0.0)
    assert comparison.einstein_radius_parent == comparison.einstein_radius_twin
    assert len(comparison.radii) > 0


def test__compare__transformed_twin(witness_baseline, witness_twin):
    comparison = al.twins.compare(
        witness_baseline, witness_twin, aperture_settings=COARSE_APERTURE
    )
    record = witness_twin.records[-1]

    assert comparison.delta_chi_squared_image > 0.0
    assert comparison.einstein_radius_parent == pytest.approx(
        record.procedure["einstein_radius_parent"], rel=1e-12
    )
    assert comparison.einstein_radius_twin == pytest.approx(
        record.procedure["einstein_radius_twin"], rel=1e-12
    )
    assert len(comparison.tangential_critical_curves_parent) > 0
    assert len(comparison.tangential_critical_curves_twin) > 0
    assert np.any(comparison.convergence_difference.array != 0.0)

    # The aperture mass inside theta_E,parent is the transformation's one preservation.
    index = int(np.argmin(np.abs(comparison.radii - comparison.einstein_radius_parent)))
    assert comparison.radii[index] == pytest.approx(comparison.einstein_radius_parent)
    assert comparison.mass_enclosed_twin[index] == pytest.approx(
        comparison.mass_enclosed_parent[index], rel=1.0e-3
    )


def test__compare_to_data(witness_baseline, witness_twin, masked_imaging_7x7):
    delta = al.twins.compare_to_data(witness_twin, masked_imaging_7x7)
    assert np.isfinite(delta)
    assert delta != 0.0

    assert al.twins.compare_to_data(witness_baseline, masked_imaging_7x7) == 0.0


def test__compare_to_data__pixelized_source_raises(
    witness_baseline, masked_imaging_7x7, tracer_x2_plane_inversion_7x7
):
    twin = al.twins.Twin(
        reference=witness_baseline.reference, tracer=tracer_x2_plane_inversion_7x7
    )
    with pytest.raises(NotImplementedError, match="pixelized"):
        al.twins.compare_to_data(twin, masked_imaging_7x7)


def test__subplot_twin_comparison__writes_png(
    witness_baseline, witness_twin, tmp_path, monkeypatch
):
    import autoarray.plot.utils as plot_utils

    monkeypatch.setattr(plot_utils, "_FAST_PLOTS", False)
    monkeypatch.delenv("PYAUTO_OUTPUT_MODE", raising=False)

    comparison = al.twins.compare(
        witness_baseline, witness_twin, aperture_settings=COARSE_APERTURE
    )
    aplt.subplot_twin_comparison(
        comparison, output_path=str(tmp_path), output_format="png"
    )

    assert (tmp_path / "twin_comparison.png").exists()
