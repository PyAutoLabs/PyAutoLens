"""
The ``model-figures`` epic's three lens acceptance cases, over real classes.

Phases 1 and 2 rendered these from structural doubles inside ``test_autofit``.
This module renders the real thing (:mod:`test_autolens.model_figure.lens_models`)
and asserts, in this order:

1. the **structural invariants** -- what the figure must say about the model,
   which is what the epic and its independent review are actually about;
2. the **counts**, pinned as measured, with the epic's own numbers recorded in a
   ``# epic target:`` comment wherever the two differ (the epic's numbers were
   measured on the doubles, whose priors differ from the real classes');
3. the **rendered PNG** -- inside the width budget, and identical across two
   independent builds of the same model.
"""

import matplotlib

matplotlib.use("Agg")

import pytest

import autofit as af

from test_autolens.model_figure import lens_models


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def walk_nodes(node):
    yield node
    for child in node.children:
        yield from walk_nodes(child)


def rows_of(spec):
    rows = {}
    for node in walk_nodes(spec.root):
        for row in node.rows:
            rows[".".join(row.path)] = row
    return rows


def plates_of(spec):
    return [node for node in walk_nodes(spec.root) if node.plate is not None]


def cards_of(presentation):
    return list(presentation.walk())


def pill_named(card, name):
    for pill in card.pills:
        if pill.text == name or pill.text.startswith(f"{name} ·"):
            return pill
    return None


def card_of(presentation, key):
    for card in presentation.walk():
        if card.key == key:
            return card
    raise AssertionError(
        f"no card keyed {key!r}: {[card.key for card in presentation.walk()]}"
    )


def png_size(path):
    """``(width, height)`` in pixels, read from the PNG IHDR chunk."""
    import struct

    with open(path, "rb") as f:
        header = f.read(24)

    assert header[:8] == b"\x89PNG\r\n\x1a\n"

    return struct.unpack(">II", header[16:24])


# ---------------------------------------------------------------------------
# (a) the simple lens
# ---------------------------------------------------------------------------


def test__simple_lens__six_cards_in_model_info_order():
    model = lens_models.simple_lens()

    presentation = af.ModelPlotter(model).presentation()

    component_titles = [
        card.title for card in cards_of(presentation) if card.kind == "model"
    ]

    assert component_titles == [
        "lens · Galaxy",
        "bulge · Sersic",
        "mass · Isothermal",
        "shear · ExternalShear",
        "source · Galaxy",
        "bulge · SersicCore",
    ]


def test__simple_lens__counts():
    model = lens_models.simple_lens()

    spec = af.GraphSpec.from_model(model)

    # 8 components = the six drawn cards plus the root `Collection` and the
    # `galaxies` `Collection` frame, which are nodes of the tree too.
    assert spec.counts["components"] == 8
    assert spec.counts["unique_sampled_scalars"] == model.prior_count == 21
    assert spec.counts["shared_priors"] == 0
    assert spec.counts["solved"] == 0
    assert spec.counts["missing"] == 0
    # `SersicCore`'s `radius_break`, `gamma` and `alpha`, and the two redshifts.
    assert spec.counts["fixed_leaf_slots"] == 5


# ---------------------------------------------------------------------------
# (b) the MGE lens light + pixelized source
# ---------------------------------------------------------------------------


@pytest.fixture(name="mge_spec", scope="module")
def make_mge_spec():
    return af.GraphSpec.from_model(lens_models.mge_pixelized())


def test__mge__two_plates_of_thirty_split_on_ell_comps(mge_spec):
    plates = [node for node in plates_of(mge_spec) if node.plate.count == 30]

    assert len(plates) == 2

    # The two plates exist *because* each basis holds its own `ell_comps` pair:
    # the shared `centre` priors are carried by all 60 members and so cannot
    # discriminate, while the `ell_comps` priors differ between the bases.
    ell_comps_ids = {
        tuple(
            component.prior_id
            for row in plate.rows
            if row.name == "ell_comps"
            for component in row.components
        )
        for plate in plates
    }

    assert len(ell_comps_ids) == 2


def test__mge__every_member_intensity_is_solved(mge_spec):
    plates = [node for node in plates_of(mge_spec) if node.plate.count == 30]

    for plate in plates:
        intensity = next(row for row in plate.rows if row.name == "intensity")

        assert intensity.sampling == "solved"
        assert intensity.in_model_info is False


def test__mge__basis_itself_declares_no_solved_intensity(mge_spec):
    basis = next(node for node in walk_nodes(mge_spec.root) if node.cls_name == "Basis")

    assert [row.name for row in basis.rows if row.name == "intensity"] == []


def test__mge__shared_centre_is_badged_on_its_owner():
    model = lens_models.mge_pixelized()

    presentation = af.ModelPlotter(model).presentation()

    badges = {
        pill.badge
        for card in cards_of(presentation)
        for pill in card.pills
        if pill.text.startswith("centre") and pill.badge is not None
    }

    assert "shared across group" in badges


def test__mge__areas_factor_is_missing_on_the_mesh(mge_spec):
    row = rows_of(mge_spec)["galaxies.source.pixelization.mesh.areas_factor"]

    assert row.sampling == "missing"
    assert row.prior_cls_name == "ConfigException"


def test__mge__reconstruction_is_solved_on_the_pixelization_card():
    model = lens_models.mge_pixelized()

    presentation = af.ModelPlotter(model).presentation()

    card = card_of(presentation, "galaxies/source/pixelization")
    pill = pill_named(card, "reconstruction")

    assert pill is not None
    assert pill.state == "solved"
    assert pill.text == "reconstruction · solved"


def test__mge__counts(mge_spec):
    model = lens_models.mge_pixelized()

    assert mge_spec.counts["unique_sampled_scalars"] == model.prior_count == 14
    # epic target: 16 unique sampled scalars. The epic measured the phase-1
    # structural double, whose source was a `Sersic` (7 priors); this model's
    # source is pixelized, so its only sampled parameter is the `ConstantSplit`
    # coefficient (1 prior). 13 lens priors + 1 = 14.
    assert mge_spec.counts["components"] == 13
    assert mge_spec.counts["components_raw"] == 71
    assert mge_spec.counts["plates"] == 2
    # `centre_0`, `centre_1`, and each basis's own `ell_comps` pair.
    assert mge_spec.counts["shared_priors"] == 6
    assert mge_spec.counts["fixed_leaf_slots"] == 64
    # 60 member intensities plus the source reconstruction.
    assert mge_spec.counts["solved"] == 61
    # `Delaunay.areas_factor`, which has no prior configured.
    assert mge_spec.counts["missing"] == 1


# ---------------------------------------------------------------------------
# (e) the group-scale model
# ---------------------------------------------------------------------------


@pytest.fixture(name="group_spec", scope="module")
def make_group_spec():
    return af.GraphSpec.from_model(lens_models.group_scale())


def test__group__one_plate_of_eight(group_spec):
    plates = plates_of(group_spec)

    assert len(plates) == 1
    assert plates[0].plate.count == 8


def test__group__member_centres_are_fixed_and_vary_by_member():
    model = lens_models.group_scale()

    presentation = af.ModelPlotter(model).presentation()

    texts = [pill.text for pill in presentation.pills()]

    assert "centre · fixed, varies by member" in texts


def test__group__member_sigma_priors_are_independent():
    model = lens_models.group_scale()

    presentation = af.ModelPlotter(model).presentation()

    sigma = [pill for pill in presentation.pills() if pill.text.startswith("sigma")]

    assert sigma
    assert any(pill.badge == "independent" for pill in sigma)


def test__group__counts(group_spec):
    model = lens_models.group_scale()

    assert group_spec.counts["unique_sampled_scalars"] == model.prior_count == 69
    assert group_spec.counts["components"] == 12
    assert group_spec.counts["components_raw"] == 33
    assert group_spec.counts["plates"] == 1
    assert group_spec.counts["shared_priors"] == 0
    # Eight extra galaxies x (fixed centre pair + r_core, r_cut, the two
    # redshifts, H0, Om0) plus the main lens and source redshifts.
    assert group_spec.counts["fixed_leaf_slots"] == 77
    assert group_spec.counts["solved"] == 0
    assert group_spec.counts["missing"] == 0


# ---------------------------------------------------------------------------
# the renders
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "builder, filename",
    [
        (lens_models.simple_lens, "acceptance_simple_lens"),
        (lens_models.mge_pixelized, "acceptance_mge_pixelized"),
        (lens_models.group_scale, "acceptance_group_scale"),
    ],
)
def test__renders_inside_the_width_budget(builder, filename, tmp_path):
    af.ModelPlotter(builder()).figure(
        path=str(tmp_path), filename=filename, format="png", width=14.0
    )

    png = tmp_path / f"{filename}.png"

    assert png.exists()

    width, _ = png_size(png)

    assert width <= 1400


@pytest.mark.parametrize(
    "builder",
    [lens_models.simple_lens, lens_models.mge_pixelized, lens_models.group_scale],
)
def test__layout_is_identical_across_two_builds_of_the_same_model(builder):
    first = af.ModelPlotter(builder()).layout(width=14.0).to_dict()
    second = af.ModelPlotter(builder()).layout(width=14.0).to_dict()

    assert first == second
