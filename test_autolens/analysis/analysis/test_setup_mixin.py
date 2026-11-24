"""
The lens analyses share PyAutoGalaxy's one setup-resolution path (`AnalysisSetupMixin`).
"""
import autolens as al

from autogalaxy.analysis.analysis.setup import AnalysisSetupMixin

from autolens.analysis.analysis.setup import AnalysisSetupNoOperator


def test__lens_analyses__setup_mixin_in_mro_once():
    assert issubclass(AnalysisSetupNoOperator, AnalysisSetupMixin)

    for cls in (
        al.AnalysisPoint,
        al.AnalysisWeak,
        al.AnalysisImaging,
        al.AnalysisInterferometer,
    ):
        assert cls.__mro__.count(AnalysisSetupMixin) == 1


def test__setup_no_operator__key_and_note_from_setup_key():
    class Analysis(AnalysisSetupNoOperator):
        def setup_key_from(self, model):
            return ("point", "isothermal")

    assert Analysis().setup_key_and_note_from(model=None) == (
        ("point", "isothermal"),
        None,
    )
