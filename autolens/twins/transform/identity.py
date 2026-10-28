from autolens.twins.record import TransformationRecord


class Identity:
    """
    The do-nothing transformation: returns the twin's tracer and setup unchanged
    and a record named ``"Identity"`` with empty ``changed``, ``side_effects`` and
    ``preserved``. It exists so the transformation chain and its JSON are exercised
    before any physical transformation is available.
    """

    def apply(self, twin):
        record = TransformationRecord(
            name="Identity",
            requested={},
            procedure={
                "text": "No change: the tracer and observation setup are returned unchanged."
            },
        )
        return twin.tracer, twin.setup, record
