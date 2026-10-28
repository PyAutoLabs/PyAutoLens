"""
Transformations of a twin.

A transformation is any object with ``apply(twin) -> (tracer, setup, record)``:
it returns the new ground-truth tracer, the (possibly unchanged) observation setup
and a ``TransformationRecord`` describing the change. ``Twin.apply`` wraps the
result into a new ``Twin`` with the record appended; a transformation must not
mutate the twin, its tracer or its setup.
"""

from typing import Protocol, Tuple, runtime_checkable

from autolens.twins.transform.identity import Identity


@runtime_checkable
class Transformation(Protocol):
    def apply(self, twin) -> Tuple[object, object, object]:
        """Return ``(tracer, setup, record)`` for ``twin`` without mutating it."""
        ...
