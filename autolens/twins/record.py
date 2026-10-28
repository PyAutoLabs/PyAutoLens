"""
``TransformationRecord``: what one transformation of a twin asked for, how it did
it, and what it changed, as plain JSON-able data.

``preserved`` is empty by default: preserving a parent property (e.g. the Einstein
radius) is a choice a transformation records explicitly, never an implicit
default.
"""

import copy
from typing import Dict, List, Optional

from autolens.twins._util import Frozen


class TransformationRecord(Frozen):
    def __init__(
        self,
        name: str,
        requested: Optional[Dict] = None,
        procedure: Optional[Dict] = None,
        changed: Optional[List] = None,
        side_effects: Optional[List] = None,
        preserved: Optional[List] = None,
    ):
        """
        The record a transformation appends to a twin's chain.

        Parameters
        ----------
        name
            The transformation's name, e.g. ``"Identity"``.
        requested
            The parameters the user asked for.
        procedure
            How the change was made: a ``"text"`` description plus any derived numbers.
        changed
            The galaxy / component paths that were changed.
        side_effects
            Properties that changed as a consequence (e.g. a lensing observable).
        preserved
            Parent properties explicitly held fixed. Empty unless a transformation
            chooses to preserve something.

        ``al.from_dict`` drops ``None`` values inside dictionaries (PyAutoFit's dict
        parser), so ``requested`` and ``procedure`` should not use ``None`` values.
        """
        self._freeze_set("name", str(name))
        self._freeze_set("requested", copy.deepcopy(dict(requested or {})))
        self._freeze_set("procedure", copy.deepcopy(dict(procedure or {})))
        self._freeze_set("changed", list(changed or []))
        self._freeze_set("side_effects", list(side_effects or []))
        self._freeze_set("preserved", list(preserved or []))

    def __repr__(self):
        return (
            f"TransformationRecord(name={self.name!r}, changed={self.changed}, "
            f"side_effects={self.side_effects}, preserved={self.preserved})"
        )

    def __eq__(self, other):
        if not isinstance(other, TransformationRecord):
            return NotImplemented
        return all(
            getattr(self, name) == getattr(other, name)
            for name in (
                "name",
                "requested",
                "procedure",
                "changed",
                "side_effects",
                "preserved",
            )
        )

    __hash__ = None
