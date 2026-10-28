"""
How ``al.twins`` addresses the galaxies of a tracer.

A tracer loaded from a model-fit's ``files/tracer.json`` holds its galaxies in an
``af.ModelInstance`` keyed by the model's names (``af.Collection(lens=..., source=...)``
gives ``"lens"`` and ``"source"``); those keys are the galaxy names. A tracer built
from a plain list (or a list-backed ``ModelInstance``) names its galaxies by position,
``"galaxy_0"``, ``"galaxy_1"``, ..., as ``al.interop.coolest`` does. These are the
names that ``ReferenceModel`` provenance ``roles`` (galaxy name -> role) are keyed by,
so a transformation also accepts a role, resolved through the reference's ``roles``.

A component path is ``"galaxies.<name>.<attribute>"`` (e.g. ``"galaxies.lens.mass"``),
and a tracer-level field is ``"fields.<index>"``.
"""

from typing import Dict


def galaxy_name_list_from(tracer) -> list:
    """The galaxy names of ``tracer``, in the order of ``tracer.galaxies``."""
    return list(galaxy_dict_from(tracer))


def galaxy_dict_from(tracer) -> Dict[str, object]:
    """
    The galaxies of ``tracer`` by name: the keys of a dict-backed ``af.ModelInstance``
    (a tracer from ``tracer.json``), otherwise ``"galaxy_<index>"``.
    """
    galaxies = tracer.galaxies
    items = galaxies.dict.items() if hasattr(galaxies, "dict") else None
    if items is not None and all(isinstance(key, str) for key, _ in items):
        return dict(items)
    return {f"galaxy_{index}": galaxy for index, galaxy in enumerate(galaxies)}


def galaxy_name_from(twin, galaxy: str) -> str:
    """
    Resolve ``galaxy`` (a galaxy name, or a role from the reference's provenance
    ``roles``) to the name of one galaxy of ``twin.tracer``.

    Raises
    ------
    ValueError
        If ``galaxy`` is neither a galaxy name nor a role held by exactly one galaxy.
    """
    names = galaxy_name_list_from(twin.tracer)
    if galaxy in names:
        return galaxy

    roles = twin.reference.provenance.get("roles", {}) or {}
    matches = [name for name, role in roles.items() if role == galaxy]
    if len(matches) == 1 and matches[0] in names:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"The role {galaxy!r} is held by several galaxies ({matches}); name the "
            f"galaxy instead."
        )
    raise ValueError(
        f"No galaxy named {galaxy!r} in the twin's tracer, and no galaxy has that role. "
        f"Galaxy names: {names}; roles: {roles}. A tracer loaded from tracer.json names "
        f"its galaxies by the model's keys (e.g. 'lens'); a list of galaxies is named "
        f"'galaxy_0', 'galaxy_1', ..."
    )
