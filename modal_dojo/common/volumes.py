"""The Volumes Dojo creates: v2, whose hard links Miles' GPU-delta publications need."""

from __future__ import annotations

from modal import Volume
from modal.exception import AlreadyExistsError


def dojo_volume(name: str) -> Volume:
    """A Volume Dojo owns, created as v2 when it doesn't exist yet. An existing one keeps its version: ``from_name``
    with a version refuses a Volume of another."""
    try:
        Volume.objects.create(name, version=2)
    except AlreadyExistsError:
        pass
    return Volume.from_name(name)
