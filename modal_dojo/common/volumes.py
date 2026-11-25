"""The Volumes Dojo creates: v2, whose hard links Miles' GPU-delta publications need."""

from __future__ import annotations

from modal import Volume
from modal.exception import AlreadyExistsError, InvalidError

from modal_dojo.common.errors import DojoConfigError


def dojo_volume(name: str) -> Volume:
    """A Volume Dojo owns, created as v2 when it doesn't exist yet. An existing one keeps its version: ``from_name``
    with a version refuses a Volume of another."""
    try:
        Volume.objects.create(name, version=2)
    except AlreadyExistsError:
        pass
    return Volume.from_name(name)


def require_v2_volume(name: str, need: str) -> None:
    """Refuse a v1 Volume where a run needs v2, before it starts."""
    try:
        Volume.from_name(name, version=2).hydrate()
    except InvalidError:
        raise DojoConfigError(
            f"Volume {name!r} is v1; {need}, which a v1 Volume refuses. Copy it into a new v2 Volume "
            f"(`modal volume create --version 2 <new>`), then swap their names with `modal volume rename`."
        ) from None
