from types import SimpleNamespace

import pytest
from modal.exception import AlreadyExistsError, InvalidError

from modal_dojo.common import volumes
from modal_dojo.common.errors import DojoConfigError


class FakeVolumes:
    """Modal's Volume API over named Volumes and their versions."""

    def __init__(self, existing):
        self.existing = dict(existing)
        self.objects = SimpleNamespace(create=self.create)

    def create(self, name, *, version=None):
        if name in self.existing:
            raise AlreadyExistsError(f"Volume {name!r} already exists")
        self.existing[name] = version or 1

    def from_name(self, name, *, version=None):
        def hydrate():
            if version is not None and self.existing[name] != version:
                raise InvalidError(
                    f"Volume {name!r} exists but has version v{self.existing[name]}"
                )

        return SimpleNamespace(name=name, version=version, hydrate=hydrate)


@pytest.fixture
def fake(monkeypatch):
    fake = FakeVolumes({"old-v1": 1, "old-v2": 2})
    monkeypatch.setattr(volumes, "Volume", fake)
    return fake


def test_a_new_volume_is_created_as_v2_and_an_existing_one_keeps_its_version(fake):
    for name in ("new", "old-v1", "old-v2"):
        assert volumes.dojo_volume(name).version is None

    assert fake.existing == {"old-v1": 1, "old-v2": 2, "new": 2}


def test_a_run_that_needs_v2_refuses_a_v1_volume(fake):
    volumes.require_v2_volume("old-v2", "a need")

    with pytest.raises(
        DojoConfigError,
        match=r"'old-v1' is v1; a need.*modal volume create --version 2",
    ):
        volumes.require_v2_volume("old-v1", "a need")
