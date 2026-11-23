from types import SimpleNamespace

import pytest
from modal.exception import AlreadyExistsError, InvalidError

from modal_dojo.common import volumes


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
