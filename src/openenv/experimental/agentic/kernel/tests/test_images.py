# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Tests for ImageStore."""

from pathlib import Path

import pytest
from agentic.kernel.core.storage.images import ImageStore


@pytest.fixture
def image_store(tmp_path: Path) -> ImageStore:
    return ImageStore(tmp_path / "images")


class TestImageStore:
    def test_create_writes_manifest(self, image_store: ImageStore):
        image = image_store.create("img1", "test-image")
        assert (image.path / "manifest.json").exists()

    def test_create_returns_image_with_path(self, image_store: ImageStore):
        image = image_store.create("img1", "test-image")
        assert image.path.is_dir()
        assert image.id == "img1"
        assert image.name == "test-image"

    def test_get_roundtrip(self, image_store: ImageStore):
        image_store.create("roundtrip", "my-agent")

        loaded = image_store.get("roundtrip")
        assert loaded is not None
        assert loaded.id == "roundtrip"
        assert loaded.name == "my-agent"

    def test_get_nonexistent_returns_none(self, image_store: ImageStore):
        assert image_store.get("nonexistent") is None

    def test_exists(self, image_store: ImageStore):
        assert not image_store.exists("img1")
        image_store.create("img1", "test-image")
        assert image_store.exists("img1")

    def test_get_path(self, image_store: ImageStore):
        image_store.create("img1", "test-image")
        path = image_store.get_path("img1")
        assert path.is_dir()
        assert (path / "manifest.json").exists()

    def test_get_path_nonexistent_raises(self, image_store: ImageStore):
        with pytest.raises(KeyError, match="not found"):
            image_store.get_path("nonexistent")

    def test_multiple_images(self, image_store: ImageStore):
        image_store.create("img1", "first")
        image_store.create("img2", "second")

        first = image_store.get("img1")
        second = image_store.get("img2")
        assert first.name == "first"
        assert second.name == "second"

    def test_create_with_registry_tag(self, image_store: ImageStore):
        image = image_store.create(
            "oci-img", "worker", registry_tag="registry.example.com/worker:abc123"
        )
        assert str(image.path) == "registry.example.com/worker:abc123"

    def test_get_roundtrip_with_registry_tag(self, image_store: ImageStore):
        image_store.create(
            "oci-rt", "worker", registry_tag="registry.example.com/worker:v1"
        )
        loaded = image_store.get("oci-rt")
        assert loaded is not None
        assert loaded.name == "worker"
        assert str(loaded.path) == "registry.example.com/worker:v1"

    def test_create_without_registry_tag_uses_local_dir(self, image_store: ImageStore):
        image = image_store.create("local-img", "agent")
        assert image.path.is_dir()
        assert (image.path / "manifest.json").exists()
