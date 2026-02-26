# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Tests for LocalPackagingService."""

from pathlib import Path

import pytest
from agentic.kernel.core.backends.local.packaging import LocalPackagingService
from agentic.kernel.core.config import SourceBundle
from agentic.kernel.core.storage.blob import LocalBlobStore
from agentic.kernel.core.storage.images import ImageStore
from agentic.kernel.core.storage.uri import URIDownloader


@pytest.fixture
def blob_store(tmp_path: Path) -> LocalBlobStore:
    return LocalBlobStore(tmp_path / "blobs")


@pytest.fixture
def image_store(tmp_path: Path) -> ImageStore:
    return ImageStore(tmp_path / "images")


@pytest.fixture
def packaging_service(
    blob_store: LocalBlobStore, image_store: ImageStore
) -> LocalPackagingService:
    downloader = URIDownloader(blob_store=blob_store)
    return LocalPackagingService(downloader, image_store)


class TestLocalPackagingService:
    @pytest.mark.asyncio
    async def test_build_image_no_bundles(
        self, packaging_service: LocalPackagingService
    ):
        job = await packaging_service.create_agent_image(
            name="simple",
        )
        assert job.status == "succeeded"
        assert job.image is not None
        assert job.image.name == "simple"

    @pytest.mark.asyncio
    async def test_build_image_with_file_bundle(
        self,
        packaging_service: LocalPackagingService,
        blob_store: LocalBlobStore,
        tmp_path: Path,
    ):
        # Create a file and upload it
        spec_file = tmp_path / "spec.md"
        spec_file.write_text("# My Spec\nDetails here.")
        spec_uri = blob_store.upload(spec_file)

        job = await packaging_service.create_agent_image(
            name="with-spec",
            bundles=[
                SourceBundle(uri=spec_uri, labels={"type": "reference", "name": "spec"})
            ],
        )
        assert job.status == "succeeded"

        # Verify bundle was copied into image
        bundles_dir = job.image.path / "bundles"
        assert bundles_dir.exists()
        # The bundle is a directory (blob stores files inside a hash dir)
        spec_bundle = bundles_dir / "spec"
        assert spec_bundle.exists()

    @pytest.mark.asyncio
    async def test_build_image_with_dir_bundle(
        self,
        packaging_service: LocalPackagingService,
        blob_store: LocalBlobStore,
        tmp_path: Path,
    ):
        # Create a directory with files and upload it
        helpers_dir = tmp_path / "helpers"
        helpers_dir.mkdir()
        (helpers_dir / "calc.py").write_text("def add(a, b): return a + b")
        (helpers_dir / "utils.py").write_text("def noop(): pass")
        helpers_uri = blob_store.upload_dir(helpers_dir)

        job = await packaging_service.create_agent_image(
            name="with-helpers",
            bundles=[
                SourceBundle(
                    uri=helpers_uri, labels={"type": "helpers", "name": "myhelpers"}
                )
            ],
        )
        assert job.status == "succeeded"

        # Verify directory structure preserved
        myhelpers = job.image.path / "bundles" / "myhelpers"
        assert (myhelpers / "calc.py").read_text() == "def add(a, b): return a + b"
        assert (myhelpers / "utils.py").read_text() == "def noop(): pass"

    @pytest.mark.asyncio
    async def test_build_image_with_invalid_blob_fails(
        self, packaging_service: LocalPackagingService
    ):
        job = await packaging_service.create_agent_image(
            name="bad",
            bundles=[
                SourceBundle(uri="blob://nonexistent", labels={"type": "helpers"})
            ],
        )
        assert job.status == "failed"
        assert job.error is not None

    @pytest.mark.asyncio
    async def test_built_image_persisted_in_store(
        self,
        packaging_service: LocalPackagingService,
        image_store: ImageStore,
    ):
        job = await packaging_service.create_agent_image(name="persisted")
        assert job.status == "succeeded"

        # Verify we can load it back from the store
        loaded = image_store.get(job.image.id)
        assert loaded is not None
        assert loaded.name == "persisted"
