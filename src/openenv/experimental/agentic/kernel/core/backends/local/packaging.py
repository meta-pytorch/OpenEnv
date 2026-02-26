# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""LocalPackagingService - packages agent images from bundles on local disk."""

import logging
import uuid

from ...config import PackageJob, SourceBundle
from ...storage.images import ImageStore
from ...storage.uri import URIDownloader

logger = logging.getLogger(__name__)


class LocalPackagingService:
    """Service for packaging agent images from base + bundles on local disk.

    Copies bundle contents into a local image directory and returns a
    PackageJob with the resulting AgentImage.  For OCI/registry-based
    packaging, see OciPackagingService.
    """

    def __init__(self, downloader: URIDownloader, image_store: ImageStore) -> None:
        self._downloader = downloader
        self._image_store = image_store

    async def create_agent_image(
        self,
        name: str,
        bundles: list[SourceBundle] | None = None,
    ) -> PackageJob:
        """Package an agent image from bundles.

        Creates the image directory, copies bundle contents into it,
        and writes the manifest.
        """
        image_id = str(uuid.uuid4())
        build_id = str(uuid.uuid4())

        try:
            image = self._image_store.create(image_id, name)
            image_dir = image.path

            # Copy bundles into the image
            if bundles:
                bundles_dir = image_dir / "bundles"
                bundles_dir.mkdir(exist_ok=True)

                for bundle in bundles:
                    bundle_name = bundle.labels.get(
                        "name", bundle.uri.split("//")[-1][:12]
                    )
                    dest = bundles_dir / bundle_name
                    await self._downloader.download(bundle.uri, dest)

            logger.info(
                "Built image %s (name=%s, %d bundle(s))",
                image_id,
                name,
                len(bundles or []),
            )
            return PackageJob(id=build_id, status="succeeded", image=image)

        except Exception as e:
            logger.error("Failed to build image %s: %s", name, e, exc_info=True)
            return PackageJob(id=build_id, status="failed", error=str(e))

    async def get_build(self, build_id: str) -> PackageJob | None:
        """Get a build job by ID. Currently a no-op since builds are synchronous."""
        return None
