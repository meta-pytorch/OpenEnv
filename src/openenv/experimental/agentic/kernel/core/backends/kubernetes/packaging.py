# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""OciPackagingService - builds OCI container images and pushes to a registry.

This is the Kubernetes-backend counterpart to LocalPackagingService.
Instead of copying files into a local directory, it:

1. Layers source bundles on top of a base image using ``podman build``
2. Pushes the resulting image to a container registry via ``podman push``
3. Returns a PackageJob whose AgentImage.path points to the registry URI

Proxy configuration is inherited from the environment (HTTP_PROXY, NO_PROXY)
and extended to bypass the configured registry host.
"""

import asyncio
import logging
import os
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlparse

from ...config import PackageJob, SourceBundle
from ...storage.images import ImageStore
from ...storage.uri import URIDownloader

logger = logging.getLogger(__name__)

_NO_PROXY_DEFAULTS = [
    "localhost",
    "127.0.0.1",
    "::1",
]


def _make_proxy_env(
    registry_url: str,
    extra_no_proxy: list[str] | None = None,
) -> dict[str, str]:
    """Build a subprocess env dict inheriting proxy vars from the environment.

    Extends NO_PROXY with the registry hostname and any extras so that
    internal registries are reached directly while external resources
    (Docker Hub, PyPI) go through the configured proxy.
    """
    no_proxy_entries = list(_NO_PROXY_DEFAULTS)
    # Preserve any existing NO_PROXY entries from the environment.
    existing = os.environ.get("NO_PROXY", "")
    for entry in existing.split(","):
        entry = entry.strip()
        if entry and entry not in no_proxy_entries:
            no_proxy_entries.append(entry)
    if extra_no_proxy:
        no_proxy_entries.extend(extra_no_proxy)

    # Extract the registry hostname and add it to NO_PROXY as well.
    if registry_url:
        parsed = urlparse(
            registry_url if "://" in registry_url else f"https://{registry_url}"
        )
        if parsed.hostname:
            no_proxy_entries.append(parsed.hostname)

    no_proxy = ",".join(dict.fromkeys(no_proxy_entries))  # dedupe, preserve order

    env = dict(os.environ)
    env["no_proxy"] = no_proxy
    env["NO_PROXY"] = no_proxy
    return env


class OciPackagingService:
    """Packages agent images as OCI containers for Kubernetes deployment.

    Builds a new image by layering bundles on top of the base image,
    then pushes it to the configured registry.  Requires ``podman``
    on the host.
    """

    def __init__(
        self,
        registry_url: str,
        base_image: str,
        downloader: URIDownloader,
        image_store: ImageStore,
    ) -> None:
        self._registry_url = registry_url
        self._base_image = base_image
        self._downloader = downloader
        self._image_store = image_store
        self._proxy_env = _make_proxy_env(registry_url)

    async def create_agent_image(
        self,
        name: str,
        bundles: list[SourceBundle] | None = None,
    ) -> PackageJob:
        """Build an OCI image from bundles and push to the registry."""
        build_id = str(uuid.uuid4())
        image_id = str(uuid.uuid4())

        # No bundles → just use base image directly
        if not bundles:
            tag = (
                f"{self._registry_url}/{name}:latest"
                if self._registry_url
                else f"{name}:latest"
            )
            image = self._image_store.create(image_id, name, registry_tag=tag)
            return PackageJob(id=build_id, status="succeeded", image=image)

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                build_dir = Path(tmpdir)

                # Download bundles via URIDownloader
                bundles_dir = build_dir / "bundles"
                bundles_dir.mkdir()
                for bundle in bundles:
                    bundle_name = bundle.labels.get(
                        "name", bundle.uri.split("//")[-1][:12]
                    )
                    dest = bundles_dir / bundle_name
                    await self._downloader.download(bundle.uri, dest)

                # Check if any bundle has a requirements.txt
                has_requirements = any(
                    (bundles_dir / d.name / "requirements.txt").exists()
                    for d in bundles_dir.iterdir()
                    if d.is_dir()
                )

                # Generate Dockerfile
                dockerfile_lines = [
                    f"FROM {self._base_image}",
                    "COPY bundles/ /image/bundles/",
                ]
                if has_requirements:
                    dockerfile_lines.append(
                        "RUN for f in /image/bundles/*/requirements.txt; do "
                        '[ -f "$f" ] && pip install -r "$f"; done'
                    )
                dockerfile = build_dir / "Dockerfile"
                dockerfile.write_text("\n".join(dockerfile_lines) + "\n")

                # Build
                tag = (
                    f"{self._registry_url}/{name}:{build_id}"
                    if self._registry_url
                    else f"{name}:{build_id}"
                )
                proc = await asyncio.create_subprocess_exec(
                    "podman",
                    "build",
                    "--network=host",
                    "-t",
                    tag,
                    ".",
                    cwd=str(build_dir),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=self._proxy_env,
                )
                stdout, stderr = await proc.communicate()
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"podman build failed (rc={proc.returncode}): {stderr.decode()}"
                    )
                logger.info("Built image %s", tag)

                # Push
                if self._registry_url:
                    proc = await asyncio.create_subprocess_exec(
                        "podman",
                        "push",
                        tag,
                        cwd=str(build_dir),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        env=self._proxy_env,
                    )
                    stdout, stderr = await proc.communicate()
                    if proc.returncode != 0:
                        raise RuntimeError(
                            f"podman push failed (rc={proc.returncode}): "
                            f"{stderr.decode()}"
                        )
                    logger.info("Pushed image %s", tag)

            image = self._image_store.create(image_id, name, registry_tag=tag)
            return PackageJob(id=build_id, status="succeeded", image=image)

        except Exception as e:
            logger.error("Failed to build image %s: %s", name, e, exc_info=True)
            return PackageJob(id=build_id, status="failed", error=str(e))

    async def get_build(self, build_id: str) -> PackageJob | None:
        """Look up an OCI build job by ID. Currently a no-op since builds are synchronous."""
        return None
