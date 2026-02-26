# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""URIDownloader - downloads content from blob://, s3://, and file:// URIs."""

import asyncio
import logging
import shutil
from pathlib import Path
from urllib.parse import urlparse

from .blob import LocalBlobStore

logger = logging.getLogger(__name__)


class URIDownloader:
    """Downloads URI content to a local directory.

    Supports blob://, s3://, and file:// URI schemes.
    Used by both packaging (build-time) and runner (runtime).
    """

    def __init__(self, blob_store: LocalBlobStore | None = None) -> None:
        self._blob_store = blob_store

    async def download(self, uri: str, dest: Path) -> Path:
        """Download URI content into dest directory. Returns path to downloaded content.

        Args:
            uri: URI to download (blob://, s3://, or file://).
            dest: Destination path. For directories, content is copied into this path.
                  For files, content is copied to this path.

        Returns:
            Path to the downloaded content.

        Raises:
            ValueError: If URI scheme is unsupported or blob_store not configured for blob:// URIs.
            FileNotFoundError: If source content doesn't exist.
            RuntimeError: If S3 download fails.
        """
        parsed = urlparse(uri)
        scheme = parsed.scheme

        if scheme == "blob":
            return await self._download_blob(uri, dest)
        elif scheme == "s3":
            return await self._download_s3(uri, dest)
        elif scheme == "file":
            return await self._download_file(uri, dest)
        else:
            raise ValueError(f"Unsupported URI scheme: {scheme!r} (uri={uri})")

    async def _download_blob(self, uri: str, dest: Path) -> Path:
        """Download from blob:// URI using the local blob store."""
        if self._blob_store is None:
            raise ValueError("blob_store required for blob:// URIs")

        blob_path = self._blob_store.get_path(uri)
        dest.parent.mkdir(parents=True, exist_ok=True)

        if blob_path.is_dir():
            shutil.copytree(blob_path, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(blob_path, dest)

        logger.debug("Downloaded blob %s -> %s", uri, dest)
        return dest

    async def _download_s3(self, uri: str, dest: Path) -> Path:
        """Download from s3:// URI using aws CLI."""
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Use --recursive for potential directories
        proc = await asyncio.create_subprocess_exec(
            "aws",
            "s3",
            "cp",
            uri,
            str(dest),
            "--recursive",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            # Retry without --recursive (single file)
            proc = await asyncio.create_subprocess_exec(
                "aws",
                "s3",
                "cp",
                uri,
                str(dest),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                raise RuntimeError(
                    f"aws s3 cp failed (rc={proc.returncode}): {stderr.decode()}"
                )

        logger.debug("Downloaded s3 %s -> %s", uri, dest)
        return dest

    async def _download_file(self, uri: str, dest: Path) -> Path:
        """Download from file:// URI (local copy)."""
        # file:///path/to/thing -> /path/to/thing
        source = Path(urlparse(uri).path)
        if not source.exists():
            raise FileNotFoundError(f"Source not found: {source} (uri={uri})")

        dest.parent.mkdir(parents=True, exist_ok=True)

        if source.is_dir():
            shutil.copytree(source, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(source, dest)

        logger.debug("Downloaded file %s -> %s", uri, dest)
        return dest
