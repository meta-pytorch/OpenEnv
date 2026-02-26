# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""S3-backed content-addressable blob storage.

Uses the ``aws`` CLI via subprocess for S3 operations, following the same
pattern as OciPackagingService (which uses ``podman`` via subprocess).
"""

import asyncio
import hashlib
import logging
import tarfile
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class S3BlobStore:
    """Content-addressable blob storage backed by S3.

    Files are stored by their SHA-256 hash. Directories are tarred
    before upload. The URI format is ``s3://{bucket}/{prefix}/{sha256}``.
    """

    def __init__(self, bucket: str, prefix: str = "blobs") -> None:
        self._bucket = bucket
        self._prefix = prefix

    def _s3_key(self, blob_hash: str) -> str:
        """Build the full S3 key for a blob hash."""
        return f"{self._prefix}/{blob_hash}"

    def _s3_uri(self, blob_hash: str) -> str:
        """Build the full s3:// URI for a blob hash."""
        return f"s3://{self._bucket}/{self._s3_key(blob_hash)}"

    async def upload(self, path: str | Path) -> str:
        """Upload a single file to S3.

        Returns an s3:// URI that can be used to reference the file.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Not a file: {path}")

        content = path.read_bytes()
        blob_hash = hashlib.sha256(content).hexdigest()

        # Check if already uploaded
        uri = self._s3_uri(blob_hash)
        if await self.exists(uri):
            logger.debug("Blob already exists: %s", uri)
            return uri

        # Upload
        s3_dest = f"s3://{self._bucket}/{self._s3_key(blob_hash)}"
        proc = await asyncio.create_subprocess_exec(
            "aws",
            "s3",
            "cp",
            str(path),
            s3_dest,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"aws s3 cp failed (rc={proc.returncode}): {stderr.decode()}"
            )

        logger.info("Uploaded %s -> %s", path, uri)
        return uri

    async def upload_dir(self, path: str | Path) -> str:
        """Upload a directory to S3 as a tar.gz archive.

        Returns an s3:// URI.
        """
        path = Path(path)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")

        # Hash directory contents deterministically
        hasher = hashlib.sha256()
        for file_path in sorted(path.rglob("*")):
            if file_path.is_file():
                rel = file_path.relative_to(path)
                hasher.update(str(rel).encode())
                hasher.update(file_path.read_bytes())
        blob_hash = hasher.hexdigest()

        uri = self._s3_uri(blob_hash)
        if await self.exists(uri):
            logger.debug("Blob already exists: %s", uri)
            return uri

        # Create tar.gz in a temp file and upload
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=True) as tmp:
            with tarfile.open(tmp.name, "w:gz") as tar:
                tar.add(str(path), arcname=".")
            tmp.flush()

            s3_dest = f"s3://{self._bucket}/{self._s3_key(blob_hash)}"
            proc = await asyncio.create_subprocess_exec(
                "aws",
                "s3",
                "cp",
                tmp.name,
                s3_dest,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(
                    f"aws s3 cp failed (rc={proc.returncode}): {stderr.decode()}"
                )

        logger.info("Uploaded dir %s -> %s", path, uri)
        return uri

    async def get_url(self, uri: str) -> str:
        """Return the raw S3 URL for a given s3:// URI.

        Converts s3://bucket/key to https://bucket.s3.amazonaws.com/key.
        """
        if not uri.startswith("s3://"):
            raise ValueError(f"Not an s3:// URI: {uri}")
        # s3://bucket/prefix/hash -> bucket, prefix/hash
        without_scheme = uri[5:]
        bucket, _, key = without_scheme.partition("/")
        return f"https://{bucket}.s3.amazonaws.com/{key}"

    async def exists(self, uri: str) -> bool:
        """Check if a URI exists in S3."""
        if not uri.startswith("s3://"):
            raise ValueError(f"Not an s3:// URI: {uri}")

        proc = await asyncio.create_subprocess_exec(
            "aws",
            "s3",
            "ls",
            uri,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        return proc.returncode == 0
