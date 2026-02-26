# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Local blob storage for agent code bundles and artifacts."""

import hashlib
import shutil
from pathlib import Path


class LocalBlobStore:
    """Content-addressable local blob storage.

    Files and directories are stored by their content hash (SHA-256).
    This deduplicates identical uploads and provides stable URIs.
    """

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def upload(self, path: str | Path) -> str:
        """Upload a single file to blob storage.

        Returns a blob:// URI that can be used to reference the file.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Not a file: {path}")

        content = path.read_bytes()
        blob_hash = hashlib.sha256(content).hexdigest()
        blob_dir = self.base_dir / blob_hash
        blob_dir.mkdir(exist_ok=True)

        dest = blob_dir / path.name
        dest.write_bytes(content)

        return f"blob://{blob_hash}"

    def upload_dir(self, path: str | Path) -> str:
        """Upload a directory to blob storage.

        The entire directory tree is copied under a content-addressed key.
        Returns a blob:// URI.
        """
        path = Path(path)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")

        # Hash the directory contents deterministically
        hasher = hashlib.sha256()
        for file_path in sorted(path.rglob("*")):
            if file_path.is_file():
                rel = file_path.relative_to(path)
                hasher.update(str(rel).encode())
                hasher.update(file_path.read_bytes())

        blob_hash = hasher.hexdigest()
        blob_dir = self.base_dir / blob_hash

        if not blob_dir.exists():
            shutil.copytree(path, blob_dir, dirs_exist_ok=True)

        return f"blob://{blob_hash}"

    def get_path(self, uri: str) -> Path:
        """Resolve a blob:// URI to its local path.

        Raises KeyError if the blob doesn't exist.
        """
        blob_hash = uri.removeprefix("blob://")
        blob_dir = self.base_dir / blob_hash
        if not blob_dir.exists():
            raise KeyError(f"Blob not found: {uri}")
        return blob_dir

    def exists(self, uri: str) -> bool:
        """Check if a blob URI exists in the store."""
        blob_hash = uri.removeprefix("blob://")
        return (self.base_dir / blob_hash).exists()
