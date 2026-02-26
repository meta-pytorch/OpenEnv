# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Image storage for agent images (built from bundles + base)."""

import json
from pathlib import Path

from ..config import AgentImage


class ImageStore:
    """Manages agent images on the local filesystem.

    Each image is a directory containing:
    - manifest.json: image metadata
    - bundles/: extracted bundle contents
    """

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create(
        self, image_id: str, image_name: str, registry_tag: str | None = None
    ) -> AgentImage:
        """Create a new image directory and write its manifest.

        For OCI images, pass ``registry_tag`` — the spawner will use it
        as the container image reference instead of the local directory.
        """
        image_dir = self.base_dir / image_id
        image_dir.mkdir(parents=True, exist_ok=True)

        manifest: dict[str, str] = {
            "id": image_id,
            "name": image_name,
        }
        if registry_tag:
            manifest["registry_tag"] = registry_tag

        manifest_path = image_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

        path = Path(registry_tag) if registry_tag else image_dir
        return AgentImage(id=image_id, name=image_name, path=path)

    def get(self, image_id: str) -> AgentImage | None:
        """Load an image from disk by its ID."""
        image_dir = self.base_dir / image_id
        manifest_path = image_dir / "manifest.json"
        if not manifest_path.exists():
            return None

        manifest = json.loads(manifest_path.read_text())
        registry_tag = manifest.get("registry_tag")
        path = Path(registry_tag) if registry_tag else image_dir
        return AgentImage(
            id=manifest["id"],
            name=manifest["name"],
            path=path,
        )

    def exists(self, image_id: str) -> bool:
        """Check if an image exists."""
        return (self.base_dir / image_id / "manifest.json").exists()

    def get_path(self, image_id: str) -> Path:
        """Get the filesystem path for an image.

        Raises KeyError if the image doesn't exist.
        """
        image_dir = self.base_dir / image_id
        if not (image_dir / "manifest.json").exists():
            raise KeyError(f"Image not found: {image_id}")
        return image_dir
