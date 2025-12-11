"""
Unified OpenEnv package bundling the CLI and core runtime.
"""

from importlib import metadata

__all__ = ["core", "cli"]

try:
    __version__ = metadata.version("openenv")  # type: ignore[arg-type]
except metadata.PackageNotFoundError:  # pragma: no cover - local dev
    __version__ = "0.0.0"



