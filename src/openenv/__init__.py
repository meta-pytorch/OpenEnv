"""
Unified OpenEnv package bundling the CLI and core runtime.
"""

from importlib import metadata

from .auto import AutoAction, AutoEnv

__all__ = ["core", "cli", "AutoEnv", "AutoAction"]

try:
    __version__ = metadata.version("openenv")  # type: ignore[arg-type]
except metadata.PackageNotFoundError:  # pragma: no cover - local dev
    __version__ = "0.0.0"
