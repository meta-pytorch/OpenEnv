from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


@dataclass
class Manifest:
    raw: Dict[str, Any]

    @property
    def spec_version(self) -> int | None:
        v = self.raw.get("spec_version")
        return int(v) if isinstance(v, int) else None

    @property
    def name(self) -> str | None:
        v = self.raw.get("name")
        return v if isinstance(v, str) and v.strip() else None

    @property
    def type(self) -> str | None:
        v = self.raw.get("type")
        return v if isinstance(v, str) else None

    @property
    def runtime(self) -> str | None:
        v = self.raw.get("runtime")
        return v if isinstance(v, str) else None

    @property
    def app(self) -> str | None:
        v = self.raw.get("app")
        return v if isinstance(v, str) and ":" in v else None

    @property
    def port(self) -> int | None:
        v = self.raw.get("port")
        return int(v) if isinstance(v, int) else None


def load_manifest(env_root: Path) -> Manifest | None:
    path = env_root / "openenv.yaml"
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        data = {}
    return Manifest(raw=data)


def validate_manifest(man: Manifest, env_root: Path) -> List[str]:
    errors: List[str] = []
    if man.spec_version is None:
        errors.append("spec_version must be an integer (e.g., 1)")
    if man.name is None:
        errors.append("name must be a non-empty string")
    if man.type != "space":
        errors.append("type must be 'space'")
    if man.runtime != "fastapi":
        errors.append("runtime must be 'fastapi'")
    if man.app is None:
        errors.append("app must be a dotted path ending with :app (e.g., envs.<name>.server.app:app)")
    if man.port is None or not (1024 <= man.port <= 65535):
        errors.append("port must be an integer between 1024 and 65535")

    # Structural checks
    server_dir = env_root / "server"
    if not server_dir.exists() or not server_dir.is_dir():
        errors.append("server/ directory is required at the environment root")
    app_py = server_dir / "app.py"
    if not app_py.exists():
        errors.append("server/app.py is required for fastapi runtime")

    return errors


