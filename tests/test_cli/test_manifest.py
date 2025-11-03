from pathlib import Path
import pytest

from openenv_cli.utils.manifest import load_manifest, validate_manifest


def write_manifest(root: Path, content: str) -> None:
    (root / "openenv.yaml").write_text(content)
    (root / "server").mkdir(parents=True, exist_ok=True)
    (root / "server" / "app.py").write_text("from fastapi import FastAPI\napp = FastAPI()\n")


def test_manifest_valid(tmp_path):
    write_manifest(
        tmp_path,
        """
spec_version: 1
name: myenv
type: space
runtime: fastapi
app: envs.myenv.server.app:app
port: 8000
""".strip(),
    )
    man = load_manifest(tmp_path)
    assert man is not None
    errors = validate_manifest(man, tmp_path)
    assert errors == []


def test_manifest_missing_fields(tmp_path):
    write_manifest(tmp_path, "{}")
    man = load_manifest(tmp_path)
    assert man is not None
    errors = validate_manifest(man, tmp_path)
    assert any("spec_version" in e for e in errors)
    assert any("name" in e for e in errors)
    assert any("type" in e for e in errors)
    assert any("runtime" in e for e in errors)
    assert any("app" in e for e in errors)
    assert any("port" in e for e in errors)


def test_manifest_structural_checks(tmp_path):
    # no server/app.py
    (tmp_path / "server").mkdir(parents=True, exist_ok=True)
    (tmp_path / "openenv.yaml").write_text(
        """
spec_version: 1
name: myenv
type: space
runtime: fastapi
app: envs.myenv.server.app:app
port: 8000
""".strip()
    )
    man = load_manifest(tmp_path)
    assert man is not None
    errors = validate_manifest(man, tmp_path)
    assert any("server/app.py" in e for e in errors)


