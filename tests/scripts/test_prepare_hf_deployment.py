"""Tests for the Hugging Face deployment shell helper."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_prepare_hf_deployment_repo_id_override(tmp_path: Path) -> None:
    """An exact repo override should target the canonical repo and README URLs."""
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "prepare_hf_deployment.sh"
    staging_dir = tmp_path / "hf-staging"

    env = os.environ.copy()
    env["OPENENV_VERSION"] = "main"

    result = subprocess.run(
        [
            "bash",
            str(script_path),
            "--env",
            "repl_env",
            "--repo-id",
            "openenv/repl",
            "--dry-run",
            "--skip-collection",
            "--staging-dir",
            str(staging_dir),
        ],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[dry-run] Would create/update space: openenv/repl" in result.stdout

    generated_readme = staging_dir / "openenv" / "repl" / "README.md"
    assert generated_readme.exists()
    readme_text = generated_readme.read_text()
    assert "https://huggingface.co/spaces/openenv/repl" in readme_text
    assert "https://huggingface.co/spaces/openenv/repl_env" not in readme_text
