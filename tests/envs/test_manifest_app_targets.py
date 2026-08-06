# SPDX-License-Identifier: BSD-3-Clause

"""Every `openenv.yaml` `app` field must name a module that exists.

The cloud providers do not read the Dockerfile `CMD` when a manifest is
present: `ModalProvider._discover_server_cmd` (and its Daytona twin) locate
`openenv.yaml` inside the sandbox, take the `app` field verbatim, and run
`cd <env_root> && python -m uvicorn <app>`. A typo there is invisible locally --
`docker run` uses the Dockerfile `CMD` and works fine -- and only surfaces as a
`ModuleNotFoundError` when someone launches the env on Modal or Daytona.

The check resolves the module against the env directory on disk rather than
importing it. Importing would pull in playwright, carla, dm_control and the rest
of the optional-dependency tail, so the module would skip on exactly the CI
machines that should be guarding this, which is how the drift this test pins
went unnoticed.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ENVS_DIR = REPO_ROOT / "envs"


def _manifests_with_app() -> list[pathlib.Path]:
    """Env dirs whose manifest declares an `app` field.

    A handful of envs still carry the pre-spec manifest format, which has no
    `app` field at all. Those are a separate problem; skipping them here keeps
    this test about the one thing it can prove.
    """
    found = []
    for manifest in sorted(ENVS_DIR.glob("*/openenv.yaml")):
        data = yaml.safe_load(manifest.read_text()) or {}
        if isinstance(data, dict) and isinstance(data.get("app"), str):
            found.append(manifest)
    return found


MANIFESTS = _manifests_with_app()


def test_manifests_with_app_field_were_found() -> None:
    """Guard against the glob silently matching nothing."""
    assert len(MANIFESTS) > 20


@pytest.mark.parametrize("manifest", MANIFESTS, ids=lambda p: p.parent.name)
def test_manifest_app_target_resolves(manifest: pathlib.Path) -> None:
    env_dir = manifest.parent
    app = yaml.safe_load(manifest.read_text())["app"]

    assert ":" in app, (
        f"{manifest.relative_to(REPO_ROOT)} declares app={app!r}, which is not "
        "in uvicorn's '<module>:<attribute>' form."
    )

    module_path = app.split(":", 1)[0]
    relative = pathlib.Path(*module_path.split("."))

    # The env directory is the container's env root: the image is built with the
    # env dir as its context, so `<env>/server/app.py` lands at
    # `/app/env/server/app.py` and `server.app:app` resolves from there.
    module_file = env_dir / relative.with_suffix(".py")
    package_init = env_dir / relative / "__init__.py"

    assert module_file.is_file() or package_init.is_file(), (
        f"{manifest.relative_to(REPO_ROOT)} declares app={app!r}, but module "
        f"{module_path!r} does not exist under {env_dir.relative_to(REPO_ROOT)}"
        f" (looked for {module_file.relative_to(env_dir)} and "
        f"{(relative / '__init__.py')}). The cloud providers run "
        f"'cd /app/env && python -m uvicorn {app}', so this env fails to start "
        "on Modal and Daytona."
    )
