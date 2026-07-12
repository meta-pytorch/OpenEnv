# SPDX-License-Identifier: BSD-3-Clause

"""Static source-inspection helpers shared by validation entry points."""

from __future__ import annotations

import ast
import re
from pathlib import Path


_OPENENV_DOCKER_INSTALL_RE = re.compile(
    r"(?<![a-z0-9_.-])openenv(?:\s*(?:$|[<>=!~@;])|\[)"
)


def _is_safe_regular_file(root: Path, candidate: Path) -> bool:
    """Return whether a regular file is contained without symlink traversal."""
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return False
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            return False
    try:
        resolved_root = root.resolve(strict=True)
        resolved_candidate = candidate.resolve(strict=True)
    except OSError:
        return False
    return bool(
        candidate.is_file() and resolved_candidate.is_relative_to(resolved_root)
    )


def _has_main_guard_call(app_content: str) -> bool:
    """Return True when the module calls main() under a __main__ guard."""
    try:
        tree = ast.parse(app_content)
    except SyntaxError:
        return (
            "__name__" in app_content
            and "__main__" in app_content
            and "main(" in app_content
        )

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.If) or not _is_main_guard(node.test):
            continue

        if any(_contains_main_call(guarded_node) for guarded_node in node.body):
            return True

    return False


def _is_main_guard(test: ast.expr) -> bool:
    """Return True for `if __name__ == "__main__"` tests."""
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


def _contains_main_call(node: ast.AST) -> bool:
    """Return True when an AST node contains a direct `main(...)` call."""
    return any(
        isinstance(candidate, ast.Call)
        and isinstance(candidate.func, ast.Name)
        and candidate.func.id == "main"
        for candidate in ast.walk(node)
    )


def _dockerfile_installs_openenv_runtime(env_path: Path) -> bool:
    """Return True when a Docker deployment installs OpenEnv outside pyproject."""
    for dockerfile_path in (
        env_path / "server" / "Dockerfile",
        env_path / "Dockerfile",
    ):
        if not _is_safe_regular_file(env_path, dockerfile_path):
            continue

        try:
            dockerfile = dockerfile_path.read_text(encoding="utf-8")
        except OSError:
            continue

        for line in dockerfile.splitlines():
            stripped = line.strip().lower()
            if not stripped or stripped.startswith("#"):
                continue
            if _OPENENV_DOCKER_INSTALL_RE.search(stripped):
                return True
            if "openenv-core" in stripped:
                return True

    return False
