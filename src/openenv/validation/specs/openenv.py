# SPDX-License-Identifier: BSD-3-Clause

"""Trusted adapter for served OpenEnv environment source trees."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import yaml

from .base import (
    AdapterIdentity,
    DetectionMode,
    ExecutionModel,
    SpecIdentity,
    SpecLoad,
    SpecLoadState,
    ValidationSubject,
)
from .harbor import find_harbor_verifier_script, load_harbor_requirements


_MAX_SPEC_BYTES = 1024 * 1024
_MAX_VERIFIER_BYTES = 16 * 1024 * 1024


def _verifier_digest(path: Path) -> str:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as verifier_file:
        while chunk := verifier_file.read(1024 * 1024):
            size += len(chunk)
            if size > _MAX_VERIFIER_BYTES:
                raise ValueError(
                    "declared verifier exceeds the 16 MiB validation limit"
                )
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


class OpenEnvSpecAdapter:
    """Inspect `openenv.yaml` without importing submitted environment code."""

    spec_id = "openenv"
    adapter_id = "openenv-yaml"
    adapter_version = "1"
    execution_model = ExecutionModel.SERVED
    signature_files = ("openenv.yaml",)

    @property
    def _adapter(self) -> AdapterIdentity:
        return AdapterIdentity(self.adapter_id, self.adapter_version)

    def _identity(self, version: str | None) -> SpecIdentity:
        return SpecIdentity(
            spec_id=self.spec_id,
            spec_version=version,
            adapter=self._adapter,
            execution_model=self.execution_model,
        )

    def detect(self, root: Path) -> bool:
        """Detect the OpenEnv signature at the requested root only."""
        signature = root / "openenv.yaml"
        return signature.exists() or signature.is_symlink()

    def inspect(self, root: Path) -> SpecLoad:
        """Safely inspect OpenEnv identity and normalized auxiliary requirements."""
        signature = root / "openenv.yaml"
        unknown_identity = self._identity(None)
        if not signature.exists() and not signature.is_symlink():
            return SpecLoad(state=SpecLoadState.ABSENT, identity=unknown_identity)
        if signature.is_symlink() or not signature.is_file():
            return SpecLoad(
                state=SpecLoadState.INVALID,
                identity=unknown_identity,
                error="openenv.yaml must be a regular file, not a symbolic link",
            )

        try:
            with signature.open("rb") as spec_file:
                document = spec_file.read(_MAX_SPEC_BYTES + 1)
            if len(document) > _MAX_SPEC_BYTES:
                return SpecLoad(
                    state=SpecLoadState.INVALID,
                    identity=unknown_identity,
                    error="openenv.yaml exceeds the 1 MiB validation limit",
                )
            parsed: Any = yaml.safe_load(document)
        except (OSError, RecursionError, UnicodeError, yaml.YAMLError) as exc:
            return SpecLoad(
                state=SpecLoadState.INVALID,
                identity=unknown_identity,
                error=f"Unable to parse openenv.yaml ({type(exc).__name__})",
            )
        if not isinstance(parsed, dict):
            return SpecLoad(
                state=SpecLoadState.INVALID,
                identity=unknown_identity,
                error="openenv.yaml must contain a mapping",
            )

        raw_version = parsed.get("spec_version")
        version = (
            str(raw_version)
            if isinstance(raw_version, (int, str)) and not isinstance(raw_version, bool)
            else None
        )
        identity = self._identity(version)
        if type(raw_version) is not int or raw_version != 1:
            return SpecLoad(
                state=SpecLoadState.UNSUPPORTED,
                identity=identity,
                error=f"Unsupported OpenEnv spec version {raw_version!r}; expected 1",
            )

        requirements = load_harbor_requirements(root)
        verifier_script = find_harbor_verifier_script(root)
        subject = ValidationSubject(
            spec=identity,
            signature_path="openenv.yaml",
            detection_mode=DetectionMode.AUTO,
            requirements=requirements,
            verifier_script=verifier_script,
            verifier_path=(
                os.path.relpath(verifier_script, start=root)
                if verifier_script is not None
                else None
            ),
            verifier_digest=(
                _verifier_digest(verifier_script)
                if verifier_script is not None
                else None
            ),
            document_digest=f"sha256:{hashlib.sha256(document).hexdigest()}",
        )
        return SpecLoad(state=SpecLoadState.LOADED, subject=subject)
