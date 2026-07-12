# SPDX-License-Identifier: BSD-3-Clause

"""Tests for spec detection and the normalized subject boundary."""

from __future__ import annotations

from pathlib import Path

import pytest
from openenv.validation.specs import (
    AdapterIdentity,
    DetectionMode,
    ExecutionModel,
    NetworkMode,
    RequirementsLoad,
    RequirementsState,
    SpecIdentity,
    SpecLoad,
    SpecLoadState,
    ValidationRequirements,
    ValidationSpecRegistry,
    ValidationSubject,
)


class _FakeSpecAdapter:
    adapter_id = "fake-adapter"
    adapter_version = "7"
    execution_model = ExecutionModel.ONE_SHOT
    signature_files = ("task.fake",)

    def __init__(self, spec_id: str = "fake") -> None:
        self.spec_id = spec_id

    def detect(self, root: Path) -> bool:
        return (root / "task.fake").exists()

    def inspect(self, root: Path) -> SpecLoad:
        identity = SpecIdentity(
            spec_id=self.spec_id,
            spec_version="2",
            adapter=AdapterIdentity(self.adapter_id, self.adapter_version),
            execution_model=self.execution_model,
        )
        if not self.detect(root):
            return SpecLoad(state=SpecLoadState.ABSENT, identity=identity)
        subject = ValidationSubject(
            spec=identity,
            signature_path="task.fake",
            detection_mode=DetectionMode.AUTO,
            requirements=RequirementsLoad(
                state=RequirementsState.LOADED,
                requirements=ValidationRequirements(),
            ),
        )
        return SpecLoad(state=SpecLoadState.LOADED, subject=subject)


def _write_openenv_spec(root: Path) -> None:
    root.mkdir(exist_ok=True)
    (root / "openenv.yaml").write_text(
        "spec_version: 1\n"
        "name: example\n"
        "runtime: fastapi\n"
        "app: server.app:app\n"
        "port: 8000\n"
    )


def test_default_registry_detects_openenv_and_adapts_harbor_requirements(
    tmp_path: Path,
) -> None:
    from openenv.validation.specs import DEFAULT_SPEC_REGISTRY

    _write_openenv_spec(tmp_path)
    (tmp_path / "task.toml").write_text(
        'schema_version = "1.1"\n[environment]\nallow_internet = false\n'
    )

    loaded = DEFAULT_SPEC_REGISTRY.resolve(tmp_path)

    assert loaded.state is SpecLoadState.LOADED
    assert loaded.subject is not None
    assert loaded.subject.spec.spec_id == "openenv"
    assert loaded.subject.spec.execution_model is ExecutionModel.SERVED
    assert loaded.subject.detection_mode is DetectionMode.AUTO
    assert loaded.subject.requirements.state is RequirementsState.LOADED
    requirements = loaded.subject.requirements.requirements
    assert requirements is not None
    assert requirements.environment.network.mode is NetworkMode.DENY_ALL
    assert loaded.subject.document_digest is not None


def test_registry_supports_explicit_selection_and_structured_absence(
    tmp_path: Path,
) -> None:
    registry = ValidationSpecRegistry((_FakeSpecAdapter(),))

    absent = registry.resolve(tmp_path)
    explicit_absent = registry.resolve(tmp_path, spec_id="fake")

    assert absent.state is SpecLoadState.ABSENT
    assert absent.matches == ()
    assert explicit_absent.state is SpecLoadState.ABSENT
    assert explicit_absent.spec is not None
    assert explicit_absent.spec.spec_id == "fake"

    (tmp_path / "task.fake").write_text("signature")
    loaded = registry.resolve(tmp_path, spec_id="fake")
    assert loaded.state is SpecLoadState.LOADED
    assert loaded.subject is not None
    assert loaded.subject.detection_mode is DetectionMode.EXPLICIT


def test_registry_rejects_duplicate_and_ambiguous_specs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unique"):
        ValidationSpecRegistry((_FakeSpecAdapter(), _FakeSpecAdapter()))

    first = _FakeSpecAdapter("first")
    second = _FakeSpecAdapter("second")
    registry = ValidationSpecRegistry((first, second))
    (tmp_path / "task.fake").write_text("signature")

    loaded = registry.resolve(tmp_path)

    assert loaded.state is SpecLoadState.AMBIGUOUS
    assert loaded.matches == ("first", "second")
    assert loaded.error == "Multiple validation specs matched: first, second"


def test_malformed_openenv_signature_is_not_treated_as_absent(tmp_path: Path) -> None:
    from openenv.validation.specs import DEFAULT_SPEC_REGISTRY

    (tmp_path / "openenv.yaml").write_text("spec_version: [")

    loaded = DEFAULT_SPEC_REGISTRY.resolve(tmp_path)

    assert loaded.state is SpecLoadState.INVALID
    assert loaded.matches == ("openenv",)
    assert loaded.error is not None
    assert loaded.spec is not None
    assert loaded.spec.adapter == AdapterIdentity("openenv-yaml", "1")
    assert loaded.spec.execution_model is ExecutionModel.SERVED


def test_registry_generated_failures_retain_adapter_identity(tmp_path: Path) -> None:
    class _ExplodingAdapter(_FakeSpecAdapter):
        def inspect(self, root: Path) -> SpecLoad:
            raise RuntimeError("boom")

    (tmp_path / "task.fake").write_text("signature")
    loaded = ValidationSpecRegistry((_ExplodingAdapter(),)).resolve(tmp_path)

    assert loaded.state is SpecLoadState.INVALID
    assert loaded.spec == SpecIdentity(
        spec_id="fake",
        spec_version=None,
        adapter=AdapterIdentity("fake-adapter", "7"),
        execution_model=ExecutionModel.ONE_SHOT,
    )


def test_spec_load_rejects_conflicting_subject_identity() -> None:
    identity = SpecIdentity(
        spec_id="fake",
        spec_version="2",
        adapter=AdapterIdentity("fake-adapter", "7"),
        execution_model=ExecutionModel.ONE_SHOT,
    )
    subject = ValidationSubject(
        spec=identity,
        signature_path="task.fake",
        detection_mode=DetectionMode.AUTO,
        requirements=RequirementsLoad(state=RequirementsState.ABSENT),
    )
    conflicting = SpecIdentity(
        spec_id="other",
        spec_version="2",
        adapter=AdapterIdentity("fake-adapter", "7"),
        execution_model=ExecutionModel.ONE_SHOT,
    )

    with pytest.raises(ValueError, match="identity"):
        SpecLoad(
            state=SpecLoadState.LOADED,
            subject=subject,
            identity=conflicting,
        )


@pytest.mark.parametrize("version", ["true", "1.0"])
def test_openenv_spec_version_must_be_the_strict_integer_one(
    tmp_path: Path, version: str
) -> None:
    from openenv.validation.specs import DEFAULT_SPEC_REGISTRY

    (tmp_path / "openenv.yaml").write_text(f"spec_version: {version}\n")

    loaded = DEFAULT_SPEC_REGISTRY.resolve(tmp_path)

    assert loaded.state is SpecLoadState.UNSUPPORTED
    assert loaded.error is not None


def test_dangling_openenv_signature_is_invalid_not_absent(tmp_path: Path) -> None:
    from openenv.validation.specs import DEFAULT_SPEC_REGISTRY

    try:
        (tmp_path / "openenv.yaml").symlink_to(tmp_path / "missing.yaml")
    except OSError:
        pytest.skip("symbolic links are unavailable on this platform")

    loaded = DEFAULT_SPEC_REGISTRY.resolve(tmp_path)

    assert loaded.state is SpecLoadState.INVALID
    assert "symbolic link" in (loaded.error or "")
