# SPDX-License-Identifier: BSD-3-Clause

"""Tests for validation policy planning and capability selection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from openenv.validation.models import (
    CheckOutcome,
    ValidationCapability,
    ValidationCheck,
    ValidationPolicy,
    ValidationProfile,
    ValidationRequirementBinding,
    ValidationSeverity,
)
from openenv.validation.planner import build_validation_plan, is_canonical_policy_plan
from openenv.validation.specs import (
    AdapterIdentity,
    DEFAULT_SPEC_REGISTRY,
    DetectionMode,
    EnvironmentRequirements,
    ExecutionModel,
    NetworkMode,
    NetworkRequirements,
    PhaseRequirements,
    RequirementsLoad,
    RequirementsState,
    SpecIdentity,
    SpecLoad,
    SpecLoadState,
    ValidationRequirements,
    ValidationSpecRegistry,
    ValidationSubject,
)

from ._helpers import write_harbor_task, write_valid_env


def test_plan_carries_sanitized_harbor_execution_requirements(tmp_path: Path) -> None:
    env_dir = tmp_path / "test_env"
    write_valid_env(env_dir)
    (env_dir / "task.toml").write_text(
        'schema_version = "1.1"\n'
        'artifacts = ["/workspace/result.json"]\n'
        "[task]\n"
        'name = "org/task"\n'
        'description = "https://alice:hunter2@example.com/private"\n'
        "[agent]\n"
        "timeout_sec = 120\n"
        "[verifier]\n"
        "timeout_sec = 30\n"
        'env = { API_TOKEN = "secret-value" }\n'
        "[environment]\n"
        "build_timeout_sec = 300\n"
        "cpus = 4\n"
        "memory_mb = 8192\n"
        "gpus = 1\n"
        "allow_internet = false\n"
        "[environment.healthcheck]\n"
        'command = "curl -f http://127.0.0.1:8000/health"\n'
        "retries = 5\n"
    )

    plan, _context = build_validation_plan(env_dir)
    payload = plan.to_dict()
    serialized = json.dumps(payload)

    assert payload["requirements"]["resources"]["cpus"] == 4
    assert payload["requirements"]["resources"]["memory_mb"] == 8192
    assert payload["requirements"]["resources"]["gpus"] == 1
    assert payload["requirements"]["network"]["mode"] == "deny_all"
    assert payload["requirements"]["timeouts"] == {
        "build_s": 300.0,
        "agent_s": 120.0,
        "verifier_s": 30.0,
    }
    assert payload["requirements"]["healthcheck"]["retries"] == 5
    assert payload["requirements"]["artifacts"][0]["source"] == (
        "/workspace/result.json"
    )
    assert "API_TOKEN" in serialized
    assert "secret-value" not in serialized
    assert "alice:hunter2" not in serialized
    assert payload["spec"]["id"] == "openenv"
    assert payload["spec"]["adapter"] == {"id": "openenv-yaml", "version": "1"}
    assert payload["spec"]["execution_model"] == "served"
    assert payload["spec"]["requirements"]["id"] == "harbor"
    assert payload["spec"]["requirements"]["document_digest"].startswith("sha256:")


def test_plan_includes_harbor_step_requirements_and_sibling_verifier(
    tmp_path: Path,
) -> None:
    task_root = tmp_path / "harbor-task"
    environment = write_harbor_task(task_root)

    plan, _context = build_validation_plan(
        environment,
        profile=ValidationProfile.RUNTIME,
    )

    assert "custom.verifier" in {check.criterion_id for check in plan.checks}
    requirements = plan.to_dict()["requirements"]
    assert requirements["steps"][0]["min_reward"] == {
        "correctness": 0.8,
        "style": 0.5,
    }
    assert requirements["steps"][0]["artifacts"][0]["source"] == (
        "/workspace/result.json"
    )


class _OneShotAdapter:
    spec_id = "one-shot-test"
    adapter_id = "one-shot-test-adapter"
    adapter_version = "1"
    execution_model = ExecutionModel.ONE_SHOT
    signature_files = ("task.oneshot",)

    def detect(self, root: Path) -> bool:
        return (root / "task.oneshot").is_file()

    def inspect(self, root: Path) -> SpecLoad:
        identity = SpecIdentity(
            spec_id=self.spec_id,
            spec_version="1",
            adapter=AdapterIdentity(self.adapter_id, self.adapter_version),
            execution_model=self.execution_model,
        )
        if not self.detect(root):
            return SpecLoad(state=SpecLoadState.ABSENT, identity=identity)
        verifier = root / "verify.sh"
        verifier_declared = verifier.is_file()
        return SpecLoad(
            state=SpecLoadState.LOADED,
            subject=ValidationSubject(
                spec=identity,
                signature_path="task.oneshot",
                detection_mode=DetectionMode.AUTO,
                requirements=RequirementsLoad(
                    state=RequirementsState.LOADED,
                    requirements=ValidationRequirements(
                        verifier=PhaseRequirements(timeout_s=17.0),
                        environment=EnvironmentRequirements(
                            network=NetworkRequirements(mode=NetworkMode.DENY_ALL)
                        ),
                    ),
                ),
                verifier_script=(verifier if verifier_declared else None),
                verifier_path=("verify.sh" if verifier_declared else None),
                verifier_digest=(
                    f"sha256:{hashlib.sha256(verifier.read_bytes()).hexdigest()}"
                    if verifier_declared
                    else None
                ),
            ),
        )


def test_custom_spec_uses_its_policy_without_openenv_checks(tmp_path: Path) -> None:
    (tmp_path / "task.oneshot").write_text("signature")
    registry = ValidationSpecRegistry((_OneShotAdapter(),))
    check = ValidationCheck(
        criterion_id="spec.one_shot.structure",
        requirement="The one-shot package has its required structure",
        capabilities=frozenset({ValidationCapability.SOURCE}),
        severity=ValidationSeverity.BLOCKING,
        timeout_s=1.0,
        evaluator=lambda _context: CheckOutcome.pass_(),
    )
    policy = ValidationPolicy(
        version="one-shot-v1",
        supported_subjects=frozenset({("one-shot-test", ExecutionModel.ONE_SHOT)}),
        checks=(check,),
    )

    plan, context = build_validation_plan(
        tmp_path, spec_registry=registry, policy=policy
    )

    assert plan.policy_version == "one-shot-v1"
    assert [planned.criterion_id for planned in plan.checks] == [
        "spec.one_shot.structure"
    ]
    assert plan.spec is context.spec_load.subject
    assert "source.openenv_manifest" not in {
        planned.criterion_id for planned in plan.checks
    }


def test_openenv_policy_rejects_an_incompatible_execution_model(
    tmp_path: Path,
) -> None:
    (tmp_path / "task.oneshot").write_text("signature")

    with pytest.raises(ValueError, match="does not support"):
        build_validation_plan(
            tmp_path,
            spec_registry=ValidationSpecRegistry((_OneShotAdapter(),)),
        )


def test_openenv_policy_rejects_an_invalid_incompatible_spec(
    tmp_path: Path,
) -> None:
    class _InvalidOneShotAdapter(_OneShotAdapter):
        def inspect(self, root: Path) -> SpecLoad:
            return SpecLoad(
                state=SpecLoadState.INVALID,
                identity=SpecIdentity(
                    spec_id=self.spec_id,
                    spec_version=None,
                    adapter=AdapterIdentity(self.adapter_id, self.adapter_version),
                    execution_model=self.execution_model,
                ),
                error="malformed one-shot task",
            )

    (tmp_path / "task.oneshot").write_text("signature")

    with pytest.raises(ValueError, match="does not support"):
        build_validation_plan(
            tmp_path,
            spec_registry=ValidationSpecRegistry((_InvalidOneShotAdapter(),)),
        )


def test_openenv_policy_rejects_an_unresolved_custom_registry(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="matching validation policy"):
        build_validation_plan(
            tmp_path,
            spec_registry=ValidationSpecRegistry((_OneShotAdapter(),)),
        )


def test_injected_spec_load_cannot_claim_canonical_policy(tmp_path: Path) -> None:
    write_valid_env(tmp_path)
    loaded = DEFAULT_SPEC_REGISTRY.resolve(tmp_path)

    plan, context = build_validation_plan(tmp_path, spec_load=loaded)

    assert is_canonical_policy_plan(plan, context) is False


def test_policy_adaptation_uses_normalized_requirements_for_any_spec(
    tmp_path: Path,
) -> None:
    (tmp_path / "task.oneshot").write_text("signature")
    (tmp_path / "verify.sh").write_text("#!/bin/sh\n")
    checks = (
        ValidationCheck(
            criterion_id="spec.one_shot.verifier",
            requirement="Run the declared verifier",
            capabilities=frozenset(),
            severity=ValidationSeverity.BLOCKING,
            timeout_s=600.0,
            evaluator=lambda _context: CheckOutcome.pass_(),
            requirement_binding=ValidationRequirementBinding.VERIFIER,
        ),
        ValidationCheck(
            criterion_id="spec.one_shot.egress",
            requirement="Enforce network requirements",
            capabilities=frozenset(),
            severity=ValidationSeverity.ADVISORY,
            timeout_s=10.0,
            evaluator=lambda _context: CheckOutcome.pass_(),
            requirement_binding=ValidationRequirementBinding.NETWORK,
        ),
    )
    policy = ValidationPolicy(
        version="one-shot-v1",
        supported_subjects=frozenset({("one-shot-test", ExecutionModel.ONE_SHOT)}),
        checks=checks,
    )

    plan, _context = build_validation_plan(
        tmp_path,
        profile=ValidationProfile.FULL,
        spec_registry=ValidationSpecRegistry((_OneShotAdapter(),)),
        policy=policy,
    )

    planned = {check.criterion_id: check for check in plan.checks}
    assert planned["spec.one_shot.verifier"].timeout_s == 17.0
    assert planned["spec.one_shot.egress"].severity is ValidationSeverity.BLOCKING


def test_canonical_plan_revalidates_bound_verifier_file(tmp_path: Path) -> None:
    environment = write_harbor_task(tmp_path / "harbor-task")
    plan, context = build_validation_plan(environment)
    assert plan.spec is not None
    assert plan.spec.verifier_script is not None
    outside = tmp_path / "outside.sh"
    outside.write_text("#!/bin/sh\nexit 0\n")
    forged_subject = replace(plan.spec, verifier_script=outside)
    forged_load = replace(context.spec_load, subject=forged_subject)
    context.spec_load = forged_load
    forged_plan = replace(plan, spec=forged_subject)

    assert is_canonical_policy_plan(forged_plan, context) is False
