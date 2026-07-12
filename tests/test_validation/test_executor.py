# SPDX-License-Identifier: BSD-3-Clause

"""Tests for validation plan execution and report safety."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from pathlib import Path

from openenv.validation.executor import execute_validation_plan
from openenv.validation.models import (
    CheckOutcome,
    RunnerCapabilities,
    ValidationCapability,
    ValidationCheck,
    ValidationContext,
    ValidationPlan,
    ValidationProfile,
    ValidationSeverity,
    ValidationStatus,
)
from openenv.validation.planner import build_validation_plan
from openenv.validation.specs import (
    AdapterIdentity,
    DetectionMode,
    ExecutionModel,
    RequirementsLoad,
    RequirementsState,
    SpecIdentity,
    SpecLoad,
    SpecLoadState,
    ValidationSubject,
)

from ._helpers import write_valid_env


_RUNTIME_CRITERIA = (
    "openapi_version_available",
    "health_endpoint",
    "metadata_endpoint",
    "schema_endpoint",
    "mcp_endpoint",
    "mode_endpoint_consistency",
)


def _passing_runtime_discovery() -> dict[str, object]:
    discovery: dict[str, object] = {
        "runtime_report": {
            "criteria": [
                {"id": criterion_id, "passed": True}
                for criterion_id in _RUNTIME_CRITERIA
            ]
        },
        "runtime_declarations": {
            name: {"valid": True}
            for name in (
                "websocket",
                "tools",
                "tasks",
                "rewards",
                "seeds",
                "trajectories",
            )
        },
    }
    declarations = discovery["runtime_declarations"]
    assert isinstance(declarations, dict)
    declarations["tools"] = {"valid": True, "count": 1, "names": ["echo"]}
    return discovery


def _absent_spec() -> SpecLoad:
    return SpecLoad(state=SpecLoadState.ABSENT)


def _blocking_capability_skip_report(tmp_path: Path, profile: ValidationProfile):
    check = ValidationCheck(
        criterion_id="test.publish-readiness",
        requirement="Publish readiness requires source inspection",
        capabilities=frozenset({ValidationCapability.SOURCE}),
        severity=ValidationSeverity.BLOCKING,
        timeout_s=1.0,
        evaluator=lambda _context: CheckOutcome.pass_(),
    )
    plan = ValidationPlan(
        target=str(tmp_path),
        profile=profile,
        policy_version="test-policy",
        capabilities=RunnerCapabilities(runner="local"),
        checks=(check,),
    )
    context = ValidationContext(target=tmp_path, spec_load=_absent_spec())
    return execute_validation_plan(plan, context)


def test_publish_profile_makes_a_blocking_skip_fail_readiness(
    tmp_path: Path,
) -> None:
    report = _blocking_capability_skip_report(
        tmp_path,
        ValidationProfile.PUBLISH,
    )

    assert report.results[0].status is ValidationStatus.SKIP
    assert report.passed is False


def test_existing_profiles_keep_blocking_skips_nonfatal(tmp_path: Path) -> None:
    for profile in (ValidationProfile.STATIC, ValidationProfile.RUNTIME):
        report = _blocking_capability_skip_report(tmp_path, profile)

        assert report.results[0].status is ValidationStatus.SKIP
        assert report.passed is True


def test_echo_env_local_and_remote_executors_share_criterion_results() -> None:
    env_dir = Path(__file__).resolve().parents[2] / "envs" / "echo_env"
    local_plan, local_context = build_validation_plan(
        env_dir,
        capabilities=RunnerCapabilities(
            runner="local",
            available=frozenset({ValidationCapability.SOURCE}),
        ),
    )
    remote_plan, remote_context = build_validation_plan(
        env_dir,
        capabilities=RunnerCapabilities(
            runner="remote",
            available=frozenset({ValidationCapability.SOURCE}),
        ),
    )

    local_results = execute_validation_plan(local_plan, local_context).results
    remote_results = execute_validation_plan(remote_plan, remote_context).results

    assert [result.criterion_id for result in local_results] == [
        result.criterion_id for result in remote_results
    ]
    assert [result.status for result in local_results] == [
        result.status for result in remote_results
    ]
    assert [result.evidence for result in local_results] == [
        result.evidence for result in remote_results
    ]


def test_executor_marks_unavailable_remote_capabilities_as_skipped(
    tmp_path: Path,
) -> None:
    env_dir = tmp_path / "test_env"
    write_valid_env(env_dir)
    capabilities = RunnerCapabilities(
        runner="local",
        available=frozenset({ValidationCapability.SOURCE}),
    )
    plan, context = build_validation_plan(
        env_dir,
        profile=ValidationProfile.FULL,
        capabilities=capabilities,
    )

    report = execute_validation_plan(plan, context)

    by_id = {result.criterion_id: result for result in report.results}
    assert by_id["remote.cross_host_reproducibility"].status.value == "skip"
    assert by_id["remote.egress_enforcement"].status.value == "skip"
    assert by_id["artifact.image_identity"].status.value == "skip"
    assert report.certified is False


def test_custom_check_cannot_suppress_builtin_failure(tmp_path: Path) -> None:
    capabilities = RunnerCapabilities(
        runner="test",
        available=frozenset({ValidationCapability.SOURCE}),
    )
    built_in = ValidationCheck(
        criterion_id="builtin.failure",
        requirement="RFC 008 built-in requirement",
        capabilities=frozenset({ValidationCapability.SOURCE}),
        severity=ValidationSeverity.BLOCKING,
        timeout_s=1.0,
        evaluator=lambda _context: CheckOutcome.fail({"reason": "broken"}),
    )
    custom = ValidationCheck(
        criterion_id="custom.verifier",
        requirement="Environment-specific verifier",
        capabilities=frozenset({ValidationCapability.SOURCE}),
        severity=ValidationSeverity.BLOCKING,
        timeout_s=1.0,
        evaluator=lambda _context: CheckOutcome.pass_({"exit_code": 0}),
        built_in=False,
    )
    plan = ValidationPlan(
        target=str(tmp_path),
        profile=ValidationProfile.RUNTIME,
        policy_version="test-policy",
        capabilities=capabilities,
        checks=(built_in, custom),
    )
    context = ValidationContext(
        target=tmp_path,
        spec_load=_absent_spec(),
    )

    report = execute_validation_plan(plan, context)

    assert report.passed is False
    assert [result.status.value for result in report.results] == ["fail", "pass"]


def test_check_timeout_returns_without_waiting_for_timed_out_work(
    tmp_path: Path,
) -> None:
    check = ValidationCheck(
        criterion_id="test.timeout",
        requirement="Timeout contract",
        capabilities=frozenset(),
        severity=ValidationSeverity.BLOCKING,
        timeout_s=0.05,
        evaluator=lambda _context: (time.sleep(1.0), CheckOutcome.pass_())[1],
    )
    plan = ValidationPlan(
        target=str(tmp_path),
        profile=ValidationProfile.STATIC,
        policy_version="test",
        capabilities=RunnerCapabilities(runner="test"),
        checks=(check,),
    )
    context = ValidationContext(target=tmp_path, spec_load=_absent_spec())

    started = time.perf_counter()
    report = execute_validation_plan(plan, context)

    assert time.perf_counter() - started < 0.5
    assert report.results[0].status.value == "error"
    assert report.results[0].evidence["reason"] == "timeout"


def test_empty_plan_can_never_be_certified(tmp_path: Path) -> None:
    plan = ValidationPlan(
        target=str(tmp_path),
        profile=ValidationProfile.FULL,
        policy_version="rfc008-v1",
        capabilities=RunnerCapabilities(
            runner="remote",
            official=True,
            isolation_mode="dedicated",
        ),
        checks=(),
    )
    context = ValidationContext(
        target=tmp_path,
        spec_load=_absent_spec(),
        repo_sha="a" * 40,
        image_digest=f"sha256:{'b' * 64}",
        discovered={
            "runner_attestation_verified": True,
            "signed_report_verified": True,
            "provenance_binding_verified": True,
        },
    )

    report = execute_validation_plan(plan, context)

    assert report.passed is True
    assert report.certification_eligible is False
    assert report.certified is False


def test_modified_canonical_plan_can_never_be_certified(tmp_path: Path) -> None:
    env_dir = tmp_path / "test_env"
    write_valid_env(env_dir)
    canonical, context = build_validation_plan(
        env_dir,
        profile=ValidationProfile.FULL,
    )
    fake_checks = tuple(
        replace(check, evaluator=lambda _context: CheckOutcome.pass_())
        for check in canonical.checks
    )
    spoofed = replace(
        canonical,
        checks=fake_checks,
        capabilities=RunnerCapabilities(
            runner="remote",
            available=frozenset(ValidationCapability),
            official=True,
            isolation_mode="dedicated",
        ),
    )
    context.repo_sha = "a" * 40
    context.image_digest = f"sha256:{'b' * 64}"
    context.discovered.update(
        {
            "runner_attestation_verified": True,
            "signed_report_verified": True,
            "provenance_binding_verified": True,
        }
    )

    report = execute_validation_plan(spoofed, context)

    assert report.passed is True
    assert report.certification_eligible is False
    assert report.certified is False


def test_executor_only_produces_signature_ready_eligible_report(
    tmp_path: Path,
) -> None:
    env_dir = tmp_path / "test_env"
    write_valid_env(env_dir)
    (env_dir / "task.toml").write_text('schema_version = "1.1"\n')
    discovered = _passing_runtime_discovery()
    for criterion_id in (
        "remote.egress_enforcement",
        "remote.host_containment",
        "artifact.image_identity",
        "artifact.sbom",
        "artifact.signature",
        "remote.cross_host_reproducibility",
        "remote.reference_model",
    ):
        discovered[criterion_id] = {"status": "pass", "evidence": {}}
    discovered.update(
        {
            "runner_attestation_verified": True,
            # These pre-execution flags must never self-certify the payload.
            "signed_report_verified": True,
            "provenance_binding_verified": True,
        }
    )
    capabilities = RunnerCapabilities(
        runner="remote",
        available=frozenset(ValidationCapability),
        official=True,
        isolation_mode="dedicated",
    )
    plan, context = build_validation_plan(
        env_dir,
        profile=ValidationProfile.FULL,
        capabilities=capabilities,
        discovered=discovered,
        repo_sha="a" * 40,
        image_digest=f"sha256:{'b' * 64}",
    )

    report = execute_validation_plan(plan, context)

    assert report.passed is True
    assert report.certification_eligible is True
    assert report.certified is False
    payload = report.to_dict()
    assert payload["spec"]["id"] == "openenv"
    assert payload["spec"]["adapter"] == {
        "id": "openenv-yaml",
        "version": "1",
    }
    assert payload["spec"]["execution_model"] == "served"
    assert payload["spec"]["document_digest"].startswith("sha256:")

    assert plan.spec is not None
    spoofed_identity = replace(
        plan.spec.spec,
        adapter=replace(plan.spec.spec.adapter, adapter_version="untrusted"),
    )
    spoofed_subject = replace(plan.spec, spec=spoofed_identity)
    original_spec_load = context.spec_load
    context.spec_load = SpecLoad(
        state=SpecLoadState.LOADED,
        subject=spoofed_subject,
    )
    provenance_spoof = execute_validation_plan(
        replace(
            plan,
            spec=spoofed_subject,
            spec_identity=spoofed_identity,
        ),
        context,
    )
    assert provenance_spoof.passed is True
    assert provenance_spoof.certification_eligible is False
    context.spec_load = original_spec_load

    context.discovered.pop("provenance_binding_verified")
    unbound = execute_validation_plan(plan, context)
    assert unbound.passed is True
    assert unbound.certification_eligible is False
    context.discovered["provenance_binding_verified"] = True

    for invalid_sha, invalid_digest in (
        ("a" * 41, f"sha256:{'b' * 64}"),
        ("a" * 40, f"sha256:{'B' * 64}"),
    ):
        invalid_plan, invalid_context = build_validation_plan(
            env_dir,
            profile=ValidationProfile.FULL,
            capabilities=capabilities,
            discovered=discovered,
            repo_sha=invalid_sha,
            image_digest=invalid_digest,
        )
        invalid_provenance = execute_validation_plan(invalid_plan, invalid_context)
        assert invalid_provenance.passed is True
        assert invalid_provenance.certification_eligible is False

    context.discovered["remote.host_containment"] = CheckOutcome.fail(
        {"contained": False},
        severity=ValidationSeverity.ADVISORY,
    )
    downgrade_attempt = execute_validation_plan(plan, context)
    containment = next(
        result
        for result in downgrade_attempt.results
        if result.criterion_id == "remote.host_containment"
    )
    assert containment.severity is ValidationSeverity.BLOCKING
    assert downgrade_attempt.passed is False
    assert downgrade_attempt.certification_eligible is False

    context.discovered["remote.host_containment"] = {
        "status": "pass",
        "evidence": {},
    }
    runtime_declarations = context.discovered["runtime_declarations"]
    assert isinstance(runtime_declarations, dict)
    runtime_declarations["websocket"] = None
    malformed_runtime = execute_validation_plan(plan, context)
    websocket = next(
        result
        for result in malformed_runtime.results
        if result.criterion_id == "runtime.websocket"
    )
    assert websocket.status.value == "error"
    assert websocket.severity is ValidationSeverity.BLOCKING
    assert malformed_runtime.certification_eligible is False

    runtime_declarations["websocket"] = {"valid": True}
    runtime_declarations["tools"] = {
        "valid": True,
        "count": 2,
        "names": ["echo", "reset"],
    }
    exposed_control = execute_validation_plan(plan, context)
    boundary = next(
        result
        for result in exposed_control.results
        if result.criterion_id == "runtime.mcp_control_boundary"
    )
    assert boundary.status.value == "fail"
    assert boundary.severity is ValidationSeverity.BLOCKING
    assert exposed_control.certification_eligible is False


def test_canonical_plan_attestation_binds_target(tmp_path: Path) -> None:
    env_dir = tmp_path / "test_env"
    write_valid_env(env_dir)
    plan, context = build_validation_plan(
        env_dir,
        profile=ValidationProfile.FULL,
    )
    mismatched = replace(plan, target="other-org/other-space")

    report = execute_validation_plan(mismatched, context)

    assert report.target == "other-org/other-space"
    assert report.certification_eligible is False


def test_missing_websocket_evidence_remains_blocking(tmp_path: Path) -> None:
    env_dir = tmp_path / "test_env"
    write_valid_env(env_dir)
    discovered = _passing_runtime_discovery()
    runtime_declarations = discovered["runtime_declarations"]
    assert isinstance(runtime_declarations, dict)
    runtime_declarations.pop("websocket")
    plan, context = build_validation_plan(
        env_dir,
        profile=ValidationProfile.RUNTIME,
        capabilities=RunnerCapabilities(
            runner="local",
            available=frozenset(
                {ValidationCapability.SOURCE, ValidationCapability.RUNTIME}
            ),
        ),
        discovered=discovered,
    )

    report = execute_validation_plan(plan, context)
    websocket = next(
        result
        for result in report.results
        if result.criterion_id == "runtime.websocket"
    )

    assert websocket.status.value == "error"
    assert websocket.severity is ValidationSeverity.BLOCKING
    assert report.passed is False


def test_worker_thread_rejects_check_before_unenforceable_timeout(
    tmp_path: Path,
) -> None:
    evaluator_called = threading.Event()

    def evaluator(_context: ValidationContext) -> CheckOutcome:
        evaluator_called.set()
        time.sleep(1.0)
        return CheckOutcome.pass_()

    check = ValidationCheck(
        criterion_id="test.worker-timeout",
        requirement="Timeout contract",
        capabilities=frozenset(),
        severity=ValidationSeverity.BLOCKING,
        timeout_s=0.05,
        evaluator=evaluator,
    )
    plan = ValidationPlan(
        target=str(tmp_path),
        profile=ValidationProfile.STATIC,
        policy_version="test",
        capabilities=RunnerCapabilities(runner="test"),
        checks=(check,),
    )
    context = ValidationContext(target=tmp_path, spec_load=_absent_spec())
    reports = []
    worker = threading.Thread(
        target=lambda: reports.append(execute_validation_plan(plan, context))
    )

    worker.start()
    worker.join(timeout=0.5)

    assert worker.is_alive() is False
    assert evaluator_called.is_set() is False
    assert reports[0].results[0].status.value == "error"
    assert reports[0].results[0].evidence["reason"] == "timeout_context_unsupported"


def test_report_redacts_credential_evidence(tmp_path: Path) -> None:
    check = ValidationCheck(
        criterion_id="test.redaction",
        requirement="Secret hygiene",
        capabilities=frozenset(),
        severity=ValidationSeverity.BLOCKING,
        timeout_s=1.0,
        evaluator=lambda _context: CheckOutcome.pass_(
            {
                "HF_TOKEN": "hf_1234567890abcdef",
                "url": "https://user:password@example.com/path?token=secret",
                "embedded": (
                    "request failed at "
                    "https://alice:hunter2@example.com/path?api_key=secret"
                ),
                "database": "postgresql://alice:hunter2@db/app",
                "cache": "redis://:hunter2@cache/0",
                "signed": ("https://example.com/object?auth=one&key=two&sig=three"),
                "github": "ghp_1234567890abcdef",
                "aws": "AKIA1234567890ABCDEF",
                "aws_temporary": "ASIA1234567890ABCDEF",
                "pem": (
                    "-----BEGIN PRIVATE KEY-----\nprivate\n-----END PRIVATE KEY-----"
                ),
                "truncated_pem": (
                    "-----BEGIN PRIVATE KEY-----\nTRUNCATED_PRIVATE_MATERIAL"
                ),
            }
        ),
    )
    plan = ValidationPlan(
        target=str(tmp_path),
        profile=ValidationProfile.STATIC,
        policy_version="test",
        capabilities=RunnerCapabilities(runner="test"),
        checks=(check,),
    )
    context = ValidationContext(target=tmp_path, spec_load=_absent_spec())

    serialized = json.dumps(execute_validation_plan(plan, context).to_dict())

    assert "hf_1234567890abcdef" not in serialized
    assert "user:password" not in serialized
    assert "alice:hunter2" not in serialized
    assert "token=secret" not in serialized
    assert "auth=one" not in serialized
    assert "key=two" not in serialized
    assert "sig=three" not in serialized
    assert "ghp_1234567890abcdef" not in serialized
    assert "AKIA1234567890ABCDEF" not in serialized
    assert "ASIA1234567890ABCDEF" not in serialized
    assert "private\\n" not in serialized
    assert "TRUNCATED_PRIVATE_MATERIAL" not in serialized
    assert "[REDACTED]" in serialized


def test_report_preserves_valid_token_like_structural_ids(tmp_path: Path) -> None:
    identity = SpecIdentity(
        spec_id="hf_environment",
        spec_version="1",
        adapter=AdapterIdentity("hf_environment_adapter", "hf_adapter_version"),
        execution_model=ExecutionModel.ONE_SHOT,
    )
    subject = ValidationSubject(
        spec=identity,
        signature_path="task.fake",
        detection_mode=DetectionMode.EXPLICIT,
        requirements=RequirementsLoad(state=RequirementsState.ABSENT),
    )
    check = ValidationCheck(
        criterion_id="hf_criterion_identifier",
        requirement="hf_requirement_identifier",
        capabilities=frozenset(),
        severity=ValidationSeverity.BLOCKING,
        timeout_s=1.0,
        evaluator=lambda _context: CheckOutcome.fail(),
    )
    plan = ValidationPlan(
        target=str(tmp_path),
        profile=ValidationProfile.STATIC,
        policy_version="hf_policy_identifier",
        capabilities=RunnerCapabilities(runner="hf_runner_identifier"),
        checks=(check,),
        spec=subject,
        spec_identity=identity,
    )
    context = ValidationContext(
        target=tmp_path,
        spec_load=SpecLoad(state=SpecLoadState.LOADED, subject=subject),
    )

    plan_payload = plan.to_dict()
    report_payload = execute_validation_plan(plan, context).to_dict()

    assert plan_payload["policy_version"] == "hf_policy_identifier"
    assert plan_payload["spec"]["id"] == "hf_environment"
    assert plan_payload["spec"]["adapter"] == {
        "id": "hf_environment_adapter",
        "version": "hf_adapter_version",
    }
    assert plan_payload["checks"][0]["id"] == "hf_criterion_identifier"
    assert report_payload["runner"]["kind"] == "hf_runner_identifier"
    assert report_payload["criteria"][0]["id"] == "hf_criterion_identifier"
    assert report_payload["summary"]["failed_criteria"] == ["hf_criterion_identifier"]


def test_report_redacts_top_level_spec_provenance(tmp_path: Path) -> None:
    env_dir = tmp_path / "test_env"
    write_valid_env(env_dir)
    (env_dir / "task.toml").write_text(
        'schema_version = "https://alice:hunter2@example.com/v"\n'
    )
    plan, context = build_validation_plan(env_dir)

    serialized = json.dumps(execute_validation_plan(plan, context).to_dict())
    plan_serialized = json.dumps(plan.to_dict())

    assert "alice:hunter2" not in serialized
    assert "alice:hunter2" not in plan_serialized
    assert "https://example.com/v" in serialized


def test_report_rejects_excessively_nested_evidence(tmp_path: Path) -> None:
    nested: dict[str, object] = {}
    cursor = nested
    for _ in range(100):
        child: dict[str, object] = {}
        cursor["child"] = child
        cursor = child
    check = ValidationCheck(
        criterion_id="test.deep-evidence",
        requirement="Bounded evidence",
        capabilities=frozenset(),
        severity=ValidationSeverity.BLOCKING,
        timeout_s=1.0,
        evaluator=lambda _context: CheckOutcome.pass_({"nested": nested}),
    )
    plan = ValidationPlan(
        target=str(tmp_path),
        profile=ValidationProfile.STATIC,
        policy_version="test",
        capabilities=RunnerCapabilities(runner="test"),
        checks=(check,),
    )
    context = ValidationContext(target=tmp_path, spec_load=_absent_spec())

    result = execute_validation_plan(plan, context).results[0]

    assert result.status.value == "error"
    assert result.evidence["reason"] == "invalid_evidence"


def test_report_handles_invalid_url_redaction_and_message_shape(
    tmp_path: Path,
) -> None:
    invalid_url = "http://example.invalid:99999"
    check = ValidationCheck(
        criterion_id="test.invalid-message",
        requirement="Safe messages",
        capabilities=frozenset(),
        severity=ValidationSeverity.BLOCKING,
        timeout_s=1.0,
        evaluator=lambda _context: CheckOutcome.pass_(message=invalid_url),
    )
    plan = ValidationPlan(
        target=invalid_url,
        profile=ValidationProfile.STATIC,
        policy_version="test",
        capabilities=RunnerCapabilities(runner="test"),
        checks=(check,),
    )
    context = ValidationContext(target=tmp_path, spec_load=_absent_spec())

    report = execute_validation_plan(plan, context)

    assert report.target == "[REDACTED_INVALID_URL]"
    assert report.results[0].message == "[REDACTED_INVALID_URL]"

    malformed = replace(
        check,
        evaluator=lambda _context: CheckOutcome(
            status=ValidationStatus.PASS,
            evidence={},
            message=42,  # type: ignore[arg-type]
        ),
    )
    malformed_report = execute_validation_plan(
        replace(plan, checks=(malformed,)), context
    )
    assert malformed_report.results[0].status.value == "error"
    assert malformed_report.results[0].evidence["reason"] == "invalid_outcome_shape"
