# SPDX-License-Identifier: BSD-3-Clause

"""Build versioned validation plans from subjects and runner capabilities."""

from __future__ import annotations

import hashlib
import hmac
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from .checks import get_openenv_checks
from .models import (
    RunnerCapabilities,
    ValidationCapability,
    ValidationCheck,
    ValidationContext,
    ValidationPlan,
    ValidationPolicy,
    ValidationProfile,
    ValidationRequirementBinding,
    ValidationSeverity,
)
from .specs import (
    DEFAULT_SPEC_REGISTRY,
    ExecutionModel,
    NetworkMode,
    SpecLoad,
    SpecLoadState,
    ValidationSpecRegistry,
)


VALIDATION_POLICY_VERSION = "rfc008-v1"
OPENENV_VALIDATION_POLICY = ValidationPolicy(
    version=VALIDATION_POLICY_VERSION,
    supported_subjects=frozenset({("openenv", ExecutionModel.SERVED)}),
    checks=get_openenv_checks(),
)
_POLICY_ATTESTATION = object()


def _repo_sha(path: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    sha = completed.stdout.strip()
    return sha or None


def _profile(value: ValidationProfile | str) -> ValidationProfile:
    if isinstance(value, ValidationProfile):
        return value
    return ValidationProfile(value)


def _adapt_policy(
    checks: Iterable[ValidationCheck], spec_load: SpecLoad
) -> tuple[ValidationCheck, ...]:
    subject = spec_load.subject
    requirements = subject.requirements.requirements if subject is not None else None
    adapted: list[ValidationCheck] = []
    for check in checks:
        replacement = check
        if (
            check.requirement_binding is ValidationRequirementBinding.VERIFIER
            and requirements is not None
        ):
            replacement = replace(
                check, timeout_s=requirements.verifier.timeout_s or 600.0
            )
        if check.requirement_binding is ValidationRequirementBinding.NETWORK:
            network_mode = (
                requirements.environment.network.mode
                if requirements is not None
                else NetworkMode.UNSPECIFIED
            )
            replacement = replace(
                replacement,
                severity=(
                    ValidationSeverity.BLOCKING
                    if network_mode in {NetworkMode.DENY_ALL, NetworkMode.ALLOWLIST}
                    else ValidationSeverity.ADVISORY
                ),
            )
        adapted.append(replacement)
    return tuple(adapted)


def _select_policy_checks(
    checks: Iterable[ValidationCheck],
    *,
    profile: ValidationProfile,
    spec_load: SpecLoad,
) -> tuple[ValidationCheck, ...]:
    subject = spec_load.subject
    selected = tuple(check for check in checks if profile in check.profiles)
    selected = tuple(
        check
        for check in selected
        if check.requirement_binding is not ValidationRequirementBinding.VERIFIER
        or (subject is not None and subject.verifier_script is not None)
    )
    return _adapt_policy(selected, spec_load)


def _policy_requirements(spec_load: SpecLoad) -> dict[str, Any]:
    subject = spec_load.subject
    if subject is None:
        return {
            "state": "unavailable",
            "policy_defaults": True,
        }
    evidence = subject.requirements.to_evidence()
    evidence["policy_defaults"] = subject.requirements.requirements is None
    return evidence


def _canonical_fingerprint(plan: ValidationPlan, context: ValidationContext) -> str:
    payload = plan.to_dict()
    payload["repo_sha"] = context.repo_sha
    payload["image_digest"] = context.image_digest
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _policy_for_subject(
    policy: ValidationPolicy | None, spec_load: SpecLoad
) -> tuple[ValidationPolicy, bool]:
    selected = policy or OPENENV_VALIDATION_POLICY
    identity = spec_load.spec
    if identity is not None and not selected.supports_identity(identity):
        raise ValueError(
            f"Validation policy {selected.version!r} does not support "
            f"spec {identity.spec_id!r} with execution model "
            f"{identity.execution_model.value!r}"
        )
    return selected, policy is None


def _matches_current_default_subject(context: ValidationContext, subject: Any) -> bool:
    if not isinstance(context.target, Path):
        return False
    try:
        current = DEFAULT_SPEC_REGISTRY.resolve(context.target)
    except Exception:
        return False
    return bool(current.state is SpecLoadState.LOADED and current.subject == subject)


def is_canonical_policy_plan(plan: ValidationPlan, context: ValidationContext) -> bool:
    """Return whether a plan is the unmodified trusted OpenEnv policy output."""
    subject = context.spec_load.subject
    if (
        plan._policy_attestation is not _POLICY_ATTESTATION
        or plan._policy_fingerprint is None
        or subject is None
        or not OPENENV_VALIDATION_POLICY.supports(subject)
        or not _matches_current_default_subject(context, subject)
    ):
        return False
    expected_checks = _select_policy_checks(
        OPENENV_VALIDATION_POLICY.checks,
        profile=plan.profile,
        spec_load=context.spec_load,
    )
    structure_matches = bool(
        plan.policy_version == OPENENV_VALIDATION_POLICY.version
        and plan.target == str(context.target)
        and plan.spec == subject
        and plan.spec_identity == subject.spec
        and plan.checks == expected_checks
        and dict(plan.requirements) == _policy_requirements(context.spec_load)
    )
    return bool(
        structure_matches
        and hmac.compare_digest(
            plan._policy_fingerprint, _canonical_fingerprint(plan, context)
        )
    )


def build_validation_plan(
    target: str | Path,
    *,
    profile: ValidationProfile | str = ValidationProfile.STATIC,
    capabilities: RunnerCapabilities | None = None,
    spec_load: SpecLoad | None = None,
    spec_id: str | None = None,
    spec_registry: ValidationSpecRegistry | None = None,
    policy: ValidationPolicy | None = None,
    runtime_url: str | None = None,
    discovered: dict[str, Any] | None = None,
    repo_sha: str | None = None,
    image_digest: str | None = None,
) -> tuple[ValidationPlan, ValidationContext]:
    """Build a plan without importing or executing submitted code."""
    selected_profile = _profile(profile)
    target_path = target if isinstance(target, Path) else Path(target)
    is_source = target_path.exists() and target_path.is_dir()

    if spec_load is not None and (spec_id is not None or spec_registry is not None):
        raise ValueError("spec_load cannot be combined with spec selection options")
    resolved_by_default_registry = spec_load is None and spec_registry is None
    registry = spec_registry or DEFAULT_SPEC_REGISTRY
    if spec_load is None:
        spec_load = (
            registry.resolve(target_path, spec_id=spec_id)
            if is_source
            else SpecLoad(state=SpecLoadState.ABSENT)
        )

    if policy is None and spec_registry is not None and spec_load.spec is None:
        raise ValueError(
            "The OpenEnv validation policy cannot be used for an unresolved "
            "custom spec registry; provide the matching validation policy"
        )

    selected_policy, is_default_policy = _policy_for_subject(policy, spec_load)
    if capabilities is None:
        available: set[ValidationCapability] = set()
        if is_source:
            available.add(ValidationCapability.SOURCE)
        if runtime_url is not None:
            available.add(ValidationCapability.RUNTIME)
        capabilities = RunnerCapabilities(
            runner="local", available=frozenset(available), official=False
        )

    checks = _select_policy_checks(
        selected_policy.checks,
        profile=selected_profile,
        spec_load=spec_load,
    )
    subject = spec_load.subject
    canonical_subject = bool(
        subject is not None and OPENENV_VALIDATION_POLICY.supports(subject)
    )
    uses_canonical_policy = bool(
        is_default_policy
        and resolved_by_default_registry
        and spec_id in {None, "openenv"}
        and canonical_subject
    )

    resolved_repo_sha = repo_sha
    if resolved_repo_sha is None and is_source:
        resolved_repo_sha = _repo_sha(target_path)
    context = ValidationContext(
        target=target_path if is_source else str(target),
        spec_load=spec_load,
        runtime_url=runtime_url,
        discovered=dict(discovered or {}),
        repo_sha=resolved_repo_sha,
        image_digest=image_digest,
    )
    plan = ValidationPlan(
        target=str(target),
        profile=selected_profile,
        policy_version=selected_policy.version,
        capabilities=capabilities,
        checks=checks,
        spec=subject,
        spec_identity=spec_load.spec,
        requirements=_policy_requirements(spec_load),
        _policy_attestation=(_POLICY_ATTESTATION if uses_canonical_policy else None),
    )
    if uses_canonical_policy:
        plan = replace(plan, _policy_fingerprint=_canonical_fingerprint(plan, context))
    return plan, context
