# SPDX-License-Identifier: BSD-3-Clause

"""Shared models for local validation and remote certification runners."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

from .serialization import json_safe, redact_string
from .specs.base import ExecutionModel, SpecIdentity, SpecLoad, ValidationSubject


_STABLE_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_TRUSTED_METADATA_STRING_KEYS = frozenset(
    {
        "id",
        "isolation_mode",
        "kind",
        "policy_version",
    }
)


def _restore_structural_spec_identity(
    serialized: dict[str, Any], raw: Mapping[str, Any]
) -> None:
    """Restore adapter-owned identity fields after free-text redaction."""
    for key in ("id", "adapter", "execution_model"):
        if key in raw:
            serialized[key] = raw[key]
    raw_requirements = raw.get("requirements")
    safe_requirements = serialized.get("requirements")
    if isinstance(raw_requirements, Mapping) and isinstance(safe_requirements, dict):
        for key in ("id", "adapter"):
            if key in raw_requirements:
                safe_requirements[key] = raw_requirements[key]


class ValidationStatus(str, Enum):
    """Outcome of one validation criterion."""

    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    ERROR = "error"


class ValidationSeverity(str, Enum):
    """Policy effect of a validation result."""

    BLOCKING = "blocking"
    ADVISORY = "advisory"


class ValidationProfile(str, Enum):
    """Set of validation checks requested by a caller."""

    STATIC = "static"
    RUNTIME = "runtime"
    FULL = "full"
    PUBLISH = "publish"


class ValidationCapability(str, Enum):
    """Execution capabilities that checks may require."""

    SOURCE = "source"
    RUNTIME = "runtime"
    CONTAINER_IMAGE = "container_image"
    NETWORK_ENFORCEMENT = "network_enforcement"
    GPU = "gpu"
    CROSS_HOST = "cross_host"
    VERIFIER_ISOLATION = "verifier_isolation"
    REFERENCE_MODEL = "reference_model"
    SIGNATURE_VERIFICATION = "signature_verification"


class ValidationRequirementBinding(str, Enum):
    """Normalized requirement that can adapt a policy criterion."""

    VERIFIER = "verifier"
    NETWORK = "network"


def _validate_relative_path(value: str, *, field_name: str) -> None:
    path = Path(value)
    if (
        not value
        or len(value) > 4096
        or "\x00" in value
        or path.is_absolute()
        or ".." in path.parts
    ):
        raise ValueError(f"{field_name} must be a bounded relative path")


@dataclass(frozen=True)
class DiagnosticLocation:
    """Repository-relative location for an author-facing diagnostic."""

    path: str
    pointer: str | None = None
    line: int | None = None
    column: int | None = None

    def __post_init__(self) -> None:
        _validate_relative_path(self.path, field_name="diagnostic path")
        if self.pointer is not None and (
            not self.pointer.startswith("/")
            or len(self.pointer) > 4096
            or "\x00" in self.pointer
        ):
            raise ValueError("diagnostic pointer must be a bounded document pointer")
        for name, value in (("line", self.line), ("column", self.column)):
            if value is not None and (not isinstance(value, int) or value < 1):
                raise ValueError(f"diagnostic {name} must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe source location."""
        return {
            "path": self.path,
            "pointer": self.pointer,
            "line": self.line,
            "column": self.column,
        }


@dataclass(frozen=True)
class ValidationDiagnostic:
    """Trusted, typed explanation of a validation problem."""

    code: str
    message: str
    location: DiagnosticLocation | None = None

    def __post_init__(self) -> None:
        if not _STABLE_IDENTIFIER.fullmatch(self.code):
            raise ValueError("diagnostic code must be a stable lowercase identifier")
        if not self.message.strip() or len(self.message) > 4096:
            raise ValueError("diagnostic message must be non-empty and bounded")

    def to_dict(self) -> dict[str, Any]:
        """Return the report representation with free text redacted."""
        return {
            "code": self.code,
            "message": redact_string(self.message),
            "location": self.location.to_dict() if self.location is not None else None,
        }


@dataclass(frozen=True)
class ValidationRemediation:
    """Display-only, policy-authored guidance for resolving a diagnostic."""

    kind: str
    message: str
    argv: tuple[str, ...] = ()
    cwd: str | None = None
    path: str | None = None
    pointer: str | None = None
    url: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"command", "documentation", "edit", "retry"}:
            raise ValueError("unsupported remediation kind")
        if not self.message.strip() or len(self.message) > 4096:
            raise ValueError("remediation message must be non-empty and bounded")
        if any(
            not isinstance(argument, str)
            or not argument
            or len(argument) > 4096
            or "\x00" in argument
            for argument in self.argv
        ):
            raise ValueError("remediation argv must contain bounded strings")
        if self.kind == "command" and not self.argv:
            raise ValueError("command remediation requires argv")
        if self.kind != "command" and self.argv:
            raise ValueError("only command remediation may declare argv")
        if self.cwd is not None:
            _validate_relative_path(self.cwd, field_name="remediation cwd")
        if self.path is not None:
            _validate_relative_path(self.path, field_name="remediation path")
        if self.pointer is not None and (
            not self.pointer.startswith("/")
            or len(self.pointer) > 4096
            or "\x00" in self.pointer
        ):
            raise ValueError("remediation pointer must be a bounded document pointer")
        if self.url is not None:
            parsed = urlsplit(self.url)
            if parsed.scheme != "https" or not parsed.hostname or parsed.username:
                raise ValueError(
                    "remediation URL must be an HTTPS URL without credentials"
                )

    def to_dict(self) -> dict[str, Any]:
        """Return structured guidance; argv is never interpreted as a shell string."""
        return {
            "kind": self.kind,
            "message": redact_string(self.message),
            "argv": [redact_string(argument) for argument in self.argv],
            "cwd": self.cwd,
            "path": self.path,
            "pointer": self.pointer,
            "url": redact_string(self.url) if self.url is not None else None,
        }


@dataclass(frozen=True)
class RunnerCapabilities:
    """Capabilities and trust properties exposed by a validation runner."""

    runner: str
    available: frozenset[ValidationCapability] = frozenset()
    official: bool = False
    isolation_mode: str | None = None

    def __post_init__(self) -> None:
        if not _STABLE_IDENTIFIER.fullmatch(self.runner):
            raise ValueError("runner kind must be a stable lowercase identifier")
        if self.isolation_mode is not None and not _STABLE_IDENTIFIER.fullmatch(
            self.isolation_mode
        ):
            raise ValueError("isolation mode must be a stable lowercase identifier")

    def supports(self, required: frozenset[ValidationCapability]) -> bool:
        """Return whether all requested capabilities are available."""
        return required.issubset(self.available)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe runner description."""
        return {
            "kind": self.runner,
            "capabilities": sorted(capability.value for capability in self.available),
            "official": self.official,
            "isolation_mode": self.isolation_mode,
        }


@dataclass(frozen=True)
class CheckOutcome:
    """Raw outcome returned by a validation check implementation."""

    status: ValidationStatus
    evidence: Mapping[str, Any] = field(default_factory=dict)
    message: str | None = None
    severity: ValidationSeverity | None = None
    diagnostics: tuple[ValidationDiagnostic, ...] = ()
    remediation: tuple[ValidationRemediation, ...] = ()

    @classmethod
    def pass_(
        cls,
        evidence: Mapping[str, Any] | None = None,
        *,
        message: str | None = None,
        severity: ValidationSeverity | None = None,
        diagnostics: tuple[ValidationDiagnostic, ...] = (),
        remediation: tuple[ValidationRemediation, ...] = (),
    ) -> "CheckOutcome":
        """Build a passing outcome."""
        return cls(
            ValidationStatus.PASS,
            evidence or {},
            message=message,
            severity=severity,
            diagnostics=diagnostics,
            remediation=remediation,
        )

    @classmethod
    def fail(
        cls,
        evidence: Mapping[str, Any] | None = None,
        *,
        message: str | None = None,
        severity: ValidationSeverity | None = None,
        diagnostics: tuple[ValidationDiagnostic, ...] = (),
        remediation: tuple[ValidationRemediation, ...] = (),
    ) -> "CheckOutcome":
        """Build a failing outcome."""
        return cls(
            ValidationStatus.FAIL,
            evidence or {},
            message=message,
            severity=severity,
            diagnostics=diagnostics,
            remediation=remediation,
        )

    @classmethod
    def skip(
        cls,
        evidence: Mapping[str, Any] | None = None,
        *,
        message: str | None = None,
        severity: ValidationSeverity | None = None,
        diagnostics: tuple[ValidationDiagnostic, ...] = (),
        remediation: tuple[ValidationRemediation, ...] = (),
    ) -> "CheckOutcome":
        """Build a skipped outcome."""
        return cls(
            ValidationStatus.SKIP,
            evidence or {},
            message=message,
            severity=severity,
            diagnostics=diagnostics,
            remediation=remediation,
        )

    @classmethod
    def error(
        cls,
        evidence: Mapping[str, Any] | None = None,
        *,
        message: str | None = None,
        severity: ValidationSeverity | None = None,
        diagnostics: tuple[ValidationDiagnostic, ...] = (),
        remediation: tuple[ValidationRemediation, ...] = (),
    ) -> "CheckOutcome":
        """Build an infrastructure-error outcome."""
        return cls(
            ValidationStatus.ERROR,
            evidence or {},
            message=message,
            severity=severity,
            diagnostics=diagnostics,
            remediation=remediation,
        )


CheckEvaluator = Callable[["ValidationContext"], CheckOutcome]


@dataclass(frozen=True)
class ValidationCheck:
    """Policy-independent implementation contract for one criterion."""

    criterion_id: str
    requirement: str
    capabilities: frozenset[ValidationCapability]
    severity: ValidationSeverity
    timeout_s: float
    evaluator: CheckEvaluator | None
    profiles: frozenset[ValidationProfile] = field(
        default_factory=lambda: frozenset(ValidationProfile)
    )
    built_in: bool = True
    requirement_binding: ValidationRequirementBinding | None = None

    def __post_init__(self) -> None:
        if not _STABLE_IDENTIFIER.fullmatch(self.criterion_id):
            raise ValueError("ValidationCheck criterion_id must be a stable identifier")
        if not self.requirement.strip():
            raise ValueError("ValidationCheck requirement cannot be empty")
        if self.timeout_s <= 0:
            raise ValueError("ValidationCheck timeout_s must be positive")

    @property
    def rfc_requirement(self) -> str:
        """Compatibility alias for the normative requirement."""
        return self.requirement

    @property
    def required_capabilities(self) -> frozenset[ValidationCapability]:
        """Compatibility alias for capabilities required by this check."""
        return self.capabilities

    @property
    def default_severity(self) -> ValidationSeverity:
        """Compatibility alias for the policy default severity."""
        return self.severity


@dataclass(frozen=True)
class ValidationPolicy:
    """Versioned check catalog applicable to explicit specs and lifecycles."""

    version: str
    supported_subjects: frozenset[tuple[str, ExecutionModel]]
    checks: tuple[ValidationCheck, ...]

    def __post_init__(self) -> None:
        if not _STABLE_IDENTIFIER.fullmatch(self.version):
            raise ValueError("ValidationPolicy version must be a stable identifier")
        if not self.supported_subjects or any(
            not _STABLE_IDENTIFIER.fullmatch(spec_id)
            or not isinstance(model, ExecutionModel)
            for spec_id, model in self.supported_subjects
        ):
            raise ValueError("ValidationPolicy must declare supported subject pairs")
        criterion_ids = [check.criterion_id for check in self.checks]
        if len(criterion_ids) != len(set(criterion_ids)):
            raise ValueError("ValidationPolicy contains duplicate criterion IDs")

    def supports(self, subject: ValidationSubject) -> bool:
        """Return whether this policy applies to a loaded subject."""
        return self.supports_identity(subject.spec)

    def supports_identity(self, identity: SpecIdentity) -> bool:
        """Return whether this policy applies to a detected spec identity."""
        return bool(
            (identity.spec_id, identity.execution_model) in self.supported_subjects
        )


@dataclass
class ValidationContext:
    """Inputs and discovery data shared by checks in one execution."""

    target: Path | str
    spec_load: SpecLoad
    runtime_url: str | None = None
    discovered: dict[str, Any] = field(default_factory=dict)
    repo_sha: str | None = None
    image_digest: str | None = None


@dataclass(frozen=True)
class ValidationPlan:
    """Deterministic set of checks selected for a target and runner."""

    target: str
    profile: ValidationProfile
    policy_version: str
    capabilities: RunnerCapabilities
    checks: tuple[ValidationCheck, ...]
    spec: ValidationSubject | None = None
    spec_identity: SpecIdentity | None = None
    requirements: Mapping[str, Any] = field(default_factory=dict)
    _policy_attestation: object | None = field(default=None, repr=False, compare=False)
    _policy_fingerprint: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not _STABLE_IDENTIFIER.fullmatch(self.policy_version):
            raise ValueError(
                "ValidationPlan policy version must be a stable identifier"
            )
        criterion_ids = [check.criterion_id for check in self.checks]
        if len(criterion_ids) != len(set(criterion_ids)):
            raise ValueError("ValidationPlan contains duplicate criterion IDs")
        if (
            self.spec is not None
            and self.spec_identity is not None
            and self.spec_identity != self.spec.spec
        ):
            raise ValueError("ValidationPlan spec identity must match its subject")

    def to_dict(self) -> dict[str, Any]:
        """Return the safe, execution-independent plan representation."""
        payload = {
            "target": self.target,
            "profile": self.profile.value,
            "policy_version": self.policy_version,
            "spec": (
                self.spec.to_dict()
                if self.spec is not None
                else (
                    self.spec_identity.to_dict()
                    if self.spec_identity is not None
                    else None
                )
            ),
            "runner": self.capabilities.to_dict(),
            "requirements": dict(self.requirements),
            "checks": [
                {
                    "id": check.criterion_id,
                    "requirement": check.requirement,
                    "required_capabilities": sorted(
                        capability.value for capability in check.capabilities
                    ),
                    "severity": check.severity.value,
                    "timeout_s": check.timeout_s,
                    "built_in": check.built_in,
                    "requirement_binding": (
                        check.requirement_binding.value
                        if check.requirement_binding is not None
                        else None
                    ),
                }
                for check in self.checks
            ],
        }
        serialized = json_safe(
            payload,
            trusted_string_keys=_TRUSTED_METADATA_STRING_KEYS,
        )
        assert isinstance(serialized, dict)
        if isinstance(payload["spec"], Mapping) and isinstance(
            serialized.get("spec"), dict
        ):
            _restore_structural_spec_identity(serialized["spec"], payload["spec"])
        serialized_checks = serialized.get("checks")
        if isinstance(serialized_checks, list):
            for safe_check, raw_check in zip(
                serialized_checks, payload["checks"], strict=True
            ):
                if isinstance(safe_check, dict):
                    safe_check["requirement"] = raw_check["requirement"]
        return serialized


@dataclass(frozen=True)
class ValidationResult:
    """Structured result of executing one validation criterion."""

    criterion_id: str
    requirement: str
    status: ValidationStatus
    severity: ValidationSeverity
    evidence: Mapping[str, Any]
    duration_s: float
    timeout_s: float
    required_capabilities: frozenset[ValidationCapability]
    built_in: bool = True
    message: str | None = None
    diagnostics: tuple[ValidationDiagnostic, ...] = ()
    remediation: tuple[ValidationRemediation, ...] = ()

    @property
    def passed(self) -> bool:
        """Return whether this criterion explicitly passed."""
        return self.status is ValidationStatus.PASS

    def to_dict(self) -> dict[str, Any]:
        """Return the shared local/remote criterion schema."""
        payload: dict[str, Any] = {
            "id": self.criterion_id,
            "description": self.requirement,
            "requirement": self.requirement,
            "status": self.status.value,
            "severity": self.severity.value,
            "passed": self.passed,
            "required": self.severity is ValidationSeverity.BLOCKING,
            "built_in": self.built_in,
            "required_capabilities": sorted(
                capability.value for capability in self.required_capabilities
            ),
            "evidence": json_safe(self.evidence),
            "duration_s": round(self.duration_s, 6),
            "duration_ms": round(self.duration_s * 1000, 3),
            "timeout_s": self.timeout_s,
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "remediation": [item.to_dict() for item in self.remediation],
        }
        if self.message is not None:
            payload["details"] = redact_string(self.message)
        for compatibility_key in ("expected", "actual"):
            if compatibility_key in self.evidence:
                payload[compatibility_key] = self.evidence[compatibility_key]
        return payload


@dataclass(frozen=True)
class ValidationReport:
    """Versioned report shared by local and remote validation executors."""

    target: str
    profile: ValidationProfile
    policy_version: str
    runner: RunnerCapabilities
    results: tuple[ValidationResult, ...]
    duration_s: float
    started_at: str
    finished_at: str
    spec: ValidationSubject | None = None
    spec_identity: SpecIdentity | None = None
    repo_sha: str | None = None
    image_digest: str | None = None
    certified: bool = False
    certification_eligible: bool = False
    report_schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if not _STABLE_IDENTIFIER.fullmatch(self.policy_version):
            raise ValueError(
                "ValidationReport policy version must be a stable identifier"
            )
        if (
            self.spec is not None
            and self.spec_identity is not None
            and self.spec_identity != self.spec.spec
        ):
            raise ValueError("ValidationReport spec identity must match its subject")

    @property
    def passed(self) -> bool:
        """Return whether blocking criteria satisfy this profile's gate."""
        if self.profile is ValidationProfile.PUBLISH:
            return not any(
                result.severity is ValidationSeverity.BLOCKING
                and result.status is not ValidationStatus.PASS
                for result in self.results
            )
        return not any(
            result.severity is ValidationSeverity.BLOCKING
            and result.status in {ValidationStatus.FAIL, ValidationStatus.ERROR}
            for result in self.results
        )

    @property
    def status(self) -> ValidationStatus:
        """Return the aggregate report status."""
        return ValidationStatus.PASS if self.passed else ValidationStatus.FAIL

    def _summary(self) -> dict[str, Any]:
        counts = {
            status.value: sum(1 for result in self.results if result.status is status)
            for status in ValidationStatus
        }
        failed = [
            result.criterion_id
            for result in self.results
            if result.status in {ValidationStatus.FAIL, ValidationStatus.ERROR}
        ]
        blocking_failed = [
            result.criterion_id
            for result in self.results
            if result.severity is ValidationSeverity.BLOCKING
            and result.status in {ValidationStatus.FAIL, ValidationStatus.ERROR}
        ]
        blocking_skipped = [
            result.criterion_id
            for result in self.results
            if result.severity is ValidationSeverity.BLOCKING
            and result.status is ValidationStatus.SKIP
        ]
        return {
            "passed_count": counts[ValidationStatus.PASS.value],
            "failed_count": counts[ValidationStatus.FAIL.value],
            "skipped_count": counts[ValidationStatus.SKIP.value],
            "error_count": counts[ValidationStatus.ERROR.value],
            "total_count": len(self.results),
            "failed_criteria": failed,
            "blocking_failed_criteria": blocking_failed,
            "blocking_skipped_criteria": blocking_skipped,
            "required_passed_count": sum(
                1
                for result in self.results
                if result.severity is ValidationSeverity.BLOCKING
                and result.status is ValidationStatus.PASS
            ),
            "required_total_count": sum(
                1
                for result in self.results
                if result.severity is ValidationSeverity.BLOCKING
            ),
            "status_counts": counts,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned JSON report."""
        payload = {
            "report_schema_version": self.report_schema_version,
            "policy_version": self.policy_version,
            "spec": (
                self.spec.to_dict()
                if self.spec is not None
                else (
                    self.spec_identity.to_dict()
                    if self.spec_identity is not None
                    else None
                )
            ),
            "target": self.target,
            "validation_type": "openenv_validation",
            "profile": self.profile.value,
            "runner": self.runner.to_dict(),
            "status": self.status.value,
            "passed": self.passed,
            "certified": self.certified,
            "certification_eligible": self.certification_eligible,
            "repo_sha": self.repo_sha,
            "image_digest": self.image_digest,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": round(self.duration_s, 6),
            "summary": self._summary(),
            "criteria": [result.to_dict() for result in self.results],
        }
        serialized = json_safe(
            payload,
            trusted_string_keys=_TRUSTED_METADATA_STRING_KEYS,
        )
        assert isinstance(serialized, dict)
        if isinstance(payload["spec"], Mapping) and isinstance(
            serialized.get("spec"), dict
        ):
            _restore_structural_spec_identity(serialized["spec"], payload["spec"])
        serialized_criteria = serialized.get("criteria")
        if isinstance(serialized_criteria, list):
            for safe_result, raw_result in zip(
                serialized_criteria, payload["criteria"], strict=True
            ):
                if isinstance(safe_result, dict):
                    safe_result["description"] = raw_result["description"]
                    safe_result["requirement"] = raw_result["requirement"]
        serialized_summary = serialized.get("summary")
        if isinstance(serialized_summary, dict):
            serialized_summary["failed_criteria"] = payload["summary"][
                "failed_criteria"
            ]
            serialized_summary["blocking_failed_criteria"] = payload["summary"][
                "blocking_failed_criteria"
            ]
            serialized_summary["blocking_skipped_criteria"] = payload["summary"][
                "blocking_skipped_criteria"
            ]
        return serialized
