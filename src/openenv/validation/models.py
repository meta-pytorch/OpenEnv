# SPDX-License-Identifier: BSD-3-Clause

"""Shared models for local validation and remote certification runners."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping

from .specs.base import SpecLoad


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


@dataclass(frozen=True)
class RunnerCapabilities:
    """Capabilities and trust properties exposed by a validation runner."""

    runner: str
    available: frozenset[ValidationCapability] = frozenset()
    official: bool = False
    isolation_mode: str | None = None

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

    @classmethod
    def pass_(
        cls,
        evidence: Mapping[str, Any] | None = None,
        *,
        message: str | None = None,
        severity: ValidationSeverity | None = None,
    ) -> "CheckOutcome":
        """Build a passing outcome."""
        return cls(
            ValidationStatus.PASS,
            evidence or {},
            message=message,
            severity=severity,
        )

    @classmethod
    def fail(
        cls,
        evidence: Mapping[str, Any] | None = None,
        *,
        message: str | None = None,
        severity: ValidationSeverity | None = None,
    ) -> "CheckOutcome":
        """Build a failing outcome."""
        return cls(
            ValidationStatus.FAIL,
            evidence or {},
            message=message,
            severity=severity,
        )

    @classmethod
    def skip(
        cls,
        evidence: Mapping[str, Any] | None = None,
        *,
        message: str | None = None,
        severity: ValidationSeverity | None = None,
    ) -> "CheckOutcome":
        """Build a skipped outcome."""
        return cls(
            ValidationStatus.SKIP,
            evidence or {},
            message=message,
            severity=severity,
        )

    @classmethod
    def error(
        cls,
        evidence: Mapping[str, Any] | None = None,
        *,
        message: str | None = None,
        severity: ValidationSeverity | None = None,
    ) -> "CheckOutcome":
        """Build an infrastructure-error outcome."""
        return cls(
            ValidationStatus.ERROR,
            evidence or {},
            message=message,
            severity=severity,
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

    def __post_init__(self) -> None:
        if not self.criterion_id.strip():
            raise ValueError("ValidationCheck criterion_id cannot be empty")
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
    requirements: Mapping[str, Any] = field(default_factory=dict)
    _policy_attestation: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        criterion_ids = [check.criterion_id for check in self.checks]
        if len(criterion_ids) != len(set(criterion_ids)):
            raise ValueError("ValidationPlan contains duplicate criterion IDs")

    def to_dict(self) -> dict[str, Any]:
        """Return the safe, execution-independent plan representation."""
        return {
            "target": self.target,
            "profile": self.profile.value,
            "policy_version": self.policy_version,
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
                }
                for check in self.checks
            ],
        }


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
            "evidence": dict(self.evidence),
            "duration_s": round(self.duration_s, 6),
            "duration_ms": round(self.duration_s * 1000, 3),
            "timeout_s": self.timeout_s,
        }
        if self.message is not None:
            payload["details"] = self.message
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
    repo_sha: str | None = None
    image_digest: str | None = None
    certified: bool = False
    certification_eligible: bool = False
    report_schema_version: str = "1.0"

    @property
    def passed(self) -> bool:
        """Return whether all executed blocking criteria avoid failure/error."""
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
        return {
            "passed_count": counts[ValidationStatus.PASS.value],
            "failed_count": counts[ValidationStatus.FAIL.value],
            "skipped_count": counts[ValidationStatus.SKIP.value],
            "error_count": counts[ValidationStatus.ERROR.value],
            "total_count": len(self.results),
            "failed_criteria": failed,
            "blocking_failed_criteria": blocking_failed,
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
        return {
            "report_schema_version": self.report_schema_version,
            "policy_version": self.policy_version,
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
