# SPDX-License-Identifier: BSD-3-Clause

"""Capability-aware executor for shared validation plans."""

from __future__ import annotations

import re
import signal
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, Mapping

from .models import (
    CheckOutcome,
    ValidationContext,
    ValidationDiagnostic,
    ValidationPlan,
    ValidationProfile,
    ValidationRemediation,
    ValidationReport,
    ValidationResult,
    ValidationSeverity,
    ValidationStatus,
)
from .planner import is_canonical_policy_plan
from .serialization import json_safe, redact_string


class _CheckTimedOut(BaseException):
    pass


class _TimeoutContextUnsupported(BaseException):
    pass


@contextmanager
def _timeout(seconds: float) -> Iterator[None]:
    supported = (
        hasattr(signal, "SIGALRM")
        and threading.current_thread() is threading.main_thread()
    )
    if not supported:
        raise _TimeoutContextUnsupported

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)

    def handle_timeout(_signum: int, _frame: Any) -> None:
        raise _CheckTimedOut

    signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


def _execute_check(
    plan: ValidationPlan,
    context: ValidationContext,
    index: int,
) -> ValidationResult:
    check = plan.checks[index]
    missing = check.capabilities - plan.capabilities.available
    if missing:
        declared_reasons = context.discovered.get("capability_unavailable_reasons", {})
        reasons = (
            {
                capability.value: declared_reasons[capability.value]
                for capability in missing
                if capability.value in declared_reasons
                and isinstance(declared_reasons[capability.value], str)
            }
            if isinstance(declared_reasons, Mapping)
            else {}
        )
        return ValidationResult(
            criterion_id=check.criterion_id,
            requirement=check.requirement,
            status=ValidationStatus.SKIP,
            severity=check.severity,
            evidence={
                "reason": "runner_capability_unavailable",
                "required": sorted(
                    capability.value for capability in check.capabilities
                ),
                "missing": sorted(capability.value for capability in missing),
                "available": sorted(
                    capability.value for capability in plan.capabilities.available
                ),
                "unavailable_reasons": reasons,
            },
            duration_s=0.0,
            timeout_s=check.timeout_s,
            required_capabilities=check.capabilities,
            built_in=check.built_in,
            message="Runner does not provide the capabilities required by this check",
            diagnostics=(
                ValidationDiagnostic(
                    code="runner_capability_unavailable",
                    message=(
                        "This blocking check could not run with the selected "
                        "validation runner"
                    ),
                ),
            ),
        )

    started = time.perf_counter()
    if check.evaluator is None:
        return ValidationResult(
            criterion_id=check.criterion_id,
            requirement=check.requirement,
            status=ValidationStatus.ERROR,
            severity=check.severity,
            evidence={"reason": "missing_check_implementation"},
            duration_s=time.perf_counter() - started,
            timeout_s=check.timeout_s,
            required_capabilities=check.capabilities,
            built_in=check.built_in,
            message="No check implementation is registered",
        )

    try:
        with _timeout(check.timeout_s):
            outcome = check.evaluator(context)
        if not isinstance(outcome, CheckOutcome):
            outcome = CheckOutcome.error(
                {"actual_type": type(outcome).__name__},
                message="Check implementation returned an invalid outcome",
            )
        elif (
            not isinstance(outcome.status, ValidationStatus)
            or not isinstance(outcome.evidence, Mapping)
            or (outcome.message is not None and not isinstance(outcome.message, str))
            or not isinstance(outcome.diagnostics, tuple)
            or not all(
                isinstance(item, ValidationDiagnostic) for item in outcome.diagnostics
            )
            or not isinstance(outcome.remediation, tuple)
            or not all(
                isinstance(item, ValidationRemediation) for item in outcome.remediation
            )
        ):
            outcome = CheckOutcome.error(
                {"reason": "invalid_outcome_shape"},
                message="Check implementation returned malformed status, evidence, or message",
            )
        elif time.perf_counter() - started > check.timeout_s:
            outcome = CheckOutcome.error(
                {"reason": "timeout", "timeout_s": check.timeout_s},
                message=f"Check exceeded its {check.timeout_s:g}s timeout",
            )
    except _CheckTimedOut:
        outcome = CheckOutcome.error(
            {"reason": "timeout", "timeout_s": check.timeout_s},
            message=f"Check exceeded its {check.timeout_s:g}s timeout",
        )
    except _TimeoutContextUnsupported:
        outcome = CheckOutcome.error(
            {
                "reason": "timeout_context_unsupported",
                "timeout_s": check.timeout_s,
            },
            message=(
                "Timed checks must run on the POSIX main thread or in a "
                "runner-managed worker process"
            ),
        )
    except Exception as exc:
        outcome = CheckOutcome.error(
            {"reason": "uncaught_exception", "error_type": type(exc).__name__},
            message="Check implementation raised an unexpected exception",
        )

    try:
        evidence = json_safe(outcome.evidence)
    except Exception as exc:
        outcome = CheckOutcome.error(
            {
                "reason": "invalid_evidence",
                "error_type": type(exc).__name__,
            },
            message="Check evidence is not safe to serialize",
        )
        evidence = json_safe(outcome.evidence)

    diagnostics = outcome.diagnostics
    if (
        not diagnostics
        and check.built_in
        and outcome.status is not ValidationStatus.PASS
        and not (
            check.severity is ValidationSeverity.ADVISORY
            and outcome.status is ValidationStatus.SKIP
        )
    ):
        diagnostics = (
            ValidationDiagnostic(
                code=f"criterion_{outcome.status.value}",
                message=(
                    "Review this criterion's safe evidence and correct the environment "
                    "configuration before retrying"
                ),
            ),
        )

    return ValidationResult(
        criterion_id=check.criterion_id,
        requirement=check.requirement,
        status=outcome.status,
        # Severity is policy, not runner evidence. A subject or lane must not
        # downgrade a canonical blocking criterion through CheckOutcome.
        severity=check.severity,
        evidence=evidence,
        duration_s=time.perf_counter() - started,
        timeout_s=check.timeout_s,
        required_capabilities=check.capabilities,
        built_in=check.built_in,
        message=(
            redact_string(outcome.message) if outcome.message is not None else None
        ),
        diagnostics=diagnostics,
        remediation=outcome.remediation,
    )


def execute_validation_plan(
    plan: ValidationPlan, context: ValidationContext
) -> ValidationReport:
    """Execute every planned check and preserve partial results on errors."""
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    results = tuple(
        _execute_check(plan, context, index) for index in range(len(plan.checks))
    )
    finished_at = datetime.now(timezone.utc)

    blocking_skips = any(
        result.severity is ValidationSeverity.BLOCKING
        and result.status is ValidationStatus.SKIP
        for result in results
    )
    all_blocking_pass = not any(
        result.severity is ValidationSeverity.BLOCKING
        and result.status is not ValidationStatus.PASS
        for result in results
    )
    policy_complete = bool(
        plan.profile is ValidationProfile.FULL
        and is_canonical_policy_plan(plan, context)
    )
    repo_sha_valid = bool(
        isinstance(context.repo_sha, str)
        and re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", context.repo_sha)
    )
    image_digest_valid = bool(
        isinstance(context.image_digest, str)
        and re.fullmatch(r"sha256:[0-9a-f]{64}", context.image_digest)
    )
    certification_eligible = bool(
        policy_complete
        and plan.capabilities.official
        and plan.capabilities.isolation_mode == "dedicated"
        and context.discovered.get("runner_attestation_verified") is True
        and context.discovered.get("provenance_binding_verified") is True
        and repo_sha_valid
        and image_digest_valid
        and not blocking_skips
        and all_blocking_pass
    )
    return ValidationReport(
        target=(
            redact_string(plan.target)
            if isinstance(plan.target, str)
            else "[INVALID_TARGET]"
        ),
        profile=plan.profile,
        policy_version=plan.policy_version,
        runner=plan.capabilities,
        results=results,
        duration_s=time.perf_counter() - started,
        started_at=started_at.isoformat(),
        finished_at=finished_at.isoformat(),
        spec=plan.spec,
        spec_identity=plan.spec_identity,
        repo_sha=(
            redact_string(context.repo_sha)
            if isinstance(context.repo_sha, str)
            else None
        ),
        image_digest=(
            redact_string(context.image_digest)
            if isinstance(context.image_digest, str)
            else None
        ),
        # Certification is a property of the signed registry envelope produced
        # after this immutable payload exists. An executor cannot verify a
        # signature over a report it has not produced yet.
        certified=False,
        certification_eligible=certification_eligible,
    )
