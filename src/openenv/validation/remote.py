# SPDX-License-Identifier: BSD-3-Clause

"""Author-triggered validation in a dedicated Hugging Face Sandbox."""

from __future__ import annotations

import json
import re
import tarfile
from contextlib import suppress
from dataclasses import replace
from importlib import metadata
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping

from .models import (
    DiagnosticLocation,
    RunnerCapabilities,
    ValidationCapability,
    ValidationDiagnostic,
    ValidationProfile,
    ValidationRemediation,
    ValidationReport,
    ValidationResult,
    ValidationSeverity,
    ValidationStatus,
)
from .specs import (
    AdapterIdentity,
    DetectionMode,
    ExecutionModel,
    RequirementsLoad,
    RequirementsProvenance,
    RequirementsState,
    SpecIdentity,
    ValidationRequirements,
    ValidationSubject,
)


DEFAULT_SANDBOX_IMAGE = "python:3.12"
DEFAULT_SANDBOX_FLAVOR = "cpu-basic"
_REMOTE_REVISION_ARCHIVE = "/tmp/openenv-revision.tar.gz"
_REMOTE_VALIDATOR_ARCHIVE = "/tmp/openenv-validator.tar.gz"
_REMOTE_REPORT = "/workspace/validation-report.json"
_REVISION_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_EXCLUDED_PARTS = frozenset(
    {
        ".git",
        ".hg",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        ".venv",
        "__pycache__",
        "node_modules",
        "venv",
    }
)
_EXCLUDED_NAMES = frozenset(
    {
        ".netrc",
        ".npmrc",
        ".pypirc",
        "credentials.json",
        "validation-report.json",
    }
)


class RemoteValidationError(RuntimeError):
    """Remote author validation could not produce a trustworthy report payload."""


def _archive_path_allowed(relative: Path) -> bool:
    if any(part in _EXCLUDED_PARTS for part in relative.parts):
        return False
    name = relative.name
    if name in _EXCLUDED_NAMES or name == ".env" or name.startswith(".env."):
        return False
    if relative.as_posix() == ".openenv/validation-report.json":
        return False
    return True


def _add_tree(archive: tarfile.TarFile, root: Path, *, prefix: Path) -> None:
    def normalize(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        info.mtime = 0
        return info

    for candidate in sorted(root.rglob("*")):
        relative = candidate.relative_to(root)
        if (
            not _archive_path_allowed(relative)
            or candidate.is_symlink()
            or not (candidate.is_file() or candidate.is_dir())
        ):
            continue
        archive.add(
            candidate,
            arcname=(prefix / relative).as_posix(),
            recursive=False,
            filter=normalize,
        )


def _create_revision_archive(source: Path, destination: Path) -> None:
    with tarfile.open(destination, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
        _add_tree(archive, source, prefix=Path("."))


def _validator_project_root() -> Path | None:
    package = Path(__file__).resolve().parents[1]
    project = package.parent.parent
    if (project / "pyproject.toml").is_file() and (
        project / "src" / "openenv"
    ) == package:
        return project
    return None


def _add_bytes(archive: tarfile.TarFile, *, name: str, content: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(content)
    info.mode = 0o644
    info.mtime = 0
    archive.addfile(info, BytesIO(content))


def _installed_validator_pyproject() -> bytes:
    try:
        version = metadata.version("openenv")
        requirements = metadata.requires("openenv") or []
    except metadata.PackageNotFoundError as exc:
        raise RemoteValidationError(
            "Remote validation could not resolve the installed OpenEnv package metadata"
        ) from exc
    dependencies = ",\n  ".join(json.dumps(item) for item in requirements)
    return (
        "[build-system]\n"
        'requires = ["setuptools>=77", "wheel"]\n'
        'build-backend = "setuptools.build_meta"\n\n'
        "[project]\n"
        'name = "openenv"\n'
        f"version = {json.dumps(version)}\n"
        'requires-python = ">=3.10"\n'
        f"dependencies = [\n  {dependencies}\n]\n\n"
        "[tool.setuptools]\n"
        'package-dir = {"" = "src"}\n\n'
        "[tool.setuptools.packages.find]\n"
        'where = ["src"]\n'
    ).encode("utf-8")


def _create_validator_archive(destination: Path) -> None:
    project = _validator_project_root()
    package = Path(__file__).resolve().parents[1]
    with tarfile.open(destination, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
        _add_tree(archive, package, prefix=Path("src/openenv"))
        if project is None:
            _add_bytes(
                archive,
                name="pyproject.toml",
                content=_installed_validator_pyproject(),
            )
        else:
            for name in ("pyproject.toml", "README.md", "LICENSE"):
                candidate = project / name
                if candidate.is_file() and not candidate.is_symlink():
                    _add_bytes(
                        archive,
                        name=name,
                        content=candidate.read_bytes(),
                    )


def _remote_command(
    *, profile: ValidationProfile, runtime_timeout_s: float
) -> list[str]:
    script = r"""
set -eu
revision_archive="$1"
validator_archive="$2"
report_path="$3"
profile="$4"
runtime_timeout="$5"
environment_root=/workspace/environment
validator_root=/workspace/validator
mkdir -p "$environment_root" "$validator_root" || exit 70
tar -xzf "$revision_archive" -C "$environment_root" || exit 70
tar -xzf "$validator_archive" -C "$validator_root" || exit 70
python -m pip install --disable-pip-version-check --no-input "$validator_root" || exit 70
python -m pip install --disable-pip-version-check --no-input "$environment_root" || exit 71
set +e
PYTHONPATH="$validator_root/src" python -m openenv.cli.__main__ validate \
  "$environment_root" --profile "$profile" --json --output "$report_path" \
  --timeout "$runtime_timeout"
validation_status=$?
set -e
test -f "$report_path" || exit 72
exit "$validation_status"
""".strip()
    return [
        "/bin/sh",
        "-c",
        script,
        "openenv-remote-validation",
        _REMOTE_REVISION_ARCHIVE,
        _REMOTE_VALIDATOR_ARCHIVE,
        _REMOTE_REPORT,
        profile.value,
        f"{runtime_timeout_s:g}",
    ]


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    return float(value)


def _adapter(payload: Mapping[str, Any]) -> AdapterIdentity:
    return AdapterIdentity(
        adapter_id=_string(payload.get("id"), field="spec.adapter.id"),
        adapter_version=_string(payload.get("version"), field="spec.adapter.version"),
    )


def _spec_identity(payload: Mapping[str, Any]) -> SpecIdentity:
    adapter = _adapter(_mapping(payload.get("adapter"), field="spec.adapter"))
    version = payload.get("version")
    if version is not None and not isinstance(version, str):
        raise ValueError("spec.version must be a string or null")
    return SpecIdentity(
        spec_id=_string(payload.get("id"), field="spec.id"),
        spec_version=version,
        adapter=adapter,
        execution_model=ExecutionModel(
            _string(payload.get("execution_model"), field="spec.execution_model")
        ),
    )


def _requirements_provenance(value: Any) -> RequirementsProvenance | None:
    if value is None:
        return None
    payload = _mapping(value, field="spec.requirements")
    version = payload.get("version")
    digest = payload.get("document_digest")
    if version is not None and not isinstance(version, str):
        raise ValueError("spec.requirements.version must be a string or null")
    if digest is not None and not isinstance(digest, str):
        raise ValueError("spec.requirements.document_digest must be a string or null")
    return RequirementsProvenance(
        source_id=_string(payload.get("id"), field="spec.requirements.id"),
        source_version=version,
        adapter=_adapter(
            _mapping(payload.get("adapter"), field="spec.requirements.adapter")
        ),
        path=_string(payload.get("path"), field="spec.requirements.path"),
        document_digest=digest,
    )


def _subject_or_identity(
    value: Any,
) -> tuple[ValidationSubject | None, SpecIdentity | None]:
    if value is None:
        return None, None
    payload = _mapping(value, field="spec")
    identity = _spec_identity(payload)
    detection_mode = payload.get("detection_mode")
    if detection_mode is None:
        return None, identity
    requirements_state = RequirementsState(
        _string(payload.get("requirements_state"), field="spec.requirements_state")
    )
    requirements = RequirementsLoad(
        state=requirements_state,
        provenance=_requirements_provenance(payload.get("requirements")),
        requirements=(
            ValidationRequirements()
            if requirements_state is RequirementsState.LOADED
            else None
        ),
        error=(
            "Remote requirements were invalid"
            if requirements_state
            in {RequirementsState.INVALID, RequirementsState.UNSUPPORTED}
            else None
        ),
    )
    verifier = payload.get("verifier")
    verifier_path: str | None = None
    verifier_digest: str | None = None
    if verifier is not None:
        verifier_payload = _mapping(verifier, field="spec.verifier")
        verifier_path = _string(
            verifier_payload.get("path"), field="spec.verifier.path"
        )
        verifier_digest = _string(
            verifier_payload.get("document_digest"),
            field="spec.verifier.document_digest",
        )
    signature_path = payload.get("signature_path")
    document_digest = payload.get("document_digest")
    if signature_path is not None and not isinstance(signature_path, str):
        raise ValueError("spec.signature_path must be a string or null")
    if document_digest is not None and not isinstance(document_digest, str):
        raise ValueError("spec.document_digest must be a string or null")
    subject = ValidationSubject(
        spec=identity,
        signature_path=signature_path,
        detection_mode=DetectionMode(_string(detection_mode, field="detection_mode")),
        requirements=requirements,
        verifier_script=Path(verifier_path) if verifier_path is not None else None,
        verifier_path=verifier_path,
        verifier_digest=verifier_digest,
        document_digest=document_digest,
    )
    return subject, identity


def _diagnostic(value: Any) -> ValidationDiagnostic:
    payload = _mapping(value, field="criterion.diagnostic")
    location_payload = payload.get("location")
    location = None
    if location_payload is not None:
        raw = _mapping(location_payload, field="criterion.diagnostic.location")
        location = DiagnosticLocation(
            path=_string(raw.get("path"), field="diagnostic.location.path"),
            pointer=raw.get("pointer"),
            line=raw.get("line"),
            column=raw.get("column"),
        )
    return ValidationDiagnostic(
        code=_string(payload.get("code"), field="diagnostic.code"),
        message=_string(payload.get("message"), field="diagnostic.message"),
        location=location,
    )


def _remediation(value: Any) -> ValidationRemediation:
    payload = _mapping(value, field="criterion.remediation")
    argv = payload.get("argv", [])
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise ValueError("remediation.argv must be a string array")
    return ValidationRemediation(
        kind=_string(payload.get("kind"), field="remediation.kind"),
        message=_string(payload.get("message"), field="remediation.message"),
        argv=tuple(argv),
        cwd=payload.get("cwd"),
        path=payload.get("path"),
        pointer=payload.get("pointer"),
        url=payload.get("url"),
    )


def _result(value: Any) -> ValidationResult:
    payload = _mapping(value, field="criterion")
    capabilities = payload.get("required_capabilities")
    diagnostics = payload.get("diagnostics", [])
    remediation = payload.get("remediation", [])
    if not isinstance(capabilities, list):
        raise ValueError("criterion.required_capabilities must be an array")
    if not isinstance(diagnostics, list) or not isinstance(remediation, list):
        raise ValueError("criterion guidance must be arrays")
    evidence = _mapping(payload.get("evidence"), field="criterion.evidence")
    message = payload.get("details")
    if message is not None and not isinstance(message, str):
        raise ValueError("criterion.details must be a string or null")
    return ValidationResult(
        criterion_id=_string(payload.get("id"), field="criterion.id"),
        requirement=_string(
            payload.get("requirement", payload.get("description")),
            field="criterion.requirement",
        ),
        status=ValidationStatus(
            _string(payload.get("status"), field="criterion.status")
        ),
        severity=ValidationSeverity(
            _string(payload.get("severity"), field="criterion.severity")
        ),
        evidence=dict(evidence),
        duration_s=_number(payload.get("duration_s"), field="criterion.duration_s"),
        timeout_s=_number(payload.get("timeout_s"), field="criterion.timeout_s"),
        required_capabilities=frozenset(
            ValidationCapability(_string(item, field="criterion.capability"))
            for item in capabilities
        ),
        built_in=payload.get("built_in", True) is True,
        message=message,
        diagnostics=tuple(_diagnostic(item) for item in diagnostics),
        remediation=tuple(_remediation(item) for item in remediation),
    )


def _parse_report(
    payload: Any, *, expected_profile: ValidationProfile
) -> ValidationReport:
    document = _mapping(payload, field="report")
    if document.get("validation_type") != "openenv_validation":
        raise ValueError("report.validation_type is invalid")
    if document.get("report_schema_version") != "1.0":
        raise ValueError("report schema version is unsupported")
    profile = ValidationProfile(
        _string(document.get("profile"), field="report.profile")
    )
    if profile is not expected_profile:
        raise ValueError("report profile does not match the requested profile")
    runner_payload = _mapping(document.get("runner"), field="report.runner")
    capabilities = runner_payload.get("capabilities")
    if not isinstance(capabilities, list):
        raise ValueError("report.runner.capabilities must be an array")
    results_payload = document.get("criteria")
    if not isinstance(results_payload, list):
        raise ValueError("report.criteria must be an array")
    subject, identity = _subject_or_identity(document.get("spec"))
    report = ValidationReport(
        target=_string(document.get("target"), field="report.target"),
        profile=profile,
        policy_version=_string(
            document.get("policy_version"), field="report.policy_version"
        ),
        runner=RunnerCapabilities(
            runner=_string(runner_payload.get("kind"), field="report.runner.kind"),
            available=frozenset(
                ValidationCapability(_string(item, field="report.runner.capability"))
                for item in capabilities
            ),
            official=runner_payload.get("official") is True,
            isolation_mode=runner_payload.get("isolation_mode"),
        ),
        results=tuple(_result(item) for item in results_payload),
        duration_s=_number(document.get("duration_s"), field="report.duration_s"),
        started_at=_string(document.get("started_at"), field="report.started_at"),
        finished_at=_string(document.get("finished_at"), field="report.finished_at"),
        spec=subject,
        spec_identity=identity,
        repo_sha=document.get("repo_sha"),
        image_digest=document.get("image_digest"),
        certified=document.get("certified") is True,
        certification_eligible=document.get("certification_eligible") is True,
        report_schema_version="1.0",
    )
    if document.get("passed") is not report.passed:
        raise ValueError("report pass state is inconsistent with its criteria")
    return report


def run_remote_validation(
    source: str | Path,
    *,
    profile: ValidationProfile | str = ValidationProfile.PUBLISH,
    repo_sha: str | None = None,
    sandbox_image: str = DEFAULT_SANDBOX_IMAGE,
    flavor: str = DEFAULT_SANDBOX_FLAVOR,
    runtime_timeout_s: float = 5.0,
    sandbox_timeout_s: float = 900.0,
) -> ValidationReport:
    """Run an unofficial author validation in a fresh dedicated HF Sandbox."""
    selected_profile = (
        profile
        if isinstance(profile, ValidationProfile)
        else ValidationProfile(profile)
    )
    source_path = Path(source).resolve()
    if not source_path.is_dir():
        raise RemoteValidationError(
            "Remote validation requires an existing local source directory"
        )
    if repo_sha is not None and _REVISION_PATTERN.fullmatch(repo_sha) is None:
        raise RemoteValidationError("Remote validation repo SHA is malformed")
    if runtime_timeout_s <= 0 or sandbox_timeout_s <= 0:
        raise RemoteValidationError("Remote validation timeouts must be positive")
    if not isinstance(flavor, str) or not flavor.strip():
        raise RemoteValidationError("Remote validation hardware flavor is invalid")

    try:
        from huggingface_hub import Sandbox
    except ImportError as exc:  # pragma: no cover - depends on installed extra
        raise RemoteValidationError(
            "Remote validation requires `openenv[hf-sandbox]`"
        ) from exc

    sandbox = None
    with TemporaryDirectory(prefix="openenv-remote-validation-") as temporary:
        temporary_path = Path(temporary)
        revision_archive = temporary_path / "revision.tar.gz"
        validator_archive = temporary_path / "validator.tar.gz"
        downloaded_report = temporary_path / "validation-report.json"
        _create_revision_archive(source_path, revision_archive)
        _create_validator_archive(validator_archive)

        try:
            sandbox = Sandbox.create(
                image=sandbox_image,
                flavor=flavor,
                idle_timeout=max(60.0, sandbox_timeout_s),
                forward_hf_token=False,
                start_timeout=min(300.0, sandbox_timeout_s),
            )
            sandbox.files.upload(revision_archive, _REMOTE_REVISION_ARCHIVE)
            sandbox.files.upload(validator_archive, _REMOTE_VALIDATOR_ARCHIVE)
            command_result = sandbox.run(
                _remote_command(
                    profile=selected_profile,
                    runtime_timeout_s=runtime_timeout_s,
                ),
                shell=False,
                timeout=sandbox_timeout_s,
                check=False,
            )
            if command_result.timed_out or command_result.exit_code not in {0, 1}:
                raise RemoteValidationError(
                    "Remote validation failed before a report was produced"
                )
            sandbox.files.download(_REMOTE_REPORT, downloaded_report)
            try:
                payload = json.loads(downloaded_report.read_text(encoding="utf-8"))
                report = _parse_report(payload, expected_profile=selected_profile)
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                raise RemoteValidationError(
                    "Remote validation report is invalid or malformed"
                ) from exc
            if repo_sha is not None and report.repo_sha not in {None, repo_sha}:
                raise RemoteValidationError(
                    "Remote validation report is not bound to the requested revision"
                )
            return replace(
                report,
                target=str(source_path),
                profile=selected_profile,
                runner=RunnerCapabilities(
                    runner="hf-sandbox",
                    available=frozenset(
                        {
                            ValidationCapability.SOURCE,
                            ValidationCapability.RUNTIME,
                        }
                    ),
                    official=False,
                    isolation_mode="dedicated",
                ),
                repo_sha=repo_sha or report.repo_sha,
                certified=False,
                certification_eligible=False,
            )
        except RemoteValidationError:
            raise
        except Exception as exc:
            raise RemoteValidationError(
                f"Remote validation failed ({type(exc).__name__})"
            ) from exc
        finally:
            if sandbox is not None:
                with suppress(Exception):
                    sandbox.close()
