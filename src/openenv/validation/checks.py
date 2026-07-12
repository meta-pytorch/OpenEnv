# SPDX-License-Identifier: BSD-3-Clause

"""Built-in validation check registry for RFC 008 policy version 1."""

from __future__ import annotations

import ast
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import yaml

from .models import (
    CheckOutcome,
    DiagnosticLocation,
    ValidationCapability,
    ValidationCheck,
    ValidationContext,
    ValidationDiagnostic,
    ValidationProfile,
    ValidationRemediation,
    ValidationRequirementBinding,
    ValidationSeverity,
    ValidationStatus,
)
from .source_inspection import (
    _dockerfile_installs_openenv_runtime,
    _has_main_guard_call,
    _is_safe_regular_file,
)
from .specs import RequirementsState, SpecLoadState

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


_STATIC_PROFILES = frozenset(
    {
        ValidationProfile.STATIC,
        ValidationProfile.RUNTIME,
        ValidationProfile.FULL,
        ValidationProfile.PUBLISH,
    }
)
_RUNTIME_PROFILES = frozenset(
    {ValidationProfile.RUNTIME, ValidationProfile.FULL, ValidationProfile.PUBLISH}
)
_FULL_PROFILE = frozenset({ValidationProfile.FULL})
_OPENENV_DEPENDENCY = re.compile(r"^openenv(?:\s*(?:$|[<>=!~@;])|\[)")
_OPENENV_CORE_DEPENDENCY = re.compile(r"^openenv-core(?:\s*(?:$|[<>=!~@;])|\[)")
_MCP_CONTROL_TOOL_NAMES = frozenset(
    {"reset", "reset_async", "step", "step_async", "state"}
)


def _target_path(context: ValidationContext) -> Path:
    if not isinstance(context.target, Path):
        return Path(str(context.target))
    return context.target


def _load_pyproject(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    pyproject_path = path / "pyproject.toml"
    if not _is_safe_regular_file(path, pyproject_path):
        return None, "Missing or unsafe pyproject.toml"
    try:
        with pyproject_path.open("rb") as pyproject_file:
            parsed = tomllib.load(pyproject_file)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return None, f"Unable to parse pyproject.toml ({type(exc).__name__})"
    if not isinstance(parsed, dict):
        return None, "pyproject.toml must contain a TOML table"
    return parsed, None


def _validation_spec(context: ValidationContext) -> CheckOutcome:
    loaded = context.spec_load
    if loaded.state is SpecLoadState.ABSENT:
        return CheckOutcome.fail(
            loaded.to_dict(),
            message="No supported validation spec was detected",
            diagnostics=(
                ValidationDiagnostic(
                    code="validation_spec_missing",
                    message="Add a supported environment manifest",
                    location=DiagnosticLocation(path="openenv.yaml"),
                ),
            ),
            remediation=(
                ValidationRemediation(
                    kind="edit",
                    message="Create openenv.yaml with the environment identity and runtime settings",
                    path="openenv.yaml",
                ),
            ),
        )
    if loaded.state in {
        SpecLoadState.INVALID,
        SpecLoadState.UNSUPPORTED,
        SpecLoadState.AMBIGUOUS,
    }:
        return CheckOutcome.fail(
            loaded.to_dict(),
            message="The validation spec could not be loaded",
            diagnostics=(
                ValidationDiagnostic(
                    code="validation_spec_invalid",
                    message="Fix the OpenEnv manifest before validation can continue",
                    location=DiagnosticLocation(path="openenv.yaml"),
                ),
            ),
            remediation=(
                ValidationRemediation(
                    kind="edit",
                    message="Correct the invalid fields or syntax in openenv.yaml",
                    path="openenv.yaml",
                ),
            ),
        )
    if loaded.subject is None:
        return CheckOutcome.error(
            {"state": loaded.state.value, "reason": "loader returned no subject"}
        )
    return CheckOutcome.pass_(loaded.to_dict())


def _validation_requirements(context: ValidationContext) -> CheckOutcome:
    subject = context.spec_load.subject
    if subject is None:
        return CheckOutcome.skip(
            {"requirements_present": False, "reason": "spec_unavailable"},
            message="Requirements cannot be loaded until the source spec is valid",
            diagnostics=(
                ValidationDiagnostic(
                    code="requirements_blocked_by_spec",
                    message="Requirements cannot be inspected until the environment manifest is valid",
                    location=DiagnosticLocation(path="openenv.yaml"),
                ),
            ),
        )
    loaded = subject.requirements
    if loaded.state is RequirementsState.ABSENT:
        return CheckOutcome.skip(
            {
                **loaded.to_evidence(),
                "migration": "Missing requirements are locally non-failing in rfc008-v1",
            },
            message="No requirements envelope found; policy defaults apply during migration",
            diagnostics=(
                ValidationDiagnostic(
                    code="requirements_missing",
                    message="A publish-ready environment must declare its execution requirements",
                    location=DiagnosticLocation(path="task.toml"),
                ),
            ),
            remediation=(
                ValidationRemediation(
                    kind="edit",
                    message='Add a Harbor schema 1.1 requirements envelope; start with schema_version = "1.1"',
                    path="task.toml",
                    pointer="/schema_version",
                ),
            ),
        )
    if loaded.state in {RequirementsState.INVALID, RequirementsState.UNSUPPORTED}:
        return CheckOutcome.fail(
            loaded.to_evidence(),
            message="The requirements envelope could not be loaded",
            diagnostics=(
                ValidationDiagnostic(
                    code="requirements_invalid",
                    message="Fix unsupported fields or invalid TOML in the requirements envelope",
                    location=DiagnosticLocation(path="task.toml"),
                ),
            ),
            remediation=(
                ValidationRemediation(
                    kind="edit",
                    message='Use the supported Harbor schema_version "1.1" configuration',
                    path="task.toml",
                    pointer="/schema_version",
                ),
            ),
        )
    if loaded.requirements is None:
        return CheckOutcome.error(
            {"state": loaded.state.value, "reason": "loader returned no requirements"}
        )
    return CheckOutcome.pass_(loaded.to_evidence())


def _openenv_manifest(context: ValidationContext) -> CheckOutcome:
    root = _target_path(context)
    path = root / "openenv.yaml"
    if not _is_safe_regular_file(root, path):
        return CheckOutcome.fail(
            {"path": str(path), "missing_or_unsafe": True},
            message="Missing or unsafe openenv.yaml",
            diagnostics=(
                ValidationDiagnostic(
                    code="manifest_missing",
                    message="The environment root must contain a regular openenv.yaml file",
                    location=DiagnosticLocation(path="openenv.yaml"),
                ),
            ),
            remediation=(
                ValidationRemediation(
                    kind="edit",
                    message="Create a regular openenv.yaml manifest in the environment root",
                    path="openenv.yaml",
                ),
            ),
        )
    try:
        manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return CheckOutcome.fail(
            {"path": str(path), "error_type": type(exc).__name__},
            message="Unable to parse openenv.yaml",
            diagnostics=(
                ValidationDiagnostic(
                    code="manifest_syntax_invalid",
                    message="openenv.yaml contains invalid YAML syntax",
                    location=DiagnosticLocation(path="openenv.yaml"),
                ),
            ),
            remediation=(
                ValidationRemediation(
                    kind="edit",
                    message="Correct the YAML syntax in openenv.yaml",
                    path="openenv.yaml",
                ),
            ),
        )
    if not isinstance(manifest, dict):
        return CheckOutcome.fail(
            {"path": str(path), "actual_type": type(manifest).__name__},
            message="openenv.yaml must contain a mapping",
        )
    required = ("spec_version", "name", "runtime", "app", "port")
    missing = [field for field in required if field not in manifest]
    if missing:
        return CheckOutcome.fail(
            {"path": str(path), "missing_fields": missing},
            message="openenv.yaml is missing required fields",
            diagnostics=tuple(
                ValidationDiagnostic(
                    code="manifest_field_missing",
                    message=f"Add the required `{field}` setting",
                    location=DiagnosticLocation(
                        path="openenv.yaml", pointer=f"/{field}"
                    ),
                )
                for field in missing
            ),
            remediation=tuple(
                ValidationRemediation(
                    kind="edit",
                    message=f"Configure `{field}` in openenv.yaml",
                    path="openenv.yaml",
                    pointer=f"/{field}",
                )
                for field in missing
            ),
        )
    invalid_fields: dict[str, Any] = {}
    spec_version = manifest.get("spec_version")
    if type(spec_version) is not int or spec_version != 1:
        invalid_fields["spec_version"] = manifest.get("spec_version")
    if not isinstance(manifest.get("name"), str) or not manifest["name"].strip():
        invalid_fields["name"] = type(manifest.get("name")).__name__
    if manifest.get("runtime") != "fastapi":
        invalid_fields["runtime"] = manifest.get("runtime")
    app = manifest.get("app")
    if not isinstance(app, str) or ":" not in app:
        invalid_fields["app"] = app
    port = manifest.get("port")
    if not isinstance(port, int) or isinstance(port, bool) or not 0 < port < 65536:
        invalid_fields["port"] = port
    if invalid_fields:
        return CheckOutcome.fail(
            {"path": str(path), "invalid_fields": invalid_fields},
            message="openenv.yaml contains unsupported or invalid field values",
            diagnostics=tuple(
                ValidationDiagnostic(
                    code="manifest_field_invalid",
                    message=f"Configure a supported value for `{field}`",
                    location=DiagnosticLocation(
                        path="openenv.yaml", pointer=f"/{field}"
                    ),
                )
                for field in invalid_fields
            ),
            remediation=tuple(
                ValidationRemediation(
                    kind="edit",
                    message=f"Correct `{field}` in openenv.yaml",
                    path="openenv.yaml",
                    pointer=f"/{field}",
                )
                for field in invalid_fields
            ),
        )
    return CheckOutcome.pass_(
        {
            "path": str(path),
            "spec_version": manifest.get("spec_version"),
            "name": manifest.get("name"),
            "runtime": manifest.get("runtime"),
            "app": manifest.get("app"),
            "port": manifest.get("port"),
        }
    )


def _project_layout(context: ValidationContext) -> CheckOutcome:
    path = _target_path(context)
    pyproject, error = _load_pyproject(path)
    issues: list[str] = []
    if error is not None:
        issues.append(error)
    server_app = path / "server" / "app.py"
    if not _is_safe_regular_file(path, server_app):
        issues.append("Missing or unsafe server/app.py")
    else:
        try:
            app_source = server_app.read_text(encoding="utf-8")
        except OSError:
            issues.append("Unable to read server/app.py")
        else:
            if "def main(" not in app_source:
                issues.append("server/app.py missing main() function")
            if not _has_main_guard_call(app_source):
                issues.append(
                    "server/app.py main() function not callable "
                    "(missing if __name__ == '__main__')"
                )
    if pyproject is not None:
        scripts = pyproject.get("project", {}).get("scripts", {})
        if not isinstance(scripts, dict) or "server" not in scripts:
            issues.append("Missing [project.scripts] server entry point")
        else:
            server_entry = scripts.get("server")
            if not isinstance(server_entry, str) or ":main" not in server_entry:
                issues.append(
                    f"Server entry point should reference main function, got: {server_entry}"
                )

    evidence = {
        "environment_root": str(path),
        "required_paths": ["pyproject.toml", "server/app.py"],
        "issues": issues,
    }
    if issues:
        return CheckOutcome.fail(
            evidence,
            message=f"{len(issues)} layout issue(s)",
            diagnostics=tuple(
                ValidationDiagnostic(
                    code="project_layout_issue",
                    message=issue,
                    location=DiagnosticLocation(path="."),
                )
                for issue in issues
            ),
            remediation=(
                ValidationRemediation(
                    kind="edit",
                    message="Add the missing project files or correct the server entry point",
                    path="pyproject.toml",
                ),
            ),
        )
    return CheckOutcome.pass_(evidence)


def _dependencies(context: ValidationContext) -> CheckOutcome:
    path = _target_path(context)
    pyproject, error = _load_pyproject(path)
    if error is not None or pyproject is None:
        return CheckOutcome.fail(
            {"error": error}, message="Cannot validate dependencies"
        )
    project = pyproject.get("project", {})
    dependencies = project.get("dependencies", []) if isinstance(project, dict) else []
    if not isinstance(dependencies, list):
        return CheckOutcome.fail(
            {"actual_type": type(dependencies).__name__},
            message="project.dependencies must be a list",
        )
    normalized = [str(dependency).strip().lower() for dependency in dependencies]
    declared = any(
        _OPENENV_DEPENDENCY.match(dependency)
        or _OPENENV_CORE_DEPENDENCY.match(dependency)
        for dependency in normalized
    )
    dockerfile_declared = _dockerfile_installs_openenv_runtime(path)
    if not declared and not dockerfile_declared:
        return CheckOutcome.fail(
            {
                "project_dependency_count": len(normalized),
                "dockerfile_installs_openenv": False,
            },
            message="Missing required dependency: openenv>=0.2.0",
            diagnostics=(
                ValidationDiagnostic(
                    code="openenv_dependency_missing",
                    message="The project must install the OpenEnv runtime dependency",
                    location=DiagnosticLocation(
                        path="pyproject.toml", pointer="/project/dependencies"
                    ),
                ),
            ),
            remediation=(
                ValidationRemediation(
                    kind="command",
                    message="Add the OpenEnv runtime dependency and refresh the lockfile",
                    argv=("uv", "add", "openenv>=0.2.0"),
                    cwd=".",
                ),
            ),
        )
    return CheckOutcome.pass_(
        {
            "declared_in_pyproject": declared,
            "declared_in_dockerfile": dockerfile_declared,
        }
    )


def _lockfile(context: ValidationContext) -> CheckOutcome:
    root = _target_path(context)
    lockfile = root / "uv.lock"
    if not _is_safe_regular_file(root, lockfile):
        return CheckOutcome.fail(
            {"path": str(lockfile), "missing_or_unsafe": True},
            message="Missing or unsafe uv.lock - run 'uv lock' to generate it",
            diagnostics=(
                ValidationDiagnostic(
                    code="lockfile_missing",
                    message="A publish-ready environment must include a regular uv.lock file",
                    location=DiagnosticLocation(path="uv.lock"),
                ),
            ),
            remediation=(
                ValidationRemediation(
                    kind="command",
                    message="Generate the dependency lockfile",
                    argv=("uv", "lock"),
                    cwd=".",
                ),
            ),
        )
    try:
        with lockfile.open("rb") as lockfile_handle:
            lock_data = tomllib.load(lockfile_handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return CheckOutcome.fail(
            {"path": str(lockfile), "error_type": type(exc).__name__},
            message="uv.lock is empty or invalid TOML",
        )
    if not isinstance(lock_data.get("version"), int):
        return CheckOutcome.fail(
            {"path": str(lockfile), "missing_fields": ["version"]},
            message="uv.lock does not contain a lockfile version",
        )
    return CheckOutcome.pass_(
        {"path": str(lockfile), "lock_version": lock_data["version"]}
    )


def _schema_sources(context: ValidationContext) -> CheckOutcome:
    path = _target_path(context)
    candidates = [path / "server" / "app.py", path / "models.py", path / "client.py"]
    checked: list[str] = []
    errors: list[dict[str, Any]] = []
    for candidate in candidates:
        if not _is_safe_regular_file(path, candidate):
            if candidate.exists() or candidate.is_symlink():
                errors.append(
                    {
                        "path": str(candidate.relative_to(path)),
                        "error_type": "unsafe_file",
                        "line": None,
                    }
                )
            continue
        checked.append(str(candidate.relative_to(path)))
        try:
            source = candidate.read_text(encoding="utf-8")
            ast.parse(source, filename=str(candidate))
        except (OSError, SyntaxError) as exc:
            errors.append(
                {
                    "path": str(candidate.relative_to(path)),
                    "error_type": type(exc).__name__,
                    "line": getattr(exc, "lineno", None),
                }
            )
    if errors:
        return CheckOutcome.fail(
            {"checked": checked, "errors": errors},
            message="Python schema/application sources do not compile",
            diagnostics=tuple(
                ValidationDiagnostic(
                    code="python_source_invalid",
                    message="Correct the Python syntax error in this source file",
                    location=DiagnosticLocation(
                        path=str(error["path"]), line=error.get("line")
                    ),
                )
                for error in errors
            ),
            remediation=(
                ValidationRemediation(
                    kind="edit",
                    message="Fix each reported Python syntax or file-safety error",
                    path=str(errors[0]["path"]),
                ),
            ),
        )
    return CheckOutcome.pass_({"checked": checked})


def _dockerfile(context: ValidationContext) -> CheckOutcome:
    path = _target_path(context)
    subject = context.spec_load.subject
    requirements_load = subject.requirements if subject is not None else None
    requirements = (
        requirements_load.requirements if requirements_load is not None else None
    )
    declared_image = (
        requirements.environment.container_image if requirements is not None else None
    )
    if declared_image:
        provenance = (
            requirements_load.provenance.to_dict()
            if requirements_load is not None
            and requirements_load.provenance is not None
            else None
        )
        return CheckOutcome.pass_(
            {"source": provenance, "declared_image": declared_image}
        )
    for candidate in (path / "server" / "Dockerfile", path / "Dockerfile"):
        if not _is_safe_regular_file(path, candidate):
            continue
        try:
            content = candidate.read_text(encoding="utf-8")
        except OSError:
            return CheckOutcome.fail(
                {"path": str(candidate)}, message="Unable to read Dockerfile"
            )
        has_from = any(
            line.lstrip().upper().startswith("FROM ") for line in content.splitlines()
        )
        if not has_from:
            return CheckOutcome.fail(
                {"path": str(candidate), "has_from": False},
                message="Dockerfile has no FROM instruction",
            )
        return CheckOutcome.pass_(
            {"source": "dockerfile", "path": str(candidate), "has_from": True}
        )
    return CheckOutcome.fail(
        {"searched": ["server/Dockerfile", "Dockerfile"]},
        message="Missing Dockerfile and normalized container-image declaration",
        diagnostics=(
            ValidationDiagnostic(
                code="container_declaration_missing",
                message="Declare a container image or add a Dockerfile",
                location=DiagnosticLocation(path="server/Dockerfile"),
            ),
        ),
        remediation=(
            ValidationRemediation(
                kind="edit",
                message="Add server/Dockerfile with a supported FROM instruction",
                path="server/Dockerfile",
            ),
        ),
    )


def _custom_verifier_declaration(context: ValidationContext) -> CheckOutcome:
    subject = context.spec_load.subject
    script = subject.verifier_script if subject is not None else None
    if script is None:
        return CheckOutcome.skip(
            {"declared": False},
            message="No optional isolated verifier is declared by this spec",
            severity=ValidationSeverity.ADVISORY,
        )
    try:
        first_line = script.read_text(encoding="utf-8").splitlines()[:1]
    except OSError:
        return CheckOutcome.fail(
            {"path": str(script)}, message="Unable to read the declared verifier"
        )
    return CheckOutcome.pass_(
        {
            "declared": True,
            "path": str(script),
            "has_shebang": bool(first_line and first_line[0].startswith("#!")),
            "execution": "requires an isolated verifier runner",
        }
    )


def _legacy_runtime_criterion(
    criterion_id: str,
) -> Callable[[ValidationContext], CheckOutcome]:
    def evaluate(context: ValidationContext) -> CheckOutcome:
        probe_error = context.discovered.get("runtime_probe_error")
        if probe_error is not None:
            return CheckOutcome.error(
                {"error_type": str(probe_error)},
                message="Runtime discovery could not be completed",
            )
        report = context.discovered.get("runtime_report")
        if not isinstance(report, dict):
            return CheckOutcome.skip(
                {"runtime_url": context.runtime_url},
                message="No runtime discovery report is available",
            )
        criteria = report.get("criteria", [])
        criterion = next(
            (
                candidate
                for candidate in criteria
                if isinstance(candidate, dict) and candidate.get("id") == criterion_id
            ),
            None,
        )
        if criterion is None:
            return CheckOutcome.error(
                {"runtime_url": context.runtime_url, "criterion": criterion_id},
                message="Runtime probe omitted a required criterion",
            )
        evidence = {
            key: criterion[key] for key in ("expected", "actual") if key in criterion
        }
        if "details" in criterion:
            evidence["probe_details"] = criterion["details"]
        if criterion.get("passed") is True:
            return CheckOutcome.pass_(evidence)
        return CheckOutcome.fail(evidence, message=criterion.get("details"))

    return evaluate


def _runtime_declaration(
    name: str, *, required: bool = False
) -> Callable[[ValidationContext], CheckOutcome]:
    def evaluate(context: ValidationContext) -> CheckOutcome:
        declarations = context.discovered.get("runtime_declarations", {})
        if not isinstance(declarations, dict) or name not in declarations:
            evidence = {"declaration": name, "available": False}
            if required:
                return CheckOutcome.error(
                    evidence,
                    message=(
                        f"The capable runtime runner did not provide required {name} "
                        "validation evidence"
                    ),
                )
            return CheckOutcome.skip(
                evidence,
                message=f"The running API does not declare {name} validation data",
            )
        value = declarations[name]
        if not isinstance(value, dict) or "valid" not in value:
            return CheckOutcome.error(
                {"declaration": name, "actual_type": type(value).__name__},
                message=f"Runtime {name} evidence is malformed",
            )
        if value.get("valid") is False:
            return CheckOutcome.fail(value)
        if value.get("valid") is not True:
            return CheckOutcome.error(
                {"declaration": name, "valid": value.get("valid")},
                message=f"Runtime {name} evidence must declare valid as a boolean",
            )
        return CheckOutcome.pass_({"declaration": name, "value": value})

    return evaluate


def _mcp_control_boundary(context: ValidationContext) -> CheckOutcome:
    declarations = context.discovered.get("runtime_declarations", {})
    tools = declarations.get("tools") if isinstance(declarations, dict) else None
    names = tools.get("names") if isinstance(tools, dict) else None
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        return CheckOutcome.error(
            {"tools_evidence_available": False},
            message="Runtime discovery did not provide a valid MCP tool-name list",
        )
    exposed_controls = sorted(
        name for name in names if name.casefold() in _MCP_CONTROL_TOOL_NAMES
    )
    if exposed_controls:
        return CheckOutcome.fail(
            {"exposed_control_tools": exposed_controls},
            message="MCP must not expose infrastructure reset, step, or state controls",
        )
    return CheckOutcome.pass_({"tool_count": len(names), "exposed_control_tools": []})


def _external_evidence(
    criterion_id: str,
) -> Callable[[ValidationContext], CheckOutcome]:
    def evaluate(context: ValidationContext) -> CheckOutcome:
        value = context.discovered.get(criterion_id)
        if value is None:
            return CheckOutcome.error(
                {"criterion": criterion_id},
                message="Capable runner did not supply criterion evidence",
            )
        if isinstance(value, CheckOutcome):
            return value
        if isinstance(value, bool):
            return (
                CheckOutcome.pass_({"value": value})
                if value
                else CheckOutcome.fail({"value": value})
            )
        if isinstance(value, dict):
            if "status" not in value:
                return CheckOutcome.error(
                    {"criterion": criterion_id},
                    message="Runner evidence must declare an explicit status",
                )
            raw_status = value["status"]
            try:
                status = ValidationStatus(raw_status)
            except ValueError:
                return CheckOutcome.error(
                    {"criterion": criterion_id},
                    message="Runner supplied an invalid evidence status",
                )
            evidence = value.get("evidence", value)
            return CheckOutcome(
                status=status,
                evidence=evidence
                if isinstance(evidence, dict)
                else {"value": evidence},
                message=value.get("message"),
            )
        return CheckOutcome.error(
            {"criterion": criterion_id, "actual_type": type(value).__name__},
            message="Runner supplied unsupported criterion evidence",
        )

    return evaluate


def _check(
    criterion_id: str,
    requirement: str,
    evaluator: Callable[[ValidationContext], CheckOutcome] | None,
    *,
    capabilities: frozenset[ValidationCapability],
    profiles: frozenset[ValidationProfile],
    severity: ValidationSeverity = ValidationSeverity.BLOCKING,
    timeout_s: float = 10.0,
    built_in: bool = True,
    requirement_binding: ValidationRequirementBinding | None = None,
) -> ValidationCheck:
    return ValidationCheck(
        criterion_id=criterion_id,
        requirement=requirement,
        capabilities=capabilities,
        severity=severity,
        timeout_s=timeout_s,
        evaluator=evaluator,
        profiles=profiles,
        built_in=built_in,
        requirement_binding=requirement_binding,
    )


@lru_cache(maxsize=1)
def get_openenv_checks() -> tuple[ValidationCheck, ...]:
    """Return the deterministic served-OpenEnv policy-v1 check catalog."""
    source = frozenset({ValidationCapability.SOURCE})
    runtime = frozenset({ValidationCapability.RUNTIME})
    checks = [
        _check(
            "source.validation_spec",
            "RFC 008 § Core Abstractions: detect and load the source spec",
            _validation_spec,
            capabilities=source,
            profiles=_STATIC_PROFILES,
        ),
        _check(
            "source.validation_requirements",
            "RFC 008 § Core Abstractions: normalize execution requirements",
            _validation_requirements,
            capabilities=source,
            profiles=_STATIC_PROFILES,
        ),
        _check(
            "source.openenv_manifest",
            "RFC 008 § Local Execution: validate the OpenEnv manifest",
            _openenv_manifest,
            capabilities=source,
            profiles=_STATIC_PROFILES,
        ),
        _check(
            "source.project_layout",
            "RFC 008 § Local Execution: validate project layout and entry point",
            _project_layout,
            capabilities=source,
            profiles=_STATIC_PROFILES,
        ),
        _check(
            "source.dependencies",
            "RFC 008 § Local Execution: validate runtime dependencies",
            _dependencies,
            capabilities=source,
            profiles=_STATIC_PROFILES,
        ),
        _check(
            "source.lockfile",
            "RFC 008 § Local Execution: validate the dependency lockfile",
            _lockfile,
            capabilities=source,
            profiles=_STATIC_PROFILES,
        ),
        _check(
            "source.schema_sources",
            "RFC 008 § Local Execution: validate schema and application sources",
            _schema_sources,
            capabilities=source,
            profiles=_STATIC_PROFILES,
        ),
        _check(
            "source.dockerfile",
            "RFC 008 § Local Execution: validate the container image declaration",
            _dockerfile,
            capabilities=source,
            profiles=_STATIC_PROFILES,
        ),
        _check(
            "source.custom_verifier",
            "RFC 008 § Validation Model: discover an optional isolated verifier",
            _custom_verifier_declaration,
            capabilities=source,
            profiles=_STATIC_PROFILES,
            severity=ValidationSeverity.ADVISORY,
        ),
    ]

    runtime_descriptions = {
        "openapi_version_available": "the runtime publishes a versioned OpenAPI contract",
        "health_endpoint": "the runtime health endpoint becomes ready",
        "metadata_endpoint": "the runtime publishes environment metadata",
        "schema_endpoint": "the runtime publishes action, observation, and state schemas",
        "mcp_endpoint": "the agent MCP endpoint is reachable",
        "mode_endpoint_consistency": "runtime endpoints match the declared OpenEnv mode",
    }
    for criterion_id, description in runtime_descriptions.items():
        checks.append(
            _check(
                criterion_id,
                f"RFC 008 § Local Execution: {description}",
                _legacy_runtime_criterion(criterion_id),
                capabilities=runtime,
                profiles=_RUNTIME_PROFILES,
                timeout_s=15.0,
            )
        )
    for declaration in (
        "websocket",
        "tools",
        "tasks",
        "rewards",
        "seeds",
        "trajectories",
    ):
        checks.append(
            _check(
                f"runtime.{declaration}",
                f"RFC 008 § Local Execution: validate runtime {declaration}",
                _runtime_declaration(declaration, required=declaration == "websocket"),
                capabilities=runtime,
                profiles=_RUNTIME_PROFILES,
                severity=(
                    ValidationSeverity.BLOCKING
                    if declaration == "websocket"
                    else ValidationSeverity.ADVISORY
                ),
            )
        )

    checks.append(
        _check(
            "runtime.mcp_control_boundary",
            "OpenEnv invariant: agent MCP tools cannot expose reset, step, or state controls",
            _mcp_control_boundary,
            capabilities=runtime,
            profiles=_RUNTIME_PROFILES,
        )
    )

    checks.extend(
        [
            _check(
                "custom.verifier",
                "RFC 008 § Validation Model: execute task criteria in an isolated verifier",
                _external_evidence("custom.verifier"),
                capabilities=frozenset(
                    {
                        ValidationCapability.RUNTIME,
                        ValidationCapability.VERIFIER_ISOLATION,
                    }
                ),
                profiles=_RUNTIME_PROFILES,
                timeout_s=600.0,
                built_in=False,
                requirement_binding=ValidationRequirementBinding.VERIFIER,
            ),
            _check(
                "remote.egress_enforcement",
                "RFC 008 § Remote Runner: enforce normalized network requirements",
                _external_evidence("remote.egress_enforcement"),
                capabilities=frozenset({ValidationCapability.NETWORK_ENFORCEMENT}),
                profiles=_FULL_PROFILE,
                requirement_binding=ValidationRequirementBinding.NETWORK,
            ),
            _check(
                "remote.host_containment",
                "RFC 008 § Remote Runner: contain the subject from runner hosts",
                _external_evidence("remote.host_containment"),
                capabilities=frozenset({ValidationCapability.NETWORK_ENFORCEMENT}),
                profiles=_FULL_PROFILE,
            ),
            _check(
                "artifact.image_identity",
                "RFC 008 § Security and Results: bind results to an immutable image digest",
                _external_evidence("artifact.image_identity"),
                capabilities=frozenset({ValidationCapability.CONTAINER_IMAGE}),
                profiles=_FULL_PROFILE,
            ),
            _check(
                "artifact.sbom",
                "RFC 008 § Remote Runner: inspect the image software bill of materials",
                _external_evidence("artifact.sbom"),
                capabilities=frozenset({ValidationCapability.CONTAINER_IMAGE}),
                profiles=_FULL_PROFILE,
                severity=ValidationSeverity.ADVISORY,
            ),
            _check(
                "artifact.signature",
                "RFC 008 § Security and Results: verify report and image signatures",
                _external_evidence("artifact.signature"),
                capabilities=frozenset({ValidationCapability.SIGNATURE_VERIFICATION}),
                profiles=_FULL_PROFILE,
            ),
            _check(
                "remote.cross_host_reproducibility",
                "RFC 008 § Remote Runner: replay trajectories on an independent host",
                _external_evidence("remote.cross_host_reproducibility"),
                capabilities=frozenset({ValidationCapability.CROSS_HOST}),
                profiles=_FULL_PROFILE,
                severity=ValidationSeverity.ADVISORY,
            ),
            _check(
                "remote.reference_model",
                "RFC 008 § Remote Runner: run the pinned official reference model",
                _external_evidence("remote.reference_model"),
                capabilities=frozenset(
                    {
                        ValidationCapability.GPU,
                        ValidationCapability.REFERENCE_MODEL,
                    }
                ),
                profiles=_FULL_PROFILE,
                severity=ValidationSeverity.ADVISORY,
                timeout_s=3600.0,
            ),
        ]
    )

    criterion_ids = [check.criterion_id for check in checks]
    if len(criterion_ids) != len(set(criterion_ids)):
        raise RuntimeError(
            "The default validation registry has duplicate criterion IDs"
        )
    return tuple(checks)
