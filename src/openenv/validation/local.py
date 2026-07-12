# SPDX-License-Identifier: BSD-3-Clause

"""Local/static and runtime-capable validation execution."""

from __future__ import annotations

import errno
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator
from urllib.parse import urlparse

import requests
import yaml

from .executor import execute_validation_plan
from .models import (
    RunnerCapabilities,
    ValidationCapability,
    ValidationPolicy,
    ValidationProfile,
    ValidationReport,
)
from .planner import build_validation_plan
from .runtime_probe import _normalize_runtime_url, validate_running_environment
from .specs import (
    DEFAULT_SPEC_REGISTRY,
    ExecutionModel,
    runtime_openenv_spec_load,
    SpecLoad,
    ValidationSpecRegistry,
)


def _looks_like_url(value: str) -> bool:
    lowered = value.strip().lower()
    return lowered.startswith("http://") or lowered.startswith("https://")


def _runtime_configuration(env_path: Path) -> tuple[str, int]:
    manifest_path = env_path / "openenv.yaml"
    try:
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError("Unable to load openenv.yaml for local runtime") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("openenv.yaml must contain a mapping")
    app = manifest.get("app")
    port = manifest.get("port", 8000)
    if not isinstance(app, str) or not app.strip():
        raise RuntimeError("openenv.yaml must declare an app module")
    if not isinstance(port, int) or not 0 < port < 65536:
        raise RuntimeError("openenv.yaml port must be a valid integer")
    return app, port


def _runtime_environment(
    env_path: Path, *, temporary_home: Path | None = None
) -> dict[str, str]:
    """Build a minimal environment that does not forward ambient credentials."""
    safe_names = {
        "HOME",
        "LANG",
        "LC_ALL",
        "PATH",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "VIRTUAL_ENV",
    }
    process_env = {
        name: value for name, value in os.environ.items() if name in safe_names
    }
    if temporary_home is not None:
        process_env["HOME"] = str(temporary_home)
    source_root = Path(__file__).resolve().parents[2]
    python_path = [str(env_path), str(env_path.parent), str(source_root)]
    existing_python_path = os.environ.get("PYTHONPATH")
    if existing_python_path:
        python_path.append(existing_python_path)
    process_env["PYTHONPATH"] = os.pathsep.join(python_path)
    process_env["OPENENV_VALIDATION"] = "1"
    return process_env


def _ensure_port_available(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.25)
        result = probe.connect_ex(("127.0.0.1", port))
    if result == 0:
        raise RuntimeError(f"Local runtime port {port} is already in use")
    if result in {errno.EACCES, errno.EPERM}:
        raise PermissionError(f"Cannot inspect local runtime port {port}")


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if os.name == "posix":
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
    elif process.poll() is None:
        process.terminate()
    if process.poll() is None:
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=5.0)
    if os.name == "posix":
        try:
            os.killpg(process.pid, 0)
        except (PermissionError, ProcessLookupError):
            return
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    elif process.poll() is None:
        process.kill()
    if process.poll() is None:
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=5.0)


def _server_command(env_path: Path, app: str, port: int) -> list[str]:
    command = [
        "python",
        "-m",
        "uvicorn",
        app,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]
    uv = shutil.which("uv")
    if uv is not None:
        return [uv, "run", "--project", str(env_path), *command]
    command[0] = sys.executable
    return command


def _contains_property(document: dict[str, Any], node: Any, name: str) -> bool:
    seen_refs: set[str] = set()

    def visit(value: Any) -> bool:
        if isinstance(value, dict):
            reference = value.get("$ref")
            if isinstance(reference, str) and reference.startswith("#/"):
                if reference in seen_refs:
                    return False
                seen_refs.add(reference)
                resolved: Any = document
                for part in reference[2:].split("/"):
                    if not isinstance(resolved, dict) or part not in resolved:
                        return False
                    resolved = resolved[part]
                if visit(resolved):
                    return True
            properties = value.get("properties")
            if isinstance(properties, dict) and name in properties:
                return True
            return any(visit(item) for key, item in value.items() if key != "$ref")
        if isinstance(value, list):
            return any(visit(item) for item in value)
        return False

    return visit(node)


def _discover_runtime_declarations(
    runtime_url: str, timeout_s: float
) -> dict[str, Any]:
    declarations: dict[str, Any] = {}
    try:
        openapi_response = requests.get(
            f"{runtime_url}/openapi.json",
            timeout=timeout_s,
            allow_redirects=False,
        )
        openapi = openapi_response.json()
    except (requests.RequestException, ValueError):
        openapi = None
    try:
        schema_response = requests.get(
            f"{runtime_url}/schema",
            timeout=timeout_s,
            allow_redirects=False,
        )
        schemas = schema_response.json()
    except (requests.RequestException, ValueError):
        schemas = None

    if isinstance(openapi, dict):
        paths = openapi.get("paths", {})
        if isinstance(paths, dict):
            task_paths = sorted(
                path
                for path in paths
                if path == "/list_environments"
                or path.endswith("/tasks")
                or path.endswith("/num_tasks")
            )
            if task_paths:
                declarations["tasks"] = {"valid": True, "paths": task_paths}
            declarations["seeds"] = {
                "valid": _contains_property(openapi, paths.get("/reset"), "seed"),
                "endpoint": "/reset",
            }
            trajectory_paths = sorted(
                path
                for path in paths
                if "trajectory" in path.lower() or "episode" in path.lower()
            )
            if trajectory_paths:
                declarations["trajectories"] = {
                    "valid": True,
                    "paths": trajectory_paths,
                }

    if isinstance(schemas, dict):
        declarations["rewards"] = {
            "valid": _contains_property(
                schemas, schemas.get("observation", {}), "reward"
            ),
            "schema": "observation.reward",
        }

    try:
        tools_response = requests.post(
            f"{runtime_url}/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            timeout=timeout_s,
            allow_redirects=False,
        )
        tools_payload = tools_response.json()
    except (requests.RequestException, ValueError):
        tools_payload = None
    if isinstance(tools_payload, dict):
        result = tools_payload.get("result")
        tools = result.get("tools") if isinstance(result, dict) else None
        if isinstance(tools, list):
            names = [tool.get("name") for tool in tools if isinstance(tool, dict)]
            valid = len(names) == len(tools) and all(
                isinstance(name, str) and name for name in names
            )
            declarations["tools"] = {
                "valid": valid and len(set(names)) == len(names),
                "count": len(tools),
                "names": names,
            }

    try:
        from websockets.sync.client import connect as ws_sync_connect

        parsed = urlparse(runtime_url)
        websocket_scheme = "wss" if parsed.scheme == "https" else "ws"
        websocket_url = f"{websocket_scheme}://{parsed.netloc}/ws"
        with ws_sync_connect(websocket_url, open_timeout=timeout_s):
            pass
    except Exception as exc:
        declarations["websocket"] = {
            "valid": False,
            "error_type": type(exc).__name__,
        }
    else:
        declarations["websocket"] = {"valid": True, "endpoint": "/ws"}
    return declarations


@contextmanager
def _launch_environment(env_path: Path, timeout_s: float) -> Iterator[str]:
    app, port = _runtime_configuration(env_path)
    base_url = f"http://127.0.0.1:{port}"
    _ensure_port_available(port)
    with TemporaryDirectory(prefix="openenv-validation-") as temporary_home:
        process_env = _runtime_environment(
            env_path, temporary_home=Path(temporary_home)
        )
        process = subprocess.Popen(
            _server_command(env_path, app, port),
            cwd=env_path,
            env=process_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + timeout_s
            health_url = f"{base_url}/health"
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Local environment server exited with code {process.returncode}"
                    )
                try:
                    response = requests.get(
                        health_url,
                        timeout=min(1.0, timeout_s),
                        allow_redirects=False,
                    )
                except requests.RequestException:
                    time.sleep(0.1)
                    continue
                if response.status_code == 200:
                    break
                time.sleep(0.1)
            else:
                raise TimeoutError(
                    f"Local environment did not become ready within {timeout_s:g}s"
                )
            yield base_url
        finally:
            _stop_process(process)


def _run(
    target: str | Path,
    *,
    profile: ValidationProfile,
    source_path: Path | None,
    spec_load: SpecLoad,
    policy: ValidationPolicy | None,
    runtime_url: str | None,
    runtime_error: str | None,
    runtime_unavailable_reason: str | None,
    timeout_s: float,
) -> ValidationReport:
    discovered: dict[str, Any] = {}
    available: set[ValidationCapability] = set()
    if source_path is not None:
        available.add(ValidationCapability.SOURCE)
    if runtime_url is not None or (
        runtime_error is not None and runtime_unavailable_reason is None
    ):
        available.add(ValidationCapability.RUNTIME)
    if runtime_url is not None:
        try:
            discovered["runtime_report"] = validate_running_environment(
                runtime_url, timeout_s=timeout_s
            )
        except Exception as exc:
            discovered["runtime_probe_error"] = type(exc).__name__
        else:
            try:
                discovered["runtime_declarations"] = _discover_runtime_declarations(
                    runtime_url, timeout_s
                )
            except Exception as exc:
                discovered["runtime_declarations_error"] = type(exc).__name__
    elif runtime_error is not None:
        discovered["runtime_probe_error"] = runtime_error
    if runtime_unavailable_reason is not None:
        discovered["capability_unavailable_reasons"] = {
            ValidationCapability.RUNTIME.value: runtime_unavailable_reason
        }

    capabilities = RunnerCapabilities(
        runner="local",
        available=frozenset(available),
        official=False,
        isolation_mode=None,
    )
    plan, context = build_validation_plan(
        source_path or str(target),
        profile=profile,
        capabilities=capabilities,
        spec_load=spec_load,
        policy=policy,
        runtime_url=runtime_url,
        discovered=discovered,
    )
    return execute_validation_plan(plan, context)


def run_local_validation(
    target: str | Path,
    *,
    profile: ValidationProfile | str = ValidationProfile.STATIC,
    runtime_url: str | None = None,
    timeout_s: float = 5.0,
    spec_id: str | None = None,
    spec_registry: ValidationSpecRegistry | None = None,
    policy: ValidationPolicy | None = None,
) -> ValidationReport:
    """Run the selected profile locally and never claim remote certification."""
    selected_profile = (
        profile
        if isinstance(profile, ValidationProfile)
        else ValidationProfile(profile)
    )
    raw_target = str(target)
    target_is_url = _looks_like_url(raw_target)
    if target_is_url and selected_profile is ValidationProfile.STATIC:
        raise ValueError("The static profile requires a local source directory")
    source_path = None if target_is_url else Path(target)
    if source_path is not None:
        source_path = source_path.resolve()
        if not source_path.is_dir():
            raise ValueError("Validation target must be an existing source directory")
    effective_runtime_url = runtime_url or (raw_target if target_is_url else None)
    if effective_runtime_url is not None:
        effective_runtime_url = _normalize_runtime_url(effective_runtime_url)

    if source_path is None:
        if spec_id not in {None, "openenv"} or spec_registry is not None:
            raise ValueError("Runtime URLs currently support only the OpenEnv spec")
        spec_load = runtime_openenv_spec_load()
    else:
        registry = spec_registry or DEFAULT_SPEC_REGISTRY
        spec_load = registry.resolve(source_path, spec_id=spec_id)

    if selected_profile is ValidationProfile.STATIC:
        return _run(
            target,
            profile=selected_profile,
            source_path=source_path,
            spec_load=spec_load,
            policy=policy,
            runtime_url=None,
            runtime_error=None,
            runtime_unavailable_reason=None,
            timeout_s=timeout_s,
        )

    subject = spec_load.subject
    execution_model = subject.spec.execution_model if subject is not None else None
    if subject is None or execution_model is not ExecutionModel.SERVED:
        reason = (
            "source_spec_unavailable"
            if execution_model is None
            else f"unsupported_execution_model:{execution_model.value}"
        )
        return _run(
            target,
            profile=selected_profile,
            source_path=source_path,
            spec_load=spec_load,
            policy=policy,
            runtime_url=None,
            runtime_error=None,
            runtime_unavailable_reason=reason,
            timeout_s=timeout_s,
        )

    if effective_runtime_url is not None:
        return _run(
            target,
            profile=selected_profile,
            source_path=source_path,
            spec_load=spec_load,
            policy=policy,
            runtime_url=effective_runtime_url,
            runtime_error=None,
            runtime_unavailable_reason=None,
            timeout_s=timeout_s,
        )

    if source_path is None:
        return _run(
            target,
            profile=selected_profile,
            source_path=None,
            spec_load=spec_load,
            policy=policy,
            runtime_url=None,
            runtime_error=None,
            runtime_unavailable_reason="runtime_target_unavailable",
            timeout_s=timeout_s,
        )

    try:
        with _launch_environment(source_path, timeout_s) as launched_url:
            return _run(
                target,
                profile=selected_profile,
                source_path=source_path,
                spec_load=spec_load,
                policy=policy,
                runtime_url=launched_url,
                runtime_error=None,
                runtime_unavailable_reason=None,
                timeout_s=timeout_s,
            )
    except Exception as exc:
        return _run(
            target,
            profile=selected_profile,
            source_path=source_path,
            spec_load=spec_load,
            policy=policy,
            runtime_url=None,
            runtime_error=type(exc).__name__,
            runtime_unavailable_reason=None,
            timeout_s=timeout_s,
        )


def format_shared_validation_report(
    report: ValidationReport, *, verbose: bool = False
) -> str:
    """Render a concise human-readable shared report."""
    marker = "OK" if report.passed else "FAIL"
    lines = [
        f"[{marker}] {report.target}: {report.profile.value} validation",
        f"Policy: {report.policy_version}",
    ]
    identity = report.spec.spec if report.spec is not None else report.spec_identity
    if identity is not None:
        lines.append(
            f"Spec: {identity.spec_id} {identity.spec_version or 'unknown'} "
            f"({identity.execution_model.value}; "
            f"adapter {identity.adapter.adapter_id}@{identity.adapter.adapter_version})"
        )
    for result in report.results:
        lines.append(
            f"  [{result.status.value.upper()}] {result.criterion_id} "
            f"({result.severity.value})"
        )
        if result.message:
            lines.append(f"    {result.message}")
        for diagnostic in result.diagnostics:
            safe_diagnostic = diagnostic.to_dict()
            location = ""
            if diagnostic.location is not None:
                location = diagnostic.location.path
                if diagnostic.location.pointer is not None:
                    location += diagnostic.location.pointer
                if diagnostic.location.line is not None:
                    location += f":{diagnostic.location.line}"
                location = f" ({location})"
            lines.append(
                f"    Diagnostic [{diagnostic.code}]{location}: "
                f"{safe_diagnostic['message']}"
            )
        for remediation in result.remediation:
            safe_remediation = remediation.to_dict()
            lines.append(f"    Fix: {safe_remediation['message']}")
            if remediation.argv:
                lines.append(f"      $ {shlex.join(remediation.argv)}")
            elif remediation.path is not None:
                destination = remediation.path
                if remediation.pointer is not None:
                    destination += remediation.pointer
                lines.append(f"      Edit: {destination}")
            elif remediation.url is not None:
                lines.append(f"      Docs: {remediation.url}")
        if verbose and result.evidence:
            rendered_evidence = json.dumps(
                result.evidence, indent=2, sort_keys=True, default=str
            )
            lines.append("    Evidence:")
            lines.extend(f"      {line}" for line in rendered_evidence.splitlines())
    if report.profile is ValidationProfile.PUBLISH and not report.passed:
        blockers = report.to_dict()["summary"]
        blocking_ids = [
            *blockers["blocking_failed_criteria"],
            *blockers["blocking_skipped_criteria"],
        ]
        if blocking_ids:
            lines.append(f"Publish blockers: {', '.join(blocking_ids)}")
    lines.append("Certification: not claimed by author validation")
    return "\n".join(lines)
