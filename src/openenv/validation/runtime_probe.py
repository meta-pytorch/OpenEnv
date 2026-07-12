# SPDX-License-Identifier: BSD-3-Clause

"""Runtime URL normalization and legacy API conformance probing."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import requests


def _make_criterion(
    criterion_id: str,
    description: str,
    passed: bool,
    *,
    required: bool = True,
    details: str | None = None,
    expected: Any | None = None,
    actual: Any | None = None,
) -> dict[str, Any]:
    """Create a standard criterion result payload."""
    criterion: dict[str, Any] = {
        "id": criterion_id,
        "description": description,
        "passed": passed,
        "required": required,
    }
    if details is not None:
        criterion["details"] = details
    if expected is not None:
        criterion["expected"] = expected
    if actual is not None:
        criterion["actual"] = actual
    return criterion


def _normalize_runtime_url(base_url: str) -> str:
    """Normalize and validate a runtime target URL."""
    target = base_url.strip()
    if not target:
        raise ValueError("Runtime URL cannot be empty")

    if "://" not in target:
        target = f"http://{target}"

    try:
        parsed = urlparse(target)
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError as exc:
        raise ValueError(f"Invalid runtime URL: {base_url}") from exc
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.netloc
        or not hostname
    ):
        raise ValueError(f"Invalid runtime URL: {base_url}")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Runtime URL must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("Runtime URL must not contain a query string or fragment")

    return target.rstrip("/")


def _runtime_standard_profile(api_version: str) -> str:
    """Resolve the runtime standard profile for an API version."""
    if api_version.startswith("1."):
        return "openenv-http/1.x"
    return "openenv-http/unknown"


def _build_summary(criteria: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a compact pass/fail summary for a criteria list."""
    total_count = len(criteria)
    passed_count = sum(1 for criterion in criteria if criterion.get("passed", False))
    failed_criteria = [
        criterion.get("id", "unknown")
        for criterion in criteria
        if not criterion.get("passed", False)
    ]
    required_criteria = [
        criterion for criterion in criteria if criterion.get("required", True)
    ]
    required_total_count = len(required_criteria)
    required_passed_count = sum(
        1 for criterion in required_criteria if criterion.get("passed", False)
    )

    return {
        "passed_count": passed_count,
        "total_count": total_count,
        "failed_criteria": failed_criteria,
        "required_passed_count": required_passed_count,
        "required_total_count": required_total_count,
    }


def validate_running_environment(
    base_url: str, timeout_s: float = 5.0
) -> dict[str, Any]:
    """
    Validate a running OpenEnv server against runtime API standards.

    The returned JSON report contains an overall pass/fail result and
    per-criterion outcomes that can be consumed in CI.
    """
    normalized_url = _normalize_runtime_url(base_url)
    criteria: list[dict[str, Any]] = []

    report: dict[str, Any] = {
        "target": normalized_url,
        "validation_type": "running_environment",
        "standard_version": "unknown",
        "standard_profile": "openenv-http/unknown",
        "mode": "unknown",
        "passed": False,
        "summary": {},
        "criteria": criteria,
    }

    openapi_paths: dict[str, Any] = {}
    api_version = "unknown"

    # Criterion: OpenAPI endpoint reachable with a declared version.
    try:
        openapi_response = requests.get(
            f"{normalized_url}/openapi.json",
            timeout=timeout_s,
            allow_redirects=False,
        )
    except requests.RequestException as exc:
        criteria.append(
            _make_criterion(
                "openapi_version_available",
                "GET /openapi.json returns OpenAPI info.version",
                False,
                details=f"Request failed: {type(exc).__name__}: {exc}",
                expected={"status_code": 200, "info.version": "string"},
            )
        )
    else:
        try:
            openapi_json = openapi_response.json()
        except ValueError:
            openapi_json = None

        openapi_ok = (
            openapi_response.status_code == 200
            and isinstance(openapi_json, dict)
            and isinstance(openapi_json.get("info"), dict)
            and isinstance(openapi_json["info"].get("version"), str)
        )

        if openapi_ok:
            api_version = str(openapi_json["info"]["version"])
            openapi_paths = openapi_json.get("paths", {})
            criteria.append(
                _make_criterion(
                    "openapi_version_available",
                    "GET /openapi.json returns OpenAPI info.version",
                    True,
                    expected={"status_code": 200, "info.version": "string"},
                    actual={
                        "status_code": openapi_response.status_code,
                        "info.version": api_version,
                    },
                )
            )
        else:
            criteria.append(
                _make_criterion(
                    "openapi_version_available",
                    "GET /openapi.json returns OpenAPI info.version",
                    False,
                    details="Response missing required OpenAPI info.version field",
                    expected={"status_code": 200, "info.version": "string"},
                    actual={
                        "status_code": openapi_response.status_code,
                        "body_type": (
                            type(openapi_json).__name__
                            if openapi_json is not None
                            else "non_json"
                        ),
                    },
                )
            )

    report["standard_version"] = api_version
    report["standard_profile"] = _runtime_standard_profile(api_version)

    # Criterion: Health endpoint.
    try:
        health_response = requests.get(
            f"{normalized_url}/health",
            timeout=timeout_s,
            allow_redirects=False,
        )
    except requests.RequestException as exc:
        criteria.append(
            _make_criterion(
                "health_endpoint",
                "GET /health returns healthy status",
                False,
                details=f"Request failed: {type(exc).__name__}: {exc}",
                expected={"status_code": 200, "status": "healthy"},
            )
        )
    else:
        try:
            health_json = health_response.json()
        except ValueError:
            health_json = None

        health_ok = (
            health_response.status_code == 200
            and isinstance(health_json, dict)
            and health_json.get("status") == "healthy"
        )
        criteria.append(
            _make_criterion(
                "health_endpoint",
                "GET /health returns healthy status",
                health_ok,
                expected={"status_code": 200, "status": "healthy"},
                actual={
                    "status_code": health_response.status_code,
                    "status": (
                        health_json.get("status")
                        if isinstance(health_json, dict)
                        else None
                    ),
                },
            )
        )

    # Criterion: Metadata endpoint has required fields.
    try:
        metadata_response = requests.get(
            f"{normalized_url}/metadata",
            timeout=timeout_s,
            allow_redirects=False,
        )
    except requests.RequestException as exc:
        criteria.append(
            _make_criterion(
                "metadata_endpoint",
                "GET /metadata returns name and description",
                False,
                details=f"Request failed: {type(exc).__name__}: {exc}",
                expected={"status_code": 200, "fields": ["name", "description"]},
            )
        )
    else:
        try:
            metadata_json = metadata_response.json()
        except ValueError:
            metadata_json = None

        metadata_ok = (
            metadata_response.status_code == 200
            and isinstance(metadata_json, dict)
            and isinstance(metadata_json.get("name"), str)
            and isinstance(metadata_json.get("description"), str)
        )
        criteria.append(
            _make_criterion(
                "metadata_endpoint",
                "GET /metadata returns name and description",
                metadata_ok,
                expected={"status_code": 200, "fields": ["name", "description"]},
                actual={
                    "status_code": metadata_response.status_code,
                    "name": (
                        metadata_json.get("name")
                        if isinstance(metadata_json, dict)
                        else None
                    ),
                    "description": (
                        metadata_json.get("description")
                        if isinstance(metadata_json, dict)
                        else None
                    ),
                },
            )
        )

    # Criterion: Schema endpoint returns action/observation/state.
    try:
        schema_response = requests.get(
            f"{normalized_url}/schema",
            timeout=timeout_s,
            allow_redirects=False,
        )
    except requests.RequestException as exc:
        criteria.append(
            _make_criterion(
                "schema_endpoint",
                "GET /schema returns action, observation, and state schemas",
                False,
                details=f"Request failed: {type(exc).__name__}: {exc}",
                expected={
                    "status_code": 200,
                    "fields": ["action", "observation", "state"],
                },
            )
        )
    else:
        try:
            schema_json = schema_response.json()
        except ValueError:
            schema_json = None

        schema_ok = (
            schema_response.status_code == 200
            and isinstance(schema_json, dict)
            and isinstance(schema_json.get("action"), dict)
            and isinstance(schema_json.get("observation"), dict)
            and isinstance(schema_json.get("state"), dict)
        )
        criteria.append(
            _make_criterion(
                "schema_endpoint",
                "GET /schema returns action, observation, and state schemas",
                schema_ok,
                expected={
                    "status_code": 200,
                    "fields": ["action", "observation", "state"],
                },
                actual={
                    "status_code": schema_response.status_code,
                    "has_action": (
                        isinstance(schema_json.get("action"), dict)
                        if isinstance(schema_json, dict)
                        else False
                    ),
                    "has_observation": (
                        isinstance(schema_json.get("observation"), dict)
                        if isinstance(schema_json, dict)
                        else False
                    ),
                    "has_state": (
                        isinstance(schema_json.get("state"), dict)
                        if isinstance(schema_json, dict)
                        else False
                    ),
                },
            )
        )

    # Criterion: MCP endpoint is reachable.
    try:
        mcp_response = requests.post(
            f"{normalized_url}/mcp",
            json={},
            timeout=timeout_s,
            allow_redirects=False,
        )
    except requests.RequestException as exc:
        criteria.append(
            _make_criterion(
                "mcp_endpoint",
                "POST /mcp is reachable and returns JSON-RPC payload",
                False,
                details=f"Request failed: {type(exc).__name__}: {exc}",
                expected={"status_code": 200, "jsonrpc": "2.0"},
            )
        )
    else:
        try:
            mcp_json = mcp_response.json()
        except ValueError:
            mcp_json = None

        mcp_ok = (
            mcp_response.status_code == 200
            and isinstance(mcp_json, dict)
            and mcp_json.get("jsonrpc") == "2.0"
        )
        criteria.append(
            _make_criterion(
                "mcp_endpoint",
                "POST /mcp is reachable and returns JSON-RPC payload",
                mcp_ok,
                expected={"status_code": 200, "jsonrpc": "2.0"},
                actual={
                    "status_code": mcp_response.status_code,
                    "jsonrpc": (
                        mcp_json.get("jsonrpc") if isinstance(mcp_json, dict) else None
                    ),
                },
            )
        )

    # Criterion: mode endpoint contract consistency via OpenAPI paths.
    if isinstance(openapi_paths, dict) and openapi_paths:
        has_reset = "/reset" in openapi_paths
        has_step = "/step" in openapi_paths
        has_state = "/state" in openapi_paths

        if has_reset:
            report["mode"] = "simulation"
            mode_ok = has_step and has_state
            expected_paths = {"/reset": True, "/step": True, "/state": True}
        else:
            report["mode"] = "production"
            mode_ok = not has_step and not has_state
            expected_paths = {"/reset": False, "/step": False, "/state": False}

        criteria.append(
            _make_criterion(
                "mode_endpoint_consistency",
                "OpenAPI endpoint set matches OpenEnv mode contract",
                mode_ok,
                expected=expected_paths,
                actual={
                    "/reset": has_reset,
                    "/step": has_step,
                    "/state": has_state,
                },
            )
        )
    else:
        criteria.append(
            _make_criterion(
                "mode_endpoint_consistency",
                "OpenAPI endpoint set matches OpenEnv mode contract",
                False,
                details="Cannot determine mode without OpenAPI paths",
                expected={"openapi.paths": "present"},
                actual={"openapi.paths": "missing"},
            )
        )

    report["passed"] = all(
        criterion["passed"] for criterion in criteria if criterion.get("required", True)
    )
    report["summary"] = _build_summary(criteria)
    return report
