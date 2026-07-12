# SPDX-License-Identifier: BSD-3-Clause

"""Bounded, credential-safe serialization for validation artifacts."""

from __future__ import annotations

import json
import math
import re
from enum import Enum
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


_SENSITIVE_MAPPING_KEY = re.compile(
    r"(?i)(authorization|cookie|credential|password|passwd|private.?key|secret|"
    r"token|api.?key|access.?key|(?:^|[_-])auth(?:$|[_-])|"
    r"(?:^|[_-])jwt(?:$|[_-])|(?:^|[_-])pat(?:$|[_-])|askpass)"
)
_SENSITIVE_QUERY_KEYS = frozenset(
    {
        "access_key",
        "apikey",
        "api_key",
        "auth",
        "authorization",
        "awsaccesskeyid",
        "credential",
        "googleaccessid",
        "key",
        "password",
        "secret",
        "security_token",
        "sig",
        "signature",
        "token",
        "x_amz_credential",
        "x_amz_security_token",
        "x_amz_signature",
        "x_goog_credential",
        "x_goog_signature",
    }
)
_SECRET_VALUE_PATTERNS = (
    re.compile(r"(?i)(bearer\s+)[^\s,;]+"),
    re.compile(
        r"\b(?:hf_|sk-|github_pat_|gh[pousr]_)[A-Za-z0-9_-]{8,}\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?"
        r"(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)",
        re.DOTALL,
    ),
)
_URI_VALUE = re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s<>'\"]+")
_URI_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*$", re.IGNORECASE)
_MAX_DEPTH = 32
_MAX_NODES = 10_000
_MAX_STRING_LENGTH = 1_000_000


def _sensitive_query_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    return (
        normalized in _SENSITIVE_QUERY_KEYS
        or _SENSITIVE_MAPPING_KEY.search(key) is not None
    )


def _redact_uri(value: str) -> str:
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except (UnicodeError, ValueError):
        return "[REDACTED_INVALID_URL]"
    if not _URI_SCHEME.fullmatch(parsed.scheme) or not hostname:
        return "[REDACTED_INVALID_URL]"
    host = hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if port is not None:
        host = f"{host}:{port}"
    query = [
        (key, "[REDACTED]" if _sensitive_query_key(key) else item)
        for key, item in parse_qsl(parsed.query, keep_blank_values=True)
    ]
    return urlunsplit((parsed.scheme, host, parsed.path, urlencode(query), ""))


def redact_string(value: str) -> str:
    """Redact credential-like tokens and URI user information from text."""
    redacted = value
    for pattern in _SECRET_VALUE_PATTERNS:
        redacted = pattern.sub(
            lambda match: (
                f"{match.group(1)}[REDACTED]" if match.lastindex else "[REDACTED]"
            ),
            redacted,
        )
    return _URI_VALUE.sub(lambda match: _redact_uri(match.group(0)), redacted)


def json_safe(
    value: Any,
    *,
    key: str | None = None,
    seen: set[int] | None = None,
    depth: int = 0,
    budget: list[int] | None = None,
    trusted_string_keys: frozenset[str] = frozenset(),
) -> Any:
    """Return a bounded JSON-safe value with credential-bearing fields redacted."""
    if depth > _MAX_DEPTH:
        raise ValueError("Value exceeds the maximum nesting depth")
    if budget is None:
        budget = [_MAX_NODES]
    budget[0] -= 1
    if budget[0] < 0:
        raise ValueError("Value exceeds the maximum node count")
    if key is not None and _SENSITIVE_MAPPING_KEY.search(key):
        return "[REDACTED]"
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Value contains a non-finite float")
        return value
    if isinstance(value, str):
        if len(value) > _MAX_STRING_LENGTH:
            raise ValueError("Value string exceeds the maximum length")
        if key in trusted_string_keys:
            return value
        return redact_string(value)
    if isinstance(value, Enum):
        return json_safe(
            value.value,
            seen=seen,
            depth=depth + 1,
            budget=budget,
            trusted_string_keys=trusted_string_keys,
        )
    if isinstance(value, Path):
        return json_safe(
            str(value),
            seen=seen,
            depth=depth + 1,
            budget=budget,
            trusted_string_keys=trusted_string_keys,
        )
    if seen is None:
        seen = set()
    container_id = id(value)
    if isinstance(value, (Mapping, list, tuple, set, frozenset)):
        if container_id in seen:
            raise ValueError("Value contains a reference cycle")
        seen.add(container_id)
    if isinstance(value, Mapping):
        items: list[tuple[str, Any]] = []
        for index, (item_key, item) in enumerate(value.items()):
            if index >= _MAX_NODES:
                raise ValueError("Value mapping exceeds the maximum size")
            if not isinstance(item_key, (str, int, float, bool)):
                raise ValueError("Value mapping contains an unsupported key")
            if isinstance(item_key, float) and not math.isfinite(item_key):
                raise ValueError("Value mapping contains a non-finite key")
            key_text = str(item_key)
            if len(key_text) > _MAX_STRING_LENGTH:
                raise ValueError("Value key exceeds the maximum length")
            items.append((key_text, item))
        items.sort(key=lambda pair: pair[0])
        result: dict[str, Any] = {}
        for item_key, item in items:
            if item_key in result:
                raise ValueError("Value mapping contains duplicate JSON keys")
            result[item_key] = json_safe(
                item,
                key=item_key,
                seen=seen,
                depth=depth + 1,
                budget=budget,
                trusted_string_keys=trusted_string_keys,
            )
        seen.remove(container_id)
        return result
    if isinstance(value, (list, tuple)):
        result = [
            json_safe(
                item,
                seen=seen,
                depth=depth + 1,
                budget=budget,
                trusted_string_keys=trusted_string_keys,
            )
            for item in value
        ]
        seen.remove(container_id)
        return result
    if isinstance(value, (set, frozenset)):
        converted = [
            json_safe(
                item,
                seen=seen,
                depth=depth + 1,
                budget=budget,
                trusted_string_keys=trusted_string_keys,
            )
            for item in value
        ]
        seen.remove(container_id)
        return sorted(converted, key=lambda item: json.dumps(item, sort_keys=True))
    return {"type": type(value).__name__}
