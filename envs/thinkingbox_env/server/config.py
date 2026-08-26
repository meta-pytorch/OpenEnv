"""Resolve trusted runtime settings and immutable benchmark identities.

Operational defaults remain environment-variable driven, while installed
ThinkingBox provenance is derived from PEP 610 metadata instead of duplicated
source constants.
"""

import copy
import hashlib
import importlib.util
import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import yaml
from thinkingbox.common.config_types import ConfigFile, SessionProxyConfig
from thinkingbox.common.credential_factory import create_credential
from thinkingbox.common.mcp_proxy_client import MCPProxyClient


DATA_RELEASE_NAME = "thinkingbox-bench-v1.0"
DATA_COMMIT = "fcaba4c1a9debec42fda7f15bf29fe6d6b46c431"
DATA_BUNDLE_SHA256 = "16803d95d161ca06309d95f0dbbcbbb7b3fdf8a61e2b4eacfe920781b12f18ff"
DATA_MANIFEST_PATH = "releases/thinkingbox_bench_v1/testlist_thinkingbox_bench_v1.yaml"
DATA_ARCHIVE_URL = (
    f"https://codeload.github.com/microsoft/thinkingbox-data/tar.gz/{DATA_COMMIT}"
)

SESSION_PROXY_URL = os.environ.get("OPENENV_TB_PROXY_URL", "http://127.0.0.1:7111")
DATASET_PATH = os.environ.get("OPENENV_TB_DATASET", "")
AGENT = os.environ.get("OPENENV_TB_AGENT", "think")
THINKINGBOX_CONFIG = os.environ.get("OPENENV_TB_CONFIG", "")
PROXY_TIMEOUT = float(os.environ.get("OPENENV_TB_PROXY_TIMEOUT", "120"))
DATA_TIMEOUT = float(os.environ.get("OPENENV_TB_DATA_TIMEOUT", "120"))
MAX_CONCURRENT_ENVS = int(os.environ.get("OPENENV_TB_MAX_CONCURRENT_ENVS", "8"))

_cache_home = Path(
    os.environ.get(
        "XDG_CACHE_HOME",
        str(Path.home() / ".cache"),
    )
)
DATA_CACHE = Path(
    os.environ.get(
        "OPENENV_TB_DATA_CACHE",
        str(_cache_home / "openenv" / "thinkingbox_bench"),
    )
)


@dataclass(frozen=True)
class ThinkingBoxRuntimeProvenance:
    """Describe the installed ThinkingBox package without exposing local paths.

    Args:
        identity (`str`):
            Commit identity for VCS installs or a deterministic source hash
            identity for local and editable installs.
        version (`str`):
            Installed ThinkingBox distribution version.
        source_type (`str`):
            PEP 610 source category, such as `vcs`, `editable`, or `installed`.
        source_sha256 (`str`):
            Deterministic hash of the installed `thinkingbox` package tree.
        commit_id (`str`, *optional*):
            PEP 610 VCS commit identifier when available.
        requested_revision (`str`, *optional*):
            Revision requested by the installer when available.
        source_url (`str`, *optional*):
            Credential-free VCS source URL when available.
        editable (`bool`, *optional*, defaults to `False`):
            Whether PEP 610 marks the local source as editable.
        canonical (`bool`, *optional*, defaults to `False`):
            Whether the installed VCS source exactly matches this package's
            pinned dependency.
    """

    identity: str
    version: str
    source_type: str
    source_sha256: str
    commit_id: str | None = None
    requested_revision: str | None = None
    source_url: str | None = None
    editable: bool = False
    canonical: bool = False


@dataclass(frozen=True)
class RuntimeSettings:
    """Hold private native runtime settings for one episode.

    Args:
        proxy (`thinkingbox.common.config_types.SessionProxyConfig`):
            Session Proxy connection and authentication settings.
        judge_model (`object`, *optional*):
            Native judge-model configuration.
        judge_type (`str`, *optional*, defaults to `"legacy"`):
            Native ThinkingBox judge implementation.
        user_model (`object`, *optional*):
            Native simulated-user model configuration.
        user_can_end_conversation (`bool`, *optional*, defaults to `False`):
            Whether the simulated user can terminate the episode.
        config_sha256 (`str`, *optional*):
            Fingerprint of the exact native YAML payload.
    """

    proxy: SessionProxyConfig
    judge_model: Any | None = None
    judge_type: str = "legacy"
    user_model: Any | None = None
    user_can_end_conversation: bool = False
    config_sha256: str | None = None


def _normalized_vcs_url(value: str) -> str:
    raw = value.removeprefix("git+")
    parsed = urlsplit(raw)
    host = parsed.hostname or ""
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunsplit(
        (
            parsed.scheme.casefold(),
            host.casefold(),
            parsed.path.rstrip("/"),
            "",
            "",
        )
    )


def _parse_thinkingbox_requirement(requirement: str) -> tuple[str, str] | None:
    match = re.match(
        r"^\s*thinkingbox\s*@\s*git\+(.+)@([0-9a-fA-F]{40})(?:\s*;.*)?\s*$",
        requirement,
    )
    if match is None:
        return None
    return _normalized_vcs_url(match.group(1)), match.group(2).lower()


def _expected_thinkingbox_vcs_pin() -> tuple[str, str] | None:
    try:
        requirements = metadata.distribution("thinkingbox-env").requires or []
    except metadata.PackageNotFoundError:
        requirements = []
    for requirement in requirements:
        parsed = _parse_thinkingbox_requirement(requirement)
        if parsed is not None:
            return parsed
    return None


def _thinkingbox_source_root() -> Path:
    spec = importlib.util.find_spec("thinkingbox")
    locations = spec.submodule_search_locations if spec is not None else None
    if not locations:
        raise RuntimeError("Installed ThinkingBox package source is unavailable")
    return Path(next(iter(locations))).resolve()


def _source_tree_sha256(root: Path) -> str:
    ignored = {".git", ".mypy_cache", ".pytest_cache", "__pycache__"}
    files = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix not in {".pyc", ".pyo"}
            and not any(part in ignored for part in path.relative_to(root).parts)
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not files:
        raise RuntimeError("Installed ThinkingBox package source is empty")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@lru_cache(maxsize=1)
def load_thinkingbox_runtime_provenance() -> ThinkingBoxRuntimeProvenance:
    """Load deterministic provenance for the installed ThinkingBox runtime.

    Returns:
        [`ThinkingBoxRuntimeProvenance`]:
            PEP 610 VCS provenance or a deterministic local source identity.

    Raises:
        `RuntimeError`:
            If installed package metadata or package source cannot be read.
    """
    try:
        distribution = metadata.distribution("thinkingbox")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("ThinkingBox distribution is not installed") from exc

    source_sha256 = _source_tree_sha256(_thinkingbox_source_root())
    raw_direct_url = distribution.read_text("direct_url.json")
    direct_url: dict[str, Any] | None = None
    if raw_direct_url is not None:
        try:
            parsed = json.loads(raw_direct_url)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("ThinkingBox PEP 610 metadata is invalid") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("ThinkingBox PEP 610 metadata is invalid")
        direct_url = parsed

    if direct_url is not None and isinstance(direct_url.get("vcs_info"), dict):
        vcs_info = direct_url["vcs_info"]
        commit_id = vcs_info.get("commit_id")
        source_url = direct_url.get("url")
        vcs = vcs_info.get("vcs")
        if (
            vcs != "git"
            or not isinstance(source_url, str)
            or not isinstance(commit_id, str)
            or re.fullmatch(r"[0-9a-fA-F]{40}", commit_id) is None
        ):
            raise RuntimeError("ThinkingBox PEP 610 VCS metadata is invalid")
        normalized_url = _normalized_vcs_url(source_url)
        commit_id = commit_id.lower()
        expected = _expected_thinkingbox_vcs_pin()
        canonical = expected == (normalized_url, commit_id)
        requested_revision = vcs_info.get("requested_revision")
        if requested_revision is not None and not isinstance(requested_revision, str):
            raise RuntimeError("ThinkingBox requested revision metadata is invalid")
        return ThinkingBoxRuntimeProvenance(
            identity=commit_id,
            version=distribution.version,
            source_type="vcs",
            source_sha256=source_sha256,
            commit_id=commit_id,
            requested_revision=requested_revision,
            source_url=normalized_url,
            canonical=canonical,
        )

    dir_info = direct_url.get("dir_info") if direct_url is not None else None
    editable = (
        bool(dir_info.get("editable", False)) if isinstance(dir_info, dict) else False
    )
    source_type = (
        "editable" if editable else ("local" if dir_info is not None else "installed")
    )
    return ThinkingBoxRuntimeProvenance(
        identity=f"source-sha256:{source_sha256}",
        version=distribution.version,
        source_type=source_type,
        source_sha256=source_sha256,
        editable=editable,
    )


def require_canonical_thinkingbox_runtime() -> ThinkingBoxRuntimeProvenance:
    """Return installed runtime provenance only when it matches the VCS pin.

    Returns:
        [`ThinkingBoxRuntimeProvenance`]:
            Canonical installed VCS provenance.

    Raises:
        `RuntimeError`:
            If the installed source is local, editable, unpinned, or does not
            match the dependency declared by `thinkingbox-env`.
    """
    provenance = load_thinkingbox_runtime_provenance()
    if not provenance.canonical:
        raise RuntimeError(
            "Installed ThinkingBox runtime does not match the pinned VCS dependency"
        )
    return provenance


def _parse_thinkingbox_config(payload: bytes) -> ConfigFile:
    raw = yaml.safe_load(payload.decode("utf-8"))
    validated = copy.deepcopy(raw)
    xhigh_paths: list[tuple[str, ...]] = []
    for path in (
        ("agent_model",),
        ("orchestrator", "agent_model"),
        ("judge_model",),
        ("user_model",),
    ):
        source = raw
        target = validated
        for part in path[:-1]:
            if not isinstance(source, dict) or not isinstance(source.get(part), dict):
                break
            source = source[part]
            target = target[part]
        else:
            model = source.get(path[-1]) if isinstance(source, dict) else None
            if isinstance(model, dict) and model.get("reasoning_effort") == "xhigh":
                target[path[-1]]["reasoning_effort"] = "high"
                xhigh_paths.append(path)

    parsed = ConfigFile.model_validate(validated)
    if ("agent_model",) in xhigh_paths or (
        "orchestrator",
        "agent_model",
    ) in xhigh_paths:
        parsed.orchestrator.agent_model.reasoning_effort = "xhigh"
    if ("judge_model",) in xhigh_paths and parsed.judge_model is not None:
        parsed.judge_model.reasoning_effort = "xhigh"
    if ("user_model",) in xhigh_paths and parsed.user_model is not None:
        parsed.user_model.reasoning_effort = "xhigh"
    return parsed


def load_thinkingbox_config_with_sha256(
    config_path: str | Path,
) -> tuple[ConfigFile, str]:
    """Load one immutable config payload and its content fingerprint.

    Args:
        config_path (`str` or `pathlib.Path`):
            Native ThinkingBox YAML configuration.

    Returns:
        `tuple[thinkingbox.common.config_types.ConfigFile, str]`:
            Validated configuration and SHA-256 fingerprint.
    """
    payload = Path(config_path).expanduser().read_bytes()
    return _parse_thinkingbox_config(payload), hashlib.sha256(payload).hexdigest()


def load_thinkingbox_config(config_path: str | Path) -> ConfigFile:
    """Load a native config, including AOAI reasoning efforts newer than the pin.

    Args:
        config_path (`str` or `pathlib.Path`):
            Native ThinkingBox YAML configuration.

    Returns:
        `thinkingbox.common.config_types.ConfigFile`:
            Validated native configuration.
    """
    parsed, _ = load_thinkingbox_config_with_sha256(config_path)
    return parsed


def load_runtime_settings(
    config_path: str | None,
    default_proxy: SessionProxyConfig,
) -> RuntimeSettings:
    """Load native ThinkingBox configuration without exposing it on the wire.

    Args:
        config_path (`str`, *optional*):
            Server-visible native configuration path.
        default_proxy (`thinkingbox.common.config_types.SessionProxyConfig`):
            Proxy fallback used when no configuration is supplied.

    Returns:
        [`RuntimeSettings`]:
            Private settings for one environment episode.
    """
    if not config_path:
        return RuntimeSettings(proxy=default_proxy)
    parsed, config_sha256 = load_thinkingbox_config_with_sha256(config_path)
    return RuntimeSettings(
        proxy=parsed.mcp_proxy,
        config_sha256=config_sha256,
        judge_model=parsed.judge_model,
        judge_type=parsed.judge_type,
        user_model=parsed.user_model,
        user_can_end_conversation=parsed.user_can_end_conversation,
    )


def make_proxy_client(settings: RuntimeSettings) -> MCPProxyClient:
    """Construct the native Session Proxy client.

    Args:
        settings ([`RuntimeSettings`]):
            Private episode settings.

    Returns:
        `thinkingbox.common.mcp_proxy_client.MCPProxyClient`:
            Configured native proxy client.
    """
    proxy = settings.proxy
    credential = (
        create_credential(proxy.credential) if proxy.credential is not None else None
    )
    return MCPProxyClient(
        endpoint=proxy.endpoint_url,
        timeout=proxy.timeout,
        use_dns_cache=proxy.use_dns_cache,
        credential=credential,
        client_certificate=proxy.client_certificate,
        trust_ca_path=proxy.trust_ca_path,
        headers=proxy.headers,
        max_retries_server_error=proxy.max_retries_server_error,
        retryable_server_errors=proxy.retryable_server_errors,
        always_json_output=proxy.always_json_output,
        geteffects_proxy_info=proxy.geteffects_proxy_info,
    )
