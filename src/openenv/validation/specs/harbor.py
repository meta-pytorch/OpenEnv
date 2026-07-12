# SPDX-License-Identifier: BSD-3-Clause

"""Harbor v0.5 task.toml adapter for normalized validation requirements."""

from __future__ import annotations

import hashlib
import os
import re
from decimal import Decimal, DecimalException
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
    ValidationError,
)

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from .base import (
    AdapterIdentity,
    ArtifactRequirement,
    EnvironmentRequirements,
    HealthcheckRequirement,
    NetworkMode,
    NetworkRequirements,
    PackageIdentity,
    PhaseRequirements,
    RequirementsLoad,
    RequirementsProvenance,
    RequirementsState,
    StepRequirements,
    ValidationRequirements,
)


_MAX_RESOURCE_MB = 2**31 - 1
_MAX_RESOURCE_COUNT = 1024
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_MANIFEST_DEPTH = 64
_MAX_MANIFEST_NODES = 100_000

_HARBOR_ADAPTER = AdapterIdentity("harbor-task-toml", "1")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class _TaskIdentityModel(_StrictModel):
    name: str
    description: str = ""
    authors: list["_AuthorModel"] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        pattern = r"^[A-Za-z0-9_-][A-Za-z0-9._-]*/[A-Za-z0-9_-][A-Za-z0-9._-]*$"
        if not re.match(pattern, value) or ".." in value:
            raise ValueError("task name must use the org/name format")
        return value


class _AuthorModel(_StrictModel):
    name: str = Field(min_length=1)
    email: str | None = None


class _AgentModel(_StrictModel):
    timeout_sec: float | None = Field(default=None, gt=0)
    user: str | int | None = None


class _VerifierModel(_StrictModel):
    timeout_sec: float = Field(default=600.0, gt=0)
    env: dict[str, str] = Field(default_factory=dict)
    user: str | int | None = None


class _SolutionModel(_StrictModel):
    env: dict[str, str] = Field(default_factory=dict)


class _HealthcheckModel(_StrictModel):
    command: str = Field(min_length=1)
    interval_sec: float = Field(default=5.0, gt=0)
    timeout_sec: float = Field(default=30.0, gt=0)
    start_period_sec: float = Field(default=0.0, ge=0)
    start_interval_sec: float = Field(default=5.0, gt=0)
    retries: int = Field(default=3, gt=0)


class _EnvironmentModel(_StrictModel):
    build_timeout_sec: float = Field(default=600.0, gt=0)
    docker_image: str | None = None
    cpus: int = Field(default=1, gt=0, le=_MAX_RESOURCE_COUNT)
    memory_mb: int = Field(default=2048, gt=0, le=_MAX_RESOURCE_MB)
    storage_mb: int = Field(default=10240, gt=0, le=_MAX_RESOURCE_MB)
    gpus: int = Field(default=0, ge=0, le=_MAX_RESOURCE_COUNT)
    gpu_types: list[str] | None = None
    allow_internet: bool = True
    mcp_servers: list["_MCPServerModel"] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    skills_dir: str | None = None
    healthcheck: _HealthcheckModel | None = None
    workdir: str | None = None
    memory: str | None = None
    storage: str | None = None

    @field_validator("docker_image")
    @classmethod
    def _nonempty_image(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("docker_image cannot be empty")
        return value


class _ArtifactModel(_StrictModel):
    source: str = Field(min_length=1)
    destination: str | None = None
    exclude: list[str] = Field(default_factory=list)


class _MCPServerModel(_StrictModel):
    name: str = Field(min_length=1)
    transport: str = "sse"
    url: str | None = None
    command: str | None = None
    args: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_transport_fields(self) -> "_MCPServerModel":
        if self.transport in {"sse", "streamable-http"} and not self.url:
            raise ValueError(f"url is required for transport {self.transport}")
        if self.transport == "stdio" and not self.command:
            raise ValueError("command is required for stdio transport")
        return self


class _StepModel(_StrictModel):
    name: str = Field(min_length=1)
    agent: _AgentModel = Field(default_factory=_AgentModel)
    verifier: _VerifierModel = Field(default_factory=_VerifierModel)
    min_reward: float | dict[str, float] | None = None
    healthcheck: _HealthcheckModel | None = None
    artifacts: list[str | _ArtifactModel] = Field(default_factory=list)


class _TaskManifestModel(_StrictModel):
    schema_version: str = "1.1"
    task: _TaskIdentityModel | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    verifier: _VerifierModel = Field(default_factory=_VerifierModel)
    agent: _AgentModel = Field(default_factory=_AgentModel)
    solution: _SolutionModel = Field(default_factory=_SolutionModel)
    environment: _EnvironmentModel = Field(default_factory=_EnvironmentModel)
    source: str | None = None
    artifacts: list[str | _ArtifactModel] = Field(default_factory=list)
    steps: list[_StepModel] | None = None
    multi_step_reward_strategy: Literal["mean", "final"] | None = None


def _size_to_mb(value: str) -> int:
    normalized = value.strip().upper()
    factors = {"G": Decimal(1024), "M": Decimal(1), "K": Decimal(1) / 1024}
    suffix = normalized[-1:] if normalized else ""
    if suffix not in factors:
        raise ValueError("size must use K, M, or G units")
    try:
        amount = Decimal(normalized[:-1])
        size_mb = amount * factors[suffix]
        if not size_mb.is_finite() or size_mb <= 0 or size_mb > _MAX_RESOURCE_MB:
            raise ValueError("size is outside the supported resource range")
        return max(1, int(size_mb))
    except DecimalException as exc:
        raise ValueError("size must contain a finite supported number") from exc


def _normalize_legacy_resources(raw: dict[str, Any]) -> dict[str, Any]:
    environment = raw.get("environment")
    if not isinstance(environment, dict):
        return raw
    normalized = dict(raw)
    normalized_environment = dict(environment)
    for legacy_name, current_name in (
        ("memory", "memory_mb"),
        ("storage", "storage_mb"),
    ):
        legacy_value = normalized_environment.get(legacy_name)
        if legacy_value is None:
            continue
        if not isinstance(legacy_value, str):
            continue
        converted = _size_to_mb(legacy_value)
        if (
            current_name in normalized_environment
            and normalized_environment[current_name] != converted
        ):
            raise ValueError(f"conflicting {legacy_name} and {current_name} fields")
        normalized_environment[current_name] = converted
    normalized["environment"] = normalized_environment
    return normalized


def _validate_manifest_structure(raw: dict[str, Any]) -> None:
    stack: list[tuple[Any, int]] = [(raw, 0)]
    visited = 0
    while stack:
        value, depth = stack.pop()
        visited += 1
        if visited > _MAX_MANIFEST_NODES:
            raise ValueError("task.toml exceeds the maximum structural size")
        if depth > _MAX_MANIFEST_DEPTH:
            raise ValueError("task.toml exceeds the maximum nesting depth")
        if isinstance(value, dict):
            stack.extend((item, depth + 1) for item in value.values())
        elif isinstance(value, list):
            stack.extend((item, depth + 1) for item in value)


def _artifact(model: str | _ArtifactModel) -> ArtifactRequirement:
    if isinstance(model, str):
        return ArtifactRequirement(source=model)
    return ArtifactRequirement(
        source=model.source,
        destination=model.destination,
        exclude=tuple(model.exclude),
    )


def _phase(
    model: _AgentModel | _VerifierModel,
) -> PhaseRequirements:
    env = model.env if isinstance(model, _VerifierModel) else {}
    return PhaseRequirements(
        timeout_s=model.timeout_sec,
        user=model.user,
        env_names=tuple(sorted(env)),
    )


def _convert(
    model: _TaskManifestModel, present_fields: tuple[str, ...]
) -> ValidationRequirements:
    healthcheck_model = model.environment.healthcheck
    healthcheck = (
        HealthcheckRequirement(
            command=healthcheck_model.command,
            interval_s=healthcheck_model.interval_sec,
            timeout_s=healthcheck_model.timeout_sec,
            start_period_s=healthcheck_model.start_period_sec,
            start_interval_s=healthcheck_model.start_interval_sec,
            retries=healthcheck_model.retries,
        )
        if healthcheck_model is not None
        else None
    )
    environment = EnvironmentRequirements(
        build_timeout_s=model.environment.build_timeout_sec,
        container_image=model.environment.docker_image,
        cpus=model.environment.cpus,
        memory_mb=model.environment.memory_mb,
        storage_mb=model.environment.storage_mb,
        gpus=model.environment.gpus,
        gpu_types=tuple(model.environment.gpu_types or ()),
        network=NetworkRequirements(
            mode=(
                NetworkMode.ALLOW_ALL
                if model.environment.allow_internet
                else NetworkMode.DENY_ALL
            )
        ),
        env_names=tuple(sorted(model.environment.env)),
        skills_dir=model.environment.skills_dir,
        workdir=model.environment.workdir,
        mcp_server_count=len(model.environment.mcp_servers),
        healthcheck=healthcheck,
    )
    steps = tuple(
        StepRequirements(
            name=step.name,
            agent=_phase(step.agent),
            verifier=_phase(step.verifier),
            min_reward=step.min_reward,
            healthcheck=(
                HealthcheckRequirement(
                    command=step.healthcheck.command,
                    interval_s=step.healthcheck.interval_sec,
                    timeout_s=step.healthcheck.timeout_sec,
                    start_period_s=step.healthcheck.start_period_sec,
                    start_interval_s=step.healthcheck.start_interval_sec,
                    retries=step.healthcheck.retries,
                )
                if step.healthcheck is not None
                else None
            ),
            artifacts=tuple(_artifact(artifact) for artifact in step.artifacts),
        )
        for step in model.steps or ()
    )
    task = model.task
    return ValidationRequirements(
        identity=PackageIdentity(
            name=task.name if task is not None else None,
            description=task.description if task is not None else "",
            authors=(
                tuple(author.model_dump() for author in task.authors)
                if task is not None
                else ()
            ),
            keywords=tuple(task.keywords) if task is not None else (),
        ),
        agent=_phase(model.agent),
        verifier=_phase(model.verifier),
        environment=environment,
        artifacts=tuple(_artifact(artifact) for artifact in model.artifacts),
        steps=steps,
        metadata_keys=tuple(sorted(model.metadata)),
        solution_env_names=tuple(sorted(model.solution.env)),
        multi_step_reward_strategy=model.multi_step_reward_strategy,
        source_declared=model.source is not None,
        present_fields=present_fields,
    )


def _manifest_path(root: Path) -> Path:
    direct = root / "task.toml"
    if direct.exists() or direct.is_symlink() or root.name != "environment":
        return direct
    return root.parent / "task.toml"


def _contained_regular_file(root: Path, candidate: Path) -> bool:
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return False
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            return False
    try:
        return bool(
            candidate.is_file()
            and candidate.resolve(strict=True).is_relative_to(root.resolve(strict=True))
        )
    except OSError:
        return False


def find_harbor_verifier_script(root: str | Path) -> Path | None:
    """Find tests/test.sh for an OpenEnv root or Harbor environment directory."""
    path = Path(root)
    direct = path / "tests" / "test.sh"
    if _contained_regular_file(path, direct):
        return direct
    if path.name == "environment" and (path.parent / "task.toml").is_file():
        sibling = path.parent / "tests" / "test.sh"
        if _contained_regular_file(path.parent, sibling):
            return sibling
    return None


def _validation_error(exc: ValidationError) -> str:
    issues = []
    for error in exc.errors(include_input=False, include_url=False):
        location = ".".join(str(part) for part in error.get("loc", ())) or "task.toml"
        issues.append(f"{location}: {error.get('msg', error.get('type', 'invalid'))}")
    return "; ".join(issues)


def _provenance(
    root: Path,
    path: Path,
    version: str | None,
    document_digest: str | None = None,
) -> RequirementsProvenance:
    return RequirementsProvenance(
        source_id="harbor",
        source_version=version,
        adapter=_HARBOR_ADAPTER,
        path=os.path.relpath(path, start=root),
        document_digest=document_digest,
    )


def load_harbor_requirements(root: str | Path) -> RequirementsLoad:
    """Load Harbor v0.5/schema 1.1 into normalized validation requirements."""
    root_path = Path(root)
    path = _manifest_path(root_path)
    if not path.exists() and not path.is_symlink():
        return RequirementsLoad(state=RequirementsState.ABSENT)
    if path.is_symlink():
        return RequirementsLoad(
            state=RequirementsState.INVALID,
            provenance=_provenance(root_path, path, None),
            error="task.toml must be a regular file, not a symbolic link",
        )
    if not path.is_file():
        return RequirementsLoad(
            state=RequirementsState.INVALID,
            provenance=_provenance(root_path, path, None),
            error="task.toml must be a regular file",
        )

    document_digest: str | None = None
    try:
        with path.open("rb") as manifest_file:
            manifest_bytes = manifest_file.read(_MAX_MANIFEST_BYTES + 1)
        if len(manifest_bytes) > _MAX_MANIFEST_BYTES:
            return RequirementsLoad(
                state=RequirementsState.INVALID,
                provenance=_provenance(root_path, path, None),
                error="task.toml exceeds the 1 MiB validation limit",
            )
        document_digest = f"sha256:{hashlib.sha256(manifest_bytes).hexdigest()}"
        raw = tomllib.loads(manifest_bytes.decode("utf-8"))
    except (OSError, RecursionError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        return RequirementsLoad(
            state=RequirementsState.INVALID,
            provenance=_provenance(root_path, path, None, document_digest),
            error=f"Unable to parse task.toml ({type(exc).__name__})",
        )

    if not isinstance(raw, dict):
        return RequirementsLoad(
            state=RequirementsState.INVALID,
            provenance=_provenance(root_path, path, None, document_digest),
            error="task.toml must contain a TOML table",
        )

    try:
        _validate_manifest_structure(raw)
    except (RecursionError, ValueError) as exc:
        return RequirementsLoad(
            state=RequirementsState.INVALID,
            provenance=_provenance(root_path, path, None, document_digest),
            error=f"Invalid task.toml structure ({type(exc).__name__})",
        )

    normalized = dict(raw)
    legacy_version = normalized.pop("version", None)
    if legacy_version is not None:
        current_version = normalized.get("schema_version")
        if current_version is not None and current_version != legacy_version:
            return RequirementsLoad(
                state=RequirementsState.INVALID,
                provenance=_provenance(
                    root_path, path, str(current_version), document_digest
                ),
                error="task.toml has conflicting version and schema_version fields",
            )
        normalized["schema_version"] = legacy_version

    try:
        normalized = _normalize_legacy_resources(normalized)
        model = _TaskManifestModel.model_validate(normalized)
    except ValidationError as exc:
        return RequirementsLoad(
            state=RequirementsState.INVALID,
            provenance=_provenance(
                root_path,
                path,
                str(normalized.get("schema_version", "1.1")),
                document_digest,
            ),
            error=_validation_error(exc),
        )
    except (
        DecimalException,
        OverflowError,
        RecursionError,
        TypeError,
        ValueError,
    ) as exc:
        return RequirementsLoad(
            state=RequirementsState.INVALID,
            provenance=_provenance(
                root_path,
                path,
                str(normalized.get("schema_version", "1.1")),
                document_digest,
            ),
            error=f"Invalid task.toml configuration ({type(exc).__name__})",
        )

    provenance = _provenance(root_path, path, model.schema_version, document_digest)
    if model.schema_version != "1.1":
        return RequirementsLoad(
            state=RequirementsState.UNSUPPORTED,
            provenance=provenance,
            error=(
                "Unsupported Harbor schema version "
                f"{model.schema_version!r}; expected '1.1'"
            ),
        )
    present_fields = tuple(sorted(str(key) for key in raw))
    requirements = _convert(model, present_fields)
    artifacts = list(requirements.artifacts)
    for step in requirements.steps:
        artifacts.extend(step.artifacts)
    if any(not artifact.source.startswith("/") for artifact in artifacts):
        return RequirementsLoad(
            state=RequirementsState.INVALID,
            provenance=provenance,
            error="Harbor artifact sources must use absolute container paths",
        )
    environment = requirements.environment
    if environment.gpus == 0 and environment.gpu_types:
        return RequirementsLoad(
            state=RequirementsState.INVALID,
            provenance=provenance,
            error="Harbor gpu_types requires a positive gpus value",
        )
    return RequirementsLoad(
        state=RequirementsState.LOADED,
        provenance=provenance,
        requirements=requirements,
    )
