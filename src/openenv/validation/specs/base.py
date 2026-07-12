# SPDX-License-Identifier: BSD-3-Clause

"""Spec-neutral contracts for validation subjects and requirements."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, Sequence


_STABLE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SHA256_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class ExecutionModel(str, Enum):
    """Lifecycle used to execute a validation subject."""

    SERVED = "served"
    ONE_SHOT = "one_shot"
    EXTERNAL = "external"


class SpecLoadState(str, Enum):
    """Outcome of detecting and safely inspecting a source spec."""

    ABSENT = "absent"
    LOADED = "loaded"
    INVALID = "invalid"
    UNSUPPORTED = "unsupported"
    AMBIGUOUS = "ambiguous"


class RequirementsState(str, Enum):
    """Outcome of loading normalized execution requirements."""

    ABSENT = "absent"
    LOADED = "loaded"
    INVALID = "invalid"
    UNSUPPORTED = "unsupported"


class DetectionMode(str, Enum):
    """How a spec adapter was selected."""

    AUTO = "auto"
    EXPLICIT = "explicit"
    RUNTIME = "runtime"


class NetworkMode(str, Enum):
    """Normalized outbound-network requirement."""

    UNSPECIFIED = "unspecified"
    DENY_ALL = "deny_all"
    ALLOW_ALL = "allow_all"
    ALLOWLIST = "allowlist"


@dataclass(frozen=True)
class AdapterIdentity:
    """Stable identity of trusted adapter code."""

    adapter_id: str
    adapter_version: str

    def __post_init__(self) -> None:
        if not _STABLE_ID.fullmatch(self.adapter_id):
            raise ValueError("adapter ID must be a stable lowercase identifier")
        if not self.adapter_version.strip() or len(self.adapter_version) > 128:
            raise ValueError("adapter version must be a non-empty bounded string")

    def to_dict(self) -> dict[str, str]:
        return {"id": self.adapter_id, "version": self.adapter_version}


@dataclass(frozen=True)
class SpecIdentity:
    """Source-format and execution-model provenance."""

    spec_id: str
    spec_version: str | None
    adapter: AdapterIdentity
    execution_model: ExecutionModel

    def __post_init__(self) -> None:
        if not _STABLE_ID.fullmatch(self.spec_id):
            raise ValueError("spec ID must be a stable lowercase identifier")
        if self.spec_version is not None and (
            not self.spec_version.strip() or len(self.spec_version) > 128
        ):
            raise ValueError("spec version must be a non-empty bounded string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.spec_id,
            "version": self.spec_version,
            "adapter": self.adapter.to_dict(),
            "execution_model": self.execution_model.value,
        }


@dataclass(frozen=True)
class RequirementsProvenance:
    """Origin of a normalized requirements envelope."""

    source_id: str
    source_version: str | None
    adapter: AdapterIdentity
    path: str
    document_digest: str | None = None

    def __post_init__(self) -> None:
        if not _STABLE_ID.fullmatch(self.source_id):
            raise ValueError(
                "requirements source ID must be a stable lowercase identifier"
            )
        if self.source_version is not None and (
            not self.source_version.strip() or len(self.source_version) > 128
        ):
            raise ValueError("requirements source version must be bounded")
        if (
            not self.path
            or len(self.path) > 4096
            or "\x00" in self.path
            or Path(self.path).is_absolute()
        ):
            raise ValueError("requirements source path must be relative")
        if self.document_digest is not None and not _SHA256_DIGEST.fullmatch(
            self.document_digest
        ):
            raise ValueError(
                "requirements document digest must be a lowercase SHA-256 digest"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.source_id,
            "version": self.source_version,
            "adapter": self.adapter.to_dict(),
            "path": self.path,
            "document_digest": self.document_digest,
        }


@dataclass(frozen=True)
class ArtifactRequirement:
    """Artifact copied from a subject environment to its verifier."""

    source: str
    destination: str | None = None
    exclude: tuple[str, ...] = ()


@dataclass(frozen=True)
class HealthcheckRequirement:
    """Normalized healthcheck settings."""

    command: str
    interval_s: float = 5.0
    timeout_s: float = 30.0
    start_period_s: float = 0.0
    start_interval_s: float = 5.0
    retries: int = 3


@dataclass(frozen=True)
class PackageIdentity:
    """Optional package identity declared by the source spec."""

    name: str | None = None
    description: str = ""
    authors: tuple[dict[str, str], ...] = ()
    keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class PhaseRequirements:
    """Timeout, user, and secret-name declarations for one phase."""

    timeout_s: float | None = None
    user: str | int | None = None
    env_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class NetworkRequirements:
    """Normalized egress policy independent of a source spec."""

    mode: NetworkMode = NetworkMode.UNSPECIFIED
    endpoints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.mode is not NetworkMode.ALLOWLIST and self.endpoints:
            raise ValueError("network endpoints require allowlist mode")


@dataclass(frozen=True)
class EnvironmentRequirements:
    """Resource and containment requirements for a subject environment."""

    build_timeout_s: float | None = None
    container_image: str | None = None
    cpus: int | None = None
    memory_mb: int | None = None
    storage_mb: int | None = None
    gpus: int | None = None
    gpu_types: tuple[str, ...] = ()
    network: NetworkRequirements = NetworkRequirements()
    env_names: tuple[str, ...] = ()
    skills_dir: str | None = None
    workdir: str | None = None
    mcp_server_count: int = 0
    healthcheck: HealthcheckRequirement | None = None


@dataclass(frozen=True)
class StepRequirements:
    """Normalized per-step overrides for a multi-step subject."""

    name: str
    agent: PhaseRequirements = PhaseRequirements()
    verifier: PhaseRequirements = PhaseRequirements(timeout_s=600.0)
    min_reward: float | dict[str, float] | None = None
    healthcheck: HealthcheckRequirement | None = None
    artifacts: tuple[ArtifactRequirement, ...] = ()


@dataclass(frozen=True)
class ValidationRequirements:
    """Sanitized planner inputs independent of their source format."""

    identity: PackageIdentity = PackageIdentity()
    agent: PhaseRequirements = PhaseRequirements()
    verifier: PhaseRequirements = PhaseRequirements(timeout_s=600.0)
    environment: EnvironmentRequirements = EnvironmentRequirements()
    artifacts: tuple[ArtifactRequirement, ...] = ()
    steps: tuple[StepRequirements, ...] = ()
    metadata_keys: tuple[str, ...] = ()
    solution_env_names: tuple[str, ...] = ()
    multi_step_reward_strategy: str | None = None
    source_declared: bool = False
    present_fields: tuple[str, ...] = ()

    def to_evidence(self) -> dict[str, Any]:
        """Return report-safe requirements without environment values."""

        def health_evidence(
            value: HealthcheckRequirement | None,
        ) -> dict[str, Any] | None:
            if value is None:
                return None
            return {
                "command_declared": True,
                "command_digest": (
                    f"sha256:{hashlib.sha256(value.command.encode('utf-8')).hexdigest()}"
                ),
                "interval_s": value.interval_s,
                "timeout_s": value.timeout_s,
                "start_period_s": value.start_period_s,
                "start_interval_s": value.start_interval_s,
                "retries": value.retries,
            }

        def artifact_evidence(value: ArtifactRequirement) -> dict[str, Any]:
            return {
                "source": value.source,
                "destination": value.destination,
                "exclude": list(value.exclude),
            }

        environment = self.environment
        return {
            "identity": {
                "name": self.identity.name,
                "description": self.identity.description,
                "author_count": len(self.identity.authors),
                "keywords": list(self.identity.keywords),
            },
            "timeouts": {
                "build_s": environment.build_timeout_s,
                "agent_s": self.agent.timeout_s,
                "verifier_s": self.verifier.timeout_s,
            },
            "resources": {
                "cpus": environment.cpus,
                "memory_mb": environment.memory_mb,
                "storage_mb": environment.storage_mb,
                "gpus": environment.gpus,
                "gpu_types": list(environment.gpu_types),
            },
            "network": {
                "mode": environment.network.mode.value,
                "endpoints": list(environment.network.endpoints),
            },
            "healthcheck": health_evidence(environment.healthcheck),
            "container_image": environment.container_image,
            "execution": {
                "agent_user": self.agent.user,
                "verifier_user": self.verifier.user,
                "workdir": environment.workdir,
                "skills_dir": environment.skills_dir,
                "mcp_server_count": environment.mcp_server_count,
                "source_declared": self.source_declared,
            },
            "artifacts": [artifact_evidence(value) for value in self.artifacts],
            "step_names": [step.name for step in self.steps],
            "multi_step_reward_strategy": self.multi_step_reward_strategy,
            "steps": [
                {
                    "name": step.name,
                    "agent_timeout_s": step.agent.timeout_s,
                    "verifier_timeout_s": step.verifier.timeout_s,
                    "agent_user": step.agent.user,
                    "verifier_user": step.verifier.user,
                    "min_reward": step.min_reward,
                    "healthcheck": health_evidence(step.healthcheck),
                    "artifacts": [artifact_evidence(value) for value in step.artifacts],
                    "environment_variable_names": {
                        "agent": list(step.agent.env_names),
                        "verifier": list(step.verifier.env_names),
                    },
                }
                for step in self.steps
            ],
            "environment_variable_names": {
                "environment": list(environment.env_names),
                "agent": list(self.agent.env_names),
                "verifier": list(self.verifier.env_names),
                "solution": list(self.solution_env_names),
            },
            "metadata_keys": list(self.metadata_keys),
            "present_fields": list(self.present_fields),
        }


@dataclass(frozen=True)
class RequirementsLoad:
    """Structured result of adapting an auxiliary requirements document."""

    state: RequirementsState
    provenance: RequirementsProvenance | None = None
    requirements: ValidationRequirements | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.state is RequirementsState.LOADED and self.requirements is None:
            raise ValueError("loaded requirements must include normalized values")
        if self.state is not RequirementsState.LOADED and self.requirements is not None:
            raise ValueError("only loaded requirements may include normalized values")
        if self.state in {RequirementsState.INVALID, RequirementsState.UNSUPPORTED}:
            if not self.error:
                raise ValueError("invalid or unsupported requirements need an error")

    def to_evidence(self) -> dict[str, Any]:
        evidence: dict[str, Any] = {
            "state": self.state.value,
            "present": self.state is not RequirementsState.ABSENT,
        }
        if self.provenance is not None:
            evidence["source"] = self.provenance.to_dict()
        if self.requirements is not None:
            evidence.update(self.requirements.to_evidence())
        if self.error is not None:
            evidence["error"] = self.error
        return evidence


@dataclass(frozen=True)
class ValidationSubject:
    """Safely inspected subject supplied to planners and checks."""

    spec: SpecIdentity
    signature_path: str | None
    detection_mode: DetectionMode
    requirements: RequirementsLoad
    verifier_script: Path | None = None
    verifier_path: str | None = None
    verifier_digest: str | None = None
    document_digest: str | None = None

    def __post_init__(self) -> None:
        if self.signature_path is not None:
            signature = Path(self.signature_path)
            if signature.is_absolute() or ".." in signature.parts:
                raise ValueError("spec signature path must stay within the target root")
        if self.document_digest is not None and not _SHA256_DIGEST.fullmatch(
            self.document_digest
        ):
            raise ValueError("spec document digest must be a lowercase SHA-256 digest")
        verifier_fields = (
            self.verifier_script,
            self.verifier_path,
            self.verifier_digest,
        )
        if any(value is not None for value in verifier_fields) and any(
            value is None for value in verifier_fields
        ):
            raise ValueError(
                "verifier path, file, and digest must be declared together"
            )
        if self.verifier_path is not None and (
            not self.verifier_path
            or len(self.verifier_path) > 4096
            or "\x00" in self.verifier_path
            or Path(self.verifier_path).is_absolute()
        ):
            raise ValueError("verifier path must be a bounded relative path")
        if self.verifier_digest is not None and not _SHA256_DIGEST.fullmatch(
            self.verifier_digest
        ):
            raise ValueError("verifier digest must be a lowercase SHA-256 digest")

    def to_dict(self) -> dict[str, Any]:
        payload = self.spec.to_dict()
        payload.update(
            {
                "detection_mode": self.detection_mode.value,
                "signature_path": self.signature_path,
                "document_digest": self.document_digest,
                "requirements_state": self.requirements.state.value,
                "requirements": (
                    self.requirements.provenance.to_dict()
                    if self.requirements.provenance is not None
                    else None
                ),
                "verifier": (
                    {
                        "path": self.verifier_path,
                        "document_digest": self.verifier_digest,
                    }
                    if self.verifier_path is not None
                    else None
                ),
            }
        )
        return payload


@dataclass(frozen=True)
class SpecLoad:
    """Result of selecting and safely inspecting a validation spec."""

    state: SpecLoadState
    subject: ValidationSubject | None = None
    identity: SpecIdentity | None = None
    error: str | None = None
    matches: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.state is SpecLoadState.LOADED and self.subject is None:
            raise ValueError("loaded spec must include a validation subject")
        if self.state is not SpecLoadState.LOADED and self.subject is not None:
            raise ValueError("only a loaded spec may include a validation subject")
        if (
            self.subject is not None
            and self.identity is not None
            and self.identity != self.subject.spec
        ):
            raise ValueError("spec load identity must match its loaded subject")
        if (
            self.state
            in {
                SpecLoadState.INVALID,
                SpecLoadState.UNSUPPORTED,
                SpecLoadState.AMBIGUOUS,
            }
            and not self.error
        ):
            raise ValueError("invalid, unsupported, or ambiguous specs need an error")

    @property
    def spec(self) -> SpecIdentity | None:
        return self.subject.spec if self.subject is not None else self.identity

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "spec": self.subject.to_dict()
            if self.subject is not None
            else (self.identity.to_dict() if self.identity is not None else None),
            "matches": list(self.matches),
            "error": self.error,
        }


class ValidationSpecAdapter(Protocol):
    """Trusted structural adapter for one validation source format."""

    spec_id: str
    adapter_id: str
    adapter_version: str
    execution_model: ExecutionModel
    signature_files: tuple[str, ...]

    def detect(self, root: Path) -> bool:
        """Detect only by trusted signature paths; never import source code."""
        ...

    def inspect(self, root: Path) -> SpecLoad:
        """Return a sanitized subject without executing submitted code."""
        ...


class ValidationSpecRegistry:
    """Deterministic in-memory registry of installed spec adapters."""

    def __init__(
        self,
        adapters: Sequence[ValidationSpecAdapter],
    ) -> None:
        self._adapters = tuple(adapters)
        ids = [adapter.spec_id for adapter in self._adapters]
        if any(not _STABLE_ID.fullmatch(spec_id) for spec_id in ids):
            raise ValueError("spec adapter IDs must be stable lowercase identifiers")
        if len(ids) != len(set(ids)):
            raise ValueError("spec adapter IDs must be unique")
        for adapter in self._adapters:
            AdapterIdentity(adapter.adapter_id, adapter.adapter_version)
            if not isinstance(adapter.execution_model, ExecutionModel):
                raise ValueError("spec adapters must declare an execution model")
            if not adapter.signature_files or any(
                not signature
                or Path(signature).is_absolute()
                or ".." in Path(signature).parts
                for signature in adapter.signature_files
            ):
                raise ValueError(
                    "spec adapter signatures must be non-empty relative paths"
                )

    @property
    def supported_specs(self) -> tuple[str, ...]:
        return tuple(adapter.spec_id for adapter in self._adapters)

    def get(self, spec_id: str) -> ValidationSpecAdapter:
        for adapter in self._adapters:
            if adapter.spec_id == spec_id:
                return adapter
        supported = ", ".join(self.supported_specs) or "none"
        raise ValueError(
            f"Unsupported validation spec {spec_id!r}. Supported: {supported}"
        )

    @staticmethod
    def _unknown_identity(adapter: ValidationSpecAdapter) -> SpecIdentity:
        return SpecIdentity(
            spec_id=adapter.spec_id,
            spec_version=None,
            adapter=AdapterIdentity(adapter.adapter_id, adapter.adapter_version),
            execution_model=adapter.execution_model,
        )

    def _inspect(
        self,
        adapter: ValidationSpecAdapter,
        root: Path,
        mode: DetectionMode,
    ) -> SpecLoad:
        try:
            loaded = adapter.inspect(root)
        except Exception as exc:
            return SpecLoad(
                state=SpecLoadState.INVALID,
                identity=self._unknown_identity(adapter),
                error=f"Spec adapter inspection failed ({type(exc).__name__})",
                matches=(adapter.spec_id,),
            )
        if not isinstance(loaded, SpecLoad):
            return SpecLoad(
                state=SpecLoadState.INVALID,
                identity=self._unknown_identity(adapter),
                error="Spec adapter returned an invalid inspection result",
                matches=(adapter.spec_id,),
            )
        identity = loaded.spec
        expected_adapter = AdapterIdentity(adapter.adapter_id, adapter.adapter_version)
        if (
            identity is None
            or identity.spec_id != adapter.spec_id
            or identity.adapter != expected_adapter
            or identity.execution_model is not adapter.execution_model
        ):
            return SpecLoad(
                state=SpecLoadState.INVALID,
                identity=self._unknown_identity(adapter),
                error="Spec adapter returned inconsistent provenance",
                matches=(adapter.spec_id,),
            )
        if (
            loaded.subject is not None
            and loaded.subject.signature_path not in adapter.signature_files
        ):
            return SpecLoad(
                state=SpecLoadState.INVALID,
                identity=identity,
                error="Spec adapter returned an undeclared signature path",
                matches=(adapter.spec_id,),
            )
        if loaded.subject is None:
            return replace(loaded, matches=(adapter.spec_id,))
        subject = replace(
            loaded.subject,
            detection_mode=mode,
        )
        return replace(loaded, subject=subject, matches=(adapter.spec_id,))

    def resolve(self, root: str | Path, spec_id: str | None = None) -> SpecLoad:
        """Resolve an explicit spec or reject absent/ambiguous auto-detection."""
        path = Path(root)
        if spec_id is not None:
            return self._inspect(self.get(spec_id), path, DetectionMode.EXPLICIT)

        matches: list[ValidationSpecAdapter] = []
        for adapter in self._adapters:
            try:
                detected = adapter.detect(path)
            except Exception as exc:
                return SpecLoad(
                    state=SpecLoadState.INVALID,
                    identity=self._unknown_identity(adapter),
                    error=f"Spec adapter detection failed ({type(exc).__name__})",
                    matches=(adapter.spec_id,),
                )
            if not isinstance(detected, bool):
                return SpecLoad(
                    state=SpecLoadState.INVALID,
                    identity=self._unknown_identity(adapter),
                    error="Spec adapter detection must return a boolean",
                    matches=(adapter.spec_id,),
                )
            if detected:
                matches.append(adapter)
        if not matches:
            return SpecLoad(state=SpecLoadState.ABSENT)
        if len(matches) > 1:
            ids = tuple(sorted(adapter.spec_id for adapter in matches))
            return SpecLoad(
                state=SpecLoadState.AMBIGUOUS,
                error=f"Multiple validation specs matched: {', '.join(ids)}",
                matches=ids,
            )
        return self._inspect(matches[0], path, DetectionMode.AUTO)
