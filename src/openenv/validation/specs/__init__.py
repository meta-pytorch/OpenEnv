# SPDX-License-Identifier: BSD-3-Clause

"""Trusted source-spec adapters and normalized validation requirements."""

from .base import (
    AdapterIdentity,
    ArtifactRequirement,
    DetectionMode,
    EnvironmentRequirements,
    ExecutionModel,
    HealthcheckRequirement,
    NetworkMode,
    NetworkRequirements,
    PackageIdentity,
    PhaseRequirements,
    RequirementsLoad,
    RequirementsProvenance,
    RequirementsState,
    SpecIdentity,
    SpecLoad,
    SpecLoadState,
    StepRequirements,
    ValidationRequirements,
    ValidationSpecAdapter,
    ValidationSpecRegistry,
    ValidationSubject,
)
from .harbor import find_harbor_verifier_script, load_harbor_requirements
from .openenv import OpenEnvSpecAdapter, runtime_openenv_spec_load


DEFAULT_SPEC_REGISTRY = ValidationSpecRegistry((OpenEnvSpecAdapter(),))

__all__ = [
    "AdapterIdentity",
    "ArtifactRequirement",
    "DEFAULT_SPEC_REGISTRY",
    "DetectionMode",
    "EnvironmentRequirements",
    "ExecutionModel",
    "HealthcheckRequirement",
    "NetworkMode",
    "NetworkRequirements",
    "OpenEnvSpecAdapter",
    "PackageIdentity",
    "PhaseRequirements",
    "RequirementsLoad",
    "RequirementsProvenance",
    "RequirementsState",
    "SpecIdentity",
    "SpecLoad",
    "SpecLoadState",
    "StepRequirements",
    "ValidationRequirements",
    "ValidationSpecAdapter",
    "ValidationSpecRegistry",
    "ValidationSubject",
    "find_harbor_verifier_script",
    "load_harbor_requirements",
    "runtime_openenv_spec_load",
]
