"""Cognitive Manufacturing Environment - OpenEnv compliant manufacturing simulation.

This package provides an environment for AI agents to manage and optimize
a simulated manufacturing system with real-time sensor data, health monitoring,
and multi-objective optimization.

Phase 0 MVP Features:
- Single simulated production machine
- 5 tools: ReadSensors, CheckHealth, AdjustSpeed, ScheduleMaintenance, SendAlert
- Multi-objective rewards: Safety, Throughput, Quality, Cost, Sustainability
- Realistic physics simulation with temperature, vibration, and wear dynamics
"""

from .models import (
    ManufacturingAction,
    ManufacturingObservation,
    ManufacturingState,
    MachineStatus,
    Alert,
)
from .client import CognitiveManufacturingEnv
from .server import CognitiveManufacturingEnvironment

__version__ = "0.1.0"

__all__ = [
    "ManufacturingAction",
    "ManufacturingObservation",
    "ManufacturingState",
    "MachineStatus",
    "Alert",
    "CognitiveManufacturingEnv",
    "CognitiveManufacturingEnvironment",
]
