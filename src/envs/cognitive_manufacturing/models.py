"""
Data models for Cognitive Manufacturing Environment.

Defines Action, Observation, and State types for the manufacturing environment.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

from core.env_server.types import Action, Observation, State


# ============================================================================
# Actions
# ============================================================================


@dataclass(kw_only=True)
class ManufacturingAction(Action):
    """
    Action for cognitive manufacturing environment.

    Uses tool-calling paradigm where agent specifies which tool to use
    and provides parameters for that tool.

    Attributes:
        tool_name: Name of the tool to execute (e.g., 'ReadSensors', 'AdjustSpeed')
        parameters: Tool-specific parameters as a dictionary
        reasoning: Optional reasoning for this action (for logging/debugging)
    """

    tool_name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    reasoning: str | None = None


# ============================================================================
# Observations
# ============================================================================


@dataclass
class MachineStatus:
    """Status of a single machine."""

    machine_id: str
    status: Literal["running", "idle", "maintenance", "failed"]
    temperature: float  # Celsius
    vibration: float  # arbitrary units (0-1 scale)
    speed: float  # percentage of max speed (0-100)
    health_score: float  # 0-100, lower is worse
    wear_level: float  # 0-1 scale, cumulative wear
    production_output: float  # units per hour


@dataclass
class Alert:
    """Alert or warning from the system."""

    alert_id: str
    severity: Literal["info", "warning", "critical"]
    machine_id: str | None
    message: str
    timestamp: float
    category: str = "other"  # temperature, vibration, wear, health, production, safety, other


@dataclass(kw_only=True)
class ManufacturingObservation(Observation):
    """
    Observation from cognitive manufacturing environment.

    Inherits from Observation:
        - done: bool (episode terminated?)
        - reward: float (immediate reward)
        - metadata: dict (additional info)

    Additional attributes:
        tool_result: Result from the executed tool
        machine_status: Current status of the machine
        alerts: List of active alerts/warnings
        simulation_time: Current simulation time in hours
    """

    tool_result: dict[str, Any] = field(default_factory=dict)
    machine_status: MachineStatus | None = None
    alerts: list[Alert] = field(default_factory=list)
    simulation_time: float = 0.0


# ============================================================================
# State
# ============================================================================


@dataclass
class ManufacturingState(State):
    """
    Extended state for manufacturing environment.

    Inherits from State:
        - episode_id: str
        - step_count: int

    Additional attributes for manufacturing-specific state tracking.
    """

    simulation_time: float = 0.0  # Hours since episode start
    total_units_produced: int = 0
    total_defects: int = 0
    total_downtime: float = 0.0  # Hours
    maintenance_scheduled: bool = False
    maintenance_time: float | None = None

# ============================================================================
# Phase 1: Multi-Machine Production Line Models
# ============================================================================


@dataclass
class ProductUnit:
    """A single unit moving through the production line."""

    unit_id: str
    created_at: float  # Simulation time when created
    stage: str  # prep, assembly, finishing, qc
    quality: float = 1.0  # 0-1 scale
    passed_qc: bool | None = None
    defect_type: str | None = None


@dataclass
class Buffer:
    """Material buffer between machines."""

    buffer_id: str  # e.g., "M1_M2"
    capacity: int = 10
    current_level: int = 0
    units: list[ProductUnit] = field(default_factory=list)

    def add_unit(self, unit: ProductUnit) -> bool:
        """Add unit to buffer if space available."""
        if self.current_level < self.capacity:
            self.units.append(unit)
            self.current_level += 1
            return True
        return False

    def remove_units(self, count: int) -> list[ProductUnit]:
        """Remove and return units from buffer."""
        count = min(count, self.current_level)
        removed = self.units[:count]
        self.units = self.units[count:]
        self.current_level -= count
        return removed

    def peek(self, count: int = 1) -> list[ProductUnit]:
        """View units without removing."""
        return self.units[:count]


@dataclass
class LineMetrics:
    """Production line performance metrics."""

    total_produced: int = 0
    total_defects: int = 0
    total_scrapped: int = 0
    throughput_rate: float = 0.0  # units/hour
    qc_pass_rate: float = 1.0  # 0-1
    line_efficiency: float = 1.0  # 0-1
    bottleneck_machine: str | None = None
    total_energy: float = 0.0  # kWh
    energy_per_unit: float = 0.0  # kWh/unit
