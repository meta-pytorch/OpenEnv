"""
Production Machine Simulator.

Simulates a single production machine with realistic physics:
- Temperature dynamics (heat generation, dissipation)
- Vibration (based on wear and speed)
- Wear accumulation over time
- Failure probability based on conditions
- Production output
"""

import random
from dataclasses import dataclass
from typing import Literal


# Constants
AMBIENT_TEMP = 20.0  # Celsius
MAX_TEMP = 100.0  # Celsius
CRITICAL_TEMP = 95.0  # Celsius - triggers critical alerts
WARNING_TEMP = 85.0  # Celsius - triggers warnings
MAX_SPEED = 100.0  # Maximum speed percentage
CRITICAL_VIBRATION = 0.8  # Critical vibration level
WARNING_VIBRATION = 0.6  # Warning vibration level
BASE_PRODUCTION_RATE = 100.0  # Units per hour at max speed
MAINTENANCE_DURATION = 2.0  # Hours


@dataclass
class SimulatedMachine:
    """
    Simulates a single production machine with physical properties.

    Attributes:
        machine_id: Unique identifier
        status: Current operational status
        temperature: Current temperature in Celsius
        vibration: Vibration level (0-1)
        speed: Current speed (0-100%)
        wear_level: Cumulative wear (0-1, higher is worse)
        health_score: Overall health (0-100)
        last_maintenance_time: When maintenance last occurred
        total_runtime: Total hours of operation
        units_produced: Total units produced
        defects_produced: Total defective units
    """

    machine_id: str = "M1"
    status: Literal["running", "idle", "maintenance", "failed"] = "idle"
    temperature: float = AMBIENT_TEMP
    vibration: float = 0.1
    speed: float = 0.0
    wear_level: float = 0.0
    health_score: float = 100.0
    last_maintenance_time: float = 0.0
    total_runtime: float = 0.0
    units_produced: int = 0
    defects_produced: int = 0

    def step(self, dt: float) -> dict:
        """
        Advance simulation by dt hours.

        Args:
            dt: Time step in hours

        Returns:
            Dictionary with step results (units produced, defects, etc.)
        """
        if self.status == "running":
            # Update physical properties
            self._update_temperature(dt)
            self._update_vibration()
            self._update_wear(dt)
            self._update_health_score()

            # Check for failures
            if self._check_failure():
                self.status = "failed"
                return {
                    "units_produced": 0,
                    "defects": 0,
                    "failed": True,
                    "failure_reason": "Random failure based on conditions",
                }

            # Produce output
            units, defects = self._produce_output(dt)
            self.units_produced += units
            self.defects_produced += defects
            self.total_runtime += dt

            return {
                "units_produced": units,
                "defects": defects,
                "failed": False,
            }

        elif self.status == "idle":
            # Machine is idle - slowly cool down
            self._cool_down(dt)
            return {"units_produced": 0, "defects": 0, "failed": False}

        elif self.status == "maintenance":
            # Machine under maintenance - can't produce
            return {"units_produced": 0, "defects": 0, "failed": False}

        elif self.status == "failed":
            # Machine failed - no production
            return {"units_produced": 0, "defects": 0, "failed": True}

        return {"units_produced": 0, "defects": 0, "failed": False}

    def _update_temperature(self, dt: float):
        """Update temperature based on operation."""
        # Heat generation proportional to speed
        heat_generation = (self.speed / MAX_SPEED) * 10.0  # degrees/hour

        # Additional heat from poor conditions
        if self.wear_level > 0.5:
            heat_generation *= 1.0 + (self.wear_level - 0.5)

        # Heat dissipation (Newton's cooling)
        heat_dissipation = (self.temperature - AMBIENT_TEMP) * 0.15

        # Update temperature
        self.temperature += (heat_generation - heat_dissipation) * dt

        # Clamp temperature
        self.temperature = max(AMBIENT_TEMP, min(MAX_TEMP, self.temperature))

    def _cool_down(self, dt: float):
        """Cool down when idle."""
        cooling_rate = (self.temperature - AMBIENT_TEMP) * 0.3
        self.temperature -= cooling_rate * dt
        self.temperature = max(AMBIENT_TEMP, self.temperature)

    def _update_vibration(self):
        """Update vibration based on wear and speed."""
        base_vibration = 0.05

        # Wear increases vibration
        wear_factor = self.wear_level * 0.4

        # Speed increases vibration
        speed_factor = (self.speed / MAX_SPEED) * 0.2

        # Temperature above optimal increases vibration
        if self.temperature > 70:
            temp_factor = (self.temperature - 70) / 30 * 0.2
        else:
            temp_factor = 0.0

        self.vibration = base_vibration + wear_factor + speed_factor + temp_factor

        # Add small random noise
        self.vibration += random.uniform(-0.02, 0.02)

        # Clamp vibration
        self.vibration = max(0.0, min(1.0, self.vibration))

    def _update_wear(self, dt: float):
        """Update wear level based on operating conditions."""
        if self.status != "running":
            return

        # Base wear rate
        wear_rate = 0.001  # per hour

        # High temperature increases wear
        if self.temperature > WARNING_TEMP:
            wear_rate *= 1.0 + ((self.temperature - WARNING_TEMP) / 15) * 0.5

        # High speed increases wear
        speed_factor = self.speed / MAX_SPEED
        wear_rate *= 1.0 + (speed_factor * 0.3)

        # High vibration increases wear
        if self.vibration > WARNING_VIBRATION:
            wear_rate *= 1.0 + (self.vibration - WARNING_VIBRATION) * 0.5

        # Update wear
        self.wear_level += wear_rate * dt

        # Clamp wear
        self.wear_level = min(1.0, self.wear_level)

    def _update_health_score(self):
        """Calculate overall health score (0-100)."""
        # Start at 100
        health = 100.0

        # Wear reduces health
        health -= self.wear_level * 50

        # High temperature reduces health
        if self.temperature > WARNING_TEMP:
            health -= ((self.temperature - WARNING_TEMP) / 15) * 20

        # High vibration reduces health
        if self.vibration > WARNING_VIBRATION:
            health -= ((self.vibration - WARNING_VIBRATION) / 0.4) * 15

        # Clamp health
        self.health_score = max(0.0, min(100.0, health))

    def _check_failure(self) -> bool:
        """Check if machine should fail based on conditions."""
        # Base failure probability (very low)
        failure_prob = 0.00001  # per hour

        # Wear dramatically increases failure probability
        if self.wear_level > 0.7:
            failure_prob *= 1.0 + ((self.wear_level - 0.7) / 0.3) * 50

        # Critical temperature increases failure probability
        if self.temperature > CRITICAL_TEMP:
            failure_prob *= 20

        # Critical vibration increases failure probability
        if self.vibration > CRITICAL_VIBRATION:
            failure_prob *= 10

        # Very low health dramatically increases failure
        if self.health_score < 20:
            failure_prob *= 15

        return random.random() < failure_prob

    def _produce_output(self, dt: float) -> tuple[int, int]:
        """
        Calculate production output and defects.

        Returns:
            (units_produced, defects)
        """
        # Base rate proportional to speed
        rate = BASE_PRODUCTION_RATE * (self.speed / MAX_SPEED)

        # Poor conditions reduce output
        if self.temperature > WARNING_TEMP:
            rate *= 0.8
        if self.vibration > WARNING_VIBRATION:
            rate *= 0.85
        if self.wear_level > 0.6:
            rate *= 0.9

        # Calculate units
        units = int(rate * dt)

        # Calculate defects using defect_rate property
        defects = int(units * self.defect_rate)

        return units, defects

    def set_speed(self, new_speed: float):
        """Set machine speed (0-100)."""
        self.speed = max(0.0, min(MAX_SPEED, new_speed))
        if self.speed > 0 and self.status == "idle":
            self.status = "running"
        elif self.speed == 0 and self.status == "running":
            self.status = "idle"

    def perform_maintenance(self, simulation_time: float):
        """Perform maintenance on the machine."""
        self.status = "maintenance"
        self.wear_level = 0.0
        self.temperature = AMBIENT_TEMP
        self.vibration = 0.05
        self.health_score = 100.0
        self.last_maintenance_time = simulation_time

    def complete_maintenance(self):
        """Complete maintenance and return to idle."""
        self.status = "idle"

    def repair_failure(self, simulation_time: float):
        """Repair a failed machine (emergency maintenance)."""
        self.status = "maintenance"
        self.wear_level = 0.1  # Not perfect after emergency repair
        self.temperature = AMBIENT_TEMP
        self.vibration = 0.1
        self.health_score = 90.0
        self.last_maintenance_time = simulation_time

    def get_hours_since_maintenance(self, current_time: float) -> float:
        """Get hours since last maintenance."""
        return current_time - self.last_maintenance_time

    @property
    def production_output(self) -> float:
        """Get current production rate in units per hour."""
        if self.status != "running":
            return 0.0
        # Production rate is proportional to speed
        # Max production at 100% speed is 10 units/hour
        return (self.speed / MAX_SPEED) * 10.0

    @property
    def defect_rate(self) -> float:
        """Get current defect rate (0-1)."""
        rate = 0.01  # Base 1% defect rate
        if self.temperature > WARNING_TEMP:
            rate += 0.02
        if self.vibration > WARNING_VIBRATION:
            rate += 0.03
        if self.wear_level > 0.5:
            rate += 0.02
        return min(rate, 1.0)
