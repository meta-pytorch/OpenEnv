"""Reward computation for cognitive manufacturing environment.

Multi-objective reward function balancing:
- Safety (highest priority)
- Throughput (production efficiency)
- Quality (defect minimization)
- Cost (operational expenses)
- Sustainability (energy efficiency)
"""

from typing import TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from .simulator import SimulatedMachine
    from ..models import ManufacturingState


@dataclass
class RewardWeights:
    """Configurable weights for multi-objective optimization."""

    safety: float = 10.0  # Highest priority
    throughput: float = 1.0
    quality: float = 2.0
    cost: float = 0.5
    sustainability: float = 0.3


class RewardCalculator:
    """Computes multi-objective rewards for manufacturing environment."""

    def __init__(self, weights: RewardWeights | None = None):
        """Initialize reward calculator with configurable weights.

        Args:
            weights: Custom weights for objectives. Uses defaults if None.
        """
        self.weights = weights or RewardWeights()

    def compute_reward(
        self,
        machine: "SimulatedMachine",
        state: "ManufacturingState",
        dt: float,
    ) -> tuple[float, dict[str, float]]:
        """Compute total reward and component breakdown.

        Args:
            machine: The simulated machine
            state: Current environment state
            dt: Time step duration

        Returns:
            Tuple of (total_reward, component_breakdown)
        """
        # Compute individual components
        safety_reward = self._compute_safety_reward(machine)
        throughput_reward = self._compute_throughput_reward(machine, dt)
        quality_reward = self._compute_quality_reward(machine, dt)
        cost_reward = self._compute_cost_reward(machine, dt)
        sustainability_reward = self._compute_sustainability_reward(machine, dt)

        # Weighted sum
        total_reward = (
            self.weights.safety * safety_reward
            + self.weights.throughput * throughput_reward
            + self.weights.quality * quality_reward
            + self.weights.cost * cost_reward
            + self.weights.sustainability * sustainability_reward
        )

        # Component breakdown for analysis
        breakdown = {
            "safety": safety_reward,
            "throughput": throughput_reward,
            "quality": quality_reward,
            "cost": cost_reward,
            "sustainability": sustainability_reward,
            "total": total_reward,
        }

        return total_reward, breakdown

    def _compute_safety_reward(self, machine: "SimulatedMachine") -> float:
        """Safety reward: penalize dangerous conditions.

        Returns:
            Reward in range [-10, 0]. Zero is safe, -10 is critical.
        """
        reward = 0.0

        # Critical penalties
        if machine.status == "failed":
            reward -= 10.0  # Failure is worst case
        elif machine.temperature > 95.0:
            reward -= 5.0  # Overheating is dangerous
        elif machine.temperature > 85.0:
            reward -= 2.0  # High temperature warning

        # Vibration safety
        if machine.vibration > 0.8:
            reward -= 3.0  # Excessive vibration
        elif machine.vibration > 0.6:
            reward -= 1.0  # High vibration warning

        # Health-based safety
        if machine.health_score < 30.0:
            reward -= 4.0  # Critical health
        elif machine.health_score < 50.0:
            reward -= 1.0  # Low health warning

        return reward

    def _compute_throughput_reward(self, machine: "SimulatedMachine", dt: float) -> float:
        """Throughput reward: reward production output.

        Returns:
            Reward in range [0, 1] based on production rate.
        """
        # Reward is proportional to production output
        # Max production is ~10 units/hour at full speed
        # Normalize to [0, 1] range
        max_production_rate = 10.0 * dt  # Max possible in this timestep
        actual_production = machine.production_output * dt

        if max_production_rate > 0:
            normalized_throughput = min(actual_production / max_production_rate, 1.0)
        else:
            normalized_throughput = 0.0

        return normalized_throughput

    def _compute_quality_reward(self, machine: "SimulatedMachine", dt: float) -> float:
        """Quality reward: penalize defects.

        Returns:
            Reward in range [-1, 0]. Zero defects = 0, many defects = -1.
        """
        # Penalize based on defect rate
        # Defect rate increases with high temperature and wear
        defect_probability = machine.defect_rate

        # Expected defects in this timestep
        expected_defects = defect_probability * machine.production_output * dt

        # Normalize penalty (assume max 2 defects per timestep is worst case)
        penalty = -min(expected_defects / 2.0, 1.0)

        return penalty

    def _compute_cost_reward(self, machine: "SimulatedMachine", dt: float) -> float:
        """Cost reward: penalize operational costs.

        Returns:
            Reward in range [-1, 0] based on normalized costs.
        """
        # Cost factors:
        # 1. Energy consumption (proportional to speed)
        # 2. Maintenance costs (proportional to wear)
        # 3. Failure costs (high penalty)

        energy_cost = (machine.speed / 100.0) * 0.1  # $0.1 per hour at full speed
        maintenance_cost = machine.wear_level * 0.05  # Increases with wear
        failure_cost = 1.0 if machine.status == "failed" else 0.0

        total_cost = (energy_cost + maintenance_cost + failure_cost) * dt

        # Normalize to [-1, 0] range (assume max cost of $1.15 per hour)
        max_cost = 1.15 * dt
        normalized_cost = -min(total_cost / max_cost, 1.0) if max_cost > 0 else 0.0

        return normalized_cost

    def _compute_sustainability_reward(self, machine: "SimulatedMachine", dt: float) -> float:
        """Sustainability reward: reward energy efficiency.

        Returns:
            Reward in range [-1, 1] based on energy efficiency.
        """
        # Energy efficiency = production output / energy consumed
        # Higher efficiency = better sustainability

        if machine.speed > 0:
            energy_consumed = machine.speed / 100.0  # Normalized energy
            production_output = machine.production_output

            if energy_consumed > 0:
                efficiency = production_output / energy_consumed
                # Normalize: efficiency of 10 units/energy = 1.0 reward
                normalized_efficiency = min(efficiency / 10.0, 1.0)
            else:
                normalized_efficiency = 0.0
        else:
            # Idle machine: slight penalty for not producing
            normalized_efficiency = -0.1

        return normalized_efficiency


def compute_cumulative_metrics(state: "ManufacturingState") -> dict[str, float]:
    """Compute cumulative performance metrics.

    Args:
        state: Current environment state

    Returns:
        Dictionary of cumulative metrics
    """
    return {
        "total_reward": state.cumulative_reward,
        "simulation_time": state.simulation_time,
        "avg_reward_per_hour": state.cumulative_reward / max(state.simulation_time, 1.0),
        "total_alerts": len(state.alerts),
        "critical_alerts": sum(1 for alert in state.alerts if alert.severity == "critical"),
    }
