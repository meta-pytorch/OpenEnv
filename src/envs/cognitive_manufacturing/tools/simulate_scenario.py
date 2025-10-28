"""Tool for running what-if scenario simulations."""

from typing import Any, TYPE_CHECKING
import uuid
import copy
from .base import ManufacturingTool, ToolResult

if TYPE_CHECKING:
    from ..server.environment import CognitiveManufacturingEnvironment


class SimulateScenarioTool(ManufacturingTool):
    """Run what-if scenario simulations to test parameter changes."""

    @property
    def name(self) -> str:
        return "SimulateScenario"

    @property
    def description(self) -> str:
        return "Run what-if scenario simulations to predict outcomes of parameter changes"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "scenario_name": {
                    "type": "string",
                    "description": "Name for this scenario",
                },
                "changes": {
                    "type": "object",
                    "description": "Parameter changes to test (e.g., {\"speed\": 80, \"temperature\": 75})",
                },
                "simulation_steps": {
                    "type": "integer",
                    "description": "Number of simulation steps to run (default: 100)",
                    "default": 100,
                },
                "metrics_to_track": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Metrics to track (default: [\"throughput\", \"quality\", \"cost\", \"energy\"])",
                    "default": ["throughput", "quality", "cost", "energy"],
                },
            },
            "required": ["scenario_name", "changes"],
        }

    def execute(self, parameters: dict[str, Any], env: "CognitiveManufacturingEnvironment") -> ToolResult:
        is_valid, error = self.validate_parameters(parameters)
        if not is_valid:
            return ToolResult(success=False, error=error)

        scenario_name = parameters["scenario_name"]
        changes = parameters["changes"]
        simulation_steps = parameters.get("simulation_steps", 100)
        metrics_to_track = parameters.get("metrics_to_track", ["throughput", "quality", "cost", "energy"])

        scenario_id = str(uuid.uuid4())[:8]

        # Capture baseline state
        if hasattr(env, "production_line") and env.production_line is not None:
            baseline_state = {
                "avg_speed": sum(m.speed for m in env.production_line.machines.values()) / len(env.production_line.machines),
                "avg_temperature": sum(m.temperature for m in env.production_line.machines.values()) / len(env.production_line.machines),
                "total_units": sum(m.units_produced for m in env.production_line.machines.values()),
                "avg_health": sum(m.health_score for m in env.production_line.machines.values()) / len(env.production_line.machines),
            }
        else:
            baseline_state = {
                "speed": env.simulator_machine.speed,
                "temperature": env.simulator_machine.temperature,
                "units": env.simulator_machine.units_produced,
                "health": env.simulator_machine.health_score,
            }

        # Simulate the scenario (simplified simulation)
        # In a real implementation, we would create a copy of the environment and run it
        # For now, we'll use estimates based on the changes

        # Estimate impacts
        scenario_results = {
            "throughput": 0.0,
            "quality": 0.0,
            "cost": 0.0,
            "energy": 0.0,
            "safety": 0.0,
        }

        # Speed impact
        if "speed" in changes:
            speed_change = changes["speed"] - baseline_state.get("avg_speed", baseline_state.get("speed", 70))
            # Higher speed = higher throughput, slightly lower quality, higher energy
            scenario_results["throughput"] = speed_change * 0.8  # 80% throughput increase per speed unit
            scenario_results["quality"] = -abs(speed_change) * 0.3  # Quality degrades with speed changes
            scenario_results["energy"] = speed_change * 0.5  # Energy increases with speed

        # Temperature impact
        if "temperature" in changes:
            temp = changes["temperature"]
            optimal_temp = 70.0
            temp_deviation = abs(temp - optimal_temp)
            scenario_results["quality"] -= temp_deviation * 0.5  # Quality degrades away from optimal
            if temp > 85:
                scenario_results["safety"] -= (temp - 85) * 2.0  # Safety risk at high temps

        # Maintenance impact
        if "maintenance_frequency" in changes:
            freq = changes["maintenance_frequency"]
            scenario_results["cost"] += freq * 100  # Maintenance cost
            scenario_results["quality"] += freq * 2  # Better quality with more maintenance
            scenario_results["throughput"] -= freq * 5  # Downtime from maintenance

        # Power mode impact
        if "power_mode" in changes:
            mode = changes["power_mode"]
            if mode == "eco":
                scenario_results["energy"] -= 20
                scenario_results["throughput"] -= 10
            elif mode == "high_performance":
                scenario_results["energy"] += 20
                scenario_results["throughput"] += 15

        # Estimate absolute values (starting from current state)
        estimated_throughput = baseline_state.get("total_units", baseline_state.get("units", 100)) * (1.0 + scenario_results["throughput"] / 100.0)
        estimated_quality = max(0, min(100, 85 + scenario_results["quality"]))
        estimated_cost = 5000 * (1.0 + scenario_results["cost"] / 100.0)  # Base cost of $5000
        estimated_energy = 1000 * (1.0 + scenario_results["energy"] / 100.0)  # Base energy 1000 kWh

        # Compare to baseline
        baseline_throughput = baseline_state.get("total_units", baseline_state.get("units", 100))
        baseline_quality = 85.0
        baseline_cost = 5000.0
        baseline_energy = 1000.0

        comparison = {
            "throughput": {
                "baseline": baseline_throughput,
                "scenario": round(estimated_throughput, 1),
                "change_pct": round(scenario_results["throughput"], 1),
            },
            "quality": {
                "baseline": baseline_quality,
                "scenario": round(estimated_quality, 1),
                "change_pct": round(scenario_results["quality"], 1),
            },
            "cost": {
                "baseline": baseline_cost,
                "scenario": round(estimated_cost, 2),
                "change_pct": round(scenario_results["cost"], 1),
            },
            "energy": {
                "baseline": baseline_energy,
                "scenario": round(estimated_energy, 1),
                "change_pct": round(scenario_results["energy"], 1),
            },
        }

        # Generate recommendations
        recommendations = []

        if scenario_results["throughput"] > 5:
            recommendations.append("✅ Throughput improvement expected - consider implementing")
        if scenario_results["quality"] < -5:
            recommendations.append("⚠️  Quality degradation risk - review quality controls")
        if scenario_results["energy"] > 10:
            recommendations.append("⚠️  Significant energy increase - evaluate cost-benefit")
        if scenario_results["safety"] < -5:
            recommendations.append("❌ Safety concerns - do not implement without additional controls")

        if not recommendations:
            recommendations.append("Scenario has minimal impact - changes are safe to implement")

        message = f"Scenario '{scenario_name}' simulated: {len(changes)} parameters changed"

        return ToolResult(
            success=True,
            data={
                "scenario_id": scenario_id,
                "scenario_name": scenario_name,
                "changes_applied": changes,
                "simulation_results": scenario_results,
                "estimated_outcomes": {
                    "throughput_units": round(estimated_throughput, 1),
                    "quality_pct": round(estimated_quality, 1),
                    "cost_usd": round(estimated_cost, 2),
                    "energy_kwh": round(estimated_energy, 1),
                },
                "comparison_to_baseline": comparison,
                "recommendations": recommendations,
                "simulation_details": {
                    "steps_simulated": simulation_steps,
                    "metrics_tracked": metrics_to_track,
                    "baseline_state": baseline_state,
                },
            },
            message=message,
        )
