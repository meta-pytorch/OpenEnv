"""Tool for monitoring energy usage and costs."""

from typing import Any, TYPE_CHECKING
from .base import ManufacturingTool, ToolResult

if TYPE_CHECKING:
    from ..server.environment import CognitiveManufacturingEnvironment


class MonitorEnergyUsageTool(ManufacturingTool):
    """Monitor power consumption and energy costs."""

    @property
    def name(self) -> str:
        return "MonitorEnergyUsage"

    @property
    def description(self) -> str:
        return "Monitor power consumption, energy usage, and costs"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "time_window": {
                    "type": "integer",
                    "description": "Time window in hours to analyze (default: 24)",
                    "default": 24,
                },
                "breakdown_by": {
                    "type": "string",
                    "enum": ["machine", "operation", "total"],
                    "description": "How to break down energy usage",
                    "default": "machine",
                },
                "include_cost": {
                    "type": "boolean",
                    "description": "Include cost estimates",
                    "default": True,
                },
            },
            "required": [],
        }

    def execute(self, parameters: dict[str, Any], env: "CognitiveManufacturingEnvironment") -> ToolResult:
        is_valid, error = self.validate_parameters(parameters)
        if not is_valid:
            return ToolResult(success=False, error=error)

        time_window = parameters.get("time_window", 24)
        breakdown_by = parameters.get("breakdown_by", "machine")
        include_cost = parameters.get("include_cost", True)

        # Energy pricing ($/kWh)
        electricity_rate = 0.12  # $0.12 per kWh

        # Get machine states
        if hasattr(env, "production_line") and env.production_line is not None:
            machines = env.production_line.machines
        else:
            machines = {env.machine_id: env.simulator_machine}

        # Calculate energy usage per machine
        breakdown = {}
        total_kwh = 0.0

        for machine_id, machine in machines.items():
            # Energy model based on speed and operating state
            # Idle: 5kW, Running: 10-50kW depending on speed
            if machine.speed == 0:
                power_kw = 5.0  # Idle power
            else:
                # Power scales with speed (10kW at 0% to 50kW at 100%)
                power_kw = 10.0 + (machine.speed / 100.0) * 40.0

            # Calculate energy for the time window
            kwh = power_kw * time_window

            # Apply power mode multiplier if set
            power_mode = getattr(machine, "_power_mode", "normal")
            if power_mode == "eco":
                kwh *= 0.8  # 20% savings
            elif power_mode == "high_performance":
                kwh *= 1.2  # 20% more consumption

            total_kwh += kwh

            breakdown[machine_id] = {
                "current_power_kw": round(power_kw, 2),
                "energy_kwh": round(kwh, 2),
                "power_mode": power_mode,
                "operating_hours": time_window,
                "average_speed": round(machine.speed, 1),
            }

            if include_cost:
                breakdown[machine_id]["cost_usd"] = round(kwh * electricity_rate, 2)

        # Calculate total cost
        total_cost = total_kwh * electricity_rate if include_cost else 0.0

        # Calculate efficiency score (lower energy per unit produced = better)
        if hasattr(env, "production_line"):
            total_units = sum(m.units_produced for m in machines.values())
        else:
            total_units = env.simulator_machine.units_produced

        if total_units > 0:
            energy_per_unit = total_kwh / total_units
            # Efficiency score: 100 = excellent (< 1 kWh/unit), 0 = poor (> 10 kWh/unit)
            efficiency_score = max(0.0, min(100.0, 100.0 * (1.0 - min(energy_per_unit / 10.0, 1.0))))
        else:
            energy_per_unit = 0.0
            efficiency_score = 50.0  # Neutral score if no production

        # Generate recommendations
        recommendations = []

        # Check for high power mode usage
        high_power_machines = [m_id for m_id, m in breakdown.items() if m.get("power_mode") == "high_performance"]
        if high_power_machines:
            recommendations.append(f"Consider switching {', '.join(high_power_machines)} to normal mode to save energy")

        # Check for idle machines with high power
        idle_machines = [m_id for m_id, info in breakdown.items()
                        if info["average_speed"] == 0 and info["current_power_kw"] > 3]
        if idle_machines:
            recommendations.append(f"Idle machines consuming power: {', '.join(idle_machines)} - consider eco mode")

        # Check efficiency
        if efficiency_score < 50:
            recommendations.append("Low energy efficiency - review production parameters and maintenance needs")
        elif efficiency_score > 80:
            recommendations.append("Excellent energy efficiency - current settings are optimal")

        # Peak demand warning
        total_power_kw = sum(info["current_power_kw"] for info in breakdown.values())
        if total_power_kw > 150:
            recommendations.append(f"High power demand ({total_power_kw:.0f} kW) - consider load balancing")

        message = f"Energy usage: {total_kwh:.1f} kWh over {time_window}h"
        if include_cost:
            message += f" (${total_cost:.2f})"

        return ToolResult(
            success=True,
            data={
                "total_kwh": round(total_kwh, 2),
                "cost": round(total_cost, 2) if include_cost else None,
                "breakdown": breakdown if breakdown_by in ["machine", "total"] else {},
                "efficiency_score": round(efficiency_score, 1),
                "energy_per_unit": round(energy_per_unit, 3),
                "recommendations": recommendations,
                "summary": {
                    "time_window_hours": time_window,
                    "total_power_kw": round(total_power_kw, 2),
                    "average_power_kw": round(total_kwh / time_window, 2),
                    "units_produced": total_units,
                    "electricity_rate": electricity_rate,
                },
            },
            message=message,
        )
