"""Tool for adjusting machine power modes."""

from typing import Any, TYPE_CHECKING
from .base import ManufacturingTool, ToolResult

if TYPE_CHECKING:
    from ..server.environment import CognitiveManufacturingEnvironment


class SetPowerModeTool(ManufacturingTool):
    """Adjust machine power consumption modes."""

    @property
    def name(self) -> str:
        return "SetPowerMode"

    @property
    def description(self) -> str:
        return "Adjust machine power consumption modes for energy optimization"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "machine_id": {
                    "type": "string",
                    "description": "Machine ID or 'all' for all machines",
                },
                "power_mode": {
                    "type": "string",
                    "enum": ["eco", "normal", "high_performance"],
                    "description": "Power mode: eco (low power), normal (balanced), high_performance (max power)",
                },
                "schedule": {
                    "type": "object",
                    "description": "Optional schedule for power mode changes",
                    "properties": {
                        "start_hour": {"type": "integer"},
                        "end_hour": {"type": "integer"},
                    },
                },
            },
            "required": ["machine_id", "power_mode"],
        }

    def execute(self, parameters: dict[str, Any], env: "CognitiveManufacturingEnvironment") -> ToolResult:
        is_valid, error = self.validate_parameters(parameters)
        if not is_valid:
            return ToolResult(success=False, error=error)

        machine_id = parameters["machine_id"]
        power_mode = parameters["power_mode"]
        schedule = parameters.get("schedule")

        # Get machines to update
        if hasattr(env, "production_line") and env.production_line is not None:
            if machine_id == "all":
                machines_to_update = list(env.production_line.machines.values())
                machine_ids = list(env.production_line.machines.keys())
            elif machine_id in env.production_line.machines:
                machines_to_update = [env.production_line.machines[machine_id]]
                machine_ids = [machine_id]
            else:
                return ToolResult(success=False, error=f"Machine '{machine_id}' not found")
        else:
            if machine_id == "all" or machine_id == env.machine_id:
                machines_to_update = [env.simulator_machine]
                machine_ids = [env.machine_id]
            else:
                return ToolResult(success=False, error=f"Machine '{machine_id}' not found")

        # Power mode effects
        mode_effects = {
            "eco": {
                "power_savings_pct": 20.0,
                "performance_impact_pct": -10.0,  # 10% slower
                "description": "Reduced power consumption, slightly lower throughput",
            },
            "normal": {
                "power_savings_pct": 0.0,
                "performance_impact_pct": 0.0,
                "description": "Balanced power and performance",
            },
            "high_performance": {
                "power_savings_pct": -20.0,  # 20% more power
                "performance_impact_pct": 15.0,  # 15% faster
                "description": "Maximum performance, higher power consumption",
            },
        }

        # Apply power mode to machines
        for machine in machines_to_update:
            machine._power_mode = power_mode

            # Adjust speed if in eco or high_performance mode
            if power_mode == "eco":
                # Eco mode: slightly reduce max speed
                machine._power_mode_speed_multiplier = 0.9
            elif power_mode == "high_performance":
                # High performance: allow higher speeds
                machine._power_mode_speed_multiplier = 1.15
            else:  # normal
                machine._power_mode_speed_multiplier = 1.0

        # Get mode effects
        effects = mode_effects[power_mode]

        # Log the change if database available
        if hasattr(env, "db_manager") and env.db_manager:
            try:
                if hasattr(env, "current_run_id") and env.current_run_id:
                    env.db_manager.record_event(
                        run_id=env.current_run_id,
                        machine_id=machine_id,
                        simulation_time=env._state.simulation_time,
                        event_type="power_mode_change",
                        data={
                            "machines": machine_ids,
                            "power_mode": power_mode,
                            "schedule": schedule,
                            "effects": effects,
                        }
                    )
            except Exception:
                pass

        # Generate impact summary
        annual_hours = 8760  # Hours per year
        avg_power_kw = 25.0  # Average power per machine
        electricity_rate = 0.12  # $/kWh

        savings_per_machine = (avg_power_kw * annual_hours * abs(effects["power_savings_pct"]) / 100.0) * electricity_rate
        total_annual_savings = savings_per_machine * len(machines_to_update)

        if effects["power_savings_pct"] > 0:
            impact_msg = f"Expected annual savings: ${total_annual_savings:.0f}"
        elif effects["power_savings_pct"] < 0:
            impact_msg = f"Expected annual cost increase: ${abs(total_annual_savings):.0f}"
        else:
            impact_msg = "No change in power consumption"

        message = f"Power mode set to '{power_mode}' for {len(machines_to_update)} machine(s)"

        return ToolResult(
            success=True,
            data={
                "applied": True,
                "power_mode": power_mode,
                "machines_affected": machine_ids,
                "expected_savings_pct": effects["power_savings_pct"],
                "performance_impact_pct": effects["performance_impact_pct"],
                "description": effects["description"],
                "impact_summary": {
                    "annual_savings_usd": round(total_annual_savings, 2) if effects["power_savings_pct"] > 0 else round(-total_annual_savings, 2),
                    "monthly_savings_usd": round(total_annual_savings / 12, 2) if effects["power_savings_pct"] > 0 else round(-total_annual_savings / 12, 2),
                    "impact_message": impact_msg,
                },
                "schedule": schedule if schedule else "immediate",
            },
            message=message,
        )
