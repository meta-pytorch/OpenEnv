"""Tool for checking machine health status."""

from typing import Any, TYPE_CHECKING
from .base import ManufacturingTool, ToolResult

if TYPE_CHECKING:
    from ..server.environment import CognitiveManufacturingEnvironment


class CheckHealthTool(ManufacturingTool):
    """Check machine health and get diagnostic information."""

    @property
    def name(self) -> str:
        return "CheckHealth"

    @property
    def description(self) -> str:
        return "Get machine health score and diagnostic information"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"machine_id": {"type": "string", "description": "Machine identifier"}},
            "required": ["machine_id"],
        }

    def execute(self, parameters: dict[str, Any], env: "CognitiveManufacturingEnvironment") -> ToolResult:
        is_valid, error = self.validate_parameters(parameters)
        if not is_valid:
            return ToolResult(success=False, error=error)

        machine_id = parameters["machine_id"]

        # Get machine (support both single and multi-machine mode)
        if hasattr(env, "production_line") and env.production_line is not None:
            # Multi-machine mode: look up machine in production line
            if machine_id not in env.production_line.machines:
                return ToolResult(success=False, error=f"Machine '{machine_id}' not found")
            machine = env.production_line.machines[machine_id]
        else:
            # Single machine mode
            machine = env.simulator_machine
            if machine.machine_id != machine_id:
                return ToolResult(success=False, error=f"Machine '{machine_id}' not found")

        health_status = "FAILED" if machine.status == "failed" else (
            "GOOD" if machine.health_score >= 80 else
            "FAIR" if machine.health_score >= 60 else
            "POOR" if machine.health_score >= 40 else "CRITICAL"
        )

        diagnostics = {
            "health_score": round(machine.health_score, 1),
            "status": health_status,
            "wear_level": round(machine.wear_level, 3),
            "hours_since_maintenance": round(machine.get_hours_since_maintenance(env._state.simulation_time), 1),
            "issues": [],
            "recommendations": [],
        }

        if machine.temperature > 85:
            diagnostics["issues"].append(f"High temperature: {machine.temperature:.1f}C")
        if machine.vibration > 0.6:
            diagnostics["issues"].append(f"High vibration: {machine.vibration:.2f}")
        if machine.wear_level > 0.6:
            diagnostics["issues"].append(f"High wear: {machine.wear_level:.2%}")
        if machine.status == "failed":
            diagnostics["issues"].append("Machine has failed")

        if machine.wear_level > 0.7:
            diagnostics["recommendations"].append("Schedule maintenance soon")
        if machine.temperature > 90:
            diagnostics["recommendations"].append("Reduce speed to cool down")
        if machine.status == "failed":
            diagnostics["recommendations"].append("Perform emergency repair")

        return ToolResult(
            success=True,
            data=diagnostics,
            message=f"Health check for {machine_id}: {health_status} (score: {machine.health_score:.1f})",
        )
