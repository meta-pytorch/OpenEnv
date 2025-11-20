"""Tool for scheduling machine maintenance."""

from typing import Any, TYPE_CHECKING
from .base import ManufacturingTool, ToolResult

if TYPE_CHECKING:
    from ..server.environment import CognitiveManufacturingEnvironment


class ScheduleMaintenanceTool(ManufacturingTool):
    """Schedule or perform maintenance on a machine."""

    @property
    def name(self) -> str:
        return "ScheduleMaintenance"

    @property
    def description(self) -> str:
        return "Schedule or perform immediate maintenance on a machine. Resets wear and health."

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "machine_id": {"type": "string", "description": "Machine identifier"},
                "maintenance_type": {
                    "type": "string",
                    "enum": ["immediate", "scheduled"],
                    "description": "Whether to perform maintenance now or schedule it",
                },
                "reason": {"type": "string", "description": "Reason for maintenance"},
            },
            "required": ["machine_id", "maintenance_type"],
        }

    def execute(self, parameters: dict[str, Any], env: "CognitiveManufacturingEnvironment") -> ToolResult:
        is_valid, error = self.validate_parameters(parameters)
        if not is_valid:
            return ToolResult(success=False, error=error)

        machine_id = parameters["machine_id"]
        maintenance_type = parameters["maintenance_type"]
        reason = parameters.get("reason", "Routine maintenance")

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

        if maintenance_type == "immediate":
            # Perform maintenance now
            old_health = machine.health_score
            old_wear = machine.wear_level
            old_status = machine.status

            # Reset machine to healthy state
            machine.perform_maintenance()

            return ToolResult(
                success=True,
                data={
                    "machine_id": machine_id,
                    "maintenance_type": "immediate",
                    "reason": reason,
                    "before": {
                        "health_score": round(old_health, 1),
                        "wear_level": round(old_wear, 3),
                        "status": old_status,
                    },
                    "after": {
                        "health_score": round(machine.health_score, 1),
                        "wear_level": round(machine.wear_level, 3),
                        "status": machine.status,
                    },
                    "estimated_downtime": 2.0,  # hours
                },
                message=f"Performed immediate maintenance on {machine_id}. Health restored from {old_health:.1f} to {machine.health_score:.1f}",
            )
        else:
            # Schedule maintenance for later
            scheduled_time = env._state.simulation_time + 8.0  # Schedule 8 hours ahead

            return ToolResult(
                success=True,
                data={
                    "machine_id": machine_id,
                    "maintenance_type": "scheduled",
                    "reason": reason,
                    "scheduled_time": scheduled_time,
                    "current_time": env._state.simulation_time,
                    "estimated_downtime": 2.0,
                },
                message=f"Scheduled maintenance for {machine_id} at time {scheduled_time:.1f}. Reason: {reason}",
            )
