"""Tool for adjusting machine speed."""

from typing import Any, TYPE_CHECKING
from .base import ManufacturingTool, ToolResult

if TYPE_CHECKING:
    from ..server.environment import CognitiveManufacturingEnvironment


class AdjustSpeedTool(ManufacturingTool):
    """Adjust the operating speed of a machine."""

    @property
    def name(self) -> str:
        return "AdjustSpeed"

    @property
    def description(self) -> str:
        return "Adjust machine operating speed (0-100%). Higher speeds increase throughput but may increase wear and temperature."

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "machine_id": {"type": "string", "description": "Machine identifier"},
                "target_speed": {
                    "type": "number",
                    "description": "Target speed percentage (0-100)",
                    "minimum": 0,
                    "maximum": 100,
                },
            },
            "required": ["machine_id", "target_speed"],
        }

    def execute(self, parameters: dict[str, Any], env: "CognitiveManufacturingEnvironment") -> ToolResult:
        is_valid, error = self.validate_parameters(parameters)
        if not is_valid:
            return ToolResult(success=False, error=error)

        machine_id = parameters["machine_id"]
        target_speed = parameters["target_speed"]

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

        # Check if machine can have its speed adjusted
        if machine.status == "failed":
            return ToolResult(
                success=False,
                error=f"Cannot adjust speed: machine is in failed state. Maintenance required.",
            )

        if machine.status == "maintenance":
            return ToolResult(
                success=False,
                error=f"Cannot adjust speed: machine is under maintenance",
            )

        old_speed = machine.speed
        machine.set_speed(target_speed)

        # Update status based on new speed
        if target_speed > 0:
            machine.status = "running"
        else:
            machine.status = "idle"

        return ToolResult(
            success=True,
            data={
                "machine_id": machine_id,
                "old_speed": round(old_speed, 1),
                "new_speed": round(target_speed, 1),
                "status": machine.status,
            },
            message=f"Adjusted {machine_id} speed from {old_speed:.1f}% to {target_speed:.1f}%",
        )
