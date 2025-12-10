"""Tool for getting production line status."""

from typing import Any, TYPE_CHECKING
from .base import ManufacturingTool, ToolResult

if TYPE_CHECKING:
    from ..server.environment import CognitiveManufacturingEnvironment


class GetLineStatusTool(ManufacturingTool):
    """Get status of entire production line."""

    @property
    def name(self) -> str:
        return "GetLineStatus"

    @property
    def description(self) -> str:
        return "Get comprehensive status of the entire production line including all machines, buffers, and line metrics"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "include_buffers": {
                    "type": "boolean",
                    "description": "Include buffer level details (default: true)",
                    "default": True,
                },
                "include_metrics": {
                    "type": "boolean",
                    "description": "Include line performance metrics (default: true)",
                    "default": True,
                },
            },
            "required": [],
        }

    def execute(self, parameters: dict[str, Any], env: "CognitiveManufacturingEnvironment") -> ToolResult:
        is_valid, error = self.validate_parameters(parameters)
        if not is_valid:
            return ToolResult(success=False, error=error)

        include_buffers = parameters.get("include_buffers", True)
        include_metrics = parameters.get("include_metrics", True)

        # Check if environment has production line simulator
        if not hasattr(env, "production_line") or env.production_line is None:
            # Fall back to single machine mode
            machine = env.simulator_machine
            return ToolResult(
                success=True,
                data={
                    "mode": "single_machine",
                    "machine": {
                        "machine_id": machine.machine_id,
                        "status": machine.status,
                        "speed": machine.speed,
                        "temperature": machine.temperature,
                        "health_score": machine.health_score,
                        "production_output": machine.production_output,
                        "units_produced": machine.units_produced,
                    },
                },
                message="Single machine mode - no production line configured",
            )

        # Get full line status from production line simulator
        line_status = env.production_line.get_line_status()

        # Build response based on what's requested
        response_data = {
            "mode": "production_line",
            "machines": line_status["machines"],
        }

        if include_buffers:
            response_data["buffers"] = line_status["buffers"]

        if include_metrics:
            response_data["metrics"] = line_status["metrics"]

        # Generate summary message
        bottleneck = line_status["metrics"].get("bottleneck", "unknown")
        throughput = line_status["metrics"].get("throughput_rate", 0.0)
        efficiency = line_status["metrics"].get("line_efficiency", 0.0)

        message = (
            f"Production line status: {len(line_status['machines'])} machines, "
            f"bottleneck={bottleneck}, throughput={throughput:.2f} units/hr, "
            f"efficiency={efficiency:.1%}"
        )

        return ToolResult(
            success=True,
            data=response_data,
            message=message,
        )
