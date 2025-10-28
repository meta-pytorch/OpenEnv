"""Tool for reading sensor data from machines."""

from typing import Any, TYPE_CHECKING
from .base import ManufacturingTool, ToolResult

if TYPE_CHECKING:
    from ..server.environment import CognitiveManufacturingEnvironment


class ReadSensorsTool(ManufacturingTool):
    """Read real-time sensor data from a machine."""

    @property
    def name(self) -> str:
        return "ReadSensors"

    @property
    def description(self) -> str:
        return "Read current sensor values (temperature, vibration, speed) from a machine"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "machine_id": {"type": "string", "description": "Machine identifier"},
                "sensors": {"type": "string", "enum": ["all", "temperature", "vibration", "speed"]},
            },
            "required": ["machine_id"],
        }

    def execute(self, parameters: dict[str, Any], env: "CognitiveManufacturingEnvironment") -> ToolResult:
        is_valid, error = self.validate_parameters(parameters)
        if not is_valid:
            return ToolResult(success=False, error=error)

        machine_id = parameters["machine_id"]
        sensors = parameters.get("sensors", "all")

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

        if sensors == "all":
            sensor_data = {
                "temperature": round(machine.temperature, 2),
                "vibration": round(machine.vibration, 3),
                "speed": round(machine.speed, 1),
            }
            message = f"Read all sensors from {machine_id}"
        else:
            value = getattr(machine, sensors)
            sensor_data = {sensors: round(value, 2)}
            message = f"Read {sensors} from {machine_id}: {value:.2f}"

        return ToolResult(
            success=True,
            data={"machine_id": machine_id, "sensors": sensor_data, "timestamp": env._state.simulation_time},
            message=message,
        )
