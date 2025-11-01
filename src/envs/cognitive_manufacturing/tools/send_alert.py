"""Tool for sending alerts to operators."""

import uuid
from typing import Any, TYPE_CHECKING
from .base import ManufacturingTool, ToolResult
from ..models import Alert

if TYPE_CHECKING:
    from ..server.environment import CognitiveManufacturingEnvironment


class SendAlertTool(ManufacturingTool):
    """Send an alert notification to operators."""

    @property
    def name(self) -> str:
        return "SendAlert"

    @property
    def description(self) -> str:
        return "Send an alert notification to operators about machine status or issues"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "machine_id": {"type": "string", "description": "Machine identifier"},
                "severity": {
                    "type": "string",
                    "enum": ["info", "warning", "critical"],
                    "description": "Alert severity level",
                },
                "message": {"type": "string", "description": "Alert message content"},
                "category": {
                    "type": "string",
                    "enum": ["temperature", "vibration", "wear", "health", "production", "safety", "other"],
                    "description": "Alert category",
                },
            },
            "required": ["machine_id", "severity", "message"],
        }

    def execute(self, parameters: dict[str, Any], env: "CognitiveManufacturingEnvironment") -> ToolResult:
        is_valid, error = self.validate_parameters(parameters)
        if not is_valid:
            return ToolResult(success=False, error=error)

        machine_id = parameters["machine_id"]
        severity = parameters["severity"]
        message = parameters["message"]
        category = parameters.get("category", "other")
        machine = env.simulator_machine

        if machine.machine_id != machine_id:
            return ToolResult(success=False, error=f"Machine '{machine_id}' not found")

        # Create alert
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            machine_id=machine_id,
            severity=severity,
            message=message,
            category=category,
            timestamp=env._state.simulation_time,
        )

        # Add to environment's alert list
        env._alerts.append(alert)

        return ToolResult(
            success=True,
            data={
                "alert_id": alert.alert_id,
                "machine_id": machine_id,
                "severity": severity,
                "category": category,
                "message": message,
                "timestamp": alert.timestamp,
            },
            message=f"Sent {severity} alert for {machine_id}: {message}",
        )
