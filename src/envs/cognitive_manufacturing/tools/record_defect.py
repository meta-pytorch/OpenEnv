"""Tool for recording product defects."""

from typing import Any, TYPE_CHECKING
import uuid
from datetime import datetime
from .base import ManufacturingTool, ToolResult

if TYPE_CHECKING:
    from ..server.environment import CognitiveManufacturingEnvironment


class RecordDefectTool(ManufacturingTool):
    """Record a product defect in the system."""

    @property
    def name(self) -> str:
        return "RecordDefect"

    @property
    def description(self) -> str:
        return "Record a product defect in the quality management system"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "unit_id": {
                    "type": "string",
                    "description": "ID of the defective unit",
                },
                "defect_type": {
                    "type": "string",
                    "description": "Type of defect (e.g., 'surface_damage', 'dimensional_error')",
                },
                "severity": {
                    "type": "string",
                    "enum": ["minor", "major", "critical"],
                    "description": "Severity level of the defect",
                },
                "description": {
                    "type": "string",
                    "description": "Detailed description of the defect",
                },
                "machine_id": {
                    "type": "string",
                    "description": "Machine that produced the defective unit (optional)",
                },
            },
            "required": ["unit_id", "defect_type", "severity", "description"],
        }

    def execute(self, parameters: dict[str, Any], env: "CognitiveManufacturingEnvironment") -> ToolResult:
        is_valid, error = self.validate_parameters(parameters)
        if not is_valid:
            return ToolResult(success=False, error=error)

        unit_id = parameters["unit_id"]
        defect_type = parameters["defect_type"]
        severity = parameters["severity"]
        description = parameters["description"]
        machine_id = parameters.get("machine_id")

        # Generate defect ID
        defect_id = str(uuid.uuid4())[:8]
        recorded_at = datetime.utcnow()

        # Determine if immediate action is required
        requires_action = severity == "critical"

        # Store defect in database if available
        if hasattr(env, "db_manager") and env.db_manager:
            try:
                # We would store in defects table, but for now just log as event
                if hasattr(env, "current_run_id") and env.current_run_id:
                    env.db_manager.record_event(
                        run_id=env.current_run_id,
                        machine_id=machine_id or "unknown",
                        simulation_time=env._state.simulation_time,
                        event_type="defect_recorded",
                        data={
                            "defect_id": defect_id,
                            "unit_id": unit_id,
                            "defect_type": defect_type,
                            "severity": severity,
                            "description": description,
                            "recorded_at": recorded_at.isoformat(),
                        }
                    )
            except Exception as e:
                # Continue even if database recording fails
                pass

        # Send alert if critical
        if severity == "critical":
            alert_msg = f"CRITICAL DEFECT: {defect_type} in {unit_id} - {description}"
            env._alerts.append({
                "type": "critical_defect",
                "message": alert_msg,
                "timestamp": recorded_at.isoformat(),
                "defect_id": defect_id,
            })

        # Determine recommended actions
        recommended_actions = []
        if severity == "critical":
            recommended_actions.append("Halt production and inspect machine")
            recommended_actions.append("Quarantine all recent units from same machine")
            if machine_id:
                recommended_actions.append(f"Schedule immediate maintenance for {machine_id}")
        elif severity == "major":
            recommended_actions.append("Increase inspection frequency")
            recommended_actions.append("Review production parameters")
        else:  # minor
            recommended_actions.append("Monitor for pattern of similar defects")

        message = f"Defect recorded: {defect_id} - {severity.upper()} {defect_type} in {unit_id}"

        return ToolResult(
            success=True,
            data={
                "defect_id": defect_id,
                "recorded_at": recorded_at.isoformat(),
                "requires_action": requires_action,
                "recommended_actions": recommended_actions,
                "alert_sent": severity == "critical",
                "details": {
                    "unit_id": unit_id,
                    "defect_type": defect_type,
                    "severity": severity,
                    "description": description,
                    "machine_id": machine_id,
                },
            },
            message=message,
        )
