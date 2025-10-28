"""Tool for updating quality control thresholds."""

from typing import Any, TYPE_CHECKING
from .base import ManufacturingTool, ToolResult

if TYPE_CHECKING:
    from ..server.environment import CognitiveManufacturingEnvironment


class UpdateQCThresholdsTool(ManufacturingTool):
    """Update quality control thresholds and parameters."""

    @property
    def name(self) -> str:
        return "UpdateQCThresholds"

    @property
    def description(self) -> str:
        return "Update quality control thresholds and inspection parameters"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "threshold_type": {
                    "type": "string",
                    "enum": ["quality_min", "defect_rate_max", "inspection_frequency", "temperature_max", "vibration_max"],
                    "description": "Type of threshold to update",
                },
                "new_value": {
                    "type": "number",
                    "description": "New threshold value",
                },
                "apply_to": {
                    "type": "string",
                    "description": "Apply to 'all' machines or specific machine_id",
                    "default": "all",
                },
            },
            "required": ["threshold_type", "new_value"],
        }

    def execute(self, parameters: dict[str, Any], env: "CognitiveManufacturingEnvironment") -> ToolResult:
        is_valid, error = self.validate_parameters(parameters)
        if not is_valid:
            return ToolResult(success=False, error=error)

        threshold_type = parameters["threshold_type"]
        new_value = parameters["new_value"]
        apply_to = parameters.get("apply_to", "all")

        # Initialize QC config if not exists
        if not hasattr(env, "_qc_config"):
            env._qc_config = {
                "quality_min": 70.0,
                "defect_rate_max": 0.05,  # 5%
                "inspection_frequency": 10,  # Every 10 units
                "temperature_max": 85.0,
                "vibration_max": 10.0,
            }

        # Get old value
        old_value = env._qc_config.get(threshold_type, 0.0)

        # Validate new value
        if threshold_type == "quality_min":
            if not (0 <= new_value <= 100):
                return ToolResult(success=False, error="quality_min must be between 0 and 100")
        elif threshold_type == "defect_rate_max":
            if not (0 <= new_value <= 1):
                return ToolResult(success=False, error="defect_rate_max must be between 0 and 1")
        elif threshold_type == "inspection_frequency":
            if new_value < 1:
                return ToolResult(success=False, error="inspection_frequency must be >= 1")

        # Update threshold
        env._qc_config[threshold_type] = new_value

        # Determine affected machines
        if hasattr(env, "production_line") and env.production_line is not None:
            if apply_to == "all":
                affected_machines = list(env.production_line.machines.keys())
            else:
                affected_machines = [apply_to] if apply_to in env.production_line.machines else []
        else:
            affected_machines = [env.machine_id] if apply_to == "all" or apply_to == env.machine_id else []

        # Calculate impact
        change_pct = ((new_value - old_value) / old_value * 100) if old_value != 0 else 100.0

        # Determine impact description
        impact_description = []
        if threshold_type == "quality_min":
            if new_value > old_value:
                impact_description.append("Higher quality standards - may reduce throughput")
                impact_description.append("Expected defect rate decrease")
            else:
                impact_description.append("Lower quality standards - may increase throughput")
                impact_description.append("Expected defect rate may increase")
        elif threshold_type == "defect_rate_max":
            if new_value > old_value:
                impact_description.append("More tolerant of defects")
            else:
                impact_description.append("Less tolerant of defects - stricter quality control")
        elif threshold_type == "inspection_frequency":
            if new_value > old_value:
                impact_description.append("Less frequent inspections - faster throughput")
            else:
                impact_description.append("More frequent inspections - better quality detection")

        # Log the change if database available
        if hasattr(env, "db_manager") and env.db_manager:
            try:
                if hasattr(env, "current_run_id") and env.current_run_id:
                    env.db_manager.record_event(
                        run_id=env.current_run_id,
                        machine_id="system",
                        simulation_time=env._state.simulation_time,
                        event_type="qc_threshold_update",
                        data={
                            "threshold_type": threshold_type,
                            "old_value": old_value,
                            "new_value": new_value,
                            "apply_to": apply_to,
                            "affected_machines": affected_machines,
                        }
                    )
            except Exception:
                pass

        message = f"Updated {threshold_type}: {old_value} → {new_value} ({change_pct:+.1f}%)"

        return ToolResult(
            success=True,
            data={
                "updated": True,
                "threshold_type": threshold_type,
                "old_value": round(old_value, 2),
                "new_value": round(new_value, 2),
                "change_percent": round(change_pct, 1),
                "affected_machines": affected_machines,
                "impact_description": impact_description,
                "current_qc_config": env._qc_config.copy(),
            },
            message=message,
        )
