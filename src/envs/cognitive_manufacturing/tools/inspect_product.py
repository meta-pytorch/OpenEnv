"""Tool for detailed product quality inspection."""

from typing import Any, TYPE_CHECKING
import random
from datetime import datetime
from .base import ManufacturingTool, ToolResult

if TYPE_CHECKING:
    from ..server.environment import CognitiveManufacturingEnvironment


class InspectProductTool(ManufacturingTool):
    """Perform detailed quality inspection on a production unit."""

    @property
    def name(self) -> str:
        return "InspectProduct"

    @property
    def description(self) -> str:
        return "Perform detailed quality inspection on a production unit"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "unit_id": {
                    "type": "string",
                    "description": "ID of the unit to inspect",
                },
                "inspection_type": {
                    "type": "string",
                    "enum": ["visual", "dimensional", "functional", "full"],
                    "description": "Type of inspection to perform",
                    "default": "full",
                },
                "auto_log_defects": {
                    "type": "boolean",
                    "description": "Automatically log defects found (default: true)",
                    "default": True,
                },
            },
            "required": ["unit_id"],
        }

    def execute(self, parameters: dict[str, Any], env: "CognitiveManufacturingEnvironment") -> ToolResult:
        is_valid, error = self.validate_parameters(parameters)
        if not is_valid:
            return ToolResult(success=False, error=error)

        unit_id = parameters["unit_id"]
        inspection_type = parameters.get("inspection_type", "full")
        auto_log_defects = parameters.get("auto_log_defects", True)

        # Get machine state to determine quality factors
        if hasattr(env, "production_line") and env.production_line is not None:
            # Multi-machine: use average conditions
            machines = list(env.production_line.machines.values())
            avg_temp = sum(m.temperature for m in machines) / len(machines)
            avg_vibration = sum(m.vibration for m in machines) / len(machines)
            avg_wear = sum(m.wear_level for m in machines) / len(machines)
            avg_health = sum(m.health_score for m in machines) / len(machines)
        else:
            # Single machine
            machine = env.simulator_machine
            avg_temp = machine.temperature
            avg_vibration = machine.vibration
            avg_wear = machine.wear_level
            avg_health = machine.health_score

        # Simulate inspection based on machine conditions
        defects_found = []
        quality_score = 100.0

        # Visual inspection
        if inspection_type in ["visual", "full"]:
            if avg_temp > 80:
                defects_found.append({
                    "type": "surface_discoloration",
                    "severity": "minor",
                    "description": f"Temperature-related surface marks ({avg_temp:.1f}°C)",
                })
                quality_score -= 5

            if avg_wear > 60:
                defects_found.append({
                    "type": "surface_roughness",
                    "severity": "minor",
                    "description": f"Tooling wear marks (wear level: {avg_wear:.0f}%)",
                })
                quality_score -= 3

        # Dimensional inspection
        if inspection_type in ["dimensional", "full"]:
            if avg_vibration > 8:
                defects_found.append({
                    "type": "dimensional_variance",
                    "severity": "major",
                    "description": f"High vibration caused tolerance issues ({avg_vibration:.1f} mm/s)",
                })
                quality_score -= 15

            if avg_temp > 85 or avg_temp < 50:
                defects_found.append({
                    "type": "thermal_expansion",
                    "severity": "minor",
                    "description": f"Temperature outside optimal range ({avg_temp:.1f}°C)",
                })
                quality_score -= 5

        # Functional inspection
        if inspection_type in ["functional", "full"]:
            if avg_health < 70:
                defects_found.append({
                    "type": "functional_failure",
                    "severity": "critical",
                    "description": f"Machine health degradation affected functionality ({avg_health:.0f}%)",
                })
                quality_score -= 25

            # Random functional test (simulated)
            if random.random() > (avg_health / 100.0):
                defects_found.append({
                    "type": "performance_degradation",
                    "severity": "major",
                    "description": "Unit failed performance test",
                })
                quality_score -= 20

        # Ensure quality score doesn't go negative
        quality_score = max(0.0, quality_score)

        # Determine pass/fail
        qc_threshold = 70.0  # Can be configured via UpdateQCThresholds
        passed = quality_score >= qc_threshold and all(d["severity"] != "critical" for d in defects_found)

        # Auto-log defects if enabled and database available
        defect_ids = []
        if auto_log_defects and defects_found and hasattr(env, "db_manager") and env.db_manager:
            for defect in defects_found:
                # Use RecordDefect functionality
                defect_id = f"DEF-{unit_id}-{len(defect_ids)+1}"
                defect_ids.append(defect_id)
                defect["defect_id"] = defect_id

        # Inspection details
        inspection_details = {
            "unit_id": unit_id,
            "inspection_type": inspection_type,
            "inspected_at": datetime.utcnow().isoformat(),
            "inspector": "automated_qc",
            "test_results": {
                "visual": inspection_type in ["visual", "full"],
                "dimensional": inspection_type in ["dimensional", "full"],
                "functional": inspection_type in ["functional", "full"],
            },
            "conditions": {
                "temperature": round(avg_temp, 1),
                "vibration": round(avg_vibration, 2),
                "wear_level": round(avg_wear, 1),
                "health_score": round(avg_health, 1),
            },
        }

        message = f"Inspection {'PASSED' if passed else 'FAILED'}: {unit_id} - Quality score: {quality_score:.0f}%"
        if defects_found:
            message += f", {len(defects_found)} defects found"

        return ToolResult(
            success=True,
            data={
                "passed": passed,
                "quality_score": round(quality_score, 1),
                "defects_found": defects_found,
                "defect_count": len(defects_found),
                "inspection_details": inspection_details,
                "defect_ids": defect_ids if auto_log_defects else [],
            },
            message=message,
        )
