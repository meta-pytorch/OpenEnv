"""Tool for detecting anomalies in sensor patterns using ML."""

from typing import Any, TYPE_CHECKING
from .base import ManufacturingTool, ToolResult

if TYPE_CHECKING:
    from ..server.environment import CognitiveManufacturingEnvironment


class DetectAnomalyTool(ManufacturingTool):
    """Detect abnormal sensor patterns using ML."""

    @property
    def name(self) -> str:
        return "DetectAnomaly"

    @property
    def description(self) -> str:
        return "Detect anomalies in sensor patterns for early warning of equipment issues"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "machine_id": {
                    "type": "string",
                    "description": "Machine identifier",
                },
                "window_size": {
                    "type": "integer",
                    "description": "Number of recent readings to analyze (default: 50)",
                    "minimum": 10,
                    "maximum": 500,
                    "default": 50,
                },
                "sensitivity": {
                    "type": "number",
                    "description": "Detection sensitivity 0-1, higher=more sensitive (default: 0.5)",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.5,
                },
            },
            "required": ["machine_id"],
        }

    def execute(self, parameters: dict[str, Any], env: "CognitiveManufacturingEnvironment") -> ToolResult:
        is_valid, error = self.validate_parameters(parameters)
        if not is_valid:
            return ToolResult(success=False, error=error)

        if not hasattr(env, "ml_service") or env.ml_service is None:
            return ToolResult(
                success=False,
                error="ML features not enabled. Create environment with enable_ml=True",
            )

        machine_id = parameters["machine_id"]
        window_size = parameters.get("window_size", 50)
        sensitivity = parameters.get("sensitivity", 0.5)

        # Get machine
        if hasattr(env, "production_line") and env.production_line is not None:
            if machine_id not in env.production_line.machines:
                return ToolResult(success=False, error=f"Machine '{machine_id}' not found")
            machine = env.production_line.machines[machine_id]
        else:
            machine = env.simulator_machine
            if machine.machine_id != machine_id:
                return ToolResult(success=False, error=f"Machine '{machine_id}' not found")

        # For demo, create synthetic recent history
        sensor_readings = []
        for i in range(min(window_size, 50)):
            sensor_readings.append({
                'temperature': machine.temperature + (i * 0.1 - 2.5),
                'vibration': machine.vibration + (i * 0.01 - 0.25),
                'speed': machine.speed,
                'production_output': machine.production_output,
            })

        # Detect anomalies
        result = env.ml_service.detect_anomalies(sensor_readings)

        # Generate explanation
        explanation = {}
        for sensor in result.get('anomalous_sensors', []):
            if sensor == 'temperature':
                explanation[sensor] = f"Temperature deviation detected: {machine.temperature:.1f}°C"
            elif sensor == 'vibration':
                explanation[sensor] = f"Vibration spike detected: {machine.vibration:.3f}"
            elif sensor == 'speed':
                explanation[sensor] = f"Unusual speed pattern: {machine.speed:.1f}%"

        # Recommended action
        if result['anomaly_detected']:
            if result['severity'] == 'high':
                recommended_action = f"Immediately investigate {machine_id} - reduce speed and schedule maintenance"
            elif result['severity'] == 'medium':
                recommended_action = f"Monitor {machine_id} closely, prepare for maintenance"
            else:
                recommended_action = f"Continue monitoring {machine_id}"
        else:
            recommended_action = "No action needed, all sensors within normal range"

        return ToolResult(
            success=True,
            data={
                "machine_id": machine_id,
                "anomaly_detected": result['anomaly_detected'],
                "anomaly_score": round(result['anomaly_score'], 3),
                "anomalous_sensors": result['anomalous_sensors'],
                "explanation": explanation,
                "severity": result['severity'],
                "recommended_action": recommended_action,
                "window_size": len(sensor_readings),
                "sensitivity": sensitivity,
            },
            message=f"Anomaly detection for {machine_id}: "
                   f"{'ANOMALY' if result['anomaly_detected'] else 'NORMAL'} "
                   f"(score: {result['anomaly_score']:.2f}, severity: {result['severity']})",
        )
