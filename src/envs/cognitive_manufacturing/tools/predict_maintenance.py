"""Tool for predicting maintenance needs using ML."""

from typing import Any, TYPE_CHECKING
from .base import ManufacturingTool, ToolResult

if TYPE_CHECKING:
    from ..server.environment import CognitiveManufacturingEnvironment


class PredictMaintenanceTool(ManufacturingTool):
    """Predict when maintenance will be needed before failures occur."""

    @property
    def name(self) -> str:
        return "PredictMaintenance"

    @property
    def description(self) -> str:
        return "Use ML to predict when maintenance will be needed (predictive maintenance)"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "machine_id": {
                    "type": "string",
                    "description": "Machine identifier",
                },
                "prediction_horizon": {
                    "type": "integer",
                    "description": "Hours ahead to predict (default: 24)",
                    "minimum": 1,
                    "maximum": 168,  # 1 week
                    "default": 24,
                },
                "confidence_threshold": {
                    "type": "number",
                    "description": "Minimum confidence for recommendation (default: 0.8)",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.8,
                },
            },
            "required": ["machine_id"],
        }

    def execute(self, parameters: dict[str, Any], env: "CognitiveManufacturingEnvironment") -> ToolResult:
        is_valid, error = self.validate_parameters(parameters)
        if not is_valid:
            return ToolResult(success=False, error=error)

        # Check if ML is enabled
        if not hasattr(env, "ml_service") or env.ml_service is None:
            return ToolResult(
                success=False,
                error="ML features not enabled. Create environment with enable_ml=True",
            )

        machine_id = parameters["machine_id"]
        prediction_horizon = parameters.get("prediction_horizon", 24)
        confidence_threshold = parameters.get("confidence_threshold", 0.8)

        # Get machine (support both single and multi-machine mode)
        if hasattr(env, "production_line") and env.production_line is not None:
            # Multi-machine mode
            if machine_id not in env.production_line.machines:
                return ToolResult(success=False, error=f"Machine '{machine_id}' not found")
            machine = env.production_line.machines[machine_id]
        else:
            # Single machine mode
            machine = env.simulator_machine
            if machine.machine_id != machine_id:
                return ToolResult(success=False, error=f"Machine '{machine_id}' not found")

        # Get current machine state
        temperature = machine.temperature
        vibration = machine.vibration
        wear_level = machine.wear_level
        health_score = machine.health_score
        speed = machine.speed

        # Make prediction
        prediction = env.ml_service.predict_maintenance_need(
            temperature=temperature,
            vibration=vibration,
            wear_level=wear_level,
            health_score=health_score,
            speed=speed,
        )

        # Analyze factors
        factors = {}
        if temperature > 75:
            factors["temperature_trend"] = "high" if temperature > 80 else "increasing"
        if vibration > 0.3:
            factors["vibration_level"] = "high" if vibration > 0.4 else "elevated"
        if wear_level > 0.5:
            factors["wear_rate"] = "high" if wear_level > 0.7 else "moderate"
        if health_score < 70:
            factors["health_declining"] = True

        # Generate recommendation
        if prediction["maintenance_needed"] and prediction["confidence"] >= confidence_threshold:
            if prediction["hours_until_failure"] < 12:
                recommendation = f"Schedule immediate maintenance for {machine_id} within next 12 hours"
            elif prediction["hours_until_failure"] < prediction_horizon:
                recommendation = f"Schedule maintenance for {machine_id} within next {prediction['hours_until_failure']:.0f} hours"
            else:
                recommendation = f"Monitor {machine_id} closely, maintenance may be needed soon"
        else:
            recommendation = f"No immediate maintenance needed for {machine_id}"

        return ToolResult(
            success=True,
            data={
                "machine_id": machine_id,
                "maintenance_needed": prediction["maintenance_needed"],
                "probability": round(prediction["probability"], 3),
                "hours_until_failure": round(prediction["hours_until_failure"], 1),
                "confidence": round(prediction["confidence"], 3),
                "prediction_horizon": prediction_horizon,
                "factors": factors,
                "recommendation": recommendation,
                "current_state": {
                    "temperature": round(temperature, 1),
                    "vibration": round(vibration, 3),
                    "wear_level": round(wear_level, 3),
                    "health_score": round(health_score, 1),
                },
                "model_method": prediction.get("method", "ml"),
            },
            message=f"Maintenance prediction for {machine_id}: "
                   f"{prediction['probability']:.1%} probability, "
                   f"{prediction['hours_until_failure']:.0f}h until potential failure",
        )
