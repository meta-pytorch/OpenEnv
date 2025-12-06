"""Tool for predicting product quality using ML."""

from typing import Any, TYPE_CHECKING
from .base import ManufacturingTool, ToolResult

if TYPE_CHECKING:
    from ..server.environment import CognitiveManufacturingEnvironment


class PredictQualityTool(ManufacturingTool):
    """Predict product quality before completion."""

    @property
    def name(self) -> str:
        return "PredictQuality"

    @property
    def description(self) -> str:
        return "Predict product quality based on current production parameters"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "machine_id": {
                    "type": "string",
                    "description": "Machine identifier",
                },
                "parameters": {
                    "type": "object",
                    "description": "Production parameters to evaluate (optional, uses current if not provided)",
                    "properties": {
                        "speed": {"type": "number"},
                        "temperature": {"type": "number"},
                    },
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
        custom_params = parameters.get("parameters", {})

        # Get machine
        if hasattr(env, "production_line") and env.production_line is not None:
            if machine_id not in env.production_line.machines:
                return ToolResult(success=False, error=f"Machine '{machine_id}' not found")
            machine = env.production_line.machines[machine_id]
        else:
            machine = env.simulator_machine
            if machine.machine_id != machine_id:
                return ToolResult(success=False, error=f"Machine '{machine_id}' not found")

        # Use custom parameters or current state
        speed = custom_params.get('speed', machine.speed)
        temperature = custom_params.get('temperature', machine.temperature)
        vibration = machine.vibration
        wear_level = machine.wear_level

        # Predict quality
        prediction = env.ml_service.predict_quality(
            speed=speed,
            temperature=temperature,
            vibration=vibration,
            wear_level=wear_level,
        )

        # Analyze quality factors
        quality_factors = {}
        if speed > 85:
            quality_factors['speed'] = 'too_high'
        elif speed < 40:
            quality_factors['speed'] = 'too_low'
        else:
            quality_factors['speed'] = 'optimal'

        if temperature > 80:
            quality_factors['temperature'] = 'too_high'
        elif temperature < 60:
            quality_factors['temperature'] = 'acceptable'
        else:
            quality_factors['temperature'] = 'optimal'

        if vibration > 0.4:
            quality_factors['vibration'] = 'high'
        elif vibration > 0.25:
            quality_factors['vibration'] = 'moderate'
        else:
            quality_factors['vibration'] = 'acceptable'

        # Generate improvement suggestions
        improvement_suggestions = []
        if temperature > 75:
            improvement_suggestions.append({
                'parameter': 'temperature',
                'current': round(temperature, 1),
                'recommended': 72.0,
                'quality_gain': round((80 - temperature) / 100, 2),
            })
        if speed > 85:
            improvement_suggestions.append({
                'parameter': 'speed',
                'current': round(speed, 1),
                'recommended': 80.0,
                'quality_gain': round((speed - 80) / 200, 2),
            })

        return ToolResult(
            success=True,
            data={
                "machine_id": machine_id,
                "predicted_quality": round(prediction['predicted_quality'], 3),
                "confidence_interval": [
                    round(prediction['confidence_interval'][0], 3),
                    round(prediction['confidence_interval'][1], 3),
                ],
                "pass_probability": round(prediction['pass_probability'], 3),
                "quality_factors": quality_factors,
                "improvement_suggestions": improvement_suggestions,
                "current_parameters": {
                    "speed": round(speed, 1),
                    "temperature": round(temperature, 1),
                    "vibration": round(vibration, 3),
                    "wear_level": round(wear_level, 3),
                },
                "model_method": prediction.get('method', 'ml'),
            },
            message=f"Quality prediction for {machine_id}: {prediction['predicted_quality']:.1%} "
                   f"(pass probability: {prediction['pass_probability']:.1%})",
        )
