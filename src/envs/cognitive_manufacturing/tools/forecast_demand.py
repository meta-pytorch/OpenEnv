"""Tool for demand forecasting using time series analysis."""

from typing import Any, TYPE_CHECKING
from .base import ManufacturingTool, ToolResult

if TYPE_CHECKING:
    from ..server.environment import CognitiveManufacturingEnvironment


class ForecastDemandTool(ManufacturingTool):
    """Forecast future production demand using time series forecasting."""

    @property
    def name(self) -> str:
        return "ForecastDemand"

    @property
    def description(self) -> str:
        return "Forecast future production demand using time series analysis"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "forecast_horizon": {
                    "type": "integer",
                    "description": "Number of hours ahead to forecast (default: 168 = 1 week)",
                    "default": 168,
                },
                "confidence_level": {
                    "type": "number",
                    "description": "Confidence interval level (default: 0.95)",
                    "default": 0.95,
                },
                "include_seasonality": {
                    "type": "boolean",
                    "description": "Account for seasonal patterns (default: true)",
                    "default": True,
                },
            },
            "required": [],
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

        forecast_horizon = parameters.get("forecast_horizon", 168)
        confidence_level = parameters.get("confidence_level", 0.95)
        include_seasonality = parameters.get("include_seasonality", True)

        # Get forecast from ML service
        forecast = env.ml_service.forecast_demand(horizon=forecast_horizon)

        # Analyze trend
        if len(forecast) >= 2:
            early_avg = sum(f['demand'] for f in forecast[:len(forecast)//3]) / (len(forecast)//3)
            late_avg = sum(f['demand'] for f in forecast[-len(forecast)//3:]) / (len(forecast)//3)

            if late_avg > early_avg * 1.1:
                trend = "increasing"
            elif late_avg < early_avg * 0.9:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"

        # Check for seasonality (simple check: variance in forecast)
        import numpy as np
        demands = [f['demand'] for f in forecast]
        seasonality_detected = include_seasonality and (np.std(demands) / max(np.mean(demands), 1)) > 0.15

        # Generate recommendations based on forecast
        recommendations = []

        # Check for capacity needs
        avg_demand = sum(f['demand'] for f in forecast) / len(forecast)
        peak_demand = max(f['demand'] for f in forecast)

        # Estimate current capacity
        if hasattr(env, "production_line") and env.production_line is not None:
            total_produced = sum(m.units_produced for m in env.production_line.machines.values())
            current_capacity = total_produced / max(env._state.step_count / 36000, 1)  # Units per hour
        else:
            current_capacity = 100  # Default estimate

        if current_capacity > 0 and peak_demand > current_capacity * 0.9:
            capacity_increase = ((peak_demand / current_capacity) - 1) * 100
            recommendations.append(
                f"Increase production capacity by {capacity_increase:.0f}% to meet peak demand"
            )

        # Find low-demand periods for maintenance
        low_demand_periods = []
        for i, f in enumerate(forecast):
            if f['demand'] < avg_demand * 0.7:
                low_demand_periods.append(i + 1)

        if low_demand_periods:
            if len(low_demand_periods) >= 4:
                # Find continuous periods
                period_start = low_demand_periods[0]
                period_end = low_demand_periods[0]
                for h in low_demand_periods[1:]:
                    if h == period_end + 1:
                        period_end = h
                    else:
                        break

                if period_end - period_start >= 4:
                    recommendations.append(
                        f"Schedule maintenance during low-demand period (hours {period_start}-{period_end})"
                    )
            else:
                recommendations.append(
                    f"Consider scheduling maintenance during low-demand hours: {low_demand_periods[:3]}"
                )

        # Warning for increasing trend
        if trend == "increasing":
            recommendations.append(
                "Demand is trending upward. Consider preparing additional capacity."
            )

        # Calculate forecast accuracy if we have history
        forecast_accuracy = 0.85  # Default
        if len(env.ml_service.demand_history) >= 10:
            # Simple accuracy estimate based on data variance
            import numpy as np
            historical_values = [d['volume'] for d in env.ml_service.demand_history]
            cv = np.std(historical_values) / max(np.mean(historical_values), 1)  # Coefficient of variation
            forecast_accuracy = max(0.5, min(0.95, 1.0 - cv))

        return ToolResult(
            success=True,
            data={
                "forecast": forecast,
                "trend": trend,
                "seasonality_detected": seasonality_detected,
                "forecast_accuracy": round(forecast_accuracy, 2),
                "recommendations": recommendations,
                "summary": {
                    "avg_demand": round(avg_demand, 1),
                    "peak_demand": round(peak_demand, 1),
                    "min_demand": round(min(f['demand'] for f in forecast), 1),
                    "forecast_horizon_hours": forecast_horizon,
                },
            },
            message=f"Forecast generated: {trend} trend, avg demand {avg_demand:.0f} units/hour",
        )
