"""Tool for optimizing production schedules."""

from typing import Any, TYPE_CHECKING
from datetime import datetime, timedelta
from .base import ManufacturingTool, ToolResult

if TYPE_CHECKING:
    from ..server.environment import CognitiveManufacturingEnvironment


class OptimizeProductionScheduleTool(ManufacturingTool):
    """Optimize production schedule for efficiency."""

    @property
    def name(self) -> str:
        return "OptimizeProductionSchedule"

    @property
    def description(self) -> str:
        return "Optimize production schedule to maximize efficiency and minimize costs"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "optimization_goal": {
                    "type": "string",
                    "enum": ["throughput", "cost", "quality", "energy", "balanced"],
                    "description": "Primary optimization objective",
                    "default": "balanced",
                },
                "time_horizon": {
                    "type": "integer",
                    "description": "Planning horizon in hours (default: 168 = 1 week)",
                    "default": 168,
                },
                "constraints": {
                    "type": "object",
                    "description": "Constraints (min_quality, max_energy, etc.)",
                    "default": {},
                },
                "include_maintenance": {
                    "type": "boolean",
                    "description": "Include scheduled maintenance in optimization",
                    "default": True,
                },
            },
            "required": [],
        }

    def execute(self, parameters: dict[str, Any], env: "CognitiveManufacturingEnvironment") -> ToolResult:
        is_valid, error = self.validate_parameters(parameters)
        if not is_valid:
            return ToolResult(success=False, error=error)

        optimization_goal = parameters.get("optimization_goal", "balanced")
        time_horizon = parameters.get("time_horizon", 168)
        constraints = parameters.get("constraints", {})
        include_maintenance = parameters.get("include_maintenance", True)

        # Get current machine states
        if hasattr(env, "production_line") and env.production_line is not None:
            machines = list(env.production_line.machines.keys())
            current_health = {m_id: m.health_score for m_id, m in env.production_line.machines.items()}
        else:
            machines = [env.machine_id]
            current_health = {env.machine_id: env.simulator_machine.health_score}

        # Get demand forecast if available
        forecast_demand = []
        if hasattr(env, "ml_service") and env.ml_service:
            try:
                forecast = env.ml_service.forecast_demand(horizon=time_horizon)
                forecast_demand = [f["demand"] for f in forecast]
            except:
                # Use default flat demand
                avg_demand = 100
                forecast_demand = [avg_demand] * time_horizon

        if not forecast_demand:
            forecast_demand = [100] * time_horizon  # Default demand

        # Generate optimized schedule
        schedule = []
        current_time = datetime.utcnow()

        # Optimization strategy based on goal
        if optimization_goal == "throughput":
            base_speed = 90  # High speed
            base_power_mode = "high_performance"
        elif optimization_goal == "energy":
            base_speed = 70  # Moderate speed
            base_power_mode = "eco"
        elif optimization_goal == "quality":
            base_speed = 60  # Slower for quality
            base_power_mode = "normal"
        elif optimization_goal == "cost":
            base_speed = 75  # Balanced
            base_power_mode = "eco"
        else:  # balanced
            base_speed = 75
            base_power_mode = "normal"

        # Schedule maintenance windows for machines needing it
        maintenance_schedule = []
        if include_maintenance:
            for m_id, health in current_health.items():
                if health < 70:  # Schedule maintenance for unhealthy machines
                    # Find low-demand period
                    min_demand_hour = forecast_demand.index(min(forecast_demand))
                    maintenance_time = current_time + timedelta(hours=min_demand_hour)
                    maintenance_schedule.append({
                        "machine_id": m_id,
                        "scheduled_time": maintenance_time.isoformat(),
                        "duration_hours": 4,
                        "reason": f"Preventive maintenance (health: {health:.0f}%)",
                    })

        # Generate hourly schedule
        for hour in range(min(24, time_horizon)):  # First 24 hours detailed schedule
            schedule_time = current_time + timedelta(hours=hour)
            demand = forecast_demand[hour] if hour < len(forecast_demand) else 100

            # Adjust speed based on demand
            if demand > 120:
                speed = min(100, base_speed + 10)
                power_mode = "high_performance"
            elif demand < 80:
                speed = max(50, base_speed - 10)
                power_mode = "eco"
            else:
                speed = base_speed
                power_mode = base_power_mode

            # Check if maintenance scheduled for this hour
            maint_this_hour = [m for m in maintenance_schedule
                              if abs((datetime.fromisoformat(m["scheduled_time"]) - schedule_time).total_seconds()) < 3600]

            schedule.append({
                "hour": hour,
                "time": schedule_time.isoformat(),
                "recommended_speed": speed,
                "power_mode": power_mode,
                "forecasted_demand": round(demand, 1),
                "maintenance_planned": len(maint_this_hour) > 0,
                "machines_in_maintenance": [m["machine_id"] for m in maint_this_hour],
            })

        # Calculate expected improvement
        baseline_throughput = sum(forecast_demand[:24]) * 0.75  # 75% efficiency baseline
        optimized_throughput = sum(s["recommended_speed"] / 100.0 * s["forecasted_demand"]
                                  for s in schedule if not s["maintenance_planned"])

        improvement_pct = ((optimized_throughput - baseline_throughput) / baseline_throughput * 100) if baseline_throughput > 0 else 0

        # Generate recommended actions
        recommended_actions = []

        # Speed adjustments
        if schedule[0]["recommended_speed"] != 75:
            recommended_actions.append({
                "tool_name": "AdjustSpeed",
                "parameters": {"machine_id": "all", "new_speed": schedule[0]["recommended_speed"]},
                "reason": f"Optimize for {optimization_goal}",
                "priority": "high",
            })

        # Power mode
        if schedule[0]["power_mode"] != "normal":
            recommended_actions.append({
                "tool_name": "SetPowerMode",
                "parameters": {"machine_id": "all", "power_mode": schedule[0]["power_mode"]},
                "reason": "Energy optimization",
                "priority": "medium",
            })

        # Maintenance
        for maint in maintenance_schedule:
            recommended_actions.append({
                "tool_name": "ScheduleMaintenance",
                "parameters": {
                    "machine_id": maint["machine_id"],
                    "schedule_for": maint["scheduled_time"],
                },
                "reason": maint["reason"],
                "priority": "high" if current_health[maint["machine_id"]] < 50 else "medium",
            })

        # Schedule details summary
        schedule_summary = {
            "total_hours_planned": time_horizon,
            "detailed_hours": len(schedule),
            "maintenance_windows": len(maintenance_schedule),
            "avg_recommended_speed": round(sum(s["recommended_speed"] for s in schedule) / len(schedule), 1),
            "total_forecasted_demand": round(sum(s["forecasted_demand"] for s in schedule), 1),
        }

        message = f"Production schedule optimized for '{optimization_goal}' over {time_horizon}h"

        return ToolResult(
            success=True,
            data={
                "optimized_schedule": schedule,
                "expected_improvement_pct": round(improvement_pct, 1),
                "schedule_details": schedule_summary,
                "maintenance_schedule": maintenance_schedule,
                "recommended_actions": recommended_actions,
                "optimization_summary": {
                    "goal": optimization_goal,
                    "time_horizon_hours": time_horizon,
                    "constraints_applied": constraints,
                    "machines_optimized": machines,
                    "expected_throughput": round(optimized_throughput, 1),
                    "baseline_throughput": round(baseline_throughput, 1),
                },
            },
            message=message,
        )
