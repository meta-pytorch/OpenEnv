"""Tool for optimizing production line speed."""

from typing import Any, TYPE_CHECKING
from .base import ManufacturingTool, ToolResult

if TYPE_CHECKING:
    from ..server.environment import CognitiveManufacturingEnvironment


class OptimizeLineSpeedTool(ManufacturingTool):
    """Automatically balance all machine speeds for optimal throughput."""

    @property
    def name(self) -> str:
        return "OptimizeLineSpeed"

    @property
    def description(self) -> str:
        return "Automatically optimize all machine speeds to maximize throughput, quality, energy efficiency, or balance"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "optimization_target": {
                    "type": "string",
                    "enum": ["throughput", "quality", "energy", "balanced"],
                    "description": "Optimization objective: maximize throughput, maximize quality, minimize energy, or balance all factors",
                    "default": "balanced",
                },
                "constraints": {
                    "type": "object",
                    "description": "Optional speed constraints per machine (e.g., {'M1': {'min': 30, 'max': 80}})",
                    "properties": {
                        "M1": {
                            "type": "object",
                            "properties": {
                                "min": {"type": "number", "minimum": 0, "maximum": 100},
                                "max": {"type": "number", "minimum": 0, "maximum": 100},
                            },
                        },
                        "M2": {
                            "type": "object",
                            "properties": {
                                "min": {"type": "number", "minimum": 0, "maximum": 100},
                                "max": {"type": "number", "minimum": 0, "maximum": 100},
                            },
                        },
                        "M3": {
                            "type": "object",
                            "properties": {
                                "min": {"type": "number", "minimum": 0, "maximum": 100},
                                "max": {"type": "number", "minimum": 0, "maximum": 100},
                            },
                        },
                        "M4": {
                            "type": "object",
                            "properties": {
                                "min": {"type": "number", "minimum": 0, "maximum": 100},
                                "max": {"type": "number", "minimum": 0, "maximum": 100},
                            },
                        },
                    },
                },
            },
            "required": [],
        }

    def execute(self, parameters: dict[str, Any], env: "CognitiveManufacturingEnvironment") -> ToolResult:
        is_valid, error = self.validate_parameters(parameters)
        if not is_valid:
            return ToolResult(success=False, error=error)

        optimization_target = parameters.get("optimization_target", "balanced")
        constraints = parameters.get("constraints", {})

        # Check if environment has production line simulator
        if not hasattr(env, "production_line") or env.production_line is None:
            return ToolResult(
                success=False,
                error="OptimizeLineSpeed only works in production line mode (multi-machine)",
            )

        production_line = env.production_line

        # Get optimization results from production line
        optimization_result = production_line.optimize_line_speed(target=optimization_target)

        # Apply constraints if provided
        if constraints:
            for machine_id, machine_constraints in constraints.items():
                if machine_id not in production_line.machines:
                    continue

                machine = production_line.machines[machine_id]
                current_speed = machine.speed

                min_speed = machine_constraints.get("min", 0.0)
                max_speed = machine_constraints.get("max", 100.0)

                # Clamp speed to constraints
                if current_speed < min_speed:
                    machine.set_speed(min_speed)
                    optimization_result["new_speeds"][machine_id] = min_speed
                elif current_speed > max_speed:
                    machine.set_speed(max_speed)
                    optimization_result["new_speeds"][machine_id] = max_speed

        # Calculate speed changes
        speed_changes = {}
        for machine_id in optimization_result["old_speeds"]:
            old = optimization_result["old_speeds"][machine_id]
            new = optimization_result["new_speeds"][machine_id]
            change = new - old
            speed_changes[machine_id] = {
                "old": old,
                "new": new,
                "change": change,
                "change_pct": (change / old * 100) if old > 0 else 0.0,
            }

        # Build result
        result_data = {
            "optimization_target": optimization_target,
            "speed_changes": speed_changes,
            "bottleneck": optimization_result["bottleneck"],
            "expected_throughput": optimization_result["expected_throughput"],
            "expected_energy": optimization_result["expected_energy"],
        }

        if constraints:
            result_data["constraints_applied"] = constraints

        # Generate summary message
        avg_change = sum(abs(sc["change"]) for sc in speed_changes.values()) / len(speed_changes)
        message = (
            f"Optimized line for '{optimization_target}': "
            f"bottleneck={optimization_result['bottleneck']}, "
            f"expected throughput={optimization_result['expected_throughput']:.2f} units/hr, "
            f"avg speed change={avg_change:.1f}%"
        )

        return ToolResult(
            success=True,
            data=result_data,
            message=message,
        )
