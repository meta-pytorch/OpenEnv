"""Tool for RL-based production optimization."""

from typing import Any, TYPE_CHECKING
from .base import ManufacturingTool, ToolResult

if TYPE_CHECKING:
    from ..server.environment import CognitiveManufacturingEnvironment


class OptimizeWithRLTool(ManufacturingTool):
    """Use reinforcement learning for intelligent production optimization."""

    @property
    def name(self) -> str:
        return "OptimizeWithRL"

    @property
    def description(self) -> str:
        return "Use RL agent to suggest optimal actions for production optimization"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "optimization_goal": {
                    "type": "string",
                    "enum": ["throughput", "quality", "cost", "energy", "balanced"],
                    "description": "Optimization objective",
                    "default": "balanced",
                },
                "constraints": {
                    "type": "object",
                    "description": "Optional constraints (speed limits, etc.)",
                },
                "learning_mode": {
                    "type": "string",
                    "enum": ["exploit", "explore"],
                    "description": "Exploit learned policy or explore new actions",
                    "default": "exploit",
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

        optimization_goal = parameters.get("optimization_goal", "balanced")
        constraints = parameters.get("constraints", {})
        learning_mode = parameters.get("learning_mode", "exploit")

        # Get current state
        if hasattr(env, "production_line") and env.production_line is not None:
            # Multi-machine state
            machines = env.production_line.machines
            avg_health = sum(m.health_score for m in machines.values()) / len(machines)
            avg_temp = sum(m.temperature for m in machines.values()) / len(machines)
            avg_speed = sum(m.speed for m in machines.values()) / len(machines)
        else:
            # Single machine state
            machine = env.simulator_machine
            avg_health = machine.health_score
            avg_temp = machine.temperature
            avg_speed = machine.speed

        state = {
            'health_score': avg_health,
            'temperature': avg_temp,
            'speed': avg_speed,
        }

        # Define available actions based on current state
        available_actions = []

        # Speed adjustment actions
        if avg_speed < 80:
            available_actions.append({
                'tool_name': 'AdjustSpeed',
                'parameters': {'machine_id': 'M1', 'target_speed': min(avg_speed + 10, 85)},
                'description': 'Increase speed'
            })
        if avg_speed > 40:
            available_actions.append({
                'tool_name': 'AdjustSpeed',
                'parameters': {'machine_id': 'M1', 'target_speed': max(avg_speed - 10, 35)},
                'description': 'Decrease speed'
            })

        # Maintenance action
        if avg_health < 80:
            available_actions.append({
                'tool_name': 'ScheduleMaintenance',
                'parameters': {'machine_id': 'M1', 'maintenance_type': 'scheduled'},
                'description': 'Schedule maintenance'
            })

        if not available_actions:
            available_actions.append({
                'tool_name': 'ReadSensors',
                'parameters': {'machine_id': 'M1', 'sensors': 'all'},
                'description': 'Monitor sensors'
            })

        # Get RL recommendation
        rl_result = env.ml_service.select_action_rl(
            state=state,
            available_actions=available_actions,
            learning_mode=learning_mode,
        )

        # Format recommended actions
        recommended_actions = [{
            "tool_name": rl_result['action']['tool_name'],
            "parameters": rl_result['action']['parameters'],
            "expected_reward": round(rl_result['expected_reward'], 2),
            "confidence": round(rl_result['confidence'], 2),
            "description": rl_result['action'].get('description', 'Recommended action'),
        }]

        return ToolResult(
            success=True,
            data={
                "optimization_goal": optimization_goal,
                "recommended_actions": recommended_actions,
                "policy_type": "q_learning",
                "exploration_rate": env.ml_service.exploration_rate if learning_mode == "explore" else 0.0,
                "learning_mode": learning_mode,
                "current_state": {
                    "avg_health": round(avg_health, 1),
                    "avg_temperature": round(avg_temp, 1),
                    "avg_speed": round(avg_speed, 1),
                },
                "available_actions_count": len(available_actions),
            },
            message=f"RL recommendation ({learning_mode} mode): {recommended_actions[0]['tool_name']} "
                   f"(expected reward: {recommended_actions[0]['expected_reward']:.2f})",
        )
