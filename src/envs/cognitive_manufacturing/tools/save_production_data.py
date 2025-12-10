"""Tool for saving production data to database."""

from typing import Any, TYPE_CHECKING
from .base import ManufacturingTool, ToolResult

if TYPE_CHECKING:
    from ..server.environment import CognitiveManufacturingEnvironment


class SaveProductionDataTool(ManufacturingTool):
    """Save current production run data to database."""

    @property
    def name(self) -> str:
        return "SaveProductionData"

    @property
    def description(self) -> str:
        return "Save the current production run data to the database for later analysis"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "run_name": {
                    "type": "string",
                    "description": "Optional name for this production run",
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional metadata (tags, notes, experiment info)",
                },
            },
            "required": [],
        }

    def execute(self, parameters: dict[str, Any], env: "CognitiveManufacturingEnvironment") -> ToolResult:
        is_valid, error = self.validate_parameters(parameters)
        if not is_valid:
            return ToolResult(success=False, error=error)

        # Check if database is enabled
        if not hasattr(env, "db_manager") or env.db_manager is None:
            return ToolResult(
                success=False,
                error="Database not enabled. Create environment with enable_database=True",
            )

        run_name = parameters.get("run_name")
        metadata = parameters.get("metadata", {})

        # If run doesn't exist yet, create it
        if not hasattr(env, "current_run_id") or env.current_run_id is None:
            mode = "multi_machine" if env.multi_machine else "single"
            run_id = env.db_manager.create_run(
                mode=mode,
                run_name=run_name,
                metadata=metadata,
            )
            env.current_run_id = run_id
        else:
            run_id = env.current_run_id

        # Save current state
        state = env._state
        env.db_manager.complete_run(
            run_id=run_id,
            total_steps=state.step_count,
            simulation_time=state.simulation_time,
            cumulative_reward=env._cumulative_reward,
        )

        # Save all sensor readings (collect from current machines)
        if env.multi_machine:
            # Save from all machines
            for machine_id, machine in env.production_line.machines.items():
                env.db_manager.save_sensor_reading(
                    run_id=run_id,
                    machine_id=machine_id,
                    simulation_time=state.simulation_time,
                    temperature=machine.temperature,
                    vibration=machine.vibration,
                    speed=machine.speed,
                    health_score=machine.health_score,
                    wear_level=machine.wear_level,
                    production_output=machine.production_output,
                    status=machine.status,
                )
        else:
            # Save from single machine
            machine = env.simulator_machine
            env.db_manager.save_sensor_reading(
                run_id=run_id,
                machine_id=machine.machine_id,
                simulation_time=state.simulation_time,
                temperature=machine.temperature,
                vibration=machine.vibration,
                speed=machine.speed,
                health_score=machine.health_score,
                wear_level=machine.wear_level,
                production_output=machine.production_output,
                status=machine.status,
            )

        return ToolResult(
            success=True,
            data={
                "run_id": run_id,
                "run_name": run_name,
                "total_steps": state.step_count,
                "simulation_time": state.simulation_time,
                "cumulative_reward": env._cumulative_reward,
                "machines_saved": len(env.production_line.machines) if env.multi_machine else 1,
            },
            message=f"Saved production run '{run_name or run_id}' to database",
        )
