"""Tool for manually transferring material between machines."""

from typing import Any, TYPE_CHECKING
from .base import ManufacturingTool, ToolResult

if TYPE_CHECKING:
    from ..server.environment import CognitiveManufacturingEnvironment


class TransferMaterialTool(ManufacturingTool):
    """Manually trigger material transfer between machines."""

    @property
    def name(self) -> str:
        return "TransferMaterial"

    @property
    def description(self) -> str:
        return "Manually transfer material units between machines through buffers"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "from_machine": {
                    "type": "string",
                    "description": "Source machine ID (M1, M2, M3, M4)",
                    "enum": ["M1", "M2", "M3", "M4"],
                },
                "to_machine": {
                    "type": "string",
                    "description": "Destination machine ID (M1, M2, M3, M4)",
                    "enum": ["M1", "M2", "M3", "M4"],
                },
                "units": {
                    "type": "integer",
                    "description": "Number of units to transfer",
                    "minimum": 1,
                },
            },
            "required": ["from_machine", "to_machine", "units"],
        }

    def execute(self, parameters: dict[str, Any], env: "CognitiveManufacturingEnvironment") -> ToolResult:
        is_valid, error = self.validate_parameters(parameters)
        if not is_valid:
            return ToolResult(success=False, error=error)

        from_machine = parameters["from_machine"]
        to_machine = parameters["to_machine"]
        units = parameters["units"]

        # Check if environment has production line simulator
        if not hasattr(env, "production_line") or env.production_line is None:
            return ToolResult(
                success=False,
                error="TransferMaterial only works in production line mode (multi-machine)",
            )

        production_line = env.production_line

        # Validate machines exist
        if from_machine not in production_line.machines:
            return ToolResult(success=False, error=f"Source machine '{from_machine}' not found")

        if to_machine not in production_line.machines:
            return ToolResult(success=False, error=f"Destination machine '{to_machine}' not found")

        # Check if machines are adjacent (can only transfer between adjacent machines)
        valid_transfers = {
            ("M1", "M2"): "M1_M2",
            ("M2", "M3"): "M2_M3",
            ("M3", "M4"): "M3_M4",
        }

        transfer_key = (from_machine, to_machine)
        if transfer_key not in valid_transfers:
            return ToolResult(
                success=False,
                error=f"Cannot transfer from {from_machine} to {to_machine}. "
                f"Only adjacent machines can transfer: M1→M2, M2→M3, M3→M4",
            )

        buffer_id = valid_transfers[transfer_key]
        buffer = production_line.buffers[buffer_id]

        # Check if source buffer has enough units
        if buffer.current_level < units:
            return ToolResult(
                success=False,
                error=f"Insufficient units in buffer {buffer_id}. "
                f"Requested: {units}, Available: {buffer.current_level}",
            )

        # Check if destination can accept units
        # For simplicity, we'll just move units within the buffer system
        # The actual processing will happen in the normal step() cycle

        # Remove units from source side
        removed_units = buffer.remove_units(units)
        transferred = len(removed_units)

        # Add to the next buffer (if exists) or mark as processed
        if to_machine == "M4":
            # M4 is final stage, units would go to finished products
            # But manual transfer to M4 doesn't make sense in production flow
            # Re-add units back
            for unit in removed_units:
                buffer.add_unit(unit)
            return ToolResult(
                success=False,
                error="Cannot manually transfer to M4 (QC stage). Units flow automatically.",
            )

        return ToolResult(
            success=True,
            data={
                "transferred": transferred,
                "from_machine": from_machine,
                "to_machine": to_machine,
                "buffer_id": buffer_id,
                "buffer_level_after": buffer.current_level,
                "buffer_capacity": buffer.capacity,
            },
            message=f"Transferred {transferred} units from {from_machine} to {to_machine} (buffer: {buffer_id})",
        )
