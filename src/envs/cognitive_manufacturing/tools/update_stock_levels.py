"""Tool for updating inventory stock levels."""

from typing import Any, TYPE_CHECKING
import uuid
from datetime import datetime
from .base import ManufacturingTool, ToolResult

if TYPE_CHECKING:
    from ..server.environment import CognitiveManufacturingEnvironment


class UpdateStockLevelsTool(ManufacturingTool):
    """Update material inventory stock levels."""

    @property
    def name(self) -> str:
        return "UpdateStockLevels"

    @property
    def description(self) -> str:
        return "Update material inventory stock levels (add, remove, or set)"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "material_id": {
                    "type": "string",
                    "description": "ID of the material to update",
                },
                "change_type": {
                    "type": "string",
                    "enum": ["add", "remove", "set"],
                    "description": "Type of update: add to stock, remove from stock, or set absolute level",
                },
                "quantity": {
                    "type": "number",
                    "description": "Quantity to add/remove or new absolute level",
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for stock update",
                },
            },
            "required": ["material_id", "change_type", "quantity", "reason"],
        }

    def execute(self, parameters: dict[str, Any], env: "CognitiveManufacturingEnvironment") -> ToolResult:
        is_valid, error = self.validate_parameters(parameters)
        if not is_valid:
            return ToolResult(success=False, error=error)

        material_id = parameters["material_id"]
        change_type = parameters["change_type"]
        quantity = parameters["quantity"]
        reason = parameters["reason"]

        # Initialize inventory if not exists
        if not hasattr(env, "_inventory"):
            env._inventory = {}

        if material_id not in env._inventory:
            return ToolResult(
                success=False,
                error=f"Material '{material_id}' not found in inventory system"
            )

        # Get old level
        old_level = env._inventory[material_id]["level"]

        # Calculate new level based on change type
        if change_type == "add":
            new_level = old_level + quantity
        elif change_type == "remove":
            new_level = old_level - quantity
            if new_level < 0:
                return ToolResult(
                    success=False,
                    error=f"Insufficient stock: trying to remove {quantity}, but only {old_level} available"
                )
        else:  # set
            new_level = quantity

        # Update the level
        env._inventory[material_id]["level"] = new_level

        # Generate transaction ID
        transaction_id = str(uuid.uuid4())[:8]

        # Initialize transaction history
        if not hasattr(env, "_inventory_transactions"):
            env._inventory_transactions = []

        # Record transaction
        transaction = {
            "transaction_id": transaction_id,
            "material_id": material_id,
            "change_type": change_type,
            "quantity": quantity,
            "old_level": old_level,
            "new_level": new_level,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
        }

        env._inventory_transactions.append(transaction)

        # Log transaction if database available
        if hasattr(env, "db_manager") and env.db_manager:
            try:
                if hasattr(env, "current_run_id") and env.current_run_id:
                    env.db_manager.record_event(
                        run_id=env.current_run_id,
                        machine_id="system",
                        simulation_time=env._state.simulation_time,
                        event_type="inventory_update",
                        data=transaction
                    )
            except Exception:
                pass

        # Check if stock is now low
        reorder_point = env._inventory[material_id]["reorder_point"]
        stock_status = "ok"
        warnings = []

        if new_level < reorder_point:
            stock_status = "low"
            warnings.append(f"Stock below reorder point ({new_level:.0f} < {reorder_point:.0f})")

        if new_level < reorder_point * 0.5:
            stock_status = "critical"
            warnings.append("Stock critically low - consider urgent reorder")

        # Calculate change percentage
        change_pct = ((new_level - old_level) / old_level * 100) if old_level != 0 else 100.0

        message = f"Stock updated: {material_id} {change_type} {quantity} ({old_level:.0f} → {new_level:.0f})"

        return ToolResult(
            success=True,
            data={
                "updated": True,
                "transaction_id": transaction_id,
                "material_id": material_id,
                "old_level": round(old_level, 2),
                "new_level": round(new_level, 2),
                "change": round(new_level - old_level, 2),
                "change_percent": round(change_pct, 1),
                "stock_status": stock_status,
                "warnings": warnings,
                "transaction_details": transaction,
            },
            message=message,
        )
