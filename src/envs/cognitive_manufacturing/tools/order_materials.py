"""Tool for ordering raw materials."""

from typing import Any, TYPE_CHECKING
import uuid
from datetime import datetime, timedelta
from .base import ManufacturingTool, ToolResult

if TYPE_CHECKING:
    from ..server.environment import CognitiveManufacturingEnvironment


class OrderMaterialsTool(ManufacturingTool):
    """Order raw materials from suppliers."""

    @property
    def name(self) -> str:
        return "OrderMaterials"

    @property
    def description(self) -> str:
        return "Order raw materials from suppliers"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "material_id": {
                    "type": "string",
                    "description": "ID of the material to order",
                },
                "quantity": {
                    "type": "number",
                    "description": "Quantity to order",
                },
                "priority": {
                    "type": "string",
                    "enum": ["normal", "urgent"],
                    "description": "Order priority",
                    "default": "normal",
                },
                "auto_approve": {
                    "type": "boolean",
                    "description": "Auto-approve the order without manual review",
                    "default": False,
                },
            },
            "required": ["material_id", "quantity"],
        }

    def execute(self, parameters: dict[str, Any], env: "CognitiveManufacturingEnvironment") -> ToolResult:
        is_valid, error = self.validate_parameters(parameters)
        if not is_valid:
            return ToolResult(success=False, error=error)

        material_id = parameters["material_id"]
        quantity = parameters["quantity"]
        priority = parameters.get("priority", "normal")
        auto_approve = parameters.get("auto_approve", False)

        # Check if material exists in inventory
        if not hasattr(env, "_inventory"):
            env._inventory = {}

        if material_id not in env._inventory:
            return ToolResult(
                success=False,
                error=f"Material '{material_id}' not found in inventory system"
            )

        # Initialize orders tracking
        if not hasattr(env, "_material_orders"):
            env._material_orders = []

        # Generate order ID
        order_id = str(uuid.uuid4())[:8]

        # Calculate delivery time based on priority
        if priority == "urgent":
            delivery_days = 1
            cost_multiplier = 1.5  # 50% premium for urgent
        else:
            delivery_days = 5
            cost_multiplier = 1.0

        estimated_delivery = datetime.utcnow() + timedelta(days=delivery_days)

        # Estimate cost (simplified pricing model)
        material_info = env._inventory[material_id]
        base_prices = {
            "raw_material": 5.0,  # $5 per kg
            "components": 2.0,    # $2 per unit
            "finished_goods": 50.0,  # $50 per unit
        }
        unit_price = base_prices.get(material_info["type"], 1.0)
        cost_estimate = quantity * unit_price * cost_multiplier

        # Determine approval status
        if auto_approve:
            approval_status = "approved"
            status = "ordered"
        else:
            # Auto-approve small orders, require approval for large ones
            if cost_estimate < 1000:
                approval_status = "auto_approved"
                status = "ordered"
            else:
                approval_status = "pending_approval"
                status = "pending"

        # Create order record
        order = {
            "order_id": order_id,
            "material_id": material_id,
            "quantity": quantity,
            "priority": priority,
            "cost_estimate": round(cost_estimate, 2),
            "status": status,
            "approval_status": approval_status,
            "ordered_at": datetime.utcnow().isoformat(),
            "estimated_delivery": estimated_delivery.isoformat(),
            "delivery_days": delivery_days,
        }

        env._material_orders.append(order)

        # Log order if database available
        if hasattr(env, "db_manager") and env.db_manager:
            try:
                if hasattr(env, "current_run_id") and env.current_run_id:
                    env.db_manager.record_event(
                        run_id=env.current_run_id,
                        machine_id="system",
                        simulation_time=env._state.simulation_time,
                        event_type="material_order",
                        data=order
                    )
            except Exception:
                pass

        message = f"Order {order_id} created: {quantity} {material_info['unit']} of {material_id}"
        if priority == "urgent":
            message += " (URGENT)"

        return ToolResult(
            success=True,
            data={
                "order_id": order_id,
                "estimated_delivery": estimated_delivery.isoformat(),
                "delivery_days": delivery_days,
                "cost_estimate": round(cost_estimate, 2),
                "approval_status": approval_status,
                "order_details": order,
                "next_steps": [
                    f"Order will arrive in {delivery_days} days" if status == "ordered" else "Waiting for approval",
                    f"Total cost: ${cost_estimate:.2f}",
                    "Use UpdateStockLevels when materials arrive",
                ],
            },
            message=message,
        )
