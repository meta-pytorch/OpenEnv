"""Tool for checking material inventory levels."""

from typing import Any, TYPE_CHECKING
from .base import ManufacturingTool, ToolResult

if TYPE_CHECKING:
    from ..server.environment import CognitiveManufacturingEnvironment


class CheckInventoryTool(ManufacturingTool):
    """Check material inventory levels and get reorder recommendations."""

    @property
    def name(self) -> str:
        return "CheckInventory"

    @property
    def description(self) -> str:
        return "Check material inventory levels and get reorder recommendations"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "material_type": {
                    "type": "string",
                    "enum": ["raw_material", "components", "finished_goods", "all"],
                    "description": "Type of materials to check",
                    "default": "all",
                },
                "include_forecast": {
                    "type": "boolean",
                    "description": "Include demand forecast in recommendations",
                    "default": False,
                },
            },
            "required": [],
        }

    def execute(self, parameters: dict[str, Any], env: "CognitiveManufacturingEnvironment") -> ToolResult:
        is_valid, error = self.validate_parameters(parameters)
        if not is_valid:
            return ToolResult(success=False, error=error)

        material_type = parameters.get("material_type", "all")
        include_forecast = parameters.get("include_forecast", False)

        # Initialize inventory if not exists
        if not hasattr(env, "_inventory"):
            env._inventory = {
                "steel_sheets": {"type": "raw_material", "level": 1000.0, "unit": "kg", "reorder_point": 200.0},
                "aluminum_bars": {"type": "raw_material", "level": 500.0, "unit": "kg", "reorder_point": 100.0},
                "fasteners": {"type": "components", "level": 5000.0, "unit": "units", "reorder_point": 1000.0},
                "circuit_boards": {"type": "components", "level": 200.0, "unit": "units", "reorder_point": 50.0},
                "finished_units": {"type": "finished_goods", "level": 0.0, "unit": "units", "reorder_point": 0.0},
            }

        # Filter by material type
        inventory_levels = {}
        for material_id, info in env._inventory.items():
            if material_type == "all" or info["type"] == material_type:
                inventory_levels[material_id] = {
                    "current_level": info["level"],
                    "unit": info["unit"],
                    "reorder_point": info["reorder_point"],
                    "material_type": info["type"],
                    "status": "ok" if info["level"] > info["reorder_point"] else "low",
                }

        # Find low stock items
        low_stock_items = []
        for material_id, info in inventory_levels.items():
            if info["status"] == "low":
                criticality = "critical" if info["current_level"] < info["reorder_point"] * 0.5 else "warning"
                low_stock_items.append({
                    "material_id": material_id,
                    "current_level": info["current_level"],
                    "reorder_point": info["reorder_point"],
                    "shortfall": info["reorder_point"] - info["current_level"],
                    "criticality": criticality,
                })

        # Generate reorder recommendations
        reorder_recommendations = []
        for item in low_stock_items:
            # Calculate recommended order quantity (2x reorder point)
            recommended_qty = item["reorder_point"] * 2

            # Adjust based on forecast if enabled
            if include_forecast and hasattr(env, "ml_service") and env.ml_service:
                try:
                    forecast = env.ml_service.forecast_demand(horizon=168)  # 1 week
                    avg_demand = sum(f["demand"] for f in forecast) / len(forecast)
                    # Estimate material consumption (assuming 1 unit uses 10kg raw material)
                    weekly_consumption = avg_demand * 10
                    recommended_qty = max(recommended_qty, weekly_consumption * 1.5)  # 1.5 weeks buffer
                except Exception:
                    pass  # Use default if forecast fails

            reorder_recommendations.append({
                "material_id": item["material_id"],
                "recommended_quantity": round(recommended_qty, 2),
                "priority": "urgent" if item["criticality"] == "critical" else "normal",
                "reason": f"Stock below reorder point ({item['current_level']:.0f} < {item['reorder_point']:.0f})",
            })

        # Calculate total inventory value (simplified)
        total_items = sum(info["current_level"] for info in inventory_levels.values())
        storage_utilization = min(100.0, (total_items / 10000) * 100)  # Assume 10k capacity

        message = f"Inventory check: {len(inventory_levels)} items tracked, {len(low_stock_items)} low stock"

        return ToolResult(
            success=True,
            data={
                "inventory_levels": inventory_levels,
                "low_stock_items": low_stock_items,
                "reorder_recommendations": reorder_recommendations,
                "summary": {
                    "total_items_tracked": len(inventory_levels),
                    "low_stock_count": len(low_stock_items),
                    "reorder_needed": len(reorder_recommendations),
                    "storage_utilization_pct": round(storage_utilization, 1),
                },
                "forecast_included": include_forecast,
            },
            message=message,
        )
