"""Manufacturing tools for cognitive manufacturing environment."""

from .base import ManufacturingTool, ToolResult
# Phase 0 tools
from .read_sensors import ReadSensorsTool
from .check_health import CheckHealthTool
from .adjust_speed import AdjustSpeedTool
from .schedule_maintenance import ScheduleMaintenanceTool
from .send_alert import SendAlertTool
# Phase 1 tools
from .get_line_status import GetLineStatusTool
from .transfer_material import TransferMaterialTool
from .optimize_line_speed import OptimizeLineSpeedTool
# Phase 2 tools
from .save_production_data import SaveProductionDataTool
from .query_production_history import QueryProductionHistoryTool
from .execute_sql import ExecuteSQLTool
from .export_to_csv import ExportToCSVTool
from .import_from_csv import ImportFromCSVTool
from .search_knowledge import SearchKnowledgeTool
from .add_knowledge import AddKnowledgeTool
# Phase 3 tools
from .predict_maintenance import PredictMaintenanceTool
from .detect_anomaly import DetectAnomalyTool
from .predict_quality import PredictQualityTool
from .optimize_with_rl import OptimizeWithRLTool
from .forecast_demand import ForecastDemandTool
# Phase 4 tools
from .inspect_product import InspectProductTool
from .record_defect import RecordDefectTool
from .update_qc_thresholds import UpdateQCThresholdsTool
from .check_inventory import CheckInventoryTool
from .order_materials import OrderMaterialsTool
from .update_stock_levels import UpdateStockLevelsTool
from .monitor_energy_usage import MonitorEnergyUsageTool
from .set_power_mode import SetPowerModeTool
from .simulate_scenario import SimulateScenarioTool
from .optimize_production_schedule import OptimizeProductionScheduleTool

__all__ = [
    "ManufacturingTool",
    "ToolResult",
    # Phase 0
    "ReadSensorsTool",
    "CheckHealthTool",
    "AdjustSpeedTool",
    "ScheduleMaintenanceTool",
    "SendAlertTool",
    # Phase 1
    "GetLineStatusTool",
    "TransferMaterialTool",
    "OptimizeLineSpeedTool",
    # Phase 2
    "SaveProductionDataTool",
    "QueryProductionHistoryTool",
    "ExecuteSQLTool",
    "ExportToCSVTool",
    "ImportFromCSVTool",
    "SearchKnowledgeTool",
    "AddKnowledgeTool",
    # Phase 3
    "PredictMaintenanceTool",
    "DetectAnomalyTool",
    "PredictQualityTool",
    "OptimizeWithRLTool",
    "ForecastDemandTool",
    # Phase 4
    "InspectProductTool",
    "RecordDefectTool",
    "UpdateQCThresholdsTool",
    "CheckInventoryTool",
    "OrderMaterialsTool",
    "UpdateStockLevelsTool",
    "MonitorEnergyUsageTool",
    "SetPowerModeTool",
    "SimulateScenarioTool",
    "OptimizeProductionScheduleTool",
]
