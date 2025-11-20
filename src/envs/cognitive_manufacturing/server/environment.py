"""Core environment implementation for cognitive manufacturing.

This module implements the OpenEnv-compliant environment that orchestrates:
- Tool execution
- Simulator stepping
- Reward computation
- State management
"""

from __future__ import annotations
from typing import Any
from core.env_server import Environment
from ..models import (
    ManufacturingAction,
    ManufacturingObservation,
    ManufacturingState,
    MachineStatus,
)
from ..tools import (
    # Phase 0
    ReadSensorsTool,
    CheckHealthTool,
    AdjustSpeedTool,
    ScheduleMaintenanceTool,
    SendAlertTool,
    # Phase 1
    GetLineStatusTool,
    TransferMaterialTool,
    OptimizeLineSpeedTool,
    # Phase 2
    SaveProductionDataTool,
    QueryProductionHistoryTool,
    ExecuteSQLTool,
    ExportToCSVTool,
    ImportFromCSVTool,
    SearchKnowledgeTool,
    AddKnowledgeTool,
    # Phase 3
    PredictMaintenanceTool,
    DetectAnomalyTool,
    PredictQualityTool,
    OptimizeWithRLTool,
    ForecastDemandTool,
    # Phase 4
    InspectProductTool,
    RecordDefectTool,
    UpdateQCThresholdsTool,
    CheckInventoryTool,
    OrderMaterialsTool,
    UpdateStockLevelsTool,
    MonitorEnergyUsageTool,
    SetPowerModeTool,
    SimulateScenarioTool,
    OptimizeProductionScheduleTool,
)
from .simulator import SimulatedMachine
from .production_line import ProductionLineSimulator
from .rewards import RewardCalculator, compute_cumulative_metrics


class CognitiveManufacturingEnvironment(Environment):
    """OpenEnv-compliant environment for cognitive manufacturing scenarios.

    This environment can operate in multiple modes:
    1. Single Machine Mode (Phase 0): One machine with 5 tools
    2. Production Line Mode (Phase 1): 4-machine production line with 8 tools
    3. Database Mode (Phase 2): Add 7 data tools (15 tools total)
    4. ML Mode (Phase 3): Add 5 ML-powered tools (20 tools total)
    5. Advanced Management (Phase 4): Add 10 advanced tools (30 tools total)

    Available tools (Phase 0):
    1. ReadSensors - Get current sensor readings
    2. CheckHealth - Perform health diagnostics
    3. AdjustSpeed - Change machine operating speed
    4. ScheduleMaintenance - Schedule or perform maintenance
    5. SendAlert - Send notifications to operators

    Additional tools (Phase 1):
    6. GetLineStatus - Get status of entire production line
    7. TransferMaterial - Manually transfer material between machines
    8. OptimizeLineSpeed - Optimize all machine speeds

    Additional tools (Phase 2):
    9. SaveProductionData - Save production run to database
    10. QueryProductionHistory - Query historical data
    11. ExecuteSQL - Execute custom SQL queries
    12. ExportToCSV - Export data to CSV
    13. ImportFromCSV - Import data from CSV
    14. SearchKnowledge - Semantic search in knowledge base
    15. AddKnowledge - Add documents to knowledge base

    Additional tools (Phase 3):
    16. PredictMaintenance - Predict when maintenance will be needed
    17. DetectAnomaly - Detect abnormal sensor patterns
    18. PredictQuality - Predict product quality
    19. OptimizeWithRL - Use reinforcement learning for optimization
    20. ForecastDemand - Forecast future production demand

    Additional tools (Phase 4):
    21. InspectProduct - Detailed quality inspection
    22. RecordDefect - Log defects and issues
    23. UpdateQCThresholds - Adjust quality control parameters
    24. CheckInventory - Check material inventory levels
    25. OrderMaterials - Order raw materials
    26. UpdateStockLevels - Update inventory records
    27. MonitorEnergyUsage - Track power consumption
    28. SetPowerMode - Adjust machine power modes
    29. SimulateScenario - Run what-if simulations
    30. OptimizeProductionSchedule - Optimize production schedule

    The environment tracks multi-objective rewards:
    - Safety (highest priority)
    - Throughput (production efficiency)
    - Quality (defect minimization)
    - Cost (operational expenses)
    - Sustainability (energy efficiency)
    """

    def __init__(
        self,
        machine_id: str = "M1",
        timestep: float = 0.1,
        multi_machine: bool = False,
        enable_database: bool = False,
        db_connection_string: str | None = None,
        enable_ml: bool = False,
    ):
        """Initialize the manufacturing environment.

        Args:
            machine_id: Identifier for the simulated machine (single machine mode)
            timestep: Simulation timestep in hours (default 0.1 = 6 minutes)
            multi_machine: If True, use production line mode with 4 machines (Phase 1)
            enable_database: If True, enable database and Phase 2 tools
            db_connection_string: PostgreSQL connection string (if enable_database=True)
                Example: "postgresql://user:pass@localhost/dbname"
            enable_ml: If True, enable ML features and Phase 3 tools (requires enable_database=True)
        """
        super().__init__()
        self.machine_id = machine_id
        self.timestep = timestep
        self.multi_machine = multi_machine
        self.enable_database = enable_database
        self.enable_ml = enable_ml

        # Initialize simulator (single or multi-machine)
        if multi_machine:
            self.production_line = ProductionLineSimulator()
            self.simulator_machine = self.production_line.machines["M1"]  # For backward compatibility
        else:
            self.simulator_machine = SimulatedMachine(machine_id=machine_id)
            self.production_line = None

        # Initialize reward calculator
        self.reward_calculator = RewardCalculator()

        # Initialize Phase 2 services if database enabled
        self.db_manager = None
        self.embedding_service = None
        self.csv_service = None
        self.current_run_id = None

        if enable_database:
            # Import Phase 2 services
            try:
                from .database import DatabaseManager
                from .embeddings import EmbeddingService
                from .csv_service import CSVService

                # Use default SQLite if no connection string provided
                if db_connection_string is None:
                    import os
                    db_path = os.path.join(os.getcwd(), "data", "cognitive_manufacturing.db")
                    os.makedirs(os.path.dirname(db_path), exist_ok=True)
                    db_connection_string = f"sqlite:///{db_path}"

                self.db_manager = DatabaseManager(db_connection_string)
                self.db_manager.create_tables()

                self.embedding_service = EmbeddingService()
                self.csv_service = CSVService(export_dir="data/exports")

            except ImportError as e:
                raise ImportError(
                    f"Phase 2 dependencies not installed: {e}\n"
                    "Install with: pip install sqlalchemy sentence-transformers pandas"
                )

        # Initialize Phase 3 services if ML enabled
        self.ml_service = None

        if enable_ml:
            if not enable_database:
                raise ValueError("ML features require database (enable_database=True)")

            # Import Phase 3 services
            try:
                from .ml_models import MLModelsService

                self.ml_service = MLModelsService()

                # Train models if database has sufficient data
                if self.db_manager.get_run_count() > 10:
                    # Train maintenance model from sensor readings
                    sensor_data = self.db_manager.get_sensor_readings(limit=1000)
                    if len(sensor_data) >= 10:
                        self.ml_service.train_maintenance_model(sensor_data)

                    # Train anomaly detector from normal operation data
                    normal_data = [s for s in sensor_data if s.get('health_score', 100) > 70]
                    if len(normal_data) >= 20:
                        self.ml_service.train_anomaly_detector(normal_data)

                    # Train quality predictor from production units
                    production_data = self.db_manager.get_production_units(limit=1000)
                    if len(production_data) >= 10:
                        self.ml_service.train_quality_predictor(production_data)

            except ImportError as e:
                raise ImportError(
                    f"Phase 3 dependencies not installed: {e}\n"
                    "Install with: pip install scikit-learn numpy scipy"
                )

        # Initialize tools (Phase 0 tools always available)
        self.tools = {
            "ReadSensors": ReadSensorsTool(),
            "CheckHealth": CheckHealthTool(),
            "AdjustSpeed": AdjustSpeedTool(),
            "ScheduleMaintenance": ScheduleMaintenanceTool(),
            "SendAlert": SendAlertTool(),
        }

        # Add Phase 1 tools if in multi-machine mode
        if multi_machine:
            self.tools.update({
                "GetLineStatus": GetLineStatusTool(),
                "TransferMaterial": TransferMaterialTool(),
                "OptimizeLineSpeed": OptimizeLineSpeedTool(),
            })

        # Add Phase 2 tools if database enabled
        if enable_database:
            self.tools.update({
                "SaveProductionData": SaveProductionDataTool(),
                "QueryProductionHistory": QueryProductionHistoryTool(),
                "ExecuteSQL": ExecuteSQLTool(),
                "ExportToCSV": ExportToCSVTool(),
                "ImportFromCSV": ImportFromCSVTool(),
                "SearchKnowledge": SearchKnowledgeTool(),
                "AddKnowledge": AddKnowledgeTool(),
            })

        # Add Phase 3 tools if ML enabled
        if enable_ml:
            self.tools.update({
                "PredictMaintenance": PredictMaintenanceTool(),
                "DetectAnomaly": DetectAnomalyTool(),
                "PredictQuality": PredictQualityTool(),
                "OptimizeWithRL": OptimizeWithRLTool(),
                "ForecastDemand": ForecastDemandTool(),
            })

        # Add Phase 4 tools (always available)
        self.tools.update({
            # Quality Management
            "InspectProduct": InspectProductTool(),
            "RecordDefect": RecordDefectTool(),
            "UpdateQCThresholds": UpdateQCThresholdsTool(),
            # Inventory Management
            "CheckInventory": CheckInventoryTool(),
            "OrderMaterials": OrderMaterialsTool(),
            "UpdateStockLevels": UpdateStockLevelsTool(),
            # Energy Management
            "MonitorEnergyUsage": MonitorEnergyUsageTool(),
            "SetPowerMode": SetPowerModeTool(),
            # Advanced Optimization
            "SimulateScenario": SimulateScenarioTool(),
            "OptimizeProductionSchedule": OptimizeProductionScheduleTool(),
        })

        # Initialize state
        self._state = ManufacturingState(simulation_time=0.0)
        self._cumulative_reward = 0.0
        self._last_reward = 0.0
        self._alerts = []
        self._done = False

    def reset(self) -> ManufacturingObservation:
        """Reset the environment to initial state.

        Returns:
            Initial observation with machine status
        """
        # Reset simulator (single or multi-machine)
        if self.multi_machine:
            self.production_line = ProductionLineSimulator()
            self.simulator_machine = self.production_line.machines["M1"]  # For backward compatibility
        else:
            self.simulator_machine = SimulatedMachine(machine_id=self.machine_id)

        # Reset state
        self._state = ManufacturingState(simulation_time=0.0)
        self._cumulative_reward = 0.0
        self._last_reward = 0.0
        self._alerts = []
        self._done = False

        # Return initial observation
        machine_status = MachineStatus(
            machine_id=self.simulator_machine.machine_id,
            status=self.simulator_machine.status,
            temperature=round(self.simulator_machine.temperature, 2),
            vibration=round(self.simulator_machine.vibration, 3),
            speed=round(self.simulator_machine.speed, 1),
            health_score=round(self.simulator_machine.health_score, 1),
            wear_level=round(self.simulator_machine.wear_level, 3),
            production_output=round(self.simulator_machine.production_output, 2),
        )

        mode_message = "production line" if self.multi_machine else "single machine"
        return ManufacturingObservation(
            tool_result={"message": f"Environment reset ({mode_message} mode)", "initial_status": "ready"},
            machine_status=machine_status,
            alerts=[],
            simulation_time=0.0,
        )

    def step(self, action: ManufacturingAction) -> tuple[ManufacturingObservation, float, bool, dict[str, Any]]:
        """Execute one environment step with the given action.

        Args:
            action: Tool-calling action to execute

        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Execute tool
        tool_result = self._execute_tool(action)

        # Step simulator forward in time
        if self.multi_machine:
            # Step entire production line
            line_result = self.production_line.step(self.timestep)
            sim_result = line_result
            # Update metrics for production line
            self._state.total_produced = self.production_line.metrics.total_produced
        else:
            # Step single machine
            sim_result = self.simulator_machine.step(self.timestep)

        # Compute reward
        reward, reward_breakdown = self.reward_calculator.compute_reward(
            self.simulator_machine,
            self._state,
            self.timestep,
        )

        # Update state
        self._state.simulation_time += self.timestep
        self._state.step_count += 1
        self._last_reward = reward
        self._cumulative_reward += reward

        # Check termination conditions
        self._done = self._check_done()

        # Create machine status snapshot (M1 for multi-machine mode)
        machine_status = MachineStatus(
            machine_id=self.simulator_machine.machine_id,
            status=self.simulator_machine.status,
            temperature=round(self.simulator_machine.temperature, 2),
            vibration=round(self.simulator_machine.vibration, 3),
            speed=round(self.simulator_machine.speed, 1),
            health_score=round(self.simulator_machine.health_score, 1),
            wear_level=round(self.simulator_machine.wear_level, 3),
            production_output=round(self.simulator_machine.production_output, 2),
        )

        # Create observation
        observation = ManufacturingObservation(
            tool_result=tool_result,
            machine_status=machine_status,
            alerts=self._alerts[-5:],  # Last 5 alerts
            simulation_time=round(self._state.simulation_time, 2),
        )

        # Additional info
        info = {
            "reward_breakdown": reward_breakdown,
            "simulator_events": sim_result,
            "cumulative_metrics": {
                "total_reward": self._cumulative_reward,
                "simulation_time": self._state.simulation_time,
                "avg_reward_per_hour": self._cumulative_reward / max(self._state.simulation_time, 1.0),
                "total_alerts": len(self._alerts),
                "critical_alerts": sum(1 for alert in self._alerts if alert.severity == "critical"),
            },
        }

        # Add production line metrics if in multi-machine mode
        if self.multi_machine:
            info["production_line"] = {
                "total_produced": self.production_line.metrics.total_produced,
                "throughput_rate": self.production_line.metrics.throughput_rate,
                "qc_pass_rate": self.production_line.metrics.qc_pass_rate,
                "line_efficiency": self.production_line.metrics.line_efficiency,
                "bottleneck": self.production_line.metrics.bottleneck_machine,
                "finished_products": len(self.production_line.finished_products),
            }

        return observation, reward, self._done, info

    def state(self) -> dict[str, Any]:
        """Get current environment state for inspection.

        Returns:
            Dictionary containing full environment state
        """
        return {
            "simulation_time": self._state.simulation_time,
            "step_count": self._state.step_count,
            "cumulative_reward": self._cumulative_reward,
            "last_reward": self._last_reward,
            "done": self._done,
            "machine": {
                "machine_id": self.simulator_machine.machine_id,
                "status": self.simulator_machine.status,
                "temperature": round(self.simulator_machine.temperature, 2),
                "vibration": round(self.simulator_machine.vibration, 3),
                "speed": round(self.simulator_machine.speed, 1),
                "health_score": round(self.simulator_machine.health_score, 1),
                "wear_level": round(self.simulator_machine.wear_level, 3),
                "production_output": round(self.simulator_machine.production_output, 2),
            },
            "alerts": [
                {
                    "alert_id": alert.alert_id,
                    "severity": alert.severity,
                    "message": alert.message,
                    "category": alert.category,
                    "timestamp": alert.timestamp,
                }
                for alert in self._alerts[-10:]  # Last 10 alerts
            ],
            "available_tools": list(self.tools.keys()),
        }

    def _execute_tool(self, action: ManufacturingAction) -> dict[str, Any]:
        """Execute the requested tool.

        Args:
            action: Action specifying tool and parameters

        Returns:
            Tool execution result as dictionary
        """
        tool_name = action.tool_name
        parameters = action.parameters

        # Check if tool exists
        if tool_name not in self.tools:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}. Available tools: {list(self.tools.keys())}",
            }

        # Execute tool
        tool = self.tools[tool_name]
        result = tool.execute(parameters, self)

        # ToolResult is already a dict, return it directly
        return result

    def _check_done(self) -> bool:
        """Check if episode should terminate.

        Returns:
            True if episode is done, False otherwise
        """
        # Episode ends if:
        # 1. Machine has failed
        # 2. Simulation time exceeds 24 hours
        # 3. Too many critical alerts (safety concern)

        if self.simulator_machine.status == "failed":
            return True

        if self._state.simulation_time >= 24.0:
            return True

        critical_alerts = sum(1 for alert in self._alerts if alert.severity == "critical")
        if critical_alerts >= 10:
            return True

        return False
