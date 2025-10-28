"""Tool for exporting data to CSV files."""

from typing import Any, TYPE_CHECKING
from datetime import datetime
from .base import ManufacturingTool, ToolResult

if TYPE_CHECKING:
    from ..server.environment import CognitiveManufacturingEnvironment


class ExportToCSVTool(ManufacturingTool):
    """Export production data to CSV file."""

    @property
    def name(self) -> str:
        return "ExportToCSV"

    @property
    def description(self) -> str:
        return "Export production data (runs, sensors, events, units) to CSV file for analysis"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "data_type": {
                    "type": "string",
                    "enum": ["runs", "sensors", "events", "units"],
                    "description": "Type of data to export",
                },
                "filename": {
                    "type": "string",
                    "description": "Output filename (with or without .csv extension)",
                },
                "filters": {
                    "type": "object",
                    "description": "Optional filters (same as QueryProductionHistory)",
                    "properties": {
                        "start_date": {"type": "string"},
                        "end_date": {"type": "string"},
                        "min_reward": {"type": "number"},
                        "status": {"type": "string"},
                        "mode": {"type": "string"},
                        "machine_id": {"type": "string"},
                    },
                },
            },
            "required": ["data_type", "filename"],
        }

    def execute(self, parameters: dict[str, Any], env: "CognitiveManufacturingEnvironment") -> ToolResult:
        is_valid, error = self.validate_parameters(parameters)
        if not is_valid:
            return ToolResult(success=False, error=error)

        # Check if database and CSV service are enabled
        if not hasattr(env, "db_manager") or env.db_manager is None:
            return ToolResult(
                success=False,
                error="Database not enabled. Create environment with enable_database=True",
            )

        if not hasattr(env, "csv_service") or env.csv_service is None:
            return ToolResult(
                success=False,
                error="CSV service not enabled",
            )

        data_type = parameters["data_type"]
        filename = parameters["filename"]
        filters = parameters.get("filters", {})

        # Parse date filters if provided
        if "start_date" in filters:
            try:
                filters["start_date"] = datetime.fromisoformat(filters["start_date"].replace("Z", "+00:00"))
            except ValueError as e:
                return ToolResult(success=False, error=f"Invalid start_date format: {e}")

        if "end_date" in filters:
            try:
                filters["end_date"] = datetime.fromisoformat(filters["end_date"].replace("Z", "+00:00"))
            except ValueError as e:
                return ToolResult(success=False, error=f"Invalid end_date format: {e}")

        try:
            # Query data based on type
            if data_type == "runs":
                data = env.db_manager.query_runs(filters=filters, limit=10000)

            elif data_type == "sensors":
                # Query sensor readings (need to add this method to db_manager)
                # For now, use SQL query
                query = "SELECT * FROM sensor_readings"
                where_clauses = []

                if "start_date" in filters:
                    where_clauses.append(f"timestamp >= '{filters['start_date'].isoformat()}'")
                if "end_date" in filters:
                    where_clauses.append(f"timestamp <= '{filters['end_date'].isoformat()}'")
                if "machine_id" in filters:
                    where_clauses.append(f"machine_id = '{filters['machine_id']}'")

                if where_clauses:
                    query += " WHERE " + " AND ".join(where_clauses)

                query += " ORDER BY timestamp DESC LIMIT 10000"
                data = env.db_manager.execute_sql(query)

            elif data_type == "events":
                # Query machine events
                query = "SELECT * FROM machine_events"
                where_clauses = []

                if "start_date" in filters:
                    where_clauses.append(f"timestamp >= '{filters['start_date'].isoformat()}'")
                if "end_date" in filters:
                    where_clauses.append(f"timestamp <= '{filters['end_date'].isoformat()}'")
                if "machine_id" in filters:
                    where_clauses.append(f"machine_id = '{filters['machine_id']}'")

                if where_clauses:
                    query += " WHERE " + " AND ".join(where_clauses)

                query += " ORDER BY timestamp DESC LIMIT 10000"
                data = env.db_manager.execute_sql(query)

            elif data_type == "units":
                # Query production units
                query = "SELECT * FROM production_units"
                where_clauses = []

                if "start_date" in filters:
                    where_clauses.append(f"created_at >= '{filters['start_date'].isoformat()}'")
                if "end_date" in filters:
                    where_clauses.append(f"created_at <= '{filters['end_date'].isoformat()}'")

                if where_clauses:
                    query += " WHERE " + " AND ".join(where_clauses)

                query += " ORDER BY created_at DESC LIMIT 10000"
                data = env.db_manager.execute_sql(query)

            else:
                return ToolResult(success=False, error=f"Unknown data type: {data_type}")

            # Export to CSV
            filepath = env.csv_service.export_to_csv(data, filename)

            return ToolResult(
                success=True,
                data={
                    "filepath": filepath,
                    "data_type": data_type,
                    "row_count": len(data),
                    "filters_applied": filters,
                },
                message=f"Exported {len(data)} {data_type} records to {filepath}",
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Export failed: {str(e)}",
            )
