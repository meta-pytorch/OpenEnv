"""Tool for querying production history from database."""

from typing import Any, TYPE_CHECKING
from datetime import datetime
from .base import ManufacturingTool, ToolResult

if TYPE_CHECKING:
    from ..server.environment import CognitiveManufacturingEnvironment


class QueryProductionHistoryTool(ManufacturingTool):
    """Query historical production data from database."""

    @property
    def name(self) -> str:
        return "QueryProductionHistory"

    @property
    def description(self) -> str:
        return "Query historical production runs with optional filters (date range, reward threshold, status, etc.)"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": ["runs", "recent_runs", "best_runs"],
                    "description": "Type of query: all runs, recent runs, or best performing runs",
                    "default": "runs",
                },
                "filters": {
                    "type": "object",
                    "description": "Optional filters for the query",
                    "properties": {
                        "start_date": {
                            "type": "string",
                            "description": "Start date in ISO format (e.g., '2024-01-01T00:00:00')",
                        },
                        "end_date": {
                            "type": "string",
                            "description": "End date in ISO format",
                        },
                        "min_reward": {
                            "type": "number",
                            "description": "Minimum cumulative reward",
                        },
                        "status": {
                            "type": "string",
                            "enum": ["in_progress", "completed", "failed"],
                            "description": "Run status",
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["single", "multi_machine"],
                            "description": "Production mode",
                        },
                    },
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 100)",
                    "minimum": 1,
                    "maximum": 1000,
                    "default": 100,
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

        query_type = parameters.get("query_type", "runs")
        filters = parameters.get("filters", {})
        limit = parameters.get("limit", 100)

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

        # Execute query
        try:
            runs = env.db_manager.query_runs(filters=filters, limit=limit)

            # Apply query type specific processing
            if query_type == "recent_runs":
                # Already sorted by date desc in query_runs
                pass
            elif query_type == "best_runs":
                # Sort by reward descending
                runs = sorted(runs, key=lambda x: x.get("cumulative_reward", 0), reverse=True)

            # Generate summary statistics
            if runs:
                total_runs = len(runs)
                avg_reward = sum(r.get("cumulative_reward", 0) for r in runs) / total_runs
                avg_steps = sum(r.get("total_steps", 0) for r in runs) / total_runs
                completed_runs = sum(1 for r in runs if r.get("status") == "completed")

                summary = {
                    "total_runs": total_runs,
                    "completed_runs": completed_runs,
                    "avg_reward": round(avg_reward, 2),
                    "avg_steps": round(avg_steps, 1),
                    "best_reward": max(r.get("cumulative_reward", 0) for r in runs),
                }
            else:
                summary = {
                    "total_runs": 0,
                    "completed_runs": 0,
                    "avg_reward": 0,
                    "avg_steps": 0,
                    "best_reward": 0,
                }

            message = f"Found {len(runs)} production runs. Avg reward: {summary['avg_reward']:.2f}"

            return ToolResult(
                success=True,
                data={
                    "query_type": query_type,
                    "results": runs,
                    "summary": summary,
                    "count": len(runs),
                },
                message=message,
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Database query failed: {str(e)}",
            )
