"""Tool for executing custom SQL queries."""

from typing import Any, TYPE_CHECKING
from .base import ManufacturingTool, ToolResult

if TYPE_CHECKING:
    from ..server.environment import CognitiveManufacturingEnvironment


class ExecuteSQLTool(ManufacturingTool):
    """Execute custom SQL queries for advanced analytics."""

    @property
    def name(self) -> str:
        return "ExecuteSQL"

    @property
    def description(self) -> str:
        return "Execute custom SQL SELECT queries for advanced analytics (read-only, safe execution)"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "SQL SELECT query to execute (only SELECT allowed, no mutations)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of rows to return (default: 1000)",
                    "minimum": 1,
                    "maximum": 10000,
                    "default": 1000,
                },
            },
            "required": ["query"],
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

        query = parameters["query"].strip()
        limit = parameters.get("limit", 1000)

        # Add LIMIT clause if not present
        query_upper = query.upper()
        if "LIMIT" not in query_upper:
            query = f"{query.rstrip(';')} LIMIT {limit}"

        try:
            # Execute query (safety checks are in db_manager.execute_sql)
            results = env.db_manager.execute_sql(query)

            # Format results
            row_count = len(results)

            # Get column names
            columns = list(results[0].keys()) if results else []

            message = f"Query executed successfully. {row_count} rows returned"
            if row_count >= limit:
                message += f" (limited to {limit})"

            return ToolResult(
                success=True,
                data={
                    "results": results,
                    "row_count": row_count,
                    "columns": columns,
                    "query": query,
                },
                message=message,
            )

        except ValueError as e:
            # Safety check failed
            return ToolResult(
                success=False,
                error=f"Query validation failed: {str(e)}",
            )
        except Exception as e:
            # Query execution failed
            return ToolResult(
                success=False,
                error=f"Query execution failed: {str(e)}",
            )
