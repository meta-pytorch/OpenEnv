"""
Base class for manufacturing tools.

All tools inherit from ManufacturingTool and implement the execute method.
"""

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..server.environment import CognitiveManufacturingEnvironment


class ToolResult(dict):
    """
    Standardized tool result.

    Always includes:
        success: bool - Whether tool executed successfully
        data: dict - Tool-specific result data
        message: str - Human-readable message
        error: str | None - Error message if failed
    """

    def __init__(
        self,
        success: bool,
        data: dict[str, Any] | None = None,
        message: str = "",
        error: str | None = None,
    ):
        super().__init__(
            success=success,
            data=data or {},
            message=message,
            error=error,
        )


class ManufacturingTool(ABC):
    """Base class for all manufacturing tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name (must match action.tool_name)."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable tool description."""
        pass

    @property
    @abstractmethod
    def parameters_schema(self) -> dict:
        """
        JSON schema for tool parameters.

        Example:
            {
                "type": "object",
                "properties": {
                    "machine_id": {"type": "string"},
                    "threshold": {"type": "number", "minimum": 0}
                },
                "required": ["machine_id"]
            }
        """
        pass

    @abstractmethod
    def execute(
        self,
        parameters: dict[str, Any],
        env: "CognitiveManufacturingEnvironment",
    ) -> ToolResult:
        """
        Execute the tool.

        Args:
            parameters: Tool-specific parameters from agent
            env: Reference to environment for accessing state, simulator, etc.

        Returns:
            ToolResult with success status, data, and messages
        """
        pass

    def validate_parameters(self, parameters: dict) -> tuple[bool, str | None]:
        """
        Validate parameters against schema.

        Returns:
            (is_valid, error_message)
        """
        schema = self.parameters_schema
        required = schema.get("required", [])

        # Check required parameters
        for param in required:
            if param not in parameters:
                return False, f"Missing required parameter: {param}"

        # Basic type checking
        properties = schema.get("properties", {})
        for param_name, value in parameters.items():
            if param_name in properties:
                expected_type = properties[param_name].get("type")
                if expected_type == "string" and not isinstance(value, str):
                    return False, f"Parameter '{param_name}' must be a string"
                elif expected_type == "number" and not isinstance(value, (int, float)):
                    return False, f"Parameter '{param_name}' must be a number"
                elif expected_type == "boolean" and not isinstance(value, bool):
                    return False, f"Parameter '{param_name}' must be a boolean"

        return True, None
