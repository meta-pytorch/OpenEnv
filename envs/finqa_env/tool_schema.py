# envs/finqa_env/tool_schema.py
"""
Utilities for auto-generating OpenAI tool schemas from function definitions.
"""

import inspect
from typing import Any, Dict, List, Tuple, get_type_hints


def parse_docstring(docstring: str) -> Tuple[str, Dict[str, str]]:
    """Parse a docstring to extract description and parameter descriptions.

    Args:
        docstring: The function docstring

    Returns:
        Tuple of (description, {param_name: param_description})
    """
    if not docstring:
        return "", {}

    lines = docstring.strip().split("\n")
    description_lines = []
    params = {}

    in_args = False
    in_returns = False
    current_param = None

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("Args:"):
            in_args = True
            in_returns = False
            continue
        elif stripped.startswith("Returns:"):
            in_args = False
            in_returns = True
            continue

        if in_args:
            # Check for param line: "param_name: description"
            if ":" in stripped and not stripped.startswith(" "):
                parts = stripped.split(":", 1)
                current_param = parts[0].strip()
                params[current_param] = parts[1].strip() if len(parts) > 1 else ""
            elif current_param and stripped:
                # Continuation of previous param description
                params[current_param] += " " + stripped
        elif in_returns:
            # Skip Returns section
            continue
        elif stripped:
            description_lines.append(stripped)

    description = " ".join(description_lines)
    return description, params


def function_to_openai_schema(func) -> Dict[str, Any]:
    """Convert a function to OpenAI tool schema using introspection.

    Args:
        func: The function to convert

    Returns:
        OpenAI-compatible tool schema dict
    """
    # Get function signature
    sig = inspect.signature(func)

    # Get type hints
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    # Parse docstring
    description, param_docs = parse_docstring(func.__doc__ or "")

    # Build parameters
    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue

        # Get type from hints
        # Note: Only handles basic types (str, int, float, bool).
        # FinQA tools only use str params, so this is sufficient.
        param_type = hints.get(param_name, str)
        json_type = "string"  # Default to string
        if param_type == int:
            json_type = "integer"
        elif param_type == float:
            json_type = "number"
        elif param_type == bool:
            json_type = "boolean"

        properties[param_name] = {
            "type": json_type,
            "description": param_docs.get(param_name, "")
        }

        # All params without defaults are required
        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }


def generate_tool_schemas(tool_class, method_names: List[str]) -> List[Dict[str, Any]]:
    """Generate OpenAI tool schemas from class methods.

    Args:
        tool_class: The class containing tool methods
        method_names: List of method names to generate schemas for

    Returns:
        List of OpenAI-compatible tool schemas
    """
    schemas = []
    for method_name in method_names:
        method = getattr(tool_class, method_name)
        schemas.append(function_to_openai_schema(method))
    return schemas
