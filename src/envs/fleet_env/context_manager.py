"""
Context Management for Fleet Task Environments.

This module provides tools for managing conversation context during long trajectories,
inspired by Toolathlon's context management approach. It allows models to:
1. Check how much context they've used
2. Drop old turns to free up context space
3. Search through dropped history
4. Navigate truncated tool outputs

These tools are designed for step-wise RL training where each turn is a separate
training sample. When context is dropped, the training framework re-tokenizes
the modified chat_history, so the model learns from the reduced context.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

# Context management tool definitions (OpenAI function calling format)
CONTEXT_TOOLS = [
    # --- Context/History Management ---
    {
        "type": "function",
        "function": {
            "name": "check_context",
            "description": "Check current context: visible/total turn counts",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "manage_context",
            "description": "Drop old turns to free up context space",
            "parameters": {
                "type": "object",
                "properties": {
                    "keep_recent_turns": {
                        "type": "integer",
                        "description": "Number of recent turns to keep (drops older ones)",
                    }
                },
                "required": ["keep_recent_turns"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_history",
            "description": "Search all history (including dropped) by pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Text pattern to search",
                    }
                },
                "required": ["pattern"],
            },
        },
    },
    # --- Overlong Tool Output Handling ---
    {
        "type": "function",
        "function": {
            "name": "search_tool_output",
            "description": "Search the last truncated tool output by pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Text pattern to search",
                    }
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_tool_output",
            "description": "View a page of the last truncated tool output",
            "parameters": {
                "type": "object",
                "properties": {
                    "page": {
                        "type": "integer",
                        "description": "Page number (1-indexed)",
                    },
                    "page_size": {
                        "type": "integer",
                        "description": "Lines per page (default 50)",
                    },
                },
                "required": ["page"],
            },
        },
    },
]

CONTEXT_TOOL_NAMES = {t["function"]["name"] for t in CONTEXT_TOOLS}


class ContextManager:
    """Manages conversation context for long-running agent trajectories.

    This class provides utilities for:
    1. Tracking full conversation history (never dropped)
    2. Managing visible context (can be trimmed)
    3. Handling truncated tool outputs
    4. Executing context management tool calls

    Designed to work with any training framework that maintains chat_history.
    The framework passes its chat_history to execute_tool(), which may modify it.

    Example:
        >>> ctx = ContextManager(max_output_chars=10000)
        >>> # Get tools to add to the model's available tools
        >>> tools = ctx.get_tools()
        >>> # Track messages as they're added
        >>> ctx.track_message({"role": "assistant", "content": "..."})
        >>> # Check if a tool call is a context tool
        >>> if ctx.is_context_tool("manage_context"):
        ...     result, chat_history = ctx.execute_tool("manage_context", {"keep_recent_turns": 5}, chat_history)
        >>> # Truncate long outputs
        >>> output = ctx.truncate_output(long_tool_result)
    """

    def __init__(self, max_output_chars: int = 10000):
        """Initialize the context manager.

        Args:
            max_output_chars: Maximum characters for tool output before truncation.
                Truncated outputs can be accessed via search_tool_output/view_tool_output.
        """
        self.max_output_chars = max_output_chars
        self.full_history: List[Dict[str, Any]] = []
        self.last_full_output: Optional[str] = None

    def reset(self):
        """Reset state for a new episode."""
        self.full_history = []
        self.last_full_output = None

    def get_tools(self) -> List[Dict[str, Any]]:
        """Get the context management tool definitions.

        Returns:
            List of tool definitions in OpenAI function calling format.
        """
        return CONTEXT_TOOLS.copy()

    def is_context_tool(self, tool_name: str) -> bool:
        """Check if a tool name is a context management tool.

        Args:
            tool_name: Name of the tool to check.

        Returns:
            True if it's a context tool that should be handled locally.
        """
        return tool_name in CONTEXT_TOOL_NAMES

    def track_message(self, message: Dict[str, Any]):
        """Track a message in the full history.

        Call this for every message added to chat_history. The full_history
        is never trimmed, allowing search_history to find dropped messages.

        Args:
            message: Message dict with "role" and "content" keys.
        """
        self.full_history.append(message.copy())

    def truncate_output(self, output: str) -> str:
        """Truncate a tool output if it exceeds max_output_chars.

        If truncated, the full output is stored and can be accessed via
        search_tool_output or view_tool_output tools.

        Args:
            output: The tool output string.

        Returns:
            Original output if within limit, truncated version with notice otherwise.
        """
        if not isinstance(output, str):
            return output

        if len(output) > self.max_output_chars:
            self.last_full_output = output
            return (
                output[: self.max_output_chars]
                + f"\n\n[TRUNCATED - {len(output)} chars total. "
                + "Use search_tool_output or view_tool_output to access full content.]"
            )
        else:
            self.last_full_output = None
            return output

    def execute_tool(
        self, tool_name: str, args: Dict[str, Any], chat_history: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Execute a context management tool.

        Args:
            tool_name: Name of the context tool to execute.
            args: Tool arguments.
            chat_history: Current visible chat history (may be modified).

        Returns:
            Tuple of (result_string, modified_chat_history).
            The chat_history is modified in-place for manage_context.
        """
        if tool_name == "check_context":
            return self._check_context(chat_history), chat_history

        elif tool_name == "manage_context":
            return self._manage_context(args, chat_history)

        elif tool_name == "search_history":
            return self._search_history(args), chat_history

        elif tool_name == "search_tool_output":
            return self._search_tool_output(args), chat_history

        elif tool_name == "view_tool_output":
            return self._view_tool_output(args), chat_history

        else:
            return (
                json.dumps({"error": f"Unknown context tool: {tool_name}"}),
                chat_history,
            )

    def _check_context(self, chat_history: List[Dict[str, Any]]) -> str:
        """Check current context: visible vs total turns."""
        visible_turns = len([m for m in chat_history if m.get("role") == "assistant"])
        total_turns = len(
            [m for m in self.full_history if m.get("role") == "assistant"]
        )
        return json.dumps(
            {
                "visible_turns": visible_turns,
                "total_turns": total_turns,
                "dropped_turns": total_turns - visible_turns,
            }
        )

    def _manage_context(
        self, args: Dict[str, Any], chat_history: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Drop old turns to free up context space."""
        n = args.get("keep_recent_turns", 5)

        # Keep system message + last n turns (each turn = assistant + user message)
        system = [m for m in chat_history if m.get("role") == "system"]
        non_system = [m for m in chat_history if m.get("role") != "system"]
        keep_count = n * 2  # n turns = n assistant + n user messages

        if len(non_system) > keep_count:
            dropped = len(non_system) - keep_count
            new_history = system + non_system[-keep_count:]
            return (
                f"Dropped {dropped} messages. {len(new_history)} remaining.",
                new_history,
            )
        else:
            return f"Nothing to drop. {len(chat_history)} messages.", chat_history

    def _search_history(self, args: Dict[str, Any]) -> str:
        """Search all history (including dropped) by pattern."""
        pattern = args.get("pattern", "").lower()
        if not pattern:
            return json.dumps({"error": "pattern is required"})

        matches = []
        for i, msg in enumerate(self.full_history):
            content = msg.get("content", "")
            if isinstance(content, str) and pattern in content.lower():
                matches.append(
                    {
                        "index": i,
                        "role": msg.get("role"),
                        "snippet": content[:200],
                    }
                )
        return json.dumps({"matches": matches[:10]})

    def _search_tool_output(self, args: Dict[str, Any]) -> str:
        """Search the last truncated tool output by pattern."""
        if not self.last_full_output:
            return "No truncated output available."

        pattern = args.get("pattern", "").lower()
        if not pattern:
            return json.dumps({"error": "pattern is required"})

        lines = self.last_full_output.split("\n")
        matches = []
        for i, line in enumerate(lines):
            if pattern in line.lower():
                matches.append({"line": i + 1, "content": line[:200]})
        return json.dumps({"matches": matches[:20]})

    def _view_tool_output(self, args: Dict[str, Any]) -> str:
        """View a page of the last truncated tool output."""
        if not self.last_full_output:
            return "No truncated output available."

        page = args.get("page", 1)
        page_size = args.get("page_size", 50)
        lines = self.last_full_output.split("\n")
        total_pages = (len(lines) + page_size - 1) // page_size
        start = (page - 1) * page_size
        end = start + page_size
        page_lines = lines[start:end]
        return json.dumps(
            {
                "page": page,
                "total_pages": total_pages,
                "total_lines": len(lines),
                "content": "\n".join(page_lines),
            }
        )
