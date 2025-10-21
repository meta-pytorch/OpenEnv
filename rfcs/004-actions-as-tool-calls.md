# RFC: Support multiple tool calls via Action wrapper abstraction

**Status**: In Review
**Created**: 10/15/2025
**Authors**: @Darktex, @pankit-eng
**RFC ID**: 004

**Note**: This RFC defines the unified action interface that applies to all environment types. RFC 003 describes how MCP tools integrate with this action system to enable tool calling in both traditional and CodeAct paradigms.

## Summary

This RFC proposes treating environment actions using a standardized pattern inspired by MCP (Model Context Protocol), where each action represents a discrete, named operation with typed parameters. This approach aligns OpenEnv with modern LLM agent frameworks while maintaining type safety and providing better introspection capabilities for agent training and debugging.

Instead of arbitrary `Action` subclasses with domain-specific fields, actions follow a tool-call pattern with a `tool_name` and structured `parameters`, making the framework more composable and easier to integrate with tool-using agents.

**Important**: While inspired by MCP's tool-calling pattern, this abstraction extends beyond external tools and code execution to **any environment action** - including game moves, navigation commands, configuration changes, and domain-specific operations that don't involve tools at all.

## Motivation

### Problem Statement

Current action design in OpenEnv treats actions as dataclasses:

```python
@dataclass
class CodeAction(Action):
    code: str

@dataclass
class BashAction(Action):
    command: str
    cwd: Optional[str] = None

@dataclass
class MoveAction(Action):
    direction: str  # "up", "down", "left", "right"

@dataclass
class GameAction(Action):
    action_id: int
    player_id: str
```

This approach has several limitations:

1. **Lack of Introspection**: No standard way to discover what actions an environment supports
2. **LLM Integration Friction**: Modern LLM agents use tool-calling patterns with JSON schemas, requiring translation layers
3. **Inconsistent Patterns**: Each environment invents its own action structure without standardization
4. **Poor Discoverability**: Agents can't programmatically determine valid actions and their parameters

### Goals

1. **Standardize Action Structure**: Define a consistent pattern for representing all environment actions, inspired by MCP's tool-calling design
2. **Enable Action Discovery**: Provide APIs to introspect available actions in any environment
3. **Improve LLM Integration**: Native compatibility with tool-calling patterns used by Claude, GPT-4, and other models
4. **Maintain Type Safety**: Preserve strong typing while adopting the unified action pattern
5. **Universal Applicability**: Support any type of action - tools, code execution, game moves, navigation, configuration, etc.

### Inspiration: MCP (Model Context Protocol)

This RFC is heavily inspired by [MCP](https://spec.modelcontextprotocol.io/), which standardized how external tools are exposed to AI agents. MCP introduced:
- Standardized tool definitions with JSON Schema
- Tool discovery via `list_tools()` API
- Tool execution via `call_tool(name, parameters)` RPC
- Language-agnostic design

We adopt these principles but **generalize beyond tools** to cover all environment actions. For example:
- A chess environment's "move piece" action is not a "tool" in the MCP sense
- A navigation environment's "go_north" action doesn't involve external tool calls
- A configuration environment's "set_parameter" action isn't code execution

Yet all benefit from the same standardized action pattern.

## Design

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Agent/RL Code                        │
│                                                         │
│  # Action discovery (works for ANY environment)         │
│  actions = env.actions()                                │
│  # -> [ActionDefinition(name="execute_code", ...)]      │
│  # -> [ActionDefinition(name="move_piece", ...)]        │
│  # -> [ActionDefinition(name="set_config", ...)]        │
│                                                         │
│  # Execute action (unified interface)                   │
│  action = ToolCallAction(                               │
│      tool_name="execute_code",                          │
│      parameters={"code": "print('Hello')"}              │
│  )                                                      │
│  observation = env.step(action)                         │
└─────────────────────────────────────────────────────────┘
                        │ HTTP
                        ▼
┌─────────────────────────────────────────────────────────┐
│              Environment (Docker Container)             │
│                                                         │
│  class PythonCodeActEnv(Environment):                   │
│                                                         │
│      @action("execute_code")                            │
│      def execute_code(self, code: str) -> CodeResult:   │
│          return self._executor.run(code)                │
│                                                         │
│      def step(self, action: ToolCallAction):            │
│          action_fn = self._get_action(action.tool_name) │
│          result = action_fn(**action.parameters)        │
│          return self._make_observation(result)          │
└─────────────────────────────────────────────────────────┘
```

### Core Abstractions

#### 1. ToolCallAction

```python
from typing import Any, Dict
from dataclasses import dataclass, field

@dataclass(kw_only=True)
class ToolCallAction(Action):
    """Action representing a named operation with typed parameters.

    Inspired by MCP's tool-calling pattern, but generalized to represent
    ANY environment action - not just tool calls or code execution.

    Examples:
    - Tool calls: tool_name="search_web", parameters={"query": "..."}
    - Code execution: tool_name="execute_code", parameters={"code": "..."}
    - Game moves: tool_name="move_piece", parameters={"from": "e2", "to": "e4"}
    - Navigation: tool_name="go_north", parameters={}
    - Configuration: tool_name="set_timeout", parameters={"seconds": 30}

    This is the standard action type for all OpenEnv environments.
    Environments dispatch based on tool_name to handle different action types.
    """

    tool_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
```

#### 2. ToolDefinition

```python
from typing import Any, Callable, Dict, List
from dataclasses import dataclass

@dataclass
class ToolParameter:
    """Definition of a tool parameter."""

    name: str
    type: str  # JSON Schema type: "string", "number", "boolean", "object", "array"
    description: str
    required: bool = True
    default: Any = None

@dataclass
class ToolDefinition:
    """Specification of an action that can be taken in an environment.

    Inspired by MCP's tool definition format and compatible with LLM tool-calling
    APIs (Claude, OpenAI, etc.), but represents ANY action type - not just tools.

    This can describe:
    - External tool calls (search_web, read_file)
    - Code execution (execute_python, run_bash)
    - Game actions (move_piece, attack, defend)
    - Navigation commands (go_north, turn_left)
    - Configuration changes (set_parameter, update_config)
    - Any domain-specific action
    """

    name: str
    description: str
    parameters: List[ToolParameter]

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format for LLM tool calling."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    p.name: {
                        "type": p.type,
                        "description": p.description,
                    }
                    for p in self.parameters
                },
                "required": [p.name for p in self.parameters if p.required],
            },
        }
```

#### 3. Enhanced Environment Interface

```python
from typing import List, Optional

class Environment(ABC):
    """Base class for all environment servers."""

    @abstractmethod
    def reset(self) -> Observation:
        """Reset the environment and return initial observation."""
        pass

    @abstractmethod
    def step(self, action: Action) -> Observation:
        """Take a step in the environment."""
        pass

    @property
    @abstractmethod
    def state(self) -> State:
        """Get current environment state."""
        pass

    def actions(self) -> List[ToolDefinition]:
        """Return list of available actions in this environment.

        This method enables action discovery for any environment type.
        Actions can represent tools, code execution, game moves, navigation,
        or any domain-specific operations.

        For backward compatibility, environments can return an empty list,
        though implementing this method is strongly encouraged.
        """
        return []

    def tools(self) -> List[ToolDefinition]:
        """Alias for actions() for backward compatibility with RFC 003.

        Deprecated: Use actions() instead.
        """
        return self.actions()
```

### Key Design Decisions

#### Decision 1: Unified Action Type vs. Per-Tool Action Classes

**Chosen Approach**: Use a single `ToolCallAction` class with `tool_name` and `parameters` fields rather than creating separate action classes per tool.

**Rationale**:
- **Simplicity**: Single action type is easier to understand and work with
- **Flexibility**: Adding new actions doesn't require new action classes
- **MCP Compatibility**: Matches the structure used by MCP for tool calling, enabling easy integration
- **Type Safety**: JSON Schema validation can still enforce parameter types
- **Universality**: Works for any action type - tools, game moves, navigation, configuration, etc.
- **Composability**: Multi-action environments work naturally

**Trade-offs**:
- Advantages:
  - Less boilerplate (no action class per tool)
  - Natural support for dynamic tool sets
- Disadvantages:
  - Action parameters are `Dict[str, Any]` instead of strongly-typed fields (mitigated by JSON Schema validation)

#### Decision 2: Action Discovery via `actions()` Method

**Chosen Approach**: Add an `actions()` method to the `Environment` base class that returns `List[ToolDefinition]`.

**Rationale**:
- **Universal Introspection**: Agents can discover available actions in any environment type
- **LLM Integration**: Action definitions can be passed directly to LLM APIs (they use the same format as tool definitions)
- **Documentation**: Self-documenting environments via decorator pattern
- **MCP Alignment**: Follows MCP's `list_tools()` pattern but generalized to all actions

**Note**: We keep `ToolDefinition` as the return type name for compatibility with LLM APIs and MCP, even though it represents any action type.


## Examples

### Example 1: Code Execution Environment

```python
from core.env_server import Environment, Observation, State, ToolCallAction
from core.tools import PyExecutor

class PythonCodeActEnv(Environment):
    """Environment for executing Python code via tool calls."""

    def __init__(self):
        self._executor = PyExecutor()
        self._state = CodeState()

    @action("execute_code", "Execute Python code and return stdout, stderr, and exit code")
    def execute_code(self, code: str) -> Dict[str, Any]:
        """Execute Python code.

        Args:
            code: Python code to execute

        Returns:
            Dict with stdout, stderr, and exit_code keys
        """
        result = self._executor.run(code)
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
        }

    def reset(self) -> Observation:
        self._state = CodeState(episode_id=str(uuid.uuid4()))
        return CodeObservation(stdout="", stderr="", exit_code=0)

    def step(self, action: Action) -> Observation:
        if not isinstance(action, ToolCallAction):
            raise ValueError(f"Expected ToolCallAction, got {type(action)}")

        # Dispatch to action method
        if action.tool_name == "execute_code":
            result = self.execute_code(**action.parameters)
            reward = 1 if result["exit_code"] == 0 else -1
            self._state.step_count += 1
            return CodeObservation(reward=reward, **result)
        else:
            raise ValueError(f"Unknown action: {action.tool_name}")

    @property
    def state(self) -> State:
        return self._state
```


### Example 2: Game Environment (Non-Tool Actions)

```python
from core.env_server import Environment, Observation, State, ToolCallAction

class ChessEnv(Environment):
    """Chess environment - actions are game moves, not tools."""

    def __init__(self):
        self._board = chess.Board()
        self._state = GameState()

    @action("move_piece", "Move a chess piece from one square to another")
    def move_piece(self, from_square: str, to_square: str) -> Dict[str, Any]:
        """Move a chess piece.

        Args:
            from_square: Starting square (e.g., "e2")
            to_square: Destination square (e.g., "e4")

        Returns:
            Dict with move validity and game state
        """
        move = chess.Move.from_uci(f"{from_square}{to_square}")
        if move in self._board.legal_moves:
            self._board.push(move)
            return {
                "valid": True,
                "game_over": self._board.is_game_over(),
                "fen": self._board.fen(),
            }
        return {"valid": False, "error": "Illegal move"}

    def reset(self) -> Observation:
        self._board = chess.Board()
        self._state = GameState(episode_id=str(uuid.uuid4()))
        return ChessObservation(fen=self._board.fen(), legal_moves=list(self._board.legal_moves))

    def step(self, action: Action) -> Observation:
        if not isinstance(action, ToolCallAction):
            raise ValueError(f"Expected ToolCallAction, got {type(action)}")

        # Dispatch to action method
        if action.tool_name == "move_piece":
            result = self.move_piece(**action.parameters)
            reward = 1 if result.get("valid") else -1
            done = result.get("game_over", False)
            self._state.step_count += 1
            return ChessObservation(
                reward=reward,
                done=done,
                fen=result.get("fen"),
                valid_move=result.get("valid"),
            )
        else:
            raise ValueError(f"Unknown action: {action.tool_name}")

    @property
    def state(self) -> State:
        return self._state

    def actions(self) -> List[ToolDefinition]:
        """Return available actions (game moves, not tools)."""
        return [
            ToolDefinition(
                name="move_piece",
                description="Move a chess piece from one square to another",
                parameters=[
                    ToolParameter(name="from_square", type="string", description="Starting square (e.g., 'e2')"),
                    ToolParameter(name="to_square", type="string", description="Destination square (e.g., 'e4')"),
                ],
            )
        ]
```

### Example 3: Client-Side Usage with LLM

```python
from anthropic import Anthropic
from envs.coding_env import CodingEnv

# Initialize environment
env = CodingEnv.from_docker_image("coding-env:latest")

# Get available actions
actions = env.actions()  # Returns List[ToolDefinition]

# Convert to Claude's tool format (works for any action type!)
claude_tools = [action.to_json_schema() for action in actions]

# Initialize Claude client
client = Anthropic()

# Agent loop
observation = env.reset()
messages = [{"role": "user", "content": "Calculate fibonacci(10)"}]

while not observation.done:
    # Get model response with tools
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=messages,
        tools=claude_tools,
    )

    # If model wants to take an action
    if response.stop_reason == "tool_use":
        tool_use = response.content[0]

        # Create action from LLM's tool call
        # (works for code execution, game moves, or any action type)
        action = ToolCallAction(
            tool_name=tool_use.name,
            parameters=tool_use.input,
            tool_call_id=tool_use.id,
        )

        # Execute in environment
        observation = env.step(action)

        # Add action result to messages
        messages.append({
            "role": "assistant",
            "content": response.content,
        })
        messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",  # LLM APIs still call it "tool_result"
                "tool_use_id": tool_use.id,
                "content": str(observation),
            }],
        })
        print(observation.reward)
    else:
        break

env.close()
```

## References

- [Model Context Protocol (MCP) Specification](https://spec.modelcontextprotocol.io/) - Primary inspiration for this RFC
- [Anthropic Tool Use Documentation](https://docs.anthropic.com/claude/docs/tool-use)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- RFC 000: OpenEnv Project Phases
- RFC 001: OpenEnv Basic Abstractions
- RFC 002: OpenEnv Framework Spec
- RFC 003: MCP Support
