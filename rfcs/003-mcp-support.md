# RFC: MCP (Model Context Protocol) Support

**Status**: In Review
**Created**: 10/21/2025
**Authors**: @Darktex, @pankit-eng
**RFC ID:** 003

## Summary

This RFC defines how OpenEnv integrates with MCP (Model Context Protocol) to expose external tools to agents. We propose supporting both traditional function-calling paradigms and CodeAct-style execution by implementing an MCP client that exposes remote MCP server tools as Python functions within our execution environments.

## Motivation

### Problem Statement

Modern AI agents need access to external tools (web search, file operations, database queries, etc.). While MCP provides a standardized protocol for defining and exposing these tools, there are two distinct usage patterns that need support:

1. **Traditional Tool Calling**: Agent explicitly calls a tool by name with structured parameters (e.g., `call_tool("search_web", {"query": "python patterns"})`)
2. **CodeAct Paradigm**: Agent writes Python code that directly imports and calls tools as if they were native Python functions (e.g., `from tools import search_web; results = search_web(query="python patterns")`)

MCP's RPC-based, language-agnostic design works naturally for the first pattern but requires additional infrastructure for the second.

### Goals

1. **MCP Compatibility**: Support standard MCP servers without modification
2. **Dual Paradigm Support**: Enable both traditional tool calling and CodeAct execution styles
3. **Language Independence**: Leverage MCP's language-agnostic design to support tools written in any language
4. **Deployment Simplicity**: Provide patterns for deploying MCP servers alongside environments
5. **Developer Experience**: Make tools feel native to Python in CodeAct mode

## Background: MCP Architecture

### Overview

MCP (Model Context Protocol) is a protocol for exposing tools to AI models. It uses:
- **JSON Schema** for tool definitions and parameter validation
- **RPC-based communication** (typically over stdio, HTTP, or SSE)
- **Language independence** - servers can be written in any language

### Standard MCP Flow

```
┌─────────────┐                          ┌─────────────┐
│             │   1. list_tools()        │             │
│ MCP Client  │─────────────────────────>│ MCP Server  │
│             │                          │             │
│             │<─────────────────────────│             │
│             │   2. Tool definitions    │             │
│             │      (JSON Schema)       │             │
│             │                          │             │
│             │   3. call_tool(name,     │             │
│             │      params)             │             │
│             │─────────────────────────>│             │
│             │                          │             │
│             │<─────────────────────────│             │
│             │   4. Tool result (JSON)  │             │
└─────────────┘                          └─────────────┘
```

### Why MCP Works for Traditional Tool Calling

In RFC 004, we defined `ToolCallAction` with a `tool_name` and `parameters` structure. This maps naturally to MCP:

```python
# RFC 004 style
action = ToolCallAction(
    tool_name="search_web",
    parameters={"query": "python patterns", "max_results": 5}
)
observation = env.step(action)

# Maps to MCP call_tool RPC
mcp_client.call_tool(
    name="search_web",
    arguments={"query": "python patterns", "max_results": 5}
)
```

The environment can act as an MCP client, forwarding tool calls to the MCP server and returning results in observations.

### Why MCP Needs Adaptation for CodeAct

In CodeAct, agents write Python code that executes directly. **Best Practice**: Tools should be pre-imported in the execution environment, and the model should be informed of available tools via a system prompt. Any import statements written by the model should be ignored.

```python
# Agent generates this code (no imports!)
# Tools are already available: search_web, read_file

results = search_web(query="python patterns", max_results=5)
config = read_file(path="/workspace/config.json")
print(f"Found {len(results)} results")
```

**Pros**:
- **Security**: Prevents arbitrary module imports
- **Determinism**: Environment controls exactly what's available
- **Simplicity**: Model doesn't need to guess import syntax or module names
- **Reliability**: Avoids import errors and version conflicts

This requires:
1. **Pre-import tools**: Inject tool functions into the execution namespace before running agent code
2. **Function call translation**: Converting Python function calls to MCP RPC calls
3. **Type marshaling**: Converting between Python types and JSON for MCP communication
4. **Import filtering**: Strip or ignore import statements from agent-generated code

## Design

### Architecture Overview

#### 3. System Prompt for Tool Availability

When using CodeAct with pre-imported tools, the agent needs to know what's available. We provide this via a system prompt:

```python
def generate_tool_system_prompt(registry: MCPToolRegistry) -> str:
    """Generate a system prompt describing available tools.

    Args:
        registry: Tool registry containing available tools

    Returns:
        System prompt text describing available tools
    """
    prompt_parts = [
        "You are writing Python code that will be executed.",
        "The following tools are pre-imported and available for use:",
        ""
    ]

    for tool_name in registry.list_tools():
        tool = registry.get_tool(tool_name)
        prompt_parts.append(f"- {tool_name}: {tool.__doc__ or 'No description'}")

    prompt_parts.extend([
        "",
        "IMPORTANT: Do NOT write import statements. All tools are already available.",
        "Simply call the functions directly by name.",
        "",
        "Example:",
        "results = search_web(query='python async', max_results=5)",
        "content = read_file(path='/workspace/config.json')",
    ])

    return "\n".join(prompt_parts)
```

```
┌────────────────────────────────────────────────────────────────┐
│                    Docker Environment                          │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           Agent Code (Python)                           │   │
│  │                                                         │   │
│  │  # Traditional style (RFC 003)                          │   │
│  │  action = ToolCallAction(                               │   │
│  │      tool_name="search_web",                            │   │
│  │      parameters={"query": "..."}                        │   │
│  │  )                                                      │   │
│  │  env.step(action)                                       │   │
│  │                                                         │   │
│  │  # CodeAct style (NEW)                                  │   │
│  │  # Tools pre-imported, no import statements needed     │   │
│  │  results = search_web(query="...")                      │   │
│  └────────────────────┬────────────────────────────────────┘   │
│                       │                                        │
│                       │ Python import/call                     │
│                       ▼                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │         MCP Client (Python Library)                     │   │
│  │                                                         │   │
│  │  - Tool discovery & caching                             │   │
│  │  - Dynamic Python function generation                   │   │
│  │  - Type marshaling (Python ↔ JSON)                      │   │
│  │  - RPC communication with MCP servers                   │   │
│  └────────────────────┬────────────────────────────────────┘   │
│                       │ HTTP/SSE                               │
└───────────────────────┼────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ MCP Server 1 │ │ MCP Server 2 │ │ MCP Server N │
│ (Search)     │ │ (Files)      │ │ (Database)   │
└──────────────┘ └──────────────┘ └──────────────┘
```

### Core Components

#### 1. MCP Client Library

We need an MCP client that can run inside our Python execution environments. We have three options:

**Option A: Build our own** - Full control but requires maintenance
**Option B: Use FastMCP** - Popular, well-maintained Python MCP client
**Option C: Use mcp-use** - Alternative Python MCP library

**Recommendation**: Start with **FastMCP** or **mcp-use** as they provide:
- Standard MCP protocol implementation
- Tool discovery and schema parsing
- RPC communication primitives
- Active maintenance and community support

#### 2. Tool Registry & Namespace Injection

```python
from typing import Any, Callable, Dict
import inspect

class MCPToolRegistry:
    """Registry that exposes MCP tools as Python functions.

    This bridges MCP's RPC-based tool calling to Python's function
    call syntax, enabling CodeAct-style tool usage.

    Tools are injected into the execution namespace before running
    agent code, eliminating the need for import statements.
    """

    def __init__(self, mcp_clients: list[MCPClient]):
        """Initialize registry with one or more MCP clients.

        Args:
            mcp_clients: List of MCP clients connected to different servers
        """
        self.mcp_clients = mcp_clients
        self._tool_map: Dict[str, tuple[MCPClient, ToolDefinition]] = {}
        self._discover_tools()

    def _discover_tools(self) -> None:
        """Discover all tools from connected MCP servers."""
        for client in self.mcp_clients:
            tools = client.list_tools()
            for tool in tools:
                self._tool_map[tool.name] = (client, tool)

    def get_tool(self, name: str) -> Callable:
        """Get a Python-callable wrapper for an MCP tool.

        Args:
            name: Tool name

        Returns:
            Callable that executes the MCP tool

        Example:
            # Get a single tool wrapper
            search = registry.get_tool("search_web")
            results = search(query="python", max_results=5)
        """
        if name not in self._tool_map:
            raise ValueError(f"Tool '{name}' not found in registry")

        client, tool_def = self._tool_map[name]

        def tool_wrapper(**kwargs: Any) -> Any:
            """Generated wrapper function that calls MCP tool."""
            # Validate parameters against JSON schema
            self._validate_params(tool_def, kwargs)

            # Call MCP server
            result = client.call_tool(name, kwargs)

            # Parse and return result
            return result

        # Set function metadata for better introspection
        tool_wrapper.__name__ = name
        tool_wrapper.__doc__ = tool_def.description

        # Generate type hints from JSON schema (optional enhancement)
        # tool_wrapper.__annotations__ = self._schema_to_annotations(tool_def)

        return tool_wrapper

      def list_tools(self) -> list[str]:
        """List all available tool names."""
        return list(self._tool_map.keys())

    def get_all_tools(self) -> Dict[str, Callable]:
        """Get all tools as a dictionary for namespace injection.

        Returns:
            Dictionary mapping tool names to callable wrappers

        Example:
            tools = registry.get_all_tools()
            # Inject into execution namespace
            exec_globals = {**globals(), **tools}
            exec(agent_code, exec_globals)
        """
        return {name: self.get_tool(name) for name in self.list_tools()}

    def _validate_params(self, tool_def: ToolDefinition, params: Dict[str, Any]) -> None:
        """Validate parameters against tool's JSON schema."""
        # Use jsonschema library for validation
        # This ensures type safety even with dynamic calls
        pass


def filter_imports(code: str) -> str:
    """Remove import statements from agent-generated code.

    This prevents models from attempting to import modules,
    since all tools are pre-imported into the namespace.

    Args:
        code: Python code that may contain import statements

    Returns:
        Code with import statements removed

    Example:
        code = '''
        from tools import search_web
        import os

        result = search_web(query="test")
        '''

        filtered = filter_imports(code)
        # filtered = "result = search_web(query='test')"
    """
    import re
    # Remove 'import ...' and 'from ... import ...' lines
    lines = code.split('\n')
    filtered_lines = [
        line for line in lines
        if not re.match(r'^\s*(import\s+|from\s+.*\s+import\s+)', line)
    ]
    return '\n'.join(filtered_lines)
```

### Integration with Environment Interface

#### Traditional Tool Calling (RFC 003 Style)

Environments act as MCP clients:

```python
from core.env_server import Environment, Observation
from mcp_client import MCPClient

class ToolCallingEnvironment(Environment):
    """Environment that forwards ToolCallActions to MCP servers."""

    def __init__(self, mcp_servers: list[str]):
        self.mcp_clients = [MCPClient(url) for url in mcp_servers]
        self.registry = MCPToolRegistry(self.mcp_clients)

    def step(self, action: Action) -> Observation:
        if isinstance(action, ToolCallAction):
            # Forward to MCP server
            tool = self.registry.get_tool(action.tool_name)
            result = tool(**action.parameters)

            # Convert result to observation
            return self._make_observation(result)
        else:
            raise ValueError(f"Expected ToolCallAction, got {type(action)}")

    def tools(self) -> list[ToolDefinition]:
        """RFC 003 tool discovery API."""
        return [tool_def for _, tool_def in self.registry._tool_map.values()]
```

#### CodeAct Style

Python code execution environments pre-import tools into the execution namespace:

```python
from core.env_server import Environment
from core.tools import PyExecutor
from mcp_client import MCPClient, MCPToolRegistry, filter_imports

class CodeActEnvironment(Environment):
    """Environment for CodeAct with MCP tool access."""

    def __init__(self, mcp_servers: list[str]):
        self.executor = PyExecutor()

        # Initialize MCP clients and registry
        mcp_clients = [MCPClient(url) for url in mcp_servers]
        self.registry = MCPToolRegistry(mcp_clients)

        # Pre-import all tools into execution namespace
        self.tool_namespace = self.registry.get_all_tools()

    def step(self, action: CodeAction) -> Observation:
        # Filter out any import statements from agent code
        filtered_code = filter_imports(action.code)

        # Execute with tools pre-injected into namespace
        result = self.executor.run(
            filtered_code,
            extra_globals=self.tool_namespace
        )
        return self._make_observation(result)

    def get_system_prompt(self) -> str:
        """Get system prompt describing available tools."""
        return generate_tool_system_prompt(self.registry)
```

### MCP Server Deployment

#### Deployment Pattern

MCP servers should be deployed alongside environment containers. We propose a Docker Compose pattern:

```yaml
# docker-compose.yml
version: '3.8'

services:
  # Main environment container
  environment:
    build: ./environment
    ports:
      - "8000:8000"
    environment:
      - MCP_SERVERS=http://mcp-search:8001,http://mcp-files:8002
    depends_on:
      - mcp-search
      - mcp-files
    networks:
      - agent-network

  # MCP server for web search
  mcp-search:
    image: mcp-search-server:latest
    ports:
      - "8001:8001"
    environment:
      - SEARCH_API_KEY=${SEARCH_API_KEY}
    networks:
      - agent-network

  # MCP server for file operations
  mcp-files:
    image: mcp-files-server:latest
    ports:
      - "8002:8002"
    volumes:
      - workspace:/workspace:ro
    networks:
      - agent-network

networks:
  agent-network:
    driver: bridge

volumes:
  workspace:
```

#### Environment Configuration

Environments specify required MCP servers in their configuration:

```python
# environment_config.py
from dataclasses import dataclass
from typing import List

@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str
    image: str
    port: int
    env_vars: dict[str, str] = None

@dataclass
class EnvironmentConfig:
    """Environment configuration including MCP dependencies."""
    name: str
    image: str
    mcp_servers: List[MCPServerConfig]

# Example configuration
CODING_ENV_CONFIG = EnvironmentConfig(
    name="coding-env",
    image="coding-env:latest",
    mcp_servers=[
        MCPServerConfig(
            name="search",
            image="mcp-search-server:latest",
            port=8001,
            env_vars={"SEARCH_API_KEY": "${SEARCH_API_KEY}"}
        ),
        MCPServerConfig(
            name="files",
            image="mcp-files-server:latest",
            port=8002,
        ),
    ]
)
```

#### Build & Deployment Tools

We provide utilities to generate Docker Compose files from environment configs:

```python
from pathlib import Path
import yaml

def generate_compose_file(config: EnvironmentConfig, output_path: Path) -> None:
    """Generate docker-compose.yml from environment config."""
    compose = {
        "version": "3.8",
        "services": {},
        "networks": {"agent-network": {"driver": "bridge"}},
    }

    # Main environment service
    mcp_urls = [f"http://{s.name}:{s.port}" for s in config.mcp_servers]
    compose["services"]["environment"] = {
        "image": config.image,
        "ports": ["8000:8000"],
        "environment": {
            "MCP_SERVERS": ",".join(mcp_urls)
        },
        "depends_on": [s.name for s in config.mcp_servers],
        "networks": ["agent-network"],
    }

    # MCP server services
    for server in config.mcp_servers:
        compose["services"][server.name] = {
            "image": server.image,
            "ports": [f"{server.port}:{server.port}"],
            "networks": ["agent-network"],
        }
        if server.env_vars:
            compose["services"][server.name]["environment"] = server.env_vars

    # Write compose file
    output_path.write_text(yaml.dump(compose))

# Usage
generate_compose_file(CODING_ENV_CONFIG, Path("docker-compose.yml"))
```

## Key Design Decisions

### Decision 1: MCP Client Implementation

**Chosen Approach**: Use existing Python MCP client library (FastMCP or mcp-use) rather than building our own.

**Rationale**:
- **Faster development**: Leverage existing, tested implementations
- **Standard compliance**: These libraries follow MCP spec changes
- **Community support**: Benefit from community bug fixes and features
- **Focus on value-add**: Spend effort on CodeAct integration, not protocol details

**Trade-offs**:
- External dependency (mitigated by vendoring if needed)
- Less control over implementation details

### Decision 2: Pre-Import Tools vs Import Statements

**Chosen Approach**: Pre-import all tools into the execution namespace and filter out import statements from agent code.

**Rationale**:
- **Security**: Prevents arbitrary module imports that could access system resources
- **Determinism**: Environment has full control over available tools
- **Reliability**: Eliminates import errors and module not found issues
- **Simplicity**: Model doesn't need to know correct import syntax
- **Best Practice**: Aligns with sandboxed code execution principles

**Trade-offs**:
- Requires filtering/stripping import statements from agent code
- Need clear system prompts to inform model of available tools
- Less "natural" than writing actual imports (but safer and more reliable)

### Decision 3: Docker Compose for MCP Server Orchestration

**Chosen Approach**: Use Docker Compose to deploy MCP servers alongside environment containers.

**Rationale**:
- **Declarative**: Clear specification of dependencies
- **Standard tooling**: Docker Compose is widely understood
- **Networking**: Built-in network isolation and service discovery
- **Development experience**: Easy local testing with `docker-compose up`

**Trade-offs**:
- Additional complexity for simple environments without tools
- May need adaptation for Kubernetes deployments (future RFC)

## Examples

### Example 1: Traditional Tool Calling with MCP

```python
from envs.tool_calling_env import ToolCallingEnv, ToolCallAction

# Start environment with MCP servers
env = ToolCallingEnv.from_docker_compose("docker-compose.yml")

# Discover available tools
tools = env.tools()
print(f"Available tools: {[t.name for t in tools]}")
# Output: ['search_web', 'read_file', 'write_file']

# Reset environment
obs = env.reset()

# Agent makes tool call
action = ToolCallAction(
    tool_name="search_web",
    parameters={"query": "Python async patterns", "max_results": 5}
)

obs = env.step(action)
print(obs.metadata["tool_result"])
# Output: [{"title": "...", "url": "...", ...}, ...]

env.close()
```

### Example 2: CodeAct with MCP Tools

```python
from envs.codeact_env import CodeActEnv, CodeAction

# Start environment (MCP servers started automatically)
env = CodeActEnv.from_docker_compose("docker-compose.yml")

# Reset environment
obs = env.reset()

# Get system prompt to inform agent of available tools
system_prompt = env.get_system_prompt()
print(system_prompt)
# Output:
# You are writing Python code that will be executed.
# The following tools are pre-imported and available for use:
#
# - search_web: Search the web for information
# - read_file: Read contents of a file
# ...

# Agent generates code that uses MCP tools (no imports!)
agent_code = """
# Tools are already available, just use them directly
results = search_web(query="Python async patterns", max_results=5)
print(f"Found {len(results)} results")

# Read configuration
config = read_file(path="/workspace/config.json")
print(f"Config: {config}")
"""

# Execute code (tools available transparently)
action = CodeAction(code=agent_code)
obs = env.step(action)

print(obs.stdout)
# Output:
# Found 5 results
# Config: {...}

env.close()
```

### Example 3: Building MCP Server for Custom Tools

```python
# my_tools_server.py
from fastmcp import FastMCP

mcp = FastMCP("My Custom Tools")

@mcp.tool()
def analyze_sentiment(text: str) -> dict:
    """Analyze sentiment of text.

    Args:
        text: Text to analyze

    Returns:
        Dictionary with sentiment scores
    """
    # Your implementation
    return {
        "positive": 0.8,
        "negative": 0.1,
        "neutral": 0.1
    }

@mcp.tool()
def summarize_text(text: str, max_length: int = 100) -> str:
    """Summarize text to specified length.

    Args:
        text: Text to summarize
        max_length: Maximum length of summary

    Returns:
        Summarized text
    """
    # Your implementation
    return text[:max_length] + "..."

if __name__ == "__main__":
    mcp.run()
```

```dockerfile
# Dockerfile for custom MCP server
FROM python:3.10-slim

WORKDIR /app

RUN pip install fastmcp

COPY my_tools_server.py .

EXPOSE 8001

CMD ["python", "my_tools_server.py"]
```
## Open Questions

1. **Caching**: Should we cache tool results, and if so, what's the invalidation strategy?
2. **Streaming**: How to handle streaming responses from MCP servers (e.g., long-running operations)?
3. **Error Handling**: Should MCP errors be propagated as exceptions or returned in observations?
4. **Versioning**: How to handle version compatibility between MCP clients and servers?

## References

- [Model Context Protocol Specification](https://spec.modelcontextprotocol.io/)
- [FastMCP Python Library](https://github.com/jlowin/fastmcp)
- [mcp-use Python Library](https://github.com/mcp-use/mcp-use)
- RFC 000: OpenEnv Project Phases
- RFC 001: OpenEnv Basic Abstractions
- RFC 002: OpenEnv Framework Spec
- RFC 004: Support multiple tool calls via Action wrapper abstraction
