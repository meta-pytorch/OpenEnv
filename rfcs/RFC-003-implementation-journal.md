# RFC-003 Implementation Journal

## Overview
This journal tracks the implementation of RFC-003: MCP (Model Context Protocol) Support for OpenEnv.

**RFC Goal**: Integrate MCP as the universal interface for all actions exposed to agents, supporting both tool-calling and CodeAct paradigms.

**Implementation Phases**:
- **PR #1** (Current): Core MCP infrastructure + echo_env conversion
- **PR #2** (Future): Migrate remaining environments
- **PR #3** (Future): CodeAct environment with MCP integration
- **PR #4** (Future): Fix double marshaling with callable introspection

---

## PR #1: Core MCP Infrastructure + Echo Env

**Branch**: (will be set when starting sapling commit)

**Goals**:
1. Add MCP client/server base classes to core package
2. Add new action types (ListToolsAction, CallToolAction)
3. Integrate MCP into Environment base class
4. Add /mcp JSON-RPC endpoint to FastAPI
5. Convert echo_env as reference implementation
6. Maintain backward compatibility with existing envs

### Architecture Decisions

#### 1. Dual Interface Model
**Decision**: Environments expose both HTTP (orchestration) and MCP (agent actions) interfaces.

- **HTTP `/step` endpoint**: Accepts ListToolsAction/CallToolAction, routes to MCP client internally
- **HTTP `/mcp` endpoint**: Direct JSON-RPC access to MCP servers (for production/inference)
- **HTTP `/reset`, `/state`**: Unchanged (orchestration only, not exposed as tools)

**Rationale**:
- Training code uses `/step` API for gym-style control
- Production agents can use `/mcp` directly without step wrapper
- Minimizes delta between training and production

#### 2. In-Process MCP Server
**Decision**: Run MCP server in same Python process as FastAPI, using asyncio.

**Implementation approach**:
- FastMCP server and FastAPI share same event loop
- Communication via stdio using asyncio subprocess
- No threading required

**Rationale**:
- Simpler deployment (one process)
- Lower latency for local tools
- Easier debugging

#### 3. Library Choice
**Decision**: Use FastMCP for server creation, MCP SDK for client.

**Rationale**:
- FastMCP provides high-level decorators for tool registration
- Official MCP SDK for standard client implementation
- Both maintained by Anthropic

#### 4. No Callable Introspection (Yet)
**Decision**: Defer callable introspection optimization to PR #4.

**Rationale**:
- Keep PR #1 focused on infrastructure
- Callable introspection only needed for CodeAct (PR #3)
- Easier to review incrementally

---

### Implementation Checklist

#### Core Infrastructure
- [x] Create `src/core/mcp/__init__.py`
- [x] Create `src/core/mcp/client.py` (MCPClient class)
- [x] Create `src/core/mcp/server.py` (LocalMCPServer class)
- [x] Add dependencies to `src/core/pyproject.toml`

#### Type System
- [x] Add `ListToolsAction` to `src/core/env_server/types.py`
- [x] Add `CallToolAction` to `src/core/env_server/types.py`
- [x] Export new types from `src/core/env_server/__init__.py`

#### Environment Integration
- [x] Add `mcp_client` parameter to `Environment.__init__`
- [x] Add `_handle_mcp_action()` method to `Environment`
- [x] Update HTTP server /step endpoint to route MCP actions

#### HTTP Server
- [x] Add `/mcp` endpoint to `create_fastapi_app()`
- [x] Add mcp_server parameter to HTTPEnvServer
- [x] Update action deserialization to handle MCP actions
- [x] Ensure backward compatibility with existing endpoints

#### Echo Env Conversion
- [x] Deprecate `src/envs/echo_env/models.py` (EchoAction deprecated)
- [x] Create `src/envs/echo_env/server/mcp_server.py`
- [x] Define `echo_message` tool using FastMCP
- [x] Update `src/envs/echo_env/server/echo_environment.py`
- [x] Update `src/envs/echo_env/server/app.py` to initialize MCP
- [x] Update `src/envs/echo_env/client.py` with MCP actions
- [ ] Update echo_env Dockerfile dependencies

#### Tests
- [ ] Create `tests/core/mcp/test_client.py`
- [ ] Create `tests/core/mcp/test_server.py`
- [ ] Update `tests/envs/test_echo_env.py`
- [ ] Verify other envs still work (no regressions)

#### Documentation
- [ ] Create example `examples/echo_mcp_demo.py`
- [ ] Update `CLAUDE.md` with MCP notes
- [ ] Update echo_env README

---

### Implementation Notes

#### Session: 2025-11-24 (Part 1)

**Status**: Core infrastructure and echo_env conversion COMPLETED

**What Was Implemented**:

1. **Core MCP Infrastructure** (`src/core/mcp/`):
   - Created `MCPClient` class for communicating with MCP servers
   - Created `LocalMCPServer` wrapper around FastMCP
   - Added `compose_servers()` function for combining multiple MCP servers
   - In-process communication (FastMCP Server passed directly to client)

2. **New Action Types** (`src/core/env_server/types.py`):
   - `ListToolsAction`: Requests available tools from MCP servers
   - `CallToolAction`: Calls a specific tool with parameters

3. **Environment Integration** (`src/core/env_server/interfaces.py`):
   - Added optional `mcp_client` parameter to `Environment.__init__`
   - Implemented `_handle_mcp_action()` async method for processing MCP actions
   - Returns observations with tools list or tool call results in metadata

4. **HTTP Server Updates** (`src/core/env_server/http_server.py`):
   - Added `/mcp` POST endpoint for direct JSON-RPC access
   - Updated `/step` endpoint to handle ListToolsAction and CallToolAction
   - Enhanced `_deserialize_action()` to detect and deserialize MCP action types
   - Added `mcp_server` parameter to `HTTPEnvServer` and `create_fastapi_app()`

5. **Echo Env Conversion**:
   - Created `mcp_server.py` with `echo_message` tool using FastMCP decorators
   - Rewrote `EchoEnvironment` to use MCP client instead of custom actions
   - Updated `app.py` to initialize MCP server, client, and wire them together
   - Deprecated `EchoAction` and `EchoObservation` in `models.py` with warnings
   - Updated `EchoEnv` client to use `CallToolAction` with convenience methods

**Key Implementation Details**:

- **MCP Client**: Caches tool list on initialization for performance
- **Error Handling**: Tool call errors returned in observation metadata with "error" key
- **Action Deserialization**: Checks for "type" or "action_type" field to detect MCP actions
- **Backward Compatibility**: Existing environments unaffected (mcp_client is optional)

**Next Steps**:
1. Write tests for MCP infrastructure
2. Test echo_env end-to-end
3. Create example demonstrating both `/step` and `/mcp` interfaces
4. Update documentation

---

### Open Questions

1. **Error Handling**: How should MCP errors be surfaced?
   - ~~Option A: Raise exceptions (breaks step)~~
   - **Option B: Return error in observation** ✅ CHOSEN
   - **Decision**: Errors returned in `observation.metadata["error"]` to maintain step flow

2. **Tool Result Format**: How to structure tool call results in observations?
   - **Current implementation**: `observation.metadata["result"]` contains tool output ✅
   - **List tools**: `observation.metadata["tools"]` contains array of tool schemas ✅

3. **Caching**: Should tool results be cached?
   - **Current**: Tool list cached on client initialization, tool calls not cached ✅
   - Future: May add tool call caching in later PR

4. **Testing**: How to test MCP integration without external dependencies?
   - **Approach**: Use in-process FastMCP servers for testing
   - Mock not needed since FastMCP can be instantiated directly

---

### Testing Strategy

1. **Unit Tests**:
   - Test MCPClient can list tools
   - Test MCPClient can call tools
   - Test LocalMCPServer tool registration
   - Mock not needed since FastMCP can be instantiated directly

2. **Integration Tests**:
   - Test echo_env with ListToolsAction
   - Test echo_env with CallToolAction
   - Test /mcp endpoint directly
   - Verify other envs unchanged

3. **Manual Testing**:
   - Build echo_env Docker image
   - Run examples/echo_mcp_demo.py
   - Verify both /step and /mcp interfaces work

---

### Running the Server and Tests

#### Setup

```bash
# Create virtual environment
uv venv

# Install core package with MCP dependencies
uv pip install -e src/core/

# Clear Python cache (if needed)
find src/envs/echo_env -type d -name "__pycache__" -exec rm -rf {} +
```

#### Running Echo Env Server

```bash
# From project root
cd src
uvicorn envs.echo_env.server.app:app --port 8000 --reload

# Or using absolute path to uvicorn from venv
../.venv/bin/uvicorn envs.echo_env.server.app:app --port 8000
```

Server will start on http://localhost:8000

#### Testing Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Reset environment
curl -X POST http://localhost:8000/reset

# List tools via /step endpoint
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"type": "ListToolsAction"}}'

# Call tool via /step endpoint
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "type": "CallToolAction",
      "tool_name": "echo_message",
      "parameters": {"message": "Hello MCP!"}
    }
  }'

# Direct MCP access via /mcp endpoint (JSON-RPC)
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/list",
    "id": 1
  }'

curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "echo_message",
      "arguments": {"message": "Hello from MCP endpoint!"}
    },
    "id": 2
  }'
```

#### Running Tests

```bash
# Install test dependencies
cd src/core
uv pip install -e ".[dev]"

# Run all tests
pytest

# Run specific test file
pytest tests/core/mcp/test_mcp.py

# Run with verbose output
pytest -v tests/core/mcp/test_mcp.py
```

---

### Future Work (Later PRs)

**PR #2: Environment Migration**
- Convert coding_env, browsergym_env, etc. to MCP pattern
- Document migration guide for environment authors

**PR #3: CodeAct Integration**
- Modify PythonCodeActEnv to support MCP tools
- Add `list_tools()` function in code execution context
- Inject tool functions for direct calling

**PR #4: Fix Double Marshaling**
- Add callable introspection to LocalMCPServer
- Store `_callables` dict alongside MCP protocol handlers
- Inject raw Python functions into CodeAct namespace
- Benchmark performance improvement

---

## References

- [RFC-003](../../rfcs/003-mcp-support.md)
- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
