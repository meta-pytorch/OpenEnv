### Fleet Runtime Integration (OpenEnv) — Design Proposal

### Goal
Run OpenEnv environments on **Fleet** (remote) with **no Docker**, strictly adhering to:
- **RFC 001**: Agent interacts via MCP tools; Orchestration via HTTP.
- **RFC 003**: Standardized `ListToolsAction` and `CallToolAction`.

### Architecture

We implement a client-side adapter (`FleetEnvClient`) that aggregates Fleet's interfaces into the OpenEnv contract.

```mermaid
flowchart TB
  subgraph Client["OpenEnv Client (Local)"]
    direction TB
    Agent["Agent / Policy"]
    Orch["FleetEnvClient<br/>(Orchestrator, HTTP)"]
    Tools["FleetMCPTools<br/>(Agent, MCP)"]
  end

  subgraph Runtime["Fleet Runtime (Remote)"]
    direction TB
    HTTP["Instance Manager HTTP API<br/>/reset /step /state"]
    MCP_SVC["Fleet MCP Service<br/>(Multiple endpoints aggregated)"]
  end

  %% Orchestration (RFC 001)
  Orch --"reset()/step()/state()"--> HTTP

  %% Agent actions (RFC 003)
  Agent --"tools/list, tools/call"--> Tools
  Tools <-->|"SSE Session (Multiplexed)"| MCP_SVC
```

### 1. Combined Action Space (Client-Side Multiplexing)
Fleet instances currently expose multiple MCP endpoints (e.g., `api/v1/mcp` for browser control, `mcp` for API tools). 

**The Strategy:**
1.  **Connect to ALL**: The client establishes sessions with both `root + "api/v1/mcp"` and `root + "mcp"`.
2.  **Union Tools**: `FleetMCPTools.list_tools()` returns the union of tools from all connected endpoints.
3.  **Route Execution**: `FleetMCPTools.call_tool()` routes the call to the endpoint that owns the tool.

> **Future Work**: This client-side multiplexing is a temporary workaround. Future versions of the Fleet API will expose a single unified MCP endpoint that aggregates all tools server-side, removing the need for the client to know about specific paths like `api/v1/mcp`.

### 2. Client Implementation (`FleetEnvClient`)

This adapter replaces `LocalDockerProvider` and remains orchestration-only (HTTP). Agent tool calls are handled by `FleetMCPTools` (MCP).

```python
# Pseudocode implementation of the Client Adapter
class FleetEnvClient(HTTPEnvClient):
    @classmethod
    def from_fleet(cls, api_key, env_key, **kwargs):
        # 1. Provision Instance via Fleet SDK
        env = fleet.make(env_key, ...)
        
        # 2. Establish MCP Sessions (Streamable HTTP)
        # We connect to BOTH to provide the full browser + api toolset
        mcp_sessions = []
        for path in ["api/v1/mcp", "mcp"]:
            url = f"{env.urls.root}{path}"
            if is_reachable(url):
                mcp_sessions.append(connect_mcp(url, api_key))
                
        orch = cls(base_url=env.urls.manager.api)
        tools = FleetMCPTools(mcp_urls=mcp_sessions)
        return orch, tools

    # step/reset/state remain HTTP only
```

### 3. Usage (User Perspective)

```python
# The user simply provides keys. No Docker required.
orch, tools = FleetEnvClient.from_fleet(
    api_key=os.environ["FLEET_API_KEY"],
    env_key=os.environ["FLEET_ENV_KEY"]
)

# Orchestrator controls episode (HTTP)
orch.reset()

# Agent uses MCP tools (Browser + API)
tools_list = await tools.list_tools()
result = await tools.call_tool("computer", {...})
```

### Architecture Note: Strict Separation of Concerns

This implementation enforces a strict boundary between the **Orchestration Plane** and the **Agent Plane**, aligning with RFC 001.

- **Orchestrator (`FleetEnvClient`)**: Has access to the HTTP control plane (`reset`, `state`, `step`). It handles environment lifecycle and simulation stepping (if applicable).
- **Agent (`FleetMCPTools`)**: Has access *only* to the MCP tool capabilities (`list_tools`, `call_tool`). It cannot reset or delete the environment.

This avoids "leaking" powerful orchestration capabilities (like `reset` or `delete`) to the agent runtime. Unlike a "bridged" implementation where `env.step()` handles everything, this design requires the caller to explicitly use the correct handle for the correct intent.