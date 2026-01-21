### PR: Fleet environments (OpenEnv)

This PR documents and refines the **Fleet** runtime integration for OpenEnv.

#### What this enables
- Run OpenEnv environments on **Fleet (remote)** with **no local Docker**.
- Keep a strict split between:
  - **Orchestration (HTTP)**: `reset / step / state`
  - **Agent actions (MCP)**: `tools/list + tools/call`

#### What this is *not*
- This is **not** the local “Dockerized env server + env container” setup.
- There is **no container/provider abstraction** here; Fleet hosts the runtime remotely (HTTP env server + MCP service). The client only connects.

#### Main abstractions
- **`FleetEnvClient` (HTTP)**: orchestrator handle for reset/step/state.
- **`FleetMCPTools` (MCP)**: agent handle for listing/calling tools.
  - Unions tools across Fleet’s MCP endpoints (today often `api/v1/mcp` and `mcp`)
  - Returns tools in **OpenAI “tools” dict format** (via `convert_tool_format`)
  - Routes tool calls to the owning endpoint (cached after discovery)

#### Quickstart
- Install: `pip install "openenv-core[fleet]"`
- Set: `export FLEET_API_KEY="..."`
- Run: `python examples/fleet_env_example.py <env_key>`

#### References
- RFC 001: `rfcs/001-abstractions.md`
- RFC 003: `rfcs/003-mcp-support.md`

#### TODOs / known sharp edges
- Endpoint discovery (avoid hardcoding `api/v1/mcp` vs `mcp`)
- Reset inconsistencies across some env keys (better errors + compatibility notes)
- Tool-name collision policy across endpoints
- Retries/backoff and clearer “endpoint down” failure modes

