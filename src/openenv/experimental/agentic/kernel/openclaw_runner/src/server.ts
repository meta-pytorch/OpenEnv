#!/usr/bin/env node
// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

/**
 * openclaw-agent-server — HTTP server implementing the AgentKernel contract.
 *
 * Endpoints:
 *   GET  /health     → {"status": "ok", "agent_id": "..."}
 *   POST /v1/turn    → SSE stream of {"body": "...", "done": bool, "error": "..."}
 *   POST /v1/history → {"entries": [{"role": "...", "content": "..."}]}
 *   POST /v1/control → no-op stub
 *
 * Reads --config <path> arg, parses JSON config written by AgentKernel spawner.
 * Optionally connects to AgentBus for safety gating when agentbus_url is present.
 */

import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { loadConfig, parseArgs, type OpenClawRunnerConfig } from "./config.js";
import { OpenClawAgent } from "./agent.js";
import { AgentBusSafetyExecutor } from "./agentbus-plugin.js";

// ── Globals ───────────────────────────────────────────────────────────

let config: OpenClawRunnerConfig;
let agent: OpenClawAgent;
let safetyExecutor: AgentBusSafetyExecutor | null = null;

// ── Request helpers ───────────────────────────────────────────────────

async function readBody(req: IncomingMessage): Promise<string> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    chunks.push(typeof chunk === "string" ? Buffer.from(chunk) : chunk);
  }
  return Buffer.concat(chunks).toString("utf-8");
}

function jsonResponse(res: ServerResponse, status: number, data: any): void {
  const body = JSON.stringify(data);
  res.writeHead(status, {
    "Content-Type": "application/json",
    "Content-Length": Buffer.byteLength(body),
  });
  res.end(body);
}

function sseChunk(res: ServerResponse, data: any): void {
  res.write(`data: ${JSON.stringify(data)}\n\n`);
}

// ── Route handlers ────────────────────────────────────────────────────

function handleHealth(_req: IncomingMessage, res: ServerResponse): void {
  jsonResponse(res, 200, {
    status: "ok",
    agent_id: config.agent_id,
  });
}

async function handleTurn(req: IncomingMessage, res: ServerResponse): Promise<void> {
  let payload: any;
  try {
    const raw = await readBody(req);
    payload = JSON.parse(raw);
  } catch {
    jsonResponse(res, 400, { error: "Invalid JSON body" });
    return;
  }

  // Validate nonce
  if (payload.nonce !== config.nonce) {
    jsonResponse(res, 403, { error: "Invalid nonce" });
    return;
  }

  // Parse the body field (contains the actual messages)
  let messages: Array<{ role: string; content: string }>;
  try {
    const body = typeof payload.body === "string" ? JSON.parse(payload.body) : payload.body;
    messages = body.messages || [];
  } catch {
    jsonResponse(res, 400, { error: "Invalid body.messages" });
    return;
  }

  // Set up SSE headers
  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
  });

  // Start a heartbeat to keep the connection alive during HITL waits
  const heartbeat = setInterval(() => {
    res.write(": heartbeat\n\n");
  }, 15_000);

  try {
    for await (const chunk of agent.turn(messages)) {
      sseChunk(res, chunk);
    }
  } catch (err: any) {
    sseChunk(res, { body: "", done: true, error: err?.message || String(err) });
  } finally {
    clearInterval(heartbeat);
    res.end();
  }
}

async function handleHistory(_req: IncomingMessage, res: ServerResponse): Promise<void> {
  const entries = agent.getHistory();
  jsonResponse(res, 200, { entries });
}

async function handleControl(_req: IncomingMessage, res: ServerResponse): Promise<void> {
  // No-op stub — OpenClaw agents don't support control operations yet
  jsonResponse(res, 200, { status: "ok" });
}

// ── Router ────────────────────────────────────────────────────────────

async function handleRequest(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const method = req.method?.toUpperCase() ?? "GET";
  const url = req.url ?? "/";

  try {
    if (method === "GET" && url === "/health") {
      handleHealth(req, res);
    } else if (method === "POST" && url === "/v1/turn") {
      await handleTurn(req, res);
    } else if (method === "POST" && url === "/v1/history") {
      await handleHistory(req, res);
    } else if (method === "POST" && url === "/v1/control") {
      await handleControl(req, res);
    } else {
      jsonResponse(res, 404, { error: "Not found" });
    }
  } catch (err: any) {
    console.error(`[server] Unhandled error on ${method} ${url}: ${err}`);
    if (!res.headersSent) {
      jsonResponse(res, 500, { error: err?.message || "Internal server error" });
    }
  }
}

// ── AgentBus integration ──────────────────────────────────────────────

async function connectAgentBus(): Promise<void> {
  if (!config.agentbus_url) return;

  console.error(`[server] Connecting to AgentBus: ${config.agentbus_url}`);
  const enabled = !config.disable_safety;

  try {
    safetyExecutor = new AgentBusSafetyExecutor(config.agent_id, enabled);
    await safetyExecutor.connect(config.agentbus_url);

    // Wire the safety executor into the agent
    agent.busClient = safetyExecutor as any;
    agent.safetyEnabled = enabled;

    console.error(
      `[server] AgentBus connected (safety_enabled=${enabled})`,
    );
  } catch (err) {
    console.error(`[server] Failed to connect to AgentBus: ${err}`);
    console.error("[server] Continuing without AgentBus integration");
    safetyExecutor = null;
  }
}

// ── Main ──────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const configPath = parseArgs();
  config = loadConfig(configPath);

  console.error(
    `[server] Starting openclaw-agent-server (agent_id=${config.agent_id}, ` +
      `port=${config.http_port}, model=${config.model})`,
  );

  // Create the agent
  agent = new OpenClawAgent(config);

  // Connect to AgentBus (if configured)
  await connectAgentBus();

  // Start HTTP server
  const server = createServer((req, res) => {
    handleRequest(req, res).catch((err) => {
      console.error(`[server] Request handler error: ${err}`);
      if (!res.headersSent) {
        jsonResponse(res, 500, { error: "Internal server error" });
      }
    });
  });

  server.listen(config.http_port, "127.0.0.1", () => {
    console.error(
      `[server] Listening on http://127.0.0.1:${config.http_port}`,
    );
  });

  // Graceful shutdown
  for (const signal of ["SIGTERM", "SIGINT"] as const) {
    process.on(signal, () => {
      console.error(`[server] Received ${signal}, shutting down`);
      safetyExecutor?.close();
      server.close();
      process.exit(0);
    });
  }
}

main().catch((err) => {
  console.error(`[server] Fatal error: ${err}`);
  process.exit(1);
});
