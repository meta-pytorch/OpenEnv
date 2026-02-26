// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

/**
 * Configuration types and loader for openclaw-agent-server.
 *
 * The config JSON is written by AgentKernel's spawner and passed via
 * --config <path>.  Fields mirror the Python OpenClawSpawnInfo dataclass.
 */

import { readFileSync } from "node:fs";

export interface OpenClawRunnerConfig {
  /** Unique agent identifier (UUID from AgentKernel). */
  agent_id: string;
  /** Secret nonce for authenticating Turn requests. */
  nonce: string;
  /** Human-readable agent name. */
  name: string;
  /** Always "openclaw". */
  agent_type: string;
  /** System prompt for the agent. */
  system_prompt: string;
  /** LLM model identifier (e.g. "claude-sonnet-4-5"). */
  model: string;
  /** LLM provider (e.g. "anthropic", "openai"). */
  provider: string;
  /** Enabled tools (e.g. ["bash", "browser"]). */
  tools: string[];
  /** Thinking level: "none" | "low" | "medium" | "high". */
  thinking_level: string;
  /** HTTP port to listen on. */
  http_port: number;
  /** API key for the LLM provider. */
  api_key?: string;
  /** Base URL override for the LLM provider. */
  base_url?: string;
  /** AgentBus gRPC endpoint (e.g. "memory://9999"). */
  agentbus_url?: string;
  /** Whether to disable safety checks. */
  disable_safety?: boolean;
}

/**
 * Load config from a JSON file path.
 */
export function loadConfig(configPath: string): OpenClawRunnerConfig {
  const raw = readFileSync(configPath, "utf-8");
  const config = JSON.parse(raw) as OpenClawRunnerConfig;

  // Validate required fields
  if (!config.agent_id) throw new Error("config: agent_id is required");
  if (!config.nonce) throw new Error("config: nonce is required");
  if (!config.http_port) throw new Error("config: http_port is required");

  // Defaults
  config.model = config.model || "claude-sonnet-4-5";
  config.provider = config.provider || "anthropic";
  config.tools = config.tools || ["bash"];
  config.thinking_level = config.thinking_level || "none";
  config.system_prompt = config.system_prompt || "";

  return config;
}

/**
 * Parse --config <path> from process.argv.
 */
export function parseArgs(): string {
  const idx = process.argv.indexOf("--config");
  if (idx === -1 || idx + 1 >= process.argv.length) {
    console.error("Usage: openclaw-agent-server --config <path>");
    process.exit(1);
  }
  return process.argv[idx + 1]!;
}
