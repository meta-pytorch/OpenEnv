// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

/**
 * TypeScript gRPC client for AgentBus.
 *
 * Uses @grpc/proto-loader for dynamic proto loading (no codegen step).
 * Follows the Python AgentBusClient pattern exactly.
 */

import * as grpc from "@grpc/grpc-js";
import * as protoLoader from "@grpc/proto-loader";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Proto file is symlinked into proto/ directory
const PROTO_PATH = join(__dirname, "..", "proto", "agent_bus.proto");

/** Safety decision from AgentBus polling. */
export interface SafetyDecision {
  approved: boolean;
  reason: string;
  intentionId: number;
}

/**
 * AgentBus gRPC client.
 *
 * Wraps the Propose/Poll RPCs with high-level methods matching the
 * Python AgentBusClient interface.
 */
export class AgentBusClient {
  private client: any;
  private busId: string;
  private connected = false;

  constructor(busId: string) {
    this.busId = busId;
  }

  /**
   * Connect to the AgentBus gRPC server.
   *
   * @param address - gRPC address (e.g. "127.0.0.1:9999")
   */
  async connect(address: string): Promise<void> {
    const packageDef = protoLoader.loadSync(PROTO_PATH, {
      keepCase: true,
      longs: Number,
      enums: Number,
      defaults: true,
      oneofs: true,
    });

    const proto = grpc.loadPackageDefinition(packageDef) as any;
    const AgentBusService = proto.agent_bus.AgentBusService;

    this.client = new AgentBusService(
      address,
      grpc.credentials.createInsecure(),
    );

    // Wait for the channel to be ready
    await new Promise<void>((resolve, reject) => {
      const deadline = new Date(Date.now() + 10_000);
      this.client.waitForReady(deadline, (err: Error | null) => {
        if (err) reject(new Error(`AgentBus connection failed: ${err.message}`));
        else resolve();
      });
    });
    this.connected = true;
    console.error(`[agentbus] Connected to AgentBus at ${address}`);
  }

  /**
   * Propose a payload to the AgentBus log.
   * Returns the log position where it was stored.
   */
  private async propose(payload: any): Promise<number> {
    if (!this.connected) throw new Error("AgentBus not connected");

    return new Promise((resolve, reject) => {
      this.client.Propose(
        { agent_bus_id: this.busId, payload },
        (err: grpc.ServiceError | null, response: any) => {
          if (err) reject(new Error(`Propose failed: ${err.message}`));
          else resolve(response.log_position);
        },
      );
    });
  }

  /**
   * Poll for entries from the AgentBus log.
   */
  private async poll(
    startPosition: number,
    maxEntries: number,
    payloadTypes?: number[],
  ): Promise<{ entries: any[]; complete: boolean }> {
    if (!this.connected) throw new Error("AgentBus not connected");

    const request: any = {
      agent_bus_id: this.busId,
      start_log_position: startPosition,
      max_entries: maxEntries,
    };
    if (payloadTypes) {
      request.filter = { payload_types: payloadTypes };
    }

    return new Promise((resolve, reject) => {
      this.client.Poll(
        request,
        (err: grpc.ServiceError | null, response: any) => {
          if (err) reject(new Error(`Poll failed: ${err.message}`));
          else resolve({ entries: response.entries || [], complete: response.complete });
        },
      );
    });
  }

  /**
   * Log an intention (tool call) to AgentBus.
   * Returns the log position (intention ID).
   */
  async logIntention(code: string): Promise<number> {
    const payload = {
      intention: { string_intention: code },
    };
    return this.propose(payload);
  }

  /**
   * Poll for a Commit or Abort decision on a previously proposed intention.
   *
   * @param intentionId - Log position of the intention
   * @param timeoutMs - Maximum time to wait (default: 30s)
   */
  async waitForDecision(
    intentionId: number,
    timeoutMs = 30_000,
  ): Promise<SafetyDecision> {
    // SelectivePollType: COMMIT=4, ABORT=5
    const COMMIT = 4;
    const ABORT = 5;
    const POLL_INTERVAL = 1000;
    const maxAttempts = Math.ceil(timeoutMs / POLL_INTERVAL);

    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      const result = await this.poll(intentionId + 1, 1000, [COMMIT, ABORT]);

      for (const entry of result.entries) {
        const payload = entry.payload;
        if (!payload) continue;

        if (payload.commit && payload.commit.intention_id === intentionId) {
          return {
            approved: true,
            reason: payload.commit.reason || "approved",
            intentionId,
          };
        }
        if (payload.abort && payload.abort.intention_id === intentionId) {
          return {
            approved: false,
            reason: payload.abort.reason || "rejected",
            intentionId,
          };
        }
      }

      await new Promise((r) => setTimeout(r, POLL_INTERVAL));
    }

    return {
      approved: false,
      reason: "AgentBus safety check timed out",
      intentionId,
    };
  }

  /**
   * Log the output/result of a tool execution.
   */
  async logActionOutput(intentionId: number, result: string): Promise<void> {
    const payload = {
      action_output: {
        intention_id: intentionId,
        string_action_output: result,
      },
    };
    try {
      await this.propose(payload);
    } catch (e) {
      console.error(`[agentbus] Failed to log ActionOutput: ${e}`);
    }
  }

  /**
   * Log LLM inference input for observability.
   */
  async logInferenceInput(input: string): Promise<void> {
    const payload = {
      inference_input: { string_inference_input: input },
    };
    try {
      await this.propose(payload);
    } catch (e) {
      console.error(`[agentbus] Failed to log InferenceInput: ${e}`);
    }
  }

  /**
   * Log LLM inference output for observability.
   */
  async logInferenceOutput(output: string): Promise<void> {
    const payload = {
      inference_output: { string_inference_output: output },
    };
    try {
      await this.propose(payload);
    } catch (e) {
      console.error(`[agentbus] Failed to log InferenceOutput: ${e}`);
    }
  }

  /**
   * Set the decider policy for this agent's bus.
   */
  async setDeciderPolicy(policy: number): Promise<void> {
    const payload = { decider_policy: policy };
    await this.propose(payload);
  }

  /**
   * Poll all entries from the bus (for inspection / demo).
   */
  async pollAll(start = 0, maxEntries = 10000): Promise<any[]> {
    const result = await this.poll(start, maxEntries);
    return result.entries;
  }

  /** Close the gRPC channel. */
  close(): void {
    if (this.client) {
      this.client.close();
      this.connected = false;
    }
  }
}

/**
 * Parse an agentbus URL into a gRPC address.
 *
 * Supported schemes:
 *   memory://<port>  → 127.0.0.1:<port>
 *   remote://<host>:<port> → <host>:<port>
 */
export function parseAgentBusUrl(url: string): string {
  if (url.startsWith("memory://")) {
    const port = url.slice("memory://".length);
    return `127.0.0.1:${port}`;
  }
  if (url.startsWith("remote://")) {
    return url.slice("remote://".length);
  }
  // Fallback: assume it's a host:port
  return url;
}
