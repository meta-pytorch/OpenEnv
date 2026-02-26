// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

/**
 * AgentBus safety integration for the OpenClaw runner.
 *
 * Provides a standalone SafeToolExecutor that wraps tool calls with
 * AgentBus safety checks.
 *
 * Flow:
 *   1. Before tool execution: propose Intention → wait for Commit/Abort
 *   2. If approved (or safety disabled): allow execution
 *   3. If rejected: block with reason
 *   4. After execution: log ActionOutput
 */

import { AgentBusClient, parseAgentBusUrl } from "./agentbus-client.js";

export interface SafetyCheckResult {
  allowed: boolean;
  intentionId: number | null;
  reason?: string;
}

/**
 * AgentBus safety executor — gates tool calls through the AgentBus
 * propose/commit/abort flow.
 */
export class AgentBusSafetyExecutor {
  private client: AgentBusClient;
  private safetyEnabled: boolean;
  private connected = false;

  constructor(busId: string, safetyEnabled: boolean) {
    this.client = new AgentBusClient(busId);
    this.safetyEnabled = safetyEnabled;
  }

  async connect(agentbusUrl: string): Promise<void> {
    const address = parseAgentBusUrl(agentbusUrl);
    await this.client.connect(address);
    this.connected = true;
  }

  /**
   * Check safety for a tool call before execution.
   *
   * Proposes an Intention and waits for a Commit/Abort decision.
   * Returns whether execution is allowed.
   */
  async checkBeforeToolCall(
    toolName: string,
    params: unknown,
  ): Promise<SafetyCheckResult> {
    if (!this.connected) {
      return { allowed: true, intentionId: null };
    }

    const intentionStr = `${toolName}: ${JSON.stringify(params)}`;

    try {
      const intentionId = await this.client.logIntention(intentionStr);

      if (this.safetyEnabled) {
        const decision = await this.client.waitForDecision(intentionId);
        if (!decision.approved) {
          return {
            allowed: false,
            intentionId,
            reason: `AgentBus safety: ${decision.reason}`,
          };
        }
      }

      return { allowed: true, intentionId };
    } catch (err) {
      // AgentBus errors are non-fatal — log and allow
      console.error(`[agentbus] before_tool_call error: ${err}`);
      return { allowed: true, intentionId: null };
    }
  }

  /**
   * Log the result of a tool execution to AgentBus.
   */
  async logAfterToolCall(
    intentionId: number | null,
    result: string,
    isError = false,
  ): Promise<void> {
    if (!this.connected || intentionId === null) return;

    const text = isError ? `ERROR: ${result}` : result;
    const truncated =
      text.length > 10_000
        ? text.slice(0, 10_000) + "... [truncated]"
        : text;

    await this.client.logActionOutput(intentionId, truncated);
  }

  /**
   * Log LLM inference input for observability.
   */
  async logInferenceInput(input: string): Promise<void> {
    if (!this.connected) return;
    await this.client.logInferenceInput(input);
  }

  /**
   * Log LLM inference output for observability.
   */
  async logInferenceOutput(output: string): Promise<void> {
    if (!this.connected) return;
    await this.client.logInferenceOutput(output);
  }

  /** Close the gRPC connection. */
  close(): void {
    this.client.close();
    this.connected = false;
  }
}
