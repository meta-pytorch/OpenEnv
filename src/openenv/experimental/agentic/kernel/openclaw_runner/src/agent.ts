// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

/**
 * OpenClaw agent — direct LLM API integration for AgentKernel.
 *
 * Calls an OpenAI-compatible LLM API directly (e.g. OpenAI, Anthropic)
 * and supports tool execution (bash) with AgentBus safety gating.
 *
 * This avoids depending on the openclaw CLI binary (which has heavy native
 * deps like node-llama-cpp that are hard to install on devservers) and
 * instead implements a minimal agentic loop:
 *   1. Send messages to LLM API
 *   2. If LLM returns tool_calls, execute them (with AgentBus safety check)
 *   3. Append tool results, loop back to step 1
 *   4. When LLM returns text without tool_calls, that's the final response
 */

import { execSync } from "node:child_process";
import type { OpenClawRunnerConfig } from "./config.js";
import type { AgentBusSafetyExecutor } from "./agentbus-plugin.js";

/** A single conversation history entry. */
export interface HistoryEntry {
  role: "user" | "assistant" | "system";
  content: string;
  timestamp?: string;
}

interface LLMMessage {
  role: string;
  content: string | null;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
}

interface ToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}

const TOOLS_SCHEMA = [
  {
    type: "function" as const,
    function: {
      name: "bash",
      description: "Execute a bash command and return stdout/stderr.",
      parameters: {
        type: "object",
        properties: {
          command: { type: "string", description: "The bash command to run" },
        },
        required: ["command"],
      },
    },
  },
];

/**
 * OpenClaw agent managed by AgentKernel.
 *
 * Implements a minimal agentic loop with LLM API calls and tool execution.
 */
export class OpenClawAgent {
  private config: OpenClawRunnerConfig;
  private history: HistoryEntry[] = [];
  busClient: AgentBusSafetyExecutor | null = null;
  safetyEnabled = false;

  constructor(config: OpenClawRunnerConfig) {
    this.config = config;
  }

  /**
   * Run a single agent turn.
   *
   * Calls the LLM API, executes tool calls in a loop, and yields SSE chunks.
   */
  async *turn(
    messages: Array<{ role: string; content: string }>,
  ): AsyncGenerator<{ body: string; done: boolean; error?: string }> {
    // Add incoming messages to history
    for (const msg of messages) {
      this.history.push({
        role: msg.role as HistoryEntry["role"],
        content: msg.content,
        timestamp: new Date().toISOString(),
      });
    }

    // Extract the latest user message
    const userMessage = messages.filter((m) => m.role === "user").pop();
    if (!userMessage) {
      yield { body: "", done: true, error: "No user message provided" };
      return;
    }

    try {
      // Build LLM messages from history
      const llmMessages: LLMMessage[] = [];

      // System prompt
      if (this.config.system_prompt) {
        llmMessages.push({
          role: "system",
          content: this.config.system_prompt,
        });
      }

      // Conversation history
      for (const entry of this.history) {
        llmMessages.push({ role: entry.role, content: entry.content });
      }

      // Log inference input
      if (this.busClient) {
        await this.busClient.logInferenceInput(
          JSON.stringify(llmMessages.slice(-3)),
        );
      }

      // Agentic loop: call LLM → execute tools → repeat
      const MAX_ITERATIONS = 10;
      let responseText = "";

      for (let i = 0; i < MAX_ITERATIONS; i++) {
        const llmResponse = await this.callLLM(llmMessages);

        // Log inference output
        if (this.busClient) {
          await this.busClient.logInferenceOutput(
            JSON.stringify(llmResponse),
          );
        }

        // No tool calls — this is the final text response
        if (!llmResponse.tool_calls || llmResponse.tool_calls.length === 0) {
          responseText = llmResponse.content || "";
          break;
        }

        // Has tool calls — execute them
        llmMessages.push({
          role: "assistant",
          content: llmResponse.content,
          tool_calls: llmResponse.tool_calls,
        });

        for (const toolCall of llmResponse.tool_calls) {
          const { name, arguments: argsStr } = toolCall.function;
          let args: Record<string, unknown>;
          try {
            args = JSON.parse(argsStr);
          } catch {
            args = { command: argsStr };
          }

          // AgentBus safety check
          let intentionId: number | null = null;
          if (this.busClient) {
            const check = await this.busClient.checkBeforeToolCall(name, args);
            intentionId = check.intentionId;
            if (!check.allowed) {
              const blockMsg = check.reason || "Blocked by safety policy";
              console.error(`[agent] Tool call blocked: ${blockMsg}`);

              // Stream the block notification
              yield { body: `\n[Safety: ${blockMsg}]\n`, done: false };

              llmMessages.push({
                role: "tool",
                tool_call_id: toolCall.id,
                content: `BLOCKED: ${blockMsg}`,
              });

              if (this.busClient) {
                await this.busClient.logAfterToolCall(
                  intentionId,
                  blockMsg,
                  true,
                );
              }
              continue;
            }
          }

          // Execute the tool
          let result: string;
          let isError = false;
          if (name === "bash") {
            const cmd = (args as { command: string }).command;
            console.error(`[agent] Executing bash: ${cmd.slice(0, 100)}`);
            yield { body: `\n[Running: ${cmd.slice(0, 80)}]\n`, done: false };

            try {
              result = execSync(cmd, {
                encoding: "utf-8",
                timeout: 30_000,
                maxBuffer: 1024 * 1024,
                cwd: process.cwd(),
              });
            } catch (err: any) {
              result = err.stderr || err.stdout || err.message;
              isError = true;
            }
          } else {
            result = `Unknown tool: ${name}`;
            isError = true;
          }

          // Truncate long results
          const maxLen = 10_000;
          const truncated =
            result.length > maxLen
              ? result.slice(0, maxLen) + "\n... [truncated]"
              : result;

          // Log tool result to AgentBus
          if (this.busClient) {
            await this.busClient.logAfterToolCall(
              intentionId,
              truncated,
              isError,
            );
          }

          llmMessages.push({
            role: "tool",
            tool_call_id: toolCall.id,
            content: truncated,
          });
        }

        // If this is the last iteration, force a text response
        if (i === MAX_ITERATIONS - 1) {
          responseText = "[Agent reached maximum iterations]";
        }
      }

      // Add assistant response to history
      this.history.push({
        role: "assistant",
        content: responseText,
        timestamp: new Date().toISOString(),
      });

      yield { body: responseText, done: false };
      yield { body: "", done: true };
    } catch (err: any) {
      const errorMsg = err?.message || String(err);
      console.error(`[agent] Turn error: ${errorMsg}`);

      this.history.push({
        role: "assistant",
        content: `[Error: ${errorMsg}]`,
        timestamp: new Date().toISOString(),
      });

      yield { body: "", done: true, error: errorMsg };
    }
  }

  /**
   * Call the LLM API (OpenAI-compatible).
   */
  private async callLLM(
    messages: LLMMessage[],
  ): Promise<{ content: string | null; tool_calls?: ToolCall[] }> {
    const baseUrl =
      this.config.base_url || "https://api.openai.com/v1";
    const apiKey =
      this.config.api_key || process.env.LLM_API_KEY || "";
    const model = this.config.model || "claude-sonnet-4-5";

    const hasTools = (this.config.tools ?? []).length > 0;

    const body: Record<string, unknown> = {
      model,
      messages,
      max_tokens: 4096,
    };

    if (hasTools) {
      body.tools = TOOLS_SCHEMA.filter((t) =>
        this.config.tools!.includes(t.function.name),
      );
    }

    console.error(
      `[agent] Calling LLM: ${baseUrl}/chat/completions (model=${model}, msgs=${messages.length})`,
    );

    const response = await fetch(`${baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(
        `LLM API error ${response.status}: ${errText.slice(0, 500)}`,
      );
    }

    const data = (await response.json()) as {
      choices: Array<{
        message: {
          content: string | null;
          tool_calls?: ToolCall[];
        };
      }>;
    };

    const choice = data.choices?.[0];
    if (!choice) {
      throw new Error("LLM API returned no choices");
    }

    return {
      content: choice.message.content,
      tool_calls: choice.message.tool_calls,
    };
  }

  /**
   * Get the full conversation history.
   */
  getHistory(): HistoryEntry[] {
    return [...this.history];
  }
}
