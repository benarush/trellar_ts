/**
 * Shared test builders for LangChain-object fakes (counterpart of tests/factories.py).
 */
import { AIMessage } from "@langchain/core/messages";
import type { ChatGeneration, Generation, LLMResult } from "@langchain/core/outputs";
import { randomUUID } from "node:crypto";

export const uuid = (): string => randomUUID();

export function makeAiMessage(content: any = "", toolCalls: any[] = []): AIMessage {
  return new AIMessage({ content, tool_calls: toolCalls });
}

export function makeChatGeneration(content: any = "", toolCalls: any[] = []): ChatGeneration {
  const message = makeAiMessage(content, toolCalls);
  return { message, text: typeof content === "string" ? content : "" };
}

export function makeLlmResult(options: {
  message?: AIMessage;
  text?: string;
  tokenUsage?: Record<string, unknown>;
  generation?: unknown;
  llmOutput?: Record<string, unknown>;
}): LLMResult {
  let gen: unknown;
  if (options.generation !== undefined) gen = options.generation;
  else if (options.message !== undefined) {
    gen = {
      message: options.message,
      text: typeof options.message.content === "string" ? options.message.content : "",
    };
  } else gen = { text: options.text ?? "" };

  const llmOutput = options.llmOutput ?? (options.tokenUsage ? { token_usage: options.tokenUsage } : undefined);
  return { generations: [[gen as Generation]], llmOutput } as LLMResult;
}

/** Duck-typed stand-in for a Generation (bypasses validation for edge cases). */
export function fakeGeneration(parts: { message?: unknown; text?: unknown }): unknown {
  return { message: parts.message, text: parts.text };
}

/** Duck-typed stand-in for an LLMResult. */
export function fakeResponse(generations: unknown[][], llmOutput?: Record<string, unknown>): any {
  return { generations, llmOutput };
}

/** Script a minimal root-chain -> node run on a handler, returning the ids used. */
export function startRoot(handler: any, name = "LangGraph"): { root: string } {
  const root = uuid();
  handler.handleChainStart({ name }, { messages: [] }, root, undefined, [], {}, undefined, name);
  return { root };
}
