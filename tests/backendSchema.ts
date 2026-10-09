/**
 * Local mirror of the backend's agent-loop request schema
 * (aitl_fastapi/src/schemas/agent_loop/), ported from tests/backend_schema.py.
 * Lets tests catch a payload that the real server would reject with HTTP
 * 400/422, without needing the server.
 *
 * Keep in sync with the backend if its schema changes.
 */
import { z } from "zod";

const base = {
  graph_order: z.number().int(),
  node_name: z.string().nullable(),
  node_type: z.string().nullable(),
  run_id: z.string(),
  parent_run_id: z.string().nullable(),
  trace_id: z.string(),
  langgraph_step: z.number().int().nullable().optional(),
};

const llmInput = z.object({
  system: z.string().nullable().optional(),
  human: z.string().nullable().optional(),
});
const llmOutput = z.object({ response: z.string().nullable().optional() });

const onChainStart = z.object({ ...base, event: z.literal("on_chain_start"), inputs: z.array(z.any()) });
const onChainEnd = z.object({ ...base, event: z.literal("on_chain_end"), outputs: z.any().refine((v) => v !== undefined) });
const onChatModelStart = z.object({
  ...base,
  event: z.literal("on_chat_model_start"),
  input: llmInput,
  model: z.string(),
  tools_hash: z.string().nullable().optional(),
});
const onLlmStart = z.object({
  ...base,
  event: z.literal("on_llm_start"),
  input: llmInput,
  model: z.string().nullable().optional(),
});
const onLlmEnd = z.object({
  ...base,
  event: z.literal("on_llm_end"),
  output: llmOutput,
  token_usage: z.any().optional(),
});
const onToolStart = z.object({
  ...base,
  event: z.literal("on_tool_start"),
  input: z.record(z.string(), z.any()),
  tool: z.string().nullable(),
  tool_description: z.string().nullable(),
  invoked_by_run_id: z.string().nullable(),
});
const onToolEnd = z.object({
  ...base,
  event: z.literal("on_tool_end"),
  output: z.any(),
  is_mcp_tool: z.boolean().default(false),
});
const onError = z.object({
  ...base,
  event: z.enum(["on_llm_error", "on_tool_error", "on_chain_error"]),
  error: z.string(),
});

export const agentEvent = z.discriminatedUnion("event", [
  onChainStart,
  onChainEnd,
  onChatModelStart,
  onLlmStart,
  onLlmEnd,
  onToolStart,
  onToolEnd,
  onError,
]);

const toolSchema = z.object({
  type: z.literal("function").default("function"),
  function: z.object({
    name: z.string(),
    description: z.string().nullable().optional(),
    parameters: z.record(z.string(), z.any()).default({}),
  }),
});

export const agentLoopRequest = z.object({
  context: z.array(agentEvent),
  trace_id: z.string(),
  agent_name: z.string().nullable().optional(),
  observability_call: z.boolean().default(false),
  single_call: z.boolean().optional(),
  available_tools: z
    .array(z.object({ tools_hash: z.string(), tools: z.array(toolSchema) }))
    .default([]),
});

/** Throws with a readable message when ``payload`` would be rejected by the backend. */
export function assertValidAgentLoopRequest(payload: unknown): void {
  const parsed = agentLoopRequest.safeParse(payload);
  if (!parsed.success) {
    throw new Error("payload rejected by backend schema:\n" + z.prettifyError(parsed.error));
  }
}
