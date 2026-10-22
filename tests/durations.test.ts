/**
 * Tests for ``duration_seconds`` on model/tool end events (LangChain + Strands).
 * LangChain tests drive the callback directly with a fake clock so durations are
 * exact; the Strands test runs a real Agent with a scripted model.
 */
import { HumanMessage } from "@langchain/core/messages";
import { Agent, tool } from "@strands-agents/sdk";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { z } from "zod";

import { ObservabilityMode, evaluateConfidence } from "../src/agentLoop.js";
import { LangchainAgentCallback } from "../src/callbacks/langchain/langchainCallback.js";
import { LangchainSingleCallCallback } from "../src/callbacks/langchain/singleLangchainCallback.js";
import { trellarStrandsAgent } from "../src/strands.js";
import { releaseTrellarAgent, runWithTrellarAgent } from "../src/context.js";
import { assertValidAgentLoopRequest, mockFetch, useCleanEnv } from "./helpers.js";
import { makeAiMessage, makeLlmResult, uuid } from "./factories.js";
import { FakeModel } from "./strandsFactories.js";

let now = 0;
beforeEach(() => {
  now = 1000;
  vi.spyOn(performance, "now").mockImplementation(() => now);
});
afterEach(() => vi.restoreAllMocks());

const advance = (seconds: number): void => {
  now += seconds * 1000;
};
const find = (h: { events: Array<Record<string, unknown>> }, event: string, runId: string) =>
  h.events.find((e) => e["event"] === event && e["run_id"] === runId)!;

/** A handler with a started root chain, so per-run state is initialised. */
function makeHandler(): LangchainAgentCallback {
  const h = new LangchainAgentCallback({ agentName: "test-agent" });
  h.handleChainStart({ name: "graph" }, { messages: [] }, uuid(), undefined, [], {}, undefined, "graph");
  return h;
}
const startModel = (h: LangchainAgentCallback, id: string) =>
  h.handleChatModelStart({ name: "gpt" }, [[new HumanMessage("hi")]], id, undefined);
const startTool = (h: LangchainAgentCallback, id: string) =>
  h.handleToolStart({ name: "search" }, "{}", id, undefined, [], {}, "search");

describe("duration_seconds", () => {
  it("langchain: tool duration", () => {
    const h = makeHandler();
    const id = uuid();
    startTool(h, id);
    advance(2.5);
    h.handleToolEnd("result", id, undefined);

    expect(find(h, "on_tool_start", id)).not.toHaveProperty("duration_seconds");
    expect(find(h, "on_tool_end", id)["duration_seconds"]).toBe(2.5);
    expect(h._startTimes.size).toBe(0);
  });

  it("langchain: model duration on success and error", () => {
    const h = makeHandler();
    const okId = uuid();
    const errId = uuid();

    startModel(h, okId);
    advance(1.25);
    h.handleLLMEnd(makeLlmResult({ message: makeAiMessage("hello") }), okId, undefined);

    startModel(h, errId);
    advance(0.5);
    h.handleLLMError(new Error("boom"), errId, undefined);

    expect(find(h, "on_llm_end", okId)["duration_seconds"]).toBe(1.25);
    expect(find(h, "on_llm_error", errId)["duration_seconds"]).toBe(0.5);
  });

  it("langchain: overlapping calls get independent durations", () => {
    const h = makeHandler();
    const toolA = uuid();
    const toolB = uuid();
    const model = uuid();

    startTool(h, toolA);
    advance(1);
    startTool(h, toolB); // same tool, starts while A is still running
    startModel(h, model);
    advance(2);
    h.handleToolEnd("b", toolB, undefined); // B: 2s
    advance(3);
    h.handleLLMEnd(makeLlmResult({ message: makeAiMessage("ok") }), model, undefined); // model: 5s
    advance(4);
    h.handleToolEnd("a", toolA, undefined); // A: 10s

    expect(find(h, "on_tool_end", toolB)["duration_seconds"]).toBe(2);
    expect(find(h, "on_llm_end", model)["duration_seconds"]).toBe(5);
    expect(find(h, "on_tool_end", toolA)["duration_seconds"]).toBe(10);
    expect(h._startTimes.size).toBe(0);
  });

  it("strands: model and tool durations", async () => {
    const add = tool({
      name: "add",
      description: "Add two numbers.",
      inputSchema: z.object({ a: z.number(), b: z.number() }),
      callback: ({ a, b }) => a + b,
    });
    const trellarAgent = trellarStrandsAgent("t");
    const agent = new Agent({
      id: "calc-durations",
      name: "calc",
      model: new FakeModel([{ toolUse: { name: "add", input: { a: 1, b: 2 } } }, { text: "3" }]),
      tools: [add],
      plugins: [trellarAgent],
      printer: false,
    });
    await agent.invoke("1+2?");

    const of = (kind: string) => trellarAgent.events.filter((e) => e["event"] === kind);
    for (const kind of ["on_llm_end", "on_tool_end"]) {
      expect(of(kind).length).toBeGreaterThan(0);
      for (const e of of(kind)) expect(e["duration_seconds"]).toBeGreaterThanOrEqual(0);
    }
    for (const kind of ["on_chat_model_start", "on_tool_start"]) {
      for (const e of of(kind)) expect(e).not.toHaveProperty("duration_seconds");
    }
    expect(trellarAgent._startTimes.size).toBe(0);
  });
});

describe("total_duration_seconds (top level of the payload, not in the events)", () => {
  useCleanEnv();
  const chainStart = (h: LangchainAgentCallback, id: string, parent?: string) =>
    h.handleChainStart({ name: "n" }, { messages: [] }, id, parent, [], {}, undefined, "n");
  const noEventHasTotal = (context: Array<Record<string, unknown>>) =>
    context.every((e) => !("total_duration_seconds" in e));

  it("langchain graph: payload total is the full run when auto-evaluated at root end", async () => {
    const calls = mockFetch();
    const h = new LangchainAgentCallback({ agentName: "test-agent", observabilityMode: ObservabilityMode.ALWAYS });
    const [root, node, model, toolId] = [uuid(), uuid(), uuid(), uuid()];

    chainStart(h, root);
    advance(1);
    chainStart(h, node, root);
    h.handleChatModelStart({ name: "gpt" }, [[new HumanMessage("hi")]], model, node);
    advance(2);
    h.handleLLMEnd(makeLlmResult({ message: makeAiMessage("ok") }), model, node);
    h.handleToolStart({ name: "search" }, "{}", toolId, node, [], {}, "search");
    advance(3);
    h.handleToolEnd("out", toolId, node);
    await h.handleChainEnd({ messages: [] }, node, root);
    advance(4);
    await h.handleChainEnd({ messages: [] }, root, undefined);

    expect(calls).toHaveLength(1);
    expect(calls[0]!.body.total_duration_seconds).toBe(10);
    expect(noEventHasTotal(calls[0]!.body.context)).toBe(true);
    assertValidAgentLoopRequest(calls[0]!.body);
  });

  it("langchain graph: payload total is elapsed-so-far for a manual mid-run evaluate", async () => {
    const calls = mockFetch();
    const h = new LangchainAgentCallback({ agentName: "test-agent" });
    chainStart(h, uuid());
    advance(3);
    try {
      await runWithTrellarAgent(h, () => evaluateConfidence({ apiKey: "k" })); // earlier tests leave roots open
    } finally {
      releaseTrellarAgent(h);
    }

    expect(calls[0]!.body.total_duration_seconds).toBe(3);
  });

  it("single LLM call: payload total equals the call duration", async () => {
    const calls = mockFetch();
    const h = new LangchainSingleCallCallback({ agentName: "test-agent", observabilityMode: ObservabilityMode.ALWAYS });
    const id = uuid();
    h.handleChatModelStart({ name: "gpt" }, [[new HumanMessage("hi")]], id, undefined);
    advance(1.5);
    await h.handleLLMEnd(makeLlmResult({ message: makeAiMessage("ok") }), id, undefined);

    const body = calls[0]!.body;
    expect(body.total_duration_seconds).toBe(1.5);
    expect(body.context.at(-1).duration_seconds).toBe(1.5);
  });

  it("strands: payload total covers the agent run", async () => {
    const calls = mockFetch();
    const add = tool({
      name: "add",
      description: "Add two numbers.",
      inputSchema: z.object({ a: z.number(), b: z.number() }),
      callback: ({ a, b }) => a + b,
    });
    const trellarAgent = trellarStrandsAgent("t", ObservabilityMode.ALWAYS);
    await new Agent({
      id: "calc-total",
      name: "calc",
      model: new FakeModel([{ toolUse: { name: "add", input: { a: 1, b: 2 } } }, { text: "3" }]),
      tools: [add],
      plugins: [trellarAgent],
      printer: false,
    }).invoke("1+2?");

    expect(calls).toHaveLength(1);
    const { body } = calls[0]!;
    const stepDurations = body.context
      .map((e: Record<string, unknown>) => e["duration_seconds"])
      .filter((d: unknown): d is number => typeof d === "number");
    expect(body.total_duration_seconds).toBeGreaterThanOrEqual(Math.max(...stepDurations));
    expect(noEventHasTotal(body.context)).toBe(true);
  });
});
