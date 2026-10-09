import { AIMessage, HumanMessage, SystemMessage, ToolMessage } from "@langchain/core/messages";
import { afterEach, describe, expect, it, vi } from "vitest";

import { trellarLangchainAgent, ObservabilityMode } from "../src/langchain.js";
import { LangchainAgentCallback } from "../src/callbacks/langchain/langchainCallback.js";
import { isTrellarAgentActive } from "../src/context.js";
import * as agentLoop from "../src/agentLoop.js";
import {
  fakeGeneration,
  fakeResponse,
  makeAiMessage,
  makeLlmResult,
  uuid,
} from "./factories.js";
import { assertValidAgentLoopRequest } from "./helpers.js";

const mkTrellarAgent = (mode: ObservabilityMode = ObservabilityMode.NONE) => trellarLangchainAgent("test-agent", mode);

/** Start a root chain on ``h`` and return its run id. */
function root(h: LangchainAgentCallback, inputs: unknown = { messages: [] }, name = "LangGraph"): string {
  const id = uuid();
  h.handleChainStart({ name }, inputs, id, undefined, [], {}, undefined, name);
  return id;
}

afterEach(() => vi.restoreAllMocks());

describe("trellarLangchainAgent", () => {
  it("returns an LangchainAgentCallback bound to the agent name", () => {
    const g = trellarLangchainAgent("research-agent");
    expect(g).toBeInstanceOf(LangchainAgentCallback);
    expect(g.agentName).toBe("research-agent");
  });
  it("rejects empty and blank names with a descriptive error", () => {
    expect(() => trellarLangchainAgent("")).toThrow(/agent_name is required/);
    expect(() => trellarLangchainAgent("   ")).toThrow(/stable, descriptive name/);
  });
  it("returns a new instance every call", () => {
    expect(trellarLangchainAgent("a")).not.toBe(trellarLangchainAgent("a"));
  });
  it("rejects an invalid observability mode", () => {
    expect(() => trellarLangchainAgent("a", "sometimes" as any)).toThrow(/not a valid ObservabilityMode/);
  });
});

describe("handleChainStart / handleChainEnd", () => {
  it("normalizes inputs to a list for every input shape", () => {
    const h = mkTrellarAgent();
    const r = root(h, [new HumanMessage("hi")]);
    h.handleChainStart({}, new AIMessage("x"), uuid(), r, [], {}, undefined, "a");
    h.handleChainStart({}, { foo: 1 }, uuid(), r, [], {}, undefined, "b");
    h.handleChainStart({}, "bare", uuid(), r, [], {}, undefined, "c");
    for (const e of h.events) expect(Array.isArray(e["inputs"])).toBe(true);
    expect(h.events[0]!["inputs"]).toEqual(["HUMAN MESSAGE: hi"]);
    expect(h.events[2]!["inputs"]).toEqual([{ foo: 1 }]);
  });

  it("dict inputs with messages use labeled strings", () => {
    const h = mkTrellarAgent();
    root(h, { messages: [new HumanMessage("hello"), new SystemMessage("sys")] });
    expect(h.events[0]!["inputs"]).toEqual(["HUMAN MESSAGE: hello", "SYSTEM MESSAGE: sys"]);
  });

  it("serializes non-dict outputs and dict outputs with messages", () => {
    const h = mkTrellarAgent();
    const r = root(h);
    const child = uuid();
    h.handleChainStart({}, {}, child, r, [], {}, undefined, "n");
    h.handleChainEnd({ answer: 1, messages: [new AIMessage("hi")] }, child, r);
    h.handleChainEnd("plain", r);
    expect(h.events.at(-2)!["outputs"]).toEqual({ answer: 1, messages: ["AI MESSAGE: hi"] });
    expect(h.events.at(-1)!["outputs"]).toBe("plain");
  });

  it("falls back from runName to chain.name to the class id", () => {
    const h = mkTrellarAgent();
    const r = root(h);
    h.handleChainStart({ name: "FromSerialized" }, {}, uuid(), r, [], {});
    h.handleChainStart({ id: ["langchain", "RunnableSequence"] }, {}, uuid(), r, [], {});
    expect(h.events[1]!["node_name"]).toBe("FromSerialized");
    expect(h.events[2]!["node_name"]).toBe("RunnableSequence");
  });

  it("surfaces langgraph_step from metadata, else null", () => {
    const h = mkTrellarAgent();
    const r = root(h);
    h.handleChainStart({}, {}, uuid(), r, [], { langgraph_step: 3 }, undefined, "a");
    h.handleChainStart({}, {}, uuid(), r, [], {}, undefined, "b");
    h.handleChainStart({}, {}, uuid(), r, [], undefined, undefined, "c");
    expect(h.events.map((e) => e["langgraph_step"])).toEqual([null, 3, null, null]);
  });

  it("root start sets trace id and activates the Trellar agent", () => {
    const h = mkTrellarAgent();
    const r = root(h);
    expect(h.traceId).toBe(r);
    expect(isTrellarAgentActive(h)).toBe(true);
  });

  it("non-root start leaves the trace id alone", () => {
    const h = mkTrellarAgent();
    const r = root(h);
    h.handleChainStart({}, {}, uuid(), r, [], {}, undefined, "n");
    expect(h.traceId).toBe(r);
  });

  it("reusing a Trellar agent across two root runs does not leak events", () => {
    const h = mkTrellarAgent();
    root(h);
    h.handleToolStart({ name: "t" }, "{}", uuid(), undefined, [], {}, "t");
    const firstCount = h.events.length;
    expect(firstCount).toBeGreaterThan(1);
    root(h);
    expect(h.events).toHaveLength(1);
    expect(h.events[0]!["graph_order"]).toBe(1);
    expect(h.availableTools).toEqual({});
    expect(h._pendingLlmToolCalls).toEqual([]);
    expect(h._runRegistry.size).toBe(1);
  });

  it("releases the slot when the root chain ends or errors, but not for child runs", async () => {
    const h = mkTrellarAgent();
    const r = root(h);
    const child = uuid();
    h.handleChainStart({}, {}, child, r, [], {}, undefined, "n");
    await h.handleChainEnd({}, child, r);
    expect(isTrellarAgentActive(h)).toBe(true);
    h.handleChainError(new Error("x"), child, r);
    expect(isTrellarAgentActive(h)).toBe(true);
    await h.handleChainEnd({}, r);
    expect(isTrellarAgentActive(h)).toBe(false);

    const r2 = root(h);
    h.handleChainError(new Error("boom"), r2);
    expect(isTrellarAgentActive(h)).toBe(false);
    expect(h.events.at(-1)).toMatchObject({ event: "on_chain_error", error: "boom" });
  });

  it("does not release a different, still-active Trellar agent", async () => {
    const a = mkTrellarAgent();
    const b = mkTrellarAgent();
    const ra = root(a);
    root(b);
    await a.handleChainEnd({}, ra);
    expect(isTrellarAgentActive(a)).toBe(false);
    expect(isTrellarAgentActive(b)).toBe(true);
  });
});

describe("handleLLMStart / handleChatModelStart", () => {
  it("records model and joined prompts for a plain LLM", () => {
    const h = mkTrellarAgent();
    const r = root(h);
    const id = uuid();
    h.handleLLMStart({ kwargs: { model_name: "m" } }, ["p1", "p2"], id, r);
    expect(h.events.at(-1)).toMatchObject({
      event: "on_llm_start",
      model: "m",
      input: { system: null, human: "p1\np2" },
      node_type: "llm",
      node_name: "m",
    });
  });

  it("extracts system/human from the first batch only", () => {
    const h = mkTrellarAgent();
    const r = root(h);
    h.handleChatModelStart(
      { kwargs: { model: "gpt" } },
      [[new SystemMessage("S"), new HumanMessage("H1")], [new HumanMessage("H2")]],
      uuid(),
      r,
    );
    expect(h.events.at(-1)).toMatchObject({ input: { system: "S", human: "H1" }, model: "gpt" });
  });

  it("empty messages do not throw", () => {
    const h = mkTrellarAgent();
    const r = root(h);
    h.handleChatModelStart({ name: "X" }, [], uuid(), r);
    expect(h.events.at(-1)!["input"]).toEqual({ system: null, human: null });
  });

  const openAiTool = (name: string) => ({ type: "function", function: { name, description: `${name} d`, parameters: {} } });

  it("records bound tools keyed by content hash and stamps tools_hash", () => {
    const h = mkTrellarAgent();
    const r = root(h);
    h.handleChatModelStart({ name: "X" }, [[new HumanMessage("a")]], uuid(), r, {
      invocation_params: { tools: [openAiTool("a")] },
    });
    const hash = h.events.at(-1)!["tools_hash"] as string;
    expect(hash).toMatch(/^[0-9a-f]{16}$/);
    expect(Object.keys(h.availableTools)).toEqual([hash]);
    expect(h.availableTools[hash]).toEqual([openAiTool("a")]);
  });

  it("dedupes an identical toolset across calls", () => {
    const h = mkTrellarAgent();
    const r = root(h);
    for (let i = 0; i < 3; i++) {
      h.handleChatModelStart({ name: "X" }, [[new HumanMessage("a")]], uuid(), r, {
        invocation_params: { tools: [openAiTool("a")] },
      });
    }
    expect(Object.keys(h.availableTools)).toHaveLength(1);
  });

  it("falls back to options.tools and leaves no tools alone", () => {
    const h = mkTrellarAgent();
    const r = root(h);
    h.handleChatModelStart({ name: "X" }, [[new HumanMessage("a")]], uuid(), r, { options: { tools: [openAiTool("b")] } });
    expect(Object.keys(h.availableTools)).toHaveLength(1);
    const h2 = mkTrellarAgent();
    const r2 = root(h2);
    h2.handleChatModelStart({ name: "X" }, [[new HumanMessage("a")]], uuid(), r2);
    expect(h2.availableTools).toEqual({});
    expect(h2.events.at(-1)!["tools_hash"]).toBeNull();
  });
});

describe("handleLLMEnd", () => {
  const end = (response: any) => {
    const h = mkTrellarAgent();
    const r = root(h);
    const id = uuid();
    h.handleLLMEnd(response, id, r);
    return h;
  };

  it("reads ChatGeneration message content", () => {
    expect(end(makeLlmResult({ message: makeAiMessage("hello") })).events.at(-1)!["output"]).toEqual({
      response: "hello",
    });
  });
  it("falls back to Generation.text", () => {
    expect(end(makeLlmResult({ text: "plain" })).events.at(-1)!["output"]).toEqual({ response: "plain" });
  });
  it("falls back to .text when message content is null", () => {
    const resp = fakeResponse([[fakeGeneration({ message: { content: null, tool_calls: [] }, text: "from text" })]]);
    expect(end(resp).events.at(-1)!["output"]).toEqual({ response: "from text" });
  });
  it("json-encodes Gemini multimodal list content", () => {
    const resp = makeLlmResult({ message: makeAiMessage([{ type: "text", text: "part" }]) });
    expect(end(resp).events.at(-1)!["output"]).toEqual({ response: '[{"type": "text", "text": "part"}]' });
  });
  it("json-encodes a list-valued .text", () => {
    const resp = fakeResponse([[fakeGeneration({ text: [{ type: "text", text: "x" }] })]]);
    expect(end(resp).events.at(-1)!["output"]).toEqual({ response: '[{"type": "text", "text": "x"}]' });
  });
  it("handles plain-dict generations", () => {
    const resp = fakeResponse([[{ message: { content: "dict msg", tool_calls: [] } }]]);
    expect(end(resp).events.at(-1)!["output"]).toEqual({ response: "dict msg" });
  });
  it("tolerates empty generations", () => {
    expect(end(fakeResponse([])).events.at(-1)!["output"]).toEqual({ response: "" });
    expect(end(fakeResponse([[]])).events.at(-1)!["output"]).toEqual({ response: "" });
  });

  it("reads token usage from token_usage, tokenUsage, then usage; else null", () => {
    expect(end(makeLlmResult({ text: "x", tokenUsage: { total_tokens: 5 } })).events.at(-1)!["token_usage"]).toEqual({
      total_tokens: 5,
    });
    expect(end(makeLlmResult({ text: "x", llmOutput: { tokenUsage: { t: 1 } } })).events.at(-1)!["token_usage"]).toEqual({
      t: 1,
    });
    expect(end(makeLlmResult({ text: "x", llmOutput: { usage: { u: 2 } } })).events.at(-1)!["token_usage"]).toEqual({
      u: 2,
    });
    expect(end(makeLlmResult({ text: "x" })).events.at(-1)!["token_usage"]).toBeNull();
  });

  it("folds tool calls into the response text and registers a pending entry", () => {
    const resp = makeLlmResult({
      message: makeAiMessage("thinking", [
        { name: "buy", args: { symbols: "A,B" }, id: "1" },
        { name: "send", args: {}, id: "2" },
      ]),
    });
    const h = end(resp);
    expect(h.events.at(-1)!["output"]).toEqual({
      response: 'thinking\nTOOL CALL: buy(args={"symbols": "A,B"})\nTOOL CALL: send(args={})',
    });
    expect(h._pendingLlmToolCalls).toHaveLength(1);
    expect(h._pendingLlmToolCalls[0]!.remainingTools).toEqual(["buy", "send"]);
  });

  it("tool calls with empty text still record the call lines; no calls means no pending entry", () => {
    const h = end(makeLlmResult({ message: makeAiMessage("", [{ name: "t", args: {}, id: "1" }]) }));
    expect(h.events.at(-1)!["output"]).toEqual({ response: "TOOL CALL: t(args={})" });
    expect(end(makeLlmResult({ message: makeAiMessage("x") }))._pendingLlmToolCalls).toEqual([]);
  });
});

describe("tool events", () => {
  it("records the tool, description from bound tools, and parsed input", () => {
    const h = mkTrellarAgent();
    const r = root(h);
    h.handleChatModelStart({ name: "X" }, [[new HumanMessage("a")]], uuid(), r, {
      invocation_params: { tools: [{ type: "function", function: { name: "buy", description: "Buy it", parameters: {} } }] },
    });
    const id = uuid();
    h.handleToolStart({ name: "ignored" }, '{"symbols":"A"}', id, r, [], {}, "buy");
    expect(h.events.at(-1)).toMatchObject({
      event: "on_tool_start",
      tool: "buy",
      tool_description: "Buy it",
      node_type: "tool",
      node_name: "buy",
      input: { raw: '{"symbols":"A"}', parsed: { symbols: "A" } },
      invoked_by_run_id: r,
    });
  });

  it("non-JSON input has parsed null and a missing parent gives null invoked_by", () => {
    const h = mkTrellarAgent();
    const id = uuid();
    h.handleToolStart({ name: "t" }, "not json", id, undefined);
    expect(h.events.at(-1)).toMatchObject({ input: { raw: "not json", parsed: null }, invoked_by_run_id: null });
  });

  it("records error strings", () => {
    const h = mkTrellarAgent();
    const r = root(h);
    h.handleLLMError(new Error("llm bad"), uuid(), r);
    h.handleToolError(new Error("tool bad"), uuid(), r);
    h.handleChainError(new Error("chain bad"), uuid(), r);
    expect(h.events.slice(-3).map((e) => [e["event"], e["error"]])).toEqual([
      ["on_llm_error", "llm bad"],
      ["on_tool_error", "tool bad"],
      ["on_chain_error", "chain bad"],
    ]);
  });
});

describe("tool output attachment to the requesting LLM event", () => {
  function llmWithCalls(h: LangchainAgentCallback, parent: string, names: string[]): string {
    const id = uuid();
    h.handleLLMEnd(
      makeLlmResult({ message: makeAiMessage("", names.map((n, i) => ({ name: n, args: {}, id: `c${i}` }))) }),
      id,
      parent,
    );
    return id;
  }
  const runTool = (h: LangchainAgentCallback, parent: string, name: string, out: unknown) => {
    const id = uuid();
    h.handleToolStart({ name }, "{}", id, parent, [], {}, name);
    h.handleToolEnd(out, id, parent);
  };

  it("appends the response to the LLM event sharing the parent", () => {
    const h = mkTrellarAgent();
    const r = root(h);
    llmWithCalls(h, r, ["buy"]);
    const llmEvent = h.events.at(-1)!;
    runTool(h, r, "buy", new ToolMessage({ content: "bought", tool_call_id: "c0" }));
    expect((llmEvent["output"] as any).response).toBe("TOOL CALL: buy(args={})\nTOOL RESPONSE [buy]: TOOL MESSAGE: bought");
    expect(h._pendingLlmToolCalls).toEqual([]);
  });

  it("keeps the entry until all of its tools resolved", () => {
    const h = mkTrellarAgent();
    const r = root(h);
    llmWithCalls(h, r, ["a", "b"]);
    runTool(h, r, "a", "A");
    expect(h._pendingLlmToolCalls).toHaveLength(1);
    runTool(h, r, "b", "B");
    expect(h._pendingLlmToolCalls).toHaveLength(0);
  });

  it("does not cross-wire two parallel branches using different tools", () => {
    const h = mkTrellarAgent();
    const r = root(h);
    const p1 = uuid();
    const p2 = uuid();
    llmWithCalls(h, p1, ["a"]);
    const e1 = h.events.at(-1)!;
    llmWithCalls(h, p2, ["b"]);
    const e2 = h.events.at(-1)!;
    runTool(h, p2, "b", "B");
    runTool(h, p1, "a", "A");
    expect((e1["output"] as any).response).toContain("TOOL RESPONSE [a]: A");
    expect((e1["output"] as any).response).not.toContain("[b]");
    expect((e2["output"] as any).response).toContain("TOOL RESPONSE [b]: B");
    void r;
  });

  it("prefers the matching parent, else the most recent entry for the tool name", () => {
    const h = mkTrellarAgent();
    root(h);
    const p1 = uuid();
    const p2 = uuid();
    llmWithCalls(h, p1, ["same"]);
    const e1 = h.events.at(-1)!;
    llmWithCalls(h, p2, ["same"]);
    const e2 = h.events.at(-1)!;
    runTool(h, p1, "same", "X");
    expect((e1["output"] as any).response).toContain("TOOL RESPONSE [same]: X");
    expect((e2["output"] as any).response).not.toContain("TOOL RESPONSE");
    runTool(h, uuid(), "same", "Y"); // unknown parent -> falls back to the remaining entry
    expect((e2["output"] as any).response).toContain("TOOL RESPONSE [same]: Y");
  });

  it("is a no-op with no pending calls or an unmatched tool name", () => {
    const h = mkTrellarAgent();
    const r = root(h);
    runTool(h, r, "x", "out");
    llmWithCalls(h, r, ["a"]);
    runTool(h, r, "other", "out");
    expect(h._pendingLlmToolCalls).toHaveLength(1);
  });
});

describe("MCP detection", () => {
  it("isMcpToolRun: annotations metadata key or mcp_* artifacts", () => {
    expect(LangchainAgentCallback.isMcpToolRun({ annotations: undefined }, undefined)).toBe(true);
    expect(LangchainAgentCallback.isMcpToolRun({ annotations: { readOnlyHint: true } }, undefined)).toBe(true);
    expect(LangchainAgentCallback.isMcpToolRun({}, undefined)).toBe(false);
    expect(LangchainAgentCallback.isMcpToolRun(undefined, "plain string")).toBe(false);
    expect(LangchainAgentCallback.isMcpToolRun(undefined, { artifact: [] })).toBe(false);
    expect(LangchainAgentCallback.isMcpToolRun(undefined, { artifact: [{ type: "mcp_structured_content", data: {} }] })).toBe(true);
    expect(LangchainAgentCallback.isMcpToolRun(undefined, { artifact: [{ type: "image" }] })).toBe(false);
  });

  it("on_tool_end carries is_mcp_tool true only for MCP tools", () => {
    const h = mkTrellarAgent();
    const r = root(h);
    const mcp = uuid();
    const local = uuid();
    h.handleToolStart({ name: "create_purchase_ticket" }, "{}", mcp, r, [], { annotations: undefined }, "create_purchase_ticket");
    h.handleToolEnd(new ToolMessage({ content: "id-1", tool_call_id: "1" }), mcp, r);
    h.handleToolStart({ name: "buy_stocks" }, "{}", local, r, [], {}, "buy_stocks");
    h.handleToolEnd(new ToolMessage({ content: "bought", tool_call_id: "2" }), local, r);
    const ends = h.events.filter((e) => e["event"] === "on_tool_end");
    expect(ends.map((e) => [e["node_name"], e["is_mcp_tool"]])).toEqual([
      ["create_purchase_ticket", true],
      ["buy_stocks", false],
    ]);
  });

  it("renders the via-MCP marker in the context text", () => {
    const h = mkTrellarAgent();
    const r = root(h);
    const id = uuid();
    h.handleToolStart({ name: "t" }, "{}", id, r, [], { annotations: null }, "t");
    h.handleToolEnd("done", id, r);
    expect(h.buildContext()).toContain('output (via MCP): "done"');
  });
});

describe("events are always JSON-serializable and backend-valid", () => {
  it("survives a full tool-calling sequence with exotic values", async () => {
    const h = mkTrellarAgent();
    const r = root(h, { messages: [new HumanMessage("hi")], extra: new Map([["k", new Set([1])]]) });
    const circular: any = { name: "c" };
    circular.self = circular;
    h.handleChatModelStart({ kwargs: { model: "m" } }, [[new HumanMessage("hi")]], uuid(), r, {
      invocation_params: { tools: [{ type: "function", function: { name: "t", description: "d", parameters: { type: "object" } } }] },
    });
    h.handleChainStart({}, circular, uuid(), r, [], {}, undefined, "weird");
    const t = uuid();
    h.handleToolStart({ name: "t" }, "{}", t, r, [], {}, "t");
    h.handleToolEnd({ some: BigInt(5), fn: () => 1 }, t, r);
    await h.handleChainEnd({ out: undefined }, r);

    expect(() => JSON.stringify(h.events)).not.toThrow();
    assertValidAgentLoopRequest({
      context: h.events,
      trace_id: h.traceId,
      agent_name: h.agentName,
      observability_call: false,
      available_tools: Object.entries(h.availableTools).map(([tools_hash, tools]) => ({ tools_hash, tools })),
    });
  });
});

describe("auto-evaluate on root chain end", () => {
  const spy = () =>
    vi.spyOn(agentLoop, "evaluateWithTrellarAgent").mockResolvedValue({
      explanation: "e",
      score: 7,
      decisionIdentifier: "d",
      shouldStopNetwork: false,
    });

  async function run(mode: ObservabilityMode, evaluated = false) {
    const evaluate = spy();
    const h = mkTrellarAgent(mode);
    const r = root(h);
    h._evaluated = evaluated;
    await h.handleChainEnd({}, r);
    return { evaluate, h };
  }

  it("NONE never calls", async () => expect((await run("none")).evaluate).not.toHaveBeenCalled());
  it("ALWAYS calls with the observability flag", async () => {
    const { evaluate, h } = await run("always");
    expect(evaluate).toHaveBeenCalledWith(h, { _observabilityCall: true });
  });
  it("ALWAYS calls even if already evaluated", async () =>
    expect((await run("always", true)).evaluate).toHaveBeenCalledTimes(1));
  it("IF_NOT_EVALUATED calls when not yet evaluated", async () =>
    expect((await run("if_not_evaluated", false)).evaluate).toHaveBeenCalledTimes(1));
  it("IF_NOT_EVALUATED skips when already evaluated", async () =>
    expect((await run("if_not_evaluated", true)).evaluate).not.toHaveBeenCalled());

  it("non-root chain end never triggers auto-evaluation", async () => {
    const evaluate = spy();
    const h = mkTrellarAgent("always");
    const r = root(h);
    const child = uuid();
    h.handleChainStart({}, {}, child, r, [], {}, undefined, "n");
    await h.handleChainEnd({}, child, r);
    expect(evaluate).not.toHaveBeenCalled();
  });

  it("swallows and logs errors from the evaluation", async () => {
    vi.spyOn(agentLoop, "evaluateWithTrellarAgent").mockRejectedValue(new Error("backend down"));
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const h = mkTrellarAgent("always");
    const r = root(h);
    await expect(h.handleChainEnd({}, r)).resolves.toBeUndefined();
    expect(warn).toHaveBeenCalled();
    expect(isTrellarAgentActive(h)).toBe(false);
  });
});

describe("buildContext", () => {
  it("renders header and node names", () => {
    const h = mkTrellarAgent();
    const r = root(h);
    h.handleChatModelStart({ kwargs: { model: "m" } }, [[new SystemMessage("S"), new HumanMessage("H")]], uuid(), r);
    const text = h.buildContext();
    expect(text).toContain(`Trace ID: ${r}`);
    expect(text).toContain("Total steps: 2");
    expect(text).toContain("[Step 1] on_chain_start  (chain: LangGraph)");
    expect(text).toContain("  system: S");
    expect(text).toContain("  human: H");
  });
});
