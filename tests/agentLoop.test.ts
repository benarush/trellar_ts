import { afterEach, describe, expect, it, vi } from "vitest";

import {
  evaluateConfidence,
  NetworkHaltedError,
  ObservabilityMode,
  TrellarHTTPError,
} from "../src/index.js";
import { trellarLangchainAgent, trellarLangchainSingleCall } from "../src/langchain.js";
import { trellarStrandsAgent, trellarStrandsSingleCall } from "../src/strands.js";
import { LangchainSingleCallCallback } from "../src/callbacks/langchain/singleLangchainCallback.js";
import { activateTrellarAgent, releaseTrellarAgent, runWithTrellarAgent } from "../src/context.js";
import { assertValidAgentLoopRequest, deactivate, makeActiveHandler, mockFetch, OK_BODY, useCleanEnv } from "./helpers.js";

useCleanEnv();

describe("evaluateConfidence request", () => {
  afterEach(() => vi.useRealTimers());

  it("posts to the fixed endpoint", async () => {
    const calls = mockFetch();
    const h = makeActiveHandler();
    await evaluateConfidence({ apiKey: "k" });
    expect(calls[0]!.url).toBe("https://api.trellar.io/agent-gateway/v1/agent-loop");
    deactivate(h);
  });

  it("sends the bearer header and json content type", async () => {
    const calls = mockFetch();
    const h = makeActiveHandler();
    await evaluateConfidence({ apiKey: "secret" });
    expect(calls[0]!.init.headers["Authorization"]).toBe("Bearer secret");
    expect(calls[0]!.init.headers["Content-Type"]).toBe("application/json");
    deactivate(h);
  });

  it("falls back to the TRELLAR_API_KEY env var", async () => {
    const calls = mockFetch();
    const h = makeActiveHandler();
    await evaluateConfidence();
    expect(calls[0]!.init.headers["Authorization"]).toBe("Bearer env-key");
    deactivate(h);
  });

  it("sends the recorded events, trace_id, agent_name and flags", async () => {
    const calls = mockFetch();
    const h = makeActiveHandler("my-agent");
    h.events.push({ event: "x" });
    await evaluateConfidence();
    expect(calls[0]!.body).toEqual({
      context: [{ event: "x" }],
      trace_id: h.traceId,
      agent_name: "my-agent",
      observability_call: false,
      single_call: false,
      total_duration_seconds: null, // no root run was started on this hand-built handler
      available_tools: [],
      tags: {},
    });
    deactivate(h);
  });

  it("marks single-call Trellar agents with single_call: true", async () => {
    const calls = mockFetch();
    const h = new LangchainSingleCallCallback({ agentName: "single" });
    h.traceId = "t";
    activateTrellarAgent(h);
    await evaluateConfidence();
    expect(calls[0]!.body.single_call).toBe(true);
    releaseTrellarAgent(h);
  });

  it("sends available_tools as {tools_hash, tools} entries", async () => {
    const calls = mockFetch();
    const h = makeActiveHandler();
    h.availableTools["abc"] = [{ type: "function", function: { name: "f", description: "", parameters: {} } }];
    await evaluateConfidence();
    expect(calls[0]!.body.available_tools).toEqual([
      { tools_hash: "abc", tools: [{ type: "function", function: { name: "f", description: "", parameters: {} } }] },
    ]);
    deactivate(h);
  });

  it("uses a 30 second default timeout and forwards a custom one", async () => {
    const spy = vi.spyOn(AbortSignal, "timeout");
    mockFetch();
    const h = makeActiveHandler();
    await evaluateConfidence();
    await evaluateConfidence({ timeout: 90 });
    expect(spy.mock.calls.map((c) => c[0])).toEqual([30000, 90000]);
    deactivate(h);
  });

  it("produces a payload the backend schema accepts", async () => {
    const calls = mockFetch();
    const h = makeActiveHandler();
    await evaluateConfidence();
    assertValidAgentLoopRequest(calls[0]!.body);
    deactivate(h);
  });
});

describe("evaluateConfidence preconditions", () => {
  it("throws when there is no active Trellar agent", async () => {
    mockFetch();
    await expect(evaluateConfidence()).rejects.toThrow(/No active callback handler/);
  });

  it("throws when trace_id is not resolved", async () => {
    mockFetch();
    const h = makeActiveHandler();
    h.traceId = null;
    await expect(evaluateConfidence()).rejects.toThrow(/trace_id could not be resolved/);
    deactivate(h);
  });

  it("throws when no api key is available", async () => {
    mockFetch();
    vi.stubEnv("TRELLAR_API_KEY", "");
    const h = makeActiveHandler();
    await expect(evaluateConfidence()).rejects.toThrow(/TRELLAR_API_KEY/);
    deactivate(h);
  });

  it("throws after the Trellar agent was released (called after the run ended)", async () => {
    mockFetch();
    const h = makeActiveHandler();
    releaseTrellarAgent(h);
    await expect(evaluateConfidence()).rejects.toThrow(/No active callback handler/);
  });

  it("two active Trellar agents are ambiguous unless runWithTrellarAgent disambiguates", async () => {
    const calls = mockFetch();
    const a = makeActiveHandler("a");
    const b = makeActiveHandler("b");
    await expect(evaluateConfidence()).rejects.toThrow(/No active callback handler/);
    await runWithTrellarAgent(b, () => evaluateConfidence());
    expect(calls[0]!.body.agent_name).toBe("b");
    deactivate(a);
    deactivate(b);
  });
});

describe("evaluateConfidence result", () => {
  it("returns a frozen AgentLoopResult and marks the Trellar agent evaluated", async () => {
    mockFetch();
    const h = makeActiveHandler();
    const result = await evaluateConfidence();
    expect(result).toEqual({
      explanation: "looks good",
      score: 8,
      decisionIdentifier: "decision-1",
      shouldStopNetwork: false,
    });
    expect(Object.isFrozen(result)).toBe(true);
    expect(h._evaluated).toBe(true);
    deactivate(h);
  });

  it("raises TrellarHTTPError on non-2xx responses", async () => {
    mockFetch({ detail: "nope" }, 422, "Unprocessable Entity");
    const h = makeActiveHandler();
    const error = await evaluateConfidence().catch((e) => e);
    expect(error).toBeInstanceOf(TrellarHTTPError);
    expect(error.status).toBe(422);
    expect(error.body).toContain("nope");
    deactivate(h);
  });

  it("raises NetworkHaltedError when the backend says stop", async () => {
    mockFetch({ ...OK_BODY, should_stop_network: true, score: 2 });
    const h = makeActiveHandler();
    const error = await evaluateConfidence().catch((e) => e);
    expect(error).toBeInstanceOf(NetworkHaltedError);
    expect(error.score).toBe(2);
    expect(error.decisionIdentifier).toBe("decision-1");
    expect(error.message).toContain("decision_identifier=decision-1");
    deactivate(h);
  });

  it("does not raise NetworkHaltedError for observability calls", async () => {
    mockFetch({ ...OK_BODY, should_stop_network: true });
    const h = makeActiveHandler();
    const { evaluateWithTrellarAgent } = await import("../src/agentLoop.js");
    const result = await evaluateWithTrellarAgent(h, { _observabilityCall: true });
    expect(result.shouldStopNetwork).toBe(true);
    deactivate(h);
  });
});

describe("ObservabilityMode", () => {
  it("has the same values as Python", () => {
    expect(ObservabilityMode).toEqual({
      ALWAYS: "always",
      IF_NOT_EVALUATED: "if_not_evaluated",
      NONE: "none",
    });
  });
});

describe("tags", () => {
  const run = async (agent: ReturnType<typeof trellarLangchainAgent>, options = {}) => {
    const calls = mockFetch();
    agent.traceId = "11111111-1111-4111-8111-111111111111";
    activateTrellarAgent(agent);
    try {
      await evaluateConfidence({ apiKey: "k", ...options });
    } finally {
      releaseTrellarAgent(agent);
    }
    return calls;
  };

  it("sends creation tags", async () => {
    const calls = await run(trellarLangchainAgent("t", ObservabilityMode.NONE, { env: "prod" }));
    expect(calls[0]!.body.tags).toEqual({ env: "prod" });
  });

  it("merges per-call tags on top and does not mutate creation tags", async () => {
    const agent = trellarLangchainAgent("t", ObservabilityMode.NONE, { env: "prod", topic: "a" });
    const calls = await run(agent, { tags: { topic: "b", customer_id: 5 } });
    expect(calls[0]!.body.tags).toEqual({ env: "prod", topic: "b", customer_id: "5" });
    expect(agent.tags).toEqual({ env: "prod", topic: "a" });
  });

  it.each([
    ["langchain agent", () => trellarLangchainAgent("t", undefined, { a: 1 })],
    ["langchain single call", () => trellarLangchainSingleCall("t", undefined, { a: 1 })],
    ["strands agent", () => trellarStrandsAgent("t", undefined, { a: 1 })],
    ["strands single call", () => trellarStrandsSingleCall("t", undefined, { a: 1 })],
  ])("%s factory stores tags", (_name, make) => {
    expect(make().tags).toEqual({ a: "1" });
  });

  it.each([
    ["x", "x"],
    [5, "5"],
    [1.5, "1.5"],
    [true, "true"],
    [false, "false"],
  ])("converts %j with String()", (value, expected) => {
    expect(trellarLangchainAgent("t", undefined, { k: value }).tags).toEqual({ k: expected });
  });

  it.each([["x"], [["a"]]])("rejects invalid tags shape %j", async (bad) => {
    expect(() => trellarLangchainAgent("t", undefined, bad as any)).toThrow(TypeError);
    const calls = mockFetch();
    const h = makeActiveHandler();
    await expect(evaluateConfidence({ tags: bad as any })).rejects.toThrow(TypeError);
    expect(calls).toHaveLength(0);
    deactivate(h);
  });
});
