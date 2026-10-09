import { afterEach, describe, expect, it, vi } from "vitest";

import {
  evaluateConfidence,
  NetworkHaltedError,
  ObservabilityMode,
  TrellarHTTPError,
} from "../src/index.js";
import { SingleCallGuardCallback } from "../src/callbacks/langchain/singleLangchainCallback.js";
import { activateGuard, releaseGuard, runWithGuard } from "../src/context.js";
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
      available_tools: [],
    });
    deactivate(h);
  });

  it("marks single-call guards with single_call: true", async () => {
    const calls = mockFetch();
    const h = new SingleCallGuardCallback({ agentName: "single" });
    h.traceId = "t";
    activateGuard(h);
    await evaluateConfidence();
    expect(calls[0]!.body.single_call).toBe(true);
    releaseGuard(h);
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
  it("throws when there is no active guard", async () => {
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

  it("throws after the guard was released (called after the run ended)", async () => {
    mockFetch();
    const h = makeActiveHandler();
    releaseGuard(h);
    await expect(evaluateConfidence()).rejects.toThrow(/No active callback handler/);
  });

  it("two active guards are ambiguous unless runWithGuard disambiguates", async () => {
    const calls = mockFetch();
    const a = makeActiveHandler("a");
    const b = makeActiveHandler("b");
    await expect(evaluateConfidence()).rejects.toThrow(/No active callback handler/);
    await runWithGuard(b, () => evaluateConfidence());
    expect(calls[0]!.body.agent_name).toBe("b");
    deactivate(a);
    deactivate(b);
  });
});

describe("evaluateConfidence result", () => {
  it("returns a frozen AgentLoopResult and marks the guard evaluated", async () => {
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
    const { evaluateWithGuard } = await import("../src/agentLoop.js");
    const result = await evaluateWithGuard(h, { _observabilityCall: true });
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
