import { describe, expect, it } from "vitest";

import { activateTrellarAgent, isTrellarAgentActive, releaseTrellarAgent, resolveActiveTrellarAgent, runWithTrellarAgent } from "../src/context.js";
import { evaluateConfidence } from "../src/index.js";
import { LangchainAgentCallback } from "../src/callbacks/langchain/langchainCallback.js";
import { mockFetch, useCleanEnv } from "./helpers.js";

useCleanEnv();

const make = (name: string) => {
  const g = new LangchainAgentCallback({ agentName: name });
  g.traceId = `trace-${name}`;
  return g;
};

describe("active Trellar agent resolution", () => {
  it("is undefined when no run is open", () => {
    expect(resolveActiveTrellarAgent()).toBeUndefined();
  });

  it("falls back to the single open root run", () => {
    const g = make("solo");
    activateTrellarAgent(g);
    try {
      expect(resolveActiveTrellarAgent()).toBe(g);
    } finally {
      releaseTrellarAgent(g);
    }
    expect(resolveActiveTrellarAgent()).toBeUndefined();
  });

  it("is ambiguous with several open roots unless a scope is bound", async () => {
    const a = make("a");
    const b = make("b");
    activateTrellarAgent(a);
    activateTrellarAgent(b);
    try {
      expect(resolveActiveTrellarAgent()).toBeUndefined();
      expect(runWithTrellarAgent(a, () => resolveActiveTrellarAgent())).toBe(a);
      expect(runWithTrellarAgent(b, () => resolveActiveTrellarAgent())).toBe(b);
      await expect(evaluateConfidence({ apiKey: "k" })).rejects.toThrow(/No active callback handler/);
    } finally {
      releaseTrellarAgent(a);
      releaseTrellarAgent(b);
    }
  });

  it("runWithTrellarAgent keeps isolation across interleaved async work", async () => {
    const calls = mockFetch();
    const a = make("a");
    const b = make("b");
    activateTrellarAgent(a);
    activateTrellarAgent(b);
    try {
      const run = (g: LangchainAgentCallback, delay: number) =>
        runWithTrellarAgent(g, async () => {
          await new Promise((r) => setTimeout(r, delay));
          return evaluateConfidence({ apiKey: "k" });
        });
      await Promise.all([run(a, 20), run(b, 5)]);
    } finally {
      releaseTrellarAgent(a);
      releaseTrellarAgent(b);
    }
    expect(calls.map((c) => c.body.agent_name).sort()).toEqual(["a", "b"]);
  });

  it("a scoped Trellar agent that already ended is not used", () => {
    const g = make("ended");
    activateTrellarAgent(g);
    releaseTrellarAgent(g);
    expect(isTrellarAgentActive(g)).toBe(false);
    expect(runWithTrellarAgent(g, () => resolveActiveTrellarAgent())).toBeUndefined();
  });
});
