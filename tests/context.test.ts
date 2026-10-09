import { describe, expect, it } from "vitest";

import { activateGuard, isGuardActive, releaseGuard, resolveActiveGuard, runWithGuard } from "../src/context.js";
import { evaluateConfidence } from "../src/index.js";
import { AgentGuardCallback } from "../src/callbacks/langchain/langchainCallback.js";
import { mockFetch, useCleanEnv } from "./helpers.js";

useCleanEnv();

const make = (name: string) => {
  const g = new AgentGuardCallback({ agentName: name });
  g.traceId = `trace-${name}`;
  return g;
};

describe("active guard resolution", () => {
  it("is undefined when no run is open", () => {
    expect(resolveActiveGuard()).toBeUndefined();
  });

  it("falls back to the single open root run", () => {
    const g = make("solo");
    activateGuard(g);
    try {
      expect(resolveActiveGuard()).toBe(g);
    } finally {
      releaseGuard(g);
    }
    expect(resolveActiveGuard()).toBeUndefined();
  });

  it("is ambiguous with several open roots unless a scope is bound", async () => {
    const a = make("a");
    const b = make("b");
    activateGuard(a);
    activateGuard(b);
    try {
      expect(resolveActiveGuard()).toBeUndefined();
      expect(runWithGuard(a, () => resolveActiveGuard())).toBe(a);
      expect(runWithGuard(b, () => resolveActiveGuard())).toBe(b);
      await expect(evaluateConfidence({ apiKey: "k" })).rejects.toThrow(/No active callback handler/);
    } finally {
      releaseGuard(a);
      releaseGuard(b);
    }
  });

  it("runWithGuard keeps isolation across interleaved async work", async () => {
    const calls = mockFetch();
    const a = make("a");
    const b = make("b");
    activateGuard(a);
    activateGuard(b);
    try {
      const run = (g: AgentGuardCallback, delay: number) =>
        runWithGuard(g, async () => {
          await new Promise((r) => setTimeout(r, delay));
          return evaluateConfidence({ apiKey: "k" });
        });
      await Promise.all([run(a, 20), run(b, 5)]);
    } finally {
      releaseGuard(a);
      releaseGuard(b);
    }
    expect(calls.map((c) => c.body.agent_name).sort()).toEqual(["a", "b"]);
  });

  it("a scoped guard that already ended is not used", () => {
    const g = make("ended");
    activateGuard(g);
    releaseGuard(g);
    expect(isGuardActive(g)).toBe(false);
    expect(runWithGuard(g, () => resolveActiveGuard())).toBeUndefined();
  });
});
