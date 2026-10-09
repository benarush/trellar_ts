/** Shared test helpers (counterpart of tests/conftest.py). */
import { afterEach, beforeEach, vi } from "vitest";

import { AgentGuardCallback } from "../src/callbacks/langchain/langchainCallback.js";
import { activateGuard, releaseGuard } from "../src/context.js";
import { assertValidAgentLoopRequest } from "./backendSchema.js";

export const OK_BODY = {
  score: 8,
  explanation: "looks good",
  decision_identifier: "decision-1",
  should_stop_network: false,
};

export interface FetchCall {
  url: string;
  init: RequestInit & { headers: Record<string, string> };
  body: any;
}

/** Mock ``fetch`` for the duration of a test; returns the recorded calls. */
export function mockFetch(body: unknown = OK_BODY, status = 200, statusText = "OK"): FetchCall[] {
  const calls: FetchCall[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init: any) => {
      const parsed = JSON.parse(init.body);
      calls.push({ url, init, body: parsed });
      return new Response(JSON.stringify(body), {
        status,
        statusText,
        headers: { "Content-Type": "application/json" },
      });
    }),
  );
  return calls;
}

/** Install the standard env / fetch cleanup hooks for a test file. */
export function useCleanEnv(): void {
  beforeEach(() => {
    vi.stubEnv("TRELLAR_API_KEY", "env-key");
  });
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
  });
}

/** A guard activated with a fake trace id (counterpart of the ``active_handler`` fixture). */
export function makeActiveHandler(name = "test-agent"): AgentGuardCallback {
  const handler = new AgentGuardCallback({ agentName: name });
  handler.traceId = "11111111-1111-4111-8111-111111111111";
  activateGuard(handler);
  return handler;
}

export function deactivate(handler: AgentGuardCallback): void {
  releaseGuard(handler);
}

export { assertValidAgentLoopRequest };
