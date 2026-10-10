/**
 * Strands Agents. Requires ``@strands-agents/sdk`` >= 1.19 (Node 22+).
 *
 * ```ts
 * import { trellarStrandsAgent } from "trellar/strands";
 * import { evaluateConfidence } from "trellar";
 * ```
 */
import { ObservabilityMode } from "./agentLoop.js";
import { deprecatedAlias } from "./deprecation.js";
import {
  StrandsAgentCallback,
  StrandsSingleCallCallback,
} from "./callbacks/strands/strandsCallback.js";

export {
  evaluateConfidence,
  NetworkHaltedError,
  ObservabilityMode,
  TrellarHTTPError,
  type AgentLoopResult,
  type EvaluateConfidenceOptions,
} from "./agentLoop.js";
export { runWithTrellarAgent, runWithGuard } from "./context.js";
export type { StrandsAgentCallback, StrandsSingleCallCallback };

/**
 * Create a plugin that identifies a Strands Agents network to Trellar.
 *
 * Register the Trellar agent on the Graph/Swarm (so the whole run is one trace); every
 * node's Agent is bound automatically:
 *
 * ```ts
 * const trellarAgent = trellarStrandsAgent("research-agent");
 *
 * const agent = new Agent({ name: "searcher" });
 * const graph = new Graph({ nodes: [agent, ...], edges: [...], plugins: [trellarAgent] });
 * ```
 *
 * A standalone Agent (no Graph) needs `new Agent({ plugins: [trellarAgent] })`.
 *
 * Give every agent a stable ``name``: it is how the backend tells agents apart.
 * As with ``trellarLangchainAgent``, call ``evaluateConfidence`` from inside the run
 * (e.g. a graph node or a tool), not after it returns.
 *
 * @param agentName Unique, stable name for this agent network.
 * @param observabilityMode See ``ObservabilityMode``.
 */
export function trellarStrandsAgent(
  agentName: string,
  observabilityMode: ObservabilityMode = ObservabilityMode.NONE,
): StrandsAgentCallback {
  return new StrandsAgentCallback({ agentName, observabilityMode });
}

/**
 * Create a plugin for one Strands Agent called once (no Graph/Swarm).
 *
 * The run is over when the call returns, so a manual ``evaluateConfidence`` is
 * not possible; the run is therefore evaluated automatically
 * (``ObservabilityMode.ALWAYS`` is the default here, pass ``NONE`` to only record).
 * Read the outcome off the Trellar agent:
 *
 * ```ts
 * const trellarAgent = trellarStrandsSingleCall("faq-agent");
 * const agent = new Agent({ name: "faq", plugins: [trellarAgent] });
 * await agent.invoke("What time does the office open?");
 * const result = trellarAgent.trellarEvaluateResult;
 * ```
 *
 * Requests are marked ``single_call: true`` in the payload. Not covered:
 * ``agent.structuredOutput()`` and calling a Strands ``Model`` directly
 * (Strands fires no model-call hooks for them).
 */
export function trellarStrandsSingleCall(
  agentName: string,
  observabilityMode: ObservabilityMode = ObservabilityMode.ALWAYS,
): StrandsSingleCallCallback {
  return new StrandsSingleCallCallback({ agentName, observabilityMode });
}

/** @deprecated Use {@link trellarStrandsAgent}. */
export const getStrandsGuard = deprecatedAlias("getStrandsGuard", "trellarStrandsAgent", trellarStrandsAgent);

/** @deprecated Use {@link trellarStrandsSingleCall}. */
export const getStrandsSingleCallGuard = deprecatedAlias(
  "getStrandsSingleCallGuard",
  "trellarStrandsSingleCall",
  trellarStrandsSingleCall,
);
