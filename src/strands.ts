/**
 * Strands Agents guards. Requires ``@strands-agents/sdk`` >= 1.19 (Node 22+).
 *
 * ```ts
 * import { getStrandsGuard } from "trellar/strands";
 * import { evaluateConfidence } from "trellar";
 * ```
 */
import { ObservabilityMode } from "./agentLoop.js";
import {
  StrandsGuardCallback,
  StrandsSingleCallGuardCallback,
} from "./callbacks/strands/strandsCallback.js";

export {
  evaluateConfidence,
  NetworkHaltedError,
  ObservabilityMode,
  TrellarHTTPError,
  type AgentLoopResult,
  type EvaluateConfidenceOptions,
} from "./agentLoop.js";
export { runWithGuard } from "./context.js";
export type { StrandsGuardCallback, StrandsSingleCallGuardCallback };

/**
 * Create a plugin that identifies a Strands Agents network to Trellar.
 *
 * Register the same guard on every Agent and on the Graph/Swarm (so the whole
 * run is one trace):
 *
 * ```ts
 * const guard = getStrandsGuard("research-agent");
 *
 * const agent = new Agent({ name: "searcher", plugins: [guard] });
 * const graph = new Graph({ nodes: [...], edges: [...], plugins: [guard] });
 * ```
 *
 * Give every agent a stable ``name``: it is how the backend tells agents apart.
 * As with ``getAgentGuard``, call ``evaluateConfidence`` from inside the run
 * (e.g. a graph node or a tool), not after it returns.
 *
 * @param agentName Unique, stable name for this agent network.
 * @param observabilityMode See ``ObservabilityMode``.
 */
export function getStrandsGuard(
  agentName: string,
  observabilityMode: ObservabilityMode = ObservabilityMode.NONE,
): StrandsGuardCallback {
  return new StrandsGuardCallback({ agentName, observabilityMode });
}

/**
 * Create a plugin for one Strands Agent called once (no Graph/Swarm).
 *
 * The run is over when the call returns, so a manual ``evaluateConfidence`` is
 * not possible. Use ``ObservabilityMode.ALWAYS`` (or ``IF_NOT_EVALUATED``) and
 * read the outcome off the guard:
 *
 * ```ts
 * const guard = getStrandsSingleCallGuard("faq-agent", ObservabilityMode.ALWAYS);
 * const agent = new Agent({ name: "faq", plugins: [guard] });
 * await agent.invoke("What time does the office open?");
 * const result = guard.trellarEvaluateResult;
 * ```
 *
 * Requests are marked ``single_call: true`` in the payload. Not covered:
 * ``agent.structuredOutput()`` and calling a Strands ``Model`` directly
 * (Strands fires no model-call hooks for them).
 */
export function getStrandsSingleCallGuard(
  agentName: string,
  observabilityMode: ObservabilityMode = ObservabilityMode.NONE,
): StrandsSingleCallGuardCallback {
  return new StrandsSingleCallGuardCallback({ agentName, observabilityMode });
}
