/**
 * LangChain / LangGraph guards. Requires ``@langchain/core`` (optional peer dependency).
 *
 * ```ts
 * import { getAgentGuard } from "trellar/langchain";
 * import { evaluateConfidence } from "trellar";
 * ```
 */
import { ObservabilityMode } from "./agentLoop.js";
import { AgentGuardCallback } from "./callbacks/langchain/langchainCallback.js";
import { SingleCallGuardCallback } from "./callbacks/langchain/singleLangchainCallback.js";

export {
  evaluateConfidence,
  NetworkHaltedError,
  ObservabilityMode,
  TrellarHTTPError,
  type AgentLoopResult,
  type EvaluateConfidenceOptions,
} from "./agentLoop.js";
export { runWithGuard } from "./context.js";
export type { AgentGuardCallback, SingleCallGuardCallback };

/**
 * Create a callback handler that identifies this graph to the Trellar backend.
 *
 * ``agentName`` must be a stable, unique name for this agent graph within your
 * repository (e.g. ``'research-agent'``, ``'support-bot'``). The backend uses
 * it to look up and maintain the graph's network profile across runs.
 * Different graphs in the same repo must use different names.
 *
 * ```ts
 * const guard = getAgentGuard("research-agent");
 *
 * async function confidenceGate(state) {
 *   const result = await evaluateConfidence();
 *   // ...
 * }
 *
 * await graph.invoke(input, { callbacks: [guard] });
 * ```
 *
 * ``evaluateConfidence()`` must be called from inside a graph node, while the
 * run is still in progress -- not after ``graph.invoke()`` returns. The guard
 * is released as soon as the root run ends, so a call made after ``invoke()``
 * returns will throw.
 *
 * @param agentName Unique, stable name for this agent graph.
 * @param observabilityMode Controls whether ``evaluateConfidence()`` is
 *   auto-triggered when the graph run finishes. Defaults to ``ObservabilityMode.NONE``.
 */
export function getAgentGuard(
  agentName: string,
  observabilityMode: ObservabilityMode = ObservabilityMode.NONE,
): AgentGuardCallback {
  return new AgentGuardCallback({ agentName, observabilityMode });
}

/**
 * Create a callback handler for a single bare LLM call (no LangGraph/chain wrapper).
 *
 * Use this instead of ``getAgentGuard`` when you are calling a chat model
 * directly (e.g. ``llm.invoke(...)``) rather than invoking a graph or an agent
 * built with ``createAgent`` (which is itself a compiled graph, and already
 * works with ``getAgentGuard``).
 *
 * A bare ``llm.invoke()`` call has no node to call ``evaluateConfidence()``
 * from mid-run, and the guard is released as soon as the call finishes -- so a
 * manual call is never supported here. Use ``ObservabilityMode.ALWAYS`` (or
 * ``IF_NOT_EVALUATED``) to auto-trigger the evaluation, then read the result
 * off the guard:
 *
 * ```ts
 * const guard = getSingleCallGuard("single-llm-call", ObservabilityMode.ALWAYS);
 * await llm.invoke(messages, { callbacks: [guard] });
 * const result = guard.trellarEvaluateResult;
 * ```
 */
export function getSingleCallGuard(
  agentName: string,
  observabilityMode: ObservabilityMode = ObservabilityMode.NONE,
): SingleCallGuardCallback {
  return new SingleCallGuardCallback({ agentName, observabilityMode });
}
