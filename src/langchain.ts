/**
 * LangChain / LangGraph agents. Requires ``@langchain/core`` (optional peer dependency).
 *
 * ```ts
 * import { trellarLangchainAgent } from "trellar/langchain";
 * import { evaluateConfidence } from "trellar";
 * ```
 */
import { ObservabilityMode } from "./agentLoop.js";
import { deprecatedAlias } from "./deprecation.js";
import { LangchainAgentCallback } from "./callbacks/langchain/langchainCallback.js";
import { LangchainSingleCallCallback } from "./callbacks/langchain/singleLangchainCallback.js";

export {
  evaluateConfidence,
  NetworkHaltedError,
  ObservabilityMode,
  TrellarHTTPError,
  type AgentLoopResult,
  type EvaluateConfidenceOptions,
} from "./agentLoop.js";
export { runWithTrellarAgent, runWithGuard } from "./context.js";
export type { LangchainAgentCallback, LangchainSingleCallCallback };

/**
 * Create a callback handler that identifies this graph to the Trellar backend.
 *
 * ``agentName`` must be a stable, unique name for this agent graph within your
 * repository (e.g. ``'research-agent'``, ``'support-bot'``). The backend uses
 * it to look up and maintain the graph's network profile across runs.
 * Different graphs in the same repo must use different names.
 *
 * ```ts
 * const trellarAgent = trellarLangchainAgent("research-agent");
 *
 * async function confidenceGate(state) {
 *   const result = await evaluateConfidence();
 *   // ...
 * }
 *
 * await graph.invoke(input, { callbacks: [trellarAgent] });
 * ```
 *
 * ``evaluateConfidence()`` must be called from inside a graph node, while the
 * run is still in progress -- not after ``graph.invoke()`` returns. The Trellar agent
 * is released as soon as the root run ends, so a call made after ``invoke()``
 * returns will throw.
 *
 * @param agentName Unique, stable name for this agent graph.
 * @param observabilityMode Controls whether ``evaluateConfidence()`` is
 *   auto-triggered when the graph run finishes. Defaults to ``ObservabilityMode.NONE``.
 */
export function trellarLangchainAgent(
  agentName: string,
  observabilityMode: ObservabilityMode = ObservabilityMode.NONE,
): LangchainAgentCallback {
  return new LangchainAgentCallback({ agentName, observabilityMode });
}

/**
 * Create a callback handler for a single bare LLM call (no LangGraph/chain wrapper).
 *
 * Use this instead of ``trellarLangchainAgent`` when you are calling a chat model
 * directly (e.g. ``llm.invoke(...)``) rather than invoking a graph or an agent
 * built with ``createAgent`` (which is itself a compiled graph, and already
 * works with ``trellarLangchainAgent``).
 *
 * A bare ``llm.invoke()`` call has no node to call ``evaluateConfidence()``
 * from mid-run, and the Trellar agent is released as soon as the call finishes -- so a
 * manual call is never supported here. The evaluation is therefore triggered
 * automatically (``ObservabilityMode.ALWAYS`` is the default here, pass ``NONE``
 * to only record). Read the result off the Trellar agent:
 *
 * ```ts
 * const trellarAgent = trellarLangchainSingleCall("single-llm-call");
 * await llm.invoke(messages, { callbacks: [trellarAgent] });
 * const result = trellarAgent.trellarEvaluateResult;
 * ```
 */
export function trellarLangchainSingleCall(
  agentName: string,
  observabilityMode: ObservabilityMode = ObservabilityMode.ALWAYS,
): LangchainSingleCallCallback {
  return new LangchainSingleCallCallback({ agentName, observabilityMode });
}

/** @deprecated Use {@link trellarLangchainAgent}. */
export const getAgentGuard = deprecatedAlias("getAgentGuard", "trellarLangchainAgent", trellarLangchainAgent);

/** @deprecated Use {@link trellarLangchainSingleCall}. */
export const getSingleCallGuard = deprecatedAlias(
  "getSingleCallGuard",
  "trellarLangchainSingleCall",
  trellarLangchainSingleCall,
);
