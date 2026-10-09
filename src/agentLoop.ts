import { type GuardState, resolveActiveGuard } from "./context.js";
import * as settings from "./settings.js";

/** Result of one ``evaluateConfidence()`` call. */
export interface AgentLoopResult {
  readonly explanation: string;
  /** Confidence score from 1 (low) to 10 (high). */
  readonly score: number;
  readonly decisionIdentifier: string;
  readonly shouldStopNetwork: boolean;
}

/**
 * Controls whether the guard auto-triggers ``evaluateConfidence()`` when the
 * graph's root run finishes (i.e. ``graph.invoke()`` is about to return).
 *
 * - ``ALWAYS``           -- always auto-call at the end of the run.
 * - ``IF_NOT_EVALUATED`` -- auto-call at the end only if ``evaluateConfidence()``
 *   was not already successfully called anywhere during the run.
 * - ``NONE``             -- never auto-call (default).
 *
 * Errors raised by an auto-triggered call are caught and logged, never
 * propagated out of ``graph.invoke()``.
 */
export const ObservabilityMode = {
  ALWAYS: "always",
  IF_NOT_EVALUATED: "if_not_evaluated",
  NONE: "none",
} as const;

export type ObservabilityMode = (typeof ObservabilityMode)[keyof typeof ObservabilityMode];

/** Validate an ``ObservabilityMode`` value (Python: ``ObservabilityMode(value)``). */
export function parseObservabilityMode(value: unknown): ObservabilityMode {
  const allowed = Object.values(ObservabilityMode) as string[];
  if (typeof value === "string" && allowed.includes(value)) return value as ObservabilityMode;
  throw new Error(`${String(value)} is not a valid ObservabilityMode`);
}

/** Raised when the Trellar backend signals the agent network must stop. */
export class NetworkHaltedError extends Error {
  readonly explanation: string;
  readonly score: number;
  readonly decisionIdentifier: string;

  constructor(explanation: string, score: number, decisionIdentifier: string) {
    super(`Trellar halted the agent network (decision_identifier=${decisionIdentifier}): ${explanation}`);
    this.name = "NetworkHaltedError";
    this.explanation = explanation;
    this.score = score;
    this.decisionIdentifier = decisionIdentifier;
  }
}

/** Raised on non-2xx responses from the Trellar backend (Python: ``requests.HTTPError``). */
export class TrellarHTTPError extends Error {
  readonly status: number;
  readonly statusText: string;
  readonly body: string;
  readonly url: string;

  constructor(status: number, statusText: string, url: string, body: string) {
    super(`${status} ${statusText} for url: ${url}`);
    this.name = "TrellarHTTPError";
    this.status = status;
    this.statusText = statusText;
    this.url = url;
    this.body = body;
  }
}

export interface EvaluateConfidenceOptions {
  /** Bearer token. Defaults to the ``TRELLAR_API_KEY`` env var. */
  apiKey?: string;
  /** HTTP request timeout in seconds (default 30). */
  timeout?: number;
  /**
   * Internal -- set by the guard's auto-trigger (see ``ObservabilityMode``) to
   * mark the request as automatic rather than a manual call. Not for external use.
   */
  _observabilityCall?: boolean;
}

/**
 * Send ``guard``'s recorded run to the backend. Shared by the public
 * ``evaluateConfidence()`` and by the guards' auto-trigger, which already
 * holds the guard and must not depend on async-context lookup.
 *
 * @internal
 */
export async function evaluateWithGuard(
  guard: GuardState,
  options: EvaluateConfidenceOptions = {},
): Promise<AgentLoopResult> {
  const { apiKey, timeout = 30.0, _observabilityCall = false } = options;

  if (!guard.traceId) {
    throw new Error(
      "trace_id could not be resolved. Make sure getAgentGuard() is passed to " +
        "graph.invoke() before calling evaluateConfidence().",
    );
  }
  const resolvedTraceId = String(guard.traceId);

  const baseUrl = settings.DEFAULT_ENDPOINT.replace(/\/+$/, "");
  const key = apiKey || settings.getEnvApiKey();
  if (!key) {
    throw new Error(
      `apiKey must be provided or set via ${settings.ENV_TRELLAR_API_KEY} environment variable.`,
    );
  }

  const url = `${baseUrl}/agent-gateway/v1/agent-loop`;
  const payload = {
    context: guard.events,
    trace_id: resolvedTraceId,
    agent_name: guard.agentName,
    observability_call: _observabilityCall,
    single_call: guard.isSingleCall ?? false,
    // One entry per distinct toolset bound during this run, keyed by a
    // content hash (not the model name) so the backend can correlate it
    // back to the exact SubAgent that declared it.
    available_tools: Object.entries(guard.availableTools).map(([toolsHash, tools]) => ({
      tools_hash: toolsHash,
      tools,
    })),
  };

  const response = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${key}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(Math.round(timeout * 1000)),
  });

  if (!response.ok) {
    let body = "";
    try {
      body = await response.text();
    } catch {
      body = "";
    }
    throw new TrellarHTTPError(response.status, response.statusText, url, body);
  }

  const data = (await response.json()) as {
    explanation: string;
    score: number;
    decision_identifier: string;
    should_stop_network: boolean;
  };
  const result: AgentLoopResult = Object.freeze({
    explanation: data.explanation,
    score: data.score,
    decisionIdentifier: data.decision_identifier,
    shouldStopNetwork: data.should_stop_network,
  });
  guard._evaluated = true;

  if (result.shouldStopNetwork && !_observabilityCall) {
    throw new NetworkHaltedError(result.explanation, result.score, result.decisionIdentifier);
  }

  return result;
}

/**
 * Call the Trellar backend to get a confidence score.
 *
 * ``context``, ``trace_id``, and ``agent_name`` are all resolved automatically
 * from the active guard (``getAgentGuard`` / ``getStrandsGuard`` ...) -- no
 * manual wiring needed. Must be called from inside a graph node while the run
 * is still in progress, not after ``graph.invoke()`` returns.
 *
 * @throws Error when no active guard / trace id / api key can be resolved.
 * @throws TrellarHTTPError on non-2xx responses.
 * @throws NetworkHaltedError when the backend signals that the agent network must stop.
 */
export async function evaluateConfidence(options: EvaluateConfidenceOptions = {}): Promise<AgentLoopResult> {
  const guard = resolveActiveGuard();
  if (guard === undefined) {
    throw new Error(
      "No active callback handler found. Use getAgentGuard() to create one " +
        "and pass it to graph.invoke() before calling evaluateConfidence(). " +
        "If several runs execute concurrently, wrap the calling code in runWithGuard(guard, fn).",
    );
  }
  return evaluateWithGuard(guard, options);
}
