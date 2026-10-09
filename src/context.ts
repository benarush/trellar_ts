/**
 * Active Trellar agent tracking: the TypeScript counterpart of Python's
 * ``_current_callback`` ContextVar (``trellar/_context.py``).
 *
 * Python sets a ContextVar inside the callback and ``evaluate_confidence()``
 * reads it back. In Node an ``AsyncLocalStorage`` value set inside a callback
 * does not flow back to the code that called ``graph.invoke()``, so
 * ``resolveActiveTrellarAgent()`` looks the Trellar agent up in this order:
 *
 * 1. the trellar ``AsyncLocalStorage`` scope (``runWithTrellarAgent`` / Strands tool
 *    middleware / ``enterWith`` from a root hook),
 * 2. registered framework resolvers (e.g. the LangChain runnable-config
 *    ``AsyncLocalStorage``, which finds the Trellar agent in the current run's own
 *    callbacks -- this is what keeps concurrent ``graph.invoke()`` calls apart),
 * 3. the single process-wide active root run, if there is exactly one.
 *
 * Whatever is found must still have an open root run: once the root run ends
 * the Trellar agent is released and the lookup fails, exactly like Python.
 */
import { AsyncLocalStorage } from "node:async_hooks";
import { deprecatedAlias } from "./deprecation.js";

/** The slice of a Trellar agent that the framework-neutral core needs. */
export interface TrellarAgentState {
  agentName: string;
  traceId: string | null;
  events: Array<Record<string, unknown>>;
  availableTools: Record<string, unknown[]>;
  /** ``true`` for the single-call Trellar agents (payload ``single_call``). */
  isSingleCall?: boolean;
  /** Whether ``evaluateConfidence()`` already succeeded during this run. */
  _evaluated: boolean;
}

export interface TrellarAgentScope {
  trellarAgent: TrellarAgentState;
  /** Innermost open run id (Strands: lets agents-as-tools nest under their tool call). */
  runId?: string;
}

export type TrellarAgentResolver = () => TrellarAgentState | undefined;

interface TrellarGlobalState {
  als: AsyncLocalStorage<TrellarAgentScope>;
  activeRoots: Set<TrellarAgentState>;
  resolvers: TrellarAgentResolver[];
}

// Kept on globalThis so the ESM and CJS builds (and the per-framework
// sub-entries) all share one registry instead of each bundling their own.
const GLOBAL_KEY = Symbol.for("trellar.context.v1");

function state(): TrellarGlobalState {
  const g = globalThis as unknown as Record<symbol, TrellarGlobalState | undefined>;
  let s = g[GLOBAL_KEY];
  if (!s) {
    s = { als: new AsyncLocalStorage<TrellarAgentScope>(), activeRoots: new Set(), resolvers: [] };
    g[GLOBAL_KEY] = s;
  }
  return s;
}

/** Mark ``trellarAgent`` as having an open root run (Python: ``_current_callback.set(self)``). */
export function activateTrellarAgent(trellarAgent: TrellarAgentState): void {
  state().activeRoots.add(trellarAgent);
}

/** Release ``trellarAgent`` (Python: ``_current_callback.set(None)``, guarded by identity). */
export function releaseTrellarAgent(trellarAgent: TrellarAgentState): void {
  state().activeRoots.delete(trellarAgent);
}

export function isTrellarAgentActive(trellarAgent: TrellarAgentState): boolean {
  return state().activeRoots.has(trellarAgent);
}

/** The scope bound to the current async execution, if any. */
export function currentScope(): TrellarAgentScope | undefined {
  return state().als.getStore();
}

/**
 * Run ``fn`` with ``trellarAgent`` bound to the current async context.
 *
 * Use this when several guarded runs execute concurrently in one process and
 * ``evaluateConfidence()`` is called from code the framework does not scope
 * for you (e.g. a custom Strands graph node).
 */
export function runWithTrellarAgent<T>(trellarAgent: TrellarAgentState, fn: () => T): T {
  return state().als.run({ trellarAgent }, fn);
}

/** @deprecated Use {@link TrellarAgentState}. */
export type GuardState = TrellarAgentState;

/** @deprecated Use {@link runWithTrellarAgent}. */
export const runWithGuard = deprecatedAlias("runWithGuard", "runWithTrellarAgent", runWithTrellarAgent);

/** Run ``fn`` inside a full scope (Trellar agent plus innermost run id). */
export function runInScope<T>(scope: TrellarAgentScope, fn: () => T): T {
  return state().als.run(scope, fn);
}

/** Bind ``scope`` to the rest of the current async execution. */
export function enterScope(scope: TrellarAgentScope): void {
  state().als.enterWith(scope);
}

/** Register a framework-specific lookup (called on import of the framework entry). */
export function registerTrellarAgentResolver(resolver: TrellarAgentResolver): void {
  const s = state();
  if (!s.resolvers.includes(resolver)) s.resolvers.push(resolver);
}

/** Resolve the Trellar agent ``evaluateConfidence()`` should use, or ``undefined``. */
export function resolveActiveTrellarAgent(): TrellarAgentState | undefined {
  const s = state();

  const scoped = s.als.getStore()?.trellarAgent;
  if (scoped && s.activeRoots.has(scoped)) return scoped;

  for (const resolver of s.resolvers) {
    let found: TrellarAgentState | undefined;
    try {
      found = resolver();
    } catch {
      found = undefined;
    }
    if (found && s.activeRoots.has(found)) return found;
  }

  if (s.activeRoots.size === 1) {
    return s.activeRoots.values().next().value as TrellarAgentState;
  }
  return undefined;
}
