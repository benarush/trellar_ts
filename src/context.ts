/**
 * Active-guard tracking: the TypeScript counterpart of Python's
 * ``_current_callback`` ContextVar (``trellar/_context.py``).
 *
 * Python sets a ContextVar inside the callback and ``evaluate_confidence()``
 * reads it back. In Node an ``AsyncLocalStorage`` value set inside a callback
 * does not flow back to the code that called ``graph.invoke()``, so
 * ``resolveActiveGuard()`` looks the guard up in this order:
 *
 * 1. the trellar ``AsyncLocalStorage`` scope (``runWithGuard`` / Strands tool
 *    middleware / ``enterWith`` from a root hook),
 * 2. registered framework resolvers (e.g. the LangChain runnable-config
 *    ``AsyncLocalStorage``, which finds the guard in the current run's own
 *    callbacks -- this is what keeps concurrent ``graph.invoke()`` calls apart),
 * 3. the single process-wide active root run, if there is exactly one.
 *
 * Whatever is found must still have an open root run: once the root run ends
 * the guard is released and the lookup fails, exactly like Python.
 */
import { AsyncLocalStorage } from "node:async_hooks";

/** The slice of a guard that the framework-neutral core needs. */
export interface GuardState {
  agentName: string;
  traceId: string | null;
  events: Array<Record<string, unknown>>;
  availableTools: Record<string, unknown[]>;
  /** ``true`` for the single-call guards (payload ``single_call``). */
  isSingleCall?: boolean;
  /** Whether ``evaluateConfidence()`` already succeeded during this run. */
  _evaluated: boolean;
}

export interface GuardScope {
  guard: GuardState;
  /** Innermost open run id (Strands: lets agents-as-tools nest under their tool call). */
  runId?: string;
}

export type GuardResolver = () => GuardState | undefined;

interface TrellarGlobalState {
  als: AsyncLocalStorage<GuardScope>;
  activeRoots: Set<GuardState>;
  resolvers: GuardResolver[];
}

// Kept on globalThis so the ESM and CJS builds (and the per-framework
// sub-entries) all share one registry instead of each bundling their own.
const GLOBAL_KEY = Symbol.for("trellar.context.v1");

function state(): TrellarGlobalState {
  const g = globalThis as unknown as Record<symbol, TrellarGlobalState | undefined>;
  let s = g[GLOBAL_KEY];
  if (!s) {
    s = { als: new AsyncLocalStorage<GuardScope>(), activeRoots: new Set(), resolvers: [] };
    g[GLOBAL_KEY] = s;
  }
  return s;
}

/** Mark ``guard`` as having an open root run (Python: ``_current_callback.set(self)``). */
export function activateGuard(guard: GuardState): void {
  state().activeRoots.add(guard);
}

/** Release ``guard`` (Python: ``_current_callback.set(None)``, guarded by identity). */
export function releaseGuard(guard: GuardState): void {
  state().activeRoots.delete(guard);
}

export function isGuardActive(guard: GuardState): boolean {
  return state().activeRoots.has(guard);
}

/** The scope bound to the current async execution, if any. */
export function currentScope(): GuardScope | undefined {
  return state().als.getStore();
}

/**
 * Run ``fn`` with ``guard`` bound to the current async context.
 *
 * Use this when several guarded runs execute concurrently in one process and
 * ``evaluateConfidence()`` is called from code the framework does not scope
 * for you (e.g. a custom Strands graph node).
 */
export function runWithGuard<T>(guard: GuardState, fn: () => T): T {
  return state().als.run({ guard }, fn);
}

/** Run ``fn`` inside a full scope (guard plus innermost run id). */
export function runInScope<T>(scope: GuardScope, fn: () => T): T {
  return state().als.run(scope, fn);
}

/** Bind ``scope`` to the rest of the current async execution. */
export function enterScope(scope: GuardScope): void {
  state().als.enterWith(scope);
}

/** Register a framework-specific lookup (called on import of the framework entry). */
export function registerGuardResolver(resolver: GuardResolver): void {
  const s = state();
  if (!s.resolvers.includes(resolver)) s.resolvers.push(resolver);
}

/** Resolve the guard ``evaluateConfidence()`` should use, or ``undefined``. */
export function resolveActiveGuard(): GuardState | undefined {
  const s = state();

  const scoped = s.als.getStore()?.guard;
  if (scoped && s.activeRoots.has(scoped)) return scoped;

  for (const resolver of s.resolvers) {
    let found: GuardState | undefined;
    try {
      found = resolver();
    } catch {
      found = undefined;
    }
    if (found && s.activeRoots.has(found)) return found;
  }

  if (s.activeRoots.size === 1) {
    return s.activeRoots.values().next().value as GuardState;
  }
  return undefined;
}
