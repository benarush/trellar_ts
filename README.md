# trellar

[![npm version](https://img.shields.io/npm/v/trellar.svg)](https://www.npmjs.com/package/trellar)
[![Node](https://img.shields.io/node/v/trellar.svg)](https://www.npmjs.com/package/trellar)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/benarush/AITL/actions/workflows/ci.yml/badge.svg)](https://github.com/benarush/AITL/actions/workflows/ci.yml)

A lightweight TypeScript / Node.js client for the **Trellar** confidence evaluation API. Attach a callback to your LangChain / LangGraph or Strands Agents run, then call `evaluateConfidence()` when you want a score. Context, trace ID, and agent name are picked up automatically — no manual wiring.

This is a one-to-one port of the Python [`trellar`](../README.md) package: same callbacks, same events, same payload sent to the back-end.

This library cannot be used without an API key from [trellar.io](https://trellar.io).

---

## Create an account and API key

[trellar.io](https://trellar.io) is the only place that issues API keys for this library. Create an account there, then generate an API key from the dashboard. Without that key, `evaluateConfidence()` cannot authenticate and the client will not work.

Then pass the key into the SDK (see [Environment Variables](#environment-variables)):

- `evaluateConfidence({ apiKey: "..." })`, or
- `TRELLAR_API_KEY` in the environment

---

## Installation

```bash
npm install trellar @langchain/core @langchain/langgraph     # LangChain / LangGraph
npm install trellar @strands-agents/sdk                       # Strands Agents
```

`@langchain/core` and `@strands-agents/sdk` are **optional peer dependencies** — install only the framework you use.

| | Node.js |
|---|---|
| `trellar`, `trellar/langchain` | 20, 22, 24, 26 (tested in CI) |
| `trellar/strands` | 22+ (the Strands Agents SDK requires Node 22 or newer) |

The package ships ESM and CommonJS builds with type declarations.

---

## Imports

Mirrors the Python layout (guards live next to their framework):

```ts
import { evaluateConfidence, ObservabilityMode, runWithGuard } from "trellar";
import { getAgentGuard, getSingleCallGuard } from "trellar/langchain";
import { getStrandsGuard, getStrandsSingleCallGuard } from "trellar/strands";
```

---

## Quick Start (LangChain / LangGraph)

```ts
import { evaluateConfidence } from "trellar";
import { getAgentGuard } from "trellar/langchain";

// agentName must be a stable, unique name for this agent graph — the
// backend uses it to track the graph's network profile across runs.
const guard = getAgentGuard("research-agent");

// call evaluateConfidence() from a node, while the run is still in progress —
// context, trace_id, and agent_name are picked up from the guard automatically
async function reportConfidence(state: State) {
  const result = await evaluateConfidence();
  console.log(result.score);       // number, 1-10
  console.log(result.explanation); // string, human-readable reasoning
  return state;
}

await graph.invoke(inputs, { callbacks: [guard] });
```

> **Note:** forward the node `config` into nested runnables (`model.invoke(msgs, config)`) so child runs are reported to the guard — the same rule as for any LangChain callback.

---

## Strands Agents

```ts
import { Agent, Graph } from "@strands-agents/sdk"; // Graph also lives in "@strands-agents/sdk/multiagent"
import { evaluateConfidence } from "trellar";
import { getStrandsGuard } from "trellar/strands";

const guard = getStrandsGuard("research-agent");

// Give every agent a stable name and a unique id, and register the guard as a plugin on each agent...
const searcher = new Agent({ id: "searcher", name: "searcher", plugins: [guard] });
const reporter = new Agent({ id: "reporter", name: "reporter", plugins: [guard] });

// ...and on the Graph/Swarm, so the whole run is one trace.
const graph = new Graph({
  nodes: [searcher, reporter],
  edges: [["searcher", "reporter"]],
  plugins: [guard],
});

await graph.invoke("Find me something to report on");
```

Call `evaluateConfidence()` from inside the run (a graph node or a tool), exactly as with LangChain. `ObservabilityMode` works the same way. See `orchestrations_examples/our_lab_with_aviran/car_stocks_buy_mcp_celery.py/ts/` for full examples.

> Strands graph nodes take their id from `agent.id`, which defaults to `"agent"` — give each agent in a graph a unique `id`.

### `getStrandsSingleCallGuard`

For one Agent called once (no Graph/Swarm). The run is over when the call returns, so use `ObservabilityMode.ALWAYS` (or `IF_NOT_EVALUATED`) and read the result off the guard:

```ts
import { Agent } from "@strands-agents/sdk";
import { ObservabilityMode } from "trellar";
import { getStrandsSingleCallGuard } from "trellar/strands";

const guard = getStrandsSingleCallGuard("faq-agent", ObservabilityMode.ALWAYS);
const agent = new Agent({ name: "faq_agent", plugins: [guard] });
await agent.invoke("What time does the office open?");

const result = guard.trellarEvaluateResult; // AgentLoopResult, or null
const error = guard.trellarEvaluateError;   // the error, if the auto-triggered call failed
```

Requests are marked `single_call: true` in the payload. Not covered: `agent.structuredOutput()` and calling a Strands `Model` directly — Strands fires no model-call hooks for them.

---

## Concurrency (the `ContextVar` equivalent)

Python resolves the active guard with a `ContextVar`. In Node the equivalent is `AsyncLocalStorage`, and `evaluateConfidence()` resolves the guard in this order:

1. an explicit `runWithGuard(guard, fn)` scope (and the scopes trellar binds itself around every Strands tool call and graph node),
2. the guard attached to the current LangChain run (read from the run's own callbacks, so concurrent `graph.invoke()` calls never see each other),
3. the single run that is currently open in the process.

So several runs can execute in parallel in the same process — each `evaluateConfidence()` call evaluates its own run:

```ts
await Promise.all([
  graph.invoke(inputA, { callbacks: [getAgentGuard("agent-a")] }),
  graph.invoke(inputB, { callbacks: [getAgentGuard("agent-b")] }),
]);
```

When you call `evaluateConfidence()` from code the framework does not scope for you (for example a callback you spawn yourself while several runs are open), wrap it:

```ts
import { runWithGuard } from "trellar";

await runWithGuard(guard, () => evaluateConfidence());
```

As in Python, the guard is released as soon as the root run ends, so calling `evaluateConfidence()` after `invoke()` returns throws.

---

## Where to call `evaluateConfidence`

Call it from a graph node, at the point in the run you want scored, while the run is still in progress. The payload is the events captured **so far** — later nodes are not included.

There are two ways to use the result:

### 1. Gate — validate before the graph continues

Put the call on an edge you do not want the graph to cross until Trellar has scored the run. Use `result.score` / `result.explanation` to decide whether to proceed or stop.

```ts
async function confidenceGate(state: State) {
  const result = await evaluateConfidence();
  if (result.score < 7) return { halt: true, reason: result.explanation };
  return { halt: false };
}
```

### 2. Observe — send a validation, do not restrict the graph

Put the call in any node where you want a score recorded. Store or log `result` if you want it; do not branch on it. The graph continues either way.

```ts
async function reportConfidence() {
  const result = await evaluateConfidence();
  return { confidenceScore: result.score, confidenceExplanation: result.explanation };
}
```

---

## Environment Variables

The SDK always talks to the managed Trellar backend at `https://api.trellar.io` — this is fixed and cannot be overridden via an environment variable or function argument.

| Variable | Description | Default |
|---|---|---|
| `TRELLAR_API_KEY` | Bearer token for authentication | *(required)* |

```bash
export TRELLAR_API_KEY=your-api-key
```

```ts
const result = await evaluateConfidence(); // apiKey read from the env var
```

---

## API Reference

### `getAgentGuard` (`trellar/langchain`)

```ts
getAgentGuard(agentName: string, observabilityMode?: ObservabilityMode): AgentGuardCallback
```

| Parameter | Type | Description |
|---|---|---|
| `agentName` | `string` | Stable, unique name identifying this agent graph (e.g. `"research-agent"`) |
| `observabilityMode` | `ObservabilityMode` | Controls whether `evaluateConfidence()` is auto-triggered when the graph run finishes. Default `ObservabilityMode.NONE`. |

Returns a LangChain callback handler bound to `agentName`. Pass it to `graph.invoke(input, { callbacks: [guard] })`.

**Throws:** `Error` if `agentName` is empty or blank.

#### `ObservabilityMode`

Controls whether the guard automatically calls `evaluateConfidence()` for you when the graph run finishes (the root `graph.invoke()` call completes).

| Value | Behavior |
|---|---|
| `ObservabilityMode.NONE` (`"none"`) | Never auto-call. Default. |
| `ObservabilityMode.ALWAYS` (`"always"`) | Always call `evaluateConfidence()` when the run finishes. |
| `ObservabilityMode.IF_NOT_EVALUATED` (`"if_not_evaluated"`) | Call it when the run finishes only if it was not already successfully called earlier in the run (e.g. from a gate node). |

```ts
import { ObservabilityMode } from "trellar";
import { getAgentGuard } from "trellar/langchain";

const guard = getAgentGuard("research-agent", ObservabilityMode.IF_NOT_EVALUATED);
await graph.invoke(inputs, { callbacks: [guard] });
// evaluateConfidence() has already run automatically if no node called it.
```

Auto-triggered calls never throw: any error (missing API key, HTTP error, `NetworkHaltedError`, etc.) is caught and logged instead of propagating out of `graph.invoke()`. A manual call to `evaluateConfidence()` still throws normally.

Requests triggered this way are marked in the payload with `observability_call: true` (`false` for a normal, manually-invoked call).

### `getSingleCallGuard` (`trellar/langchain`)

```ts
getSingleCallGuard(agentName: string, observabilityMode?: ObservabilityMode): SingleCallGuardCallback
```

Use this instead of `getAgentGuard` when you are calling a chat model directly (`llm.invoke(...)`) with no wrapping LangGraph/chain. Agents built with `createAgent` are already compiled graphs under the hood, so they work with `getAgentGuard` as usual.

A bare `llm.invoke()` call has no node to call `evaluateConfidence()` from mid-run, and the guard is released as soon as the call finishes — so a manual call is never supported here. Use `ObservabilityMode.ALWAYS` (or `IF_NOT_EVALUATED`), then read the result back from the guard:

```ts
import { ObservabilityMode } from "trellar";
import { getSingleCallGuard } from "trellar/langchain";

const guard = getSingleCallGuard("single-llm-call", ObservabilityMode.ALWAYS);
await llm.invoke(messages, { callbacks: [guard] });

const result = guard.trellarEvaluateResult; // AgentLoopResult, or null if not yet evaluated
const error = guard.trellarEvaluateError;   // the error, if the auto-triggered call failed
```

Requests made through this guard are marked in the payload with `single_call: true` (`false` for `getAgentGuard`).

### `evaluateConfidence`

```ts
evaluateConfidence(options?: { apiKey?: string; timeout?: number }): Promise<AgentLoopResult>
```

| Option | Type | Description |
|---|---|---|
| `apiKey` | `string` | Bearer token. Falls back to `TRELLAR_API_KEY` |
| `timeout` | `number` | HTTP request timeout in seconds (default `30`) |

`context`, `trace_id`, and `agent_name` are resolved automatically from the active guard — there is no way to pass them manually. Requests always go to `https://api.trellar.io`; callers cannot redirect them.

**Rejects with:**
- `Error` — if no active guard is found, its trace id cannot be resolved, or the API key is missing
- `TrellarHTTPError` — on non-2xx HTTP responses (`status`, `statusText`, `url`, `body`)
- `NetworkHaltedError` — when the backend signals the run should stop (`shouldStopNetwork`)

### `AgentLoopResult`

A frozen object with:

| Field | Type | Description |
|---|---|---|
| `score` | `number` | Confidence score from 1 (low) to 10 (high) |
| `explanation` | `string` | Human-readable explanation of the score |
| `decisionIdentifier` | `string` | Unique ID for this evaluation, for cross-referencing with the backend |
| `shouldStopNetwork` | `boolean` | Whether the backend signaled the run should halt (see `NetworkHaltedError`) |

---

## Development

```bash
npm install
npm run typecheck
npm run lint
npm test
npm run build
```

## License

MIT — see [LICENSE](LICENSE) for details.
