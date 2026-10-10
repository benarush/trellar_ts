# trellar

[![npm version](https://img.shields.io/npm/v/trellar.svg)](https://www.npmjs.com/package/trellar)
[![Node](https://img.shields.io/node/v/trellar.svg)](https://www.npmjs.com/package/trellar)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/benarush/trellar_ts/actions/workflows/ci.yml/badge.svg)](https://github.com/benarush/trellar_ts/actions/workflows/ci.yml)

The TypeScript / Node.js SDK for **Trellar**, the trust layer for autonomous multi-agent systems. Trellar is an AI governance platform that sits above your existing agent frameworks (LangChain, LangGraph, Strands Agents and more) and gives your team visibility and control over what your agents do in production, so you can extend their autonomy with confidence.

Connecting takes one callback on your agent run. Trellar monitors the run outside the execution path, so it adds no latency to your agents, and it evaluates the run against what each agent is meant to do. Trellar can evaluate each run automatically when it finishes. For finer control, call `evaluateConfidence()` wherever you want a score, either to gate the next step or just to record it. Context, trace ID, and agent name are picked up automatically — no manual wiring.

This package has the same features as the Python [`trellar`](../README.md) package.

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

Mirrors the Python layout (each framework's Trellar agent lives next to it):

```ts
import { evaluateConfidence, ObservabilityMode, runWithTrellarAgent } from "trellar";
import { trellarLangchainAgent, trellarLangchainSingleCall } from "trellar/langchain";
import { trellarStrandsAgent, trellarStrandsSingleCall } from "trellar/strands";
```

---

## Quick Start (LangChain / LangGraph)

Start by observing your agent. Set your API key, add the Trellar callback to your run, and Trellar evaluates the run automatically when it finishes. You don't need to call `evaluateConfidence()` yourself.

```bash
export TRELLAR_API_KEY=your-api-key
```

```ts
import { createAgent, tool } from "langchain";
import { z } from "zod";
import { ObservabilityMode } from "trellar";
import { trellarLangchainAgent } from "trellar/langchain";

const searchDocs = tool(async () => "The office opens at 9am.", {
  name: "search_docs",
  description: "Search the internal docs.",
  schema: z.object({ query: z.string() }),
});

const graph = createAgent({ model, tools: [searchDocs] });

// agentName must be a stable, unique name for this agent graph — the
// backend uses it to track the graph's network profile across runs.
const trellarAgent = trellarLangchainAgent("research-agent", ObservabilityMode.ALWAYS);

// The run is captured and evaluated when invoke() finishes.
// Context, trace_id, and agent_name are picked up automatically.
await graph.invoke(
  { messages: [{ role: "user", content: "What time does the office open?" }] },
  { callbacks: [trellarAgent] },
);
```

The run is sent to Trellar when `invoke()` returns, and the result appears in your dashboard at [trellar.io](https://trellar.io).

> **Evaluation needs some context.** Trellar scores the run from the events it captured, so the run should include at least one agent call or tool call. A run with nothing in it gives the evaluation too little to work with.

Want to read the score in your code, or stop the graph when it's low? See [Where to call `evaluateConfidence`](#where-to-call-evaluateconfidence).

> **Note:** forward the node `config` into nested runnables (`model.invoke(msgs, config)`) so child runs are reported to the Trellar agent — the same rule as for any LangChain callback.

---

## Strands Agents

```ts
import { Agent, Graph } from "@strands-agents/sdk"; // Graph also lives in "@strands-agents/sdk/multiagent"
import { ObservabilityMode } from "trellar";
import { trellarStrandsAgent } from "trellar/strands";

// ALWAYS: the run is evaluated automatically when it finishes (no manual call needed).
const trellarAgent = trellarStrandsAgent("research-agent", ObservabilityMode.ALWAYS);

// Give every agent a stable name and a unique id.
const searcher = new Agent({ id: "searcher", name: "searcher" });
const reporter = new Agent({ id: "reporter", name: "reporter" });

// Register the Trellar agent as a plugin on the Graph/Swarm: the whole run is one trace, and
// every node's Agent is bound automatically (no need for `plugins: [trellarAgent]` on each one).
const graph = new Graph({
  nodes: [searcher, reporter],
  edges: [["searcher", "reporter"]],
  plugins: [trellarAgent],
});

await graph.invoke("Find me something to report on");
```

A standalone Agent (no Graph/Swarm) still needs `new Agent({ plugins: [trellarAgent] })`. Passing the plugin to a graph's Agents as well is harmless.

With `ObservabilityMode.ALWAYS` (as above) nothing else is needed. Without an `ObservabilityMode`, nothing is evaluated unless you call `evaluateConfidence()` from inside the run (a graph node or a tool), exactly as with LangChain. See `orchestrations_examples/our_lab_with_aviran/car_stocks_buy_mcp_celery.py/ts/` for full examples.

> Strands graph nodes take their id from `agent.id`, which defaults to `"agent"` — give each agent in a graph a unique `id`.

### `trellarStrandsSingleCall`

For one Agent called once (no Graph/Swarm). The run is over when the call returns, so a manual `evaluateConfidence()` isn't possible: the run is evaluated automatically (`ObservabilityMode.ALWAYS` is the default here; pass `ObservabilityMode.NONE` to only record). Read the result off the Trellar agent:

```ts
import { Agent } from "@strands-agents/sdk";
import { trellarStrandsSingleCall } from "trellar/strands";

const trellarAgent = trellarStrandsSingleCall("faq-agent");
const agent = new Agent({ name: "faq_agent", plugins: [trellarAgent] });
await agent.invoke("What time does the office open?");

const result = trellarAgent.trellarEvaluateResult; // AgentLoopResult, or null
const error = trellarAgent.trellarEvaluateError;   // the error, if the auto-triggered call failed
```

Requests are marked `single_call: true` in the payload. Not covered: `agent.structuredOutput()` and calling a Strands `Model` directly.

---

## Parallel runs

Several runs can execute in parallel in the same process. Give each its own Trellar agent, and each `evaluateConfidence()` call evaluates its own run:

```ts
await Promise.all([
  graph.invoke(inputA, { callbacks: [trellarLangchainAgent("agent-a")] }),
  graph.invoke(inputB, { callbacks: [trellarLangchainAgent("agent-b")] }),
]);
```

If you call `evaluateConfidence()` from code you started yourself (for example a background task spawned while several runs are open), wrap it so Trellar knows which run you mean:

```ts
import { runWithTrellarAgent } from "trellar";

await runWithTrellarAgent(trellarAgent, () => evaluateConfidence());
```

Call `evaluateConfidence()` while the run is still in progress. Once the root `invoke()` returns, the run is closed and a later call throws.

---

## Where to call `evaluateConfidence`

Call it from a graph node, at the point in the run you want scored, while the run is still in progress. The payload is the events captured **so far** — later nodes are not included.

There are two ways to get a score:

### 1. Gate — validate before the graph continues

Put the call on an edge you do not want the graph to cross until Trellar has scored the run. Use `result.score` / `result.explanation` to decide whether to proceed or stop.

```ts
async function confidenceGate(state: State) {
  const result = await evaluateConfidence();
  if (result.score < 7) return { halt: true, reason: result.explanation };
  return { halt: false };
}
```

### 2. Observe — record a score, do not restrict the graph

If you only want the run scored and recorded, you don't need a node or a manual call. Set an `ObservabilityMode` when you create the Trellar agent, and Trellar evaluates the run when it finishes. The graph is never affected.

```ts
import { ObservabilityMode } from "trellar";
import { trellarLangchainAgent } from "trellar/langchain";

const trellarAgent = trellarLangchainAgent("research-agent", ObservabilityMode.ALWAYS);
await graph.invoke(inputs, { callbacks: [trellarAgent] });
```

Use `ObservabilityMode.IF_NOT_EVALUATED` to combine both ways: gate nodes score the run where you need them, and any run that no node scored is still evaluated when it finishes. See [`ObservabilityMode`](#observabilitymode) for the full list of values.

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

> **Renamed in this release.** The `get*Guard` functions are now `trellar<Framework>Agent` / `trellar<Framework>SingleCall`, and `runWithGuard` is now `runWithTrellarAgent`. The old names still work but emit a Node `DeprecationWarning` (once per name) and will be removed in a future release.
>
> | Deprecated | Use instead |
> |---|---|
> | `getAgentGuard` | `trellarLangchainAgent` |
> | `getSingleCallGuard` | `trellarLangchainSingleCall` |
> | `getStrandsGuard` | `trellarStrandsAgent` |
> | `getStrandsSingleCallGuard` | `trellarStrandsSingleCall` |
> | `runWithGuard` | `runWithTrellarAgent` |

### `trellarLangchainAgent` (`trellar/langchain`)

```ts
trellarLangchainAgent(agentName: string, observabilityMode?: ObservabilityMode): LangchainAgentCallback
```

| Parameter | Type | Description |
|---|---|---|
| `agentName` | `string` | Stable, unique name identifying this agent graph (e.g. `"research-agent"`) |
| `observabilityMode` | `ObservabilityMode` | Controls whether `evaluateConfidence()` is auto-triggered when the graph run finishes. Default `ObservabilityMode.NONE`. |

Returns a LangChain callback handler bound to `agentName`. Pass it to `graph.invoke(input, { callbacks: [trellarAgent] })`.

**Throws:** `Error` if `agentName` is empty or blank.

#### `ObservabilityMode`

Controls whether the Trellar agent automatically calls `evaluateConfidence()` for you when the graph run finishes (the root `graph.invoke()` call completes).

| Value | Behavior |
|---|---|
| `ObservabilityMode.NONE` (`"none"`) | Never auto-call. Default. |
| `ObservabilityMode.ALWAYS` (`"always"`) | Always call `evaluateConfidence()` when the run finishes. |
| `ObservabilityMode.IF_NOT_EVALUATED` (`"if_not_evaluated"`) | Call it when the run finishes only if it was not already successfully called earlier in the run (e.g. from a gate node). |

```ts
import { ObservabilityMode } from "trellar";
import { trellarLangchainAgent } from "trellar/langchain";

const trellarAgent = trellarLangchainAgent("research-agent", ObservabilityMode.IF_NOT_EVALUATED);
await graph.invoke(inputs, { callbacks: [trellarAgent] });
// evaluateConfidence() has already run automatically if no node called it.
```

Auto-triggered calls never throw: any error (missing API key, HTTP error, `NetworkHaltedError`, etc.) is caught and logged instead of propagating out of `graph.invoke()`. A manual call to `evaluateConfidence()` still throws normally.

Requests triggered this way are marked in the payload with `observability_call: true` (`false` for a normal, manually-invoked call).

### `trellarLangchainSingleCall` (`trellar/langchain`)

```ts
trellarLangchainSingleCall(agentName: string, observabilityMode?: ObservabilityMode): LangchainSingleCallCallback
```

Use this instead of `trellarLangchainAgent` when you are calling a chat model directly (`llm.invoke(...)`) with no wrapping LangGraph/chain. Agents built with `createAgent` are already compiled graphs under the hood, so they work with `trellarLangchainAgent` as usual.

A bare `llm.invoke()` call has no node to call `evaluateConfidence()` from mid-run, and the Trellar agent is released as soon as the call finishes — so a manual call is never supported here. The run is evaluated automatically (`ObservabilityMode.ALWAYS` is the default here; pass `ObservabilityMode.NONE` to only record). Read the result back from the Trellar agent:

```ts
import { trellarLangchainSingleCall } from "trellar/langchain";

const trellarAgent = trellarLangchainSingleCall("single-llm-call");
await llm.invoke(messages, { callbacks: [trellarAgent] });

const result = trellarAgent.trellarEvaluateResult; // AgentLoopResult, or null if not yet evaluated
const error = trellarAgent.trellarEvaluateError;   // the error, if the auto-triggered call failed
```

Requests made through this Trellar agent are marked in the payload with `single_call: true` (`false` for `trellarLangchainAgent`).

### `evaluateConfidence`

```ts
evaluateConfidence(options?: { apiKey?: string; timeout?: number }): Promise<AgentLoopResult>
```

| Option | Type | Description |
|---|---|---|
| `apiKey` | `string` | Bearer token. Falls back to `TRELLAR_API_KEY` |
| `timeout` | `number` | HTTP request timeout in seconds (default `30`) |

`context`, `trace_id`, and `agent_name` are resolved automatically from the active Trellar agent — there is no way to pass them manually. Requests always go to `https://api.trellar.io`; callers cannot redirect them.

**Rejects with:**
- `Error` — if no active Trellar agent is found, its trace id cannot be resolved, or the API key is missing
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
