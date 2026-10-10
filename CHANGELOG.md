# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.7.0] - 2026-10-10

### Added
- Strands: registering the Trellar agent on the Graph/Swarm now binds every node's
  Agent automatically, so `plugins: [trellarAgent]` on each Agent is no longer needed
  (still harmless, and still required for a standalone Agent). Registration is
  idempotent: nothing is recorded twice.

### Changed
- `trellarStrandsSingleCall` and `trellarLangchainSingleCall` now default to `ObservabilityMode.ALWAYS` (was `NONE`): a single call cannot be evaluated manually, so with no mode it used to evaluate nothing. Pass `ObservabilityMode.NONE` to keep the old behavior.
- The public factory functions are renamed to product-friendly names (matching
  the Python package):
  `getAgentGuard` -> `trellarLangchainAgent`,
  `getSingleCallGuard` -> `trellarLangchainSingleCall`,
  `getStrandsGuard` -> `trellarStrandsAgent`,
  `getStrandsSingleCallGuard` -> `trellarStrandsSingleCall`.
  `runWithGuard` is now `runWithTrellarAgent` and the `GuardState` type is now
  `TrellarAgentState`. Behavior and the payload sent to the back-end are unchanged.
- Callback classes renamed accordingly (`AgentGuardCallback` ->
  `LangchainAgentCallback`, `SingleCallGuardCallback` ->
  `LangchainSingleCallCallback`, `StrandsGuardCallback` ->
  `StrandsAgentCallback`, `StrandsSingleCallGuardCallback` ->
  `StrandsSingleCallCallback`).

### Deprecated
- `getAgentGuard`, `getSingleCallGuard`, `getStrandsGuard`,
  `getStrandsSingleCallGuard` and `runWithGuard` still work but emit a Node
  `DeprecationWarning` and will be removed in a future release.

## [0.6.1] - 2026-10-06

First TypeScript release. A one-to-one port of the Python `trellar` 0.6.1: same
callbacks, same events and text formats, and the same payload sent to
`/agent-gateway/v1/agent-loop`.

### Added
- `evaluateConfidence()`, `ObservabilityMode`, `AgentLoopResult`,
  `NetworkHaltedError`, `TrellarHTTPError`.
- LangChain / LangGraph support (`trellar/langchain`): `getAgentGuard`,
  `getSingleCallGuard`.
- Strands Agents support (`trellar/strands`): `getStrandsGuard`,
  `getStrandsSingleCallGuard`.
- `AsyncLocalStorage`-based guard resolution (the equivalent of Python's
  `ContextVar`) so concurrent runs in one process stay isolated, plus the
  `runWithGuard()` escape hatch.
- ESM + CommonJS builds with type declarations; Node.js 20, 22, 24 and 26
  (Strands: Node.js 22+).

[Unreleased]: https://github.com/benarush/trellar_ts/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/benarush/trellar_ts/releases/tag/v0.7.0
[0.6.1]: https://github.com/benarush/trellar_ts/releases/tag/v0.6.1
