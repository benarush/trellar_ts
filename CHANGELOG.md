# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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

[Unreleased]: https://github.com/benarush/AITL/compare/v0.6.1...HEAD
[0.6.1]: https://github.com/benarush/AITL/releases/tag/v0.6.1
