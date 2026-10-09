// Smoke test for the *built* package: every entry point must load as ESM and as
// CommonJS, and the ESM and CJS builds must share one guard registry.
import { createRequire } from "node:module";
import assert from "node:assert/strict";

const require = createRequire(import.meta.url);
const major = Number(process.versions.node.split(".")[0]);

const esmCore = await import("../dist/index.js");
const cjsCore = require("../dist/index.cjs");
for (const core of [esmCore, cjsCore]) {
  for (const name of ["evaluateConfidence", "ObservabilityMode", "NetworkHaltedError", "TrellarHTTPError", "runWithGuard"]) {
    assert.ok(core[name] !== undefined, `missing export ${name}`);
  }
}

const esmLc = await import("../dist/langchain.js");
const cjsLc = require("../dist/langchain.cjs");
for (const lc of [esmLc, cjsLc]) {
  assert.equal(typeof lc.getAgentGuard, "function");
  assert.equal(typeof lc.getSingleCallGuard, "function");
}

// A guard created by the ESM build is visible to evaluateConfidence() from the CJS build.
const guard = esmLc.getAgentGuard("smoke");
guard.handleChainStart({ name: "root" }, {}, "11111111-1111-4111-8111-111111111111", undefined);
await assert.rejects(
  cjsCore.evaluateConfidence({ apiKey: undefined }),
  /API key|api key|TRELLAR_API_KEY/,
  "the CJS build should have found the ESM guard and then failed on the missing API key",
);

if (major >= 22) {
  const esmStrands = await import("../dist/strands.js");
  const cjsStrands = require("../dist/strands.cjs");
  for (const s of [esmStrands, cjsStrands]) {
    assert.equal(typeof s.getStrandsGuard, "function");
    assert.equal(typeof s.getStrandsSingleCallGuard, "function");
  }
}

console.log(`smoke ok (node ${process.versions.node})`);
