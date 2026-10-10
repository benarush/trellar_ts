import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { resetDeprecationWarnings } from "../src/deprecation.js";
import * as core from "../src/index.js";
import * as lc from "../src/langchain.js";
import * as strands from "../src/strands.js";
import { LangchainAgentCallback } from "../src/callbacks/langchain/langchainCallback.js";
import { LangchainSingleCallCallback } from "../src/callbacks/langchain/singleLangchainCallback.js";

describe("deprecated guard aliases", () => {
  let warn: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    resetDeprecationWarnings();
    warn = vi.spyOn(process, "emitWarning").mockImplementation(() => undefined);
  });

  afterEach(() => {
    warn.mockRestore();
  });

  it("getAgentGuard warns and returns a LangchainAgentCallback", () => {
    const handler = lc.getAgentGuard("legacy-agent");
    expect(handler).toBeInstanceOf(LangchainAgentCallback);
    expect(handler.agentName).toBe("legacy-agent");
    expect(warn).toHaveBeenCalledWith(expect.stringContaining("trellarLangchainAgent"), "DeprecationWarning");
  });

  it("getSingleCallGuard warns and returns a LangchainSingleCallCallback", () => {
    const handler = lc.getSingleCallGuard("legacy-single");
    expect(handler).toBeInstanceOf(LangchainSingleCallCallback);
    expect(warn).toHaveBeenCalledWith(expect.stringContaining("trellarLangchainSingleCall"), "DeprecationWarning");
  });

  it("warns only once per old name", () => {
    lc.getAgentGuard("a");
    lc.getAgentGuard("b");
    expect(warn).toHaveBeenCalledTimes(1);
  });

  it("the Strands aliases forward to the new functions and warn", () => {
    const agent = strands.getStrandsGuard("legacy-strands");
    expect(agent.agentName).toBe("legacy-strands");
    const single = strands.getStrandsSingleCallGuard("legacy-strands-single");
    expect(single.agentName).toBe("legacy-strands-single");
    expect(warn).toHaveBeenCalledTimes(2);
  });

  it("runWithGuard warns and behaves like runWithTrellarAgent", () => {
    const handler = new LangchainAgentCallback({ agentName: "x" });
    expect(core.runWithGuard(handler, () => 42)).toBe(42);
    expect(warn).toHaveBeenCalledWith(expect.stringContaining("runWithTrellarAgent"), "DeprecationWarning");
  });
});
