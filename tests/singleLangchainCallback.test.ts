import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { FakeListChatModel } from "@langchain/core/utils/testing";
import { afterEach, describe, expect, it, vi } from "vitest";

import * as agentLoop from "../src/agentLoop.js";
import { LangchainSingleCallCallback } from "../src/callbacks/langchain/singleLangchainCallback.js";
import { isTrellarAgentActive } from "../src/context.js";
import { trellarLangchainSingleCall, ObservabilityMode } from "../src/langchain.js";
import { uuid, makeAiMessage, makeLlmResult } from "./factories.js";
import { assertValidAgentLoopRequest, mockFetch, useCleanEnv } from "./helpers.js";

useCleanEnv();
afterEach(() => vi.restoreAllMocks());

const OK = { explanation: "e", score: 7, decisionIdentifier: "d", shouldStopNetwork: false };
const mk = (mode: ObservabilityMode = ObservabilityMode.NONE) => trellarLangchainSingleCall("single", mode);

describe("trellarLangchainSingleCall", () => {
  it("returns a single-call Trellar agent flagged isSingleCall", () => {
    const g = mk();
    expect(g).toBeInstanceOf(LangchainSingleCallCallback);
    expect(g.isSingleCall).toBe(true);
    expect(g.trellarEvaluateResult).toBeNull();
    expect(g.trellarEvaluateError).toBeNull();
  });
  it("rejects blank names", () => expect(() => trellarLangchainSingleCall(" ")).toThrow(/agent_name is required/));
});

describe("run boundary on the LLM events", () => {
  it("chat model start sets the trace id and activates the Trellar agent", () => {
    const g = mk();
    const id = uuid();
    g.handleChatModelStart({ name: "X" }, [[new HumanMessage("hi")]], id, undefined);
    expect(g.traceId).toBe(id);
    expect(isTrellarAgentActive(g)).toBe(true);
  });

  it("llm start sets the trace id and activates the Trellar agent", () => {
    const g = mk();
    const id = uuid();
    g.handleLLMStart({ name: "X" }, ["p"], id, undefined);
    expect(g.traceId).toBe(id);
    expect(isTrellarAgentActive(g)).toBe(true);
  });

  it("a non-root start does not touch the trace id", () => {
    const g = mk();
    const root = uuid();
    g.handleChatModelStart({ name: "X" }, [[new HumanMessage("hi")]], root, undefined);
    g.handleChatModelStart({ name: "X" }, [[new HumanMessage("again")]], uuid(), root);
    expect(g.traceId).toBe(root);
    expect(g.events).toHaveLength(2);
  });

  it("a second root call resets state from the first", () => {
    const g = mk();
    g.handleChatModelStart({ name: "X" }, [[new HumanMessage("one")]], uuid(), undefined);
    g.trellarEvaluateResult = OK;
    const second = uuid();
    g.handleChatModelStart({ name: "X" }, [[new HumanMessage("two")]], second, undefined);
    expect(g.events).toHaveLength(1);
    expect(g.traceId).toBe(second);
    expect(g.trellarEvaluateResult).toBeNull();
  });

  it("llm end records the response and releases the Trellar agent", async () => {
    const g = mk();
    const id = uuid();
    g.handleChatModelStart({ name: "X" }, [[new HumanMessage("hi")]], id, undefined);
    await g.handleLLMEnd(makeLlmResult({ message: makeAiMessage("answer") }), id, undefined);
    expect(g.events.at(-1)!["output"]).toEqual({ response: "answer" });
    expect(isTrellarAgentActive(g)).toBe(false);
  });

  it("a root llm error releases the Trellar agent; a non-root one does not", () => {
    const g = mk();
    const id = uuid();
    g.handleChatModelStart({ name: "X" }, [[new HumanMessage("hi")]], id, undefined);
    g.handleLLMError(new Error("child"), uuid(), id);
    expect(isTrellarAgentActive(g)).toBe(true);
    g.handleLLMError(new Error("boom"), id, undefined);
    expect(isTrellarAgentActive(g)).toBe(false);
    expect(g.events.at(-1)).toMatchObject({ event: "on_llm_error", error: "boom" });
  });
});

describe("auto-evaluation", () => {
  async function run(mode: ObservabilityMode, evaluated = false) {
    const evaluate = vi.spyOn(agentLoop, "evaluateWithTrellarAgent").mockResolvedValue(OK);
    const g = mk(mode);
    const id = uuid();
    g.handleChatModelStart({ name: "X" }, [[new HumanMessage("hi")]], id, undefined);
    g._evaluated = evaluated;
    await g.handleLLMEnd(makeLlmResult({ message: makeAiMessage("a") }), id, undefined);
    return { evaluate, g };
  }

  it("NONE never calls", async () => expect((await run("none")).evaluate).not.toHaveBeenCalled());
  it("ALWAYS calls and stores the result", async () => {
    const { evaluate, g } = await run("always");
    expect(evaluate).toHaveBeenCalledWith(g, { _observabilityCall: true });
    expect(g.trellarEvaluateResult).toEqual(OK);
    expect(g.trellarEvaluateError).toBeNull();
  });
  it("ALWAYS calls even if already evaluated", async () =>
    expect((await run("always", true)).evaluate).toHaveBeenCalledTimes(1));
  it("IF_NOT_EVALUATED calls only when not evaluated", async () => {
    expect((await run("if_not_evaluated", false)).evaluate).toHaveBeenCalledTimes(1);
    expect((await run("if_not_evaluated", true)).evaluate).not.toHaveBeenCalled();
  });

  it("the factory defaults to ALWAYS when no mode is passed", async () => {
    // A bare call can't be evaluated manually, so no mode means evaluate.
    const evaluate = vi.spyOn(agentLoop, "evaluateWithTrellarAgent").mockResolvedValue(OK);
    const g = trellarLangchainSingleCall("single");
    const id = uuid();
    g.handleChatModelStart({ name: "X" }, [[new HumanMessage("hi")]], id, undefined);
    await g.handleLLMEnd(makeLlmResult({ message: makeAiMessage("a") }), id, undefined);
    expect(evaluate).toHaveBeenCalledTimes(1);
    expect(g.trellarEvaluateResult).toEqual(OK);
  });

  it("stores a failure as trellarEvaluateError and never throws", async () => {
    const boom = new Error("backend down");
    vi.spyOn(agentLoop, "evaluateWithTrellarAgent").mockRejectedValue(boom);
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const g = mk("always");
    const id = uuid();
    g.handleChatModelStart({ name: "X" }, [[new HumanMessage("hi")]], id, undefined);
    await expect(g.handleLLMEnd(makeLlmResult({ message: makeAiMessage("a") }), id, undefined)).resolves.toBeUndefined();
    expect(g.trellarEvaluateResult).toBeNull();
    expect(g.trellarEvaluateError).toBe(boom);
  });
});

describe("a real bare llm.invoke() call", () => {
  it("sends one observability request marked single_call and exposes the result", async () => {
    const calls = mockFetch();
    process.env["TRELLAR_API_KEY"] = "k";
    const llm = new FakeListChatModel({ responses: ["the answer"] });
    const trellarAgent = mk("always");
    const reply = await llm.invoke([new SystemMessage("be brief"), new HumanMessage("question")], {
      callbacks: [trellarAgent],
    });

    expect(reply.content).toBe("the answer");
    expect(calls).toHaveLength(1);
    const body = calls[0]!.body;
    expect(body).toMatchObject({ single_call: true, observability_call: true, agent_name: "single" });
    expect(body.context.map((e: any) => e.event)).toEqual(["on_chat_model_start", "on_llm_end"]);
    expect(body.context[0].input).toEqual({ system: "be brief", human: "question" });
    expect(body.context[1].output).toEqual({ response: "the answer" });
    assertValidAgentLoopRequest(body);
    expect(trellarAgent.trellarEvaluateResult?.score).toBe(8);
    expect(isTrellarAgentActive(trellarAgent)).toBe(false);
  });
});
