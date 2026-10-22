/** Tests for ``input_tokens`` / ``output_tokens`` on ``on_llm_end`` events (LangChain + Strands). */
import { AIMessage } from "@langchain/core/messages";
import { Agent } from "@strands-agents/sdk";
import { describe, expect, it } from "vitest";

import { LangchainAgentCallback } from "../src/callbacks/langchain/langchainCallback.js";
import { trellarStrandsAgent } from "../src/strands.js";
import { assertValidAgentLoopRequest } from "./helpers.js";
import { makeAiMessage, makeLlmResult, uuid } from "./factories.js";
import { FakeModel } from "./strandsFactories.js";

/** A handler with a started root chain; returns it plus a function that ends one model call. */
function setup() {
  const h = new LangchainAgentCallback({ agentName: "test-agent" });
  h.handleChainStart({ name: "graph" }, { messages: [] }, uuid(), undefined, [], {}, undefined, "graph");
  const llmEnd = (result: any): Record<string, any> => {
    h.handleLLMEnd(result, uuid(), undefined);
    return h.events[h.events.length - 1]!;
  };
  return { h, llmEnd };
}

describe("input_tokens / output_tokens", () => {
  it("langchain: counts come from usage_metadata", () => {
    const { llmEnd } = setup();
    const message = new AIMessage({
      content: "hi",
      usage_metadata: { input_tokens: 12, output_tokens: 5, total_tokens: 17 },
    });
    const event = llmEnd(makeLlmResult({ message }));

    expect(event["input_tokens"]).toBe(12);
    expect(event["output_tokens"]).toBe(5);
  });

  it.each([
    ["token_usage", { prompt_tokens: 7, completion_tokens: 3, total_tokens: 10 }], // OpenAI style
    ["tokenUsage", { promptTokens: 7, completionTokens: 3 }], // OpenAI JS style
    ["usage", { input_tokens: 7, output_tokens: 3 }], // Anthropic style
  ])("langchain: falls back to llmOutput.%s", (key, usage) => {
    const { llmEnd } = setup();
    const event = llmEnd(makeLlmResult({ message: makeAiMessage("hi"), llmOutput: { [key]: usage } }));

    expect([event["input_tokens"], event["output_tokens"]]).toEqual([7, 3]);
    expect(event["token_usage"]).toEqual(usage); // the existing field is unchanged
  });

  it("langchain: no usage gives null", () => {
    const { llmEnd } = setup();
    const event = llmEnd(makeLlmResult({ message: makeAiMessage("hi") }));

    expect(event["input_tokens"]).toBeNull();
    expect(event["output_tokens"]).toBeNull();
  });

  it("strands: counts come from the message usage", async () => {
    const trellarAgent = trellarStrandsAgent("t");
    // FakeModel reports inputTokens=1, outputTokens=2 on every call
    const agent = new Agent({
      id: "calc-tokens",
      name: "calc",
      model: new FakeModel([{ text: "hi" }]),
      plugins: [trellarAgent],
      printer: false,
    });
    await agent.invoke("hello");

    const ends = trellarAgent.events.filter((e) => e["event"] === "on_llm_end");
    expect(ends).toHaveLength(1);
    expect(ends[0]).toMatchObject({ input_tokens: 1, output_tokens: 2 });
    assertValidAgentLoopRequest({
      context: trellarAgent.events,
      trace_id: trellarAgent.traceId,
      agent_name: trellarAgent.agentName,
    });
  });
});
