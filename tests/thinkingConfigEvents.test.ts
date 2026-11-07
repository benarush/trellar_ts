/** ``thinking_config`` on model-start events (LangChain + Strands); mirror of tests/reasoning/test_thinking_config_events.py. */
import { Agent } from "@strands-agents/sdk";
import { describe, expect, it } from "vitest";

import { extractThinkingConfig } from "../src/callbacks/langchain/utils/index.js";
import { LangchainAgentCallback } from "../src/callbacks/langchain/langchainCallback.js";
import { thinkingConfig } from "../src/callbacks/strands/utils/index.js";
import { trellarStrandsAgent } from "../src/strands.js";
import { uuid } from "./factories.js";
import { assertValidAgentLoopRequest } from "./helpers.js";
import { FakeModel } from "./strandsFactories.js";

const anthropicThinking = { type: "enabled", budget_tokens: 2000 };

/** A handler with a started root chain. */
function setup() {
  const h = new LangchainAgentCallback({ agentName: "test-agent" });
  h.handleChainStart({ name: "graph" }, { messages: [] }, uuid(), undefined, [], {}, undefined, "graph");
  return h;
}

describe("langchain events", () => {
  it("chat model start carries the constructor config, without secrets", () => {
    const h = setup();
    const serialized = {
      id: ["langchain_anthropic", "chat_models", "ChatAnthropic"],
      kwargs: { model: "claude-sonnet-4-5", thinking: anthropicThinking, api_key: "sk-super-secret" },
    };

    h.handleChatModelStart(serialized, [[]], uuid(), undefined, {});

    const event = h.events[h.events.length - 1]!;
    expect(event["thinking_config"]).toEqual({ thinking: anthropicThinking });
    expect(JSON.stringify(event)).not.toContain("sk-super-secret");
  });

  it("invocation_params override the constructor config", () => {
    const h = setup();
    const serialized = { id: ["langchain_openai", "ChatOpenAI"], kwargs: { model: "gpt-5", reasoning_effort: "low" } };

    h.handleChatModelStart(serialized, [[]], uuid(), undefined, { invocation_params: { reasoning_effort: "high" } });

    expect(h.events[h.events.length - 1]!["thinking_config"]).toEqual({ reasoning_effort: "high" });
  });

  it("llm start carries the config", () => {
    const h = setup();
    const serialized = {
      id: ["langchain_google_genai", "ChatGoogleGenerativeAI"],
      kwargs: { model: "gemini-2.5-flash", thinking_budget: 1024, include_thoughts: true },
    };

    h.handleLLMStart(serialized, ["prompt"], uuid(), undefined, {});

    expect(h.events[h.events.length - 1]!["thinking_config"]).toEqual({ thinking_budget: 1024, include_thoughts: true });
  });

  it("unset gives null", () => {
    const h = setup();

    h.handleChatModelStart({ kwargs: { model: "gpt-4o" } }, [[]], uuid(), undefined, {});

    expect(h.events[h.events.length - 1]!["thinking_config"]).toBeNull();
  });

  it("events match the backend schema", () => {
    const h = setup();
    const serialized = { id: ["ChatAnthropic"], kwargs: { model: "claude-sonnet-4-5", thinking: anthropicThinking } };
    h.handleChatModelStart(serialized, [[]], uuid(), undefined, {});
    h.handleLLMStart(serialized, ["p"], uuid(), undefined, {});

    assertValidAgentLoopRequest({ context: h.events, trace_id: h.traceId, agent_name: h.agentName });
  });

  it.each([[null], [undefined], [{}], [{ kwargs: null }], [{ id: null, kwargs: "oops" }], ["text"]])(
    "helper never throws on odd serialized (%j)",
    (serialized) => {
      expect(extractThinkingConfig(serialized, undefined)).toBeNull();
      expect([null, { reasoning_effort: "low" }]).toContainEqual(extractThinkingConfig(serialized, { reasoning_effort: "low" }));
    },
  );
});

describe("strands events", () => {
  /** FakeModel whose config also carries reasoning params and a secret. */
  class ConfiguredModel extends FakeModel {
    constructor(private readonly extraConfig: Record<string, unknown>) {
      super([{ text: "hi" }], "claude-sonnet-4-5");
    }
    override getConfig(): { modelId: string } & Record<string, unknown> {
      return { ...super.getConfig(), ...this.extraConfig };
    }
  }

  async function run(model: FakeModel, id: string) {
    const trellarAgent = trellarStrandsAgent("t");
    const agent = new Agent({ id, name: "calc", model, plugins: [trellarAgent], printer: false });
    await agent.invoke("hello");
    const starts = trellarAgent.events.filter((e) => e["event"] === "on_chat_model_start");
    expect(starts).toHaveLength(1);
    assertValidAgentLoopRequest({
      context: trellarAgent.events,
      trace_id: trellarAgent.traceId,
      agent_name: trellarAgent.agentName,
    });
    return starts[0]!;
  }

  it("params thinking is recorded, without secrets", async () => {
    const event = await run(
      new ConfiguredModel({ params: { thinking: anthropicThinking }, apiKey: "sk-super-secret" }),
      "calc-thinking",
    );

    expect(event["thinking_config"]).toEqual({ thinking: anthropicThinking });
    expect(JSON.stringify(event)).not.toContain("sk-super-secret");
  });

  it("unset gives null", async () => {
    expect((await run(new FakeModel([{ text: "hi" }]), "calc-unset"))["thinking_config"]).toBeNull();
  });

  // The Strands SDK itself reads getConfig() during a run, so a throwing model can only be
  // checked at the helper level (a real run would fail inside the SDK, not in our hook).
  it.each([
    ["getConfig throws", { model: { getConfig: () => { throw new Error("no config"); } } }],
    ["no model", {}],
    ["no getConfig", { model: {} }],
  ])("helper gives null when %s", (_name, agent) => {
    expect(thinkingConfig(agent)).toBeNull();
  });
});
