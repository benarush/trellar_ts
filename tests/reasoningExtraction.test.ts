/** Unit tests for ``extractReasoning`` and its provider modules (mirror of tests/reasoning/test_extract_reasoning.py). */
import { describe, expect, it } from "vitest";

import { extractReasoning } from "../src/callbacks/utils/index.js";
import * as anthropic from "../src/callbacks/utils/reasoningExtraction/extractAnthropic.js";
import * as gemini from "../src/callbacks/utils/reasoningExtraction/extractGemini.js";
import * as generic from "../src/callbacks/utils/reasoningExtraction/extractGeneric.js";
import * as openai from "../src/callbacks/utils/reasoningExtraction/extractOpenai.js";
import { flattenSources, pickKeys } from "../src/callbacks/utils/reasoningExtraction/helpers.js";

// Secret-looking fields every config is polluted with; none may ever be extracted.
const secretFields = () => ({
  api_key: "sk-super-secret",
  openai_api_key: { lc: 1, type: "secret", id: ["OPENAI_API_KEY"] },
  headers: { Authorization: "Bearer top-secret" },
});

const anthropicThinking = { type: "enabled", budget_tokens: 2000 };
const anthropicConfig = () => ({ model: "claude-sonnet-4-5", thinking: anthropicThinking, ...secretFields() });
const geminiConfig = () => ({ model: "gemini-2.5-flash", thinking_budget: 1024, include_thoughts: true, ...secretFields() });
const openaiConfig = () => ({ model_name: "o3-mini", reasoning_effort: "high", ...secretFields() });
// Strands keeps Gemini's thinking inside params as a nested thinking_config.
const strandsGeminiConfig = () => ({
  modelId: "gemini-2.5-flash",
  params: { temperature: 0.2, thinking_config: { thinking_budget: 512 } },
  ...secretFields(),
});
const bedrockClaudeConfig = () => ({
  modelId: "anthropic.claude-sonnet-4-5",
  additionalRequestFields: { thinking: anthropicThinking },
  ...secretFields(),
});

describe("providers", () => {
  it("anthropic thinking is returned as-is (same object)", () => {
    const result = extractReasoning([anthropicConfig()], "ChatAnthropic", "claude-sonnet-4-5");

    expect(result).toEqual({ thinking: anthropicThinking });
    expect(result!["thinking"]).toBe(anthropicThinking);
  });

  it("gemini flat options", () => {
    const result = extractReasoning([geminiConfig()], ["langchain", "ChatGoogleGenerativeAI"], "gemini-2.5-flash");

    expect(result).toEqual({ thinking_budget: 1024, include_thoughts: true });
  });

  it("gemini level stored as reasoning_effort by langchain", () => {
    expect(
      extractReasoning([{ model: "gemini-3-pro-preview", reasoning_effort: "high" }], "ChatGoogleGenerativeAI"),
    ).toEqual({ reasoning_effort: "high" });
  });

  it("gemini nested thinking_config from strands params", () => {
    expect(extractReasoning([strandsGeminiConfig()], "GeminiModel", "gemini-2.5-flash")).toEqual({
      thinking_config: { thinking_budget: 512 },
    });
  });

  it("gemini camelCase keys (JS SDKs) are kept as found", () => {
    expect(extractReasoning([{ thinkingConfig: { thinkingBudget: 10 }, includeThoughts: true }], "ChatGoogleGenerativeAI", "gemini-2.5"))
      .toEqual({ thinkingConfig: { thinkingBudget: 10 }, includeThoughts: true });
  });

  it("openai reasoning_effort and nested reasoning object", () => {
    expect(extractReasoning([openaiConfig()], "ChatOpenAI", "o3-mini")).toEqual({ reasoning_effort: "high" });
    const reasoning = { effort: "high", summary: "auto" };
    expect(extractReasoning([{ reasoning }], "ChatOpenAI", "gpt-5")).toEqual({ reasoning });
  });

  it("openai camelCase reasoningEffort", () => {
    expect(extractReasoning([{ reasoningEffort: "low" }], "ChatOpenAI", "gpt-5")).toEqual({ reasoningEffort: "low" });
  });

  it("claude on bedrock is found in additionalRequestFields", () => {
    expect(extractReasoning([bedrockClaudeConfig()], "BedrockModel")).toEqual({ thinking: anthropicThinking });
  });

  it("model name alone is enough to pick the provider", () => {
    expect(extractReasoning([{ thinking: anthropicThinking }], undefined, "claude-opus-4")).toEqual({
      thinking: anthropicThinking,
    });
  });

  it("keys of other providers are not picked", () => {
    const config = { thinking: { type: "enabled" }, reasoning_effort: "high", thinking_budget: 5 };

    expect(extractReasoning([config], "ChatAnthropic", "claude-sonnet-4-5")).toEqual({ thinking: { type: "enabled" } });
  });
});

describe("missing and empty", () => {
  it.each(["claude-sonnet-4-5", "gemini-2.5-flash", "gpt-4o", "mystery-model"])(
    "no reasoning keys returns null (%s)",
    (model) => {
      expect(extractReasoning([{ model, temperature: 0 }], undefined, model)).toBeNull();
    },
  );

  it("null/undefined values count as not set", () => {
    expect(extractReasoning([{ thinking: null, reasoning_effort: undefined }], undefined, "claude")).toBeNull();
  });

  it("falsy but real values are kept", () => {
    expect(extractReasoning([{ include_thoughts: false, thinking_budget: 0 }], undefined, "gemini-2.5")).toEqual({
      include_thoughts: false,
      thinking_budget: 0,
    });
  });

  it.each([[null], [undefined], [[]], [[null]], [[null, "text", 5, ["a"]]]])("bad or empty sources return null (%j)", (sources) => {
    expect(extractReasoning(sources as any, "ChatAnthropic", "claude")).toBeNull();
  });
});

describe("source precedence", () => {
  it("later source overrides earlier", () => {
    expect(extractReasoning([{ reasoning_effort: "low" }, { reasoning_effort: "high" }], "ChatOpenAI", "gpt-5")).toEqual({
      reasoning_effort: "high",
    });
  });

  it("sources are combined", () => {
    expect(
      extractReasoning([{ thinking_budget: 100 }, { include_thoughts: true }], "ChatGoogleGenerativeAI", "gemini-2.5"),
    ).toEqual({ thinking_budget: 100, include_thoughts: true });
  });

  it("model_kwargs and extra_body are searched", () => {
    expect(extractReasoning([{ model_kwargs: { reasoning_effort: "low" } }], "ChatOpenAI", "gpt-5")).toEqual({
      reasoning_effort: "low",
    });
    expect(extractReasoning([{ extra_body: { reasoning: { effort: "low" } } }], "ChatOpenAI", "gpt-5")).toEqual({
      reasoning: { effort: "low" },
    });
  });
});

describe("safety", () => {
  it.each([
    ["anthropic", anthropicConfig, "ChatAnthropic", "claude-sonnet-4-5"],
    ["gemini", geminiConfig, "ChatGoogleGenerativeAI", "gemini-2.5-flash"],
    ["openai", openaiConfig, "ChatOpenAI", "o3-mini"],
    ["strands gemini", strandsGeminiConfig, "GeminiModel", "gemini-2.5-flash"],
    ["bedrock claude", bedrockClaudeConfig, "BedrockModel", "anthropic.claude"],
  ])("secrets never leak (%s)", (_name, makeConfig, hint, model) => {
    const result = extractReasoning([makeConfig()], hint, model);

    expect(result).not.toBeNull();
    for (const key of Object.keys(secretFields())) expect(Object.keys(result!)).not.toContain(key);
    expect(JSON.stringify(result)).not.toContain("sk-super-secret");
    expect(JSON.stringify(result)).not.toContain("top-secret");
  });

  it("unknown provider uses the generic fallback and still skips secrets", () => {
    expect(extractReasoning([{ think: true, temperature: 1, ...secretFields() }], "SomeCustomChatModel", "mystery")).toEqual({
      think: true,
    });
  });

  it("a throwing getter returns null", () => {
    const hostile = {
      get thinking(): unknown {
        throw new Error("no access");
      },
    };

    expect(extractReasoning([hostile], "ChatAnthropic", "claude")).toBeNull();
  });
});

describe("generic fallback and merging", () => {
  it("generic fallback reads any known key", () => {
    const config = { reasoning_effort: "low", think: true, reasoningConfig: { type: "enabled" } };

    expect(extractReasoning([config], "ChatOllama", "llama3")).toEqual(config);
  });

  it("several matching providers are merged", () => {
    const config = { thinking: { type: "enabled" }, reasoning_effort: "low" };

    expect(extractReasoning([config], "ChatOpenAI", "claude-via-gateway")).toEqual(config);
  });

  it.each([
    ["anthropic", anthropic],
    ["claude-3", anthropic],
    ["gemini-2.5", gemini],
    ["vertexai", gemini],
    ["openai", openai],
    ["gpt-5", openai],
    ["o1-mini", openai],
    ["o4", openai],
  ])("matches %s", (hint, mod) => {
    expect(mod.matches(hint)).toBe(true);
  });

  it.each(["", "llama3", "mistral-large", "falcon", "bedrock"])("matches rejects unrelated (%j)", (hint) => {
    expect([anthropic, gemini, openai].some((m) => m.matches(hint))).toBe(false);
    expect(generic.matches(hint)).toBe(false);
  });
});

describe("helpers", () => {
  it("flattenSources keeps order and drops non-objects", () => {
    const outer = { params: { a: 1 }, x: 1 };

    expect(flattenSources([null, outer, "nope"])).toEqual([outer, { a: 1 }]);
  });

  it("pickKeys only returns allowlisted keys", () => {
    expect(pickKeys([{ a: 1, secret: "x" }], ["a", "b"])).toEqual({ a: 1 });
    expect(pickKeys([{ secret: "x" }], ["a"])).toBeNull();
  });
});
