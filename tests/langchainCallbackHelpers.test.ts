import { AIMessage, HumanMessage, SystemMessage, ToolMessage } from "@langchain/core/messages";
import { describe, expect, it } from "vitest";

import {
  contentToStr,
  extractLlmInput,
  extractModelName,
  findToolDescription,
  normalizeTools,
  serializeMessage,
} from "../src/callbacks/langchain/utils/index.js";

describe("extractModelName", () => {
  it("prefers kwargs.model_name, then kwargs.model", () => {
    expect(extractModelName({ name: "ChatOpenAI", kwargs: { model_name: "gpt-4o" } })).toBe("gpt-4o");
    expect(extractModelName({ name: "ChatOpenAI", kwargs: { model: "gpt-4o-mini" } })).toBe("gpt-4o-mini");
  });

  it("falls back to the class name", () => {
    expect(extractModelName({ name: "ChatOpenAI" })).toBe("ChatOpenAI");
    expect(extractModelName({ id: ["langchain", "chat_models", "ChatGoogle"] })).toBe("ChatGoogle");
    expect(extractModelName({ kwargs: {} })).toBeUndefined();
  });

  it("uses the invocation params when kwargs have no model", () => {
    expect(extractModelName({ id: ["x", "ChatGoogle"] }, { model: "gemini-2.5-flash-lite" })).toBe(
      "gemini-2.5-flash-lite",
    );
  });
});

describe("contentToStr", () => {
  it("returns strings untouched", () => expect(contentToStr("hello")).toBe("hello"));
  it("JSON-encodes lists with Python separators", () => {
    expect(contentToStr([{ type: "text", text: "hi" }])).toBe('[{"type": "text", "text": "hi"}]');
  });
  it("keeps non-ASCII characters", () => expect(contentToStr(["é"])).toBe('["é"]'));
});

describe("serializeMessage", () => {
  it("serializes role/content and extras", () => {
    const msg = new AIMessage({ content: "x", tool_calls: [{ name: "t", args: {}, id: "1" }] });
    const out = serializeMessage(msg);
    expect(out["role"]).toBe("ai");
    expect(out["content"]).toBe("x");
    expect(out["tool_calls"]).toBeTruthy();
  });
  it("falls back to raw for non-messages", () => {
    expect(serializeMessage("plain")).toEqual({ raw: "plain" });
  });
});

describe("extractLlmInput", () => {
  it("extracts system and the plain human text", () => {
    expect(extractLlmInput([new SystemMessage("sys"), new HumanMessage("hi")])).toEqual({
      system: "sys",
      human: "hi",
    });
  });

  it("returns null human when there is no human turn", () => {
    expect(extractLlmInput([new SystemMessage("sys")])).toEqual({ system: "sys", human: null });
    expect(extractLlmInput([])).toEqual({ system: null, human: null });
  });

  it("folds earlier turns into the human string", () => {
    const out = extractLlmInput([
      new SystemMessage("sys"),
      new HumanMessage("turn 1"),
      new AIMessage("answer 1"),
      new HumanMessage("turn 2"),
    ]);
    expect(out.human).toBe("Human: turn 1\nAI LLM: answer 1\n\nCurrent message - turn 2");
  });

  it("skips tool messages and tool-call-only AI messages", () => {
    const out = extractLlmInput([
      new HumanMessage("q"),
      new AIMessage({ content: "", tool_calls: [{ name: "t", args: {}, id: "1" }] }),
      new ToolMessage({ content: "result", tool_call_id: "1" }),
      new HumanMessage("follow up"),
    ]);
    expect(out.human).toBe("Human: q\n\nCurrent message - follow up");
  });

  it("uses the last human turn and ignores anything after it", () => {
    const out = extractLlmInput([new HumanMessage("a"), new AIMessage("b"), new HumanMessage("c"), new AIMessage("d")]);
    expect(out.human).toBe("Human: a\nAI LLM: b\n\nCurrent message - c");
  });

  it("accepts plain dicts with OpenAI-style roles", () => {
    const out = extractLlmInput([
      { role: "system", content: "S" },
      { role: "user", content: "u1" },
      { role: "assistant", content: "a1" },
      { role: "user", content: "u2" },
    ]);
    expect(out).toEqual({ system: "S", human: "Human: u1\nAI LLM: a1\n\nCurrent message - u2" });
  });

  it("accepts plain dicts with LangChain-style type keys", () => {
    expect(extractLlmInput([{ type: "human", content: "x" }])).toEqual({ system: null, human: "x" });
  });

  it("keeps only the first system message", () => {
    expect(extractLlmInput([new SystemMessage("one"), new SystemMessage("two"), new HumanMessage("h")]).system).toBe(
      "one",
    );
  });

  it("json-encodes multimodal list content", () => {
    const out = extractLlmInput([new HumanMessage({ content: [{ type: "text", text: "look" }] })]);
    expect(out.human).toBe('[{"type": "text", "text": "look"}]');
  });
});

describe("normalizeTools", () => {
  it("keeps OpenAI function tools and fills defaults", () => {
    expect(normalizeTools([{ type: "function", function: { name: "f" } }])).toEqual([
      { type: "function", function: { name: "f", description: "", parameters: {} } },
    ]);
  });

  it("expands Google functionDeclarations", () => {
    const out = normalizeTools([
      { functionDeclarations: [{ name: "a", description: "da", parameters: { type: "object" } }, { name: "b" }] },
    ]);
    expect(out).toEqual([
      { type: "function", function: { name: "a", description: "da", parameters: { type: "object" } } },
      { type: "function", function: { name: "b", description: "", parameters: {} } },
    ]);
  });

  it("wraps bare {name, description, schema} objects", () => {
    expect(normalizeTools([{ name: "n", description: "d", schema: { type: "object" } }])).toEqual([
      { type: "function", function: { name: "n", description: "d", parameters: { type: "object" } } },
    ]);
  });

  it("returns [] for non-arrays", () => expect(normalizeTools(undefined)).toEqual([]));
});

describe("findToolDescription", () => {
  const available = { h: [{ type: "function", function: { name: "buy", description: "Buy stuff", parameters: {} } }] };
  it("finds a description by tool name", () => expect(findToolDescription(available, "buy")).toBe("Buy stuff"));
  it("returns null for unknown / empty names", () => {
    expect(findToolDescription(available, "nope")).toBeNull();
    expect(findToolDescription(available, null)).toBeNull();
  });
});
