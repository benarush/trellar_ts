import { describe, expect, it } from "vitest";

import { buildContext, compactJson, hashTools, toJsonable } from "../src/callbacks/common.js";

describe("compactJson (parity with Python json.dumps)", () => {
  it("uses Python separators and keeps non-ASCII", () => {
    expect(compactJson({ symbols: "TSLA,F", n: [1, 2, { a: null, b: true }], u: "é ✓" })).toBe(
      '{"symbols": "TSLA,F", "n": [1, 2, {"a": null, "b": true}], "u": "é ✓"}',
    );
  });

  it("handles empty containers and escapes", () => {
    expect(compactJson([])).toBe("[]");
    expect(compactJson({})).toBe("{}");
    expect(compactJson('x"y\n')).toBe('"x\\"y\\n"');
  });

  it("maps undefined to null", () => {
    expect(compactJson({ a: undefined })).toBe('{"a": null}');
  });

  it("falls back instead of throwing on circular structures", () => {
    const a: Record<string, unknown> = {};
    a["self"] = a;
    expect(typeof compactJson(a)).toBe("string");
  });
});

describe("hashTools (parity with Python sha256(json.dumps(sort_keys=True))[:16])", () => {
  it("matches the Python golden values", () => {
    const tools = [
      {
        type: "function",
        function: {
          name: "buy",
          description: "Buy é",
          parameters: {
            type: "object",
            properties: { symbols: { type: "string" } },
            required: ["symbols"],
          },
        },
      },
    ];
    expect(hashTools(tools)).toBe("1b9c3a996674f394");
    expect(hashTools([{ b: 1, a: [1, 2, { z: null, y: "ü✓😀" }] }])).toBe("3ad5258aed2d7579");
    expect(hashTools([])).toBe("4f53cda18c2baa0c");
  });

  it("is independent of key order", () => {
    expect(hashTools([{ a: 1, b: 2 }])).toBe(hashTools([{ b: 2, a: 1 }]));
  });
});

describe("toJsonable", () => {
  it("passes primitives through and converts undefined to null", () => {
    expect(toJsonable("x")).toBe("x");
    expect(toJsonable(3)).toBe(3);
    expect(toJsonable(false)).toBe(false);
    expect(toJsonable(undefined)).toBeNull();
    expect(toJsonable({ a: undefined })).toEqual({ a: null });
  });

  it("recurses into arrays, maps, sets and toJSON objects", () => {
    expect(toJsonable({ a: [1, { b: new Set([1]) }], m: new Map([["k", 1]]) })).toEqual({
      a: [1, { b: [1] }],
      m: { k: 1 },
    });
    expect(toJsonable({ toJSON: () => ({ x: 1 }) })).toEqual({ x: 1 });
  });

  it("survives circular structures", () => {
    const a: Record<string, unknown> = {};
    a["self"] = a;
    expect(() => JSON.stringify(toJsonable(a))).not.toThrow();
  });
});

describe("buildContext", () => {
  const events = [
    { event: "on_chain_start", graph_order: 1, node_name: "root", node_type: "chain", inputs: ["hi"] },
    { event: "on_chat_model_start", graph_order: 2, node_name: "m", node_type: "llm", input: { system: "S", human: "H" } },
    { event: "on_llm_end", graph_order: 3, node_name: "m", node_type: "llm", output: { response: "R" }, token_usage: { total: 3 } },
    { event: "on_tool_start", graph_order: 4, node_name: "t", node_type: "tool", tool: "t", input: { a: 1 } },
    { event: "on_tool_end", graph_order: 5, node_name: "t", node_type: "tool", output: "ok", is_mcp_tool: true },
    { event: "on_chain_error", graph_order: 6, node_name: null, node_type: null, error: "boom" },
  ];

  it("renders the header, steps and MCP marker", () => {
    const text = buildContext(events as any, "trace-1");
    expect(text).toContain("Trace ID: trace-1");
    expect(text).toContain("Total steps: 6");
    expect(text).toContain("[Step 1] on_chain_start  (chain: root)");
    expect(text).toContain("  inputs: [\"hi\"]");
    expect(text).toContain("  system: S");
    expect(text).toContain("  human: H");
    expect(text).toContain('  token_usage: {"total": 3}');
    expect(text).toContain('  input: {"a": 1}');
    expect(text).toContain('  output (via MCP): "ok"');
    expect(text).toContain("[Step 6] on_chain_error\n  error: boom");
  });
});
