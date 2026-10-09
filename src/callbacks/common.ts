/** Framework-neutral helpers shared by the LangChain and Strands callbacks. */
import { createHash } from "node:crypto";
import { inspect } from "node:util";

export type JsonObject = Record<string, unknown>;

interface PyDumpOptions {
  sortKeys: boolean;
  ensureAscii: boolean;
}

function encodeString(value: string, ensureAscii: boolean): string {
  const encoded = JSON.stringify(value);
  if (!ensureAscii) return encoded;
  // Python's ensure_ascii=True escapes every non-ASCII UTF-16 unit as \uXXXX (lowercase hex).
  // eslint-disable-next-line no-control-regex
  return encoded.replace(/[^\x00-\x7f]/g, (ch) => "\\u" + ch.charCodeAt(0).toString(16).padStart(4, "0"));
}

/**
 * Serialize like Python's ``json.dumps`` (``", "`` / ``": "`` separators,
 * optional ``sort_keys`` / ``ensure_ascii``, unknown objects coerced with
 * ``str``). Throws on circular structures, as Python does.
 */
export function pyJsonDumps(value: unknown, options: PyDumpOptions): string {
  const stack = new Set<object>();

  const dump = (v: unknown): string => {
    if (v === null || v === undefined) return "null";
    switch (typeof v) {
      case "boolean":
        return v ? "true" : "false";
      case "number":
        if (Number.isNaN(v)) return "NaN";
        if (v === Infinity) return "Infinity";
        if (v === -Infinity) return "-Infinity";
        return String(v);
      case "bigint":
        return v.toString();
      case "string":
        return encodeString(v, options.ensureAscii);
      case "function":
      case "symbol":
        return encodeString(String(v), options.ensureAscii);
      default:
        break;
    }

    const obj = v as object;
    if (stack.has(obj)) throw new Error("Circular reference detected");

    const toJSON = (obj as { toJSON?: unknown }).toJSON;
    if (typeof toJSON === "function" && !Array.isArray(obj)) {
      stack.add(obj);
      try {
        return dump((toJSON as () => unknown).call(obj));
      } finally {
        stack.delete(obj);
      }
    }

    stack.add(obj);
    try {
      if (Array.isArray(obj)) {
        return "[" + obj.map(dump).join(", ") + "]";
      }
      if (obj instanceof Map || obj instanceof Set || obj instanceof Date || obj instanceof Error) {
        return encodeString(String(obj), options.ensureAscii);
      }
      let keys = Object.keys(obj);
      if (options.sortKeys) keys = keys.sort();
      const parts = keys.map(
        (k) => `${encodeString(k, options.ensureAscii)}: ${dump((obj as JsonObject)[k])}`,
      );
      return "{" + parts.join(", ") + "}";
    } finally {
      stack.delete(obj);
    }
  };

  return dump(value);
}

/** Render *value* as compact JSON, falling back to a debug representation on failure. */
export function compactJson(value: unknown): string {
  try {
    return pyJsonDumps(value, { sortKeys: false, ensureAscii: false });
  } catch {
    return inspect(value, { depth: 4 });
  }
}

/**
 * Stable content hash of a tool schema list.
 *
 * Used as the ``available_tools`` key and stamped on the matching
 * ``on_chat_model_start`` event so the backend can link the two.
 */
export function hashTools(tools: unknown[]): string {
  const canonical = pyJsonDumps(tools, { sortKeys: true, ensureAscii: true });
  return createHash("sha256").update(canonical, "utf8").digest("hex").slice(0, 16);
}

function isPlainScalar(value: unknown): value is string | number | boolean {
  return typeof value === "string" || typeof value === "number" || typeof value === "boolean";
}

/** Recursively coerce *value* into something ``JSON.stringify`` can handle. */
export function toJsonable(value: unknown, seen: WeakSet<object> = new WeakSet()): unknown {
  if (value === null || value === undefined) return null;
  if (isPlainScalar(value)) return value;
  if (typeof value === "bigint") {
    return Number.isSafeInteger(Number(value)) ? Number(value) : value.toString();
  }
  if (typeof value === "function" || typeof value === "symbol") return String(value);

  const obj = value as object;
  if (seen.has(obj)) return String(obj);

  if (Array.isArray(obj)) {
    seen.add(obj);
    try {
      return obj.map((v) => toJsonable(v, seen));
    } finally {
      seen.delete(obj);
    }
  }
  if (obj instanceof Map) {
    seen.add(obj);
    try {
      const out: JsonObject = {};
      for (const [k, v] of obj) out[String(k)] = toJsonable(v, seen);
      return out;
    } finally {
      seen.delete(obj);
    }
  }
  if (obj instanceof Set) {
    return toJsonable([...obj], seen);
  }
  if (obj instanceof Date) return obj.toISOString();
  if (obj instanceof Error) return String(obj);

  const toJSON = (obj as { toJSON?: unknown }).toJSON;
  if (typeof toJSON === "function") {
    seen.add(obj);
    try {
      return toJsonable((toJSON as () => unknown).call(obj), seen);
    } catch {
      return String(obj);
    } finally {
      seen.delete(obj);
    }
  }

  seen.add(obj);
  try {
    const out: JsonObject = {};
    for (const key of Object.keys(obj)) out[key] = toJsonable((obj as JsonObject)[key], seen);
    return out;
  } finally {
    seen.delete(obj);
  }
}

/** Python truthiness for the values that occur in recorded events. */
function truthy(value: unknown): boolean {
  if (value === null || value === undefined || value === false || value === 0 || value === "") return false;
  if (Array.isArray(value)) return value.length > 0;
  if (typeof value === "object") return Object.keys(value as object).length > 0;
  return true;
}

function asRecord(value: unknown): JsonObject {
  return value !== null && typeof value === "object" ? (value as JsonObject) : {};
}

function orDefault(value: unknown, fallback: unknown): unknown {
  return value === undefined ? fallback : value;
}

function pyStr(value: unknown): string {
  if (value === null || value === undefined) return "None";
  return typeof value === "string" ? value : String(value);
}

/** Render recorded events as a numbered, step-by-step narrative. */
export function buildContext(events: JsonObject[], traceId: unknown): string {
  const lines: string[] = [
    "=== Agent Run Context ===",
    `Trace ID: ${pyStr(traceId)}`,
    `Total steps: ${events.length}`,
    "",
  ];

  for (const event of events) {
    const step = orDefault(event["graph_order"], "?");
    const eventName = (orDefault(event["event"], "unknown") as string) ?? "unknown";
    const nodeName = (event["node_name"] as string | null | undefined) || "";
    const nodeType = (event["node_type"] as string | null | undefined) || "";

    let header = `[Step ${pyStr(step)}] ${eventName}`;
    if (nodeName) header += `  (${nodeType}: ${nodeName})`;
    lines.push(header);

    if (eventName === "on_llm_start" || eventName === "on_chat_model_start") {
      const inp = asRecord(orDefault(event["input"], {}));
      if (truthy(inp["system"])) lines.push(`  system: ${pyStr(inp["system"])}`);
      if (truthy(inp["human"])) lines.push(`  human: ${pyStr(inp["human"])}`);
    } else if (eventName === "on_llm_end") {
      const out = asRecord(orDefault(event["output"], {}));
      const usage = event["token_usage"];
      lines.push(`  response: ${pyStr(orDefault(out["response"], ""))}`);
      if (truthy(usage)) lines.push(`  token_usage: ${compactJson(usage)}`);
    } else if (eventName === "on_tool_start") {
      lines.push(`  tool: ${pyStr(orDefault(event["tool"], ""))}`);
      lines.push(`  input: ${compactJson(orDefault(event["input"], {}))}`);
    } else if (eventName === "on_tool_end") {
      // MCP provenance is signal for the confidence evaluator, not noise.
      const viaMcp = truthy(event["is_mcp_tool"]) ? " (via MCP)" : "";
      lines.push(`  output${viaMcp}: ${compactJson(orDefault(event["output"], ""))}`);
    } else if (eventName === "on_chain_start") {
      lines.push(`  inputs: ${compactJson(orDefault(event["inputs"], {}))}`);
    } else if (eventName === "on_chain_end") {
      lines.push(`  outputs: ${compactJson(orDefault(event["outputs"], {}))}`);
    } else if (eventName === "on_llm_error" || eventName === "on_tool_error" || eventName === "on_chain_error") {
      lines.push(`  error: ${pyStr(orDefault(event["error"], ""))}`);
    }

    lines.push(""); // blank line between steps
  }

  return lines.join("\n");
}
