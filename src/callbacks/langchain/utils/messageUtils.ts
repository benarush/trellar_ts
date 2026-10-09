import { compactJson, pyJsonDumps } from "../../common.js";

type Obj = Record<string, unknown>;

/** True for LangChain ``BaseMessage`` instances (Python: ``hasattr(msg, "type") and hasattr(msg, "content")``). */
export function isMessageLike(value: unknown): value is { type: string; content: unknown } & Obj {
  if (value === null || typeof value !== "object") return false;
  const v = value as Obj;
  const hasType = "type" in v && typeof v["type"] === "string";
  return hasType && "content" in v;
}

/** Role of a message: ``BaseMessage`` -> its type; plain dict -> ``role`` or ``type``. */
function roleOf(msg: Obj): string {
  if (typeof msg["_getType"] === "function") {
    try {
      return String((msg["_getType"] as () => unknown).call(msg)).toLowerCase();
    } catch {
      /* fall through */
    }
  }
  return String(msg["role"] || msg["type"] || "").toLowerCase();
}

/** Serialize a LangChain BaseMessage to a plain dict (role + content + extras). */
export function serializeMessage(msg: unknown): Obj {
  if (!isMessageLike(msg)) return { raw: String(msg) };

  const result: Obj = { role: msg.type, content: msg.content };

  const additional = msg["additional_kwargs"];
  if (additional && typeof additional === "object" && Object.keys(additional).length > 0) {
    // Capture tool_calls, function_call, etc.
    result["additional_kwargs"] = additional;
  }

  const toolCalls = msg["tool_calls"];
  if (Array.isArray(toolCalls) && toolCalls.length > 0) {
    result["tool_calls"] = toolCalls;
  }

  return result;
}

/**
 * Convert a message content value to a plain string.
 *
 * LangChain message content can be a str, a list of dicts (multimodal),
 * or any other JSON-serializable value for structured outputs.
 */
export function contentToStr(content: unknown): string {
  if (typeof content === "string") return content;
  try {
    return pyJsonDumps(content, { sortKeys: false, ensureAscii: false });
  } catch {
    return String(content);
  }
}

function isTruthy(value: unknown): boolean {
  if (value === null || value === undefined || value === false || value === 0 || value === "") return false;
  if (Array.isArray(value)) return value.length > 0;
  return true;
}

export interface LlmInput {
  system: string | null;
  human: string | null;
}

/**
 * Extract structured system/human fields from a list of LangChain messages.
 *
 * Returns ``{ system, human }``:
 * - ``system``: content of the first SystemMessage, or ``null``
 * - ``human``: the active prompt, or ``null``
 *
 * The active prompt is the *last* human turn, because that is what the LLM
 * is responding to. When earlier Human/AI turns exist before it, they are
 * folded into the same string (no extra key)::
 *
 *     Human: turn 1 text
 *     AI LLM: turn 1 response text
 *     Human: turn 2 text
 *     AI LLM: turn 2 response text
 *
 *     Current message - <last human text>
 *
 * With no prior history, ``human`` is just the plain last human text. Only
 * human and non-empty AI text turns are kept in the history; tool messages
 * and tool-call-only AI messages are skipped.
 *
 * Handles both LangChain ``BaseMessage`` objects and plain dicts. Also accepts
 * ``"user"`` as a synonym for ``"human"`` and ``"assistant"`` as a synonym for
 * ``"ai"`` to cover OpenAI-style role names.
 */
export function extractLlmInput(messages: unknown[]): LlmInput {
  let system: string | null = null;
  // [role, content] pairs for human/ai turns, in order; role is "human" or "ai".
  const turns: Array<["human" | "ai", string]> = [];

  for (const msg of messages) {
    let role: string;
    let content: string;
    if (msg !== null && typeof msg === "object" && (typeof (msg as Obj)["_getType"] === "function" || isMessageLike(msg))) {
      // Standard LangChain BaseMessage object
      role = roleOf(msg as Obj);
      content = contentToStr((msg as Obj)["content"]);
    } else if (msg !== null && typeof msg === "object" && !Array.isArray(msg)) {
      // Serialised dict -- may use "role" (OpenAI/Phoenix) or "type" (LangChain)
      const m = msg as Obj;
      role = String(m["role"] || m["type"] || "").toLowerCase();
      content = contentToStr(isTruthy(m["content"]) ? m["content"] : "");
    } else {
      continue;
    }

    if (role === "system" && system === null) {
      system = content;
    } else if (role === "human" || role === "user") {
      turns.push(["human", content]);
    } else if (role === "ai" || role === "assistant") {
      turns.push(["ai", content]);
    }
  }

  let lastHumanIdx = -1;
  for (let i = turns.length - 1; i >= 0; i--) {
    if (turns[i]![0] === "human") {
      lastHumanIdx = i;
      break;
    }
  }
  if (lastHumanIdx === -1) return { system, human: null };

  const current = turns[lastHumanIdx]![1];
  const historyLines = turns
    .slice(0, lastHumanIdx)
    // Tool-call-only AI messages have empty text; skip them.
    .filter(([role, content]) => content !== "" || role === "human")
    .map(([role, content]) => `${role === "human" ? "Human" : "AI LLM"}: ${content}`);
  if (historyLines.length === 0) return { system, human: current };

  const human = historyLines.join("\n") + `\n\nCurrent message - ${current}`;
  return { system, human };
}

/** Plain objects that aren't messages: JSON text (Python: ``str(dict)``). */
export function stringifyUnknown(value: unknown): string {
  if (value !== null && typeof value === "object") return compactJson(value);
  return String(value);
}
