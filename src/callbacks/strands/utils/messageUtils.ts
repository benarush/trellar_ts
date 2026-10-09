import { compactJson } from "../../common.js";

type Obj = Record<string, any>;

/** Join the text / json content blocks of a Strands message or tool result. */
export function textOfBlocks(blocks: unknown): string {
  const parts: string[] = [];
  const list = Array.isArray(blocks) ? blocks : [];
  for (const block of list) {
    if (block === null || typeof block !== "object") continue;
    const b = block as Obj;
    if (typeof b["text"] === "string") {
      parts.push(b["text"]);
    } else if ("json" in b) {
      parts.push(compactJson(b["json"]));
    }
  }
  return parts.join("\n");
}

/**
 * Build the ``input.human`` string for an LLM call from Strands messages.
 *
 * The current message is the last user turn that has text. Earlier user and
 * assistant text turns are folded into the same string::
 *
 *     Human: turn 1 text
 *     AI LLM: turn 1 response text
 *
 *     Current message - <last user text>
 *
 * With no earlier turns, returns just the plain current text. Tool-result-only
 * user messages and tool-call-only assistant messages have no text and are
 * skipped; anything after the current user turn is ignored. Returns ``null``
 * when there is no user text at all.
 */
export function llmHumanInput(messages: unknown): string | null {
  const turns: Array<["user" | "assistant", string]> = [];
  for (const msg of Array.isArray(messages) ? messages : []) {
    if (msg === null || typeof msg !== "object") continue;
    const role = (msg as Obj)["role"];
    if (role !== "user" && role !== "assistant") continue;
    const text = textOfBlocks((msg as Obj)["content"]);
    if (text) turns.push([role, text]);
  }

  let lastUserIdx = -1;
  for (let i = turns.length - 1; i >= 0; i--) {
    if (turns[i]![0] === "user") {
      lastUserIdx = i;
      break;
    }
  }
  if (lastUserIdx === -1) return null;

  const current = turns[lastUserIdx]![1];
  const historyLines = turns
    .slice(0, lastUserIdx)
    .map(([role, text]) => `${role === "user" ? "Human" : "AI LLM"}: ${text}`);
  if (historyLines.length === 0) return current;
  return historyLines.join("\n") + `\n\nCurrent message - ${current}`;
}

/** Text of the last user message (tool-result-only user turns are skipped). */
export function lastUserText(messages: unknown): string | null {
  const list = Array.isArray(messages) ? messages : [];
  for (let i = list.length - 1; i >= 0; i--) {
    const msg = list[i];
    if (msg !== null && typeof msg === "object" && (msg as Obj)["role"] === "user") {
      const text = textOfBlocks((msg as Obj)["content"]);
      if (text) return text;
    }
  }
  return null;
}

/** Text of the last assistant message (what the agent finally answered). */
export function lastAssistantText(messages: unknown): string {
  const list = Array.isArray(messages) ? messages : [];
  for (let i = list.length - 1; i >= 0; i--) {
    const msg = list[i];
    if (msg !== null && typeof msg === "object" && (msg as Obj)["role"] === "assistant") {
      return textOfBlocks((msg as Obj)["content"]);
    }
  }
  return "";
}
