/**
 * Reasoning extraction for OpenAI models (GPT-5, o-series, gpt-oss, Azure OpenAI).
 *
 * OpenAI uses ``reasoning_effort`` (``"low" | "medium" | "high"``) or a nested
 * ``reasoning`` object such as ``{ effort: "high", summary: "auto" }``. Values are
 * returned as-is. Twin of ``trellar/callbacks/utils/reasoning_extraction/extract_openai.py``.
 */
import { type Mapping, pickKeys } from "./helpers.js";

export const KEYS = ["reasoning_effort", "reasoning", "reasoningEffort"] as const;

// o1 / o3-mini / o4 ... but not a stray "o" inside another word.
const O_SERIES = /(?<![a-z0-9])o\d/;

/** True when the (lower-cased) provider/model hint looks like OpenAI. */
export function matches(hint: string): boolean {
  return ["openai", "gpt", "azure"].some((token) => hint.includes(token)) || O_SERIES.test(hint);
}

/** The OpenAI reasoning settings found in *sources*, or ``null``. */
export function extract(sources: readonly Mapping[]): Mapping | null {
  return pickKeys(sources, KEYS);
}
