/**
 * Reasoning extraction for Google Gemini models.
 *
 * Gemini exposes thinking as flat options (``thinking_budget``, ``thinking_level``,
 * ``include_thoughts``) or as one nested ``thinking_config`` object. Whatever is found
 * is returned as-is. The camelCase spellings are what the JS SDKs use; the allowlist is
 * identical to the Python one. Twin of
 * ``trellar/callbacks/utils/reasoning_extraction/extract_gemini.py``.
 */
import { type Mapping, pickKeys } from "./helpers.js";

export const KEYS = [
  "thinking_budget",
  "thinking_level",
  "include_thoughts",
  "thinking_config",
  "thinkingBudget",
  "thinkingLevel",
  "includeThoughts",
  "thinkingConfig",
] as const;

/** True when the (lower-cased) provider/model hint looks like Gemini / Google. */
export function matches(hint: string): boolean {
  return ["gemini", "google", "vertex", "genai"].some((token) => hint.includes(token));
}

/** The Gemini thinking settings found in *sources*, or ``null``. */
export function extract(sources: readonly Mapping[]): Mapping | null {
  return pickKeys(sources, KEYS);
}
