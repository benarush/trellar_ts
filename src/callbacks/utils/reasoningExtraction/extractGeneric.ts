/**
 * Fallback reasoning extraction for providers we have no dedicated module for.
 *
 * Used only when no provider module matched the hint (Ollama, DeepSeek, Bedrock Nova,
 * LiteLLM, unknown classes ...). It looks for the union of every known reasoning key plus
 * the generic ``think`` / ``reasoningConfig`` spellings, still strictly from an allowlist.
 * Twin of ``trellar/callbacks/utils/reasoning_extraction/extract_generic.py``.
 */
import * as anthropic from "./extractAnthropic.js";
import * as gemini from "./extractGemini.js";
import * as openai from "./extractOpenai.js";
import { type Mapping, pickKeys } from "./helpers.js";

export const KEYS: readonly string[] = [
  ...new Set<string>([...anthropic.KEYS, ...gemini.KEYS, ...openai.KEYS, "think", "reasoningConfig"]),
];

/** The generic extractor is never chosen by hint; it is the dispatcher's fallback. */
export function matches(_hint: string): boolean {
  return false;
}

/** Any known reasoning setting found in *sources*, or ``null``. */
export function extract(sources: readonly Mapping[]): Mapping | null {
  return pickKeys(sources, KEYS);
}
