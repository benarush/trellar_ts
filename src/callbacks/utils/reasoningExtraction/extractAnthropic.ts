/**
 * Reasoning extraction for Anthropic Claude models.
 *
 * Claude is configured with a single ``thinking`` object, e.g.
 * ``{ type: "enabled", budget_tokens: 2000 }``. It is returned as-is.
 *
 * Covers LangChain ``ChatAnthropic`` (``thinking`` on the model), Strands
 * ``AnthropicModel`` (``params.thinking``) and Claude on Bedrock
 * (``additionalRequestFields.thinking``) - the latter two are reached through
 * ``flattenSources``. Twin of ``trellar/callbacks/utils/reasoning_extraction/extract_anthropic.py``.
 */
import { type Mapping, pickKeys } from "./helpers.js";

export const KEYS = ["thinking"] as const;

/** True when the (lower-cased) provider/model hint looks like Anthropic/Claude. */
export function matches(hint: string): boolean {
  return hint.includes("anthropic") || hint.includes("claude");
}

/** The Anthropic thinking settings found in *sources*, or ``null``. */
export function extract(sources: readonly Mapping[]): Mapping | null {
  return pickKeys(sources, KEYS);
}
