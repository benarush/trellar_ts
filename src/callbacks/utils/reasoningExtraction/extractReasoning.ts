/**
 * Single entry point for "what reasoning/thinking settings was this model given?".
 *
 * How it works (read this first):
 *
 * 1. Callers hand over the model's raw config objects, a *provider hint* (class name /
 *    serialized id) and the model name.
 * 2. ``flattenSources`` turns the config objects into an ordered list of plain objects,
 *    also looking one level into ``params`` / ``model_kwargs`` / ``additionalRequestFields``.
 * 3. Each provider module (anthropic, gemini, openai) answers "does this hint look like
 *    me?"; every module that says yes extracts its own allowlisted keys. If none says
 *    yes, the generic extractor is used instead.
 * 4. The per-provider results are merged and returned exactly as found, or ``null``.
 *
 * Only allowlisted keys are ever read, so secrets in the config (``api_key``, headers ...)
 * cannot leak, and any failure degrades to ``null`` instead of throwing.
 * Twin of ``trellar/callbacks/utils/reasoning_extraction/extract_reasoning.py``.
 */
import * as anthropic from "./extractAnthropic.js";
import * as gemini from "./extractGemini.js";
import * as generic from "./extractGeneric.js";
import * as openai from "./extractOpenai.js";
import { flattenSources, type Mapping, type ReasoningProvider } from "./helpers.js";

/** Provider modules tried by hint, in a fixed order. */
const PROVIDERS: readonly ReasoningProvider[] = [anthropic, gemini, openai];

/** Lower-cased text built from the hint (string, or array such as a LangChain id) and model name. */
function hintText(providerHint: unknown, modelName: unknown): string {
  const parts: unknown[] = Array.isArray(providerHint) ? [...providerHint] : [providerHint];
  parts.push(modelName);
  return parts
    .filter((part) => part !== undefined && part !== null && part !== "")
    .map(String)
    .join(" ")
    .toLowerCase();
}

/**
 * The model's reasoning settings as an object (values untouched), or ``null``.
 *
 * @param configSources raw config objects, constructor config first and per-call config
 *   last (later ones win). Non-objects are ignored.
 * @param providerHint class name or serialized id (string or array) identifying the provider.
 * @param modelName the model id, e.g. ``"claude-sonnet-4-5"`` (also used for matching).
 */
export function extractReasoning(
  configSources: Iterable<unknown> | null | undefined,
  providerHint?: unknown,
  modelName?: string | null,
): Mapping | null {
  try {
    const sources = flattenSources(configSources);
    if (sources.length === 0) return null;
    const hint = hintText(providerHint, modelName);
    const matched = PROVIDERS.filter((provider) => provider.matches(hint));
    const extractors: readonly ReasoningProvider[] = matched.length > 0 ? matched : [generic];
    const merged: Mapping = {};
    for (const extractor of extractors) Object.assign(merged, extractor.extract(sources) ?? {});
    return Object.keys(merged).length > 0 ? merged : null;
  } catch {
    // telemetry must never break the agent
    return null;
  }
}
