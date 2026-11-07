/**
 * Small, provider-neutral building blocks for the reasoning extractors.
 *
 * Everything here is defensive: it accepts anything (``null``, non-objects, getters
 * that throw are caught by the dispatcher) so a weird model config can only ever
 * result in "nothing found", never in a broken agent run.
 * Twin of ``trellar/callbacks/utils/reasoning_extraction/_helpers.py``.
 */

export type Mapping = Record<string, unknown>;

/**
 * Sub-objects of a model config that commonly carry extra, provider-specific request
 * settings. Reasoning options often live in one of them rather than at the top level
 * (Strands: ``params`` / ``additionalRequestFields``; LangChain: ``model_kwargs`` /
 * ``extra_body``). Only these names are ever descended into, one level deep.
 */
export const NESTED_SOURCE_KEYS = [
  "params",
  "model_kwargs",
  "modelKwargs",
  "additional_request_fields",
  "additionalRequestFields",
  "extra_body",
  "extraBody",
] as const;

/** *value* if it is a plain object (not an array), else ``null``. */
export function asMapping(value: unknown): Mapping | null {
  return value !== null && typeof value === "object" && !Array.isArray(value) ? (value as Mapping) : null;
}

/**
 * Expand raw config objects into one ordered list of plain objects.
 *
 * For each source we emit the source itself followed by its known nested sub-objects
 * (see ``NESTED_SOURCE_KEYS``). Non-objects are dropped. Order matters: a later object
 * wins over an earlier one when both define the same key, so callers should pass
 * constructor config first and per-call config last.
 */
export function flattenSources(configSources: Iterable<unknown> | null | undefined): Mapping[] {
  const flat: Mapping[] = [];
  for (const source of configSources ?? []) {
    const mapping = asMapping(source);
    if (mapping === null) continue;
    flat.push(mapping);
    for (const nestedKey of NESTED_SOURCE_KEYS) {
      const nested = asMapping(mapping[nestedKey]);
      if (nested !== null) flat.push(nested);
    }
  }
  return flat;
}

/**
 * Copy *only* ``allowedKeys`` out of *sources*, values untouched.
 *
 * This is the single place where values leave a model config, which is why it takes
 * an explicit allowlist: secrets such as ``api_key`` can never be picked up because
 * they are never asked for. A key whose value is ``null``/``undefined`` counts as
 * "not set". Returns ``null`` when nothing was found.
 */
export function pickKeys(sources: readonly Mapping[], allowedKeys: readonly string[]): Mapping | null {
  const found: Mapping = {};
  for (const source of sources) {
    for (const key of allowedKeys) {
      const value = source[key];
      if (value !== undefined && value !== null) found[key] = value;
    }
  }
  return Object.keys(found).length > 0 ? found : null;
}

/** A provider module: decides from the hint whether it applies, then extracts its own keys. */
export interface ReasoningProvider {
  KEYS: readonly string[];
  matches(hint: string): boolean;
  extract(sources: readonly Mapping[]): Mapping | null;
}
