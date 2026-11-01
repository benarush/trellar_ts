/** Flat key/value labels, e.g. ``{ customer_id: 5, topic: "billing" }``. */
export type Tags = Record<string, string | number | boolean>;

/**
 * Return a copy of ``tags`` with every value converted with ``String()`` (``{}`` for
 * ``undefined``/``null``). The backend enforces count/length/character limits.
 */
export function checkTags(tags?: Tags | null): Record<string, string> {
  if (tags == null) return {};
  if (typeof tags !== "object" || Array.isArray(tags))
    throw new TypeError("tags must be an object of key/value pairs, e.g. { customer_id: 5 }");
  return Object.fromEntries(Object.entries(tags).map(([k, v]) => [k, String(v)]));
}
