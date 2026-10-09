type Obj = Record<string, unknown>;

function asObj(value: unknown): Obj {
  return value !== null && typeof value === "object" ? (value as Obj) : {};
}

function str(value: unknown): string | undefined {
  return typeof value === "string" && value !== "" ? value : undefined;
}

/**
 * Pull the real model identifier out of a serialized LLM dict.
 *
 * LangChain puts the *class* name in ``serialized.name`` / the last element of
 * ``serialized.id`` (e.g. "ChatOpenAI") but the actual model string
 * (e.g. "gpt-4o") lives inside ``kwargs`` -- and, for the JS packages, in the
 * call's ``invocation_params``.
 */
export function extractModelName(
  serialized: unknown,
  invocationParams?: unknown,
): string | undefined {
  const ser = asObj(serialized);
  const kwargs = asObj(ser["kwargs"]);
  const params = asObj(invocationParams);

  const fromKwargs =
    str(kwargs["model_name"]) ?? str(kwargs["model"]) ?? str(kwargs["modelName"]) ?? str(kwargs["model_id"]);
  if (fromKwargs) return fromKwargs;

  const fromParams =
    str(params["model_name"]) ?? str(params["model"]) ?? str(params["modelName"]) ?? str(params["model_id"]);
  if (fromParams) return fromParams;

  // Fall back to the class name so we always have something.
  const direct = str(ser["name"]);
  if (direct) return direct;
  const id = ser["id"];
  if (Array.isArray(id) && id.length > 0) return str(id[id.length - 1]);
  return undefined;
}
