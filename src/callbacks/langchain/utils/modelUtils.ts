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

function firstNumber(source: Obj, ...keys: string[]): number | null {
  for (const key of keys) {
    const value = source[key];
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return null;
}

/**
 * ``{ input, output }`` token counts for one model call; ``null`` when unreported.
 *
 * ``AIMessage.usage_metadata`` (the LangChain standard) wins; otherwise fall back
 * to ``llmOutput.token_usage`` / ``tokenUsage`` / ``usage``, which use
 * ``prompt_tokens``/``promptTokens`` + ``completion_tokens``/``completionTokens``
 * (OpenAI) or ``input_tokens`` + ``output_tokens`` (Anthropic).
 */
export function extractTokenCounts(
  usageMetadata: unknown,
  llmOutput: unknown,
): { input: number | null; output: number | null } {
  const out = asObj(llmOutput);
  for (const source of [usageMetadata, out["token_usage"], out["tokenUsage"], out["usage"]]) {
    const usage = asObj(source);
    const input = firstNumber(usage, "input_tokens", "prompt_tokens", "promptTokens");
    const output = firstNumber(usage, "output_tokens", "completion_tokens", "completionTokens");
    if (input !== null || output !== null) return { input, output };
  }
  return { input: null, output: null };
}
