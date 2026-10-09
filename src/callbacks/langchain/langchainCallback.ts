import { BaseCallbackHandler } from "@langchain/core/callbacks/base";
import { AsyncLocalStorageProviderSingleton } from "@langchain/core/singletons";

import { evaluateWithTrellarAgent, ObservabilityMode, parseObservabilityMode } from "../../agentLoop.js";
import {
  activateTrellarAgent,
  type TrellarAgentState,
  registerTrellarAgentResolver,
  releaseTrellarAgent,
} from "../../context.js";
import { logger } from "../../logger.js";
import { buildContext, compactJson, hashTools, pyJsonDumps } from "../common.js";
import {
  contentToStr,
  extractLlmInput,
  extractModelName,
  findToolDescription,
  isMessageLike,
  normalizeTools,
  stringifyUnknown,
} from "./utils/index.js";

type Obj = Record<string, any>;

/** Marker used by the async-context resolver to recognise Trellar agents. */
export const TRELLAR_AGENT = Symbol.for("trellar.langchain.agent");

interface PendingLlmToolCall {
  event: Obj;
  parentRunId: string | null;
  remainingTools: Array<string | undefined>;
}

function truthy(value: unknown): boolean {
  if (value === null || value === undefined || value === false || value === 0 || value === "") return false;
  if (Array.isArray(value)) return value.length > 0;
  if (typeof value === "object") return Object.keys(value as object).length > 0;
  return true;
}

function lastOf(id: unknown): string | undefined {
  return Array.isArray(id) && id.length > 0 && typeof id[id.length - 1] === "string"
    ? (id[id.length - 1] as string)
    : undefined;
}

/**
 * Internal LangChain callback handler that tracks agent lifecycle events.
 *
 * This class is not part of the public API. Use ``trellarLangchainAgent`` to obtain an
 * instance.
 *
 * Accumulates all graph events into ``events`` as a list of plain objects.
 * State is reset at the start of each top-level ``graph.invoke()`` call
 * (detected via ``parentRunId === undefined``), so reusing one instance across
 * multiple invocations does not leak prior-run events into later payloads.
 * Each event contains:
 *
 * - ``event``         -- callback name (e.g. ``on_chat_model_start``)
 * - ``graph_order``   -- monotonically increasing step counter across the whole run
 * - ``trace_id``      -- root run_id (graph-level trace)
 * - ``run_id``        -- this specific run
 * - ``parent_run_id`` -- direct parent run (``null`` for the root)
 * - ``node_name``     -- human-readable name of the node/chain/tool/model
 * - ``node_type``     -- ``"llm"``, ``"tool"``, or ``"chain"``
 * - extra payload fields depending on the event type, notably ``is_mcp_tool``
 *   (boolean) on ``on_tool_end`` events -- see ``isMcpToolRun``.
 */
export class LangchainAgentCallback extends BaseCallbackHandler implements TrellarAgentState {
  name = "trellar_langchain_agent";
  // LangChain JS runs callbacks in the background by default; the events must
  // be recorded before the next node (e.g. the gate that calls
  // evaluateConfidence) starts, so make LangChain wait for this handler.
  override awaitHandlers = true;
  override raiseError = false;

  readonly [TRELLAR_AGENT] = true;

  /** Maps the LangChain message `type` attribute to a human-readable prefix. */
  static readonly MESSAGE_PREFIXES: Record<string, string> = {
    ai: "AI MESSAGE",
    human: "HUMAN MESSAGE",
    system: "SYSTEM MESSAGE",
    tool: "TOOL MESSAGE",
    function: "FUNCTION MESSAGE",
    chat: "CHAT MESSAGE",
    generic: "CHAT MESSAGE", // LangChain JS ChatMessage.type
  };

  readonly agentName: string;
  readonly observabilityMode: ObservabilityMode;
  traceId: string | null = null;
  events: Array<Record<string, unknown>> = [];
  isSingleCall = false;
  _step = 0;
  /** Whether evaluateConfidence() has already succeeded during this run. */
  _evaluated = false;
  /** run_id -> {name, type, mcp} */
  _runRegistry = new Map<string, { name: string | null | undefined; type: string; mcp?: boolean }>();
  /**
   * LLM events that requested tool calls and are still awaiting the
   * corresponding tool outputs.
   */
  _pendingLlmToolCalls: PendingLlmToolCall[] = [];
  /**
   * tools_hash -> tool schemas, captured from handleChatModelStart's
   * invocation params. Keyed by a content hash of the tool list rather than
   * the model name, since the model string lives in a different namespace than
   * the node_name the backend uses to attribute this call to a SubAgent -- the
   * same hash is stamped onto the matching on_chat_model_start event so the
   * backend can correlate a declared toolset back to its real caller. One
   * entry per distinct toolset for the whole run.
   */
  availableTools: Record<string, unknown[]> = {};
  /** name -> description, supplied via ``registerTools`` (survives across runs). */
  _registeredToolDescriptions = new Map<string, string>();

  constructor(options: { agentName: string; observabilityMode?: ObservabilityMode }) {
    super();
    const { agentName, observabilityMode = ObservabilityMode.NONE } = options;
    if (!agentName || !agentName.trim()) {
      throw new Error(
        "agent_name is required. It uniquely identifies this agent graph in the " +
          "Trellar backend and is used to track its network profile across runs. " +
          "Use a stable, descriptive name such as 'research-agent' or 'support-bot'.",
      );
    }
    this.agentName = agentName;
    this.observabilityMode = parseObservabilityMode(observabilityMode);
  }

  /**
   * Tell the Trellar agent about tools so their descriptions are reported on ``on_tool_start``.
   *
   * Python's LangChain serializes the tool (name *and* description) into every tool
   * callback. LangChain JS does not: a tool that is invoked directly
   * (``myTool.invoke(args, config)``) reaches the callback with its name only, and
   * descriptions are otherwise only learned from tools bound to a model
   * (``model.bindTools``). Register directly-invoked tools here to get the same
   * ``tool_description`` Python reports. JS-only addition; optional.
   */
  registerTools(tools: Array<{ name: string; description?: string }>): this {
    for (const tool of tools) {
      if (tool && typeof tool.name === "string" && typeof tool.description === "string" && tool.description) {
        this._registeredToolDescriptions.set(tool.name, tool.description);
      }
    }
    return this;
  }

  // ------------------------------------------------------------------
  // Message serialization
  // ------------------------------------------------------------------

  /**
   * Serialize a single LangChain message object into a labeled plain string.
   * Falls back to ``String()`` for anything that is not a recognised message object.
   */
  static serializeMessageObj(msg: unknown): string {
    if (isMessageLike(msg)) {
      const prefix = LangchainAgentCallback.MESSAGE_PREFIXES[String(msg.type).toLowerCase()] ?? "MESSAGE";
      let content: unknown = msg.content;
      if (typeof content !== "string") {
        // content can be a list of dicts (e.g. multimodal messages)
        try {
          content = pyJsonDumps(content, { sortKeys: false, ensureAscii: true });
        } catch {
          content = String(content);
        }
      }
      return `${prefix}: ${content}`;
    }
    return stringifyUnknown(msg);
  }

  /**
   * Convert a list of LangChain message objects into labeled plain strings.
   *
   * Each item is formatted as ``"<TYPE PREFIX>: <content>"`` so that the result
   * is JSON-serializable and unambiguous about which role produced the content.
   */
  static serializeMessages(messages: unknown[]): string[] {
    return messages.map((msg) => LangchainAgentCallback.serializeMessageObj(msg));
  }

  /**
   * Whether a tool run came from a ``@langchain/mcp-adapters`` tool.
   *
   * ``@langchain/mcp-adapters`` builds every MCP-derived tool as a
   * ``DynamicStructuredTool`` with ``responseFormat: "content_and_artifact"``
   * and ``metadata: { annotations: ... }`` (the key is always present, even
   * when the server sent no annotations); its results carry the MCP artifacts
   * on ``ToolMessage.artifact`` (entries typed ``mcp_*``). A plain local tool
   * has neither marker.
   *
   * This is a generic fingerprint of ``@langchain/mcp-adapters`` usage --
   * present on every MCP tool call regardless of transport (stdio, SSE,
   * streamable HTTP). A hand-rolled MCP client that does not go through the
   * adapter's conversion helpers is not detected.
   */
  static isMcpToolRun(toolMetadata: unknown, output: unknown): boolean {
    if (toolMetadata !== null && typeof toolMetadata === "object" && "annotations" in (toolMetadata as object)) {
      return true;
    }
    const artifact = (output as Obj | null | undefined)?.["artifact"];
    if (Array.isArray(artifact)) {
      return artifact.some(
        (a) => a && typeof a === "object" && typeof a.type === "string" && a.type.startsWith("mcp_"),
      );
    }
    return false;
  }

  // ------------------------------------------------------------------
  // Internal helpers
  // ------------------------------------------------------------------

  _nextStep(): number {
    this._step += 1;
    return this._step;
  }

  _register(runId: string, name: string | null | undefined, nodeType: string, extra: { mcp?: boolean } = {}): void {
    this._runRegistry.set(runId, { name, type: nodeType, ...extra });
  }

  _record(event: string, runId: string, parentRunId: string | undefined | null, data: Obj = {}): void {
    const nodeInfo = this._runRegistry.get(runId);
    this.events.push(
      toJsonableEvent({
        event,
        graph_order: this._nextStep(),
        trace_id: this.traceId === null ? "None" : this.traceId,
        run_id: runId,
        parent_run_id: parentRunId ? parentRunId : null,
        node_name: nodeInfo?.name ?? null,
        node_type: nodeInfo?.type ?? null,
        ...data,
      }),
    );
  }

  /**
   * Recursively coerce *value* into something ``JSON.stringify`` can handle.
   *
   * LangChain/LangGraph hand callbacks all sorts of raw objects that are not
   * JSON-safe out of the box -- most notably message objects and class
   * instances. This makes sure nothing appended to ``events`` can ever break
   * the ``evaluateConfidence()`` HTTP call downstream.
   */
  static toJsonable(value: unknown): unknown {
    return toJsonableEvent(value);
  }

  /** Reset per-root-run state (Python: the ``parent_run_id is None`` block of on_chain_start). */
  _resetForNewRun(runId: string): void {
    this.traceId = runId;
    this.events = [];
    this._step = 0;
    this._runRegistry = new Map();
    this._pendingLlmToolCalls = [];
    this.availableTools = {};
    this._evaluated = false;
    // Self-register so evaluateConfidence() can pick us up automatically.
    activateTrellarAgent(this);
  }

  // ------------------------------------------------------------------
  // LLM events
  // ------------------------------------------------------------------

  override handleLLMStart(
    llm: Obj,
    prompts: string[],
    runId: string,
    parentRunId?: string,
    extraParams?: Obj,
  ): void {
    const model = extractModelName(llm, extraParams?.["invocation_params"]);
    this._register(runId, model, "llm");
    this._record("on_llm_start", runId, parentRunId, {
      model: model ?? null,
      input: { system: null, human: prompts.join("\n") },
    });
  }

  override handleChatModelStart(
    llm: Obj,
    messages: unknown[][],
    runId: string,
    parentRunId?: string,
    extraParams?: Obj,
  ): void {
    const invocationParams = extraParams?.["invocation_params"];
    const model = extractModelName(llm, invocationParams);
    const rawTools =
      (invocationParams as Obj | undefined)?.["tools"] ?? (extraParams?.["options"] as Obj | undefined)?.["tools"];
    let toolsHash: string | null = null;
    if (truthy(rawTools)) {
      const tools = normalizeTools(rawTools);
      if (tools.length > 0) {
        toolsHash = hashTools(tools);
        this.availableTools[toolsHash] = tools;
      }
    }
    this._register(runId, model, "llm");

    // messages is BaseMessage[][] -- one inner list per prompt batch item.
    // Use the first batch to extract system/human fields.
    const llmInput = extractLlmInput(messages && messages.length > 0 ? messages[0]! : []);
    this._record("on_chat_model_start", runId, parentRunId, {
      model: model ?? null,
      input: llmInput,
      tools_hash: toolsHash,
    });
  }

  override handleLLMEnd(response: Obj, runId: string, parentRunId?: string): void {
    // Pull the text from the first generation of the first batch.
    // Try multiple attributes in priority order to handle:
    //   - ChatGeneration (.message.content) -- standard LangChain chat models
    //   - Generation (.text) -- plain (non-chat) LLMs
    //   - Gemini multimodal content (list of parts -- serialise to JSON)
    let responseText = "";
    let toolCalls: any[] = [];
    const generations = response?.["generations"];
    if (Array.isArray(generations) && generations.length > 0) {
      const firstBatch = generations[0];
      if (Array.isArray(firstBatch) && firstBatch.length > 0) {
        const gen = firstBatch[0] as Obj;

        // 1. Try .message.content (ChatGeneration)
        let message = gen?.["message"] as Obj | null | undefined;
        if (message !== null && message !== undefined) {
          toolCalls = (message["tool_calls"] as any[] | undefined) || [];
          const content = message["content"];
          if (content !== null && content !== undefined) {
            responseText = contentToStr(content);
          } else {
            // Content is None -- fall through to .text
            message = undefined;
          }
        }

        // 2. Try .text (plain Generation or ChatGeneration fallback)
        if (!message) {
          const rawText = gen?.["text"];
          if (rawText !== null && rawText !== undefined) {
            // .text can be a list when Gemini returns multimodal content
            responseText = typeof rawText === "string" ? rawText || "" : contentToStr(rawText);
          }
        }
      }
    }

    const llmOutput = (response?.["llmOutput"] as Obj | undefined) ?? {};
    const tokenUsage = truthy(llmOutput["token_usage"] ?? llmOutput["tokenUsage"])
      ? (llmOutput["token_usage"] ?? llmOutput["tokenUsage"])
      : (llmOutput["usage"] ?? null);

    // Fold tool calls into the response text so the payload keeps its original
    // shape ({"response": <str>}). The matching tool results are appended
    // retroactively by handleToolEnd once each tool finishes.
    if (toolCalls.length > 0) {
      const parts: string[] = responseText ? [responseText] : [];
      for (const tc of toolCalls) {
        if (tc !== null && typeof tc === "object") {
          parts.push(`TOOL CALL: ${tc.name}(args=${compactJson(tc.args)})`);
        }
      }
      responseText = parts.join("\n");
    }

    this._record("on_llm_end", runId, parentRunId, {
      output: { response: responseText },
      token_usage: tokenUsage,
    });

    if (toolCalls.length > 0) {
      // _record appends a jsonable copy; keep a reference to that copy so
      // handleToolEnd can enrich it in place before the payload is built.
      this._pendingLlmToolCalls.push({
        event: this.events[this.events.length - 1] as Obj,
        parentRunId: parentRunId ? parentRunId : null,
        remainingTools: toolCalls
          .filter((tc) => tc !== null && typeof tc === "object")
          .map((tc) => tc.name as string | undefined),
      });
    }
  }

  override handleLLMError(error: unknown, runId: string, parentRunId?: string): void {
    this._record("on_llm_error", runId, parentRunId, { error: errorText(error) });
  }

  // ------------------------------------------------------------------
  // Tool events
  // ------------------------------------------------------------------

  override handleToolStart(
    tool: Obj,
    input: string,
    runId: string,
    parentRunId?: string,
    _tags?: string[],
    metadata?: Obj,
    runName?: string,
  ): void {
    const toolName = runName || (tool?.["name"] as string | undefined) || lastOf(tool?.["id"]) || null;
    const toolDescription =
      (typeof tool?.["description"] === "string" && tool["description"]) ||
      (toolName ? this._registeredToolDescriptions.get(toolName) : undefined) ||
      findToolDescription(this.availableTools, toolName);
    this._register(runId, toolName, "tool", { mcp: LangchainAgentCallback.isMcpToolRun(metadata, undefined) });

    // The parsed argument dict, when the input is JSON (LangChain JS passes the
    // arguments as a JSON string).
    let parsedInputs: unknown = null;
    try {
      parsedInputs = JSON.parse(input);
    } catch {
      parsedInputs = null;
    }

    this._record("on_tool_start", runId, parentRunId, {
      tool: toolName,
      tool_description: toolDescription || null,
      input: {
        raw: input,
        parsed: parsedInputs,
      },
      // parent_run_id is already in the record; surface it explicitly so
      // callers can link this tool call back to the LLM that invoked it.
      invoked_by_run_id: parentRunId ? parentRunId : null,
    });
  }

  override handleToolEnd(output: unknown, runId: string, parentRunId?: string): void {
    // Must run BEFORE serialization: serializeMessageObj only looks at
    // `.type`/`.content` and would otherwise silently drop `.artifact`.
    const registered = this._runRegistry.get(runId);
    const isMcpTool = Boolean(registered?.mcp) || LangchainAgentCallback.isMcpToolRun(undefined, output);
    const serializedOutput = LangchainAgentCallback.serializeMessageObj(output);
    this._record("on_tool_end", runId, parentRunId, { output: serializedOutput, is_mcp_tool: isMcpTool });
    this._attachToolResponseToLlm(registered?.name ?? undefined, serializedOutput, parentRunId);
  }

  /**
   * Retroactively attach a finished tool's output to the LLM event that
   * requested it, so the LLM's recorded output is never just an empty string.
   *
   * Matching strategy (most recent entries first):
   * 1. A pending LLM event sharing the same ``parentRunId`` (the LLM and the
   *    tool live in the same chain/node -- direct-invocation pattern).
   * 2. Any pending LLM event still awaiting this tool name (covers
   *    ToolNode/react-agent graphs where the tool runs under a different
   *    parent chain).
   */
  _attachToolResponseToLlm(
    toolName: string | undefined,
    serializedOutput: string,
    parentRunId: string | undefined,
  ): void {
    if (!toolName || this._pendingLlmToolCalls.length === 0) return;

    const parentId = parentRunId ? parentRunId : null;
    let match: PendingLlmToolCall | undefined;
    for (let i = this._pendingLlmToolCalls.length - 1; i >= 0; i--) {
      const entry = this._pendingLlmToolCalls[i]!;
      if (!entry.remainingTools.includes(toolName)) continue;
      if (entry.parentRunId === parentId) {
        match = entry;
        break;
      }
      if (match === undefined) match = entry;
    }
    if (match === undefined) return;

    match.event["output"]["response"] += `\nTOOL RESPONSE [${toolName}]: ${serializedOutput}`;
    match.remainingTools.splice(match.remainingTools.indexOf(toolName), 1);
    if (match.remainingTools.length === 0) {
      this._pendingLlmToolCalls.splice(this._pendingLlmToolCalls.indexOf(match), 1);
    }
  }

  override handleToolError(error: unknown, runId: string, parentRunId?: string): void {
    this._record("on_tool_error", runId, parentRunId, { error: errorText(error) });
  }

  // ------------------------------------------------------------------
  // Chain / Graph node events
  // ------------------------------------------------------------------

  // NOTE: LangChain JS invokes handlers with
  // (chain, inputs, runId, parentRunId, tags, metadata, runType, runName, extra).
  override handleChainStart(
    chain: Obj,
    inputs: unknown,
    runId: string,
    parentRunId?: string,
    _tags?: string[],
    metadata?: Obj,
    _runType?: string,
    runName?: string,
  ): void {
    if (parentRunId === undefined || parentRunId === null) {
      // Root invocation -- this runId is the graph-level trace ID. Guard
      // against reused instances: if this handler is passed into more than one
      // top-level graph.invoke() (sequentially), drop state from the previous
      // run instead of letting it accumulate unbounded.
      this._resetForNewRun(runId);
    }

    const chainName = runName || (chain?.["name"] as string | undefined) || lastOf(chain?.["id"]);

    // `inputs` is usually the graph/node state dict, but LangChain also fires
    // this event for internal sub-runnables (e.g. ToolNode) whose raw input is
    // a bare list of messages or a single message object. The backend's schema
    // requires `inputs` to always be a list, so every branch below normalizes
    // to one instead of a bare string/dict.
    let safeInputs: unknown[];
    const inp = inputs as Obj | unknown[] | null | undefined;
    if (inp !== null && typeof inp === "object" && !Array.isArray(inp) && "messages" in inp) {
      safeInputs = LangchainAgentCallback.serializeMessages(toArray((inp as Obj)["messages"]));
    } else if (Array.isArray(inp)) {
      safeInputs = LangchainAgentCallback.serializeMessages(inp);
    } else {
      safeInputs = [toJsonableEvent(inputs)];
    }

    // LangGraph stamps its own superstep number onto the RunnableConfig
    // metadata for each Pregel node task. Nodes sharing the same
    // langgraph_step ran in the same superstep -- i.e. in parallel -- which
    // lets the backend distinguish true fan-out branches from a sequential
    // chain. null for non-LangGraph callers.
    const langgraphStep = metadata?.["langgraph_step"];

    this._register(runId, chainName, "chain");
    this._record("on_chain_start", runId, parentRunId, {
      inputs: safeInputs,
      langgraph_step: typeof langgraphStep === "number" ? langgraphStep : null,
    });
  }

  override async handleChainEnd(outputs: unknown, runId: string, parentRunId?: string): Promise<void> {
    // See handleChainStart -- `outputs` is not always a dict.
    let out = outputs;
    if (out !== null && typeof out === "object" && !Array.isArray(out) && "messages" in (out as Obj)) {
      out = {
        ...(out as Obj),
        messages: LangchainAgentCallback.serializeMessages(toArray((out as Obj)["messages"])),
      };
    }
    this._record("on_chain_end", runId, parentRunId, { outputs: toJsonableEvent(out) });
    if (parentRunId === undefined || parentRunId === null) {
      // Root run ending -- the whole graph flow has reached its end.
      await this._maybeAutoEvaluate();
      // Release the slot so the next top-level run starts from a clean state
      // instead of inheriting this run's handler.
      releaseTrellarAgent(this);
    }
  }

  /**
   * Auto-trigger evaluateConfidence() per observabilityMode.
   *
   * Errors are caught and logged, never raised, so a passive observability
   * call can never crash the graph.
   */
  async _maybeAutoEvaluate(): Promise<void> {
    if (this.observabilityMode === ObservabilityMode.NONE) return;
    if (this.observabilityMode === ObservabilityMode.IF_NOT_EVALUATED && this._evaluated) return;
    try {
      await evaluateWithTrellarAgent(this, { _observabilityCall: true });
    } catch (error) {
      logger.warning("Auto-triggered evaluateConfidence() failed", error);
    }
  }

  override handleChainError(error: unknown, runId: string, parentRunId?: string): void {
    this._record("on_chain_error", runId, parentRunId, { error: errorText(error) });
    if (parentRunId === undefined || parentRunId === null) {
      // Root run failing -- handleChainEnd will never fire for this runId
      // (they are mutually exclusive), so release the slot here too.
      releaseTrellarAgent(this);
    }
  }

  // ------------------------------------------------------------------
  // Context serialization
  // ------------------------------------------------------------------

  /** Serialize collected events into a step-by-step string for the backend. */
  buildContext(): string {
    return buildContext(this.events, this.traceId);
  }
}

function toArray(value: unknown): unknown[] {
  if (Array.isArray(value)) return value;
  return value === undefined || value === null ? [] : [value];
}

function errorText(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}

/**
 * Coerce *value* to JSON-safe data. Message objects become labeled strings
 * (``"AI MESSAGE: ..."``), mirroring the Python callback's ``_to_jsonable``.
 */
function toJsonableEvent(value: unknown, seen: WeakSet<object> = new WeakSet()): any {
  if (value === null || value === undefined) return null;
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") return value;
  if (typeof value === "bigint") return Number.isSafeInteger(Number(value)) ? Number(value) : value.toString();
  if (typeof value === "function" || typeof value === "symbol") return String(value);

  const obj = value as Obj;
  if (seen.has(obj)) return String(obj);
  seen.add(obj);
  try {
    if (isMessageLike(obj)) return LangchainAgentCallback.serializeMessageObj(obj);
    if (Array.isArray(obj)) return obj.map((v) => toJsonableEvent(v, seen));
    if (obj instanceof Map) {
      const out: Obj = {};
      for (const [k, v] of obj) out[String(k)] = toJsonableEvent(v, seen);
      return out;
    }
    if (obj instanceof Set) return [...obj].map((v) => toJsonableEvent(v, seen));
    if (obj instanceof Date) return obj.toISOString();
    if (obj instanceof Error) return String(obj);
    if (typeof obj["toJSON"] === "function") {
      try {
        return toJsonableEvent((obj["toJSON"] as () => unknown).call(obj), seen);
      } catch {
        return String(obj);
      }
    }
    const out: Obj = {};
    for (const key of Object.keys(obj)) out[key] = toJsonableEvent(obj[key], seen);
    return out;
  } finally {
    seen.delete(obj);
  }
}

// ----------------------------------------------------------------------
// Async-context resolver (LangChain runnable config)
// ----------------------------------------------------------------------

/**
 * Inside any LangGraph node / tool / runnable, LangChain JS exposes the
 * running config through its own AsyncLocalStorage. The config's callbacks
 * contain this run's Trellar agent, so concurrent ``graph.invoke()`` calls each see
 * their own Trellar agent with no extra wiring.
 */
function resolveFromRunnableConfig(): TrellarAgentState | undefined {
  const config = AsyncLocalStorageProviderSingleton.getRunnableConfig() as Obj | undefined;
  const callbacks = config?.["callbacks"];
  if (!callbacks) return undefined;
  const handlers: unknown[] = Array.isArray(callbacks)
    ? callbacks
    : [...((callbacks as Obj)["handlers"] ?? []), ...((callbacks as Obj)["inheritableHandlers"] ?? [])];
  for (const handler of handlers) {
    if (handler && typeof handler === "object" && (handler as Obj)[TRELLAR_AGENT as unknown as string] === true) {
      return handler as TrellarAgentState;
    }
  }
  return undefined;
}

registerTrellarAgentResolver(resolveFromRunnableConfig);
