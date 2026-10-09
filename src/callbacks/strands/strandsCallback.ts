/**
 * Strands Agents guard: turns Strands hook events into the same event payload
 * the LangChain callback produces (chain / llm / tool events).
 *
 * Mapping:
 *
 *     Graph/Swarm run -> root chain      Graph node   -> chain (nodeId)
 *     Agent invoke    -> chain (agent.name)
 *     Model call      -> on_chat_model_start / on_llm_end
 *     Tool call       -> on_tool_start / on_tool_end
 */
import { randomUUID } from "node:crypto";

import {
  AfterInvocationEvent,
  AfterModelCallEvent,
  AfterToolCallEvent,
  BeforeInvocationEvent,
  BeforeModelCallEvent,
  BeforeToolCallEvent,
  ExecuteToolStage,
  MessageAddedEvent,
  type LocalAgent,
  type Plugin,
} from "@strands-agents/sdk";
import {
  AfterMultiAgentInvocationEvent,
  AfterNodeCallEvent,
  BeforeMultiAgentInvocationEvent,
  BeforeNodeCallEvent,
  type MultiAgent,
  type MultiAgentPlugin,
} from "@strands-agents/sdk/multiagent";

import { type AgentLoopResult, evaluateWithGuard, ObservabilityMode, parseObservabilityMode } from "../../agentLoop.js";
import {
  activateGuard,
  enterScope,
  type GuardState,
  currentScope,
  releaseGuard,
  runInScope,
} from "../../context.js";
import { logger } from "../../logger.js";
import { buildContext, compactJson, hashTools, toJsonable } from "../common.js";
import {
  isMcpTool,
  lastAssistantText,
  llmHumanInput,
  modelName,
  openaiTools,
  systemPromptText,
  textOfBlocks,
} from "./utils/index.js";

type Obj = Record<string, any>;

function errorText(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}

/** Render a graph / swarm input (string, content blocks, ...) as text. */
function inputText(input: unknown): string {
  if (input === undefined || input === null) return "";
  if (typeof input === "string") return input;
  if (Array.isArray(input)) {
    const text = textOfBlocks(input);
    return text || compactJson(input);
  }
  return compactJson(input);
}

/**
 * Hook provider that records a Strands run for the Trellar backend.
 *
 * Not public API; use ``getStrandsGuard``. Register it as a plugin on every
 * Agent (``new Agent({ plugins: [guard] })``) and on the Graph/Swarm
 * (``new Graph({ ..., plugins: [guard] })``).
 */
export class StrandsGuardCallback implements Plugin, MultiAgentPlugin, GuardState {
  readonly name = "trellar-strands-guard";

  readonly agentName: string;
  readonly observabilityMode: ObservabilityMode;
  isSingleCall = false;

  traceId: string | null = null;
  events: Array<Record<string, unknown>> = [];
  availableTools: Record<string, unknown[]> = {};
  _evaluated = false;
  _step = 0;
  _rootRunId: string | null = null;
  /** chain/tool run_id -> parent run_id */
  _parents = new Map<string, string | null>();
  /** agent -> open invocation run_id */
  _agentRuns = new Map<object, string>();
  /** agent -> open model-call run_id */
  _modelRuns = new Map<object, string>();
  /** toolUseId -> open tool run_id */
  _toolRuns = new Map<string, string>();
  /** toolUseId -> LLM event that asked for it */
  _toolLlmEvent = new Map<string, Obj>();
  /** open Graph/Swarm run_id */
  _multiRun: string | null = null;
  /** "<orchestratorId>\0<nodeId>" -> open node run_id */
  _nodeRuns = new Map<string, string>();
  /** agent / nested orchestrator -> the node run it executes under */
  _executorParents = new Map<object, string>();
  /** nodes whose ``stream`` was wrapped to run in this guard's scope */
  _patchedNodes = new WeakSet<object>();
  /** root / agent chain events whose input text is only known after the hook fired */
  _pendingInputEvents = new Map<string, Obj>();

  constructor(options: { agentName: string; observabilityMode?: ObservabilityMode }) {
    const { agentName, observabilityMode = ObservabilityMode.NONE } = options;
    if (!agentName || !agentName.trim()) {
      throw new Error(
        "agent_name is required. It uniquely identifies this agent network in the " +
          "Trellar backend. Use a stable, descriptive name such as 'research-agent'.",
      );
    }
    this.agentName = agentName;
    this.observabilityMode = parseObservabilityMode(observabilityMode);
    this._reset(null);
  }

  // ------------------------------------------------------------------
  // State
  // ------------------------------------------------------------------

  /** Clear all per-run state; called when a new root run starts. */
  _reset(rootRunId: string | null): void {
    this.traceId = rootRunId;
    this.events = [];
    this.availableTools = {};
    this._step = 0;
    this._evaluated = false;
    this._rootRunId = rootRunId;
    this._parents = new Map();
    this._agentRuns = new Map();
    this._modelRuns = new Map();
    this._toolRuns = new Map();
    this._toolLlmEvent = new Map();
    this._multiRun = null;
    this._nodeRuns = new Map();
    this._executorParents = new Map();
    this._pendingInputEvents = new Map();
  }

  _record(
    event: string,
    runId: string,
    parentRunId: string | null,
    nodeName: string | null | undefined,
    nodeType: string,
    data: Obj = {},
  ): Obj {
    this._step += 1;
    const record = toJsonable({
      event,
      graph_order: this._step,
      trace_id: this.traceId,
      run_id: runId,
      parent_run_id: parentRunId,
      node_name: nodeName ?? null,
      node_type: nodeType,
      ...data,
    }) as Obj;
    this.events.push(record);
    return record;
  }

  // ------------------------------------------------------------------
  // Chain helpers (graph, graph node, agent invocation)
  // ------------------------------------------------------------------

  _startChain(name: string | undefined, inputs: unknown[], explicitParent?: string): string {
    const runId = randomUUID();
    let parent: string | null;
    if (this._rootRunId === null) {
      // First run: this is the root. Reset state and become the active guard.
      this._reset(runId);
      parent = null;
      activateGuard(this);
      enterScope({ guard: this });
    } else {
      parent = explicitParent ?? currentScope()?.runId ?? this._rootRunId;
    }
    this._parents.set(runId, parent);
    const record = this._record("on_chain_start", runId, parent, name, "chain", { inputs });
    if (inputs.length === 0 || inputs[0] === "") this._pendingInputEvents.set(runId, record);
    return runId;
  }

  _endChain(runId: string, name: string | undefined, outputs: unknown): void {
    this._pendingInputEvents.delete(runId);
    const parent = this._parents.get(runId) ?? null;
    this._record("on_chain_end", runId, parent, name, "chain", { outputs });
  }

  _errorChain(runId: string, name: string | undefined, error: unknown): void {
    this._pendingInputEvents.delete(runId);
    const parent = this._parents.get(runId) ?? null;
    this._record("on_chain_error", runId, parent, name, "chain", { error: errorText(error) });
  }

  /** Fill in a chain's ``inputs`` once the framework exposes the text (it is not known when the Before* hook fires). */
  _fillInput(runId: string | undefined, text: string): void {
    if (!runId || !text) return;
    const record = this._pendingInputEvents.get(runId);
    if (!record) return;
    record["inputs"] = [text];
    this._pendingInputEvents.delete(runId);
  }

  async _finishRoot(): Promise<void> {
    await this._maybeAutoEvaluate();
    // Release the slot so the next run starts clean; a new root re-resets state.
    releaseGuard(this);
    this._rootRunId = null;
  }

  /** Auto-call evaluateConfidence() per observabilityMode; never throws. */
  async _maybeAutoEvaluate(): Promise<void> {
    if (this.observabilityMode === ObservabilityMode.NONE) return;
    if (this.observabilityMode === ObservabilityMode.IF_NOT_EVALUATED && this._evaluated) return;
    try {
      await evaluateWithGuard(this, { _observabilityCall: true });
    } catch (error) {
      logger.warning("Auto-triggered evaluateConfidence() failed", error);
    }
  }

  // ------------------------------------------------------------------
  // Hook registration
  // ------------------------------------------------------------------

  /** Strands ``Plugin`` entry point: called once per Agent the guard is attached to. */
  initAgent(agent: LocalAgent): void {
    agent.addHook(BeforeInvocationEvent, (e) => this._onAgentStart(e));
    agent.addHook(MessageAddedEvent, (e) => this._onMessageAdded(e));
    agent.addHook(AfterInvocationEvent, (e) => this._onAgentEnd(e));
    agent.addHook(BeforeModelCallEvent, (e) => this._onModelStart(e));
    agent.addHook(AfterModelCallEvent, (e) => this._onModelEnd(e));
    agent.addHook(BeforeToolCallEvent, (e) => this._onToolStart(e));
    agent.addHook(AfterToolCallEvent, (e) => this._onToolEnd(e));

    // Run every tool execution inside an async scope bound to this guard and
    // the tool's run id. Code called from a tool (``evaluateConfidence()``, or
    // an agent used as a tool) then resolves the right guard / parent run even
    // when several runs are executing concurrently in the same process.
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const guard = this;
    if (typeof agent.addMiddleware === "function") {
      agent.addMiddleware(ExecuteToolStage, async function* (context, next) {
        const scope = { guard, runId: guard._toolRuns.get(context.toolUse.toolUseId) };
        const iterator = next(context);
        while (true) {
          const step = await runInScope(scope, () => iterator.next());
          if (step.done) return step.value;
          yield step.value;
        }
      });
    }
  }

  /** Strands ``MultiAgentPlugin`` entry point: called once per Graph/Swarm. */
  initMultiAgent(orchestrator: MultiAgent): void {
    this._scopeNodes(orchestrator as unknown as Obj);
    orchestrator.addHook(BeforeMultiAgentInvocationEvent, (e) => this._onMultiStart(e));
    orchestrator.addHook(AfterMultiAgentInvocationEvent, (e) => this._onMultiEnd(e));
    orchestrator.addHook(BeforeNodeCallEvent, (e) => this._onNodeStart(e));
    orchestrator.addHook(AfterNodeCallEvent, (e) => this._onNodeEnd(e));
  }

  /**
   * Run every node of ``orchestrator`` inside an async scope bound to this guard
   * and the node's run id. A hook cannot leak an ``AsyncLocalStorage`` value into
   * the node execution, so this is what lets ``evaluateConfidence()`` called from
   * a custom node (or an agent's tool) find the right guard when several graphs
   * run concurrently in one process.
   */
  _scopeNodes(orchestrator: Obj): void {
    const nodes = orchestrator["nodes"] as Map<string, Obj> | undefined;
    if (!nodes || typeof nodes.forEach !== "function") return;
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const guard = this;
    nodes.forEach((node, nodeId) => {
      if (guard._patchedNodes.has(node) || typeof node["stream"] !== "function") return;
      guard._patchedNodes.add(node);
      const original = node["stream"].bind(node) as (...args: unknown[]) => AsyncGenerator<unknown, unknown, undefined>;
      node["stream"] = async function* (...args: unknown[]) {
        const generator = original(...args);
        const scope = () => ({
          guard,
          runId: guard._nodeRuns.get(`${orchestrator["id"]}\u0000${nodeId}`),
        });
        try {
          while (true) {
            const step = await runInScope(scope(), () => generator.next());
            if (step.done) return step.value;
            yield step.value;
          }
        } finally {
          await generator.return?.(undefined);
        }
      };
    });
  }

  // ------------------------------------------------------------------
  // Graph / Swarm events
  // ------------------------------------------------------------------

  static _multiName(source: Obj): string {
    return String(source?.["id"] || source?.constructor?.name);
  }

  _onMultiStart(event: Obj): void {
    const orchestrator = event["orchestrator"] as Obj;
    const task = inputText(event["state"]?.["_pendingInput"]);
    // A nested orchestrator (a node of a bigger graph) hangs under that node's run.
    const explicitParent = this._executorParents.get(orchestrator);
    this._multiRun = this._startChain(StrandsGuardCallback._multiName(orchestrator), [task], explicitParent);
  }

  async _onMultiEnd(event: Obj): Promise<void> {
    if (this._multiRun) {
      const runId = this._multiRun;
      this._multiRun = null;
      this._endChain(runId, StrandsGuardCallback._multiName(event["orchestrator"]), {});
      if (runId === this._rootRunId) await this._finishRoot();
    }
  }

  _onNodeStart(event: Obj): void {
    const orchestrator = event["orchestrator"] as Obj;
    const nodeId = event["nodeId"] as string;
    // The graph input is only stashed on the state after the root hook fired.
    this._fillInput(this._multiRun ?? undefined, inputText(event["state"]?.["_pendingInput"]));

    const parent = this._executorParents.get(orchestrator);
    const runId = this._startChain(nodeId, [], parent);
    this._nodeRuns.set(`${orchestrator["id"]}\u0000${nodeId}`, runId);

    // Whatever executes inside this node (an Agent, a nested Graph) hangs under its run.
    const node = orchestrator["nodes"]?.get?.(nodeId) as Obj | undefined;
    const executor = node?.["agent"] ?? node?.["orchestrator"];
    if (executor && typeof executor === "object") this._executorParents.set(executor, runId);
  }

  async _onNodeEnd(event: Obj): Promise<void> {
    const orchestrator = event["orchestrator"] as Obj;
    const nodeId = event["nodeId"] as string;
    const key = `${orchestrator["id"]}\u0000${nodeId}`;
    const runId = this._nodeRuns.get(key);
    if (runId === undefined) return;
    this._nodeRuns.delete(key);

    if (event["error"]) {
      this._errorChain(runId, nodeId, event["error"]);
    } else {
      const state = event["state"] as Obj | undefined;
      let content: unknown = state?.["node"]?.(nodeId)?.["content"];
      if (!Array.isArray(content) || content.length === 0) {
        const results = (state?.["results"] as Obj[] | undefined) ?? [];
        const last = [...results].reverse().find((r) => r?.["nodeId"] === nodeId);
        content = last?.["content"];
      }
      this._endChain(runId, nodeId, textOfBlocks(content));
    }
    if (runId === this._rootRunId) await this._finishRoot();
  }

  // ------------------------------------------------------------------
  // Agent invocation events
  // ------------------------------------------------------------------

  _onAgentStart(event: Obj): void {
    const agent = event["agent"] as Obj;
    const explicitParent = this._executorParents.get(agent);
    // The user message is appended to agent.messages only after this hook
    // fires (and a reused agent still holds the previous turn), so the prompt
    // is filled in by _onMessageAdded.
    // (Start the chain first: a root start replaces the per-run maps.)
    const runId = this._startChain(agent["name"], [""], explicitParent);
    this._agentRuns.set(agent, runId);
  }

  _onMessageAdded(event: Obj): void {
    const message = event["message"] as Obj;
    if (message?.["role"] !== "user") return;
    const agent = event["agent"] as object;
    this._fillInput(this._agentRuns.get(agent), textOfBlocks(message["content"]));
  }

  async _onAgentEnd(event: Obj): Promise<void> {
    const agent = event["agent"] as Obj;
    const runId = this._agentRuns.get(agent);
    if (runId === undefined) return;
    this._agentRuns.delete(agent);
    this._executorParents.delete(agent);
    this._endChain(runId, agent["name"], lastAssistantText(agent["messages"]));
    if (runId === this._rootRunId) await this._finishRoot();
  }

  // ------------------------------------------------------------------
  // Model events
  // ------------------------------------------------------------------

  _onModelStart(event: Obj): void {
    const agent = event["agent"] as Obj;
    const runId = randomUUID();
    this._modelRuns.set(agent, runId);

    let toolsHash: string | null = null;
    const tools = openaiTools(agent);
    if (tools.length > 0) {
      toolsHash = hashTools(tools);
      this.availableTools[toolsHash] = tools;
    }

    const model = modelName(agent);
    const parent = this._agentRuns.get(agent) ?? this._rootRunId;
    this._parents.set(runId, parent);
    this._record("on_chat_model_start", runId, parent, model, "llm", {
      model,
      input: { system: systemPromptText(agent), human: llmHumanInput(agent["messages"]) },
      tools_hash: toolsHash,
    });
  }

  _onModelEnd(event: Obj): void {
    const agent = event["agent"] as Obj;
    const runId = this._modelRuns.get(agent);
    if (runId === undefined) return;
    this._modelRuns.delete(agent);
    const parent = this._parents.get(runId) ?? null;
    const model = modelName(agent);

    const stopData = event["stopData"] as Obj | undefined;
    if (event["error"] !== undefined || stopData === undefined) {
      this._record("on_llm_error", runId, parent, model, "llm", {
        error: event["error"] !== undefined ? errorText(event["error"]) : "None",
      });
      return;
    }

    // Fold tool calls into the response text (same shape as the LangChain callback);
    // tool results are appended later by _onToolEnd.
    const content: Obj[] = stopData["message"]?.["content"] ?? [];
    const text = textOfBlocks(content);
    const parts: string[] = text ? [text] : [];
    const toolUses = content.filter(
      (b) => b && typeof b === "object" && (b["type"] === "toolUseBlock" || "toolUse" in b),
    ).map((b) => (b["toolUse"] ?? b) as Obj);
    for (const tu of toolUses) {
      parts.push(`TOOL CALL: ${tu["name"]}(args=${compactJson(tu["input"])})`);
    }

    const record = this._record("on_llm_end", runId, parent, model, "llm", {
      output: { response: parts.join("\n") },
      token_usage: null,
    });
    for (const tu of toolUses) this._toolLlmEvent.set(tu["toolUseId"], record);
  }

  // ------------------------------------------------------------------
  // Tool events
  // ------------------------------------------------------------------

  _onToolStart(event: Obj): void {
    const toolUse = event["toolUse"] as Obj;
    const agent = event["agent"] as object;
    const runId = randomUUID();
    const parent = this._agentRuns.get(agent) ?? this._rootRunId;
    this._toolRuns.set(toolUse["toolUseId"], runId);
    this._parents.set(runId, parent);

    const toolInput = toolUse["input"];
    const tool = (event["selectedTool"] ?? event["tool"]) as Obj | undefined;
    const description = tool?.["toolSpec"]?.["description"] ?? null;
    this._record("on_tool_start", runId, parent, toolUse["name"], "tool", {
      tool: toolUse["name"],
      tool_description: description,
      input: toolInput !== null && typeof toolInput === "object" && !Array.isArray(toolInput) ? toolInput : { raw: toolInput },
      invoked_by_run_id: parent,
    });
  }

  _onToolEnd(event: Obj): void {
    const toolUse = event["toolUse"] as Obj;
    const toolUseId = toolUse["toolUseId"] as string;
    const runId = this._toolRuns.get(toolUseId);
    if (runId === undefined) return;
    this._toolRuns.delete(toolUseId);
    const parent = this._parents.get(runId) ?? null;
    const name = toolUse["name"] as string;

    if (event["error"] !== undefined) {
      this._record("on_tool_error", runId, parent, name, "tool", { error: errorText(event["error"]) });
      return;
    }

    const output = textOfBlocks(event["result"]?.["content"]);
    this._record("on_tool_end", runId, parent, name, "tool", {
      output,
      is_mcp_tool: isMcpTool(event["selectedTool"] ?? event["tool"]),
    });

    // Attach the tool's output to the LLM event that requested it.
    const llmEvent = this._toolLlmEvent.get(toolUseId);
    this._toolLlmEvent.delete(toolUseId);
    if (llmEvent !== undefined) {
      llmEvent["output"]["response"] += `\nTOOL RESPONSE [${name}]: ${output}`;
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

/**
 * Guard for one Agent called once (no Graph/Swarm).
 *
 * Not public API; use ``getStrandsSingleCallGuard``. The base guard already
 * treats a bare Agent call as a root run, so this only adds the ``single_call``
 * payload flag and keeps the auto-evaluate outcome (the run is over by the time
 * the caller gets control, so it can't be fetched manually).
 */
export class StrandsSingleCallGuardCallback extends StrandsGuardCallback {
  override isSingleCall = true;

  /** Outcome of the auto-triggered evaluation; cleared on every new run. */
  trellarEvaluateResult: AgentLoopResult | null = null;
  trellarEvaluateError: unknown = null;

  override _reset(rootRunId: string | null): void {
    super._reset(rootRunId);
    this.trellarEvaluateResult = null;
    this.trellarEvaluateError = null;
  }

  /** Like the base version, but stores the result/error on the guard. */
  override async _maybeAutoEvaluate(): Promise<void> {
    if (this.observabilityMode === ObservabilityMode.NONE) return;
    if (this.observabilityMode === ObservabilityMode.IF_NOT_EVALUATED && this._evaluated) return;
    try {
      this.trellarEvaluateResult = await evaluateWithGuard(this, { _observabilityCall: true });
      this.trellarEvaluateError = null;
    } catch (error) {
      this.trellarEvaluateError = error;
      logger.warning("Auto-triggered evaluateConfidence() failed", error);
    }
  }
}
