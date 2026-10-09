/**
 * Tests for the Strands guard, driven through real Strands Agents / Graphs with a
 * scripted fake model (no network). Every recorded payload is validated against a
 * mirror of the backend schema.
 */
import {
  Agent,
  TextBlock,
  tool,
  Tool,
  ToolResultBlock,
  type ToolContext,
  type ToolSpec,
  type ToolStreamGenerator,
} from "@strands-agents/sdk";
import { Graph, Node, type NodeResultUpdate } from "@strands-agents/sdk/multiagent";
import { afterEach, describe, expect, it, vi } from "vitest";
import { z } from "zod";

import * as agentLoop from "../src/agentLoop.js";
import { llmHumanInput } from "../src/callbacks/strands/utils/index.js";
import { isGuardActive } from "../src/context.js";
import { evaluateConfidence, ObservabilityMode } from "../src/index.js";
import { getStrandsGuard, getStrandsSingleCallGuard } from "../src/strands.js";
import type { StrandsGuardCallback } from "../src/callbacks/strands/strandsCallback.js";
import { assertValidAgentLoopRequest, mockFetch, useCleanEnv } from "./helpers.js";
import { FakeModel, type Turn } from "./strandsFactories.js";

useCleanEnv();
afterEach(() => vi.restoreAllMocks());

const add = tool({
  name: "add",
  description: "Add two numbers.",
  inputSchema: z.object({ a: z.number(), b: z.number() }),
  callback: ({ a, b }) => a + b,
});
const boom = tool({
  name: "boom",
  description: "Always fails.",
  inputSchema: z.object({}),
  callback: () => {
    throw new Error("tool exploded");
  },
});

const text = (t: string): Turn => ({ text: t });
const toolTurn = (name: string, input: Record<string, unknown>): Turn => ({ toolUse: { name, input } });
const eventsOf = (guard: StrandsGuardCallback, kind: string): any[] => guard.events.filter((e) => e["event"] === kind);

let agentCounter = 0;
function makeAgent(
  guard: StrandsGuardCallback,
  turns: Turn[],
  options: { name?: string; tools?: any[]; systemPrompt?: string } = {},
): Agent {
  const name = options.name ?? "calc";
  agentCounter += 1;
  return new Agent({
    id: `${name}-${agentCounter}`, // Graph node ids come from agent.id and must be unique
    name,
    model: new FakeModel(turns),
    tools: options.tools ?? [],
    plugins: [guard],
    systemPrompt: options.systemPrompt ?? "You are a calculator.",
    printer: false,
  });
}

function assertBackendAccepts(guard: StrandsGuardCallback): void {
  assertValidAgentLoopRequest({
    context: guard.events,
    trace_id: guard.traceId,
    agent_name: guard.agentName,
    observability_call: false,
    single_call: false,
    available_tools: Object.entries(guard.availableTools).map(([hash, tools]) => ({ tools_hash: hash, tools })),
  });
}

/** Tiny non-LLM graph node (like the example's aitl_gate). */
class FunctionNode extends Node {
  readonly type = "functionNode";
  constructor(
    id: string,
    private readonly fn: () => unknown | Promise<unknown>,
  ) {
    super(id, {});
  }
  // eslint-disable-next-line require-yield
  async *handle(): AsyncGenerator<never, NodeResultUpdate, undefined> {
    await this.fn();
    return { content: [new TextBlock("done")] };
  }
}

describe("construction", () => {
  it("requires an agent name", () => {
    expect(() => getStrandsGuard("  ")).toThrow(/agent_name is required/);
    expect(() => getStrandsSingleCallGuard("")).toThrow(/agent_name is required/);
  });
});

describe("single agent", () => {
  it("records the tool-run event sequence with parent links", async () => {
    const guard = getStrandsGuard("t");
    await makeAgent(guard, [toolTurn("add", { a: 1, b: 2 }), text("3")], { tools: [add] }).invoke("what is 1+2?");

    expect(guard.events.map((e) => e["event"])).toEqual([
      "on_chain_start",
      "on_chat_model_start",
      "on_llm_end",
      "on_tool_start",
      "on_tool_end",
      "on_chat_model_start",
      "on_llm_end",
      "on_chain_end",
    ]);
    expect(guard.events.map((e) => e["graph_order"])).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    const root = guard.events[0]!;
    expect(root).toMatchObject({ node_name: "calc", parent_run_id: null, inputs: ["what is 1+2?"] });
    expect(guard.traceId).toBe(root["run_id"]);
    for (const e of guard.events.slice(1, -1)) expect(e["parent_run_id"]).toBe(root["run_id"]);
    assertBackendAccepts(guard);
  });

  it("records the model name and system/human input", async () => {
    const guard = getStrandsGuard("t");
    await makeAgent(guard, [text("hi")], { systemPrompt: "Be brief." }).invoke("hello there");
    const start = eventsOf(guard, "on_chat_model_start")[0];
    expect(start.model).toBe("fake-model");
    expect(start.input).toEqual({ system: "Be brief.", human: "hello there" });
    expect(start.tools_hash).toBeNull();
  });

  it("folds multi-turn history into the human input", async () => {
    const guard = getStrandsGuard("t");
    const agent = makeAgent(guard, [text("Hello! How can I help?"), text("KAN-5 details")], {
      systemPrompt: "Be brief.",
    });
    await agent.invoke("hello");
    await agent.invoke("give me details about KAN-5");

    const starts = eventsOf(guard, "on_chat_model_start");
    expect(starts).toHaveLength(1); // the guard resets per root run
    expect(starts[0].input.human).toBe(
      "Human: hello\nAI LLM: Hello! How can I help?\n\nCurrent message - give me details about KAN-5",
    );
  });

  it("folds tool events and responses into the LLM event", async () => {
    const guard = getStrandsGuard("t");
    await makeAgent(guard, [toolTurn("add", { a: 1, b: 2 }), text("3")], { tools: [add] }).invoke("1+2?");

    const toolStart = eventsOf(guard, "on_tool_start")[0];
    expect(toolStart).toMatchObject({ tool: "add", tool_description: "Add two numbers.", input: { a: 1, b: 2 } });
    expect(toolStart.invoked_by_run_id).toBe(toolStart.parent_run_id);

    const toolEnd = eventsOf(guard, "on_tool_end")[0];
    expect(toolEnd.output).toBe("3");
    expect(toolEnd.is_mcp_tool).toBe(false);
    expect(toolEnd.run_id).toBe(toolStart.run_id);

    const firstLlm = eventsOf(guard, "on_llm_end")[0].output.response;
    expect(firstLlm).toContain('TOOL CALL: add(args={"a": 1, "b": 2})');
    expect(firstLlm).toContain("TOOL RESPONSE [add]: 3");
  });

  it("exposes available tools in the OpenAI shape with a stable hash", async () => {
    const guard = getStrandsGuard("t");
    await makeAgent(guard, [toolTurn("add", { a: 1, b: 2 }), text("3")], { tools: [add] }).invoke("1+2?");

    const starts = eventsOf(guard, "on_chat_model_start");
    expect(new Set(starts.map((s) => s.tools_hash)).size).toBe(1);
    const hash = starts[0].tools_hash;
    expect(Object.keys(guard.availableTools)).toEqual([hash]);
    const schema: any = guard.availableTools[hash]![0];
    expect(schema.type).toBe("function");
    expect(schema.function.name).toBe("add");
    expect(schema.function.description).toBe("Add two numbers.");
    expect(schema.function.parameters.properties).toHaveProperty("a");
  });

  it("a failing tool still produces a valid payload", async () => {
    const guard = getStrandsGuard("t");
    await makeAgent(guard, [toolTurn("boom", {}), text("sorry")], { tools: [boom] }).invoke("go");
    const errors = eventsOf(guard, "on_tool_error");
    expect(errors).toHaveLength(1);
    expect(errors[0].error).toContain("tool exploded");
    assertBackendAccepts(guard);
  });

  it("a model failure records the error and releases the guard", async () => {
    const guard = getStrandsGuard("t");
    const agent = makeAgent(guard, [{ error: "model down" }]);
    await expect(agent.invoke("go")).rejects.toThrow();
    expect(eventsOf(guard, "on_llm_error")[0].error).toContain("model down");
    expect(isGuardActive(guard)).toBe(false);
    assertBackendAccepts(guard);
  });

  it("reuse resets events and the trace id", async () => {
    const guard = getStrandsGuard("t");
    await makeAgent(guard, [text("one")]).invoke("first");
    const firstTrace = guard.traceId;
    await makeAgent(guard, [text("two")]).invoke("second");
    expect(guard.traceId).not.toBe(firstTrace);
    expect(guard.events.every((e) => e["trace_id"] === guard.traceId)).toBe(true);
    expect(eventsOf(guard, "on_chain_start")).toHaveLength(1);
  });

  it("builds the narrative context", async () => {
    const guard = getStrandsGuard("t");
    await makeAgent(guard, [toolTurn("add", { a: 1, b: 2 }), text("3")], { tools: [add] }).invoke("1+2?");
    const context = guard.buildContext();
    expect(context).toContain(`Trace ID: ${guard.traceId}`);
    expect(context).toContain("tool: add");
    expect(context).toContain("human: 1+2?");
  });
});

describe("llmHumanInput", () => {
  const msg = (role: string, ...content: any[]) => ({ role, content });

  it("returns null without messages", () => {
    expect(llmHumanInput([])).toBeNull();
    expect(llmHumanInput(null)).toBeNull();
  });
  it("is plain text for a single turn", () => expect(llmHumanInput([msg("user", { text: "hi" })])).toBe("hi"));
  it("folds the history", () => {
    expect(
      llmHumanInput([
        msg("user", { text: "hello" }),
        msg("assistant", { text: "Hello! How can I help?" }),
        msg("user", { text: "turn 2" }),
        msg("assistant", { text: "reply 2" }),
        msg("user", { text: "turn 3" }),
      ]),
    ).toBe(
      "Human: hello\nAI LLM: Hello! How can I help?\nHuman: turn 2\nAI LLM: reply 2\n\nCurrent message - turn 3",
    );
  });
  it("skips tool-use and tool-result messages", () => {
    expect(
      llmHumanInput([
        msg("user", { text: "turn 1" }),
        msg("assistant", { toolUse: { toolUseId: "1", name: "add", input: {} } }),
        msg("user", { toolResult: { toolUseId: "1", content: [{ text: "3" }] } }),
        msg("assistant", { text: "reply 1" }),
        msg("user", { text: "turn 2" }),
      ]),
    ).toBe("Human: turn 1\nAI LLM: reply 1\n\nCurrent message - turn 2");
  });
  it("ignores messages after the last user turn", () => {
    expect(
      llmHumanInput([
        msg("user", { text: "turn 1" }),
        msg("assistant", { toolUse: { toolUseId: "1", name: "add", input: {} } }),
        msg("user", { toolResult: { toolUseId: "1", content: [{ text: "3" }] } }),
      ]),
    ).toBe("turn 1");
  });
});

describe("MCP detection", () => {
  it("flags tools that come from an MCP client", async () => {
    // Same class name / shape as the SDK's (non-exported) McpTool.
    class McpTool extends Tool {
      readonly name = "create_ticket";
      readonly description = "Create a ticket";
      readonly toolSpec: ToolSpec = {
        name: "create_ticket",
        description: "Create a ticket",
        inputSchema: { type: "object", properties: { symbols: { type: "string" } } },
      };
      // eslint-disable-next-line require-yield
      async *stream(ctx: ToolContext): ToolStreamGenerator {
        return new ToolResultBlock({
          toolUseId: ctx.toolUse.toolUseId,
          status: "success",
          content: [new TextBlock("T-123")],
        });
      }
    }
    const guard = getStrandsGuard("t");
    await makeAgent(guard, [toolTurn("create_ticket", { symbols: "GM" }), text("ok")], {
      tools: [new McpTool(), add],
    }).invoke("buy");
    const end = eventsOf(guard, "on_tool_end")[0];
    expect(end.is_mcp_tool).toBe(true);
    expect(end.output).toBe("T-123");
    assertBackendAccepts(guard);
  });
});

function buildGraph(guard: StrandsGuardCallback, gate: () => unknown | Promise<unknown>): Graph {
  const a1 = makeAgent(guard, [toolTurn("add", { a: 1, b: 2 }), text("3")], { name: "a1", tools: [add] });
  const a2 = makeAgent(guard, [text("report")], { name: "a2" });
  return new Graph({
    nodes: [a1, new FunctionNode("gate", gate), a2],
    edges: [
      [a1.id, "gate"],
      ["gate", a2.id],
    ],
    plugins: [guard],
  });
}

describe("graph", () => {
  it("nests graph -> node -> agent -> llm/tool", async () => {
    const guard = getStrandsGuard("t");
    await buildGraph(guard, () => {}).invoke("go");

    const chains = eventsOf(guard, "on_chain_start");
    const byRun = new Map(chains.map((e) => [e.run_id, e]));
    const root = guard.events[0]!;
    expect(root["parent_run_id"]).toBeNull();
    expect(root["node_type"]).toBe("chain");

    const names = chains.map((e) => e.node_name);
    expect(names.filter((n) => n === "a1")).toHaveLength(1);
    expect(names.filter((n) => n.startsWith("a1-"))).toHaveLength(1); // graph node chain (agent id)
    expect(names).toContain("gate");

    const llm = eventsOf(guard, "on_chat_model_start")[0];
    expect(byRun.get(llm.parent_run_id)!.node_name).toBe("a1");
    const toolStart = eventsOf(guard, "on_tool_start")[0];
    expect(byRun.get(toolStart.parent_run_id)!.node_name).toBe("a1");

    const agentChain = byRun.get(llm.parent_run_id)!;
    const nodeChain = byRun.get(agentChain.parent_run_id)!;
    expect(nodeChain.parent_run_id).toBe(root["run_id"]);
    assertBackendAccepts(guard);
  });

  it("has one trace and the root end is last", async () => {
    const guard = getStrandsGuard("t");
    await buildGraph(guard, () => {}).invoke("go");
    expect(new Set(guard.events.map((e) => e["trace_id"]))).toEqual(new Set([guard.traceId]));
    expect(guard.events.at(-1)).toMatchObject({ event: "on_chain_end", run_id: guard.traceId });
  });

  it("evaluateConfidence() works from inside a node and sends a backend-valid payload", async () => {
    const calls = mockFetch();
    const guard = getStrandsGuard("net-name");
    const results: any[] = [];
    await buildGraph(guard, async () => {
      results.push(await evaluateConfidence());
    }).invoke("go");

    expect(results[0].score).toBe(8);
    expect(calls).toHaveLength(1);
    const body = calls[0]!.body;
    expect(body).toMatchObject({ agent_name: "net-name", trace_id: guard.traceId, observability_call: false });
    assertValidAgentLoopRequest(body);
    expect(isGuardActive(guard)).toBe(false); // released after the run
  });

  it("nests an agent-as-tool call under the outer tool call", async () => {
    const guard = getStrandsGuard("t");
    const inner = makeAgent(guard, [text("inner answer")], { name: "inner" });
    const askInner = tool({
      name: "ask_inner",
      description: "Ask the inner agent.",
      inputSchema: z.object({ question: z.string() }),
      callback: async ({ question }) => String((await inner.invoke(question)).toString()),
    });
    const outer = makeAgent(guard, [toolTurn("ask_inner", { question: "q" }), text("done")], {
      name: "outer",
      tools: [askInner],
    });
    await outer.invoke("go");

    const toolStart = eventsOf(guard, "on_tool_start")[0];
    const innerChain = eventsOf(guard, "on_chain_start").find((e) => e.node_name === "inner");
    expect(innerChain.parent_run_id).toBe(toolStart.run_id);
    const roots = eventsOf(guard, "on_chain_start").filter((e) => e.parent_run_id === null);
    expect(roots.map((r) => r.node_name)).toEqual(["outer"]);
    assertBackendAccepts(guard);
  });
});

describe("conditional routing", () => {
  const routeJira = tool({ name: "route_to_jira_agent", description: "r", inputSchema: z.object({}), callback: () => "route:jira" });
  const routeEmail = tool({ name: "route_to_send_email_agent", description: "r", inputSchema: z.object({}), callback: () => "route:send_email" });
  const flag = tool({ name: "flag_email_report_request", description: "f", inputSchema: z.object({}), callback: () => "email_report_requested" });

  /** Names of the tools an agent (by name) called in this run, read back from the guard's own events. */
  function calledTools(guard: StrandsGuardCallback, agentName: string): string[] {
    const chainName = (runId: string): string | undefined =>
      (guard.events as any[]).find((e) => e.run_id === runId && e.event === "on_chain_start")?.node_name;
    return (guard.events as any[])
      .filter((e) => e.event === "on_tool_start" && chainName(e.parent_run_id) === agentName)
      .map((e) => e.tool);
  }

  // Strands Graph uses AND semantics for incoming edges, so each path gets its own gate.
  function routingGraph(guard: StrandsGuardCallback, routeTool: string, jiraFlagsEmail: boolean): Graph {
    const router = makeAgent(guard, [toolTurn(routeTool, {}), text("routed")], {
      name: "router_agent",
      tools: [routeJira, routeEmail],
    });
    const jira = makeAgent(
      guard,
      jiraFlagsEmail ? [toolTurn("flag_email_report_request", {}), text("preparing")] : [text("3 open issues")],
      { name: "jira_react_agent", tools: [flag] },
    );
    const reporterDirect = makeAgent(guard, [text("Dear user, ...")], { name: "reporter_direct" });
    const reporterAfterJira = makeAgent(guard, [text("Dear user, ...")], { name: "reporter_after_jira" });
    const emailOnly = () => calledTools(guard, "router_agent").includes("route_to_send_email_agent");

    return new Graph({
      nodes: [
        router,
        jira,
        new FunctionNode("gate_direct", () => evaluateConfidence()),
        new FunctionNode("gate_after_jira", () => evaluateConfidence()),
        reporterDirect,
        reporterAfterJira,
      ],
      edges: [
        { source: router.id, target: jira.id, handler: () => !emailOnly() },
        { source: router.id, target: "gate_direct", handler: () => emailOnly() },
        { source: "gate_direct", target: reporterDirect.id },
        {
          source: jira.id,
          target: "gate_after_jira",
          handler: () => calledTools(guard, "jira_react_agent").includes("flag_email_report_request"),
        },
        { source: "gate_after_jira", target: reporterAfterJira.id },
      ],
      plugins: [guard],
    });
  }
  const chainNames = (guard: StrandsGuardCallback) => new Set(eventsOf(guard, "on_chain_start").map((e) => e.node_name));

  it("email only skips jira", async () => {
    const calls = mockFetch();
    const guard = getStrandsGuard("t");
    await routingGraph(guard, "route_to_send_email_agent", false).invoke("email me");
    const names = chainNames(guard);
    expect(names.has("gate_direct")).toBe(true);
    expect(names.has("reporter_direct")).toBe(true);
    expect(names.has("jira_react_agent")).toBe(false);
    expect(calls).toHaveLength(1); // the gate evaluated once
    assertBackendAccepts(guard);
  });

  it("jira without an email request skips gate and reporter", async () => {
    const calls = mockFetch();
    const guard = getStrandsGuard("t");
    await routingGraph(guard, "route_to_jira_agent", false).invoke("how many issues?");
    const names = chainNames(guard);
    expect(names.has("jira_react_agent")).toBe(true);
    expect(names.has("gate_after_jira")).toBe(false);
    expect(names.has("reporter_after_jira")).toBe(false);
    expect(calls).toHaveLength(0);
    assertBackendAccepts(guard);
  });

  it("jira with the email flag goes through the gate", async () => {
    const calls = mockFetch();
    const guard = getStrandsGuard("t");
    await routingGraph(guard, "route_to_jira_agent", true).invoke("issues, then email me");
    const order = eventsOf(guard, "on_chain_start").map((e) => e.node_name);
    expect(order.indexOf("jira_react_agent")).toBeGreaterThan(-1);
    expect(order.indexOf("jira_react_agent")).toBeLessThan(order.indexOf("gate_after_jira"));
    expect(order.indexOf("gate_after_jira")).toBeLessThan(order.indexOf("reporter_after_jira"));
    expect(calls).toHaveLength(1);
    assertBackendAccepts(guard);
  });
});

describe("observability mode", () => {
  it("NONE never auto-evaluates", async () => {
    const calls = mockFetch();
    await makeAgent(getStrandsGuard("t", ObservabilityMode.NONE), [text("hi")]).invoke("go");
    expect(calls).toHaveLength(0);
  });

  it("ALWAYS evaluates at root end with the final chain end recorded", async () => {
    const calls = mockFetch();
    await buildGraph(getStrandsGuard("t", ObservabilityMode.ALWAYS), () => {}).invoke("go");
    expect(calls).toHaveLength(1);
    const body = calls[0]!.body;
    expect(body.observability_call).toBe(true);
    expect(body.context.at(-1).event).toBe("on_chain_end");
    assertValidAgentLoopRequest(body);
  });

  it("IF_NOT_EVALUATED skips after a manual call", async () => {
    const calls = mockFetch();
    await buildGraph(getStrandsGuard("t", ObservabilityMode.IF_NOT_EVALUATED), () => evaluateConfidence()).invoke("go");
    expect(calls).toHaveLength(1);
    expect(calls[0]!.body.observability_call).toBe(false);
  });

  it("IF_NOT_EVALUATED runs when there was no manual call", async () => {
    const calls = mockFetch();
    await buildGraph(getStrandsGuard("t", ObservabilityMode.IF_NOT_EVALUATED), () => {}).invoke("go");
    expect(calls).toHaveLength(1);
    expect(calls[0]!.body.observability_call).toBe(true);
  });

  it("auto-evaluate failure is swallowed", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("backend down")));
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const result = await makeAgent(getStrandsGuard("t", ObservabilityMode.ALWAYS), [text("hi")]).invoke("go");
    expect(String(result)).toContain("hi");
  });
});

describe("single call", () => {
  it("flags the payload and satisfies the backend schema", async () => {
    const calls = mockFetch();
    await makeAgent(getStrandsSingleCallGuard("t", ObservabilityMode.ALWAYS), [text("hi")]).invoke("go");
    expect(calls[0]!.body.single_call).toBe(true);
    assertValidAgentLoopRequest(calls[0]!.body);
  });

  it("a regular guard is not single_call", async () => {
    const calls = mockFetch();
    await makeAgent(getStrandsGuard("t", ObservabilityMode.ALWAYS), [text("hi")]).invoke("go");
    expect(calls[0]!.body.single_call).toBe(false);
  });

  it("stores the result on the guard and releases it", async () => {
    mockFetch();
    const guard = getStrandsSingleCallGuard("t", ObservabilityMode.ALWAYS);
    await makeAgent(guard, [text("hi")]).invoke("go");
    expect(guard.trellarEvaluateResult?.score).toBe(8);
    expect(guard.trellarEvaluateError).toBeNull();
    expect(isGuardActive(guard)).toBe(false);
  });

  it("stores a backend error instead of raising", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("backend down")));
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const guard = getStrandsSingleCallGuard("t", ObservabilityMode.ALWAYS);
    await makeAgent(guard, [text("hi")]).invoke("go");
    expect(guard.trellarEvaluateResult).toBeNull();
    expect(guard.trellarEvaluateError).toBeInstanceOf(Error);
  });

  it("a second call resets the previous result", async () => {
    mockFetch();
    const guard = getStrandsSingleCallGuard("t", ObservabilityMode.ALWAYS);
    await makeAgent(guard, [text("hi")]).invoke("go");
    const first = guard.trellarEvaluateResult;
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("down")));
    vi.spyOn(console, "warn").mockImplementation(() => {});
    await makeAgent(guard, [text("hi")]).invoke("go again");
    expect(first).not.toBeNull();
    expect(guard.trellarEvaluateResult).toBeNull();
  });

  it("NONE mode does not evaluate", async () => {
    const calls = mockFetch();
    const guard = getStrandsSingleCallGuard("t");
    await makeAgent(guard, [text("hi")]).invoke("go");
    expect(calls).toHaveLength(0);
    expect(guard.trellarEvaluateResult).toBeNull();
  });
});

describe("concurrent runs are isolated", () => {
  it("each parallel graph evaluates its own guard", async () => {
    const calls = mockFetch();
    const names = ["run-a", "run-b", "run-c"];
    const guards = names.map((n) => getStrandsGuard(n));
    await Promise.all(
      guards.map((guard, i) =>
        buildGraph(guard, async () => {
          await new Promise((resolve) => setTimeout(resolve, 10 * (3 - i)));
          await evaluateConfidence();
        }).invoke(`task ${names[i]}`),
      ),
    );

    expect(calls).toHaveLength(3);
    for (const call of calls) {
      const guard = guards.find((g) => g.agentName === call.body.agent_name)!;
      expect(call.body.trace_id).toBe(guard.traceId);
      expect(call.body.context.every((e: any) => e.trace_id === guard.traceId)).toBe(true);
      expect(call.body.context[0].inputs).toEqual([`task ${call.body.agent_name}`]);
    }
    expect(new Set(calls.map((c) => c.body.agent_name)).size).toBe(3);
  });
});

void agentLoop;
