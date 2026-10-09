/** End-to-end LangGraph runs with scripted models and a mocked backend. */
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { DynamicStructuredTool } from "@langchain/core/tools";
import { Annotation, END, MessagesAnnotation, START, StateGraph } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { describe, expect, it } from "vitest";
import { z } from "zod";

import { evaluateConfidence, ObservabilityMode } from "../src/index.js";
import { getAgentGuard } from "../src/langchain.js";
import { isGuardActive } from "../src/context.js";
import { ScriptedChatModel } from "./fakeChatModel.js";
import { assertValidAgentLoopRequest, mockFetch, useCleanEnv } from "./helpers.js";

useCleanEnv();

const buyTool = new DynamicStructuredTool({
  name: "buy_stocks",
  description: "Execute buy orders",
  schema: z.object({ symbols: z.string() }),
  func: async ({ symbols }) => `purchased ${symbols}`,
});

const mcpLikeTool = new DynamicStructuredTool({
  name: "create_purchase_ticket",
  description: "Create a ticket",
  schema: z.object({ symbols: z.string() }),
  responseFormat: "content_and_artifact",
  // @langchain/mcp-adapters always sets this metadata key.
  metadata: { annotations: undefined },
  func: async () => ["ticket-1", []],
});

function buildGraph(options: { delayMs?: number; onGate?: () => Promise<unknown> } = {}) {
  const model = new ScriptedChatModel([
    {
      toolCalls: [
        { name: "buy_stocks", args: { symbols: "TSLA,F" }, id: "call-1" },
        { name: "create_purchase_ticket", args: { symbols: "TSLA,F" }, id: "call-2" },
      ],
      delayMs: options.delayMs,
    },
    { content: "all done", delayMs: options.delayMs },
  ]).bindTools([buyTool, mcpLikeTool]);

  const toolNode = new ToolNode([buyTool, mcpLikeTool]);
  const Gate = Annotation.Root({ ...MessagesAnnotation.spec, score: Annotation<number | null>() });

  return new StateGraph(Gate)
    .addNode("agent", async (state, config) => ({
      messages: [await model.invoke([new SystemMessage("You buy stocks"), ...state.messages], config)],
    }))
    .addNode("tools", toolNode)
    .addNode("gate", async () => {
      const result = await (options.onGate ?? (() => evaluateConfidence({ apiKey: "k" })))();
      return { score: (result as { score: number }).score };
    })
    .addEdge(START, "agent")
    .addEdge("agent", "tools")
    .addEdge("tools", "gate")
    .addEdge("gate", END)
    .compile();
}

describe("LangGraph end to end", () => {
  it("captures the whole network and sends a backend-valid payload from a gate node", async () => {
    const calls = mockFetch();
    const guard = getAgentGuard("e2e-agent");
    const result = await buildGraph().invoke({ messages: [new HumanMessage("buy cars")] }, { callbacks: [guard] });

    expect(result.score).toBe(8);
    expect(calls).toHaveLength(1);
    const body = calls[0]!.body;
    expect(body.agent_name).toBe("e2e-agent");
    expect(body.observability_call).toBe(false);
    expect(body.single_call).toBe(false);
    assertValidAgentLoopRequest(body);

    // Only the events up to the gate are in the payload (the run is still in progress).
    const events: any[] = body.context;
    expect(events[0]).toMatchObject({ event: "on_chain_start", parent_run_id: null, node_name: "LangGraph" });
    expect(events.every((e) => e.trace_id === body.trace_id)).toBe(true);
    expect(events.map((e) => e.graph_order)).toEqual(events.map((_, i) => i + 1));

    const chat = events.find((e) => e.event === "on_chat_model_start");
    expect(chat).toMatchObject({ model: "scripted-model", input: { system: "You buy stocks", human: "buy cars" } });
    expect(chat.tools_hash).toMatch(/^[0-9a-f]{16}$/);
    expect(body.available_tools).toHaveLength(1);
    expect(body.available_tools[0].tools_hash).toBe(chat.tools_hash);
    expect(body.available_tools[0].tools.map((t: any) => t.function.name)).toEqual([
      "buy_stocks",
      "create_purchase_ticket",
    ]);

    const toolStarts = events.filter((e) => e.event === "on_tool_start");
    expect(toolStarts.map((e) => [e.tool, e.tool_description])).toEqual([
      ["buy_stocks", "Execute buy orders"],
      ["create_purchase_ticket", "Create a ticket"],
    ]);
    const toolEnds = events.filter((e) => e.event === "on_tool_end");
    expect(Object.fromEntries(toolEnds.map((e) => [e.node_name, e.is_mcp_tool]))).toEqual({
      buy_stocks: false,
      create_purchase_ticket: true,
    });

    // The LLM event got both tool responses attached retroactively.
    const llmEnd = events.find((e) => e.event === "on_llm_end");
    expect(llmEnd.output.response).toContain('TOOL CALL: buy_stocks(args={"symbols": "TSLA,F"})');
    expect(llmEnd.output.response).toContain("TOOL RESPONSE [buy_stocks]: TOOL MESSAGE: purchased TSLA,F");
    expect(llmEnd.output.response).toContain("TOOL RESPONSE [create_purchase_ticket]: TOOL MESSAGE: ticket-1");

    expect(events.some((e) => e.event === "on_chain_start" && e.node_name === "gate")).toBe(true);
    expect(events.some((e) => e.langgraph_step === 1)).toBe(true);
  });

  it("releases the guard after invoke() so a late evaluateConfidence throws", async () => {
    mockFetch();
    const guard = getAgentGuard("late");
    await buildGraph().invoke({ messages: [new HumanMessage("x")] }, { callbacks: [guard] });
    expect(isGuardActive(guard)).toBe(false);
    await expect(evaluateConfidence({ apiKey: "k" })).rejects.toThrow(/No active callback handler/);
  });

  it("ALWAYS mode auto-evaluates at the end with the full run and an observability flag", async () => {
    const calls = mockFetch();
    const guard = getAgentGuard("auto", ObservabilityMode.ALWAYS);
    await buildGraph().invoke({ messages: [new HumanMessage("x")] }, { callbacks: [guard] });
    // gate (manual) + automatic end-of-run call
    expect(calls.map((c) => c.body.observability_call)).toEqual([false, true]);
    const autoBody = calls[1]!.body;
    expect(autoBody.context.at(-1)).toMatchObject({ event: "on_chain_end", parent_run_id: null });
    assertValidAgentLoopRequest(autoBody);
  });

  it("IF_NOT_EVALUATED skips the automatic call after a successful manual one", async () => {
    const calls = mockFetch();
    const guard = getAgentGuard("auto2", ObservabilityMode.IF_NOT_EVALUATED);
    await buildGraph().invoke({ messages: [new HumanMessage("x")] }, { callbacks: [guard] });
    expect(calls).toHaveLength(1);
    expect(calls[0]!.body.observability_call).toBe(false);
  });

  it("auto-evaluate failures never break invoke()", async () => {
    mockFetch({ detail: "down" }, 500, "Server Error");
    const guard = getAgentGuard("auto3", ObservabilityMode.ALWAYS);
    const graph = buildGraph({ onGate: async () => ({ score: 1 }) });
    const result = await graph.invoke({ messages: [new HumanMessage("x")] }, { callbacks: [guard] });
    expect(result.score).toBe(1);
  });
});

describe("registerTools", () => {
  it("reports descriptions of directly-invoked tools (JS does not serialize them)", async () => {
    mockFetch();
    const guard = getAgentGuard("direct").registerTools([buyTool]);
    const graph = new StateGraph(MessagesAnnotation)
      .addNode("buy", async (_state, config) => {
        await buyTool.invoke({ symbols: "F" }, config);
        return {};
      })
      .addEdge(START, "buy")
      .addEdge("buy", END)
      .compile();
    await graph.invoke({ messages: [] }, { callbacks: [guard] });
    const start = (guard.events as any[]).find((e) => e.event === "on_tool_start");
    expect(start).toMatchObject({ tool: "buy_stocks", tool_description: "Execute buy orders" });
  });

  it("without registration a directly-invoked tool has no description", async () => {
    mockFetch();
    const guard = getAgentGuard("direct2");
    await buyTool.invoke({ symbols: "F" }, { callbacks: [guard] });
    const start = (guard.events as any[]).find((e) => e.event === "on_tool_start");
    expect(start.tool_description).toBeNull();
  });
});

describe("concurrent runs are isolated (Python ContextVar equivalent)", () => {
  it("each parallel invoke() evaluates its own guard", async () => {
    const calls = mockFetch();
    const names = ["run-a", "run-b", "run-c"];
    const guards = names.map((n) => getAgentGuard(n));
    await Promise.all(
      guards.map((guard, i) =>
        buildGraph({ delayMs: 15 * (3 - i) }).invoke(
          { messages: [new HumanMessage(`task ${names[i]}`)] },
          { callbacks: [guard] },
        ),
      ),
    );

    expect(calls).toHaveLength(3);
    for (const call of calls) {
      const body = call.body;
      const guard = guards.find((g) => g.agentName === body.agent_name)!;
      expect(body.trace_id).toBe(guard.traceId);
      // No cross-talk: every event belongs to this run and mentions only this run's task.
      expect(body.context.every((e: any) => e.trace_id === body.trace_id)).toBe(true);
      const human = body.context.find((e: any) => e.event === "on_chat_model_start").input.human;
      expect(human).toBe(`task ${body.agent_name}`);
    }
    expect(new Set(calls.map((c) => c.body.agent_name)).size).toBe(3);
    expect(guards.every((g) => !isGuardActive(g))).toBe(true);
  });
});
