import { toJsonable } from "../../common.js";
import { extractReasoning } from "../../utils/index.js";

type Obj = Record<string, any>;

/** Real model id (e.g. 'gemini-2.5-flash-lite'); class name as fallback. */
export function modelName(agent: Obj): string {
  let name: unknown;
  try {
    const config = agent["model"]?.getConfig?.() ?? {};
    name = config["modelId"] ?? config["model_id"] ?? config["model"] ?? agent["model"]?.modelId;
  } catch {
    name = undefined;
  }
  return String(name || agent["model"]?.constructor?.name || "unknown");
}

/**
 * The agent model's reasoning/thinking settings as configured, or ``null``.
 *
 * Reads ``agent.model.getConfig()`` (reasoning options sit in its ``params`` /
 * ``additionalRequestFields``); the provider is inferred from the model class name
 * and the model id. Never throws; see ``extractReasoning``.
 */
export function thinkingConfig(agent: Obj): Obj | null {
  try {
    const model = agent["model"];
    const config = model?.getConfig?.() ?? {};
    return extractReasoning([config], model?.constructor?.name, modelName(agent));
  } catch {
    return null;
  }
}

/** Agent's tools in the OpenAI function shape the backend expects. */
export function openaiTools(agent: Obj): unknown[] {
  const tools: unknown[] = [];
  let registered: Obj[] = [];
  try {
    registered = agent["toolRegistry"]?.list?.() ?? [];
  } catch {
    registered = [];
  }
  for (const tool of registered) {
    const spec: Obj | undefined = tool?.["toolSpec"];
    if (!spec) continue;
    tools.push({
      type: "function",
      function: {
        name: spec["name"],
        description: spec["description"] ?? "",
        parameters: spec["inputSchema"] ?? {},
      },
    });
  }
  return toJsonable(tools) as unknown[];
}

/** System prompt as plain text (string, or the text blocks of a block list). */
export function systemPromptText(agent: Obj): string | null {
  const prompt = agent["systemPrompt"];
  if (prompt === undefined || prompt === null) return null;
  if (typeof prompt === "string") return prompt;
  if (Array.isArray(prompt)) {
    return prompt
      .filter((b: Obj) => b && typeof b["text"] === "string")
      .map((b: Obj) => b["text"])
      .join("\n");
  }
  return String(prompt);
}

/** Whether a Strands tool is an MCP tool (``McpTool``, built by ``McpClient.listTools()``). */
export function isMcpTool(tool: unknown): boolean {
  if (tool === null || typeof tool !== "object") return false;
  let proto: object | null = Object.getPrototypeOf(tool);
  while (proto) {
    if (proto.constructor?.name === "McpTool") return true;
    proto = Object.getPrototypeOf(proto);
  }
  return false;
}
