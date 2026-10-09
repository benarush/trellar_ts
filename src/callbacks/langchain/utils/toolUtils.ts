import { toJsonable } from "../../common.js";

type Obj = Record<string, unknown>;

function isObj(value: unknown): value is Obj {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function toFunctionTool(name: unknown, description: unknown, parameters: unknown): Obj | undefined {
  if (typeof name !== "string" || name === "") return undefined;
  return {
    type: "function",
    function: {
      name,
      description: typeof description === "string" ? description : "",
      parameters: isObj(parameters) ? parameters : {},
    },
  };
}

/**
 * Coerce the tool list a chat model was bound with into the OpenAI function
 * shape the backend expects (``{type: "function", function: {name, description,
 * parameters}}``). LangChain JS providers hand over several shapes (OpenAI
 * tools, Google ``functionDeclarations``, bare ``{name, description, schema}``
 * objects); anything unrecognised is kept as-is.
 */
export function normalizeTools(rawTools: unknown): unknown[] {
  const tools = toJsonable(rawTools);
  if (!Array.isArray(tools)) return [];

  const out: unknown[] = [];
  for (const tool of tools) {
    if (!isObj(tool)) continue;

    const fn = tool["function"];
    if (tool["type"] === "function" && isObj(fn)) {
      out.push({
        type: "function",
        function: {
          name: fn["name"],
          description: typeof fn["description"] === "string" ? fn["description"] : "",
          parameters: isObj(fn["parameters"]) ? fn["parameters"] : {},
        },
      });
      continue;
    }

    const declarations = tool["functionDeclarations"] ?? tool["function_declarations"];
    if (Array.isArray(declarations)) {
      for (const decl of declarations) {
        if (!isObj(decl)) continue;
        const converted = toFunctionTool(
          decl["name"],
          decl["description"],
          decl["parameters"] ?? decl["parametersJsonSchema"],
        );
        if (converted) out.push(converted);
      }
      continue;
    }

    const converted = toFunctionTool(
      tool["name"],
      tool["description"],
      tool["parameters"] ?? tool["input_schema"] ?? tool["inputSchema"] ?? tool["schema"],
    );
    out.push(converted ?? tool);
  }
  return out;
}

/** Look up a tool's description by name in the toolsets bound during this run. */
export function findToolDescription(
  availableTools: Record<string, unknown[]>,
  toolName: string | null | undefined,
): string | null {
  if (!toolName) return null;
  for (const tools of Object.values(availableTools)) {
    for (const tool of tools) {
      if (!isObj(tool) || !isObj(tool["function"])) continue;
      if (tool["function"]["name"] === toolName) {
        const description = tool["function"]["description"];
        return typeof description === "string" && description !== "" ? description : null;
      }
    }
  }
  return null;
}
