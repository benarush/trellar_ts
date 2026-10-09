import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { AIMessage, type BaseMessage } from "@langchain/core/messages";
import type { ChatResult } from "@langchain/core/outputs";
import { convertToOpenAITool } from "@langchain/core/utils/function_calling";

export interface ScriptedTurn {
  content?: string;
  toolCalls?: Array<{ name: string; args: Record<string, unknown>; id: string }>;
  /** Delay before answering, to force interleaving of concurrent runs. */
  delayMs?: number;
}

/** Chat model that plays scripted turns and supports ``bindTools`` (exposed via invocation params). */
export class ScriptedChatModel extends BaseChatModel {
  /** Shared with bound clones so the script advances across ``bindTools()`` copies. */
  private counter: { calls: number } = { calls: 0 };
  boundTools: unknown[] = [];

  constructor(
    private readonly turns: ScriptedTurn[],
    private readonly modelName = "scripted-model",
  ) {
    super({});
  }

  _llmType(): string {
    return "scripted";
  }

  override invocationParams(): Record<string, unknown> {
    return { model: this.modelName, tools: this.boundTools.length > 0 ? this.boundTools : undefined };
  }

  bindTools(tools: any[]): ScriptedChatModel {
    const clone = new ScriptedChatModel(this.turns, this.modelName);
    clone.counter = this.counter;
    clone.boundTools = tools.map((t) => convertToOpenAITool(t));
    return clone;
  }

  async _generate(_messages: BaseMessage[]): Promise<ChatResult> {
    const turn = this.turns[Math.min(this.counter.calls, this.turns.length - 1)]!;
    this.counter.calls += 1;
    if (turn.delayMs) await new Promise((resolve) => setTimeout(resolve, turn.delayMs));
    const message = new AIMessage({
      content: turn.content ?? "",
      tool_calls: turn.toolCalls ?? [],
    });
    return { generations: [{ message, text: turn.content ?? "" }] };
  }
}
