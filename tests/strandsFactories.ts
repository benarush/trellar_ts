/** Fake Strands model + helpers (counterpart of tests/strands_factories.py). */
import { Model, type Message, type StreamOptions } from "@strands-agents/sdk";
import type { ModelStreamEvent } from "@strands-agents/sdk";

export type Turn = { text: string } | { error: string } | { toolUse: { name: string; input: Record<string, unknown>; id?: string } };

/** Scripted model: plays one ``Turn`` per call (the last turn repeats). */
export class FakeModel extends Model {
  private calls = 0;
  readonly seenSystemPrompts: unknown[] = [];

  constructor(
    private readonly turns: Turn[],
    private readonly fakeModelId = "fake-model",
  ) {
    super();
  }

  updateConfig(): void {}

  getConfig(): { modelId: string } {
    return { modelId: this.fakeModelId };
  }

  async *stream(_messages: Message[], options?: StreamOptions): AsyncIterable<ModelStreamEvent> {
    this.seenSystemPrompts.push(options?.systemPrompt);
    const turn = this.turns[Math.min(this.calls, this.turns.length - 1)]!;
    this.calls += 1;
    if ("error" in turn) throw new Error(turn.error);
    yield { type: "modelMessageStartEvent", role: "assistant" };
    if ("text" in turn) {
      yield { type: "modelContentBlockStartEvent" };
      yield { type: "modelContentBlockDeltaEvent", delta: { type: "textDelta", text: turn.text } };
      yield { type: "modelContentBlockStopEvent" };
      yield { type: "modelMessageStopEvent", stopReason: "endTurn" };
    } else {
      const id = turn.toolUse.id ?? `tool-${this.calls}`;
      yield {
        type: "modelContentBlockStartEvent",
        start: { type: "toolUseStart", name: turn.toolUse.name, toolUseId: id },
      };
      yield {
        type: "modelContentBlockDeltaEvent",
        delta: { type: "toolUseInputDelta", input: JSON.stringify(turn.toolUse.input) },
      };
      yield { type: "modelContentBlockStopEvent" };
      yield { type: "modelMessageStopEvent", stopReason: "toolUse" };
    }
  }
}
