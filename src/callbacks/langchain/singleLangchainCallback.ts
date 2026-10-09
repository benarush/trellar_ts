import { type AgentLoopResult, evaluateWithTrellarAgent, ObservabilityMode } from "../../agentLoop.js";
import { releaseTrellarAgent } from "../../context.js";
import { logger } from "../../logger.js";
import { LangchainAgentCallback } from "./langchainCallback.js";

type Obj = Record<string, any>;

/**
 * Callback for a single bare LLM call (no LangGraph/chain wrapper).
 *
 * This class is not part of the public API. Use ``trellarLangchainSingleCall`` to
 * obtain an instance.
 *
 * ``LangchainAgentCallback`` only sets ``traceId`` / activates the Trellar agent inside
 * ``handleChainStart``, and only auto-triggers ``evaluateConfidence()`` inside
 * ``handleChainEnd`` -- both scoped to ``parentRunId === undefined``. A bare
 * ``llm.invoke(...)`` never fires either of those events (no chain involved),
 * so this subclass does the equivalent work on the LLM run boundary instead:
 * ``handleLLMStart`` / ``handleChatModelStart`` (root-run reset + registration)
 * and ``handleLLMEnd`` (auto-evaluate trigger).
 */
export class LangchainSingleCallCallback extends LangchainAgentCallback {
  override name = "trellar_langchain_single_call";
  override isSingleCall = true;

  /**
   * Populated by ``_autoEvaluate()`` when ``observabilityMode`` triggers an
   * automatic evaluateConfidence() call. ``null`` until then, or if the caller
   * only ever evaluates manually (``observabilityMode: NONE``).
   */
  trellarEvaluateResult: AgentLoopResult | null = null;
  trellarEvaluateError: unknown = null;

  constructor(options: { agentName: string; observabilityMode?: ObservabilityMode }) {
    super(options);
  }

  override _resetForNewRun(runId: string): void {
    super._resetForNewRun(runId);
    this.trellarEvaluateResult = null;
    this.trellarEvaluateError = null;
  }

  override handleLLMStart(
    llm: Obj,
    prompts: string[],
    runId: string,
    parentRunId?: string,
    extraParams?: Obj,
  ): void {
    if (parentRunId === undefined || parentRunId === null) this._resetForNewRun(runId);
    super.handleLLMStart(llm, prompts, runId, parentRunId, extraParams);
  }

  override handleChatModelStart(
    llm: Obj,
    messages: unknown[][],
    runId: string,
    parentRunId?: string,
    extraParams?: Obj,
  ): void {
    if (parentRunId === undefined || parentRunId === null) this._resetForNewRun(runId);
    super.handleChatModelStart(llm, messages, runId, parentRunId, extraParams);
  }

  override async handleLLMEnd(response: Obj, runId: string, parentRunId?: string): Promise<void> {
    super.handleLLMEnd(response, runId, parentRunId);
    if (parentRunId === undefined || parentRunId === null) {
      await this._autoEvaluate();
      // Release the slot so the next top-level call starts clean.
      releaseTrellarAgent(this);
    }
  }

  override handleLLMError(error: unknown, runId: string, parentRunId?: string): void {
    super.handleLLMError(error, runId, parentRunId);
    if (parentRunId === undefined || parentRunId === null) {
      // Root call failing -- handleLLMEnd will never fire for this runId
      // (they are mutually exclusive), so release the slot here too.
      releaseTrellarAgent(this);
    }
  }

  /**
   * Local equivalent of ``_maybeAutoEvaluate``; stores the outcome on this
   * instance instead of discarding it.
   *
   * Errors are caught and logged, never raised -- an auto-triggered
   * observability call can never crash the caller's ``llm.invoke()``.
   */
  async _autoEvaluate(): Promise<void> {
    if (this.observabilityMode === ObservabilityMode.NONE) return;
    if (this.observabilityMode === ObservabilityMode.IF_NOT_EVALUATED && this._evaluated) return;
    try {
      this.trellarEvaluateResult = await evaluateWithTrellarAgent(this, { _observabilityCall: true });
      this.trellarEvaluateError = null;
    } catch (error) {
      this.trellarEvaluateError = error;
      logger.warning("Auto-triggered evaluateConfidence() failed", error);
    }
  }
}
