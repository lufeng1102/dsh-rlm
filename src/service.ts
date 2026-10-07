/**
 * The RLM root state machine (Algorithm 1) — the plugin's centre of gravity.
 *
 * ```text
 * state   = InitREPL(prompt = P)          # context externalized, kernel open
 * history = [Metadata(state)]             # bounded metadata only
 * while true:
 *     action = RootLM(history)            # exactly one structured action
 *     if action is eval:
 *         state, stdout = REPL(state, code)
 *         history += bounded(code) + bounded(stdout) + step metadata
 *         if state[Final] is set: return state[Final]
 *     else:
 *         validate and publish the selected final result
 * ```
 *
 * Everything the paper insists on is enforced here rather than documented:
 * the prompt is never placed in root history; the kernel is persistent across
 * iterations; `llm_query` returns to the program, not to the root; intermediate
 * values stay symbolic; depth is host-derived and strictly decreasing; and the
 * whole tree shares one budget account.
 *
 * @module dsh-rlm/service
 */

import { createHash, randomUUID } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import { Service } from '@deepseek-ai/cordis'
import {
  RlmBudgetAccount,
  RlmBudgetError,
  narrowLimits,
  type RlmBudgetReservation,
} from './budget.ts'
import type { RlmContextStore } from './context-store.ts'
import {
  RLM_EVENT_TYPES,
  type RlmEvent,
  type RlmEventSink,
  type RlmRuntimeIdentity,
} from './events.ts'
import { RlmIsolationRefusedError, evaluateIsolation, type RlmIsolationPolicy, type RlmKernelRuntime } from './kernel/index.ts'
import { addUsage, emptyUsage, estimateCostUsd, type RlmModelClient, type RlmPricing, type RlmTokenUsage } from './llm.ts'
import { buildMetadataMessage, buildRlmSystemPrompt, buildStepFeedback, truncate, type RlmModelProfile } from './prompt.ts'
import { RlmScheduler } from './scheduler.ts'
import { buildParseRetryInstruction, parseRootAction } from './finalize.ts'
import type {
  RlmArtifactRef,
  RlmChildQueryArgs,
  RlmBudgetLimits,
  RlmContextHandle,
  RlmContextInput,
  RlmContextRef,
  RlmKernel,
  RlmKernelHandle,
  RlmKernelHost,
  RlmResult,
  RlmStepResult,
  RlmStopReason,
} from './types.ts'

export const RLM_VERSION = '0.1.0'

export const DEFAULT_BUDGETS: RlmBudgetLimits = {
  maxRootIterations: 32,
  maxWallClockMs: 900_000,
  maxTotalQueries: 512,
  maxConcurrentQueries: 8,
  maxTotalKernels: 8,
  maxConcurrentKernels: 4,
  maxInputTokens: 8_000_000,
  maxOutputTokens: 1_000_000,
  maxContextReadBytes: 256 * 1024 * 1024,
  maxQueryPromptChars: 2_000_000,
  maxInlineOutputBytes: 1024 * 1024,
  maxArtifactBytes: 1024 * 1024 * 1024,
  maxCostUsd: 25,
}

export interface RlmServiceConfig {
  defaultDepth: number
  maxDepth: number
  budgets: RlmBudgetLimits
  /** Storage/admission ceiling for one prompt; may exceed the root model window. */
  maxPromptChars: number
  /** Ceiling on the root conversation, the bounded-history witness. */
  maxRootHistoryChars: number
  /** Bounded parser retries, charged to the same budget as ordinary turns. */
  maxParseRetries: number
  maxCodePreviewChars: number
  maxStdoutPreviewChars: number
  /** Ceiling on one `llm_query` prompt, in characters. */
  maxQueryPromptChars: number
  maxErrorPreviewChars: number
  finalVariableName: string
  rootMaxTokens: number
  leafMaxTokens: number
  temperature?: number
  promptProfile: RlmModelProfile
  pricing: Record<string, RlmPricing>
  requireQualifiedProfile: boolean
  isolationPolicy: RlmIsolationPolicy
}

export interface RlmServiceDeps {
  contextStore: RlmContextStore
  kernelRuntime: RlmKernelRuntime
  model: RlmModelClient
  events: RlmEventSink
  now?: () => number
  createId?: () => string
}

export interface RlmRoute {
  provider: string
  model: string
}

export interface RlmRunRequest {
  /** The user prompt. Externalized by the store; never placed in root history. */
  prompt: RlmContextInput
  sessionId?: string
  depth?: number
  rootRoute?: RlmRoute
  leafRoute?: RlmRoute
  childRoute?: RlmRoute
  budgets?: Partial<RlmBudgetLimits>
  cwd: string
  signal: AbortSignal
  toolCallId?: string
  /**
   * The question for the run. Omitted when `prompt` already is the question
   * (the paper's shape); supplied explicitly when the material is passed as
   * `context` and the question must stay out of the prompt artifact.
   */
  task?: string
  /** Sink override for this run; defaults to the plugin's durable sink. */
  events?: RlmEventSink
}

/** Raised when the request itself is inadmissible, before any kernel exists. */
export class RlmAdmissionError extends Error {
  override readonly name = 'RlmAdmissionError'
  readonly code: string

  constructor(code: string, message: string) {
    super(message)
    this.code = code
  }
}

interface InvocationContext {
  invocationId: string
  attemptId: string
  parentInvocationId?: string
  depth: number
  account: RlmBudgetAccount
  scheduler: RlmScheduler
  rootRoute: RlmRoute
  leafRoute: RlmRoute
  childRoute: RlmRoute
  sessionId?: string
  cwd: string
  signal: AbortSignal
  toolCallId?: string
  identity: RlmRuntimeIdentity
  /** Set when a soft budget refusal happened; ends the run after the step. */
  softStop?: RlmStopReason
  /** Bounded kernel diagnostics (stderr, protocol notes); never model-visible. */
  diagnostics: string[]
  /**
   * Sink for this invocation. Per-run so a tool call can add a session-backed
   * sink for the session it belongs to without changing global state.
   */
  events: RlmEventSink
  /** The query this kernel must answer; host-supplied for a child RLM. */
  task?: string
}

export class RlmService extends Service {
  private readonly deps: RlmServiceDeps
  private readonly active = new Map<string, AbortController>()
  private disposed = false

  constructor(
    ctx: import('@deepseek-ai/cordis').Context,
    readonly config: RlmServiceConfig,
    deps: RlmServiceDeps,
  ) {
    super(ctx, 'rlm')
    this.deps = deps
    const unmet = evaluateIsolation(
      deps.kernelRuntime.providerName,
      deps.kernelRuntime.capabilities,
      config.isolationPolicy,
    )
    if (unmet.length > 0) {
      // Still load (the policy allowed an unhardened provider) but make the
      // degradation impossible to miss in logs and events.
      this.recordDegradation(unmet)
    }
    if (config.promptProfile.qualified !== true && config.requireQualifiedProfile) {
      throw new RlmAdmissionError(
        'profile-not-qualified',
        `prompt profile ${config.promptProfile.id} is not marked qualified`,
      )
    }
  }

  /** Live invocations, for concurrency accounting and unload. */
  get activeInvocations(): number {
    return this.active.size
  }

  /** Declared isolation gaps for this deployment; empty when the provider meets policy. */
  readonly degradations: string[] = []

  private recordDegradation(unmet: readonly string[]): void {
    for (const gap of unmet) this.degradations.push(gap)
  }

  /**
   * Run one RLM invocation.
   *
   * @throws {RlmAdmissionError} when the prompt or routes are inadmissible.
   */
  async run(request: RlmRunRequest): Promise<RlmResult> {
    if (this.disposed) throw new RlmAdmissionError('plugin-disposed', 'the RLM plugin has been unloaded')
    const depth = Math.max(0, Math.min(request.depth ?? this.config.defaultDepth, this.config.maxDepth))
    const leafRoute = request.leafRoute ?? this.config.promptProfile.leaf
    const rootRoute = request.rootRoute ?? this.config.promptProfile.root
    const childRoute = request.childRoute ?? this.config.promptProfile.child ?? leafRoute
    assertRoute(rootRoute, 'root')
    if (depth >= 1) assertRoute(leafRoute, 'leaf')

    if (this.active.size >= this.config.budgets.maxConcurrentKernels) {
      throw new RlmAdmissionError(
        'max-concurrent-kernels',
        `already running ${this.active.size} invocations (deployment ceiling ${this.config.budgets.maxConcurrentKernels})`,
      )
    }

    const budgets = narrowLimits(this.config.budgets, { ...this.config.budgets, ...request.budgets })
    const account = new RlmBudgetAccount(budgets, undefined, {
      scope: 'invocation',
      now: this.deps.now ?? Date.now,
    })
    const invocationId = (this.deps.createId ?? randomUUID)()
    const attemptId = randomUUID()
    const controller = new AbortController()
    const abort = () => controller.abort(request.signal.reason)
    request.signal.addEventListener('abort', abort, { once: true })

    const context: InvocationContext = {
      invocationId,
      attemptId,
      depth,
      account,
      scheduler: new RlmScheduler({
        maxConcurrency: Math.max(1, Math.min(budgets.maxConcurrentQueries, this.config.budgets.maxConcurrentQueries)),
        queueTimeoutMs: 60_000,
        now: this.deps.now ?? Date.now,
      }),
      rootRoute,
      leafRoute,
      childRoute,
      ...(request.sessionId === undefined ? {} : { sessionId: request.sessionId }),
      cwd: request.cwd,
      signal: controller.signal,
      diagnostics: [],
      ...(request.toolCallId === undefined ? {} : { toolCallId: request.toolCallId }),
      ...(request.task === undefined ? {} : { task: request.task }),
      events: request.events ?? this.deps.events,
      identity: {
        rlmVersion: RLM_VERSION,
        promptProfileId: this.config.promptProfile.id,
        promptProfileVariant: this.config.promptProfile.variant,
        kernelProvider: this.deps.kernelRuntime.providerName,
        kernelIsolation: this.deps.kernelRuntime.capabilities.isolation,
        contextStoreProvider: this.deps.contextStore.providerName,
        toolSchemaVersion: '1',
      },
    }
    this.active.set(invocationId, controller)
    try {
      const ref = await this.admitContext(context, request.prompt)
      return await this.runInvocation(context, ref, context.task)
    } finally {
      request.signal.removeEventListener('abort', abort)
      this.active.delete(invocationId)
    }
  }

  /** Abort every live invocation; used by unload and by operators. */
  async abortAll(reason: string): Promise<void> {
    const controllers = [...this.active.values()]
    for (const controller of controllers) controller.abort(reason)
    // Everything already aborted resolves through its own cleanup path; this
    // method does not wait on model calls, only on bookkeeping.
    this.active.clear()
  }

  /** Reject new runs and stop everything in flight. */
  async shutdown(reason: string): Promise<void> {
    this.disposed = true
    await this.abortAll(reason)
    await this.deps.kernelRuntime.disposeAll(reason)
    await this.deps.events.flush()
  }

  // --------------------------------------------------------------------- //
  // admission
  // --------------------------------------------------------------------- //

  private async admitContext(context: InvocationContext, prompt: RlmContextInput): Promise<RlmContextRef> {
    const chars = charsOf(prompt)
    if (chars > this.config.maxPromptChars) {
      throw new RlmAdmissionError(
        'prompt-too-large',
        `prompt of ${chars} characters exceeds the ${this.config.maxPromptChars} character admission ceiling`,
      )
    }
    return this.deps.contextStore.put(prompt, {
      ...(context.sessionId === undefined ? {} : { sessionId: context.sessionId }),
      signal: context.signal,
    })
  }

  // --------------------------------------------------------------------- //
  // the root loop
  // --------------------------------------------------------------------- //

  private async runInvocation(
    context: InvocationContext,
    contextRef: RlmContextRef,
    /** Present only for a child RLM: the query that created it. */
    childQuery?: string,
  ): Promise<RlmResult> {
    const now = this.deps.now ?? Date.now
    const startedAt = now()
    this.emit({
      type: 'rlm/start',
      time: startedAt,
      invocationId: context.invocationId,
      attemptId: context.attemptId,
      ...(context.parentInvocationId === undefined ? {} : { parentInvocationId: context.parentInvocationId }),
      depth: context.depth,
      maxDepth: this.config.maxDepth,
      contextRef,
      budgets: context.account.limits,
      identity: context.identity,
      rootRoute: context.rootRoute,
      leafRoute: context.leafRoute,
      ...(context.toolCallId === undefined ? {} : { toolCallId: context.toolCallId }),
      ...(context.sessionId === undefined ? {} : { sessionId: context.sessionId }),
      startedAt,
    }, context.events)
    // Once the run exists, record the admitted artifact. The order matters for a
    // trace reader: `rlm/start` is always the first record of an invocation.
    this.emitArtifact(
      context,
      {
        id: contextRef.id,
        sha256: contextRef.sha256,
        bytes: contextRef.bytes,
        chars: contextRef.chars,
        contentType: 'text/plain',
        preview: '',
        truncated: false,
      },
      'context',
    )

    let handle: RlmKernelHandle | undefined
    let steps = 0
    let stopReason: RlmStopReason = 'max-root-iterations'
    let finalText: string | undefined
    let finalArtifact: RlmArtifactRef | undefined
    let failure: { kind: string; message: string } | undefined
    let status: RlmResult['status']

    try {
      const bound = bindingsForDepth(context.depth)
      const contextHandle = await this.deps.contextStore.open(contextRef, {
        ...(context.sessionId === undefined ? {} : { sessionId: context.sessionId }),
        signal: context.signal,
      })
      handle = await this.deps.kernelRuntime.start({
        invocationId: context.invocationId,
        attemptId: context.attemptId,
        depth: context.depth,
        maxDepth: this.config.maxDepth,
        contextRef,
        contextKind: contextRef.kind,
        bindings: bound,
        finalVarName: this.config.finalVariableName,
        maxStdoutChars: this.config.maxStdoutPreviewChars * 4,
        maxConcurrency: Math.max(1, Math.min(context.account.limits.maxConcurrentQueries, 8)),
        cwd: context.cwd,
        maxLifetimeMs: context.account.limits.maxWallClockMs,
        evalTimeoutMs: Math.min(180_000, context.account.limits.maxWallClockMs),
        host: this.createHost(context, contextHandle),
        signal: context.signal,
      })
      context.account.charge({ maxTotalKernels: 1 })

      const systemPrompt = buildRlmSystemPrompt({
        ...(childQuery === undefined ? {} : { task: childQuery }),
        contextRef,
        depth: context.depth,
        maxDepth: this.config.maxDepth,
        profile: this.config.promptProfile,
        previewChars: 0,
        maxRootIterations: context.account.limits.maxRootIterations,
        maxTotalQueries: context.account.limits.maxTotalQueries,
        evalTimeoutMs: Math.min(180_000, context.account.limits.maxWallClockMs),
        finalVarName: this.config.finalVariableName,
      })
      const history: { role: 'user'; text: string }[] = [
        { role: 'user', text: buildMetadataMessage({
          ...(childQuery === undefined ? {} : { task: childQuery }),
          contextRef,
          depth: context.depth,
          maxDepth: this.config.maxDepth,
          profile: this.config.promptProfile,
          previewChars: 0,
          maxRootIterations: context.account.limits.maxRootIterations,
          maxTotalQueries: context.account.limits.maxTotalQueries,
          evalTimeoutMs: Math.min(180_000, context.account.limits.maxWallClockMs),
          finalVarName: this.config.finalVariableName,
        }) },
      ]

      for (let step = 1; step <= context.account.limits.maxRootIterations; step += 1) {
        const guard = this.guard(context)
        if (guard) {
          stopReason = guard
          break
        }
        const turn = await this.rootTurn(context, systemPrompt, history)
        if (turn.kind === 'model-failure') {
          stopReason = 'root-model-failure'
          failure = turn.error
          break
        }
        context.account.charge({ maxRootIterations: 1 })
        steps = step

        const parsed = parseRootAction(turn.text, { codeFenceLanguages: ['repl', 'python', 'py'] })
        if (!parsed.ok) {
          const retries = history.filter((message) => message.text.startsWith('# Protocol correction')).length
          if (retries >= this.config.maxParseRetries) {
            stopReason = 'root-parse-failure'
            failure = { kind: 'parse-failure', message: parsed.failure.message }
            this.emitStep(context, step, 'parse-failure', undefined, history, undefined)
            break
          }
          history.push({ role: 'user', text: `# Protocol correction ${retries + 1}\n\n${buildParseRetryInstruction(parsed.failure, retries + 1, this.config.maxParseRetries)}` })
          this.emitStep(context, step, 'parse-failure', undefined, history, undefined)
          continue
        }

        const action = parsed.action
        if (action.type === 'eval') {
          const stepResult = await handle.kernel.eval(action.code, {
            timeoutMs: Math.min(180_000, context.account.limits.maxWallClockMs),
            signal: context.signal,
          })
          this.emitStep(context, step, 'eval', stepResult, history, action.code)
          if (stepResult.error?.fatal) {
            stopReason = fatalStopReason(stepResult.error.kind)
            failure = { kind: stepResult.error.kind, message: stepResult.error.message }
            break
          }
          if (stepResult.final) {
            const published = await this.publishFinalVariable(context, handle.kernel, stepResult.final.name)
            if (published === undefined) {
              if (!this.recoverFromFinalVariable(context, history, step, stepResult.final.name, 'the value could not be exported')) break
              continue
            }
            finalText = published.text
            finalArtifact = published.artifactRef
            stopReason = 'final-variable'
            break
          }
          history.push({
            role: 'user',
            text: buildStepFeedback({
              step,
              code: action.code,
              result: stepResult,
              maxCodeChars: this.config.maxCodePreviewChars,
              maxStdoutChars: this.config.maxStdoutPreviewChars,
              maxErrorChars: this.config.maxErrorPreviewChars,
            }),
          })
          this.assertRootHistoryBounded(history)
          continue
        }

        if (action.type === 'final-text') {
          const published = await this.publishText(context, action.text)
          finalText = published.text
          finalArtifact = published.artifactRef
          stopReason = 'final-text'
          this.emitStep(context, step, 'final-text', undefined, history, undefined)
          break
        }

        const published = await this.publishFinalVariable(context, handle.kernel, action.name)
        if (published === undefined) {
          // The paper documents models misusing FINAL_VAR (13% of turns in its
          // analysis). A wrong variable name is a protocol error to correct, not
          // a reason to throw away an otherwise healthy run.
          if (!this.recoverFromFinalVariable(context, history, step, action.name, 'no such REPL variable')) break
          continue
        }
        finalText = published.text
        finalArtifact = published.artifactRef
        stopReason = 'final-variable'
        this.emitStep(context, step, 'final-variable', undefined, history, undefined)
        break
      }
    } catch (error) {
      if (error instanceof RlmBudgetError) {
        stopReason = 'budget-exceeded'
        failure = { kind: error.dimension, message: error.message }
      } else if (error instanceof RlmIsolationRefusedError) {
        stopReason = 'kernel-failure'
        failure = { kind: 'sandbox-unavailable', message: error.message }
      } else if (context.signal.aborted) {
        stopReason = 'cancelled'
        failure = { kind: 'cancelled', message: 'invocation cancelled' }
      } else {
        stopReason = 'kernel-failure'
        failure = { kind: 'invocation-failure', message: (error as Error).message }
      }
    } finally {
      if (handle) await handle.dispose().catch(() => {})
    }

    if (context.signal.aborted && stopReason !== 'final-text' && stopReason !== 'final-variable') {
      stopReason = 'cancelled'
      failure = { kind: 'cancelled', message: 'invocation cancelled' }
    }
    status = statusFor(stopReason, finalText, finalArtifact)
    const finishedAt = now()
    const usage = context.account.usage()
    this.emit({
      type: 'rlm/end',
      time: finishedAt,
      invocationId: context.invocationId,
      attemptId: context.attemptId,
      status,
      stopReason,
      steps,
      usage,
      ...(failure ? { errorKind: failure.kind, errorMessage: failure.message } : {}),
      ...(finalArtifact ? { artifactRef: finalArtifact } : {}),
      finishedAt,
      wallClockMs: finishedAt - startedAt,
    }, context.events)

    return {
      status,
      ...(finalText === undefined ? {} : { text: finalText }),
      ...(finalArtifact === undefined ? {} : { artifactRef: finalArtifact }),
      stopReason,
      depth: context.depth,
      maxDepth: this.config.maxDepth,
      steps,
      usage,
      ...(failure ? { error: failure } : {}),
    }
  }

  private guard(context: InvocationContext): RlmStopReason | undefined {
    if (context.signal.aborted) return 'cancelled'
    if (context.softStop) return context.softStop
    if (context.account.elapsedMs() >= context.account.limits.maxWallClockMs) return 'wall-clock'
    const remaining = context.account.remainingLimits()
    if (remaining.maxRootIterations <= 0) return 'max-root-iterations'
    if (remaining.maxInputTokens <= 0 || remaining.maxOutputTokens <= 0 || remaining.maxCostUsd <= 0) {
      return 'budget-exceeded'
    }
    return undefined
  }

  private async rootTurn(
    context: InvocationContext,
    systemPrompt: string,
    history: readonly { role: 'user'; text: string }[],
  ): Promise<{ kind: 'text'; text: string } | { kind: 'model-failure'; error: { kind: string; message: string } }> {
    const budget = context.account.remainingLimits()
    const response = await this.deps.model.complete({
      provider: context.rootRoute.provider,
      model: context.rootRoute.model,
      system: systemPrompt,
      messages: history,
      maxTokens: Math.max(256, Math.min(this.config.rootMaxTokens, budget.maxOutputTokens)),
      ...(this.config.temperature === undefined ? {} : { temperature: this.config.temperature }),
      signal: context.signal,
    })
    this.chargeUsage(context, response.usage)
    if (response.error) {
      return { kind: 'model-failure', error: { kind: response.error.kind, message: response.error.message } }
    }
    return { kind: 'text', text: response.text }
  }

  private chargeUsage(context: InvocationContext, usage: RlmTokenUsage): void {
    const route = context.rootRoute
    const cost = estimateCostUsd(usage, this.config.pricing[`${route.provider}/${route.model}`])
    context.account.charge({
      ...(usage.inputTokens + usage.cacheReadTokens + usage.cacheWriteTokens > 0
        ? { maxInputTokens: usage.inputTokens + usage.cacheReadTokens + usage.cacheWriteTokens }
        : {}),
      ...(usage.outputTokens > 0 ? { maxOutputTokens: usage.outputTokens } : {}),
      ...(cost > 0 ? { maxCostUsd: cost } : {}),
    })
  }

  private assertRootHistoryBounded(history: readonly { text: string }[]): void {
    const total = history.reduce((sum, message) => sum + message.text.length, 0)
    if (total > this.config.maxRootHistoryChars) {
      throw new RlmAdmissionError(
        'root-history-unbounded',
        `root history reached ${total} characters, above the ${this.config.maxRootHistoryChars} character bound`,
      )
    }
  }

  // --------------------------------------------------------------------- //
  // finalization
  // --------------------------------------------------------------------- //

  /**
   * Export a final variable, converting a missing-variable failure into a bounded
   * protocol correction.
   *
   * @returns the published value, or `undefined` when the caller must either
   *   correct the model or end the run.
   */
  private async publishFinalVariable(
    context: InvocationContext,
    kernel: RlmKernel,
    name: string,
  ): Promise<{ text?: string; artifactRef?: RlmArtifactRef } | undefined> {
    try {
      return await this.publishVariable(context, kernel, name)
    } catch (error) {
      this.lastFinalVariableError = (error as Error).message
      return undefined
    }
  }

  private lastFinalVariableError?: string

  /** Push a correction for an unusable final variable; false means "stop". */
  private recoverFromFinalVariable(
    context: InvocationContext,
    history: { role: 'user'; text: string }[],
    step: number,
    name: string,
    reason: string,
  ): boolean {
    const corrections = history.filter((message) => message.text.startsWith('# Protocol correction')).length
    if (corrections >= this.config.maxParseRetries) {
      this.emitStep(context, step, 'parse-failure', undefined, history, undefined)
      return false
    }
    history.push({
      role: 'user',
      text: [
        `# Protocol correction ${corrections + 1}`,
        '',
        `FINAL_VAR(${name}) could not be returned: ${reason}. ${this.lastFinalVariableError ?? ''}`.trim(),
        'Reply with a ```repl block that assigns an existing variable (or prints what you need), then FINAL_VAR(<that variable>).',
        'If you already have the answer, reply with FINAL(<answer text>).',
      ].join('\n'),
    })
    this.emitStep(context, step, 'parse-failure', undefined, history, undefined)
    return true
  }

  private async publishText(
    context: InvocationContext,
    text: string,
  ): Promise<{ text?: string; artifactRef?: RlmArtifactRef }> {
    const bytes = Buffer.byteLength(text, 'utf8')
    if (bytes <= context.account.limits.maxInlineOutputBytes) return { text }
    const artifactRef = await this.deps.contextStore.writeArtifact(text, {
      ...(context.sessionId === undefined ? {} : { sessionId: context.sessionId }),
      maxBytes: context.account.limits.maxArtifactBytes,
      previewChars: 2000,
    })
    this.emitArtifact(context, artifactRef, 'output')
    return { artifactRef }
  }

  /**
   * Export a REPL variable and publish it.
   *
   * The value is serialized by the kernel into its private workspace, so even a
   * multi-gigabyte answer crosses the control channel as bounded metadata only
   * (README §4.6) — this is the property that lets the final output exceed the
   * root model's window.
   */
  private async publishVariable(
    context: InvocationContext,
    kernel: RlmKernel,
    name: string,
  ): Promise<{ text?: string; artifactRef?: RlmArtifactRef }> {
    // The kernel writes into its own private workspace, so the host never has
    // to trust a model-supplied path and never buffers the value itself.
    const exportPath = join(kernel.workspace, `final-${randomUUID()}.out`)
    const exported = await kernel.export(name, {
      path: exportPath,
      previewChars: 2000,
      maxBytes: context.account.limits.maxArtifactBytes,
      signal: context.signal,
    })
    if (exported.bytes <= context.account.limits.maxInlineOutputBytes) {
      const text = await readFile(exportPath, 'utf8')
      return { text }
    }
    const artifactRef = await this.deps.contextStore.writeArtifactFromFile(exportPath, {
      ...(context.sessionId === undefined ? {} : { sessionId: context.sessionId }),
      maxBytes: context.account.limits.maxArtifactBytes,
      previewChars: 2000,
    })
    this.emitArtifact(context, artifactRef, 'output')
    return { artifactRef }
  }

  // --------------------------------------------------------------------- //
  // kernel host bindings
  // --------------------------------------------------------------------- //

  private createHost(context: InvocationContext, contextHandle: RlmContextHandle): RlmKernelHost {
    let nextCallId = 1
    return {
      llmQuery: (args, signal) => this.llmQuery(context, args, signal, () => nextCallId++),
      rlmQuery: (args, signal) => this.rlmQuery(context, args, signal, () => nextCallId++),
      contextCall: (name, args, signal) => this.contextCall(context, contextHandle, name, args, signal),
      observe: (observation) => {
        // Kernel diagnostics are never model-visible; they are kept for the
        // operator and for the durable trace.
        if (observation.type === 'stderr') context.diagnostics.push(observation.text)
        if (context.diagnostics.length > 200) context.diagnostics.shift()
      },
    }
  }

  private async contextCall(
    context: InvocationContext,
    handle: RlmContextHandle,
    name: string,
    args: Record<string, unknown>,
    signal: AbortSignal,
  ): Promise<unknown> {
    if (signal.aborted) throw new Error('cancelled')
    const remaining = context.account.remainingLimits().maxContextReadBytes
    if (remaining <= 0) {
      context.softStop = 'budget-exceeded'
      throw new Error('context read budget exhausted')
    }
    const value = await dispatchContextCall(handle, name, args, signal)
    const bytes = estimateBytes(value)
    if (bytes > remaining) {
      context.softStop = 'budget-exceeded'
      throw new Error(`context read of ${bytes} bytes exceeds the ${remaining} bytes remaining in the read budget`)
    }
    context.account.charge({ maxContextReadBytes: bytes })
    return value
  }

  private async llmQuery(
    context: InvocationContext,
    args: { prompt: string; model?: string; maxTokens?: number },
    signal: AbortSignal,
    nextCallId: () => number,
  ): Promise<unknown> {
    if (context.depth < 1) throw new Error('llm_query is not available at depth 0')
    const prompt = args.prompt
    const limit = context.account.limits.maxQueryPromptChars
    if (prompt.length > limit) {
      throw new Error(`query prompt of ${prompt.length} characters exceeds the ${limit} character ceiling`)
    }
    return this.executeQuery(context, nextCallId(), 'llm_query', context.leafRoute, prompt, args.maxTokens, signal, args.model)
  }

  private async executeQuery(
    context: InvocationContext,
    callId: number,
    kind: 'llm_query' | 'rlm_query',
    route: RlmRoute,
    prompt: string,
    requestedMaxTokens: number | undefined,
    signal: AbortSignal,
    modelOverride?: string,
  ): Promise<unknown> {
    const maxTokens = Math.max(
      1,
      Math.min(requestedMaxTokens ?? this.config.leafMaxTokens, context.account.limits.maxOutputTokens),
    )
    const estimatedInputTokens = Math.ceil(prompt.length / 4)
    let reservation: RlmBudgetReservation
    try {
      reservation = context.account.reserve({
        maxTotalQueries: 1,
        maxInputTokens: estimatedInputTokens,
        maxOutputTokens: maxTokens,
      })
    } catch (error) {
      if (error instanceof RlmBudgetError) {
        // A soft stop: the program sees a truthful refusal, and the loop ends
        // after this step rather than being killed mid-expression.
        context.softStop = 'budget-exceeded'
        throw new Error(`query refused: ${error.message}`)
      }
      throw error
    }

    const routeUsed: RlmRoute =
      kind === 'llm_query' ? { provider: route.provider, model: modelOverride ?? route.model } : route
    const startedAt = (this.deps.now ?? Date.now)()
    this.emit({
      type: 'rlm/query-start',
      time: startedAt,
      invocationId: context.invocationId,
      attemptId: context.attemptId,
      callId,
      kind,
      depth: context.depth,
      promptChars: prompt.length,
      route: routeUsed,
      startedAt,
    }, context.events)

    try {
      const response = await context.scheduler.run(
        () =>
          this.deps.model.complete({
            provider: routeUsed.provider,
            model: routeUsed.model,
            messages: [{ role: 'user', text: prompt }],
            maxTokens,
            ...(this.config.temperature === undefined ? {} : { temperature: this.config.temperature }),
            signal,
          }),
        { signal, label: `${kind}#${callId}` },
      )
      const usage = response.usage
      const cost = estimateCostUsd(
        usage,
        this.config.pricing[`${routeUsed.provider}/${routeUsed.model}`],
      )
      reservation.settle({
        maxTotalQueries: 1,
        maxInputTokens: usage.inputTokens + usage.cacheReadTokens + usage.cacheWriteTokens,
        maxOutputTokens: usage.outputTokens,
        ...(cost > 0 ? { maxCostUsd: cost } : {}),
      })
      context.account.charge({ maxConcurrentQueries: context.scheduler.peak })
      const endedAt = (this.deps.now ?? Date.now)()
      this.emit({
        type: 'rlm/query-end',
        time: endedAt,
        invocationId: context.invocationId,
        attemptId: context.attemptId,
        callId,
        kind,
        ok: response.error === undefined,
        ...(response.error ? { errorKind: response.error.kind } : {}),
        elapsedMs: endedAt - startedAt,
        inputTokens: usage.inputTokens + usage.cacheReadTokens + usage.cacheWriteTokens,
        outputTokens: usage.outputTokens,
        resultChars: response.text.length,
      }, context.events)
      if (response.error) {
        throw new Error(`model call failed (${response.error.kind}): ${response.error.message}`)
      }
      return response.text
    } catch (error) {
      reservation.settle({ maxTotalQueries: 1 })
      const endedAt = (this.deps.now ?? Date.now)()
      this.emit({
        type: 'rlm/query-end',
        time: endedAt,
        invocationId: context.invocationId,
        attemptId: context.attemptId,
        callId,
        kind,
        ok: false,
        errorKind: error instanceof Error ? error.name : 'error',
        elapsedMs: endedAt - startedAt,
        inputTokens: 0,
        outputTokens: 0,
        resultChars: 0,
      }, context.events)
      throw error
    }
  }

  private async rlmQuery(
    context: InvocationContext,
    args: RlmChildQueryArgs,
    signal: AbortSignal,
    nextCallId: () => number,
  ): Promise<unknown> {
    const callId = nextCallId()
    if (context.depth < 2) {
      // The paper's contract: at the recursion ceiling `rlm_query` degrades to a
      // leaf call instead of spawning a child kernel.
      const materialized = await materializeChildArg(
        this.deps.contextStore,
        args.context,
        this.config.maxQueryPromptChars,
      )
      return this.executeQuery(
        context,
        callId,
        'rlm_query',
        context.leafRoute,
        `${args.query}\n\n${materialized}`,
        this.config.leafMaxTokens,
        signal,
      )
    }

    // A handle-shaped child context is passed by reference: the child gets its
    // own REPL over the same immutable artifact, with no copy and no
    // materialization into this invocation's memory.
    const inheritedRef = args.context.kind === 'handle' ? args.context.ref : undefined
    const childInput: RlmContextInput | undefined =
      args.context.kind === 'inline' ? toContextInput(args.context) : undefined
    const childChars = inheritedRef?.chars ?? (childInput ? charsOf(childInput) : 0)

    const reservation = context.account.reserve({
      maxTotalQueries: 1,
      maxTotalKernels: 1,
      maxInputTokens: Math.ceil(childChars / 4),
      maxOutputTokens: this.config.leafMaxTokens,
    })
    const remainingBeforeChild = context.account.remainingLimits()
    const childAccount = context.account.child(
      {
        maxRootIterations: Math.min(16, Math.max(1, remainingBeforeChild.maxRootIterations)),
        maxTotalQueries: Math.min(64, Math.max(1, remainingBeforeChild.maxTotalQueries)),
        maxTotalKernels: Math.min(2, Math.max(1, remainingBeforeChild.maxTotalKernels)),
        maxWallClockMs: Math.max(
          1_000,
          Math.min(context.account.limits.maxWallClockMs - context.account.elapsedMs(), 300_000),
        ),
      },
      `child-rlm@${context.depth - 1}`,
    )
    const startedAt = (this.deps.now ?? Date.now)()
    this.emit({
      type: 'rlm/query-start',
      time: startedAt,
      invocationId: context.invocationId,
      attemptId: context.attemptId,
      callId,
      kind: 'rlm_query',
      depth: context.depth,
      promptChars: args.query.length,
      route: context.childRoute,
      startedAt,
    }, context.events)
    const childContext: InvocationContext = {
      ...context,
      invocationId: (this.deps.createId ?? randomUUID)(),
      attemptId: randomUUID(),
      parentInvocationId: context.invocationId,
      depth: context.depth - 1,
      account: childAccount,
      scheduler: new RlmScheduler({
        maxConcurrency: Math.max(1, Math.min(childAccount.remainingLimits().maxConcurrentQueries, 4)),
        queueTimeoutMs: 60_000,
        now: this.deps.now ?? Date.now,
      }),
      rootRoute: this.config.promptProfile.child ?? context.childRoute,
      leafRoute: context.leafRoute,
      childRoute: context.childRoute,
      signal,
      diagnostics: [],
      events: context.events,
    }
    try {
      const ref = inheritedRef ?? (await this.admitContext(childContext, childInput!))
      const childResult = await this.runInvocation(childContext, ref, args.query)
      const endedAt = (this.deps.now ?? Date.now)()
      this.emit({
        type: 'rlm/query-end',
        time: endedAt,
        invocationId: context.invocationId,
        attemptId: context.attemptId,
        callId,
        kind: 'rlm_query',
        ok: childResult.status === 'ok',
        ...(childResult.status === 'ok' ? {} : { errorKind: childResult.stopReason }),
        elapsedMs: endedAt - startedAt,
        inputTokens: childResult.usage.inputTokens,
        outputTokens: childResult.usage.outputTokens,
        resultChars: childResult.text?.length ?? 0,
      }, context.events)
      reservation.settle({
        maxTotalQueries: 1,
        maxTotalKernels: 1,
        maxInputTokens: childResult.usage.inputTokens,
        maxOutputTokens: childResult.usage.outputTokens,
      })
      if (childResult.status !== 'ok') {
        throw new Error(
          `child RLM ended with ${childResult.stopReason}${childResult.error ? `: ${childResult.error.message}` : ''}`,
        )
      }
      return childResult.text ?? childResult.artifactRef?.preview ?? ''
    } catch (error) {
      reservation.settle({ maxTotalQueries: 1, maxTotalKernels: 1 })
      throw error
    }
  }

  // --------------------------------------------------------------------- //
  // events
  // --------------------------------------------------------------------- //

  private emit(event: RlmEvent, sink?: RlmEventSink): void {
    ;(sink ?? this.deps.events).emit(event)
  }

  private emitArtifact(context: InvocationContext, artifactRef: RlmArtifactRef, role: 'context' | 'output'): void {
    this.emit({
      type: 'rlm/artifact',
      time: (this.deps.now ?? Date.now)(),
      invocationId: context.invocationId,
      attemptId: context.attemptId,
      artifactRef,
      role,
    }, context.events)
  }

  private emitStep(
    context: InvocationContext,
    step: number,
    action: 'eval' | 'final-text' | 'final-variable' | 'parse-failure',
    result: RlmStepResult | undefined,
    history: readonly { text: string }[],
    code: string | undefined,
  ): void {
    this.emit({
      type: 'rlm/step',
      time: (this.deps.now ?? Date.now)(),
      invocationId: context.invocationId,
      attemptId: context.attemptId,
      step,
      action,
      ...(code === undefined ? {} : { codeChars: code.length }),
      ...(result === undefined
        ? {}
        : {
            stdoutChars: result.stdout.length,
            stdoutTruncated: result.stdoutTruncated,
            ...(result.error ? { errorKind: result.error.kind } : {}),
          }),
      elapsedMs: result?.elapsedMs ?? 0,
      rootHistoryChars: history.reduce((sum, message) => sum + message.text.length, 0),
      usage: context.account.usage(),
    }, context.events)
  }

  /** Event types this build can write into a session log, for the session sink. */
  static readonly eventTypes = RLM_EVENT_TYPES
}

// ------------------------------------------------------------------------- //
// helpers
// ------------------------------------------------------------------------- //

function assertRoute(route: RlmRoute, name: string): void {
  if (!route.provider || route.provider === 'unset' || !route.model || route.model === 'unset') {
    throw new RlmAdmissionError(
      'route-unset',
      `the ${name} model route is not configured (provider=${route.provider}, model=${route.model})`,
    )
  }
}

function statusFor(stopReason: RlmStopReason, text: string | undefined, artifact: RlmArtifactRef | undefined): RlmResult['status'] {
  if (stopReason === 'final-text' || stopReason === 'final-variable' || stopReason === 'ok') {
    return text !== undefined || artifact !== undefined ? 'ok' : 'failed'
  }
  if (stopReason === 'cancelled') return 'cancelled'
  if (stopReason === 'budget-exceeded' || stopReason === 'max-root-iterations' || stopReason === 'wall-clock') {
    return 'budget-exceeded'
  }
  return 'failed'
}

function fatalStopReason(kind: string): RlmStopReason {
  switch (kind) {
    case 'cancelled': return 'cancelled'
    case 'timeout': return 'budget-exceeded'
    case 'budget': return 'budget-exceeded'
    case 'protocol':
    case 'sandbox-unavailable':
    case 'worker-exit':
    case 'syntax':
    case 'name':
    case 'exception':
    default:
      return 'kernel-failure'
  }
}

function charsOf(input: RlmContextInput): number {
  switch (input.kind) {
    case 'text': return input.text.length
    case 'list': return input.items.reduce((sum, item) => sum + item.length, 0)
    case 'documents': return input.documents.reduce((sum, item) => sum + item.length, 0)
    case 'record': return JSON.stringify(input.record).length
    default: return 0
  }
}

function estimateBytes(value: unknown): number {
  if (typeof value === 'string') return Buffer.byteLength(value, 'utf8')
  if (value === undefined || value === null) return 0
  return Buffer.byteLength(JSON.stringify(value) ?? '', 'utf8')
}

async function dispatchContextCall(
  handle: RlmContextHandle,
  name: string,
  args: Record<string, unknown>,
  signal: AbortSignal,
): Promise<unknown> {
  switch (name) {
    case 'context.meta':
    case 'context.ref':
      return name === 'context.ref' ? handle.ref : { kind: handle.kind, length: handle.length, chars: handle.ref.chars, bytes: handle.ref.bytes, sha256: handle.ref.sha256 }
    case 'context.peek':
      return handle.peek(numberArg(args.start, 0), optionalNumberArg(args.end), signal)
    case 'context.slice':
      return handle.slice(numberArg(args.start, 0), numberArg(args.end, handle.ref.chars), signal)
    case 'context.items': {
      const start = numberArg(args.start, 0)
      const end = numberArg(args.end, start + 100)
      return handle.items(start, end, signal)
    }
    case 'context.get':
      if (typeof args.key !== 'string') throw new Error('context.get requires a string key')
      return handle.get(args.key, signal)
    case 'context.search':
      if (typeof args.query !== 'string') throw new Error('context.search requires a string query')
      return handle.search(args.query, {
        maxHits: optionalNumberArg(args.maxHits) ?? 20,
        ...(args.regex === true ? { regex: true } : {}),
      })
    case 'context.materialize':
      return handle.materialize({ maxBytes: numberArg(args.maxBytes, 1_000_000) }, signal)
    default:
      throw new Error(`unknown context call ${JSON.stringify(name)}`)
  }
}

function numberArg(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? Math.floor(value) : fallback
}

function optionalNumberArg(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? Math.floor(value) : undefined
}

/**
 * Binding names installed for a given remaining depth.
 *
 * `depth=0` is the ablation the README requires: the prompt is externalized and
 * the REPL is still program-driven, but no model binding exists at all, so a
 * run cannot silently become recursive.
 */
export function bindingsForDepth(depth: number): string[] {
  if (depth < 1) return []
  // `rlm_query` is installed at depth 1 too: the prompt tells the model it
  // degrades to a leaf call, and the host enforces that in `rlmQuery`.
  return ['llm_query', 'llm_query_many', 'rlm_query', 'rlm_query_many']
}

/** Convert an `rlm_query` inline argument into a storable context input. */
export function toContextInput(
  argument: { kind: 'inline'; text?: string; items?: string[]; record?: Record<string, string> },
): RlmContextInput {
  if (argument.items) return { kind: 'list', items: argument.items }
  if (argument.record) return { kind: 'record', record: argument.record }
  return { kind: 'text', text: argument.text ?? '' }
}

/** Turn an `rlm_query` context argument into the child's own context input. */
async function childContextInput(
  store: RlmContextStore,
  argument: { kind: 'handle'; ref: RlmContextRef } | { kind: 'inline'; text?: string; items?: string[]; record?: Record<string, string> },
  sessionId: string | undefined,
): Promise<RlmContextInput> {
  if (argument.kind === 'inline') return toContextInput(argument)
  // A handle-shaped argument is materialized through the store, so the parent's
  // authorization check stays on the read path.
  const text = await store.read(argument.ref, { maxBytes: 64 * 1024 * 1024 })
  void sessionId
  return { kind: 'text', text }
}

async function materializeChildArg(
  store: RlmContextStore,
  argument: { kind: 'handle'; ref: RlmContextRef } | { kind: 'inline'; text?: string; items?: string[]; record?: Record<string, string> },
  maxChars: number,
): Promise<string> {
  const input = await childContextInput(store, argument, undefined)
  const text = input.kind === 'text' ? input.text : charsOf(input) > 0 ? JSON.stringify(input) : ''
  return truncate(text, maxChars)
}

/** Stable hash of a route pair, used in fingerprints and diagnostics. */
export function routeFingerprint(route: RlmRoute): string {
  return createHash('sha256').update(`${route.provider}/${route.model}`).digest('hex').slice(0, 12)
}

export { addUsage, emptyUsage }

/** Resolve a working directory for the kernel from a session cwd or process cwd. */
export function resolveInvocationCwd(cwd: string | undefined): string {
  return cwd ? resolve(cwd) : resolve(process.cwd())
}
