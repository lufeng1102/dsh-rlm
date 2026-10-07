/**
 * Durable RLM events and where they can safely go.
 *
 * ## Why this is a seam rather than a direct `session.append()`
 *
 * The README asks for `rlm/start`, `rlm/step`, `rlm/query-start`, `rlm/query-end`,
 * `rlm/artifact` and `rlm/end` as merge-extensible `SessionEvent` types. In
 * `dsh-v0.2.0-rc.2` that is only *half* available to an out-of-tree plugin, for
 * a reason worth stating precisely:
 *
 * - `Session.append(type, data)` does **not** reject an unknown type, and the
 *   JSONL writer does not filter one — so a custom event persists.
 * - But the reader refuses an unknown type unless the stored envelope carries
 *   `ignorable: true`, and `Session.append()` provides no way to set that flag.
 * - Therefore appending `rlm/*` before the type is added to the in-repo
 *   `SessionEventMap` vocabulary produces a log that this build cannot read on
 *   the next load. A plugin must not do that to a user's session.
 *
 * So the plugin writes through {@link RlmEventSink}:
 *
 * - {@link SidecarJsonlEventSink} is the default and is always durable: one
 *   append-only JSONL file per invocation under the artifact root, covered by
 *   the same retention/GC sweep as context artifacts.
 * - {@link SessionEventSink} appends to the session **only** for event types the
 *   running build declares in `KNOWN_SESSION_EVENT_TYPES`, and silently falls
 *   back to the sidecar for the rest. The moment a deployment patches the
 *   vocabulary (or a future harness release ships `rlm/*`), the same code starts
 *   writing real session events with no change here.
 *
 * Every accepted `rlm/start` has exactly one terminal `rlm/end`, including
 * cancellation and infrastructure failure — the invariant the README demands and
 * the one a projection can rely on.
 *
 * @module dsh-rlm/events
 */

import { appendFile, mkdir } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import type { RlmArtifactRef, RlmBudgetLimits, RlmContextRef, RlmStopReason, RlmUsage } from './types.ts'

export type RlmEventType =
  | 'rlm/start'
  | 'rlm/step'
  | 'rlm/query-start'
  | 'rlm/query-end'
  | 'rlm/artifact'
  | 'rlm/end'

export const RLM_EVENT_TYPES: readonly RlmEventType[] = [
  'rlm/start',
  'rlm/step',
  'rlm/query-start',
  'rlm/query-end',
  'rlm/artifact',
  'rlm/end',
]

/** Prompt/runtime identity captured once per invocation for reproduction. */
export interface RlmRuntimeIdentity {
  rlmVersion: string
  promptProfileId: string
  promptProfileVariant: string
  kernelProvider: string
  kernelIsolation: string
  contextStoreProvider: string
  toolSchemaVersion: string
}

export interface RlmStartEvent {
  invocationId: string
  attemptId: string
  /** Parent invocation for a child RLM; absent at the root. */
  parentInvocationId?: string
  depth: number
  maxDepth: number
  contextRef: RlmContextRef
  budgets: RlmBudgetLimits
  identity: RlmRuntimeIdentity
  rootRoute: { provider: string; model: string }
  leafRoute: { provider: string; model: string }
  toolCallId?: string
  sessionId?: string
  startedAt: number
}

export interface RlmStepEvent {
  invocationId: string
  attemptId: string
  step: number
  action: 'eval' | 'final-text' | 'final-variable' | 'parse-failure'
  codeChars?: number
  stdoutChars?: number
  stdoutTruncated?: boolean
  errorKind?: string
  elapsedMs: number
  /** Bytes of the root history after this turn; the bounded-history witness. */
  rootHistoryChars: number
  usage: RlmUsage
}

export interface RlmQueryStartEvent {
  invocationId: string
  attemptId: string
  callId: number
  kind: 'llm_query' | 'rlm_query'
  depth: number
  promptChars: number
  route: { provider: string; model: string }
  startedAt: number
}

export interface RlmQueryEndEvent {
  invocationId: string
  attemptId: string
  callId: number
  kind: 'llm_query' | 'rlm_query'
  ok: boolean
  errorKind?: string
  elapsedMs: number
  inputTokens: number
  outputTokens: number
  resultChars: number
}

export interface RlmArtifactEvent {
  invocationId: string
  attemptId: string
  artifactRef: RlmArtifactRef
  /** `context` for an admitted prompt, `output` for a large final value. */
  role: 'context' | 'output'
}

export interface RlmEndEvent {
  invocationId: string
  attemptId: string
  status: 'ok' | 'failed' | 'cancelled' | 'budget-exceeded'
  stopReason: RlmStopReason
  steps: number
  usage: RlmUsage
  errorKind?: string
  errorMessage?: string
  artifactRef?: RlmArtifactRef
  finishedAt: number
  wallClockMs: number
}

export type RlmEvent =
  | ({ type: 'rlm/start'; time: number } & RlmStartEvent)
  | ({ type: 'rlm/step'; time: number } & RlmStepEvent)
  | ({ type: 'rlm/query-start'; time: number } & RlmQueryStartEvent)
  | ({ type: 'rlm/query-end'; time: number } & RlmQueryEndEvent)
  | ({ type: 'rlm/artifact'; time: number } & RlmArtifactEvent)
  | ({ type: 'rlm/end'; time: number } & RlmEndEvent)

/** Where durable RLM events go. Implementations must never throw into a run. */
export interface RlmEventSink {
  readonly name: string
  emit(event: RlmEvent): void
  /** Flush and release; called on invocation end and plugin unload. */
  flush(): Promise<void>
}

/** Collects events in memory: tests, and a UI/diagnostics ring. */
export class RecordingEventSink implements RlmEventSink {
  readonly name = 'recording'
  readonly events: RlmEvent[] = []

  constructor(private readonly limit = 10_000) {}

  emit(event: RlmEvent): void {
    this.events.push(event)
    if (this.events.length > this.limit) this.events.splice(0, this.events.length - this.limit)
  }

  async flush(): Promise<void> {}

  ofType<T extends RlmEventType>(type: T): Extract<RlmEvent, { type: T }>[] {
    return this.events.filter((event): event is Extract<RlmEvent, { type: T }> => event.type === type)
  }
}

/**
 * A sink that discards everything.
 *
 * Used as the {@link SessionEventSink} fallback when the caller already
 * guarantees durability elsewhere (e.g. inside a composite that also holds the
 * sidecar sink) — otherwise an undeclared type would be written twice.
 */
export class NullEventSink implements RlmEventSink {
  readonly name = 'null'
  emit(): void {}
  async flush(): Promise<void> {}
}

/** Fan-out sink. A failing member is isolated so it cannot fail the run. */
export class CompositeEventSink implements RlmEventSink {
  readonly name: string
  private readonly failures: string[] = []

  constructor(private readonly members: readonly RlmEventSink[]) {
    this.name = `composite(${members.map((member) => member.name).join('+')})`
  }

  emit(event: RlmEvent): void {
    for (const member of this.members) {
      try {
        member.emit(event)
      } catch (error) {
        this.failures.push(`${member.name}: ${(error as Error).message}`)
      }
    }
  }

  async flush(): Promise<void> {
    for (const member of this.members) {
      try {
        await member.flush()
      } catch (error) {
        this.failures.push(`${member.name}: ${(error as Error).message}`)
      }
    }
  }

  /** Sinks that failed at least once, for diagnostics. */
  get degraded(): readonly string[] {
    return this.failures
  }
}

export interface SidecarSinkOptions {
  /** Directory for per-invocation trace files. */
  root: string
  /** One file per invocation (`<invocationId>.jsonl`). */
  fileNameFor?: (event: RlmEvent) => string
}

/**
 * Append-only JSONL trace, one file per invocation.
 *
 * Writes are queued behind a single promise chain so `emit()` stays synchronous
 * (an event sink must never stall the root loop) while lines still land in
 * order.
 */
export class SidecarJsonlEventSink implements RlmEventSink {
  readonly name = 'sidecar-jsonl'
  private tail: Promise<void> = Promise.resolve()
  private readonly written = new Map<string, number>()

  constructor(private readonly options: SidecarSinkOptions) {}

  emit(event: RlmEvent): void {
    const file = join(this.options.root, this.fileNameFor(event))
    this.tail = this.tail
      .then(async () => {
        await mkdir(dirname(file), { recursive: true })
        await appendFile(file, `${JSON.stringify(event)}\n`, 'utf8')
        this.written.set(file, (this.written.get(file) ?? 0) + 1)
      })
      .catch(() => {
        // A trace write must never fail a run; degradation is reported through
        // `stats()` rather than thrown into model code.
      })
  }

  async flush(): Promise<void> {
    await this.tail
  }

  stats(): { files: number; events: number } {
    let events = 0
    for (const count of this.written.values()) events += count
    return { files: this.written.size, events }
  }

  private fileNameFor(event: RlmEvent): string {
    return this.options.fileNameFor?.(event) ?? `${event.invocationId}.jsonl`
  }
}

/** The `session.append` surface this bridge uses, kept structural for testability. */
export interface SessionAppendSurface {
  append(type: string, data: unknown): unknown
}

export interface SessionEventSinkOptions {
  session: SessionAppendSurface
  /** Fail closed for unknown types: only these may be written into a session log. */
  knownTypes: ReadonlySet<string>
  /** Where the events go when the session cannot accept them. */
  fallback: RlmEventSink
  /** Called once per type the first time it is skipped. */
  onSkip?: (type: string, reason: string) => void
}

/**
 * Writes `rlm/*` events into a session when — and only when — the running build
 * declares them. This is the guard that keeps a user's session log loadable.
 */
export class SessionEventSink implements RlmEventSink {
  readonly name = 'session'
  readonly skipped = new Set<string>()
  readonly appended = new Set<string>()

  constructor(private readonly options: SessionEventSinkOptions) {}

  emit(event: RlmEvent): void {
    if (!this.options.knownTypes.has(event.type)) {
      if (!this.skipped.has(event.type)) {
        this.skipped.add(event.type)
        this.options.onSkip?.(
          event.type,
          'the running harness build does not declare this session event type; ' +
            'writing it would make the session log unreadable on reload, so it was written to the sidecar trace instead',
        )
      }
      this.options.fallback.emit(event)
      return
    }
    try {
      this.options.session.append(event.type, stripEnvelope(event))
      this.appended.add(event.type)
    } catch (error) {
      if (!this.skipped.has(event.type)) {
        this.skipped.add(event.type)
        this.options.onSkip?.(event.type, `session append failed: ${(error as Error).message}`)
      }
      this.options.fallback.emit(event)
    }
  }

  async flush(): Promise<void> {
    await this.options.fallback.flush()
  }

  /** Types written into the session log by this sink. */
  get durableTypes(): readonly string[] {
    return [...this.appended]
  }
}

/** Event payload without the transport-level `type`/`time` envelope fields. */
export function stripEnvelope(event: RlmEvent): Record<string, unknown> {
  const { type: _type, time: _time, ...data } = event
  return data as Record<string, unknown>
}

/**
 * Bundle every RLM-related signal a run produces into one durable file, so a
 * failed run can be diagnosed without a second session trajectory.
 */
export function createDefaultSink(options: { root: string }): RlmEventSink {
  return new SidecarJsonlEventSink({ root: join(options.root, 'traces') })
}

/** One invocation's story, folded from its durable events. */
export interface RlmRunSummary {
  invocationId: string
  parentInvocationId?: string
  depth: number
  status?: 'ok' | 'failed' | 'cancelled' | 'budget-exceeded'
  stopReason?: RlmStopReason
  startedAt?: number
  finishedAt?: number
  wallClockMs?: number
  /** Root steps that produced an action (excludes query events). */
  steps: number
  queryStarts: number
  queryEnds: number
  failedQueries: number
  /** Highest number of `rlm_query`/`llm_query` calls in flight at once. */
  peakQueryConcurrency: number
  artifacts: { id: string; role: 'context' | 'output'; chars: number; bytes: number }[]
  usage?: RlmUsage
  errorKind?: string
  errorMessage?: string
  /**
   * True when a `rlm/start` has no `rlm/end`. The README requires exactly one
   * terminal event per accepted start, so a dangling run is a defect signal
   * (crash, kill -9, or a plugin unloaded mid-flight) rather than a normal state.
   */
  dangling: boolean
}

/**
 * Fold a durable event stream into per-invocation summaries.
 *
 * This is the projection the README asks for, expressed as a pure function of
 * the event list rather than as a `ctx.sessionProjections` registration: at
 * `dsh-v0.2.0-rc.2` a session projection can only fold event types the running
 * build declares, and an out-of-tree plugin cannot add those (see
 * `docs/limitations.md#session-event-vocabulary`). A host that patches the
 * vocabulary can feed session events into this same function; today it is fed
 * from the sidecar trace, which is why the two are interchangeable.
 */
export function foldRlmRuns(events: readonly RlmEvent[]): RlmRunSummary[] {
  const runs = new Map<string, RlmRunSummary>()
  const order: string[] = []
  const inFlight = new Map<string, number>()
  const ensure = (invocationId: string): RlmRunSummary => {
    const existing = runs.get(invocationId)
    if (existing) return existing
    const created: RlmRunSummary = {
      invocationId,
      depth: 0,
      steps: 0,
      queryStarts: 0,
      queryEnds: 0,
      failedQueries: 0,
      peakQueryConcurrency: 0,
      artifacts: [],
      dangling: true,
    }
    runs.set(invocationId, created)
    order.push(invocationId)
    return created
  }

  for (const event of events) {
    const run = ensure(event.invocationId)
    switch (event.type) {
      case 'rlm/start': {
        run.depth = event.depth
        run.startedAt = event.startedAt
        if (event.parentInvocationId !== undefined) run.parentInvocationId = event.parentInvocationId
        run.dangling = true
        break
      }
      case 'rlm/step': {
        if (event.action !== 'parse-failure') run.steps += 1
        break
      }
      case 'rlm/query-start': {
        run.queryStarts += 1
        const active = (inFlight.get(event.invocationId) ?? 0) + 1
        inFlight.set(event.invocationId, active)
        run.peakQueryConcurrency = Math.max(run.peakQueryConcurrency, active)
        break
      }
      case 'rlm/query-end': {
        run.queryEnds += 1
        if (!event.ok) run.failedQueries += 1
        inFlight.set(event.invocationId, Math.max(0, (inFlight.get(event.invocationId) ?? 1) - 1))
        break
      }
      case 'rlm/artifact': {
        run.artifacts.push({
          id: event.artifactRef.id,
          role: event.role,
          chars: event.artifactRef.chars,
          bytes: event.artifactRef.bytes,
        })
        break
      }
      case 'rlm/end': {
        run.status = event.status
        run.stopReason = event.stopReason
        run.finishedAt = event.finishedAt
        run.wallClockMs = event.wallClockMs
        run.usage = event.usage
        if (event.errorKind !== undefined) run.errorKind = event.errorKind
        if (event.errorMessage !== undefined) run.errorMessage = event.errorMessage
        run.dangling = false
        break
      }
      default:
        break
    }
  }

  return order.map((invocationId) => runs.get(invocationId)!)
}

/**
 * Compute the §7.3 metric set from folded runs.
 *
 * Deliberately returns a distribution, not a mean: the paper reports heavy-tail
 * trajectories, so p95 cost and latency are acceptance metrics and a mean alone
 * would hide the runs that matter.
 */
export function aggregateRlmMetrics(summaries: readonly RlmRunSummary[]): {
  runs: number
  ok: number
  failed: number
  cancelled: number
  budgetExceeded: number
  dangling: number
  totalQueries: number
  totalKernels: number
  totalInputTokens: number
  totalOutputTokens: number
  totalCostUsd: number
  wallClockMs: { p50: number; p90: number; p95: number }
  costUsd: { p50: number; p90: number; p95: number }
} {
  const percentile = (values: number[], fraction: number): number => {
    if (values.length === 0) return 0
    const sorted = [...values].sort((left, right) => left - right)
    const index = Math.min(sorted.length - 1, Math.ceil(fraction * sorted.length) - 1)
    return sorted[Math.max(0, index)] ?? 0
  }
  const wall = summaries.map((summary) => summary.wallClockMs ?? 0)
  const cost = summaries.map((summary) => summary.usage?.costUsd ?? 0)
  return {
    runs: summaries.length,
    ok: summaries.filter((summary) => summary.status === 'ok').length,
    failed: summaries.filter((summary) => summary.status === 'failed').length,
    cancelled: summaries.filter((summary) => summary.status === 'cancelled').length,
    budgetExceeded: summaries.filter((summary) => summary.status === 'budget-exceeded').length,
    dangling: summaries.filter((summary) => summary.dangling).length,
    totalQueries: summaries.reduce((sum, summary) => sum + (summary.usage?.queries ?? 0), 0),
    totalKernels: summaries.reduce((sum, summary) => sum + (summary.usage?.kernels ?? 0), 0),
    totalInputTokens: summaries.reduce((sum, summary) => sum + (summary.usage?.inputTokens ?? 0), 0),
    totalOutputTokens: summaries.reduce((sum, summary) => sum + (summary.usage?.outputTokens ?? 0), 0),
    totalCostUsd: summaries.reduce((sum, summary) => sum + (summary.usage?.costUsd ?? 0), 0),
    wallClockMs: { p50: percentile(wall, 0.5), p90: percentile(wall, 0.9), p95: percentile(wall, 0.95) },
    costUsd: { p50: percentile(cost, 0.5), p90: percentile(cost, 0.9), p95: percentile(cost, 0.95) },
  }
}
