/**
 * Event-sink tests, including the session-vocabulary guard.
 *
 * The guard is the reason this module exists: writing an undeclared `rlm/*` event
 * into a session log produces a log this build can no longer read. These tests
 * pin the safe behaviour, and pin that the fallback keeps the data durable.
 */

import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { readFile } from 'node:fs/promises'
import {
  CompositeEventSink,
  RecordingEventSink,
  SessionEventSink,
  SidecarJsonlEventSink,
  aggregateRlmMetrics,
  foldRlmRuns,
} from '../lib/events.js'
import { cleanup, makeTempDir } from './helpers.mjs'

const START_EVENT = {
  type: 'rlm/start',
  time: 1_700_000_000_000,
  invocationId: 'inv-1',
  attemptId: 'attempt-1',
  depth: 1,
  maxDepth: 3,
  contextRef: {
    id: 'ctx',
    sha256: 'sha',
    kind: 'text',
    bytes: 10,
    chars: 10,
    estimatedTokens: {},
    encoding: 'utf-8',
    createdAt: 0,
  },
  budgets: {},
  identity: {
    rlmVersion: '0.1.0',
    promptProfileId: 'paper-gpt5',
    promptProfileVariant: 'paper-gpt5',
    kernelProvider: 'python-subprocess',
    kernelIsolation: 'process',
    contextStoreProvider: 'local-content-addressed',
    toolSchemaVersion: '1',
  },
  rootRoute: { provider: 'p', model: 'm' },
  leafRoute: { provider: 'p', model: 'm' },
  startedAt: 1_700_000_000_000,
}

describe('recording event sink', () => {
  it('keeps events in order and filters by type', () => {
    const sink = new RecordingEventSink()
    sink.emit(START_EVENT)
    sink.emit({ ...START_EVENT, type: 'rlm/end', stopReason: 'ok', status: 'ok', steps: 1, usage: {}, finishedAt: 0, wallClockMs: 5 })
    assert.equal(sink.events.length, 2)
    assert.equal(sink.ofType('rlm/start').length, 1)
    assert.equal(sink.ofType('rlm/end')[0].stopReason, 'ok')
  })

  it('caps its ring so a long run cannot grow without bound', () => {
    const sink = new RecordingEventSink(3)
    for (let index = 0; index < 10; index += 1) sink.emit({ ...START_EVENT, invocationId: `inv-${index}` })
    assert.equal(sink.events.length, 3)
    assert.equal(sink.events[2].invocationId, 'inv-9')
  })
})

describe('sidecar jsonl sink', () => {
  it('writes one append-only file per invocation, in order', async () => {
    const root = await makeTempDir('rlm-events-')
    try {
      const sink = new SidecarJsonlEventSink({ root })
      sink.emit(START_EVENT)
      sink.emit({ ...START_EVENT, type: 'rlm/end', stopReason: 'ok', status: 'ok', steps: 1, usage: {}, finishedAt: 0, wallClockMs: 5 })
      sink.emit({ ...START_EVENT, invocationId: 'inv-2' })
      await sink.flush()
      const lines = (await readFile(`${root}/inv-1.jsonl`, 'utf8')).trim().split('\n')
      assert.equal(lines.length, 2)
      assert.equal(JSON.parse(lines[0]).type, 'rlm/start')
      assert.equal(JSON.parse(lines[1]).type, 'rlm/end')
      assert.deepEqual(sink.stats(), { files: 2, events: 3 })
    } finally {
      await cleanup(root)
    }
  })

  it('never throws into a run when the trace cannot be written', async () => {
    const sink = new SidecarJsonlEventSink({ root: '/proc/definitely-not-writable' })
    // Emitting must stay synchronous and non-throwing.
    sink.emit(START_EVENT)
    await sink.flush()
  })
})

describe('session event sink', () => {
  it('appends only event types the running build declares', () => {
    const appended = []
    const fallback = new RecordingEventSink()
    const skipped = []
    const sink = new SessionEventSink({
      session: { append: (type, data) => appended.push({ type, data }) },
      knownTypes: new Set(['rlm/start']),
      fallback,
      onSkip: (type, reason) => skipped.push({ type, reason }),
    })
    sink.emit(START_EVENT)
    sink.emit({ ...START_EVENT, type: 'rlm/step', step: 1, action: 'eval', elapsedMs: 1, rootHistoryChars: 10, usage: {} })
    assert.equal(appended.length, 1)
    assert.equal(appended[0].type, 'rlm/start')
    // The envelope fields are stripped: the session envelope supplies them.
    assert.equal(appended[0].data.type, undefined)
    assert.equal(appended[0].data.invocationId, 'inv-1')
    // The undeclared type went to the durable fallback, not into the log.
    assert.equal(fallback.events.length, 1)
    assert.equal(fallback.events[0].type, 'rlm/step')
    assert.equal(skipped.length, 1)
    assert.match(skipped[0].reason, /unreadable on reload/)
    assert.deepEqual(sink.durableTypes, ['rlm/start'])
  })

  it('falls back when a session append throws', () => {
    const fallback = new RecordingEventSink()
    const sink = new SessionEventSink({
      session: {
        append() {
          throw new Error('session is read-only')
        },
      },
      knownTypes: new Set(['rlm/start']),
      fallback,
      onSkip: () => {},
    })
    sink.emit(START_EVENT)
    assert.equal(fallback.events.length, 1)
  })

  it('reports a skip only once per type', () => {
    const skipped = []
    const sink = new SessionEventSink({
      session: { append: () => undefined },
      knownTypes: new Set(),
      fallback: new RecordingEventSink(),
      onSkip: (type) => skipped.push(type),
    })
    sink.emit(START_EVENT)
    sink.emit(START_EVENT)
    assert.deepEqual(skipped, ['rlm/start'])
  })
})

describe('composite event sink', () => {
  it('isolates a failing member so a run is not failed by telemetry', () => {
    const healthy = new RecordingEventSink()
    const sink = new CompositeEventSink([
      {
        name: 'broken',
        emit() {
          throw new Error('sink exploded')
        },
        async flush() {},
      },
      healthy,
    ])
    sink.emit(START_EVENT)
    assert.equal(healthy.events.length, 1)
    assert.equal(sink.degraded.length, 1)
  })
})

describe('trajectory fold and metrics', () => {
  const usage = {
    rootIterations: 3,
    queries: 4,
    kernels: 2,
    inputTokens: 1_000,
    outputTokens: 500,
    contextReadBytes: 2_048,
    costUsd: 0.25,
    wallClockMs: 4_000,
    maxObservedConcurrency: 2,
  }
  const parentStart = { ...START_EVENT, type: 'rlm/start', startedAt: 100 }
  const parentEnd = {
    ...START_EVENT,
    type: 'rlm/end',
    status: 'ok',
    stopReason: 'final-text',
    steps: 3,
    usage,
    finishedAt: 4_100,
    wallClockMs: 4_000,
  }
  const childStart = { ...START_EVENT, type: 'rlm/start', invocationId: 'inv-child', parentInvocationId: 'inv-1', depth: 1, startedAt: 200 }
  const childEnd = { ...parentEnd, invocationId: 'inv-child', wallClockMs: 500, finishedAt: 700 }

  it('folds one run with steps, queries, artifacts and terminal state', () => {
    const summary = foldRlmRuns([
      parentStart,
      { ...START_EVENT, type: 'rlm/artifact', artifactRef: { id: 'art_1', sha256: 'h', bytes: 10, chars: 10, contentType: 'text/plain', preview: '', truncated: false }, role: 'context' },
      { ...START_EVENT, type: 'rlm/query-start', callId: 1, kind: 'llm_query', depth: 1, promptChars: 5, route: { provider: 'p', model: 'm' }, startedAt: 300 },
      { ...START_EVENT, type: 'rlm/query-start', callId: 2, kind: 'llm_query', depth: 1, promptChars: 5, route: { provider: 'p', model: 'm' }, startedAt: 301 },
      { ...START_EVENT, type: 'rlm/query-end', callId: 1, kind: 'llm_query', ok: true, elapsedMs: 2, inputTokens: 1, outputTokens: 1, resultChars: 3 },
      { ...START_EVENT, type: 'rlm/query-end', callId: 2, kind: 'llm_query', ok: false, errorKind: 'budget', elapsedMs: 1, inputTokens: 0, outputTokens: 0, resultChars: 0 },
      { ...START_EVENT, type: 'rlm/step', step: 1, action: 'eval', elapsedMs: 5, rootHistoryChars: 900, usage },
      parentEnd,
    ])
    assert.equal(summary.length, 1)
    const run = summary[0]
    assert.equal(run.status, 'ok')
    assert.equal(run.stopReason, 'final-text')
    assert.equal(run.steps, 1)
    assert.equal(run.queryStarts, 2)
    assert.equal(run.failedQueries, 1)
    assert.equal(run.peakQueryConcurrency, 2)
    assert.equal(run.artifacts.length, 1)
    assert.equal(run.dangling, false)
    assert.equal(run.usage.costUsd, 0.25)
  })

  it('flags a start with no end as dangling, which is a defect signal', () => {
    const summary = foldRlmRuns([parentStart, childStart, childEnd])
    assert.equal(summary.length, 2)
    assert.equal(summary[0].dangling, true)
    assert.equal(summary[0].status, undefined)
    assert.equal(summary[1].dangling, false)
    assert.equal(summary[1].parentInvocationId, 'inv-1')
  })

  it('does not count a parse-failure turn as a root step', () => {
    const summary = foldRlmRuns([
      parentStart,
      { ...START_EVENT, type: 'rlm/step', step: 1, action: 'parse-failure', elapsedMs: 0, rootHistoryChars: 10, usage: {} },
      { ...START_EVENT, type: 'rlm/step', step: 2, action: 'eval', elapsedMs: 1, rootHistoryChars: 20, usage: {} },
      parentEnd,
    ])
    assert.equal(summary[0].steps, 1)
  })

  it('aggregates the §7.3 metric set as a distribution, not a mean', () => {
    const metrics = aggregateRlmMetrics(
      foldRlmRuns([
        parentStart,
        parentEnd,
        childStart,
        childEnd,
        { ...parentStart, invocationId: 'inv-3' },
        { ...parentEnd, invocationId: 'inv-3', status: 'budget-exceeded', stopReason: 'max-root-iterations', wallClockMs: 90_000 },
        { ...parentStart, invocationId: 'inv-4' },
      ]),
    )
    assert.equal(metrics.runs, 4)
    assert.equal(metrics.ok, 2)
    assert.equal(metrics.budgetExceeded, 1)
    assert.equal(metrics.dangling, 1)
    // Three finished runs, each with four queries and two kernels; the dangling
    // fourth run contributes nothing to usage, which is the point of flagging it.
    assert.equal(metrics.totalQueries, 12)
    assert.equal(metrics.totalKernels, 6)
    assert.ok(metrics.wallClockMs.p95 >= metrics.wallClockMs.p50)
    assert.ok(metrics.costUsd.p95 >= metrics.costUsd.p50)
  })
})
