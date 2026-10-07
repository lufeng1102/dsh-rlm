/**
 * Budget conservation tests.
 *
 * These are the property tests the README asks for in §7.5: a child account can
 * never widen what its parent has left, usage propagates exactly once, and
 * reservations are returned on settlement.
 */

import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { RlmBudgetAccount, RlmBudgetError, narrowLimits } from '../lib/budget.js'

const LIMITS = {
  maxRootIterations: 10,
  maxWallClockMs: 60_000,
  maxTotalQueries: 8,
  maxConcurrentQueries: 4,
  maxTotalKernels: 2,
  maxConcurrentKernels: 1,
  maxInputTokens: 1_000,
  maxOutputTokens: 500,
  maxContextReadBytes: 10_000,
  maxQueryPromptChars: 2_000,
  maxInlineOutputBytes: 1_000,
  maxArtifactBytes: 10_000,
  maxCostUsd: 10,
}

function account(now = () => 0) {
  return new RlmBudgetAccount({ ...LIMITS }, undefined, { scope: 'invocation', now })
}

describe('hierarchical budget account', () => {
  it('refuses a reservation larger than what remains', () => {
    const root = account()
    assert.throws(() => root.reserve({ maxTotalQueries: 9 }), RlmBudgetError)
  })

  it('returns unused capacity when a reservation settles', () => {
    const root = account()
    const reservation = root.reserve({ maxTotalQueries: 4 })
    assert.equal(root.remainingLimits().maxTotalQueries, 4)
    reservation.settle({ maxTotalQueries: 1 })
    // One query was actually spent, so capacity exceeds the pre-reservation level
    // by exactly the unused part of the claim.
    assert.equal(root.remainingLimits().maxTotalQueries, 7)
    assert.equal(root.usage().queries, 1)
  })

  it('propagates child usage to the parent exactly once', () => {
    const root = account()
    const child = root.child({}, 'child')
    child.charge({ maxTotalQueries: 2, maxInputTokens: 100 })
    assert.equal(child.usage().queries, 2)
    assert.equal(root.usage().queries, 2)
    assert.equal(root.usage().inputTokens, 100)
    assert.equal(root.remainingLimits().maxTotalQueries, LIMITS.maxTotalQueries - 2)
  })

  it('never lets a child widen its parent remaining capacity', () => {
    const root = account()
    root.charge({ maxTotalQueries: 5 })
    const child = root.child({ maxTotalQueries: 100 }, 'child')
    assert.equal(child.limits.maxTotalQueries, 3)
  })

  it('clamps a grandchild to the shared pool, not to a copy of the ceiling', () => {
    const root = account()
    const child = root.child({}, 'child')
    const grandchild = child.child({}, 'grandchild')
    grandchild.charge({ maxTotalQueries: 3 })
    child.charge({ maxTotalQueries: 1 })
    assert.equal(root.usage().queries, 4)
    assert.equal(grandchild.remainingLimits().maxTotalQueries, LIMITS.maxTotalQueries - 4)
  })

  it('counts a reservation against concurrency until it settles', () => {
    const root = account()
    const first = root.reserve({ maxConcurrentQueries: 1 })
    assert.equal(root.remainingLimits().maxConcurrentQueries, LIMITS.maxConcurrentQueries - 1)
    const second = root.reserve({ maxConcurrentQueries: 1 })
    assert.equal(root.remainingLimits().maxConcurrentQueries, LIMITS.maxConcurrentQueries - 2)
    first.settle()
    second.settle()
    assert.equal(root.remainingLimits().maxConcurrentQueries, LIMITS.maxConcurrentQueries)
  })

  it('reports exhaustion and maps it onto a stop reason', () => {
    const root = account()
    root.charge({ maxRootIterations: LIMITS.maxRootIterations })
    assert.deepEqual(root.exhausted(), ['maxRootIterations'])
    assert.equal(root.stopReasonForExhaustion(), 'max-root-iterations')
  })

  it('treats an elapsed wall clock as exhaustion', () => {
    let clock = 0
    const root = account(() => clock)
    clock = LIMITS.maxWallClockMs + 1
    assert.equal(root.stopReasonForExhaustion(), 'wall-clock')
  })

  it('ignores a second settle of the same reservation', () => {
    const root = account()
    const reservation = root.reserve({ maxTotalQueries: 2 })
    reservation.settle({ maxTotalQueries: 1 })
    reservation.settle({ maxTotalQueries: 1 })
    assert.equal(root.usage().queries, 1)
    assert.ok(reservation.settled)
  })

  it('keeps the tighter value on every dimension when narrowing', () => {
    const narrowed = narrowLimits({ ...LIMITS }, { ...LIMITS, maxTotalQueries: 1, maxRootIterations: 99 })
    assert.equal(narrowed.maxTotalQueries, 1)
    assert.equal(narrowed.maxRootIterations, 10)
  })

  it('never reports negative remaining capacity', () => {
    const root = account()
    root.charge({ maxInputTokens: LIMITS.maxInputTokens * 2 })
    assert.equal(root.remainingLimits().maxInputTokens, 0)
  })
})
