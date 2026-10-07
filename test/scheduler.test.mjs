/**
 * Scheduler tests: the host-side answer to "model code asked for 10,000 calls".
 */

import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { RlmAbortError, RlmQueueTimeoutError, RlmScheduler, mapBounded } from '../lib/scheduler.js'

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

describe('bounded query scheduler', () => {
  it('never exceeds the concurrency ceiling', async () => {
    const scheduler = new RlmScheduler({ maxConcurrency: 2, queueTimeoutMs: 1000 })
    let active = 0
    let peak = 0
    const jobs = Array.from({ length: 8 }, () =>
      scheduler.run(async () => {
        active += 1
        peak = Math.max(peak, active)
        await sleep(5)
        active -= 1
      }),
    )
    await Promise.all(jobs)
    assert.equal(peak, 2)
    assert.equal(scheduler.peak, 2)
    assert.equal(scheduler.admitted, 8)
    assert.equal(scheduler.queued, 0)
    assert.equal(scheduler.active, 0)
  })

  it('admits waiting calls in FIFO order', async () => {
    const scheduler = new RlmScheduler({ maxConcurrency: 1, queueTimeoutMs: 1000 })
    const order = []
    const first = scheduler.run(async () => {
      order.push('a-start')
      await sleep(10)
      order.push('a-end')
    })
    await sleep(1)
    const second = scheduler.run(async () => {
      order.push('b')
    })
    const third = scheduler.run(async () => {
      order.push('c')
    })
    await Promise.all([first, second, third])
    assert.deepEqual(order, ['a-start', 'a-end', 'b', 'c'])
  })

  it('rejects a queued call that waits too long, instead of hanging', async () => {
    const scheduler = new RlmScheduler({ maxConcurrency: 1, queueTimeoutMs: 10 })
    const held = scheduler.run(() => sleep(50))
    await assert.rejects(scheduler.run(async () => 'never'), RlmQueueTimeoutError)
    await held
    assert.equal(scheduler.rejected, 1)
  })

  it('cancels a queued call when its parent signal aborts', async () => {
    const scheduler = new RlmScheduler({ maxConcurrency: 1, queueTimeoutMs: 5000 })
    const held = scheduler.run(() => sleep(30))
    const controller = new AbortController()
    const queued = scheduler.run(async () => 'never', { signal: controller.signal })
    controller.abort('parent aborted')
    await assert.rejects(queued, RlmAbortError)
    await held
    assert.equal(scheduler.queued, 0)
  })

  it('refuses a call whose signal is already aborted, without admitting it', async () => {
    const scheduler = new RlmScheduler({ maxConcurrency: 2, queueTimeoutMs: 100 })
    const controller = new AbortController()
    controller.abort()
    await assert.rejects(scheduler.run(async () => 'never', { signal: controller.signal }), RlmAbortError)
    assert.equal(scheduler.admitted, 0)
    assert.equal(scheduler.rejected, 1)
  })

  it('gives the admitted body its own abort signal, linked to the caller', async () => {
    const scheduler = new RlmScheduler({ maxConcurrency: 1, queueTimeoutMs: 100 })
    const controller = new AbortController()
    const observed = scheduler.run(
      async (signal) => {
        assert.equal(signal.aborted, false)
        controller.abort('stop')
        await sleep(1)
        return signal.aborted
      },
      { signal: controller.signal },
    )
    assert.equal(await observed, true)
  })

  it('drops queued work on drain', async () => {
    const scheduler = new RlmScheduler({ maxConcurrency: 1, queueTimeoutMs: 5000 })
    const held = scheduler.run(() => sleep(20))
    const queued = scheduler.run(async () => 'never')
    scheduler.drainQueued('shutting down')
    await assert.rejects(queued, RlmAbortError)
    await held
    assert.equal(scheduler.queued, 0)
  })

  it('maps a list with bounded parallelism and preserves order', async () => {
    const results = await mapBounded([1, 2, 3, 4, 5], 2, async (value) => {
      await sleep(value % 2 === 0 ? 3 : 1)
      return value * 2
    })
    assert.deepEqual(results, [2, 4, 6, 8, 10])
  })
})
