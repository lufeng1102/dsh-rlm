/**
 * Bounded admission control for `llm_query` / `rlm_query` fan-out (README §4.5).
 *
 * Model-written code can request an unbounded number of sub-calls from a single
 * loop. The scheduler is the host-side answer: a fixed concurrency ceiling, a
 * FIFO queue so siblings cannot starve each other, a bounded wait in the queue,
 * and an independent cancellation path per call that is linked to the parent
 * kernel's signal.
 *
 * Rejections are typed so program code receives an honest error rather than a
 * silently dropped call.
 *
 * @module dsh-rlm/scheduler
 */

/** The caller's own abort signal fired before the work could start or finish. */
export class RlmAbortError extends Error {
  override readonly name = 'RlmAbortError'
  readonly kind = 'cancelled'

  constructor(message = 'cancelled') {
    super(message)
  }
}

/** The call waited longer than the deployment allows for a free slot. */
export class RlmQueueTimeoutError extends Error {
  override readonly name = 'RlmQueueTimeoutError'
  readonly kind = 'queue-timeout'

  constructor(message = 'query queue wait exceeded its budget') {
    super(message)
  }
}

export interface RlmSchedulerOptions {
  /** Hard ceiling on simultaneously admitted calls. */
  maxConcurrency: number
  /** Default bounded wait for a free slot. */
  queueTimeoutMs: number
  /** Called whenever a call is rejected before admission. */
  onReject?: (reason: 'queue-timeout' | 'cancelled', label: string) => void
  /** Inject the clock; tests use a fake one. */
  now?: () => number
}

export interface RlmRunOptions {
  signal?: AbortSignal
  /** Diagnostics label; never model-visible. */
  label?: string
  queueTimeoutMs?: number
}

interface Waiter {
  resolve: (release: () => void) => void
  reject: (error: Error) => void
  cleanup: () => void
}

/**
 * A FIFO, cancellation-aware concurrency limiter.
 *
 * `active` counts admitted calls, so the scheduler is also the source of truth
 * for observed concurrency in metrics.
 */
export class RlmScheduler {
  readonly maxConcurrency: number
  readonly queueTimeoutMs: number

  private readonly onReject: (reason: 'queue-timeout' | 'cancelled', label: string) => void
  private readonly now: () => number
  private readonly waiters: Waiter[] = []
  private activeCount = 0
  private peakCount = 0
  private admittedCount = 0
  private rejectedCount = 0

  constructor(options: RlmSchedulerOptions) {
    this.maxConcurrency = Math.max(1, Math.floor(options.maxConcurrency))
    this.queueTimeoutMs = Math.max(0, Math.floor(options.queueTimeoutMs))
    this.onReject = options.onReject ?? (() => {})
    this.now = options.now ?? Date.now
  }

  /** Calls currently admitted. */
  get active(): number {
    return this.activeCount
  }

  /** Calls waiting for a slot. */
  get queued(): number {
    return this.waiters.length
  }

  /** Highest number of simultaneously admitted calls. */
  get peak(): number {
    return this.peakCount
  }

  get admitted(): number {
    return this.admittedCount
  }

  get rejected(): number {
    return this.rejectedCount
  }

  /**
   * Run `fn` under the concurrency ceiling.
   *
   * @throws {RlmAbortError} when the caller's signal fires before or during the call.
   * @throws {RlmQueueTimeoutError} when no slot becomes free in time.
   */
  async run<T>(fn: (signal: AbortSignal) => Promise<T>, options: RlmRunOptions = {}): Promise<T> {
    const label = options.label ?? 'query'
    if (options.signal?.aborted) {
      this.rejectedCount += 1
      this.onReject('cancelled', label)
      throw new RlmAbortError()
    }
    await this.acquire(label, options)
    const controller = new AbortController()
    const linkAbort = () => controller.abort(options.signal?.reason)
    options.signal?.addEventListener('abort', linkAbort, { once: true })
    try {
      if (options.signal?.aborted) throw new RlmAbortError()
      return await fn(controller.signal)
    } finally {
      options.signal?.removeEventListener('abort', linkAbort)
      this.release()
    }
  }

  private acquire(label: string, options: RlmRunOptions): Promise<void> {
    if (this.activeCount < this.maxConcurrency) {
      this.activeCount += 1
      this.admittedCount += 1
      this.peakCount = Math.max(this.peakCount, this.activeCount)
      return Promise.resolve()
    }
    return new Promise<void>((resolve, reject) => {
      const timeoutMs = options.queueTimeoutMs ?? this.queueTimeoutMs
      const timer =
        timeoutMs > 0
          ? setTimeout(() => {
              this.removeWaiter(waiter)
              this.rejectedCount += 1
              this.onReject('queue-timeout', label)
              reject(new RlmQueueTimeoutError(`no slot for ${label} within ${timeoutMs} ms`))
            }, timeoutMs)
          : undefined
      const onAbort = () => {
        this.removeWaiter(waiter)
        this.rejectedCount += 1
        this.onReject('cancelled', label)
        reject(new RlmAbortError())
      }
      const waiter: Waiter = {
        resolve: (release) => {
          if (timer) clearTimeout(timer)
          options.signal?.removeEventListener('abort', onAbort)
          this.activeCount += 1
          this.admittedCount += 1
          this.peakCount = Math.max(this.peakCount, this.activeCount)
          release()
          resolve()
        },
        reject,
        cleanup: () => {
          if (timer) clearTimeout(timer)
          options.signal?.removeEventListener('abort', onAbort)
        },
      }
      options.signal?.addEventListener('abort', onAbort, { once: true })
      this.waiters.push(waiter)
    })
  }

  private release(): void {
    this.activeCount = Math.max(0, this.activeCount - 1)
    const next = this.waiters.shift()
    if (!next) return
    next.cleanup()
    next.resolve(() => {})
  }

  private removeWaiter(waiter: Waiter): void {
    const index = this.waiters.indexOf(waiter)
    if (index >= 0) this.waiters.splice(index, 1)
  }

  /** Drop every queued call; admitted calls are the caller's to cancel. */
  drainQueued(reason = 'scheduler stopped'): void {
    const waiters = this.waiters.splice(0, this.waiters.length)
    for (const waiter of waiters) {
      waiter.cleanup()
      waiter.reject(new RlmAbortError(reason))
    }
  }

  /** Metrics for durable events. */
  snapshot(): { maxConcurrency: number; active: number; queued: number; peak: number; admitted: number; rejected: number; now: number } {
    return {
      maxConcurrency: this.maxConcurrency,
      active: this.activeCount,
      queued: this.waiters.length,
      peak: this.peakCount,
      admitted: this.admittedCount,
      rejected: this.rejectedCount,
      now: this.now(),
    }
  }
}

/** Simple bounded-parallel map used by sibling fan-out helpers. */
export async function mapBounded<T, R>(
  items: readonly T[],
  limit: number,
  mapper: (item: T, index: number) => Promise<R>,
  signal?: AbortSignal,
): Promise<R[]> {
  const results: R[] = new Array(items.length)
  let cursor = 0
  const workers = Math.max(1, Math.min(Math.floor(limit), items.length))
  await Promise.all(
    Array.from({ length: workers }, async () => {
      while (true) {
        if (signal?.aborted) throw new RlmAbortError()
        const index = cursor
        cursor += 1
        if (index >= items.length) return
        results[index] = await mapper(items[index]!, index)
      }
    }),
  )
  return results
}
