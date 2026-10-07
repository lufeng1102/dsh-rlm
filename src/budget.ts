/**
 * Hierarchical budget accounting (README §4.7).
 *
 * There is exactly one root account per invocation tree. A recursive child never
 * receives a fresh copy of the deployment limits: it can only narrow what its
 * parent still has. That is the whole point of the class — depth must not
 * multiply the deployment's worst-case cost, and a child must never be able to
 * widen its own authority by asking for more.
 *
 * Two mechanics do the work:
 *
 * 1. **Propagation.** `charge()` adds to this account *and to every ancestor*,
 *    so an ancestor's `usage()` is its whole subtree. No summation happens at
 *    read time, so nothing is counted twice.
 * 2. **Reserve, then settle.** A caller reserves its worst case before starting
 *    work and reconciles after; unused capacity returns to the pool. During the
 *    flight of one child, the parent therefore counts both the reservation and
 *    that child's actuals — deliberately conservative, and resolved at settle.
 *
 * @module dsh-rlm/budget
 */

import type { RlmBudgetLimits, RlmStopReason, RlmUsage } from './types.ts'

/** Raised when an operation cannot be admitted under the current ceilings. */
export class RlmBudgetError extends Error {
  override readonly name = 'RlmBudgetError'
  readonly dimension: BudgetDimension
  readonly requested: number
  readonly available: number

  constructor(dimension: BudgetDimension, requested: number, available: number, scope: string) {
    super(`budget exceeded in ${scope}: ${dimension} requested ${requested}, available ${available}`)
    this.dimension = dimension
    this.requested = requested
    this.available = available
  }
}

export type BudgetDimension = keyof RlmBudgetLimits

/** How a usage counter maps onto the limit dimension that bounds it. */
const COUNTER_TO_LIMIT = {
  rootIterations: 'maxRootIterations',
  queries: 'maxTotalQueries',
  kernels: 'maxTotalKernels',
  inputTokens: 'maxInputTokens',
  outputTokens: 'maxOutputTokens',
  contextReadBytes: 'maxContextReadBytes',
  costUsd: 'maxCostUsd',
  wallClockMs: 'maxWallClockMs',
} as const satisfies Record<string, BudgetDimension>

type UsageCounter = keyof typeof COUNTER_TO_LIMIT

export type BudgetCharge = Partial<Record<BudgetDimension, number>>

/** A worst-case claim on capacity, settled exactly once. */
export interface RlmBudgetReservation {
  readonly id: string
  /** Record actual consumption and return the unused reservation. */
  settle(actual?: BudgetCharge): void
  /** True once {@link settle} has run; a second settle is a no-op. */
  readonly settled: boolean
  /** Claimed amounts, for diagnostics. */
  readonly claim: Readonly<BudgetCharge>
}

export interface RlmBudgetAccountOptions {
  /** Label used in error messages (`invocation`, `child-rlm`, ...). */
  scope: string
  /** Injected clock; tests use a fake one. */
  now?: () => number
}

function emptyUsage(): RlmUsage {
  return {
    rootIterations: 0,
    queries: 0,
    kernels: 0,
    inputTokens: 0,
    outputTokens: 0,
    contextReadBytes: 0,
    costUsd: 0,
    wallClockMs: 0,
    maxObservedConcurrency: 0,
  }
}

function emptyCharge(): Record<BudgetDimension, number> {
  return {
    maxRootIterations: 0,
    maxWallClockMs: 0,
    maxTotalQueries: 0,
    maxConcurrentQueries: 0,
    maxTotalKernels: 0,
    maxConcurrentKernels: 0,
    maxInputTokens: 0,
    maxOutputTokens: 0,
    maxContextReadBytes: 0,
    maxQueryPromptChars: 0,
    maxInlineOutputBytes: 0,
    maxArtifactBytes: 0,
    maxCostUsd: 0,
  }
}

/**
 * One account in the hierarchy. Limits flow strictly downward; usage flows
 * strictly upward.
 */
export class RlmBudgetAccount {
  readonly scope: string
  readonly limits: RlmBudgetLimits
  readonly parent?: RlmBudgetAccount
  readonly createdAt: number

  private readonly now: () => number
  private readonly own = emptyUsage()
  private readonly reserved = emptyCharge()
  private readonly children: RlmBudgetAccount[] = []

  constructor(limits: RlmBudgetLimits, parent?: RlmBudgetAccount, options?: RlmBudgetAccountOptions) {
    this.parent = parent
    this.scope = options?.scope ?? 'invocation'
    this.now = options?.now ?? Date.now
    this.createdAt = this.now()
    // A child is clamped to what the parent still has, at construction time.
    this.limits = parent ? narrowLimits(limits, parent.remainingLimits()) : { ...limits }
    parent?.children.push(this)
  }

  /** Wall-clock milliseconds since this account was created. */
  elapsedMs(): number {
    return Math.max(0, this.now() - this.createdAt)
  }

  /** Settled usage of this account's whole subtree (propagated, never summed). */
  usage(): RlmUsage {
    return { ...this.own }
  }

  /** Capacity still available after subtree usage and outstanding reservations. */
  remainingLimits(): RlmBudgetLimits {
    const limits: RlmBudgetLimits = { ...this.limits }
    for (const [counter, dimension] of Object.entries(COUNTER_TO_LIMIT) as [UsageCounter, BudgetDimension][]) {
      limits[dimension] = Math.max(0, limits[dimension] - this.own[counter])
    }
    for (const dimension of Object.keys(limits) as BudgetDimension[]) {
      limits[dimension] = Math.max(0, limits[dimension] - this.reserved[dimension])
    }
    // An elapsed wall clock is charged as usage; a caller that never reports
    // time still gets a truthful deadline check.
    limits.maxWallClockMs = Math.max(0, Math.min(limits.maxWallClockMs, this.limits.maxWallClockMs - this.elapsedMs()))
    // Capacity is shared with the whole tree, not merely copied at construction:
    // if a sibling branch has drained the pool, this account has nothing left
    // either. Ancestor usage already includes this subtree, so intersecting is a
    // cap rather than a second subtraction.
    const parentRemaining = this.parent?.remainingLimits()
    return parentRemaining ? narrowLimits(limits, parentRemaining) : limits
  }

  /**
   * Claim worst-case capacity. Throws {@link RlmBudgetError} when the claim
   * cannot be satisfied — the only correct moment to refuse work, because after
   * this point the model is already paying for a program that cannot run.
   */
  reserve(claim: BudgetCharge): RlmBudgetReservation {
    const remaining = this.remainingLimits()
    for (const [dimension, amount] of Object.entries(claim) as [BudgetDimension, number][]) {
      if (!amount) continue
      if (amount > remaining[dimension]) {
        throw new RlmBudgetError(dimension, amount, remaining[dimension], this.scope)
      }
    }
    for (const [dimension, amount] of Object.entries(claim) as [BudgetDimension, number][]) {
      if (!amount) continue
      this.reserved[dimension] = Math.max(0, this.reserved[dimension] + amount)
    }
    let settled = false
    const id = `rsv_${this.scope}_${this.children.length}_${this.reserved.maxTotalQueries}_${Math.floor(this.now())}`
    return {
      id,
      claim: Object.freeze({ ...claim }),
      get settled() {
        return settled
      },
      settle: (actual?: BudgetCharge) => {
        if (settled) return
        settled = true
        for (const [dimension, amount] of Object.entries(claim) as [BudgetDimension, number][]) {
          if (!amount) continue
          this.reserved[dimension] = Math.max(0, this.reserved[dimension] - amount)
        }
        if (actual) this.charge(actual)
      },
    }
  }

  /**
   * Record settled usage. Propagates to every ancestor, so an ancestor's
   * `usage()` always covers its entire subtree.
   */
  charge(actual: BudgetCharge): void {
    for (const [dimension, amount] of Object.entries(actual) as [BudgetDimension, number][]) {
      if (!amount) continue
      this.applyCharge(dimension, amount)
    }
  }

  private applyCharge(dimension: BudgetDimension, amount: number): void {
    switch (dimension) {
      case 'maxRootIterations': this.own.rootIterations += amount; break
      case 'maxTotalQueries': this.own.queries += amount; break
      case 'maxTotalKernels': this.own.kernels += amount; break
      case 'maxInputTokens': this.own.inputTokens += amount; break
      case 'maxOutputTokens': this.own.outputTokens += amount; break
      case 'maxContextReadBytes': this.own.contextReadBytes += amount; break
      case 'maxCostUsd': this.own.costUsd += amount; break
      case 'maxWallClockMs': this.own.wallClockMs = Math.max(this.own.wallClockMs, amount); break
      case 'maxConcurrentQueries':
      case 'maxConcurrentKernels':
        this.own.maxObservedConcurrency = Math.max(this.own.maxObservedConcurrency, amount)
        break
      default: break
    }
    this.parent?.applyCharge(dimension, amount)
  }

  /** How much of a dimension is currently claimed but not yet settled. */
  outstanding(dimension: BudgetDimension): number {
    return this.reserved[dimension]
  }

  /**
   * Create a strictly narrower child account. The requested limits are
   * intersected with the parent's remaining capacity, so authority can only
   * shrink with depth.
   */
  child(limits: Partial<RlmBudgetLimits>, scope: string): RlmBudgetAccount {
    const requested: RlmBudgetLimits = { ...this.limits, ...limits }
    return new RlmBudgetAccount(requested, this, { scope, now: this.now })
  }

  /** Dimensions with no room left, ignoring wall clock. */
  exhausted(): BudgetDimension[] {
    const remaining = this.remainingLimits()
    return (Object.keys(remaining) as BudgetDimension[]).filter(
      (dimension) =>
        dimension !== 'maxConcurrentQueries' &&
        dimension !== 'maxConcurrentKernels' &&
        remaining[dimension] <= 0,
    )
  }

  /** Collapse an exhausted account onto one stop reason. */
  stopReasonForExhaustion(): RlmStopReason {
    if (this.elapsedMs() >= this.limits.maxWallClockMs) return 'wall-clock'
    const reasons = this.exhausted()
    if (reasons.includes('maxRootIterations')) return 'max-root-iterations'
    return reasons.length > 0 ? 'budget-exceeded' : 'ok'
  }

  /** Immutable snapshot for durable events and diagnostics. */
  snapshot(): {
    scope: string
    limits: RlmBudgetLimits
    usage: RlmUsage
    remaining: RlmBudgetLimits
    children: ReturnType<RlmBudgetAccount['snapshot']>[] | undefined
    elapsedMs: number
  } {
    return {
      scope: this.scope,
      limits: { ...this.limits },
      usage: this.usage(),
      remaining: this.remainingLimits(),
      elapsedMs: this.elapsedMs(),
      children: this.children.length > 0 ? this.children.map((child) => child.snapshot()) : undefined,
    }
  }
}

/** Intersect two limit sets, keeping the tighter value of every dimension. */
export function narrowLimits(a: RlmBudgetLimits, b: RlmBudgetLimits): RlmBudgetLimits {
  const result = { ...a }
  for (const key of Object.keys(result) as BudgetDimension[]) {
    result[key] = Math.min(a[key], b[key])
  }
  return result
}
