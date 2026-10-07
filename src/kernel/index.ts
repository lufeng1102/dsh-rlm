/**
 * Service definition for the persistent-RLM-kernel seam.
 *
 * This seam is deliberately narrower than a code-execution runtime and
 * deliberately wider than a one-shot program runner. A provider must own:
 *
 * - **one** REPL namespace for the whole invocation, so variables survive root
 *   iterations (the defining property of Algorithm 1);
 * - the `context` handle and the `llm_query` / `rlm_query` bindings, so program
 *   code can invoke the model inside loops;
 * - lifetime, cancellation and process-tree cleanup.
 *
 * It must not own depth, budgets, model routes or admission control: those are
 * host policy, and a provider that lets program code change them has inverted
 * the trust boundary (README §4.3).
 *
 * The released `@deepseek-ai/dsh-ptc-runtime` deliberately starts a fresh
 * substrate per run; this seam exists so persisting a REPL does not require
 * changing that contract.
 *
 * @module dsh-rlm/kernel
 */

import { Service } from '@deepseek-ai/cordis'
import type { RlmKernelHandle, RlmKernelStartRequest } from '../types.ts'

declare module '@deepseek-ai/cordis' {
  interface Context {
    rlmKernelRuntime: RlmKernelRuntime
  }
}

/**
 * What a provider actually enforces. The core checks these against the
 * deployment's isolation policy at load time and refuses to start hardened runs
 * on a provider that cannot honor them — "fail closed" has to mean something
 * concrete, so it is a comparison of declared capabilities, not a comment.
 */
export interface RlmKernelCapabilities {
  /** Program language of the REPL surface. */
  language: 'python' | 'typescript'
  /** Execution substrate, for diagnostics; not a security claim by itself. */
  isolation: 'process' | 'container' | 'worker-thread' | 'in-process'
  /**
   * `disabled` means the substrate enforces it (network namespace, container,
   * sandbox profile). `guest-guard` means only the guest language blocks its own
   * socket/subprocess APIs — defense in depth, NOT a security boundary, and the
   * distinction is what stops a log from over-claiming.
   */
  network: 'disabled' | 'allowlist' | 'guest-guard' | 'unrestricted'
  filesystem: 'private-tmp' | 'read-only-context' | 'shared'
  /** Provider can bound the whole process tree's CPU/RSS. */
  processTreeLimits: boolean
  /** Provider terminates the process tree on abort/timeout/dispose. */
  processTreeTermination: boolean
  /** Provider enforces a hard lifetime regardless of model instructions. */
  hardDeadline: boolean
  /** Provider injects no host credentials or environment. */
  scrubbedEnvironment: boolean
  /** Guest-level defenses actually installed, named honestly. */
  guestGuards?: readonly string[]
}

/** Deployment's minimum bar for accepting a provider. */
export interface RlmIsolationPolicy {
  requireNetworkDisabled: boolean
  requireProcessTreeTermination: boolean
  requireHardDeadline: boolean
  requireScrubbedEnvironment: boolean
  /** Escape hatch for local, trusted, experimental use. */
  allowUnhardenedProvider: boolean
}

export const DEFAULT_ISOLATION_POLICY: RlmIsolationPolicy = {
  requireNetworkDisabled: false,
  requireProcessTreeTermination: true,
  requireHardDeadline: true,
  requireScrubbedEnvironment: true,
  allowUnhardenedProvider: true,
}

/** Structured refusal thrown when a provider cannot meet the policy. */
export class RlmIsolationRefusedError extends Error {
  override readonly name = 'RlmIsolationRefusedError'
  readonly unmet: readonly string[]

  constructor(providerName: string, unmet: readonly string[]) {
    super(
      `kernel provider ${providerName} cannot satisfy the isolation policy: ${unmet.join(', ')}. ` +
        'Set rlm.isolation.allowUnhardenedProvider for local trusted use, or install a hardened provider.',
    )
    this.unmet = unmet
  }
}

/** Compare declared capabilities against a policy. */
export function evaluateIsolation(
  providerName: string,
  capabilities: RlmKernelCapabilities,
  policy: RlmIsolationPolicy,
): string[] {
  const unmet: string[] = []
  if (policy.requireNetworkDisabled && capabilities.network !== 'disabled') unmet.push('network-not-disabled')
  if (policy.requireProcessTreeTermination && !capabilities.processTreeTermination) {
    unmet.push('no-process-tree-termination')
  }
  if (policy.requireHardDeadline && !capabilities.hardDeadline) unmet.push('no-hard-deadline')
  if (policy.requireScrubbedEnvironment && !capabilities.scrubbedEnvironment) unmet.push('inherits-host-environment')
  if (unmet.length > 0 && !policy.allowUnhardenedProvider) {
    throw new RlmIsolationRefusedError(providerName, unmet)
  }
  return unmet
}

/**
 * One `ctx.rlmKernelRuntime` implementation per deployment. Registering a second
 * provider is a Cordis conflict, which is the intent: the RLM core must not have
 * to choose between two isolation stories at runtime.
 */
export abstract class RlmKernelRuntime extends Service {
  /** Provider identifier for configuration, events and diagnostics. */
  abstract readonly providerName: string
  /** Declared enforcement; see {@link RlmKernelCapabilities}. */
  abstract readonly capabilities: RlmKernelCapabilities

  constructor(ctx: import('@deepseek-ai/cordis').Context) {
    super(ctx, 'rlmKernelRuntime')
  }

  /**
   * Start one persistent kernel and return it live.
   *
   * Resolves when the REPL is ready to accept `eval`; rejects when the kernel
   * cannot be established at all (in which case no event may claim it started).
   */
  abstract start(request: RlmKernelStartRequest): Promise<RlmKernelHandle>

  /** Number of live kernels; providers report this for concurrency accounting. */
  abstract get liveKernels(): number

  /**
   * Reject new starts and terminate every live kernel. Called on plugin unload
   * and by the manager when a hard budget is tripped.
   */
  abstract disposeAll(reason: string): Promise<void>
}

export default RlmKernelRuntime
