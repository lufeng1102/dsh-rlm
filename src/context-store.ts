/**
 * Service definition for the RLM context/artifact seam.
 *
 * The store exists for exactly one reason: to keep the complete user prompt
 * addressable by program code while the root model only ever sees bounded
 * metadata about it (README §4.1). An opaque reference alone would not be
 * enough — the REPL needs `len(context)`, range slices, item pages, keyed reads
 * and search, all of them bounded and metered.
 *
 * Implementations must:
 * - address content by hash and deduplicate by it;
 * - authorize reads per session/tenant;
 * - serve range reads without materializing the whole value;
 * - treat Unicode correctly, so a character range is not a byte range;
 * - support retention and explicit deletion of artifacts.
 *
 * @module dsh-rlm/context-store
 */

import { Service } from '@deepseek-ai/cordis'
import type {
  RlmArtifactRef,
  RlmContextHandle,
  RlmContextInput,
  RlmContextRef,
} from './types.ts'

declare module '@deepseek-ai/cordis' {
  interface Context {
    rlmContextStore: RlmContextStore
  }
}

export interface RlmContextPutOptions {
  /** Owning session used for later authorization checks. */
  sessionId?: string
  /** Skip the write when an identical content hash already exists. */
  deduplicate?: boolean
  signal?: AbortSignal
}

export interface RlmContextOpenOptions {
  /** Caller's session; required when the stored ref is session-scoped. */
  sessionId?: string
  signal?: AbortSignal
}

export interface RlmArtifactWriteOptions {
  sessionId?: string
  /** Refuse a value above this size rather than truncating it silently. */
  maxBytes: number
  contentType?: 'text/plain' | 'application/json'
  /** Characters of preview delivered alongside the reference. */
  previewChars?: number
  signal?: AbortSignal
}

export interface RlmSweepOptions {
  /** Delete artifacts older than this many milliseconds. */
  maxAgeMs: number
  /** Only sweep this session's artifacts. */
  sessionId?: string
  /** Report what would be deleted without deleting it. */
  dryRun?: boolean
  now?: number
}

export interface RlmSweepReport {
  scanned: number
  deleted: number
  reclaimedBytes: number
  artifacts: string[]
}

export interface RlmContextStoreStats {
  provider: string
  root: string
  artifacts: number
  bytes: number
  /** True when deployment policy requires encryption and the provider can do it. */
  encrypted: boolean
}

/**
 * Storage for externalized prompts and large RLM outputs.
 *
 * Registered by a provider plugin (e.g. `dsh-rlm/context-local`); the core
 * plugin only depends on this seam, so a deployment can swap in an encrypted,
 * tenant-isolated or remote store without touching RLM semantics.
 */
export abstract class RlmContextStore extends Service {
  /** Provider identifier for diagnostics and durable events. */
  abstract readonly providerName: string

  constructor(ctx: import('@deepseek-ai/cordis').Context) {
    super(ctx, 'rlmContextStore')
  }

  /** Persist a prompt/context value and return its bounded reference. */
  abstract put(input: RlmContextInput, options?: RlmContextPutOptions): Promise<RlmContextRef>

  /** Open an immutable lazy handle. Reads are authorized and range-bounded. */
  abstract open(ref: RlmContextRef, options?: RlmContextOpenOptions): Promise<RlmContextHandle>

  /** Read a stored reference back as a full value, up to a hard byte cap. */
  abstract read(ref: RlmContextRef, options?: { maxBytes?: number; signal?: AbortSignal }): Promise<string>

  /** Persist a large final value as an output artifact. */
  abstract writeArtifact(
    value: string | unknown,
    options: RlmArtifactWriteOptions,
  ): Promise<RlmArtifactRef>

  /** Read an artifact back by reference (used by the client/stream surface). */
  abstract readArtifact(ref: RlmArtifactRef, options?: { maxBytes?: number; signal?: AbortSignal }): Promise<string>

  /**
   * Adopt an already-written file as an artifact without routing its bytes
   * through a single in-memory string or control frame.
   *
   * This is the large-final path: the kernel writes the value into its own
   * private workspace, and the store moves it into managed storage.
   */
  abstract writeArtifactFromFile(path: string, options: RlmArtifactWriteOptions): Promise<RlmArtifactRef>

  /** Delete one stored value. Idempotent. */
  abstract delete(ref: RlmContextRef | RlmArtifactRef, options?: { sessionId?: string }): Promise<boolean>

  /** Retention/garbage collection pass. */
  abstract sweep(options: RlmSweepOptions): Promise<RlmSweepReport>

  /** Deployments report health/budget here for dashboards and SLOs. */
  abstract stats(options?: { sessionId?: string }): Promise<RlmContextStoreStats>
}

export default RlmContextStore
