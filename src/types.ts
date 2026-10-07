/**
 * Vocabulary for the RLM capability seams.
 *
 * These types are the host-side contract of Algorithm 1 from *Recursive
 * Language Models* (arXiv 2512.24601). They are pure types: no runtime code, so
 * the kernel provider, the context store and the root state machine can be
 * swapped independently.
 *
 * @module dsh-rlm/types
 */

// --------------------------------------------------------------------------- //
// context / artifacts
// --------------------------------------------------------------------------- //

/** Shape of an externalized prompt, mirroring the paper's context types. */
export type RlmContextKind = 'text' | 'list' | 'record' | 'documents'

/**
 * Bounded, content-addressed reference to a value that lives outside the root
 * model's history. Everything the root model is allowed to know about the
 * prompt is derived from this object.
 */
export interface RlmContextRef {
  /** Store-local identifier, opaque to callers. */
  id: string
  /** Content hash of the canonical serialization; the deduplication key. */
  sha256: string
  kind: RlmContextKind
  /** Canonical serialized size, for admission and diagnostics. */
  bytes: number
  /** Total characters across all leaf strings: the paper's `context_total_length`. */
  chars: number
  /** Per-chunk character lengths where a chunking is meaningful (list/documents). */
  chunkChars?: number[]
  /** Estimated token count per tokenizer name; never a billing claim. */
  estimatedTokens: Record<string, number>
  encoding: 'utf-8'
  /** Owning session, when the store is session-scoped. */
  sessionId?: string
  createdAt: number
}

/** Range-bounded, immutable, lazily materialized view over a stored context. */
export interface RlmContextHandle {
  readonly ref: RlmContextRef
  readonly kind: RlmContextKind
  /** Number of addressable items: characters for text, elements otherwise. */
  readonly length: number
  peek(start?: number, end?: number, signal?: AbortSignal): Promise<string>
  slice(start: number, end: number, signal?: AbortSignal): Promise<string>
  items(start: number, end: number, signal?: AbortSignal): Promise<unknown[]>
  get(key: string, signal?: AbortSignal): Promise<unknown>
  search(query: string, options?: SearchOptions): Promise<SearchHit[]>
  materialize(options: { maxBytes: number }, signal?: AbortSignal): Promise<unknown>
}

/** Options for a context search. Regex is opt-in because it is a ReDoS surface. */
export interface SearchOptions {
  maxHits?: number
  regex?: boolean
  caseSensitive?: boolean
}

/** One match inside a context. Offsets are character offsets, not bytes. */
export interface SearchHit {
  start: number
  end: number
  text: string
  item?: number
}

/** Accepted inputs for `run_rlm({ context })`. */
export type RlmContextInput =
  | { kind: 'text'; text: string }
  | { kind: 'list'; items: readonly string[] }
  | { kind: 'record'; record: Readonly<Record<string, unknown>> }
  | { kind: 'documents'; documents: readonly string[] }

/** A stored large final value; returned instead of pushing text through the root window. */
export interface RlmArtifactRef {
  id: string
  sha256: string
  bytes: number
  chars: number
  contentType: 'text/plain' | 'application/json'
  preview: string
  truncated: boolean
}

// --------------------------------------------------------------------------- //
// structured root actions (README §4.6)
// --------------------------------------------------------------------------- //

/**
 * Every root-model turn must resolve to exactly one of these. `FINAL(x)` and
 * `FINAL_VAR(x)` in model text are a *parser* compatibility surface that maps
 * onto this union; they are never executed as Python.
 */
export type RlmRootAction =
  | { type: 'eval'; code: string }
  | { type: 'final-text'; text: string }
  | { type: 'final-variable'; name: string }

/** Why a root turn could not be turned into exactly one action. */
export interface RlmParseFailure {
  reason:
    | 'empty'
    | 'mixed-code-and-final'
    | 'mixed-final-forms'
    | 'unterminated-code-fence'
    | 'unknown'
  message: string
}

// --------------------------------------------------------------------------- //
// kernel
// --------------------------------------------------------------------------- //

/** Failure taxonomy for one evaluation step. */
export type RlmStepErrorKind =
  | 'syntax'
  | 'name'
  | 'exception'
  | 'timeout'
  | 'cancelled'
  | 'protocol'
  | 'sandbox-unavailable'
  | 'worker-exit'
  | 'budget'

export interface RlmStepError {
  kind: RlmStepErrorKind
  message: string
  /** Trimmed traceback of the program's own frames. Model-visible. */
  traceback?: string
  /** True when the kernel must be considered dead after this step. */
  fatal?: boolean
}

/** Result of one `eval` in the persistent REPL. */
export interface RlmStepResult {
  stdout: string
  stdoutTruncated: boolean
  elapsedMs: number
  error?: RlmStepError
  /**
   * Set when the program assigned the configured final variable (the paper's
   * `state[Final] is set`). The value itself stays in the REPL.
   */
  final?: {
    name: string
    typeName: string
    preview: string
    truncated: boolean
    chars: number | null
  }
}

/** One host-side call the kernel can make from inside program code. */
export interface RlmKernelHost {
  /** A leaf model request (`llm_query`). Returns a string or JSON value. */
  llmQuery(args: RlmLlmQueryArgs, signal: AbortSignal): Promise<unknown>
  /** A child RLM invocation (`rlm_query`). Returns the child's final text. */
  rlmQuery(args: RlmChildQueryArgs, signal: AbortSignal): Promise<unknown>
  /** A `context.*` method call; metered as a context read. */
  contextCall(name: string, args: Record<string, unknown>, signal: AbortSignal): Promise<unknown>
  /** Observation stream for events and diagnostics. Never model-visible. */
  observe?(observation: RlmKernelObservation): void
}

export interface RlmLlmQueryArgs {
  prompt: string
  model?: string
  maxTokens?: number
}

export interface RlmChildQueryArgs {
  context: { kind: 'handle'; ref: RlmContextRef } | { kind: 'inline'; text?: string; items?: string[]; record?: Record<string, string> }
  query: string
  model?: string
}

export type RlmKernelObservation =
  | { type: 'binding-start'; callId: number; name: string; bytes: number }
  | { type: 'binding-end'; callId: number; name: string; ok: boolean; elapsedMs: number; bytes: number }
  | { type: 'stdout'; bytes: number; truncated: boolean }
  | { type: 'stderr'; text: string }
  | { type: 'exited'; code: number | null; signal: string | null; expected: boolean }

/** Request handed to a kernel provider. */
export interface RlmKernelStartRequest {
  invocationId: string
  attemptId: string
  /** Remaining recursion depth available *inside* this kernel. */
  depth: number
  /** Deployment ceiling, for prompt text and child admission. */
  maxDepth: number
  contextRef: RlmContextRef
  contextKind: RlmContextKind
  /** Binding names the program may call, decided by the host, never by the model. */
  bindings: readonly string[]
  finalVarName: string
  maxStdoutChars: number
  maxConcurrency: number
  /** Working directory for the kernel process. */
  cwd: string
  /** Hard lifetime independent of model instructions. */
  maxLifetimeMs: number
  /** Per-eval budget; exceeding it kills the kernel. */
  evalTimeoutMs: number
  host: RlmKernelHost
  signal: AbortSignal
}

/** A live persistent kernel. */
export interface RlmKernel {
  readonly id: string
  readonly providerName: string
  readonly isolation: string
  /**
   * Absolute private working directory owned by this kernel.
   *
   * The host uses it for large-value exports, which is why the kernel -- not the
   * model and not the session -- decides the path: model code never supplies a
   * filesystem location the host then trusts.
   */
  readonly workspace: string
  eval(program: string, options?: { timeoutMs?: number; signal?: AbortSignal }): Promise<RlmStepResult>
  /**
   * Serialize a REPL variable to a host-provided path and report bounded
   * metadata about it.
   *
   * This is how a final answer larger than the root window leaves the kernel:
   * the bytes go to a file, the control frame stays small, and the value never
   * enters root history (README §4.6).
   */
  export(name: string, options: RlmExportOptions): Promise<RlmExportResult>
  close(): Promise<void>
}

export interface RlmExportOptions {
  /** Absolute path inside the kernel's private workspace. */
  path: string
  previewChars: number
  /** Refuse to write more than this; the host treats it as a budget failure. */
  maxBytes: number
  signal?: AbortSignal
}

export interface RlmExportResult {
  bytes: number
  chars: number
  sha256: string
  typeName: string
  preview: string
  truncated: boolean
}

/** Handle returned by `RlmKernelRuntime.start`. */
export interface RlmKernelHandle {
  readonly kernel: RlmKernel
  /** Resolves when the kernel process is gone and all handles are released. */
  readonly exited: Promise<RlmKernelExit>
  dispose(): Promise<void>
}

export interface RlmKernelExit {
  code: number | null
  signal: NodeJS.Signals | null
  expected: boolean
}

// --------------------------------------------------------------------------- //
// budgets
// --------------------------------------------------------------------------- //

/** Host-owned limits. Every field is a hard ceiling, never a prompt instruction. */
export interface RlmBudgetLimits {
  maxRootIterations: number
  maxWallClockMs: number
  maxTotalQueries: number
  maxConcurrentQueries: number
  maxTotalKernels: number
  maxConcurrentKernels: number
  maxInputTokens: number
  maxOutputTokens: number
  maxContextReadBytes: number
  maxQueryPromptChars: number
  maxInlineOutputBytes: number
  maxArtifactBytes: number
  maxCostUsd: number
}

/** Monotone counters for one invocation tree. */
export interface RlmUsage {
  rootIterations: number
  queries: number
  kernels: number
  inputTokens: number
  outputTokens: number
  contextReadBytes: number
  costUsd: number
  wallClockMs: number
  maxObservedConcurrency: number
}

/** Why an invocation stopped. `ok` is the only success value. */
export type RlmStopReason =
  | 'ok'
  | 'final-text'
  | 'final-variable'
  | 'max-root-iterations'
  | 'wall-clock'
  | 'budget-exceeded'
  | 'cancelled'
  | 'kernel-failure'
  | 'root-parse-failure'
  | 'root-model-failure'
  | 'protocol'

export interface RlmResult {
  status: 'ok' | 'failed' | 'cancelled' | 'budget-exceeded'
  /** Present for small finals; large finals come back as {@link artifactRef}. */
  text?: string
  artifactRef?: RlmArtifactRef
  stopReason: RlmStopReason
  depth: number
  maxDepth: number
  steps: number
  usage: RlmUsage
  error?: { kind: string; message: string }
}
