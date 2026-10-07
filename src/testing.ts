/**
 * Test doubles for the RLM seams.
 *
 * These are shipped (not hidden in `test/`) on purpose: an RLM is only
 * reviewable if its root loop can be driven deterministically, and a deployment
 * that wants to evaluate a new prompt profile needs the same hooks. They are
 * also what the plugin's own suite uses, so the doubles cannot drift from the
 * production contracts without breaking tests.
 *
 * @module dsh-rlm/testing
 */

import { createHash, randomUUID } from 'node:crypto'
import { mkdir, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { Context } from '@deepseek-ai/cordis'
import { RlmKernelRuntime, type RlmKernelCapabilities } from './kernel/index.ts'
import { emptyUsage, type RlmModelClient, type RlmModelRequest, type RlmModelResponse, type RlmTokenUsage } from './llm.ts'
import type {
  RlmExportOptions,
  RlmExportResult,
  RlmKernel,
  RlmKernelHandle,
  RlmKernelStartRequest,
  RlmStepResult,
} from './types.ts'

export interface ScriptedModelOptions {
  /**
   * Root-model replies, consumed in order. A function receives the request and
   * the zero-based call index for tests that must branch on history.
   */
  root: readonly string[] | ((request: RlmModelRequest, index: number) => string)
  /** Leaf answer for one `llm_query` prompt. Defaults to echoing the prompt. */
  leaf?: (prompt: string, request: RlmModelRequest) => string
  /** Usage reported for every call; defaults to a small deterministic value. */
  usagePerCall?: Partial<RlmTokenUsage>
  /** Error to return for the Nth call (0-based), for failure-path tests. */
  failCall?: number
  failMessage?: string
}

/** Records every call and replies from a script. Never touches the network. */
export class ScriptedModelClient implements RlmModelClient {
  /** Every request, root and leaf, in order. */
  readonly calls: RlmModelRequest[] = []
  /** Root-turn inputs only (a request carrying a system prompt). */
  readonly prompts: string[] = []
  /** Leaf prompts only, in order. */
  readonly leafPrompts: string[] = []
  private rootIndex = 0

  constructor(private readonly options: ScriptedModelOptions) {}

  /** Number of root-model calls observed. */
  get rootCalls(): number {
    return this.calls.length
  }

  async complete(request: RlmModelRequest): Promise<RlmModelResponse> {
    const callIndex = this.calls.length
    this.calls.push(request)
    const isLeaf = request.system === undefined
    const prompt = request.messages.map((message) => message.text).join('\n')
    // Root turns and leaf calls advance independent scripts: a leaf call must
    // never consume a scripted root reply.
    const rootIndex = isLeaf ? -1 : this.rootIndex++
    if (isLeaf) this.leafPrompts.push(prompt)
    else this.prompts.push(prompt)
    const usage: RlmTokenUsage = { ...emptyUsage(), inputTokens: 11, outputTokens: 7, ...this.options.usagePerCall }
    if (this.options.failCall === callIndex) {
      return {
        text: '',
        usage,
        finishReason: 'error',
        error: { kind: 'error', message: this.options.failMessage ?? 'scripted failure' },
      }
    }
    const text = isLeaf
      ? (this.options.leaf?.(request.messages[0]?.text ?? '', request) ?? `echo:${(request.messages[0]?.text ?? '').slice(0, 40)}`)
      : typeof this.options.root === 'function'
        ? this.options.root(request, rootIndex)
        : (this.options.root[rootIndex] ?? 'FINAL(script exhausted)')
    return { text, usage, finishReason: 'stop' }
  }
}

export interface InMemoryKernelBehavior {
  /** Produce one step result for a program. */
  eval(program: string, kernel: InMemoryKernel): RlmStepResult | Promise<RlmStepResult>
}

export class InMemoryKernelRuntime extends RlmKernelRuntime {
  override readonly providerName = 'in-memory'
  override readonly capabilities: RlmKernelCapabilities = {
    language: 'python',
    isolation: 'in-process',
    network: 'disabled',
    filesystem: 'private-tmp',
    processTreeLimits: true,
    processTreeTermination: true,
    hardDeadline: true,
    scrubbedEnvironment: true,
    guestGuards: ['none (test double)'],
  }

  readonly started: InMemoryKernel[] = []
  private live = new Set<InMemoryKernel>()
  private readonly behavior: InMemoryKernelBehavior

  /**
   * @param behavior - per-program script for the fake kernel.
   * @param ctx - owning context. Omitted in tests, where a fresh root context is
   *   the correct owner because the fake kernel registers no other service.
   */
  constructor(
    behavior: InMemoryKernelBehavior | ((program: string, kernel: InMemoryKernel) => RlmStepResult | Promise<RlmStepResult>),
    ctx: Context = new Context(),
  ) {
    super(ctx)
    this.behavior = typeof behavior === 'function' ? { eval: behavior } : behavior
  }

  override async start(request: RlmKernelStartRequest): Promise<RlmKernelHandle> {
    const kernel = new InMemoryKernel(request, this.behavior)
    this.started.push(kernel)
    this.live.add(kernel)
    return {
      kernel,
      exited: kernel.exited,
      dispose: async () => {
        await kernel.close()
        this.live.delete(kernel)
      },
    }
  }

  override get liveKernels(): number {
    return this.live.size
  }

  override async disposeAll(): Promise<void> {
    for (const kernel of [...this.live]) await kernel.close()
    this.live.clear()
  }
}

export class InMemoryKernel implements RlmKernel {
  readonly id = `kernel_${randomUUID()}`
  readonly providerName = 'in-memory'
  readonly isolation = 'in-process'
  readonly workspace: string
  readonly programs: string[] = []
  readonly variables = new Map<string, string>()
  closed = false
  readonly exited: Promise<{ code: number | null; signal: NodeJS.Signals | null; expected: boolean }>
  private resolveExit!: (value: { code: number | null; signal: NodeJS.Signals | null; expected: boolean }) => void

  constructor(
    readonly request: RlmKernelStartRequest,
    private readonly behavior: InMemoryKernelBehavior,
  ) {
    this.workspace = join(request.cwd, `.rlm-test-kernel-${this.id}`)
    this.exited = new Promise((resolveExit) => {
      this.resolveExit = resolveExit
    })
  }

  async eval(program: string): Promise<RlmStepResult> {
    if (this.closed) {
      return { stdout: '', stdoutTruncated: false, elapsedMs: 0, error: { kind: 'worker-exit', message: 'closed', fatal: true } }
    }
    this.programs.push(program)
    return this.behavior.eval(program, this)
  }

  async export(name: string, options: RlmExportOptions): Promise<RlmExportResult> {
    const value = this.variables.get(name)
    if (value === undefined) throw new Error(`no variable named ${name}`)
    await mkdir(dirname(options.path), { recursive: true })
    await writeFile(options.path, value, 'utf8')
    const digest = createHash('sha256').update(value).digest('hex')
    return {
      bytes: Buffer.byteLength(value, 'utf8'),
      chars: value.length,
      sha256: digest,
      typeName: 'str',
      preview: value.slice(0, options.previewChars),
      truncated: value.length > options.previewChars,
    }
  }

  async close(): Promise<void> {
    this.closed = true
    this.resolveExit({ code: 0, signal: null, expected: true })
  }
}

/** Convenience wrapper for a single prompt-driven evaluation. */
export function step(stdout: string, extra: Partial<RlmStepResult> = {}): RlmStepResult {
  return { stdout, stdoutTruncated: false, elapsedMs: 1, ...extra }
}
