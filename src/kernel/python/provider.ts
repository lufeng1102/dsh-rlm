/**
 * Isolated, persistent Python RLM kernel provider.
 *
 * One OS process per invocation, one REPL namespace for its whole lifetime, a
 * length-framed JSON control channel on stdio, and no host credentials in the
 * child's environment. The driver itself is `rlm_kernel.py` next to this file.
 *
 * What this provider owns:
 * - process lifetime, private working directory, scrubbed environment;
 * - the binding dispatch table (`llm_query`, `rlm_query`, `context.*`) — but not
 *   their policy: depth, budgets and admission control are the caller's;
 * - process-tree termination on cancellation, timeout, hard lifetime or unload;
 * - turning a dead substrate into a truthful `worker-exit` step error.
 *
 * What it deliberately does NOT claim: syscall-level confinement. A plain
 * subprocess cannot disable the network or quota a process tree's RSS, so
 * {@link RlmPythonKernelRuntime.capabilities} reports `network: 'guest-guard'`
 * and `processTreeLimits: false` unless the operator supplies an
 * `isolationCommand` wrapper (bwrap, sandbox-exec, container runtime). The core
 * compares those declarations against the deployment's policy and fails closed
 * when the policy asks for more than the provider can prove.
 *
 * @module dsh-rlm/kernel/python
 */

import { spawn, type ChildProcess } from 'node:child_process'
import { createHash, randomUUID } from 'node:crypto'
import { mkdir, mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import type { Context } from '@deepseek-ai/cordis'
import {
  RlmKernelRuntime,
  type RlmKernelCapabilities,
} from '../index.ts'
import { FramedJsonReader, RlmProtocolError, writeFrame, type RlmFrame } from '../protocol.ts'
import type {
  RlmExportOptions,
  RlmExportResult,
  RlmKernel,
  RlmKernelHandle,
  RlmKernelHost,
  RlmKernelStartRequest,
  RlmStepError,
  RlmStepResult,
} from '../../types.ts'

export interface PythonKernelConfig {
  /** Interpreter used for the driver. */
  pythonPath?: string
  /** Absolute path to `rlm_kernel.py`; defaults to the copy shipped next to this module. */
  driverPath?: string
  /** Hard ceiling on one control frame in either direction. */
  maxFrameBytes?: number
  /** Wall-clock grace between SIGTERM and SIGKILL of the process group. */
  killGraceMs?: number
  /** Install the guest-level socket/subprocess guard (defense in depth only). */
  guestNetworkGuard?: boolean
  /** Wrapper command prepended to the driver invocation (e.g. `bwrap`, `sandbox-exec`). */
  isolationCommand?: string[]
  /** Declares that `isolationCommand` really denies network access. */
  isolationCommandDisablesNetwork?: boolean
  /** Extra environment for the child, on top of the scrubbed base. */
  env?: Record<string, string>
  /** Where private per-invocation working directories are created. */
  workspaceRoot?: string
}

export interface ResolvedPythonKernelConfig extends PythonKernelConfig {
  pythonPath: string
  driverPath: string
  maxFrameBytes: number
  killGraceMs: number
  guestNetworkGuard: boolean
}

interface PendingEval {
  id: number
  resolve: (result: RlmStepResult) => void
  timer?: NodeJS.Timeout
  settled: boolean
}

interface InFlightBinding {
  controller: AbortController
  name: string
  startedAt: number
}

const DEFAULT_MAX_FRAME_BYTES = 64 * 1024 * 1024
const DEFAULT_KILL_GRACE_MS = 2_000

export class RlmPythonKernelRuntime extends RlmKernelRuntime {
  readonly providerName = 'python-subprocess'

  readonly config: ResolvedPythonKernelConfig

  private readonly kernels = new Set<PythonKernelProcess>()
  private closed = false

  constructor(ctx: Context, config: PythonKernelConfig = {}) {
    super(ctx)
    this.config = {
      ...config,
      pythonPath: config.pythonPath ?? process.env.DSH_RLM_PYTHON ?? 'python3',
      driverPath: config.driverPath ?? resolve(dirname(fileURLToPath(import.meta.url)), 'rlm_kernel.py'),
      maxFrameBytes: config.maxFrameBytes ?? DEFAULT_MAX_FRAME_BYTES,
      killGraceMs: config.killGraceMs ?? DEFAULT_KILL_GRACE_MS,
      guestNetworkGuard: config.guestNetworkGuard ?? true,
    }
  }

  override get capabilities(): RlmKernelCapabilities {
    const wrapped = (this.config.isolationCommand?.length ?? 0) > 0
    const networkDisabled = wrapped && this.config.isolationCommandDisablesNetwork === true
    return {
      language: 'python',
      isolation: wrapped ? 'container' : 'process',
      network: networkDisabled ? 'disabled' : this.config.guestNetworkGuard ? 'guest-guard' : 'unrestricted',
      filesystem: 'private-tmp',
      // A bare subprocess has no cgroup quota; claiming one would be a lie.
      processTreeLimits: false,
      processTreeTermination: true,
      hardDeadline: true,
      scrubbedEnvironment: true,
      guestGuards: this.config.guestNetworkGuard
        ? ['python-socket-connect-blocked', 'python-subprocess-blocked', 'os.fork-blocked']
        : [],
    }
  }

  override get liveKernels(): number {
    return this.kernels.size
  }

  override async start(request: RlmKernelStartRequest): Promise<RlmKernelHandle> {
    if (this.closed) throw new Error('rlm kernel runtime is closed; no new kernels are accepted')
    const base = this.config.workspaceRoot ?? request.cwd ?? tmpdir()
    await mkdir(base, { recursive: true })
    const workspace = await mkdtemp(join(base, 'rlm-kernel-'))
    const process_ = new PythonKernelProcess({
      request,
      config: this.config,
      workspace,
      onExit: () => this.kernels.delete(process_),
    })
    this.kernels.add(process_)
    try {
      await process_.start()
    } catch (error) {
      this.kernels.delete(process_)
      await process_.dispose().catch(() => {})
      throw error
    }
    return {
      kernel: process_.kernel,
      exited: process_.exited,
      dispose: () => process_.dispose(),
    }
  }

  override async disposeAll(reason: string): Promise<void> {
    this.closed = true
    const live = [...this.kernels]
    this.kernels.clear()
    await Promise.all(live.map((kernel) => kernel.dispose(reason)))
  }
}

interface PythonKernelProcessOptions {
  request: RlmKernelStartRequest
  config: RlmPythonKernelRuntime['config']
  workspace: string
  onExit: () => void
}

/**
 * One live kernel process. Owns the read loop, binding dispatch table and
 * termination policy for exactly one RLM invocation.
 */
class PythonKernelProcess {
  readonly kernel: RlmKernel
  readonly exited: Promise<{ code: number | null; signal: NodeJS.Signals | null; expected: boolean }>

  private readonly request: RlmKernelStartRequest
  private readonly config: PythonKernelProcessOptions['config']
  private readonly workspace: string
  private readonly onExit: () => void

  private awaiters: { predicate: (frame: RlmFrame) => boolean; resolve: (frame: RlmFrame) => void; reject: (error: Error) => void }[] = []
  private child?: ChildProcess
  private reader?: FramedJsonReader
  private readLoop?: Promise<void>
  private resolveExit!: (value: { code: number | null; signal: NodeJS.Signals | null; expected: boolean }) => void
  private exitSettled = false
  private expectedExit = false
  private disposed = false
  private lifetimeTimer?: NodeJS.Timeout
  private nextEvalId = 1
  private pendingEval?: PendingEval
  private readonly bindings = new Map<number, InFlightBinding>()
  private exitInfo: { code: number | null; signal: NodeJS.Signals | null } = { code: null, signal: null }

  constructor(options: PythonKernelProcessOptions) {
    this.request = options.request
    this.config = options.config
    this.workspace = options.workspace
    this.onExit = options.onExit
    this.exited = new Promise((resolveExit) => {
      this.resolveExit = resolveExit
    })
    this.kernel = {
      id: `kernel_${randomUUID()}`,
      providerName: 'python-subprocess',
      isolation: 'process',
      workspace: options.workspace,
      eval: (program, evalOptions) => this.eval(program, evalOptions),
      export: (name, exportOptions) => this.exportVariable(name, exportOptions),
      close: () => this.dispose('kernel closed'),
    }
  }

  async start(): Promise<void> {
    const argv = [
      ...(this.config.isolationCommand ?? []),
      this.config.pythonPath,
      '-u',
      this.config.driverPath,
    ]
    const child = spawn(argv[0]!, argv.slice(1), {
      cwd: this.workspace,
      env: this.buildEnvironment(),
      stdio: ['pipe', 'pipe', 'pipe'],
      detached: process.platform !== 'win32',
    })
    this.child = child
    if (!child.stdout || !child.stdin) {
      await this.terminate('kernel stdio unavailable')
      const error: RlmStepError = { kind: 'sandbox-unavailable', message: 'kernel stdio unavailable', fatal: true }
      throw new Error(error.message)
    }
    child.stderr?.on('data', (chunk: Buffer) => {
      this.request.host.observe?.({ type: 'stderr', text: chunk.toString('utf8').slice(0, 4096) })
    })
    child.on('error', (error) => {
      this.request.host.observe?.({ type: 'stderr', text: `spawn error: ${error.message}` })
      // Without this the startup handshake would wait for a ready frame that can
      // never arrive, turning a missing interpreter into a silent timeout.
      this.failAllAwaiters(new Error(`kernel spawn failed: ${error.message}`))
      // Node does NOT emit 'exit' when the process could not be spawned at all,
      // so the exit promise must be settled here or disposal would wait forever.
      this.handleExit(null, null)
    })
    child.on('exit', (code, signal) => {
      this.exitInfo = { code, signal }
      this.handleExit(code, signal)
    })
    // Safety net for a kill that closes stdio without an 'exit' event.
    child.on('close', () => {
      this.handleExit(this.exitInfo.code, this.exitInfo.signal)
    })

    this.reader = new FramedJsonReader(child.stdout, { maxFrameBytes: this.config.maxFrameBytes })
    // Exactly one read loop for the kernel's whole life: startup frames and
    // program frames travel the same channel, and a second reader would steal
    // frames from the first.
    this.readLoop = this.consume()
    const ready = await this.awaitFrame(
      (frame) => frame.type === 'ready',
      'kernel did not report ready',
    )
    const protocolVersion = Number(ready.protocol ?? 0)
    if (protocolVersion !== 1) {
      await this.terminate(`unsupported kernel protocol ${protocolVersion}`)
      throw new Error(`kernel protocol ${protocolVersion} is not supported`)
    }

    await this.send({
      op: 'init',
      config: {
        depth: this.request.depth,
        maxDepth: this.request.maxDepth,
        bindings: [...this.request.bindings],
        contextKind: this.request.contextKind,
        finalVarName: this.request.finalVarName,
        maxStdoutChars: this.request.maxStdoutChars,
        maxConcurrency: this.request.maxConcurrency,
        guardNetwork: this.config.guestNetworkGuard,
      },
    })
    await this.awaitFrame((frame) => frame.type === 'ready' && frame.phase === 'initialized', 'kernel did not initialize')

    // Hard lifetime, independent of anything the model writes.
    this.lifetimeTimer = setTimeout(() => {
      void this.failPending('timeout', `kernel exceeded its ${this.request.maxLifetimeMs} ms hard lifetime`)
      void this.terminate('hard kernel lifetime exceeded')
    }, Math.max(1, this.request.maxLifetimeMs))
    this.lifetimeTimer.unref?.()

    this.request.signal.addEventListener('abort', this.onAbort, { once: true })
    if (this.request.signal.aborted) this.onAbort()
  }

  private onAbort = (): void => {
    void this.failPending('cancelled', 'parent invocation was cancelled')
    void this.terminate('parent invocation cancelled')
  }

  // --------------------------------------------------------------------- //
  // evaluation
  // --------------------------------------------------------------------- //

  private eval(program: string, options: { timeoutMs?: number; signal?: AbortSignal } = {}): Promise<RlmStepResult> {
    if (this.disposed || this.exitSettled) {
      return Promise.resolve({
        stdout: '',
        stdoutTruncated: false,
        elapsedMs: 0,
        error: {
          kind: 'worker-exit',
          message: `kernel is not alive (exit=${this.exitInfo.code ?? this.exitInfo.signal ?? 'unknown'})`,
          fatal: true,
        },
      })
    }
    if (this.pendingEval) {
      return Promise.resolve({
        stdout: '',
        stdoutTruncated: false,
        elapsedMs: 0,
        error: { kind: 'protocol', message: 'another evaluation is already in flight', fatal: true },
      })
    }
    const timeoutMs = options.timeoutMs ?? this.request.evalTimeoutMs
    const id = this.nextEvalId
    this.nextEvalId += 1
    return new Promise<RlmStepResult>((resolveResult) => {
      const pending: PendingEval = { id, resolve: resolveResult, settled: false }
      if (timeoutMs > 0) {
        pending.timer = setTimeout(() => {
          void this.failPending('timeout', `evaluation exceeded ${timeoutMs} ms; kernel terminated`)
          void this.terminate('evaluation deadline exceeded')
        }, timeoutMs)
        pending.timer.unref?.()
      }
      this.pendingEval = pending
      this.send({ op: 'eval', id, code: program, timeoutMs }).catch((error: Error) => {
        this.settlePending({
          stdout: '',
          stdoutTruncated: false,
          elapsedMs: 0,
          error: { kind: 'protocol', message: `failed to send program: ${error.message}`, fatal: true },
        })
      })
      options.signal?.addEventListener(
        'abort',
        () => {
          void this.failPending('cancelled', 'evaluation cancelled')
          void this.terminate('evaluation cancelled')
        },
        { once: true },
      )
    })
  }

  /**
   * Ask the kernel to serialize a REPL variable into `options.path`.
   *
   * The reply carries only bounded metadata (size, hash, preview), so this is the
   * path by which a final answer far larger than the root window leaves the
   * kernel without ever becoming one control frame (README §4.6).
   */
  private exportVariable(name: string, options: RlmExportOptions): Promise<RlmExportResult> {
    if (this.disposed || this.exitSettled) {
      return Promise.reject(new Error('kernel is not alive'))
    }
    const id = this.nextEvalId
    this.nextEvalId += 1
    return new Promise<RlmExportResult>((resolveExport, rejectExport) => {
      const timer = setTimeout(() => {
        this.removeAwaiter(entry)
        rejectExport(new Error(`export of ${name} timed out`))
      }, Math.max(1, this.request.evalTimeoutMs))
      timer.unref?.()
      const entry = {
        predicate: (frame: RlmFrame) => frame.type === 'export' && Number(frame.id ?? 0) === id,
        resolve: (frame: RlmFrame) => {
          clearTimeout(timer)
          if (frame.ok === true) {
            resolveExport({
              bytes: Number(frame.bytes ?? 0),
              chars: Number(frame.chars ?? 0),
              sha256: String(frame.sha256 ?? ''),
              typeName: String(frame.typeName ?? 'unknown'),
              preview: String(frame.preview ?? ''),
              truncated: frame.truncated === true,
            })
            return
          }
          const error = frame.error as { message?: string } | undefined
          rejectExport(new Error(error?.message ?? `export of ${name} failed`))
        },
        reject: (error: Error) => {
          clearTimeout(timer)
          rejectExport(error)
        },
      }
      this.awaiters.push(entry)
      this.send({
        op: 'export',
        id,
        name,
        path: options.path,
        previewChars: options.previewChars,
        maxBytes: options.maxBytes,
      }).catch((error: Error) => {
        this.removeAwaiter(entry)
        rejectExport(error)
      })
    })
  }

  private settlePending(result: RlmStepResult): void {
    const pending = this.pendingEval
    if (!pending || pending.settled) return
    pending.settled = true
    if (pending.timer) clearTimeout(pending.timer)
    this.pendingEval = undefined
    pending.resolve(result)
  }

  private async failPending(kind: RlmStepError['kind'], message: string): Promise<void> {
    this.settlePending({
      stdout: '',
      stdoutTruncated: false,
      elapsedMs: 0,
      error: { kind, message, fatal: true },
    })
  }

  // --------------------------------------------------------------------- //
  // frame dispatch
  // --------------------------------------------------------------------- //

  private async consume(): Promise<void> {
    try {
      while (true) {
        const frame = await this.reader!.read()
        if (frame === null) {
          this.failAllAwaiters(new Error('kernel control stream ended before the expected frame'))
          return
        }
        this.dispatch(frame)
      }
    } catch (error) {
      const message = error instanceof RlmProtocolError ? error.message : String(error)
      this.request.host.observe?.({ type: 'stderr', text: `protocol error: ${message}` })
      this.failAllAwaiters(new Error(message))
      await this.settlePending({
        stdout: '',
        stdoutTruncated: false,
        elapsedMs: 0,
        error: { kind: 'protocol', message, fatal: true },
      })
      await this.terminate(`protocol error: ${message}`)
    }
  }

  private dispatch(frame: RlmFrame): void {
    // Startup frames (and, in principle, any expected frame) are claimed by the
    // awaiter that asked for them before ordinary handling sees them.
    const awaiterIndex = this.awaiters.findIndex((awaiter) => awaiter.predicate(frame))
    if (awaiterIndex >= 0) {
      const [awaiter] = this.awaiters.splice(awaiterIndex, 1)
      awaiter?.resolve(frame)
      return
    }
    switch (frame.type) {
      case 'result': {
        const id = Number(frame.id ?? 0)
        if (this.pendingEval && this.pendingEval.id === id) {
          this.settlePending(toStepResult(frame))
        }
        return
      }
      case 'binding': {
        // Fire-and-forget: the read loop must stay free to deliver the reply of
        // a *different* concurrent binding call.
        void this.serveBinding(frame)
        return
      }
      case 'fatal': {
        const message = String(frame.message ?? 'kernel reported a fatal error')
        void this.settlePending({
          stdout: '',
          stdoutTruncated: false,
          elapsedMs: 0,
          error: { kind: 'protocol', message, fatal: true },
        })
        return
      }
      case 'bye':
        return
      case 'ready':
        return
      default:
        this.request.host.observe?.({ type: 'stderr', text: `unknown frame type ${String(frame.type)}` })
    }
  }

  private async serveBinding(frame: RlmFrame): Promise<void> {
    const callId = Number(frame.callId ?? 0)
    const name = String(frame.name ?? '')
    const args = (frame.args ?? {}) as Record<string, unknown>
    const startedAt = Date.now()
    const controller = new AbortController()
    this.bindings.set(callId, { controller, name, startedAt })
    this.request.host.observe?.({
      type: 'binding-start',
      callId,
      name,
      bytes: Buffer.byteLength(JSON.stringify(args), 'utf8'),
    })
    const linked = () => controller.abort(this.request.signal.reason)
    this.request.signal.addEventListener('abort', linked, { once: true })
    try {
      const value = await this.callHost(name, args, controller)
      const serialized = JSON.stringify({ ok: true, value })
      if (Buffer.byteLength(serialized, 'utf8') > this.config.maxFrameBytes) {
        throw new Error(
          `binding ${name} returned ${Buffer.byteLength(serialized, 'utf8')} bytes, above the ${this.config.maxFrameBytes} byte frame ceiling`,
        )
      }
      await this.send({ op: 'binding-result', callId, ok: true, value })
      this.request.host.observe?.({
        type: 'binding-end',
        callId,
        name,
        ok: true,
        elapsedMs: Date.now() - startedAt,
        bytes: Buffer.byteLength(serialized, 'utf8'),
      })
    } catch (error) {
      const info = describeBindingError(error)
      if (!this.disposed) {
        await this.send({ op: 'binding-result', callId, ok: false, error: info }).catch(() => {})
      }
      this.request.host.observe?.({
        type: 'binding-end',
        callId,
        name,
        ok: false,
        elapsedMs: Date.now() - startedAt,
        bytes: 0,
      })
    } finally {
      this.request.signal.removeEventListener('abort', linked)
      this.bindings.delete(callId)
    }
  }

  private callHost(name: string, args: Record<string, unknown>, controller: AbortController): Promise<unknown> {
    const host: RlmKernelHost = this.request.host
    if (name === 'llm_query') {
      return host.llmQuery(
        {
          prompt: String(args.prompt ?? ''),
          ...(args.model === undefined ? {} : { model: String(args.model) }),
          ...(args.maxTokens === undefined ? {} : { maxTokens: Number(args.maxTokens) }),
        },
        controller.signal,
      )
    }
    if (name === 'rlm_query') {
      const context = (args.context ?? { kind: 'inline', text: '' }) as Parameters<RlmKernelHost['rlmQuery']>[0]['context']
      return host.rlmQuery({ context, query: String(args.query ?? '') }, controller.signal)
    }
    if (name.startsWith('context.')) {
      return host.contextCall(name, args, controller.signal)
    }
    return Promise.reject(new Error(`unknown binding ${JSON.stringify(name)}`))
  }

  // --------------------------------------------------------------------- //
  // lifecycle
  // --------------------------------------------------------------------- //

  private handleExit(code: number | null, signal: NodeJS.Signals | null): void {
    if (this.exitSettled) return
    this.clearLifetime()
    const expected = this.expectedExit
    this.failAllAwaiters(
      new Error(`kernel exited before the expected frame (code=${code ?? 'null'}, signal=${signal ?? 'null'})`),
    )
    this.request.host.observe?.({ type: 'exited', code, signal, expected })
    if (!expected) {
      void this.settlePending({
        stdout: '',
        stdoutTruncated: false,
        elapsedMs: 0,
        error: {
          kind: 'worker-exit',
          message: `kernel exited unexpectedly (code=${code ?? 'null'}, signal=${signal ?? 'null'})`,
          fatal: true,
        },
      })
    }
    if (!this.exitSettled) {
      this.exitSettled = true
      this.resolveExit({ code, signal, expected })
    }
    this.onExit()
  }

  /**
   * Wait for a frame matching `predicate` on the shared read loop.
   *
   * This never reads the stream itself; a second reader would race the main loop
   * for frames. A timeout or spawn failure rejects every outstanding awaiter, so
   * startup can never hang silently.
   */
  private awaitFrame(predicate: (frame: RlmFrame) => boolean, errorMessage: string): Promise<RlmFrame> {
    return new Promise<RlmFrame>((resolveFrame, rejectFrame) => {
      const timer = setTimeout(() => {
        this.removeAwaiter(entry)
        rejectFrame(new Error(`${errorMessage} within the startup budget`))
      }, Math.max(1, this.request.evalTimeoutMs))
      timer.unref?.()
      const entry = {
        predicate,
        resolve: (frame: RlmFrame) => {
          clearTimeout(timer)
          resolveFrame(frame)
        },
        reject: (error: Error) => {
          clearTimeout(timer)
          rejectFrame(error)
        },
      }
      this.awaiters.push(entry)
    })
  }

  private removeAwaiter(entry: (typeof this.awaiters)[number]): void {
    const index = this.awaiters.indexOf(entry)
    if (index >= 0) this.awaiters.splice(index, 1)
  }

  private failAllAwaiters(error: Error): void {
    const awaiters = this.awaiters.splice(0, this.awaiters.length)
    for (const awaiter of awaiters) awaiter.reject(error)
  }

  private send(message: Record<string, unknown>): Promise<void> {
    if (!this.child?.stdin || this.child.stdin.destroyed) {
      return Promise.reject(new Error('kernel stdin is closed'))
    }
    return writeFrame(this.child.stdin, message)
  }

  private clearLifetime(): void {
    if (this.lifetimeTimer) clearTimeout(this.lifetimeTimer)
    this.lifetimeTimer = undefined
  }

  private buildEnvironment(): NodeJS.ProcessEnv {
    // No inherited credentials: the child gets a minimal, explicit environment
    // and a HOME/TMPDIR inside its own private directory.
    const base: NodeJS.ProcessEnv = {
      PATH: process.env.PATH ?? '/usr/local/bin:/usr/bin:/bin',
      HOME: this.workspace,
      TMPDIR: this.workspace,
      LANG: 'C.UTF-8',
      LC_ALL: 'C.UTF-8',
      PYTHONIOENCODING: 'utf-8',
      PYTHONUNBUFFERED: '1',
      PYTHONDONTWRITEBYTECODE: '1',
      PYTHONNOUSERSITE: '1',
    }
    return { ...base, ...(this.config.env ?? {}) }
  }

  /** Terminate the whole process group, escalating only after a grace period. */
  private async terminate(reason: string): Promise<void> {
    const child = this.child
    if (!child || this.exitSettled) return
    this.expectedExit = true
    this.clearLifetime()
    this.request.signal.removeEventListener('abort', this.onAbort)
    for (const binding of this.bindings.values()) binding.controller.abort(reason)
    this.bindings.clear()

    const exited = new Promise<void>((resolveExit) => {
      if (this.exitSettled) {
        resolveExit()
        return
      }
      void this.exited.then(() => resolveExit())
    })

    signalProcessTree(child, 'SIGTERM')
    const grace = new Promise<void>((resolveGrace) => {
      const timer = setTimeout(resolveGrace, this.config.killGraceMs)
      timer.unref?.()
    })
    await Promise.race([exited, grace])
    if (!this.exitSettled) {
      signalProcessTree(child, 'SIGKILL')
      await Promise.race([exited, new Promise<void>((resolveWait) => setTimeout(resolveWait, 250).unref?.())])
    }
  }

  /** Public disposal: terminate, reap, and remove the private workspace. */
  async dispose(reason = 'disposed'): Promise<void> {
    if (this.disposed) {
      await this.exited
      return
    }
    this.disposed = true
    // A shutdown the driver honors is not an "unexpected" death.
    this.expectedExit = true
    // A polite shutdown preserves the driver's own exit path; if the kernel is
    // wedged inside model code, terminate() does the real work.
    await this.send({ op: 'shutdown' }).catch(() => {})
    await this.terminate(reason)
    await this.exited
    // Drain the read loop so no frame handler outlives the kernel.
    await this.readLoop?.catch(() => {})
    await rm(this.workspace, { recursive: true, force: true }).catch(() => {})
  }
}

function signalProcessTree(child: ChildProcess, signal: NodeJS.Signals): void {
  const pid = child.pid
  if (pid === undefined) return
  if (process.platform === 'win32') {
    child.kill(signal)
    return
  }
  try {
    // Negative pid targets the process group, so grandchildren die too.
    process.kill(-pid, signal)
  } catch {
    try {
      child.kill(signal)
    } catch {
      // Already gone; the exit event will settle the handle.
    }
  }
}

function toStepResult(frame: RlmFrame): RlmStepResult {
  const errorPayload = frame.error as
    | { kind?: string; message?: string; traceback?: string }
    | null
    | undefined
  const error: RlmStepError | undefined = errorPayload
    ? {
        kind: normalizeErrorKind(errorPayload.kind),
        message: String(errorPayload.message ?? 'execution failed'),
        ...(errorPayload.traceback ? { traceback: String(errorPayload.traceback) } : {}),
        ...(frame.fatal === true ? { fatal: true } : {}),
      }
    : undefined
  const finalPayload = frame.final as
    | { name?: string; type?: string; preview?: string; truncated?: boolean; chars?: number | null }
    | undefined
  return {
    stdout: String(frame.stdout ?? ''),
    stdoutTruncated: frame.stdoutTruncated === true,
    elapsedMs: Number(frame.elapsedMs ?? 0),
    ...(error ? { error } : {}),
    ...(finalPayload
      ? {
          final: {
            name: String(finalPayload.name ?? 'Final'),
            typeName: String(finalPayload.type ?? 'object'),
            preview: String(finalPayload.preview ?? ''),
            truncated: finalPayload.truncated === true,
            chars: finalPayload.chars === undefined ? null : finalPayload.chars,
          },
        }
      : {}),
  }
}

function normalizeErrorKind(kind: string | undefined): RlmStepError['kind'] {
  switch (kind) {
    case 'syntax':
    case 'name':
    case 'exception':
    case 'timeout':
    case 'cancelled':
    case 'protocol':
    case 'sandbox-unavailable':
    case 'budget':
      return kind
    default:
      return 'exception'
  }
}

function describeBindingError(error: unknown): { kind: string; message: string } {
  if (error instanceof Error) {
    const kind = (error as Error & { kind?: string }).kind
    return { kind: typeof kind === 'string' ? kind : 'binding-error', message: error.message }
  }
  return { kind: 'binding-error', message: String(error) }
}

/** Content hash used by tests and diagnostics to identify a kernel configuration. */
export function fingerprintKernelConfig(config: PythonKernelConfig): string {
  return createHash('sha256')
    .update(JSON.stringify({ pythonPath: config.pythonPath ?? 'python3', driver: config.driverPath ?? 'default' }))
    .digest('hex')
    .slice(0, 16)
}

export default RlmPythonKernelRuntime
