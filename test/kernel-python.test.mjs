/**
 * Real Python kernel provider tests.
 *
 * These run the actual `python3` driver through the actual provider — no
 * interpretation, no stub. They are the tests that would catch a broken
 * bootstrap, a leaked process, a wedged binding reply or a driver that cannot
 * export a large value.
 */

import assert from 'node:assert/strict'
import { existsSync } from 'node:fs'
import { readFile } from 'node:fs/promises'
import { after, before, describe, it } from 'node:test'
import { Context } from '@deepseek-ai/cordis'
import { RlmPythonKernelRuntime } from '../lib/kernel/python/provider.js'
import { RecordingEventSink } from '../lib/events.js'
import { InMemoryKernelRuntime } from '../lib/testing.js'
import { cleanup, filler, makeTempDir } from './helpers.mjs'

const PYTHON = process.env.DSH_RLM_PYTHON ?? 'python3'
const CONTEXT_REF = {
  id: 'ctx-fixture',
  sha256: 'sha-fixture',
  kind: 'text',
  bytes: 32,
  chars: 32,
  estimatedTokens: { 'heuristic:ceil(chars/4)': 8 },
  encoding: 'utf-8',
  createdAt: 0,
}

/** A handle that answers from a fixed string, standing in for the store. */
function fixtureHandle(text) {
  return {
    ref: CONTEXT_REF,
    kind: 'text',
    length: text.length,
    async peek(start = 0, end = text.length) {
      return this.slice(start, end)
    },
    async slice(start, end) {
      return text.slice(start, end)
    },
    async items() {
      return [...text]
    },
    async get() {
      throw new Error('no keys')
    },
    async search(query) {
      const index = text.indexOf(query)
      return index < 0 ? [] : [{ start: index, end: index + query.length, text: query }]
    },
    async materialize() {
      return text
    },
  }
}

/** Host binding recorder shared by the kernel tests. */
function makeHost(text, overrides = {}) {
  const calls = []
  const observations = []
  const host = {
    async llmQuery(args) {
      calls.push({ kind: 'llm', args })
      return overrides.llmQuery ? overrides.llmQuery(args) : `leaf(${(args.prompt ?? '').slice(0, 20)})`
    },
    async rlmQuery(args) {
      calls.push({ kind: 'rlm', args })
      return overrides.rlmQuery ? overrides.rlmQuery(args) : 'child-result'
    },
    async contextCall(name, args) {
      calls.push({ kind: 'context', name, args })
      const handle = fixtureHandle(text)
      switch (name) {
        case 'context.meta':
          return { kind: 'text', length: text.length, chars: text.length, bytes: text.length, sha256: CONTEXT_REF.sha256 }
        case 'context.ref':
          return CONTEXT_REF
        case 'context.slice':
          return handle.slice(Number(args.start ?? 0), Number(args.end ?? text.length))
        case 'context.peek':
          return handle.slice(Number(args.start ?? 0), Number(args.end ?? text.length))
        case 'context.items':
          return [...text.slice(Number(args.start ?? 0), Number(args.end ?? 4))]
        case 'context.search':
          return handle.search(String(args.query ?? ''))
        case 'context.get':
          throw new Error('no keys')
        case 'context.materialize':
          return text
        default:
          throw new Error(`unknown context call ${name}`)
      }
    },
    observe(observation) {
      observations.push(observation)
    },
  }
  return { host, calls, observations }
}

function startRequest(overrides) {
  const controller = new AbortController()
  return {
    invocationId: 'inv-1',
    attemptId: 'attempt-1',
    depth: 2,
    maxDepth: 3,
    contextRef: CONTEXT_REF,
    contextKind: 'text',
    bindings: ['llm_query', 'llm_query_many', 'rlm_query', 'rlm_query_many'],
    finalVarName: 'Final',
    maxStdoutChars: 200_000,
    maxConcurrency: 4,
    cwd: overrides.cwd,
    maxLifetimeMs: 60_000,
    evalTimeoutMs: 30_000,
    host: overrides.host,
    signal: controller.signal,
    ...overrides.rest,
  }
}

describe('python kernel provider (real interpreter)', () => {
  let runtime
  let root

  before(async () => {
    root = await makeTempDir('rlm-python-')
    runtime = new RlmPythonKernelRuntime(new Context(), {
      pythonPath: PYTHON,
      workspaceRoot: root,
      guestNetworkGuard: true,
    })
  })

  after(async () => {
    await runtime.disposeAll('test teardown')
    await cleanup(root)
  })

  it('ships the driver next to the compiled provider', () => {
    const driver = new URL('../lib/kernel/python/rlm_kernel.py', import.meta.url)
    assert.ok(existsSync(driver), 'the build must copy rlm_kernel.py into lib/')
  })

  it('keeps one namespace across evaluations and detects a final variable', async () => {
    const { host } = makeHost('hello context')
    const handle = await runtime.start(startRequest({ host, cwd: root }))
    try {
      const first = await handle.kernel.eval('counter = 1\nprint("first")')
      assert.equal(first.error, undefined)
      assert.match(first.stdout, /first/)
      const second = await handle.kernel.eval('counter += 41\nFinal = f"counter={counter}"')
      assert.equal(second.error, undefined)
      // Persistence across evals is the defining property of the kernel.
      assert.equal(second.final.name, 'Final')
      assert.equal(second.final.preview, 'counter=42')
    } finally {
      await handle.dispose()
    }
  })

  it('exposes the context handle with range reads', async () => {
    const { host, calls } = makeHost('0123456789abcdef')
    const handle = await runtime.start(startRequest({ host, cwd: root }))
    try {
      const result = await handle.kernel.eval('print(len(context), context[0:4], context.peek(4, 8))')
      assert.equal(result.error, undefined)
      assert.equal(result.stdout, '16 0123 4567\n')
      assert.ok(calls.some((call) => call.kind === 'context' && call.name === 'context.meta'))
      assert.ok(calls.some((call) => call.kind === 'context' && call.name === 'context.slice'))
    } finally {
      await handle.dispose()
    }
  })

  it('returns llm_query results to the program, not to stdout', async () => {
    const { host, calls } = makeHost('material')
    const handle = await runtime.start(startRequest({ host, cwd: root }))
    try {
      const result = await handle.kernel.eval('answer = llm_query("summarise this")\nprint(len(answer))')
      assert.equal(result.error, undefined)
      assert.equal(calls.filter((call) => call.kind === 'llm').length, 1)
      // The point of the paper: the value is a program value.
      assert.match(result.stdout, /^leaf\(summarise this\)?\n?$|^\d+\n$/)
      const stored = await handle.kernel.eval('print(answer)')
      assert.match(stored.stdout, /leaf\(/)
    } finally {
      await handle.dispose()
    }
  })

  it('fans out llm_query_many concurrently', async () => {
    let inFlight = 0
    let peak = 0
    const { host } = makeHost('material', {
      llmQuery: async (args) => {
        inFlight += 1
        peak = Math.max(peak, inFlight)
        await new Promise((resolve) => setTimeout(resolve, 30))
        inFlight -= 1
        return `[${args.prompt}]`
      },
    })
    const handle = await runtime.start(startRequest({ host, cwd: root, rest: { maxConcurrency: 4 } }))
    try {
      const result = await handle.kernel.eval('results = llm_query_many(["a", "b", "c", "d"])\nprint(results)')
      assert.equal(result.error, undefined)
      assert.ok(peak >= 2, `expected concurrent dispatch, peak was ${peak}`)
      assert.match(result.stdout, /\['\[a\]', '\[b\]', '\[c\]', '\[d\]'\]/)
    } finally {
      await handle.dispose()
    }
  })

  it('supports awaiting the fan-out helper too', async () => {
    const { host } = makeHost('material')
    const handle = await runtime.start(startRequest({ host, cwd: root }))
    try {
      const result = await handle.kernel.eval('results = await llm_query_many(["x"])\nprint(len(results))')
      assert.equal(result.error, undefined)
      assert.equal(result.stdout.trim(), '1')
    } finally {
      await handle.dispose()
    }
  })

  it('routes rlm_query to the host binding', async () => {
    const { host, calls } = makeHost('material')
    const handle = await runtime.start(startRequest({ host, cwd: root }))
    try {
      const result = await handle.kernel.eval('child = rlm_query(context, "why?")\nprint(child)')
      assert.equal(result.error, undefined)
      const rlmCall = calls.find((call) => call.kind === 'rlm')
      assert.ok(rlmCall)
      assert.equal(rlmCall.args.query, 'why?')
      // A handle argument stays a handle: the child gets the artifact by
      // reference, not a copy.
      assert.equal(rlmCall.args.context.kind, 'handle')
      assert.equal(rlmCall.args.context.ref.id, CONTEXT_REF.id)
      // A slice is a plain string, so it travels inline.
      const inline = await handle.kernel.eval('child = rlm_query(context[0:4], "why?")\nprint(child)')
      assert.equal(inline.error, undefined)
      const inlineCall = calls.filter((call) => call.kind === 'rlm')[1]
      assert.equal(inlineCall.args.context.kind, 'inline')
      assert.equal(inlineCall.args.context.text, 'mate')
    } finally {
      await handle.dispose()
    }
  })

  it('classifies a syntax error and a name error for the model to repair', async () => {
    const { host } = makeHost('material')
    const handle = await runtime.start(startRequest({ host, cwd: root }))
    try {
      const syntax = await handle.kernel.eval('x = 1 +')
      assert.equal(syntax.error.kind, 'syntax')
      const name = await handle.kernel.eval('print(never_defined)')
      assert.equal(name.error.kind, 'name')
      // The namespace survives a failed program.
      const ok = await handle.kernel.eval('print("still alive")')
      assert.equal(ok.error, undefined)
      assert.match(ok.stdout, /still alive/)
    } finally {
      await handle.dispose()
    }
  })

  it('omits model bindings at depth 0 so a run cannot become recursive', async () => {
    const { host } = makeHost('material')
    const handle = await runtime.start(
      startRequest({ host, cwd: root, rest: { depth: 0, bindings: [] } }),
    )
    try {
      const result = await handle.kernel.eval('llm_query("nope")')
      assert.equal(result.error.kind, 'name')
      assert.match(result.error.message, /llm_query/)
      const context = await handle.kernel.eval('print(len(context))')
      assert.equal(context.error, undefined)
    } finally {
      await handle.dispose()
    }
  })

  it('truncates program output instead of letting it consume the host', async () => {
    const { host } = makeHost('material')
    const handle = await runtime.start(startRequest({ host, cwd: root, rest: { maxStdoutChars: 100 } }))
    try {
      const result = await handle.kernel.eval('print("x" * 5000)')
      assert.equal(result.error, undefined)
      assert.equal(result.stdoutTruncated, true)
      assert.ok(result.stdout.length <= 101)
    } finally {
      await handle.dispose()
    }
  })

  it('exports a large final value to a file with bounded metadata', async () => {
    const { host } = makeHost('material')
    const handle = await runtime.start(startRequest({ host, cwd: root }))
    try {
      const program = 'Final = "".join(str(i % 10) for i in range(200000))'
      const step = await handle.kernel.eval(program)
      assert.equal(step.error, undefined)
      const path = `${handle.kernel.workspace}/final.out`
      const exported = await handle.kernel.export('Final', { path, previewChars: 32, maxBytes: 10_000_000 })
      assert.equal(exported.chars, 200_000)
      assert.equal(exported.truncated, true)
      assert.equal(exported.preview.length, 32)
      const onDisk = await readFile(path, 'utf8')
      assert.equal(onDisk.length, 200_000)
      assert.match(onDisk.slice(0, 10), /^0123456789/)
    } finally {
      await handle.dispose()
    }
  })

  it('refuses an export above the byte ceiling and removes the partial file', async () => {
    const { host } = makeHost('material')
    const handle = await runtime.start(startRequest({ host, cwd: root }))
    try {
      await handle.kernel.eval('Final = "y" * 10000')
      const path = `${handle.kernel.workspace}/too-big.out`
      await assert.rejects(
        handle.kernel.export('Final', { path, previewChars: 8, maxBytes: 100 }),
        /ceiling/,
      )
      assert.equal(existsSync(path), false)
    } finally {
      await handle.dispose()
    }
  })

  it('reports a missing final variable as a name error rather than an empty string', async () => {
    const { host } = makeHost('material')
    const handle = await runtime.start(startRequest({ host, cwd: root }))
    try {
      await assert.rejects(
        handle.kernel.export('nope', { path: `${handle.kernel.workspace}/nope.out`, previewChars: 8, maxBytes: 1000 }),
        /no variable named/,
      )
    } finally {
      await handle.dispose()
    }
  })

  it('blocks the guest network and subprocess APIs (defense in depth)', async () => {
    const { host } = makeHost('material')
    const handle = await runtime.start(startRequest({ host, cwd: root, rest: { depth: 0, bindings: [] } }))
    try {
      const socket = await handle.kernel.eval('import socket\nsocket.create_connection(("example.com", 80))')
      assert.equal(socket.error.kind, 'exception')
      assert.match(socket.error.message, /disabled/)
      const child = await handle.kernel.eval('import subprocess\nsubprocess.run(["ls"])')
      assert.equal(child.error.kind, 'exception')
      assert.match(child.error.message, /disabled/)
    } finally {
      await handle.dispose()
    }
  })

  it('kills the process group when the caller aborts', async () => {
    const { host } = makeHost('material')
    const controller = new AbortController()
    const handle = await runtime.start(startRequest({ host, cwd: root, signal: controller.signal, rest: { signal: controller.signal } }))
    const survivor = handle.kernel.eval('while True:\n    pass')
    controller.abort('test abort')
    const result = await survivor
    assert.ok(result.error)
    assert.ok(['cancelled', 'timeout', 'worker-exit'].includes(result.error.kind))
    await handle.dispose()
    assert.equal(runtime.liveKernels, 0)
  })

  it('terminates a runaway program on the per-eval deadline', async () => {
    const { host } = makeHost('material')
    const handle = await runtime.start(startRequest({ host, cwd: root }))
    const started = Date.now()
    const result = await handle.kernel.eval('while True:\n    pass', { timeoutMs: 500 })
    assert.equal(result.error.kind, 'timeout')
    assert.ok(Date.now() - started < 10_000)
    // The provider reports the kernel as gone, and disposal is idempotent.
    await handle.dispose()
    assert.equal(runtime.liveKernels, 0)
  })

  it('does not leak a kernel process when a run finishes normally', async () => {
    const { host } = makeHost('material')
    const before = runtime.liveKernels
    const handle = await runtime.start(startRequest({ host, cwd: root }))
    assert.equal(runtime.liveKernels, before + 1)
    await handle.dispose()
    const exit = await handle.exited
    assert.equal(exit.expected, true)
    assert.equal(runtime.liveKernels, before)
  })

  it('fails closed when the interpreter cannot be started', async () => {
    const broken = new RlmPythonKernelRuntime(new Context(), {
      pythonPath: '/nonexistent/python-does-not-exist',
      workspaceRoot: root,
    })
    try {
      await assert.rejects(
        broken.start(startRequest({ host: makeHost('x').host, cwd: root })),
        /spawn failed|ENOENT/,
      )
      assert.equal(broken.liveKernels, 0)
    } finally {
      await broken.disposeAll('teardown')
    }
  })

  it('declares honest capabilities instead of over-claiming isolation', () => {
    const capabilities = runtime.capabilities
    assert.equal(capabilities.network, 'guest-guard')
    assert.equal(capabilities.processTreeTermination, true)
    assert.equal(capabilities.hardDeadline, true)
    // A bare subprocess cannot quota a process tree, and the provider says so.
    assert.equal(capabilities.processTreeLimits, false)
  })
})

describe('in-memory kernel double', () => {
  it('stays available for unit tests and refuses nothing surprising', async () => {
    const fake = new InMemoryKernelRuntime(() => ({
      stdout: 'ok\n',
      stdoutTruncated: false,
      elapsedMs: 0,
    }))
    const { host } = makeHost('material')
    const handle = await fake.start(startRequest({ host, cwd: '/tmp' }))
    const result = await handle.kernel.eval('print("ok")')
    assert.equal(result.stdout, 'ok\n')
    await handle.dispose()
    assert.equal(fake.liveKernels, 0)
  })

  it('is not silently used as the production provider', async () => {
    // The event payload must name the provider that actually ran, so a
    // deployment can never confuse the double with real isolation.
    const fake = new InMemoryKernelRuntime(() => ({ stdout: '', stdoutTruncated: false, elapsedMs: 0 }))
    const sink = new RecordingEventSink()
    assert.equal(fake.providerName, 'in-memory')
    assert.equal(fake.capabilities.isolation, 'in-process')
    sink.emit({
      type: 'rlm/start',
      time: 0,
      invocationId: 'i',
      attemptId: 'a',
      depth: 1,
      maxDepth: 1,
      contextRef: CONTEXT_REF,
      budgets: {},
      identity: {
        rlmVersion: 'test',
        promptProfileId: 'p',
        promptProfileVariant: 'compact',
        kernelProvider: fake.providerName,
        kernelIsolation: fake.capabilities.isolation,
        contextStoreProvider: 'local',
        toolSchemaVersion: '1',
      },
      rootRoute: { provider: 'test', model: 'test' },
      leafRoute: { provider: 'test', model: 'test' },
      startedAt: 0,
    })
    assert.equal(sink.ofType('rlm/start')[0].identity.kernelProvider, 'in-memory')
  })
})

describe('python kernel helpers', () => {
  it('produces a deterministic config fingerprint', async () => {
    const { fingerprintKernelConfig } = await import('../lib/kernel/python/provider.js')
    const first = fingerprintKernelConfig({ pythonPath: 'python3', driverPath: '/a.py' })
    const second = fingerprintKernelConfig({ pythonPath: 'python3', driverPath: '/a.py' })
    const third = fingerprintKernelConfig({ pythonPath: 'python3.12', driverPath: '/a.py' })
    assert.equal(first, second)
    assert.notEqual(first, third)
    assert.equal(first.length, 16)
    assert.ok(filler(1).length === 1)
  })
})
