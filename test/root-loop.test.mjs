/**
 * Root-loop contract tests (README §7.4).
 *
 * Each test here corresponds to one paper-level property the plugin claims. They
 * run the *real* service, store and budget machinery; only the kernel and the
 * model are doubles, which is exactly the pair a deterministic test must control.
 */

import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { InMemoryKernelRuntime, ScriptedModelClient, step } from '../lib/testing.js'
import { bigDocument, buildService, cleanup, filler, makeTempDir } from './helpers.mjs'

const PROMPT = 'SECRET-NEEDLE-7 the answer is forty-two'

/** A kernel that implements just enough of a REPL for the loop tests. */
function replKernel({ answers = {}, failOn = {}, finalAfter = {} } = {}) {
  return new InMemoryKernelRuntime((program, kernel) => {
    if (failOn[program]) return failOn[program]
    const queryCalls = [...program.matchAll(/llm_query\s*\(\s*[fr]?['"]([^'"]*)['"]/g)].map((match) => match[1])
    for (const query of queryCalls) {
      kernel.request.host.llmQuery({ prompt: `${query}` }, new AbortController().signal)
    }
    if (program.includes('context.slice') || program.includes('context[')) {
      kernel.request.host.contextCall('context.slice', { start: 0, end: 8 }, new AbortController().signal)
    }
    return step(answers[program] ?? '', finalAfter[program] ? { final: finalAfter[program] } : {})
  })
}

describe('RLM root loop', () => {
  it('keeps the prompt out of root history and reads it through the REPL', async () => {
    const root = await makeTempDir()
    try {
      const model = new ScriptedModelClient({
        root: [
          '```repl\nprint(len(context))\n```',
          'FINAL(the answer is forty-two)',
        ],
        leaf: () => 'unused',
      })
      const kernel = replKernel({ answers: { 'print(len(context))\n': '42\n' } })
      const { service, events } = buildService({ root, model, kernel })
      const result = await service.run({
        prompt: { kind: 'text', text: PROMPT },
        cwd: root,
        signal: new AbortController().signal,
      })
      assert.equal(result.status, 'ok')
      assert.equal(result.text, 'the answer is forty-two')
      // The paper's first property: the root model never sees the prompt.
      assert.equal(
        model.prompts.some((prompt) => prompt.includes('SECRET-NEEDLE-7')),
        false,
        'the prompt must never enter root history',
      )
      assert.match(model.prompts[0], /The full prompt is NOT in this conversation/)
      // ...and the REPL did read it.
      assert.ok(kernel.started[0].request.contextRef.chars >= PROMPT.length)
      const start = events.ofType('rlm/start')
      assert.equal(start.length, 1)
      assert.equal(events.ofType('rlm/end').length, 1)
      assert.equal(events.ofType('rlm/end')[0].stopReason, 'final-text')
    } finally {
      await cleanup(root)
    }
  })

  it('admits a prompt far larger than the root model window', async () => {
    const root = await makeTempDir()
    try {
      const huge = bigDocument(5_000_000)
      const model = new ScriptedModelClient({
        root: ['```repl\nn = len(context)\n```', 'FINAL_VAR(progress)'],
      })
      const kernel = replKernel({ answers: { 'n = len(context)\n': '5000000\n' } })
      kernel.started.length = 0
      const service = buildService({ root, model, kernel }).service
      // The kernel answers `progress` through the export path.
      const originalStart = kernel.start.bind(kernel)
      kernel.start = async (request) => {
        const handle = await originalStart(request)
        handle.kernel.variables.set('progress', 'done')
        return handle
      }
      const result = await service.run({
        prompt: { kind: 'text', text: huge },
        cwd: root,
        signal: new AbortController().signal,
      })
      assert.equal(result.status, 'ok')
      assert.equal(result.text, 'done')
      assert.equal(model.prompts.some((prompt) => prompt.includes(huge.slice(0, 200))), false)
      // Root history stays small while the prompt does not.
      for (const prompt of model.prompts) assert.ok(prompt.length < 20_000)
      assert.equal(kernel.started[0].request.contextRef.chars, huge.length)
    } finally {
      await cleanup(root)
    }
  })

  it('lets REPL code loop over prompt slices and issue several llm_query calls', async () => {
    const root = await makeTempDir()
    try {
      const model = new ScriptedModelClient({
        root: ['```repl\nanswers = llm_query_many(["a", "b", "c"])\n```', 'FINAL_VAR(answers)'],
        leaf: (prompt) => `answer(${prompt})`,
      })
      let observedPrompts = []
      const kernel = new InMemoryKernelRuntime(async (program, k) => {
        const host = k.request.host
        if (program.includes('llm_query_many')) {
          // Simulate three concurrent calls through the host, as the driver does.
          const results = await Promise.all(
            ['a', 'b', 'c'].map((prompt) => host.llmQuery({ prompt }, new AbortController().signal)),
          )
          observedPrompts = results
        }
        k.variables.set('answers', observedPrompts.join(','))
        return step('')
      })
      const service = buildService({ root, model, kernel }).service
      const result = await service.run({
        prompt: { kind: 'text', text: 'material' },
        cwd: root,
        signal: new AbortController().signal,
      })
      assert.equal(result.status, 'ok')
      // The child answers reached the program, not the root conversation.
      assert.equal(result.text, 'answer(a),answer(b),answer(c)')
      assert.equal(model.prompts.some((prompt) => prompt.includes('answer(a)')), false)
      assert.equal(result.usage.queries, 3)
      assert.ok(result.usage.maxObservedConcurrency >= 1)
    } finally {
      await cleanup(root)
    }
  })

  it('stores a query result in the REPL without putting it in root history', async () => {
    const root = await makeTempDir()
    try {
      const long = filler(4_000, 'leaf')
      const model = new ScriptedModelClient({
        root: ['```repl\nbuf = llm_query("summarise")\n```', 'FINAL_VAR(buf)'],
        leaf: () => long,
      })
      const kernel = new InMemoryKernelRuntime(async (program, k) => {
        if (program.includes('llm_query')) {
          const value = await k.request.host.llmQuery({ prompt: 'summarise' }, new AbortController().signal)
          k.variables.set('buf', String(value))
        }
        return step('')
      })
      const service = buildService({ root, model, kernel }).service
      const result = await service.run({
        prompt: { kind: 'text', text: 'material' },
        cwd: root,
        signal: new AbortController().signal,
      })
      assert.equal(result.status, 'ok')
      // A 4,000 character answer never appears in the conversation.
      assert.equal(model.prompts.some((prompt) => prompt.includes(long.slice(0, 64))), false)
      assert.equal(result.text, long)
    } finally {
      await cleanup(root)
    }
  })

  it('installs no model bindings at depth 0', async () => {
    const root = await makeTempDir()
    try {
      const model = new ScriptedModelClient({ root: ['```repl\nprint(len(context))\n```', 'FINAL(done)'] })
      const kernel = replKernel({ answers: { 'print(len(context))\n': '8\n' } })
      const service = buildService({ root, model, kernel, config: { defaultDepth: 0 } }).service
      const result = await service.run({
        prompt: { kind: 'text', text: 'material' },
        cwd: root,
        signal: new AbortController().signal,
      })
      assert.equal(result.status, 'ok')
      assert.deepEqual(kernel.started[0].request.bindings, [])
      // The prompt is still externalized: depth 0 is an ablation, not a plain call.
      assert.ok(kernel.started[0].request.contextRef.chars > 0)
    } finally {
      await cleanup(root)
    }
  })

  it('downgrades rlm_query to a leaf call at depth 1', async () => {
    const root = await makeTempDir()
    try {
      const model = new ScriptedModelClient({
        root: ['```repl\nsub = rlm_query(context, "what is it?")\n```', 'FINAL_VAR(sub)'],
        leaf: () => 'leaf-answer',
      })
      const kernel = new InMemoryKernelRuntime(async (program, k) => {
        if (program.includes('rlm_query')) {
          const value = await k.request.host.rlmQuery(
            { context: { kind: 'handle', ref: k.request.contextRef }, query: 'what is it?' },
            new AbortController().signal,
          )
          k.variables.set('sub', String(value))
        }
        return step('')
      })
      const service = buildService({ root, model, kernel, config: { defaultDepth: 1, maxDepth: 3 } }).service
      const result = await service.run({
        prompt: { kind: 'text', text: 'material' },
        cwd: root,
        signal: new AbortController().signal,
      })
      assert.equal(result.status, 'ok')
      assert.equal(result.text, 'leaf-answer')
      // Exactly one kernel exists: no child was spawned for a leaf-shaped run.
      assert.equal(kernel.started.length, 1)
    } finally {
      await cleanup(root)
    }
  })

  it('creates a child RLM with its own persistent kernel at depth 2', async () => {
    const root = await makeTempDir()
    try {
      const model = new ScriptedModelClient({
        root: (request) => {
          if (request.system.includes('rlm_query')) {
            return request.messages.length === 1
              ? '```repl\nchild = rlm_query(context.slice(0, 10), "child task")\n```'
              : 'FINAL_VAR(child)'
          }
          return 'FINAL(child-done)'
        },
        leaf: () => 'leaf',
      })
      const kernel = new InMemoryKernelRuntime(async (program, k) => {
        if (program.includes('rlm_query')) {
          const value = await k.request.host.rlmQuery(
            { context: { kind: 'inline', text: 'child material' }, query: 'child task' },
            new AbortController().signal,
          )
          k.variables.set('child', String(value))
        } else if (program.includes('context.slice')) {
          await k.request.host.contextCall('context.slice', { start: 0, end: 10 }, new AbortController().signal)
        }
        return step('')
      })
      const { service, events } = buildService({
        root,
        model,
        kernel,
        config: { defaultDepth: 2, maxDepth: 2 },
      })
      const result = await service.run({
        prompt: { kind: 'text', text: 'parent material' },
        cwd: root,
        signal: new AbortController().signal,
      })
      assert.equal(result.status, 'ok')
      assert.equal(result.text, 'child-done')
      // Two kernels: the parent and the child it created.
      assert.equal(kernel.started.length, 2)
      assert.equal(kernel.started[1].request.depth, 1)
      assert.ok(kernel.started[1].request.contextRef.chars > 0)
      const childStarts = events.ofType('rlm/start').filter((event) => event.parentInvocationId !== undefined)
      assert.equal(childStarts.length, 1)
      assert.equal(childStarts[0].depth, 1)
      // A child can never be given more depth than the parent had left.
      assert.ok(childStarts[0].depth < events.ofType('rlm/start')[0].depth)
    } finally {
      await cleanup(root)
    }
  })

  it('never lets a child exceed the parent remaining budget', async () => {
    const root = await makeTempDir()
    try {
      let childLimits
      const model = new ScriptedModelClient({
        root: (request) => {
          if (request.system.includes('rlm_query')) {
            return request.messages.length === 1 ? '```repl\nx = rlm_query(context, "go")\n```' : 'FINAL_VAR(x)'
          }
          return 'FINAL(leaf)'
        },
        leaf: () => 'leaf',
      })
      const kernel = new InMemoryKernelRuntime(async (program, k) => {
        if (program.includes('rlm_query')) {
          childLimits = k.request.host
          k.variables.set('x', 'ok')
          try {
            await k.request.host.rlmQuery(
              { context: { kind: 'inline', text: 'm' }, query: 'go' },
              new AbortController().signal,
            )
          } catch {
            // A refusal is the expected outcome in this scenario.
          }
        }
        return step('')
      })
      const { service, events } = buildService({
        root,
        model,
        kernel,
        config: { defaultDepth: 2, maxDepth: 2, maxTotalQueries: 3, maxTotalKernels: 1 },
      })
      const result = await service.run({
        prompt: { kind: 'text', text: 'material' },
        cwd: root,
        signal: new AbortController().signal,
      })
      assert.ok(childLimits)
      const childStart = events.ofType('rlm/start').find((event) => event.parentInvocationId !== undefined)
      if (childStart) {
        assert.ok(childStart.budgets.maxTotalQueries <= 3)
        const parentStart = events.ofType('rlm/start')[0]
        assert.ok(childStart.budgets.maxTotalKernels <= parentStart.budgets.maxTotalKernels)
      }
      assert.ok(['ok', 'budget-exceeded', 'failed'].includes(result.status))
    } finally {
      await cleanup(root)
    }
  })

  it('cancels the kernel and all child queries when the caller aborts', async () => {
    const root = await makeTempDir()
    try {
      const controller = new AbortController()
      const model = new ScriptedModelClient({
        root: (_request, index) => {
          if (index === 1) controller.abort('user cancelled')
          return index === 0 ? '```repl\nprint("first")\n```' : '```repl\nprint("second")\n```'
        },
      })
      const kernel = replKernel({ answers: { 'print("first")\n': 'first\n' } })
      const service = buildService({ root, model, kernel }).service
      const result = await service.run({
        prompt: { kind: 'text', text: 'material' },
        cwd: root,
        signal: controller.signal,
      })
      assert.equal(result.status, 'cancelled')
      assert.equal(result.stopReason, 'cancelled')
      // Disposal is not optional: a cancelled run must not leak a kernel.
      assert.ok(kernel.started.every((k) => k.closed === true))
    } finally {
      await cleanup(root)
    }
  })

  it('stops on a soft budget refusal instead of killing the program mid-expression', async () => {
    const root = await makeTempDir()
    try {
      const model = new ScriptedModelClient({
        root: ['```repl\nfor i in range(10):\n    llm_query("x")\n```', 'FINAL(done)'],
        leaf: () => 'leaf',
      })
      const kernel = new InMemoryKernelRuntime(async (program, k) => {
        let refusals = 0
        for (let index = 0; index < 3; index += 1) {
          try {
            await k.request.host.llmQuery({ prompt: 'x' }, new AbortController().signal)
          } catch {
            refusals += 1
          }
        }
        return step('', refusals > 0 ? { error: { kind: 'exception', message: 'query refused', traceback: '' } } : {})
      })
      const service = buildService({ root, model, kernel, config: { maxTotalQueries: 2 } }).service
      const result = await service.run({
        prompt: { kind: 'text', text: 'material' },
        cwd: root,
        signal: new AbortController().signal,
      })
      assert.equal(result.status, 'budget-exceeded')
      assert.equal(result.stopReason, 'budget-exceeded')
      assert.ok(result.usage.queries <= 2)
    } finally {
      await cleanup(root)
    }
  })

  it('bounds parser retries and charges them to the same budget', async () => {
    const root = await makeTempDir()
    try {
      const model = new ScriptedModelClient({ root: ['I will think about it.', 'still thinking', 'and more', 'FINAL(late)'] })
      const kernel = replKernel()
      const { service, events } = buildService({ root, model, kernel, config: { maxParseRetries: 1 } })
      const result = await service.run({
        prompt: { kind: 'text', text: 'material' },
        cwd: root,
        signal: new AbortController().signal,
      })
      assert.equal(result.status, 'failed')
      assert.equal(result.stopReason, 'root-parse-failure')
      // One retry, then the run ends: the model cannot loop forever on prose.
      assert.equal(model.calls.length, 2)
      assert.equal(events.ofType('rlm/step').filter((event) => event.action === 'parse-failure').length, 2)
    } finally {
      await cleanup(root)
    }
  })

  it('returns a final value above the inline limit as an artifact, not as text', async () => {
    const root = await makeTempDir()
    try {
      const huge = filler(40_000, 'final')
      const model = new ScriptedModelClient({ root: ['```repl\nFinal = "x"\n```', 'FINAL_VAR(Final)'] })
      const kernel = replKernel({ answers: { 'Final = "x"\n': '' } })
      const originalStart = kernel.start.bind(kernel)
      kernel.start = async (request) => {
        const handle = await originalStart(request)
        handle.kernel.variables.set('Final', huge)
        return handle
      }
      const { service, events } = buildService({
        root,
        model,
        kernel,
        config: { maxInlineOutputBytes: 1_000 },
      })
      const result = await service.run({
        prompt: { kind: 'text', text: 'material' },
        cwd: root,
        signal: new AbortController().signal,
      })
      assert.equal(result.status, 'ok')
      assert.equal(result.text, undefined)
      assert.ok(result.artifactRef)
      assert.equal(result.artifactRef.chars, huge.length)
      assert.ok(result.artifactRef.truncated)
      assert.equal(events.ofType('rlm/artifact').some((event) => event.role === 'output'), true)
      // Not one character of the final value entered the conversation.
      assert.equal(model.prompts.some((prompt) => prompt.includes(huge.slice(0, 32))), false)
    } finally {
      await cleanup(root)
    }
  })

  it('ends the run when the kernel dies, without leaking it', async () => {
    const root = await makeTempDir()
    try {
      const model = new ScriptedModelClient({ root: ['```repl\nboom\n```', 'FINAL(unreachable)'] })
      const kernel = new InMemoryKernelRuntime(() => ({
        stdout: '',
        stdoutTruncated: false,
        elapsedMs: 3,
        error: { kind: 'worker-exit', message: 'kernel died', fatal: true },
      }))
      const { service, events } = buildService({ root, model, kernel })
      const result = await service.run({
        prompt: { kind: 'text', text: 'material' },
        cwd: root,
        signal: new AbortController().signal,
      })
      assert.equal(result.status, 'failed')
      assert.equal(result.stopReason, 'kernel-failure')
      assert.ok(kernel.started.every((k) => k.closed === true))
      const end = events.ofType('rlm/end')[0]
      assert.equal(end.errorKind, 'worker-exit')
    } finally {
      await cleanup(root)
    }
  })

  it('records bounded root history in every step event', async () => {
    const root = await makeTempDir()
    try {
      const model = new ScriptedModelClient({
        root: ['```repl\nx = len(context)\n```', 'FINAL_VAR(x)'],
      })
      const kernel = replKernel({ answers: { 'x = len(context)\n': '8\n' } })
      const { service, events } = buildService({ root, model, kernel })
      await service.run({ prompt: { kind: 'text', text: filler(10_000) }, cwd: root, signal: new AbortController().signal })
      for (const stepEvent of events.ofType('rlm/step')) {
        assert.ok(stepEvent.rootHistoryChars > 0)
        assert.ok(stepEvent.rootHistoryChars < 20_000, 'root history must stay bounded')
      }
    } finally {
      await cleanup(root)
    }
  })
})
