/**
 * Plugin-level end-to-end tests.
 *
 * This is the test that answers "is it actually a dsh plugin?": a real cordis
 * context, the real `rlm-context-local` and `rlm-kernel-python` providers, the
 * real `rlm` plugin, the real `run_rlm` tool definition, a real Python REPL — and
 * only the *model* faked, through the real `ctx.llm.stream` seam.
 *
 * It also pins the load/unload contract: services appear together, and unloading
 * the plugin terminates every kernel process rather than orphaning it.
 */

import assert from 'node:assert/strict'
import { existsSync } from 'node:fs'
import { readdir } from 'node:fs/promises'
import { after, before, describe, it } from 'node:test'
import { Context } from '@deepseek-ai/cordis'
import * as contextLocalPlugin from '../lib/plugins/context-local.js'
import * as kernelPythonPlugin from '../lib/plugins/kernel-python.js'
import * as rlmPlugin from '../lib/index.js'
import { cleanup, makeTempDir } from './helpers.mjs'

const PROMPT = 'CONFIDENTIAL-MATERIAL the magic number is 42'

/**
 * A stub `ctx.llm` speaking the real streaming contract.
 *
 * One stable service object reads a mutable script, because the plugin captures
 * `ctx.llm` when it loads — replacing the service later would not reach it, and
 * faking that would test the wrong thing.
 */
function createLlmStub() {
  const state = {
    script: { root: () => 'FINAL(unused)', leaf: () => 'unused' },
    rootIndex: 0,
    rootPrompts: [],
    leafPrompts: [],
  }
  const setScript = (script) => {
    state.script = script
    state.rootIndex = 0
    state.rootPrompts.length = 0
    state.leafPrompts.length = 0
  }
  const service = {
    stream(request) {
      const isLeaf = request.system === undefined
      const prompt = request.messages
        .map((message) => (message.content ?? []).map((block) => block.text ?? '').join(''))
        .join('\n')
      const text = isLeaf ? state.script.leaf(prompt) : state.script.root(state.rootIndex++, prompt)
      if (isLeaf) state.leafPrompts.push(prompt)
      else state.rootPrompts.push(prompt)
      return (async function* generate() {
        yield { type: 'block-start', index: 0, blockType: 'text' }
        if (text.length > 0) yield { type: 'text-delta', index: 0, text }
        yield {
          type: 'usage',
          usage: { inputTokens: isLeaf ? 21 : 11, outputTokens: isLeaf ? 13 : 7, totalTokens: 41 },
        }
        yield { type: 'finish', reason: { kind: 'stop' } }
      })()
    },
  }
  return { service, state, setScript }
}

/** Minimal stand-ins for the harness services the plugin injects. */
function createToolsStub() {
  const tools = new Map()
  return {
    tools,
    service: {
      register(definition) {
        tools.set(definition.name, definition)
        return () => tools.delete(definition.name)
      },
      get(name) {
        return tools.get(name)
      },
    },
  }
}

function createSystemPromptStub() {
  const sections = new Map()
  return {
    sections,
    service: {
      section(section) {
        sections.set(section.name, section)
        return () => sections.delete(section.name)
      },
      getSectionOrder() {
        return 1_000
      },
    },
  }
}

/** A fake tool execution context, as the tool runtime would build it. */
function toolExec({ cwd, callId = 'call-1', signal = new AbortController().signal }) {
  return {
    callId,
    name: 'run_rlm',
    arguments: {},
    signal,
    rootCallId: callId,
    token: Symbol('token'),
    deferContext() {},
    concludeTurn() {},
    agent: {
      id: 'session-test',
      session: { header: { cwd }, append: () => undefined },
    },
  }
}

describe('dsh plugin end-to-end (real cordis, real python kernel)', () => {
  let ctx
  let root
  let tools
  let sections
  let llm

  before(async () => {
    root = await makeTempDir('rlm-plugin-')
    ctx = new Context()
    const toolsStub = createToolsStub()
    const promptStub = createSystemPromptStub()
    tools = toolsStub.tools
    sections = promptStub.sections
    ctx.provide('tools', toolsStub.service)
    ctx.provide('systemPrompt', promptStub.service)
    llm = createLlmStub()
    ctx.provide('llm', llm.service)

    // The two capability providers mount first; the core plugin stays PENDING
    // until both exist, exactly as cordis promises.
    ctx.plugin(contextLocalPlugin, { root: `${root}/artifacts`, charIndexStep: 2048 })
    ctx.plugin(kernelPythonPlugin, { pythonPath: process.env.DSH_RLM_PYTHON ?? 'python3' })
    await settle()
    ctx.plugin(rlmPlugin, {
      rootProvider: 'stub',
      rootModel: 'root-model',
      leafProvider: 'stub',
      leafModel: 'leaf-model',
      defaultDepth: 1,
      maxDepth: 2,
      contextStoreRoot: `${root}/artifacts`,
      maxStdoutChars: 1000,
    })
    await settle()
  })

  after(async () => {
    await ctx.fiber.dispose()
    await cleanup(root)
  })

  it('mounts every service and contribution together', () => {
    assert.ok(ctx.rlmContextStore, 'ctx.rlmContextStore must be mounted')
    assert.ok(ctx.rlmKernelRuntime, 'ctx.rlmKernelRuntime must be mounted')
    assert.ok(ctx.rlm, 'ctx.rlm must be mounted')
    assert.equal(ctx.rlmContextStore.providerName, 'local-content-addressed')
    assert.equal(ctx.rlmKernelRuntime.providerName, 'python-subprocess')
    assert.ok(tools.has('run_rlm'), 'the run_rlm tool must be registered')
    assert.ok(sections.has('tool:run_rlm'), 'the usage-policy prompt section must be registered')
  })

  it('describes the tool so it is not used for ordinary questions', () => {
    const definition = tools.get('run_rlm')
    assert.ok(definition.parameters.properties.prompt, 'the prompt parameter must be declared')
    assert.equal(typeof definition.execute, 'function')
    assert.match(definition.description, /Do NOT use it for ordinary/)
    assert.match(sections.get('tool:run_rlm').text, /externalized/)
  })

  it('runs a full RLM through the tool: externalized prompt, REPL loop, sub-calls', async () => {
    llm.setScript({
      root: (index) =>
        index === 0
          ? [
              'I will read the context in slices and ask the sub-LLM about each slice.',
              '```repl',
              'seen = []',
              'for start in range(0, len(context), 12):',
              '    seen.append(llm_query(context.peek(start, start + 12)))',
              'Final = " | ".join(seen)',
              '```',
            ].join('\n')
          : 'FINAL_VAR(Final)',
      leaf: (prompt) => `read(${prompt.length})`,
    })

    const definition = tools.get('run_rlm')
    const output = await definition.execute({ prompt: PROMPT }, toolExec({ cwd: root }))

    assert.equal(output.status, 'ok')
    assert.equal(output.depth, 1)
    assert.ok(output.steps >= 1)
    assert.ok(output.usage.queries >= 2, `expected sub-calls, saw ${output.usage.queries}`)
    assert.ok(output.usage.inputTokens > 0)
    assert.match(output.answer, /read\(/)
    // The prompt never reached the conversation...
    assert.equal(
      llm.state.rootPrompts.some((prompt) => prompt.includes('CONFIDENTIAL-MATERIAL')),
      false,
      'the tool must not leak the prompt into the conversation',
    )
    // ...while the leaf calls did carry slices of it, i.e. the REPL read it.
    assert.ok(llm.state.leafPrompts.some((prompt) => prompt.includes('CONFIDENTIAL')))
  })

  it('returns a large answer as an artifact reference instead of inlining it', async () => {
    llm.setScript({
      root: (index) => (index === 0 ? '```repl\nFinal = "z" * 50000\n```' : 'FINAL_VAR(Final)'),
      leaf: () => 'unused',
    })
    const definition = tools.get('run_rlm')
    const output = await definition.execute(
      { prompt: 'material', budget: { maxInlineOutputBytes: 1024 } },
      toolExec({ cwd: root, callId: 'call-artifact' }),
    )
    assert.equal(output.status, 'ok')
    assert.equal(output.answer, undefined)
    assert.ok(output.artifact)
    assert.equal(output.artifact.chars, 50_000)
    const rendered = definition.output.render({}, output)
    assert.equal(rendered[0].type, 'text')
    assert.match(rendered[0].text, /stored as artifact/)
    assert.ok(rendered[0].text.length < 10_000, 'a huge final answer must not be inlined')
  })

  it('rejects an oversized task when the material should travel as context', async () => {
    const definition = tools.get('run_rlm')
    await assert.rejects(
      definition.execute({ prompt: 'x'.repeat(20_000), context: 'material' }, toolExec({ cwd: root })),
      /task ceiling/,
    )
  })

  it('turns a deployment budget refusal into a tool result, not a crash', async () => {
    llm.setScript({
      root: () => '```repl\nfor i in range(5):\n    llm_query("x")\n```',
      leaf: () => 'leaf',
    })
    const definition = tools.get('run_rlm')
    const output = await definition.execute(
      { prompt: 'material', budget: { maxTotalQueries: 2 } },
      toolExec({ cwd: root, callId: 'call-budget' }),
    )
    assert.ok(['budget-exceeded', 'ok'].includes(output.status))
    assert.ok(output.usage.queries <= 2)
  })

  it('cancels the run and kills the kernel when the caller aborts', async () => {
    const controller = new AbortController()
    llm.setScript({
      root: () => '```repl\nwhile True:\n    pass\n```',
      leaf: () => 'leaf',
    })
    const definition = tools.get('run_rlm')
    const running = definition.execute(
      { prompt: 'material' },
      toolExec({ cwd: root, callId: 'call-abort', signal: controller.signal }),
    )
    setTimeout(() => controller.abort('test abort'), 400)
    const output = await running
    assert.equal(output.status, 'cancelled')
    assert.equal(output.stopReason, 'cancelled')
  })

  it('writes durable sidecar traces for every invocation', async () => {
    const traces = await readdir(`${root}/artifacts/traces`)
    assert.ok(traces.length >= 1, 'each invocation must leave a trace file')
    const { readFile } = await import('node:fs/promises')
    const lines = (await readFile(`${root}/artifacts/traces/${traces[0]}`, 'utf8')).trim().split('\n')
    const events = lines.map((line) => JSON.parse(line))
    assert.equal(events[0].type, 'rlm/start')
    assert.equal(events[events.length - 1].type, 'rlm/end')
    // Exactly one start and one end: a duplicated fan-out would silently double
    // every cost figure a telemetry reader derives from this file.
    assert.equal(events.filter((event) => event.type === 'rlm/start').length, 1)
    assert.equal(events.filter((event) => event.type === 'rlm/end').length, 1)
    // Prompt text and query results are absent from durable records by design.
    const serialized = JSON.stringify(events)
    assert.equal(serialized.includes('CONFIDENTIAL-MATERIAL'), false)
  })

  it('unloads cleanly: every service and contribution disappears together', async () => {
    await ctx.fiber.dispose()
    assert.equal(ctx.rlm, undefined)
    assert.equal(ctx.rlmContextStore, undefined)
    assert.equal(ctx.rlmKernelRuntime, undefined)
    assert.equal(tools.has('run_rlm'), false)
    assert.equal(sections.has('tool:run_rlm'), false)
  })

  it('leaves no kernel workspace behind after unload', async () => {
    const leftovers = (await readdir(root)).filter((name) => name.startsWith('rlm-kernel-'))
    assert.deepEqual(leftovers, [])
    assert.ok(existsSync(`${root}/artifacts`), 'artifacts are durable and survive unload')
  })
})

async function settle(ms = 200) {
  await new Promise((resolve) => setTimeout(resolve, ms))
}
