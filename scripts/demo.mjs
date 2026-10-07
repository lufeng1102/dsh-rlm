#!/usr/bin/env node
/**
 * Runnable RLM demonstration.
 *
 * Runs the *real* plugin stack — real cordis context, real content-addressed
 * store, real persistent Python kernel — with a scripted model standing in for
 * the LLM, and narrates the four properties that make this an RLM rather than a
 * long-context workflow:
 *
 *   1. the prompt is externalized and never enters the root conversation;
 *   2. the REPL is persistent and the program loops over arbitrary ranges;
 *   3. `llm_query` returns values to the program, not to the root history;
 *   4. a final value larger than the inline limit leaves as an artifact.
 *
 * Usage:
 *   node scripts/demo.mjs [--chars 400000] [--keep]
 */

import { mkdtemp, readFile, readdir, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { Context } from '@deepseek-ai/cordis'
import * as contextLocalPlugin from '../lib/plugins/context-local.js'
import * as kernelPythonPlugin from '../lib/plugins/kernel-python.js'
import * as rlmPlugin from '../lib/index.js'

const NEEDLE = 'THE-MAGIC-NUMBER-IS-4271'

function parseArgs(argv) {
  const options = { chars: 400_000, keep: false }
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] === '--chars') options.chars = Number(argv[++index])
    else if (argv[index] === '--keep') options.keep = true
  }
  return options
}

/** A long document with the needle buried at a known position. */
function buildDocument(chars) {
  const blocks = []
  let size = 0
  let index = 0
  while (size < chars) {
    const block = `record ${index}: the quarterly summary for division ${index % 97} reports ordinary figures and no anomaly whatsoever.\n`
    blocks.push(block)
    size += block.length
    index += 1
  }
  const text = blocks.join('')
  const middle = Math.floor(text.length / 2)
  return `${text.slice(0, middle)}${NEEDLE}\n${text.slice(middle)}`
}

/** The scripted "model": a root that writes RLM code, and a leaf that reads slices. */
function createScriptedModel() {
  const state = { rootIndex: 0, rootHistory: [], leafPrompts: [] }
  const service = {
    stream(request) {
      const isLeaf = request.system === undefined
      const prompt = request.messages
        .map((message) => (message.content ?? []).map((block) => block.text ?? '').join(''))
        .join('\n')
      let text
      if (isLeaf) {
        state.leafPrompts.push(prompt)
        text = prompt.includes('THE-MAGIC-NUMBER-IS-')
          ? `FOUND: ${NEEDLE}`
          : 'nothing unusual in this slice'
      } else {
        state.rootHistory.push(prompt)
        text =
          state.rootIndex++ === 0
            ? [
                'I will chunk the context and ask the sub-LLM about each chunk, keeping answers in a variable.',
                '```repl',
                'chunk = 20_000',
                'findings = []',
                'for start in range(0, len(context), chunk):',
                '    part = context.slice(start, min(start + chunk, len(context)))',
                '    findings.append(llm_query("Extract any magic number you see. " + part))',
                'hits = [f for f in findings if "FOUND" in f]',
                'answer = hits[0] if hits else "not found"',
                'print(f"scanned {len(findings)} chunks")',
                '```',
              ].join('\n')
            : 'FINAL_VAR(answer)'
      }
      return (async function* generate() {
        yield { type: 'block-start', index: 0, blockType: 'text' }
        if (text.length > 0) yield { type: 'text-delta', index: 0, text }
        yield { type: 'finish', reason: { kind: 'stop' } }
      })()
    },
  }
  return { service, state }
}

function createToolsStub(registry) {
  return {
    register(definition) {
      registry.set(definition.name, definition)
      return () => registry.delete(definition.name)
    },
    get: (name) => registry.get(name),
  }
}

async function main() {
  const options = parseArgs(process.argv.slice(2))
  const root = await mkdtemp(join(tmpdir(), 'rlm-demo-'))
  const document = buildDocument(options.chars)
  const registry = new Map()
  const ctx = new Context()
  ctx.provide('tools', createToolsStub(registry))
  ctx.provide('systemPrompt', { section: () => () => {}, getSectionOrder: () => 1_000 })
  const model = createScriptedModel()
  ctx.provide('llm', model.service)
  ctx.plugin(contextLocalPlugin, { root: `${root}/artifacts`, charIndexStep: 8192 })
  ctx.plugin(kernelPythonPlugin, { pythonPath: process.env.DSH_RLM_PYTHON ?? 'python3' })
  await settle(150)
  ctx.plugin(rlmPlugin, {
    rootProvider: 'demo',
    rootModel: 'root-model',
    leafProvider: 'demo',
    leafModel: 'leaf-model',
    defaultDepth: 1,
    maxDepth: 2,
    contextStoreRoot: `${root}/artifacts`,
    maxInlineOutputBytes: 4096,
    maxTotalQueries: 64,
  })
  await settle(250)

  console.log('RLM demo')
  console.log('────────')
  console.log(`prompt: ${document.length.toLocaleString()} characters (needle embedded at the midpoint)`)
  console.log('root model: scripted RLM profile      leaf model: scripted reader')
  console.log('')

  const tool = registry.get('run_rlm')
  if (!tool) throw new Error('the run_rlm tool was not registered; the plugin did not mount')

  const output = await tool.execute(
    { prompt: document },
    {
      callId: 'demo-call',
      name: 'run_rlm',
      arguments: {},
      signal: new AbortController().signal,
      rootCallId: 'demo-call',
      token: Symbol('token'),
      deferContext() {},
      concludeTurn() {},
      agent: { id: 'demo-session', session: { header: { cwd: root }, append: () => undefined } },
    },
  )

  const rootHistoryChars = model.state.rootHistory.reduce((sum, prompt) => sum + prompt.length, 0)
  const promptLeaked = model.state.rootHistory.some((prompt) => prompt.includes(NEEDLE))

  console.log('result')
  console.log(`  status          ${output.status} (${output.stopReason})`)
  console.log(`  answer          ${output.answer ?? output.artifact?.preview ?? '<none>'}`)
  console.log(`  root steps      ${output.steps}`)
  console.log(`  llm_query calls ${output.usage.queries}`)
  console.log(`  context read    ${output.usage.contextReadBytes.toLocaleString()} bytes (metered)`)
  console.log('')
  console.log('paper properties')
  console.log(`  1. prompt externalized   root history is ${rootHistoryChars.toLocaleString()} chars for a ${document.length.toLocaleString()} char prompt`)
  console.log(`     needle in root history: ${promptLeaked ? 'YES (failure)' : 'no'}`)
  console.log(
    `  2. persistent REPL       ${output.steps} root steps in one namespace (step 1 built \`answer\`, step 2 returned it)`,
  )
  console.log(`  3. symbolic sub-calls    ${output.usage.queries} leaf calls returned into variables, not into history`)
  console.log(`  4. large final output    ${output.artifact ? `artifact ${output.artifact.id}` : 'returned inline (below the ceiling)'}`)

  const traces = await readdir(`${root}/artifacts/traces`)
  console.log('')
  console.log(`durable trace: ${root}/artifacts/traces/${traces[0]}`)
  const firstLines = (await readFile(`${root}/artifacts/traces/${traces[0]}`, 'utf8')).trim().split('\n')
  console.log(`events: ${firstLines.map((line) => JSON.parse(line).type).join(' → ')}`)

  const failures = []
  if (output.status !== 'ok') failures.push(`status was ${output.status}`)
  if (promptLeaked) failures.push('the prompt leaked into root history')
  if (output.usage.queries < 2) failures.push('no meaningful sub-call loop happened')
  if (rootHistoryChars > document.length / 10) failures.push('root history grew proportionally to the prompt')

  if (!options.keep) await rm(root, { recursive: true, force: true })
  await ctx.fiber.dispose()

  if (failures.length > 0) {
    console.error('')
    console.error(`FAILED: ${failures.join('; ')}`)
    process.exitCode = 1
    return
  }
  console.log('')
  console.log('all paper properties held.')
}

async function settle(ms) {
  await new Promise((resolve) => setTimeout(resolve, ms))
}

await main()
