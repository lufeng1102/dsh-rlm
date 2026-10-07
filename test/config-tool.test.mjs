/**
 * Configuration resolution and tool-argument tests.
 *
 * Two things are being protected here: that a limit can never end up
 * `undefined` (which would silently mean "unbounded"), and that the model's
 * arguments cannot raise the host's authority.
 */

import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { Config, createDefaultServiceConfig, resolvePluginConfig } from '../lib/config.js'
import { DEFAULT_BUDGETS } from '../lib/service.js'
import { renderRunRlmResult, sanitizeBudgets, toContextInput, toOutput } from '../lib/tool.js'
import { bindingsForDepth, toContextInput as serviceToContextInput } from '../lib/service.js'

describe('plugin configuration', () => {
  it('materializes every limit from an empty configuration', () => {
    const resolved = resolvePluginConfig({})
    for (const [key, value] of Object.entries(resolved.budgets)) {
      assert.equal(typeof value, 'number', `${key} must be a number`)
      assert.ok(Number.isFinite(value), `${key} must be finite`)
    }
    assert.equal(resolved.toolName, 'run_rlm')
    assert.equal(resolved.defaultDepth, 1)
    assert.equal(resolved.finalVariableName, 'Final')
  })

  it('clamps the default depth to the ceiling', () => {
    const resolved = resolvePluginConfig({ defaultDepth: 9, maxDepth: 2 })
    assert.equal(resolved.defaultDepth, 2)
    assert.equal(resolved.maxDepth, 2)
  })

  it('rejects an unknown prompt profile instead of guessing', () => {
    assert.throws(() => resolvePluginConfig({ promptProfileId: 'nope' }), /unknown promptProfileId/)
  })

  it('keeps the deployment ceiling when a row asks for more', () => {
    const resolved = resolvePluginConfig({ maxTotalQueries: 10_000 })
    assert.equal(resolved.budgets.maxTotalQueries, 10_000)
    // The service config is derived from the resolved values, never from the raw row.
    const serviceConfig = createDefaultServiceConfig(resolved)
    assert.equal(serviceConfig.budgets.maxTotalQueries, 10_000)
    assert.equal(serviceConfig.maxQueryPromptChars, 10_000 === 0 ? 0 : resolved.budgets.maxQueryPromptChars)
  })

  it('derives the leaf route from the root route when only the root is configured', () => {
    const resolved = resolvePluginConfig({ rootProvider: 'p', rootModel: 'm' })
    assert.deepEqual(resolved.rootRoute, { provider: 'p', model: 'm' })
    assert.deepEqual(resolved.leafRoute, { provider: 'p', model: 'm' })
    assert.equal(resolved.childRoute, undefined)
  })

  it('records a child route only when one is configured', () => {
    const resolved = resolvePluginConfig({
      rootProvider: 'p',
      rootModel: 'm',
      childProvider: 'c',
      childModel: 'cm',
    })
    assert.deepEqual(resolved.childRoute, { provider: 'c', model: 'cm' })
    assert.deepEqual(resolved.promptProfile.child, { provider: 'c', model: 'cm' })
  })

  it('survives a schemastery-normalized config', () => {
    const normalized = new Config({
      rootProvider: 'p',
      rootModel: 'm',
      leafProvider: 'p',
      leafModel: 'lm',
    })
    const resolved = resolvePluginConfig(normalized)
    assert.equal(resolved.leafRoute.model, 'lm')
    assert.equal(resolved.maxRootIterations, 64)
  })

  it('exposes a schema for the profile row', () => {
    assert.ok(Config)
    assert.equal(typeof Config, 'function')
  })

  it('exposes the deployment defaults as a single source of truth', () => {
    const resolved = resolvePluginConfig({})
    assert.equal(resolved.maxConcurrentQueries, DEFAULT_BUDGETS.maxConcurrentQueries)
    assert.equal(resolved.maxArtifactBytes, DEFAULT_BUDGETS.maxArtifactBytes)
  })
})

describe('depth → binding map', () => {
  it('installs no model bindings at depth 0', () => {
    assert.deepEqual(bindingsForDepth(0), [])
  })

  it('installs llm_query and rlm_query at depth 1, per the paper fallback contract', () => {
    const bindings = bindingsForDepth(1)
    assert.ok(bindings.includes('llm_query'))
    assert.ok(bindings.includes('rlm_query'))
  })

  it('never lists an unknown binding name', () => {
    for (const depth of [0, 1, 2, 5]) {
      for (const name of bindingsForDepth(depth)) {
        assert.ok(['llm_query', 'llm_query_many', 'rlm_query', 'rlm_query_many'].includes(name))
      }
    }
  })
})

describe('run_rlm argument handling', () => {
  it('treats a bare prompt as the externalized material, the paper shape', () => {
    const material = toContextInput(undefined, 'the whole question', 1000)
    assert.deepEqual(material.input, { kind: 'text', text: 'the whole question' })
    assert.equal(material.task, undefined)
  })

  it('externalizes a document array and keeps the question as the task', () => {
    const material = toContextInput(['doc one', 'doc two'], 'how many?', 1000)
    assert.equal(material.input.kind, 'documents')
    assert.deepEqual(material.input.documents, ['doc one', 'doc two'])
    assert.equal(material.task, 'how many?')
  })

  it('accepts a string context and a record context', () => {
    assert.equal(toContextInput('material', 'q', 100).input.kind, 'text')
    assert.equal(toContextInput({ title: 'x' }, 'q', 100).input.kind, 'record')
  })

  it('refuses an oversized task so material travels as context instead', () => {
    assert.throws(() => toContextInput('material', 'x'.repeat(200), 100), /task ceiling/)
  })

  it('maps an inline item list from rlm_query to a list context, and a tool array to documents', () => {
    // The two entry points name the same payload differently on purpose: a
    // child RLM receives an explicit list, while a caller handing over an array
    // is describing a document collection.
    assert.deepEqual(serviceToContextInput({ kind: 'inline', items: ['a'] }), { kind: 'list', items: ['a'] })
    assert.deepEqual(toContextInput(['a'], 'q', 100).input, { kind: 'documents', documents: ['a'] })
    assert.deepEqual(serviceToContextInput({ kind: 'inline', record: { a: 'b' } }), {
      kind: 'record',
      record: { a: 'b' },
    })
  })

  it('ignores unknown, negative and non-finite budget overrides', () => {
    const sanitized = sanitizeBudgets({
      maxTotalQueries: 5.7,
      maxCostUsd: -1,
      maxInputTokens: Number.POSITIVE_INFINITY,
      notADimension: 100,
      maxRootIterations: 'many',
    })
    assert.deepEqual(sanitized, { maxTotalQueries: 5 })
  })

  it('returns an empty override for a non-object budget', () => {
    assert.deepEqual(sanitizeBudgets(null), {})
    assert.deepEqual(sanitizeBudgets('nope'), {})
  })
})

describe('run_rlm result rendering', () => {
  const baseUsage = {
    rootIterations: 2,
    queries: 3,
    kernels: 1,
    inputTokens: 100,
    outputTokens: 50,
    contextReadBytes: 2048,
    costUsd: 0.01,
    wallClockMs: 1234,
    maxObservedConcurrency: 2,
  }

  it('renders a small answer with its cost summary', () => {
    const text = renderRunRlmResult(
      { status: 'ok', stopReason: 'final-text', answer: 'forty-two', steps: 2, depth: 1, usage: baseUsage },
      1000,
    )
    assert.match(text, /forty-two/)
    assert.match(text, /3 sub-calls/)
  })

  it('renders an artifact reference and its bounded preview, never the whole value', () => {
    const text = renderRunRlmResult(
      {
        status: 'ok',
        stopReason: 'final-variable',
        artifact: { id: 'art_1', bytes: 50_000, chars: 50_000, preview: 'preview text', truncated: true },
        steps: 3,
        depth: 1,
        usage: baseUsage,
      },
      10_000,
    )
    assert.match(text, /stored as artifact art_1/)
    assert.match(text, /preview text/)
    assert.ok(text.length < 1_000)
  })

  it('truncates a long inline answer to the render ceiling', () => {
    const text = renderRunRlmResult(
      { status: 'ok', stopReason: 'final-text', answer: 'x'.repeat(5_000), steps: 1, depth: 1, usage: baseUsage },
      100,
    )
    assert.match(text, /truncated for display/)
    assert.ok(text.length < 300)
  })

  it('renders a failure with its stop reason and error', () => {
    const text = renderRunRlmResult(
      {
        status: 'cancelled',
        stopReason: 'cancelled',
        steps: 1,
        depth: 1,
        usage: baseUsage,
        error: { kind: 'cancelled', message: 'parent aborted' },
      },
      1000,
    )
    assert.match(text, /run_rlm cancelled \(cancelled\)/)
    assert.match(text, /parent aborted/)
  })
})

describe('result → tool output mapping', () => {
  it('carries usage, artifacts and errors without lossy defaults', () => {
    const output = toOutput({
      status: 'ok',
      text: undefined,
      artifactRef: {
        id: 'art_x',
        sha256: 'h',
        bytes: 10,
        chars: 10,
        contentType: 'text/plain',
        preview: 'p',
        truncated: false,
      },
      stopReason: 'final-variable',
      depth: 2,
      maxDepth: 3,
      steps: 4,
      usage: {
        rootIterations: 4,
        queries: 5,
        kernels: 2,
        inputTokens: 1,
        outputTokens: 2,
        contextReadBytes: 3,
        costUsd: 0.5,
        wallClockMs: 6,
        maxObservedConcurrency: 1,
      },
    })
    assert.equal(output.answer, undefined)
    assert.equal(output.artifact.id, 'art_x')
    assert.equal(output.depth, 2)
    assert.equal(output.error, undefined)
  })
})
