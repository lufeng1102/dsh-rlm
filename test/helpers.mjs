/**
 * Shared test helpers.
 *
 * The suite runs against the compiled `lib/` output, not the TypeScript source:
 * the artifact under test is what a deployment loads, and a build that forgets
 * to copy the Python driver should fail here rather than in production.
 */

import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { Context } from '@deepseek-ai/cordis'
import { LocalRlmContextStore } from '../lib/context-local.js'
import { RecordingEventSink } from '../lib/events.js'
import { createDefaultServiceConfig, resolvePluginConfig } from '../lib/config.js'
import { RlmService } from '../lib/service.js'

/** Temporary directory removed by {@link withTempDir}. */
export async function makeTempDir(prefix = 'rlm-test-') {
  return mkdtemp(join(tmpdir(), prefix))
}

export async function cleanup(dir) {
  await rm(dir, { recursive: true, force: true })
}

/** A deterministic, cheap character sequence of a given length. */
export function filler(chars, seed = 'x') {
  return Array.from({ length: chars }, (_, index) => String.fromCharCode(97 + ((index + seed.length) % 26))).join('')
}

/** A prompt large enough that copying it into root history would be obvious. */
export function bigDocument(chars) {
  const words = ['alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot', 'golf', 'hotel']
  let text = ''
  let index = 0
  while (text.length < chars) {
    text += `${words[index % words.length]}-${index} needle-${index % 7 === 0 ? 'yes' : 'no'} `
    index += 1
  }
  return text.slice(0, chars)
}

/**
 * Build a fully wired RlmService for a test.
 *
 * @param {object} options
 * @param {string} options.root - artifact/context root (a temp directory).
 * @param {import('../lib/types.js').RlmModelClient} options.model
 * @param {import('../lib/kernel/index.js').RlmKernelRuntime} options.kernel
 * @param {object} [options.config] - plugin config overrides.
 */
export function buildService(options) {
  const ctx = options.ctx ?? new Context()
  const resolved = resolvePluginConfig({
    rootProvider: 'test',
    rootModel: 'root-model',
    leafProvider: 'test',
    leafModel: 'leaf-model',
    contextStoreRoot: options.root,
    ...(options.config ?? {}),
  })
  const store = new LocalRlmContextStore(ctx, {
    root: options.root,
    charIndexStep: options.config?.charIndexStep ?? 4096,
  })
  const events = new RecordingEventSink()
  const serviceConfig = createDefaultServiceConfig(resolved)
  const service = new RlmService(ctx, serviceConfig, {
    contextStore: store,
    kernelRuntime: options.kernel,
    model: options.model,
    events,
    ...(options.now === undefined ? {} : { now: options.now }),
  })
  return { service, store, events, resolved, ctx }
}
