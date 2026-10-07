/**
 * Context store tests.
 *
 * The two properties worth pinning are the ones that are easy to get subtly
 * wrong: character ranges must not be byte ranges, and a read must never exceed
 * its ceiling. Everything else here guards deduplication, authorization and
 * retention.
 */

import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { Context } from '@deepseek-ai/cordis'
import { LocalRlmContextStore, RlmAccessError, RlmRangeError, canonicalizeInput } from '../lib/context-local.js'
import { bigDocument, cleanup, filler, makeTempDir } from './helpers.mjs'

async function withStore(run, options = {}) {
  const root = await makeTempDir()
  const store = new LocalRlmContextStore(new Context(), { root, charIndexStep: 64, ...options })
  try {
    await run(store, root)
  } finally {
    await cleanup(root)
  }
}

describe('local content-addressed context store', () => {
  it('stores text and reads it back by character range', async () => {
    await withStore(async (store) => {
      const ref = await store.put({ kind: 'text', text: 'hello world' })
      assert.equal(ref.kind, 'text')
      assert.equal(ref.chars, 11)
      const handle = await store.open(ref)
      assert.equal(await handle.slice(0, 5), 'hello')
      assert.equal(await handle.slice(6, 11), 'world')
      assert.equal(await handle.peek(0, 4), 'hell')
      assert.equal(handle.length, 11)
    })
  })

  it('deduplicates identical content by hash', async () => {
    await withStore(async (store) => {
      const first = await store.put({ kind: 'text', text: 'same' })
      const second = await store.put({ kind: 'text', text: 'same' })
      assert.equal(first.id, second.id)
    })
  })

  it('keeps multibyte characters intact across index-sample boundaries', async () => {
    await withStore(async (store) => {
      // Every character is 3 bytes in UTF-8; the index samples every 64 chars,
      // so a naive byte/char confusion would corrupt the read.
      const text = '漢'.repeat(200) + 'tail'
      const ref = await store.put({ kind: 'text', text })
      const handle = await store.open(ref)
      assert.equal(await handle.slice(0, 3), '漢漢漢')
      assert.equal(await handle.slice(64, 67), '漢漢漢')
      assert.equal(await handle.slice(127, 130), '漢漢漢')
      assert.equal(await handle.slice(200, 204), 'tail')
      assert.equal(await handle.slice(0, 200), '漢'.repeat(200))
    })
  })

  it('survives an astral character sequence without splitting a surrogate pair in the index', async () => {
    await withStore(async (store) => {
      const text = '🎯'.repeat(100)
      const ref = await store.put({ kind: 'text', text })
      const handle = await store.open(ref)
      // A split surrogate pair would surface as a replacement character.
      assert.equal(handle.ref.chars, text.length)
      assert.equal((await handle.slice(0, 64)).includes('\uFFFD'), false)
      assert.equal(await handle.slice(0, 2), '🎯')
    })
  })

  it('pages list items with a character index', async () => {
    await withStore(async (store) => {
      const items = Array.from({ length: 300 }, (_, index) => `doc ${index}`)
      const ref = await store.put({ kind: 'documents', documents: items })
      assert.equal(ref.chunkChars.length, 300)
      const handle = await store.open(ref)
      assert.equal(handle.length, 300)
      assert.deepEqual(await handle.items(298, 300), ['doc 298', 'doc 299'])
      assert.equal(await handle.slice(0, 2), 'doc 0\ndoc 1')
      assert.equal(await handle.get('5'), 'doc 5')
    })
  })

  it('reads records by key and refuses unknown keys', async () => {
    await withStore(async (store) => {
      const ref = await store.put({ kind: 'record', record: { title: 'doc', body: 'x' } })
      const handle = await store.open(ref)
      assert.equal(await handle.get('title'), 'doc')
      await assert.rejects(handle.get('missing'), /no key/)
    })
  })

  it('searches substrings and regexes with a hit ceiling', async () => {
    await withStore(async (store) => {
      const ref = await store.put({ kind: 'text', text: 'needle here, needle there, nothing' })
      const handle = await store.open(ref)
      const hits = await handle.search('needle')
      assert.equal(hits.length, 2)
      assert.equal(hits[0].start, 0)
      const regexHits = await handle.search('need\\w+', { regex: true, maxHits: 1 })
      assert.equal(regexHits.length, 1)
    })
  })

  it('refuses a read above the per-call range ceiling', async () => {
    await withStore(
      async (store) => {
        const ref = await store.put({ kind: 'text', text: filler(1000) })
        const handle = await store.open(ref)
        await assert.rejects(handle.slice(0, 900), RlmRangeError)
      },
      { maxRangeChars: 100 },
    )
  })

  it('refuses to materialize a value above the requested ceiling', async () => {
    await withStore(async (store) => {
      const ref = await store.put({ kind: 'text', text: filler(500) })
      const handle = await store.open(ref)
      await assert.rejects(handle.materialize({ maxBytes: 10 }), RlmRangeError)
    })
  })

  it('authorizes reads per session', async () => {
    await withStore(async (store) => {
      const ref = await store.put({ kind: 'text', text: 'private' }, { sessionId: 'session-a' })
      await assert.rejects(store.open(ref, { sessionId: 'session-b' }), RlmAccessError)
      const handle = await store.open(ref, { sessionId: 'session-a' })
      assert.equal(await handle.slice(0, 7), 'private')
    })
  })

  it('stores a large final value as an artifact with a bounded preview', async () => {
    await withStore(async (store) => {
      const text = bigDocument(50_000)
      const ref = await store.writeArtifact(text, { maxBytes: 1_000_000, previewChars: 100 })
      assert.equal(ref.chars, text.length)
      assert.equal(ref.preview.length, 100)
      assert.equal(ref.truncated, true)
      const readBack = await store.readArtifact(ref)
      assert.equal(readBack.length, text.length)
      assert.equal(readBack, text)
    })
  })

  it('refuses an artifact above its ceiling rather than truncating silently', async () => {
    await withStore(async (store) => {
      await assert.rejects(store.writeArtifact(filler(1000), { maxBytes: 10 }), RlmRangeError)
    })
  })

  it('adopts an existing file without buffering it as one string', async () => {
    await withStore(async (store, root) => {
      const { writeFile } = await import('node:fs/promises')
      const { join } = await import('node:path')
      const source = join(root, 'kernel-output.txt')
      const text = bigDocument(20_000)
      await writeFile(source, text, 'utf8')
      const ref = await store.writeArtifactFromFile(source, { maxBytes: 1_000_000, previewChars: 64 })
      assert.equal(ref.chars, text.length)
      assert.equal(ref.bytes, Buffer.byteLength(text, 'utf8'))
      assert.equal(await store.readArtifact(ref), text)
    })
  })

  it('sweeps expired artifacts and reports what it reclaimed', async () => {
    await withStore(async (store) => {
      const ref = await store.put({ kind: 'text', text: 'old' }, { sessionId: 's' })
      const dry = await store.sweep({ maxAgeMs: 0, dryRun: true, now: ref.createdAt + 10 })
      assert.equal(dry.deleted, 1)
      assert.equal(dry.reclaimedBytes, ref.bytes)
      const real = await store.sweep({ maxAgeMs: 0, now: ref.createdAt + 10 })
      assert.equal(real.deleted, 1)
      assert.equal(await store.delete(ref, {}), false)
    })
  })

  it('reports store statistics', async () => {
    await withStore(async (store) => {
      await store.put({ kind: 'text', text: 'one' })
      await store.writeArtifact('two', { maxBytes: 100 })
      const stats = await store.stats()
      assert.equal(stats.artifacts, 2)
      assert.equal(stats.provider, 'local-content-addressed')
      assert.equal(stats.encrypted, false)
    })
  })

  it('fails closed when encryption is required but no cipher is configured', async () => {
    const root = await makeTempDir()
    try {
      assert.throws(() => new LocalRlmContextStore(new Context(), { root, requireEncryption: true }), /cipher/)
    } finally {
      await cleanup(root)
    }
  })

  it('refuses a record above its ceiling at admission time', () => {
    assert.throws(
      () => canonicalizeInput({ kind: 'record', record: { big: filler(100) } }, 10),
      RlmRangeError,
    )
  })
})
