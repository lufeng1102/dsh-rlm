/**
 * Framing tests.
 *
 * A length-framed protocol is only as good as its hostile-input behaviour: the
 * reader must reject an oversized declared frame instead of allocating it, and
 * must treat a truncated frame as an error rather than a short read.
 */

import assert from 'node:assert/strict'
import { Readable } from 'node:stream'
import { describe, it } from 'node:test'
import { FramedJsonReader, RlmProtocolError, encodeFrame, writeFrame } from '../lib/kernel/protocol.js'

/** A readable that emits the given buffers, one per chunk. */
function streamOf(chunks) {
  return Readable.from(chunks)
}

async function readAll(chunks, options = {}) {
  const reader = new FramedJsonReader(streamOf(chunks), { maxFrameBytes: options.maxFrameBytes ?? 1_000_000 })
  const frames = []
  while (true) {
    const frame = await reader.read()
    if (frame === null) return frames
    frames.push(frame)
  }
}

describe('length-framed JSON protocol', () => {
  it('round-trips a frame', () => {
    const encoded = encodeFrame({ type: 'ready', protocol: 1 })
    return readAll([encoded]).then((frames) => {
      assert.deepEqual(frames, [{ type: 'ready', protocol: 1 }])
    })
  })

  it('reassembles a frame split across arbitrary chunk boundaries', async () => {
    const encoded = encodeFrame({ type: 'result', stdout: 'a'.repeat(500), id: 7 })
    const chunks = []
    for (let index = 0; index < encoded.length; index += 17) chunks.push(encoded.subarray(index, index + 17))
    const frames = await readAll(chunks)
    assert.equal(frames.length, 1)
    assert.equal(frames[0].stdout.length, 500)
    assert.equal(frames[0].id, 7)
  })

  it('preserves newlines inside a payload', async () => {
    const payload = 'line one\nline two\n{"not": "json"}\n'
    const frames = await readAll([encodeFrame({ type: 'result', stdout: payload })])
    assert.equal(frames[0].stdout, payload)
  })

  it('reads several frames from one chunk', async () => {
    const frames = await readAll([
      Buffer.concat([encodeFrame({ id: 1 }), encodeFrame({ id: 2 }), encodeFrame({ id: 3 })]),
    ])
    assert.deepEqual(frames.map((frame) => frame.id), [1, 2, 3])
  })

  it('rejects a frame larger than the ceiling without allocating it', async () => {
    const reader = new FramedJsonReader(streamOf([Buffer.from('999999\n')]), { maxFrameBytes: 16 })
    await assert.rejects(reader.read(), RlmProtocolError)
  })

  it('rejects an invalid header', async () => {
    const reader = new FramedJsonReader(streamOf([Buffer.from('abc\n')]), { maxFrameBytes: 16 })
    await assert.rejects(reader.read(), /invalid frame header/)
  })

  it('rejects a header that never ends', async () => {
    const reader = new FramedJsonReader(streamOf([Buffer.from('1'.repeat(200))]), { maxFrameBytes: 1024 })
    await assert.rejects(reader.read(), /without a newline/)
  })

  it('treats a truncated trailing frame as a protocol error', async () => {
    const encoded = encodeFrame({ type: 'result', stdout: 'x'.repeat(100) })
    const reader = new FramedJsonReader(streamOf([encoded.subarray(0, encoded.length - 20)]), {
      maxFrameBytes: 1024,
    })
    await assert.rejects(reader.read(), /short of/)
  })

  it('rejects a non-object payload', async () => {
    const payload = Buffer.from('[1,2,3]', 'utf8')
    const frame = Buffer.concat([Buffer.from(`${payload.length}\n`), payload])
    const reader = new FramedJsonReader(streamOf([frame]), { maxFrameBytes: 1024 })
    await assert.rejects(reader.read(), /not a JSON object/)
  })

  it('rejects a partial header at end of stream', async () => {
    const reader = new FramedJsonReader(streamOf([Buffer.from('12')]), { maxFrameBytes: 1024 })
    await assert.rejects(reader.read(), /inside a frame header/)
  })

  it('resolves null at a clean end of stream', async () => {
    assert.deepEqual(await readAll([]), [])
  })

  it('writes a frame that the reader accepts', async () => {
    const { Writable } = await import('node:stream')
    const chunks = []
    const sink = new Writable({
      write(chunk, _encoding, callback) {
        chunks.push(Buffer.from(chunk))
        callback()
      },
    })
    await writeFrame(sink, { op: 'eval', id: 1, code: 'print(1)' })
    const frames = await readAll(chunks)
    assert.equal(frames[0].op, 'eval')
    assert.equal(frames[0].code, 'print(1)')
  })

  it('reports frame sizes to the metrics hook', async () => {
    const sizes = []
    const reader = new FramedJsonReader(streamOf([encodeFrame({ type: 'x' }), encodeFrame({ type: 'y' })]), {
      maxFrameBytes: 1024,
      onFrameBytes: (bytes) => sizes.push(bytes),
    })
    while ((await reader.read()) !== null) {
      // drain
    }
    assert.equal(sizes.length, 2)
    assert.ok(sizes.every((size) => size > 0))
  })
})
