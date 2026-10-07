/**
 * Length-framed JSON transport for kernel processes.
 *
 * One frame is `<decimal byte length>\n<utf-8 JSON>`. Framing (rather than
 * newline-delimited JSON) is required because program stdout and payloads are
 * multi-megabyte strings that legitimately contain newlines — a newline protocol
 * would silently corrupt them, which is exactly the kind of failure that is
 * undetectable until a long-context run returns the wrong answer.
 *
 * The reader is defensive by construction: a declared frame length beyond the
 * configured ceiling aborts the frame stream instead of allocating it, and a
 * stream that ends mid-frame is a protocol error rather than a short read.
 *
 * @module dsh-rlm/kernel/protocol
 */

import type { Readable, Writable } from 'node:stream'

export class RlmProtocolError extends Error {
  override readonly name = 'RlmProtocolError'

  constructor(message: string) {
    super(message)
  }
}

export interface RlmFrame {
  type?: string
  [key: string]: unknown
}

export interface FramedReaderOptions {
  /** Hard ceiling on one frame; a larger declared frame is a protocol violation. */
  maxFrameBytes: number
  /** Ceiling on the ASCII header before its newline. */
  maxHeaderBytes?: number
  /** Observes every decoded frame's payload size, for metrics. */
  onFrameBytes?: (bytes: number) => void
}

/**
 * Incremental frame reader over an async iterable of buffers.
 *
 * Internally a queue of unparsed chunks: bytes are only copied when a frame
 * spans a chunk boundary, so a 64 MiB payload costs one copy, not one per read.
 */
export class FramedJsonReader {
  private readonly iterator: AsyncIterator<unknown>
  private readonly options: FramedReaderOptions
  private readonly maxHeaderBytes: number
  private chunks: Buffer[] = []
  private total = 0
  private ended = false

  constructor(stream: Readable, options: FramedReaderOptions) {
    this.iterator = (stream as AsyncIterable<unknown>)[Symbol.asyncIterator]()
    this.options = options
    this.maxHeaderBytes = options.maxHeaderBytes ?? 32
  }

  /** Unparsed bytes currently buffered. */
  get buffered(): number {
    return this.total
  }

  /**
   * Read one frame.
   *
   * @returns the frame, or `null` at a clean end of stream.
   * @throws {RlmProtocolError} on a bad header, an oversized frame, or a
   * truncated trailing frame.
   */
  async read(): Promise<RlmFrame | null> {
    const newline = await this.findHeaderEnd()
    if (newline === null) {
      if (this.total > 0) {
        throw new RlmProtocolError(`stream ended inside a frame header (${this.total} trailing bytes)`)
      }
      return null
    }
    const header = this.take(newline).toString('ascii')
    this.drop(1) // the newline itself
    const length = Number.parseInt(header.trim(), 10)
    if (!Number.isInteger(length) || length < 0) {
      throw new RlmProtocolError(`invalid frame header: ${JSON.stringify(header)}`)
    }
    if (length > this.options.maxFrameBytes) {
      throw new RlmProtocolError(
        `frame of ${length} bytes exceeds the ${this.options.maxFrameBytes} byte ceiling`,
      )
    }
    await this.fill(length)
    if (this.total < length) {
      throw new RlmProtocolError(
        `stream ended ${length - this.total} bytes short of a ${length} byte frame`,
      )
    }
    const payload = this.take(length).toString('utf8')
    this.options.onFrameBytes?.(length)
    let parsed: unknown
    try {
      parsed = JSON.parse(payload)
    } catch (error) {
      throw new RlmProtocolError(`frame payload is not valid JSON: ${(error as Error).message}`)
    }
    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
      throw new RlmProtocolError('frame payload is not a JSON object')
    }
    return parsed as RlmFrame
  }

  /** Offset of the next newline, filling as needed; `null` at end of stream. */
  private async findHeaderEnd(): Promise<number | null> {
    let searched = 0
    while (true) {
      let offset = 0
      for (const chunk of this.chunks) {
        const index = chunk.indexOf(0x0a)
        if (index >= 0) return offset + index
        offset += chunk.length
      }
      searched = this.total
      if (searched > this.maxHeaderBytes) {
        throw new RlmProtocolError(`frame header exceeds ${this.maxHeaderBytes} bytes without a newline`)
      }
      if (this.ended) return null
      await this.pull()
      if (this.total === searched && this.ended) return null
    }
  }

  private async pull(): Promise<void> {
    const { value, done } = await this.iterator.next()
    if (done) {
      this.ended = true
      return
    }
    const chunk = Buffer.isBuffer(value) ? value : Buffer.from(String(value))
    if (chunk.length === 0) return
    this.chunks.push(chunk)
    this.total += chunk.length
  }

  private async fill(target: number): Promise<void> {
    while (this.total < target && !this.ended) await this.pull()
  }

  /** Remove the first `count` bytes and return them. */
  private take(count: number): Buffer {
    if (count === 0) return Buffer.alloc(0)
    if (this.chunks[0]!.length >= count) {
      const head = this.chunks[0]!
      const result = head.subarray(0, count)
      const rest = head.subarray(count)
      if (rest.length > 0) this.chunks[0] = rest
      else this.chunks.shift()
      this.total -= count
      return result
    }
    const parts: Buffer[] = []
    let remaining = count
    while (remaining > 0) {
      const head = this.chunks[0]!
      if (head.length <= remaining) {
        parts.push(head)
        this.chunks.shift()
        remaining -= head.length
      } else {
        parts.push(head.subarray(0, remaining))
        this.chunks[0] = head.subarray(remaining)
        remaining = 0
      }
    }
    this.total -= count
    return Buffer.concat(parts, count)
  }

  private drop(count: number): void {
    this.take(count)
  }
}

/** Write one frame, awaiting the write callback (and thus backpressure). */
export function writeFrame(stream: Writable, message: unknown): Promise<void> {
  const payload = Buffer.from(JSON.stringify(message), 'utf8')
  const header = Buffer.from(`${payload.length}\n`, 'ascii')
  return new Promise((resolve, reject) => {
    stream.write(Buffer.concat([header, payload]), (error) => {
      if (error) reject(error)
      else resolve()
    })
  })
}

/** Serialize a frame the same way {@link writeFrame} does; used by tests. */
export function encodeFrame(message: unknown): Buffer {
  const payload = Buffer.from(JSON.stringify(message), 'utf8')
  return Buffer.concat([Buffer.from(`${payload.length}\n`, 'ascii'), payload])
}
