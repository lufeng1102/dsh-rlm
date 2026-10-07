/**
 * Local, content-addressed implementation of the RLM context/artifact store.
 *
 * Layout under `root`:
 *
 *     data/<id>.data     canonical payload bytes
 *     meta/<id>.json     reference + read index + authorization fields
 *
 * `id` is the content hash, so identical prompts deduplicate for free.
 *
 * Two properties matter more than speed here:
 *
 * - **Unicode safety.** Callers address *characters*, but files are read in
 *   *bytes*. Every stored text carries a sampled char→byte index, and range
 *   reads decode with a streaming `TextDecoder`, so a slice never lands in the
 *   middle of a code point.
 * - **Bounded reads.** A range read has a hard character cap, so a program
 *   cannot turn one call into an unbounded allocation on the host.
 *
 * @module dsh-rlm/context-local
 */

import { createHash, randomUUID } from 'node:crypto'
import { createReadStream } from 'node:fs'
import { mkdir, open, readdir, readFile, rename, rm, stat, writeFile } from 'node:fs/promises'
import { dirname, isAbsolute, join, resolve } from 'node:path'
import { TextDecoder } from 'node:util'
import type { Context } from '@deepseek-ai/cordis'
import {
  RlmContextStore,
  type RlmArtifactWriteOptions,
  type RlmContextOpenOptions,
  type RlmContextPutOptions,
  type RlmContextStoreStats,
  type RlmSweepOptions,
  type RlmSweepReport,
} from './context-store.ts'
import type {
  RlmArtifactRef,
  RlmContextHandle,
  RlmContextInput,
  RlmContextKind,
  RlmContextRef,
  SearchHit,
  SearchOptions,
} from './types.ts'

/** Raised when a caller reads a context it does not own. */
export class RlmAccessError extends Error {
  override readonly name = 'RlmAccessError'
}

/** Raised when a read would exceed a configured bound. */
export class RlmRangeError extends Error {
  override readonly name = 'RlmRangeError'
}

export interface LocalContextStoreConfig {
  /** Directory for payloads and metadata. */
  root: string
  /** Characters between char→byte index samples. `0` disables the index. */
  charIndexStep?: number
  /** Hard cap on one range read, in characters. */
  maxRangeChars?: number
  /** Hard cap on one stored record, in bytes. */
  maxRecordBytes?: number
  /** Do not store a per-item length list above this item count. */
  maxChunkListEntries?: number
  /** Bytes scanned by one `search()` call. */
  maxScanBytes?: number
  /** Deployment policy; encryption requires a cipher. */
  requireEncryption?: boolean
  /** Optional at-rest cipher. Absent means plaintext, which the policy may refuse. */
  cipher?: { encrypt(plain: Buffer): Promise<Buffer>; decrypt(sealed: Buffer): Promise<Buffer> }
}

interface StoredMeta {
  ref: RlmContextRef
  /**
   * Sampled mapping between character units and byte offsets (text kinds).
   * Samples never land inside a surrogate pair, so a byte offset is always a
   * valid UTF-8 sequence boundary.
   */
  charIndex?: { step: number; charOffsets: number[]; byteOffsets: number[] }
  /** Character offsets of each item (list/document kinds). */
  itemOffsets?: number[]
  /** Item count when `itemOffsets` was omitted for size reasons. */
  itemCount?: number
  /** Serialized record body for `record` kind. */
  recordJson?: string
  /**
   * Character length of the serialized payload itself.
   *
   * For a list this is *larger* than `ref.chars`, which counts only the leaf
   * strings and excludes the separators. Range reads must clamp against this
   * value, not against the leaf total.
   */
  payloadChars: number
  encrypted: boolean
  /** Internal: relative payload path, so a store can be relocated. */
  payloadPath: string
}

interface ArtifactMeta {
  ref: RlmArtifactRef
  sessionId?: string
  createdAt: number
  encrypted: boolean
  payloadPath: string
}

const DEFAULT_CHAR_INDEX_STEP = 65_536
const DEFAULT_MAX_RANGE_CHARS = 16_000_000
const DEFAULT_MAX_RECORD_BYTES = 8 * 1024 * 1024
const DEFAULT_MAX_CHUNK_LIST = 20_000
const DEFAULT_MAX_SCAN_BYTES = 16 * 1024 * 1024

export class LocalRlmContextStore extends RlmContextStore {
  override readonly providerName = 'local-content-addressed'

  private readonly root: string
  private readonly charIndexStep: number
  private readonly maxRangeChars: number
  private readonly maxRecordBytes: number
  private readonly maxChunkListEntries: number
  private readonly maxScanBytes: number
  private readonly cipher?: LocalContextStoreConfig['cipher']

  constructor(ctx: Context, config: LocalContextStoreConfig) {
    super(ctx)
    if (!config.root) throw new Error('rlm context store requires a root directory')
    if (config.requireEncryption && !config.cipher) {
      // Fail closed: an operator who asked for encryption must not silently get
      // plaintext on disk.
      throw new Error('rlm context store: requireEncryption is set but no cipher was configured')
    }
    this.root = isAbsolute(config.root) ? config.root : resolve(process.cwd(), config.root)
    this.charIndexStep = Math.max(0, Math.floor(config.charIndexStep ?? DEFAULT_CHAR_INDEX_STEP))
    this.maxRangeChars = Math.max(1, Math.floor(config.maxRangeChars ?? DEFAULT_MAX_RANGE_CHARS))
    this.maxRecordBytes = Math.max(1, Math.floor(config.maxRecordBytes ?? DEFAULT_MAX_RECORD_BYTES))
    this.maxChunkListEntries = Math.max(1, Math.floor(config.maxChunkListEntries ?? DEFAULT_MAX_CHUNK_LIST))
    this.maxScanBytes = Math.max(1, Math.floor(config.maxScanBytes ?? DEFAULT_MAX_SCAN_BYTES))
    this.cipher = config.cipher
  }

  // ----------------------------------------------------------------------- //
  // put
  // ----------------------------------------------------------------------- //

  async put(input: RlmContextInput, options: RlmContextPutOptions = {}): Promise<RlmContextRef> {
    const canonical = canonicalizeInput(input, this.maxRecordBytes, {
      charIndexStep: this.charIndexStep,
      maxChunkListEntries: this.maxChunkListEntries,
    })
    const digest = createHash('sha256')
    digest.update(canonical.kind)
    digest.update('\u0000')
    digest.update(canonical.payload)
    const id = digest.digest('hex')

    if (options.deduplicate !== false) {
      const existing = await this.readMeta(id)
      if (existing) {
        if (options.sessionId && !existing.ref.sessionId) {
          existing.ref.sessionId = options.sessionId
          await this.writeMeta(id, existing)
        }
        return existing.ref
      }
    }

    const encrypted = this.cipher !== undefined
    const payload = encrypted ? await this.cipher!.encrypt(canonical.payload) : canonical.payload
    await mkdir(this.dataDir(), { recursive: true })
    await mkdir(this.metaDir(), { recursive: true })
    const payloadPath = this.payloadPath(id)
    await writeAtomic(payloadPath, payload)

    const meta: StoredMeta = {
      ref: {
        id,
        sha256: id,
        kind: canonical.kind,
        bytes: canonical.payload.byteLength,
        chars: canonical.chars,
        ...(canonical.chunkChars ? { chunkChars: canonical.chunkChars } : {}),
        estimatedTokens: { 'heuristic:ceil(chars/4)': Math.ceil(canonical.chars / 4) },
        encoding: 'utf-8',
        ...(options.sessionId ? { sessionId: options.sessionId } : {}),
        createdAt: Date.now(),
      },
      encrypted,
      payloadPath: this.relative(payloadPath),
      payloadChars: canonical.payloadChars,
      ...(canonical.charIndex ? { charIndex: canonical.charIndex } : {}),
      ...(canonical.itemOffsets ? { itemOffsets: canonical.itemOffsets } : {}),
      ...(canonical.itemCount !== undefined ? { itemCount: canonical.itemCount } : {}),
      ...(canonical.recordJson !== undefined ? { recordJson: canonical.recordJson } : {}),
    }
    await this.writeMeta(id, meta)
    return meta.ref
  }

  // ----------------------------------------------------------------------- //
  // open / read
  // ----------------------------------------------------------------------- //

  async open(ref: RlmContextRef, options: RlmContextOpenOptions = {}): Promise<RlmContextHandle> {
    const meta = await this.requireMeta(ref.id)
    authorize(meta.ref.sessionId, options.sessionId)
    return new LocalContextHandle(this, meta)
  }

  async read(ref: RlmContextRef, options: { maxBytes?: number; signal?: AbortSignal } = {}): Promise<string> {
    const handle = await this.open(ref)
    const materialized = await handle.materialize({ maxBytes: options.maxBytes ?? this.maxRangeChars * 4 })
    return typeof materialized === 'string' ? materialized : JSON.stringify(materialized)
  }

  // ----------------------------------------------------------------------- //
  // artifacts
  // ----------------------------------------------------------------------- //

  async writeArtifact(value: string | unknown, options: RlmArtifactWriteOptions): Promise<RlmArtifactRef> {
    const isString = typeof value === 'string'
    const text = isString ? value : JSON.stringify(value)
    if (text === undefined) throw new TypeError('rlm artifact value is not serializable')
    const bytes = Buffer.byteLength(text, 'utf8')
    if (bytes > options.maxBytes) {
      throw new RlmRangeError(`artifact of ${bytes} bytes exceeds the ${options.maxBytes} byte ceiling`)
    }
    const digest = createHash('sha256').update(text).digest('hex')
    const id = `art_${digest.slice(0, 32)}`
    const previewChars = Math.max(0, Math.floor(options.previewChars ?? 2000))
    const preview = text.slice(0, previewChars)
    const ref: RlmArtifactRef = {
      id,
      sha256: digest,
      bytes,
      chars: text.length,
      contentType: options.contentType ?? (isString ? 'text/plain' : 'application/json'),
      preview,
      truncated: text.length > previewChars,
    }
    const encrypted = this.cipher !== undefined
    const payload = encrypted ? await this.cipher!.encrypt(Buffer.from(text, 'utf8')) : Buffer.from(text, 'utf8')
    await mkdir(this.artifactDir(), { recursive: true })
    const payloadPath = this.artifactPath(id)
    await writeAtomic(payloadPath, payload)
    const meta: ArtifactMeta = {
      ref,
      ...(options.sessionId ? { sessionId: options.sessionId } : {}),
      createdAt: Date.now(),
      encrypted,
      payloadPath: this.relative(payloadPath),
    }
    await writeAtomic(this.artifactMetaPath(id), Buffer.from(JSON.stringify(meta), 'utf8'))
    return ref
  }

  async readArtifact(ref: RlmArtifactRef, options: { maxBytes?: number; signal?: AbortSignal } = {}): Promise<string> {
    const raw = await readFile(this.artifactMetaPath(ref.id), 'utf8')
    const meta = JSON.parse(raw) as ArtifactMeta
    if (options.maxBytes !== undefined && meta.ref.bytes > options.maxBytes) {
      throw new RlmRangeError(`artifact of ${meta.ref.bytes} bytes exceeds the ${options.maxBytes} byte ceiling`)
    }
    const payload = await readFile(join(this.root, meta.payloadPath))
    const plain = meta.encrypted ? await this.requireCipher().decrypt(payload) : payload
    return plain.toString('utf8')
  }

  /**
   * Stream an existing file into managed artifact storage.
   *
   * Reads the source in bounded chunks and tees them into the destination while
   * hashing, so a multi-gigabyte final answer never exists as one JS string.
   */
  async writeArtifactFromFile(path: string, options: RlmArtifactWriteOptions): Promise<RlmArtifactRef> {
    const info = await stat(path)
    if (info.size > options.maxBytes) {
      throw new RlmRangeError(`artifact of ${info.size} bytes exceeds the ${options.maxBytes} byte ceiling`)
    }
    await mkdir(this.artifactDir(), { recursive: true })
    const previewChars = Math.max(0, Math.floor(options.previewChars ?? 2000))
    const digest = createHash('sha256')
    const temporary = join(this.artifactDir(), `${randomUUID()}.tmp`)
    const handle = await open(temporary, 'w')
    let preview = ''
    let chars = 0
    let bytes = 0
    // One streaming decoder for the whole file: a code point split across chunk
    // boundaries must still count as one character.
    const decoder = new TextDecoder('utf-8', { fatal: false })
    try {
      const source = createReadStream(path, { highWaterMark: 1 << 20 })
      for await (const chunk of source) {
        const buffer = chunk as Buffer
        digest.update(buffer)
        bytes += buffer.byteLength
        await handle.write(buffer)
        const decoded = decoder.decode(buffer, { stream: true })
        chars += decoded.length
        if (preview.length < previewChars) {
          preview += decoded.slice(0, previewChars - preview.length)
        }
      }
      chars += decoder.decode().length
    } finally {
      await handle.close()
    }
    const sha256 = digest.digest('hex')
    const id = `art_${sha256.slice(0, 32)}`
    const finalPath = this.artifactPath(id)
    await rename(temporary, finalPath)
    const ref: RlmArtifactRef = {
      id,
      sha256,
      bytes,
      chars,
      contentType: options.contentType ?? 'text/plain',
      preview,
      truncated: chars > previewChars,
    }
    const meta: ArtifactMeta = {
      ref,
      ...(options.sessionId ? { sessionId: options.sessionId } : {}),
      createdAt: Date.now(),
      encrypted: false,
      payloadPath: this.relative(finalPath),
    }
    await writeAtomic(this.artifactMetaPath(id), Buffer.from(JSON.stringify(meta), 'utf8'))
    return ref
  }

  // ----------------------------------------------------------------------- //
  // retention
  // ----------------------------------------------------------------------- //

  async delete(ref: RlmContextRef | RlmArtifactRef, options: { sessionId?: string } = {}): Promise<boolean> {
    try {
      if (ref.id.startsWith('art_')) {
        const meta = JSON.parse(await readFile(this.artifactMetaPath(ref.id), 'utf8')) as ArtifactMeta
        authorize(meta.sessionId, options.sessionId)
        await this.removeArtifact(ref.id)
        return true
      }
      const meta = await this.readMeta(ref.id)
      // Idempotent by contract: deleting something already gone is not an error,
      // and a retention sweep must not fail on a partially removed artifact.
      if (!meta) return false
      authorize(meta.ref.sessionId, options.sessionId)
      await this.removePayload(ref.id, meta)
      return true
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return false
      throw error
    }
  }

  /**
   * Remove one payload and its metadata.
   *
   * Deliberately unauthenticated: retention (`sweep`) is a deployment-level act
   * on expired data, and the caller-facing `delete` performs the authorization
   * check above before calling in here.
   */
  private async removePayload(id: string, meta: StoredMeta): Promise<void> {
    await rm(join(this.root, meta.payloadPath), { force: true })
    await rm(this.metaPath(id), { force: true })
  }

  private async removeArtifact(id: string): Promise<void> {
    const meta = JSON.parse(await readFile(this.artifactMetaPath(id), 'utf8')) as ArtifactMeta
    await rm(join(this.root, meta.payloadPath), { force: true })
    await rm(this.artifactMetaPath(id), { force: true })
  }

  async sweep(options: RlmSweepOptions): Promise<RlmSweepReport> {
    const now = options.now ?? Date.now()
    const report: RlmSweepReport = { scanned: 0, deleted: 0, reclaimedBytes: 0, artifacts: [] }
    for (const id of await this.listIds()) {
      const meta = await this.readMeta(id)
      if (!meta) continue
      report.scanned += 1
      if (options.sessionId && meta.ref.sessionId !== options.sessionId) continue
      if (now - meta.ref.createdAt < options.maxAgeMs) continue
      report.deleted += 1
      report.reclaimedBytes += meta.ref.bytes
      report.artifacts.push(id)
      if (!options.dryRun) await this.removePayload(id, meta)
    }
    for (const id of await this.listArtifactIds()) {
      let meta: ArtifactMeta
      try {
        meta = JSON.parse(await readFile(this.artifactMetaPath(id), 'utf8')) as ArtifactMeta
      } catch {
        continue
      }
      report.scanned += 1
      if (options.sessionId && meta.sessionId !== options.sessionId) continue
      if (now - meta.createdAt < options.maxAgeMs) continue
      report.deleted += 1
      report.reclaimedBytes += meta.ref.bytes
      report.artifacts.push(id)
      if (!options.dryRun) await this.removeArtifact(id)
    }
    return report
  }

  async stats(options: { sessionId?: string } = {}): Promise<RlmContextStoreStats> {
    let artifacts = 0
    let bytes = 0
    for (const id of [...(await this.listIds()), ...(await this.listArtifactIds())]) {
      try {
        const raw = await readFile(
          id.startsWith('art_') ? this.artifactMetaPath(id) : this.metaPath(id),
          'utf8',
        )
        const parsed = JSON.parse(raw) as StoredMeta | ArtifactMeta
        if (options.sessionId && (parsed as ArtifactMeta).sessionId !== options.sessionId) continue
        artifacts += 1
        bytes += parsed.ref.bytes
      } catch {
        continue
      }
    }
    return { provider: this.providerName, root: this.root, artifacts, bytes, encrypted: this.cipher !== undefined }
  }

  /** Release nothing but the service registration; files outlive the plugin. */
  async [Symbol.asyncDispose](): Promise<void> {
    // Deliberate no-op: artifacts are durable by contract and removed by
    // `sweep()` or an explicit `delete()`.
  }

  // ----------------------------------------------------------------------- //
  // internals used by the handle
  // ----------------------------------------------------------------------- //

  /** @internal */
  async payload(meta: StoredMeta): Promise<Buffer> {
    const payload = await readFile(join(this.root, meta.payloadPath))
    return meta.encrypted ? this.requireCipher().decrypt(payload) : payload
  }

  /** @internal Read `[start, end)` characters of a text payload, Unicode-safe. */
  async readTextRange(meta: StoredMeta, start: number, end: number): Promise<string> {
    if (end <= start) return ''
    const length = end - start
    if (length > this.maxRangeChars) {
      throw new RlmRangeError(
        `range of ${length} characters exceeds the ${this.maxRangeChars} character ceiling for one read`,
      )
    }
    const clamped = Math.max(0, Math.min(end, meta.payloadChars))
    const from = Math.max(0, Math.min(start, clamped))
    if (clamped <= from) return ''

    const index = meta.charIndex
    let byteStart = 0
    let charStart = 0
    if (index && index.charOffsets.length > 0 && from > 0) {
      // Largest sample whose character offset is <= `from`.
      let low = 0
      let high = index.charOffsets.length - 1
      while (low < high) {
        const middle = Math.ceil((low + high) / 2)
        if (index.charOffsets[middle]! <= from) low = middle
        else high = middle - 1
      }
      byteStart = index.byteOffsets[low] ?? 0
      charStart = index.charOffsets[low] ?? 0
    }

    const filePath = join(this.root, meta.payloadPath)
    if (meta.encrypted) {
      // An encrypted payload cannot be seeked; decode once and slice.
      const plain = await this.requireCipher().decrypt(await readFile(filePath))
      return plain.toString('utf8').slice(from, clamped)
    }

    const decoder = new TextDecoder('utf-8', { fatal: false })
    const stream = createReadStream(filePath, { start: byteStart, highWaterMark: 1 << 18 })
    let collected = ''
    let chars = charStart
    try {
      for await (const chunk of stream) {
        collected += decoder.decode(chunk as Buffer, { stream: true })
        chars = charStart + collected.length
        if (chars >= clamped) break
      }
      collected += decoder.decode()
    } finally {
      stream.destroy()
    }
    const offset = from - charStart
    if (offset < 0 || offset > collected.length) {
      throw new RlmRangeError(`character offset ${from} is not reachable in this payload`)
    }
    return collected.slice(offset, offset + (clamped - from))
  }

  /** @internal Read `[start, end)` items of a list/document payload. */
  async readItems(meta: StoredMeta, start: number, end: number): Promise<unknown[]> {
    const total = meta.ref.chunkChars?.length ?? meta.itemCount ?? 0
    const clamped = Math.max(0, Math.min(end, total))
    const from = Math.max(0, Math.min(start, clamped))
    if (clamped <= from) return []
    if (meta.itemOffsets) {
      // The item index turns a page into one contiguous character read.
      const first = meta.itemOffsets[from]!
      const lastStart = meta.itemOffsets[clamped - 1]!
      const lastLength = (meta.ref.chunkChars ?? [])[clamped - 1] ?? 0
      const block = await this.readTextRange(meta, first, lastStart + lastLength)
      const base = first
      const items: string[] = []
      for (let index = from; index < clamped; index += 1) {
        const itemStart = (meta.itemOffsets[index] ?? base) - base
        const itemEnd = itemStart + ((meta.ref.chunkChars ?? [])[index] ?? 0)
        items.push(block.slice(itemStart, itemEnd))
      }
      return items
    }
    const all = await this.allItems(meta)
    return all.slice(from, clamped)
  }

  /** @internal Materialize every item; only used when no item index exists. */
  async allItems(meta: StoredMeta): Promise<string[]> {
    const text = await this.payload(meta)
    const body = text.toString('utf8')
    return body.length === 0 ? [] : body.split('\n')
  }

  /** @internal */
  get scanByteCeiling(): number {
    return this.maxScanBytes
  }

  private requireCipher(): NonNullable<LocalContextStoreConfig['cipher']> {
    if (!this.cipher) throw new Error('payload is encrypted but no cipher is configured')
    return this.cipher
  }

  private relative(path: string): string {
    return path.startsWith(this.root) ? path.slice(this.root.length + 1) : path
  }

  private dataDir(): string {
    return join(this.root, 'data')
  }

  private metaDir(): string {
    return join(this.root, 'meta')
  }

  private artifactDir(): string {
    return join(this.root, 'artifacts')
  }

  private payloadPath(id: string): string {
    return join(this.dataDir(), `${id}.data`)
  }

  private metaPath(id: string): string {
    return join(this.metaDir(), `${id}.json`)
  }

  private artifactPath(id: string): string {
    return join(this.artifactDir(), `${id}.data`)
  }

  private artifactMetaPath(id: string): string {
    return join(this.artifactDir(), `${id}.meta.json`)
  }

  private async readMeta(id: string): Promise<StoredMeta | null> {
    try {
      return JSON.parse(await readFile(this.metaPath(id), 'utf8')) as StoredMeta
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return null
      throw error
    }
  }

  private async requireMeta(id: string): Promise<StoredMeta> {
    const meta = await this.readMeta(id)
    if (!meta) throw new Error(`rlm context ${id} is not available in this store`)
    return meta
  }

  private async writeMeta(id: string, meta: StoredMeta): Promise<void> {
    await writeAtomic(this.metaPath(id), Buffer.from(JSON.stringify(meta), 'utf8'))
  }

  private async listIds(): Promise<string[]> {
    return listJsonIds(this.metaDir())
  }

  private async listArtifactIds(): Promise<string[]> {
    try {
      await stat(this.artifactDir())
    } catch {
      return []
    }
    const names = await readdir(this.artifactDir())
    return names
      .filter((name) => name.endsWith('.meta.json'))
      .map((name) => name.slice(0, -'.meta.json'.length))
  }
}

/**
 * Immutable lazy handle. Kept separate from the store so that read metering,
 * authorization and range bounds live in one place per provider.
 */
class LocalContextHandle implements RlmContextHandle {
  readonly ref: RlmContextRef
  readonly kind: RlmContextKind
  readonly length: number

  constructor(private readonly store: LocalRlmContextStore, private readonly meta: StoredMeta) {
    this.ref = meta.ref
    this.kind = meta.ref.kind
    this.length = meta.ref.kind === 'text' ? meta.ref.chars : (meta.ref.chunkChars?.length ?? meta.itemCount ?? 0)
  }

  async peek(start = 0, end?: number): Promise<string> {
    return this.slice(start, end ?? Math.min(this.length, start + 10_000))
  }

  async slice(start: number, end: number): Promise<string> {
    if (this.meta.ref.kind === 'text') return this.store.readTextRange(this.meta, start, end)
    const items = await this.store.readItems(this.meta, start, end)
    return items.map((item) => (typeof item === 'string' ? item : String(item))).join('\n')
  }

  async items(start: number, end: number): Promise<unknown[]> {
    if (this.meta.ref.kind === 'text') {
      const text = await this.store.readTextRange(this.meta, start, end)
      return [...text]
    }
    return this.store.readItems(this.meta, start, end)
  }

  async get(key: string): Promise<unknown> {
    if (this.meta.recordJson) {
      const record = JSON.parse(this.meta.recordJson) as Record<string, unknown>
      if (!(key in record)) throw new Error(`context has no key ${JSON.stringify(key)}`)
      return record[key]
    }
    const items = await this.store.readItems(this.meta, 0, this.length)
    const index = Number.parseInt(key, 10)
    if (Number.isInteger(index) && index >= 0 && index < items.length) return items[index]
    throw new Error(`context has no key ${JSON.stringify(key)}`)
  }

  async search(query: string, options: SearchOptions = {}): Promise<SearchHit[]> {
    const maxHits = Math.max(1, Math.floor(options.maxHits ?? 20))
    const hits: SearchHit[] = []
    if (this.meta.ref.kind === 'text') {
      const scanned = Math.min(this.meta.ref.chars, Math.floor(this.store.scanByteCeiling))
      const text = await this.store.readTextRange(this.meta, 0, scanned)
      const haystack = options.caseSensitive ? text : text.toLowerCase()
      const needle = options.caseSensitive ? query : query.toLowerCase()
      if (options.regex) {
        const expression = new RegExp(query, options.caseSensitive ? 'g' : 'gi')
        let match: RegExpExecArray | null
        while ((match = expression.exec(text)) !== null && hits.length < maxHits) {
          hits.push({ start: match.index, end: match.index + match[0].length, text: match[0] })
          if (match[0].length === 0) expression.lastIndex += 1
        }
        return hits
      }
      let cursor = 0
      while (hits.length < maxHits) {
        const found = haystack.indexOf(needle, cursor)
        if (found < 0) break
        hits.push({ start: found, end: found + query.length, text: text.slice(found, found + query.length) })
        cursor = found + Math.max(1, query.length)
      }
      return hits
    }
    const items = await this.store.readItems(this.meta, 0, this.length)
    for (let index = 0; index < items.length && hits.length < maxHits; index += 1) {
      const item = String(items[index])
      const haystack = options.caseSensitive ? item : item.toLowerCase()
      const needle = options.caseSensitive ? query : query.toLowerCase()
      const found = haystack.indexOf(needle)
      if (found >= 0) {
        hits.push({ item: index, start: found, end: found + query.length, text: item.slice(found, found + query.length) })
      }
    }
    return hits
  }

  async materialize(options: { maxBytes: number }): Promise<unknown> {
    if (this.meta.ref.bytes > options.maxBytes) {
      throw new RlmRangeError(
        `context of ${this.meta.ref.bytes} bytes exceeds the ${options.maxBytes} byte ceiling for one materialization`,
      )
    }
    if (this.meta.recordJson) return JSON.parse(this.meta.recordJson)
    const payload = await this.store.payload(this.meta)
    const text = payload.toString('utf8')
    if (this.meta.ref.kind === 'text') return text
    return text.length === 0 ? [] : text.split('\n')
  }
}

// --------------------------------------------------------------------------- //
// helpers
// --------------------------------------------------------------------------- //

function authorize(owner: string | undefined, requester: string | undefined): void {
  if (owner !== undefined && requester !== owner) {
    throw new RlmAccessError(`rlm context belongs to session ${owner}, not ${requester ?? '<anonymous>'}`)
  }
}

async function writeAtomic(path: string, payload: Buffer): Promise<void> {
  await mkdir(dirname(path), { recursive: true })
  const temporary = `${path}.${randomUUID()}.tmp`
  await writeFile(temporary, payload)
  await rename(temporary, path)
}

async function listJsonIds(directory: string): Promise<string[]> {
  try {
    await stat(directory)
  } catch {
    return []
  }
  const names = await readdir(directory)
  return names
    .filter((name) => name.endsWith('.json'))
    .map((name) => name.slice(0, -'.json'.length))
    .filter((name) => !name.endsWith('.meta'))
}

interface CanonicalInput {
  kind: RlmContextKind
  payload: Buffer
  chars: number
  chunkChars?: number[]
  itemOffsets?: number[]
  itemCount?: number
  recordJson?: string
  charIndex?: { step: number; charOffsets: number[]; byteOffsets: number[] }
  payloadChars: number
}

/**
 * Serialize an accepted input into the canonical bytes that the content hash is
 * computed over. `text` stores raw characters; list/document shapes store
 * newline-joined items plus a character index, which is what makes item paging
 * cheap without keeping a JSON parse of a gigabyte in memory.
 */
export function canonicalizeInput(
  input: RlmContextInput,
  maxRecordBytes: number,
  options: { charIndexStep?: number; maxChunkListEntries?: number } = {},
): CanonicalInput {
  const charIndexStep = options.charIndexStep ?? DEFAULT_CHAR_INDEX_STEP
  const maxChunkListEntries = options.maxChunkListEntries ?? DEFAULT_MAX_CHUNK_LIST
  switch (input.kind) {
    case 'text': {
      const payload = Buffer.from(input.text, 'utf8')
      return {
        kind: 'text',
        payload,
        chars: input.text.length,
        payloadChars: input.text.length,
        charIndex: buildCharIndex(input.text, charIndexStep),
      }
    }
    case 'list':
    case 'documents': {
      const items = input.kind === 'list' ? input.items : input.documents
      const body = items.join('\n')
      // Above the cap the per-item index is dropped rather than stored: a huge
      // index would itself become the memory problem the store exists to avoid.
      const indexed = items.length <= maxChunkListEntries
      const itemOffsets: number[] = []
      const chunkChars: number[] = []
      if (indexed) {
        let cursor = 0
        for (const item of items) {
          itemOffsets.push(cursor)
          chunkChars.push(item.length)
          cursor += item.length + 1
        }
      }
      return {
        kind: input.kind,
        payload: Buffer.from(body, 'utf8'),
        chars: items.reduce((total, item) => total + item.length, 0),
        payloadChars: body.length,
        ...(indexed ? { itemOffsets, chunkChars } : {}),
        itemCount: items.length,
      }
    }
    case 'record': {
      const recordJson = JSON.stringify(input.record)
      const payload = Buffer.from(recordJson, 'utf8')
      if (payload.byteLength > maxRecordBytes) {
        throw new RlmRangeError(
          `record of ${payload.byteLength} bytes exceeds the ${maxRecordBytes} byte ceiling for records`,
        )
      }
      return { kind: 'record', payload, chars: recordJson.length, payloadChars: recordJson.length, recordJson }
    }
    default: {
      throw new TypeError('unsupported rlm context input')
    }
  }
}

function buildCharIndex(text: string, step: number): CanonicalInput['charIndex'] {
  const effectiveStep = Math.max(1, step)
  const charOffsets: number[] = [0]
  const byteOffsets: number[] = [0]
  let bytes = 0
  let start = 0
  while (start < text.length) {
    let end = Math.min(text.length, start + effectiveStep)
    // Never sample inside a surrogate pair: a split pair would encode as a
    // replacement character and make every later byte offset wrong.
    if (end < text.length && isHighSurrogate(text.charCodeAt(end - 1))) end += 1
    bytes += Buffer.byteLength(text.slice(start, end), 'utf8')
    charOffsets.push(end)
    byteOffsets.push(bytes)
    start = end
  }
  return { step: effectiveStep, charOffsets, byteOffsets }
}

function isHighSurrogate(code: number): boolean {
  return code >= 0xd800 && code <= 0xdbff
}

export default LocalRlmContextStore
