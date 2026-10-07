/**
 * Loadable plugin: local content-addressed RLM context/artifact store.
 *
 * Ships separately from the RLM core so a deployment can swap storage
 * (encrypted, tenant-isolated, remote) without touching RLM semantics, and so
 * telemetry can name the provider that actually served a prompt.
 *
 * @module dsh-rlm/plugins/context-local
 */

import { resolve } from 'node:path'
import type { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import { LocalRlmContextStore } from '../context-local.ts'

export const name = 'rlm-context-local'

/** No injected services: this provider depends only on the filesystem. */
export const inject: string[] = []

export interface Config {
  /** Directory for payloads, metadata and artifact traces. */
  root?: string
  /** Characters between char→byte index samples; 0 disables the index. */
  charIndexStep?: number
  /** Hard cap on one range read, in characters. */
  maxRangeChars?: number
  /** Hard cap on one stored record, in bytes. */
  maxRecordBytes?: number
  /** Refuse to start when set and no cipher is configured (fail closed). */
  requireEncryption?: boolean
  /** Bytes scanned by one `search()` call. */
  maxScanBytes?: number
}

export const Config: z<Config> = z.object({
  root: z.string().default('.dsh/rlm-artifacts'),
  charIndexStep: z.natural().default(65_536),
  maxRangeChars: z.natural().default(16_000_000),
  maxRecordBytes: z.natural().default(8 * 1024 * 1024),
  requireEncryption: z.boolean().default(false),
  maxScanBytes: z.natural().default(16 * 1024 * 1024),
})

/** Mount `ctx.rlmContextStore` for the lifetime of this plugin. */
export function apply(ctx: Context, config: Config): void {
  const resolved = resolveConfig(config)
  ctx.effect(() => {
    const store = new LocalRlmContextStore(ctx, {
      root: resolved.root,
      charIndexStep: resolved.charIndexStep,
      maxRangeChars: resolved.maxRangeChars,
      maxRecordBytes: resolved.maxRecordBytes,
      requireEncryption: resolved.requireEncryption,
      maxScanBytes: resolved.maxScanBytes,
    })
    return () => {
      void store
    }
  })
}

/** Resolve defaults defensively: a loader may call `apply` without normalization. */
function resolveConfig(config: Config | undefined): Required<Omit<Config, 'root'>> & { root: string } {
  const root = config?.root && config.root.length > 0 ? resolve(config.root) : resolve('.dsh/rlm-artifacts')
  return {
    root,
    charIndexStep: config?.charIndexStep ?? 65_536,
    maxRangeChars: config?.maxRangeChars ?? 16_000_000,
    maxRecordBytes: config?.maxRecordBytes ?? 8 * 1024 * 1024,
    requireEncryption: config?.requireEncryption ?? false,
    maxScanBytes: config?.maxScanBytes ?? 16 * 1024 * 1024,
  }
}
