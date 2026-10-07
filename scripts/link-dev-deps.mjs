#!/usr/bin/env node
/**
 * Wire the local DSH installation into `node_modules/` for development.
 *
 * A dsh plugin must share the harness's *instances* of cordis and the dsh
 * services, so its peer dependencies have to resolve to the very packages the
 * running harness uses. When this package is installed into a profile, the
 * profile's package manager does that. For local development — and for the
 * sandboxed CI of this exercise, which cannot reach a registry — the honest
 * equivalent is a symlink farm pointing at one dsh installation.
 *
 * Usage:
 *   node scripts/link-dev-deps.mjs [--deps-root <dir>] [--dry-run]
 *
 * `--deps-root` must be the directory that *contains* `@deepseek-ai/` and
 * `@types/`. Without it, the script probes, in order:
 *   $DSH_RLM_DEPS_ROOT
 *   the node_modules of the `dsh` binary found on PATH (../.. from its realpath)
 *   the node_modules of a local deepseek-harness checkout
 */

import { createRequire } from 'node:module'
import { existsSync } from 'node:fs'
import { mkdir, readlink, symlink, unlink } from 'node:fs/promises'
import { execFileSync } from 'node:child_process'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const require = createRequire(import.meta.url)
const here = resolve(dirname(fileURLToPath(import.meta.url)), '..')

/** Peer dependencies that must resolve to the harness's own copies. */
const PEERS = [
  '@deepseek-ai/cordis',
  '@deepseek-ai/dsh-agent',
  '@deepseek-ai/dsh-llm',
  '@deepseek-ai/dsh-session',
  '@deepseek-ai/dsh-session-projection',
  '@deepseek-ai/dsh-system-prompt',
  '@deepseek-ai/dsh-tools',
  '@deepseek-ai/schemastery',
]

/** Type-only packages the compiler needs. */
const TYPES = ['@types/node']

function parseArgs(argv) {
  const options = { depsRoot: process.env.DSH_RLM_DEPS_ROOT, dryRun: false }
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] === '--deps-root') options.depsRoot = argv[++index]
    else if (argv[index] === '--dry-run') options.dryRun = true
  }
  return options
}

function dshBinaryNodeModules() {
  try {
    const binary = execFileSync('which', ['dsh'], { encoding: 'utf8' }).trim()
    if (!binary) return undefined
    const real = execFileSync('readlink', ['-f', binary], { encoding: 'utf8' }).trim() || binary
    // <root>/lib/bin.js -> <root> is the @deepseek-ai/dsh package
    return join(dirname(dirname(real)), 'node_modules')
  } catch {
    return undefined
  }
}

function harnessCheckoutNodeModules() {
  const candidates = [
    '/Users/lufeng/Workspace/github/deepseek-harness/node_modules',
    resolve(here, '../../../../../../deepseek-harness/node_modules'),
  ]
  return candidates.find((candidate) => existsSync(join(candidate, '@deepseek-ai/cordis')))
}

function findDepsRoot(explicit) {
  const candidates = [explicit, dshBinaryNodeModules(), harnessCheckoutNodeModules()].filter(Boolean)
  for (const candidate of candidates) {
    if (existsSync(join(candidate, '@deepseek-ai/cordis'))) return resolve(candidate)
  }
  throw new Error(
    `could not find a dsh installation providing @deepseek-ai/cordis in: ${candidates.join(', ')}. ` +
      'Pass --deps-root <dir> or set DSH_RLM_DEPS_ROOT.',
  )
}

async function link(targetDir, packageName, source, dryRun) {
  const linkPath = join(targetDir, packageName)
  const sourcePath = join(source, packageName)
  if (!existsSync(sourcePath)) return { packageName, status: 'missing' }
  if (existsSync(linkPath)) {
    const current = await readlink(linkPath).catch(() => undefined)
    if (current === sourcePath) return { packageName, status: 'current' }
    if (current === undefined) return { packageName, status: 'present-not-link' }
    if (!dryRun) {
      await unlink(linkPath)
    }
  }
  if (!dryRun) {
    await mkdir(dirname(linkPath), { recursive: true })
    await symlink(sourcePath, linkPath, 'dir')
  }
  return { packageName, status: 'linked' }
}

async function main() {
  const options = parseArgs(process.argv.slice(2))
  const depsRoot = findDepsRoot(options.depsRoot)
  const targetDir = join(here, 'node_modules')
  console.log(`dsh dependencies: ${depsRoot}`)
  console.log(`link target:      ${targetDir}${options.dryRun ? ' (dry run)' : ''}`)
  const results = []
  for (const peer of PEERS) results.push(await link(targetDir, peer, depsRoot, options.dryRun))
  for (const type of TYPES) results.push(await link(targetDir, type, depsRoot, options.dryRun))
  for (const result of results) console.log(`  ${result.status.padEnd(18)} ${result.packageName}`)
  const missing = results.filter((result) => result.status === 'missing')
  if (missing.length > 0) {
    console.error(`missing packages in ${depsRoot}: ${missing.map((m) => m.packageName).join(', ')}`)
    process.exitCode = 1
  }
}

await main()
