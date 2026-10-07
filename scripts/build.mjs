#!/usr/bin/env node
/**
 * Build the plugin: `tsc` the TypeScript sources and copy the Python kernel
 * driver next to the compiled provider.
 *
 * The driver must sit beside `lib/kernel/python/provider.js`, because the
 * provider resolves it relative to its own module URL — the built layout has to
 * mirror the source layout or a packaged install breaks at runtime only.
 *
 * Usage:
 *   node scripts/build.mjs [--noEmit] [--quiet]
 */

import { cp, mkdir, rm } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import { execFileSync } from 'node:child_process'
import { createRequire } from 'node:module'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const require = createRequire(import.meta.url)

function findTsc() {
  const candidates = [
    join(here, 'node_modules', '.bin', 'tsc'),
    process.env.DSH_RLM_TSC,
    '/Users/lufeng/Workspace/github/deepseek-harness/node_modules/.bin/tsc',
  ].filter(Boolean)
  for (const candidate of candidates) {
    if (existsSync(candidate)) return candidate
  }
  try {
    return require.resolve('typescript/bin/tsc')
  } catch {
    throw new Error('TypeScript not found. Run scripts/link-dev-deps.mjs first, or set DSH_RLM_TSC.')
  }
}

async function main() {
  const args = process.argv.slice(2)
  const noEmit = args.includes('--noEmit')
  const quiet = args.includes('--quiet')
  const tsc = findTsc()
  const tscArgs = ['-p', join(here, 'tsconfig.json')]
  if (noEmit) tscArgs.push('--noEmit')
  if (!quiet) console.log(`tsc: ${tsc}`)
  execFileSync(tsc, tscArgs, { cwd: here, stdio: 'inherit' })
  if (noEmit) {
    if (!quiet) console.log('typecheck ok (no emit)')
    return
  }
  const sourceDriver = join(here, 'src', 'kernel', 'python', 'rlm_kernel.py')
  const targetDir = join(here, 'lib', 'kernel', 'python')
  await mkdir(targetDir, { recursive: true })
  await cp(sourceDriver, join(targetDir, 'rlm_kernel.py'))
  await rm(join(here, 'lib', 'kernel', 'python', 'test_rlm_kernel.py'), { force: true })
  if (!quiet) console.log(`copied rlm_kernel.py -> ${targetDir}`)
}

await main()
