/**
 * Loadable plugin: isolated persistent Python RLM kernel provider.
 *
 * The provider name is deliberately explicit (`python-subprocess`) and its
 * declared capabilities are deliberately modest: a bare subprocess cannot
 * enforce network isolation or process-tree CPU/RSS quotas, and this plugin
 * refuses to pretend otherwise. Operators who need that enforcement wrap the
 * interpreter with `isolationCommand` (bwrap / sandbox-exec / a container
 * runtime) and declare `isolationCommandDisablesNetwork`.
 *
 * @module dsh-rlm/plugins/kernel-python
 */

import type { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import { RlmPythonKernelRuntime } from '../kernel/python/provider.ts'

export const name = 'rlm-kernel-python'

/** No injected services: the provider owns its own processes. */
export const inject: string[] = []

export interface Config {
  /** Interpreter for the driver. */
  pythonPath?: string
  /** Absolute path to `rlm_kernel.py`; defaults to the shipped copy. */
  driverPath?: string
  /** Hard ceiling on one control frame in either direction. */
  maxFrameBytes?: number
  /** Grace period between SIGTERM and SIGKILL of the process group. */
  killGraceMs?: number
  /** Install guest-level socket/subprocess guards (defense in depth only). */
  guestNetworkGuard?: boolean
  /**
   * Wrapper command prepended to the driver invocation, e.g.
   * `['bwrap', '--unshare-net', '--ro-bind', '/', '/']`. The whole point is that
   * the isolation story lives in configuration, not in a comment.
   */
  isolationCommand?: string[]
  /** Declare that `isolationCommand` really denies network access. */
  isolationCommandDisablesNetwork?: boolean
  /** Extra environment for the child, on top of the scrubbed base. */
  env?: Record<string, string>
}

export const Config: z<Config> = z.object({
  pythonPath: z.string().default(process.env.DSH_RLM_PYTHON ?? 'python3'),
  driverPath: z.string(),
  maxFrameBytes: z.natural().default(64 * 1024 * 1024),
  killGraceMs: z.natural().default(2000),
  guestNetworkGuard: z.boolean().default(true),
  isolationCommand: z.array(z.string()),
  isolationCommandDisablesNetwork: z.boolean().default(false),
  env: z.dict(z.string()),
})

/** Mount `ctx.rlmKernelRuntime` for the lifetime of this plugin. */
export function apply(ctx: Context, config: Config): void {
  ctx.effect(() => {
    const runtime = new RlmPythonKernelRuntime(ctx, {
      ...(config?.pythonPath ? { pythonPath: config.pythonPath } : {}),
      ...(config?.driverPath ? { driverPath: config.driverPath } : {}),
      maxFrameBytes: config?.maxFrameBytes ?? 64 * 1024 * 1024,
      killGraceMs: config?.killGraceMs ?? 2000,
      guestNetworkGuard: config?.guestNetworkGuard ?? true,
      ...(config?.isolationCommand && config.isolationCommand.length > 0
        ? { isolationCommand: [...config.isolationCommand] }
        : {}),
      isolationCommandDisablesNetwork: config?.isolationCommandDisablesNetwork ?? false,
      ...(config?.env ? { env: { ...config.env } } : {}),
    })
    return async () => {
      // Unload must terminate every live kernel and every grandchild process;
      // leaving one behind is a hard release failure, not a warning.
      await runtime.disposeAll('rlm-kernel-python plugin unloaded')
    }
  })
}
