/**
 * `@deepseek-ai/dsh-rlm` — the paper-aligned Recursive Language Model plugin.
 *
 * Mounts, for the lifetime of the plugin only:
 *
 * ```text
 * ctx.rlm                 RlmService              (this plugin)
 * ctx.rlmContextStore     RlmContextStore         (dsh-rlm/context-local)
 * ctx.rlmKernelRuntime    RlmKernelRuntime        (dsh-rlm/kernel-python)
 * run_rlm                 model-facing tool       (this plugin)
 * ```
 *
 * The plugin is absent from the default composition on purpose: `inject` lists
 * `tools`, `llm`, `rlmContextStore`, `rlmKernelRuntime` and `systemPrompt`, so
 * cordis holds this plugin PENDING until a real store and a real kernel provider
 * are mounted, and silently unloads it if either disappears. There is no silent
 * fallback to an unbounded or unsandboxed implementation — a missing dependency
 * means no RLM, never a degraded one.
 *
 * @module @deepseek-ai/dsh-rlm
 */

import { resolve } from 'node:path'
import type { Context } from '@deepseek-ai/cordis'
import { createDefaultSink } from './events.ts'
import { createDefaultServiceConfig, resolvePluginConfig, type Config, type ResolvedRlmPluginConfig } from './config.ts'
import { DshModelClient } from './llm.ts'
import { RlmService } from './service.ts'
import { registerRunRlmTool } from './tool.ts'

export const name = 'rlm'

/**
 * Hard requirements. `subagents` and `tools`-adjacent capabilities that the
 * README calls optional are intentionally NOT injected: paper-correct RLM
 * execution needs a store, a kernel and a model, nothing more.
 */
export const inject = ['tools', 'llm', 'rlmContextStore', 'rlmKernelRuntime', 'systemPrompt']

export type { ResolvedRlmPluginConfig } from './config.ts'
export { Config } from './config.ts'

/**
 * Mount the RLM service, its tool and its usage-policy prompt section.
 *
 * The incoming config is the *raw* profile row; resolution happens here so a
 * programmatic `ctx.plugin(rlm, { … })` and a YAML row behave identically and no
 * limit can be silently `undefined`.
 */
export function apply(ctx: Context, config: Config | ResolvedRlmPluginConfig): void {
  const resolved = resolvePluginConfig(config as Config)
  ctx.effect(() => {
    const config = resolved
    const serviceConfig = createDefaultServiceConfig(config)
    const events = createDefaultSink({
      root: resolve(config.contextStoreRoot),
      ...(config.onSessionEventSkipped === undefined ? {} : { onSkip: config.onSessionEventSkipped }),
    })
    const service = new RlmService(ctx, serviceConfig, {
      contextStore: ctx.rlmContextStore,
      kernelRuntime: ctx.rlmKernelRuntime,
      model: new DshModelClient(ctx.llm),
      events,
    })
    const unregisterPrompt = ctx.systemPrompt.section({
      name: `tool:${config.toolName}`,
      order: ctx.systemPrompt.getSectionOrder('TOOL_WORKFLOW'),
      text: runRlmUsagePolicy(config, service),
    })
    const unregisterTool = registerRunRlmTool(ctx, service, {
      toolName: config.toolName,
      description: RUN_RLM_DESCRIPTION,
      timeoutMs: serviceConfig.budgets.maxWallClockMs,
      rootRoute: config.rootRoute,
      leafRoute: config.leafRoute,
      ...(config.childRoute === undefined ? {} : { childRoute: config.childRoute }),
      maxTaskChars: config.maxTaskChars,
      maxRenderedChars: config.maxRenderedChars,
      writeSessionEvents: config.writeSessionEvents,
      baseSink: events,
    })

    return async () => {
      unregisterTool()
      unregisterPrompt()
      // Unload order matters: reject new runs, abort kernels, flush traces.
      await service.shutdown('rlm plugin unloaded')
    }
  })
}

const RUN_RLM_DESCRIPTION = [
  'Run a Recursive Language Model (RLM): the prompt is stored outside this conversation as a symbolic handle',
  'and a persistent Python REPL processes it with programmatic llm_query() sub-calls, repeating code execution',
  'until a final answer is produced.',
  '',
  'Use it when the user offers a prompt or corpus far larger than what fits comfortably in one context, or when the',
  'task needs dense (linear or quadratic) semantic work over nearly every part of it. Do NOT use it for ordinary',
  'questions, for a single code snippet, or when a plain retrieval/compaction step is enough.',
].join('\n')

function runRlmUsagePolicy(config: ResolvedRlmPluginConfig, service: RlmService): string {
  const parts = [
    `Use the ${config.toolName} tool when the working material is much larger than one comfortable context, or when every part of it must be semantically processed (small supporting text does not need it).`,
    'The prompt is externalized: the RLM root model reads it from a REPL handle named `context` and never receives it as conversation history.',
    'Sub-calls (`llm_query`) cost real tokens and money, and the run is bounded by the deployment budget. Prefer one call on a large, related slice over many tiny calls.',
  ]
  if (config.defaultDepth === 0) {
    parts.push(
      'This deployment runs at depth 0: the REPL can read the prompt but cannot call models, so recursion is unavailable.',
    )
  } else if (config.maxDepth >= 2) {
    parts.push(
      `Depth is limited to ${config.maxDepth}: child RLMs may be spawned with \`rlm_query\`, and they inherit the remaining depth and the remaining budget rather than a fresh copy.`,
    )
  } else {
    parts.push('Depth is limited to 1: child runs are ordinary leaf model calls, not recursive RLMs.')
  }
  for (const gap of service.degradations) {
    parts.push(`NOTE (operator): the kernel provider cannot prove "${gap}"; treat model-generated code as trusted-local.`)
  }
  return parts.join(' ')
}

export { RlmService } from './service.ts'
export { LocalRlmContextStore } from './context-local.ts'
export { RlmPythonKernelRuntime } from './kernel/python/provider.ts'
export { RlmContextStore } from './context-store.ts'
export { RlmKernelRuntime } from './kernel/index.ts'
export {
  CompositeEventSink,
  RecordingEventSink,
  SessionEventSink,
  SidecarJsonlEventSink,
  aggregateRlmMetrics,
  foldRlmRuns,
} from './events.ts'
export { ScriptedModelClient, InMemoryKernelRuntime } from './testing.ts'
export { parseRootAction, extractFences, findFinalMatches } from './finalize.ts'
