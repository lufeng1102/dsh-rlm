/**
 * Plugin configuration: the YAML surface and its resolution into host policy.
 *
 * Every numeric limit here is a *hard host ceiling*, read from configuration and
 * never from a prompt or from model-written code. That is the property that
 * matters: a deployment's worst case must be knowable by an operator without
 * reading a single model trajectory (README §5).
 *
 * @module dsh-rlm/config
 */

import z from '@deepseek-ai/schemastery'
import { DEFAULT_BUDGETS, type RlmRoute, type RlmServiceConfig } from './service.ts'
import { DEFAULT_ISOLATION_POLICY } from './kernel/index.ts'
import { PAPER_PROFILES, type RlmModelProfile } from './prompt.ts'
import type { RlmBudgetLimits } from './types.ts'
import type { RlmPricing } from './llm.ts'

export interface Config {
  /** Model-facing tool name. */
  toolName: string
  /** Recursion depth used when a call does not specify one; the paper uses 1. */
  defaultDepth: number
  /** Hard ceiling on any requested depth. */
  maxDepth: number
  /** Root-loop model route. */
  rootProvider?: string
  rootModel?: string
  /** Leaf `llm_query` route; defaults to the root route. */
  leafProvider?: string
  leafModel?: string
  /** Child `rlm_query` route; defaults to the leaf route. */
  childProvider?: string
  childModel?: string
  /** Named prompt profile from the shipped set. */
  promptProfileId: string
  /** Override the profile's declared context window, in tokens. */
  rootContextWindowTokens?: number
  /** Characters a leaf sub-call can absorb, per the profile. */
  subCallChars?: number
  /** Refuse to run when the profile is not marked qualified. */
  requireQualifiedProfile: boolean
  rootMaxTokens: number
  leafMaxTokens: number
  temperature?: number
  maxRootIterations: number
  maxWallClockMs: number
  maxTotalQueries: number
  maxConcurrentQueries: number
  maxTotalKernels: number
  maxConcurrentKernels: number
  maxInputTokens: number
  maxOutputTokens: number
  maxContextReadBytes: number
  maxQueryPromptChars: number
  maxInlineOutputBytes: number
  maxArtifactBytes: number
  maxCostUsd: number
  /** Storage/admission ceiling for one prompt. */
  maxPromptChars: number
  /** Bound on the root conversation; the bounded-history witness. */
  maxRootHistoryChars: number
  maxParseRetries: number
  maxCodePreviewChars: number
  maxStdoutPreviewChars: number
  maxErrorPreviewChars: number
  /** Longest result text pushed back into the caller's model window. */
  maxRenderedChars: number
  /** Ceiling on a `prompt` argument when material travels as `context`. */
  maxTaskChars: number
  finalVariableName: string
  /** Artifact/trace root for context and RLM trajectories. */
  contextStoreRoot: string
  /** Write `rlm/*` events to the session log when the build declares them. */
  writeSessionEvents: boolean
  /** Per-route pricing, used only to fill the cost counter honestly. */
  pricing: Record<string, RlmPricing>
}

/** Schemastery schema for the profile YAML row. */
export const Config: z<Config> = z.object({
  toolName: z.string().default('run_rlm'),
  defaultDepth: z.natural().default(1),
  maxDepth: z.natural().default(3),
  rootProvider: z.string(),
  rootModel: z.string(),
  leafProvider: z.string(),
  leafModel: z.string(),
  childProvider: z.string(),
  childModel: z.string(),
  promptProfileId: z.string().default('paper-gpt5'),
  rootContextWindowTokens: z.natural(),
  subCallChars: z.natural(),
  requireQualifiedProfile: z.boolean().default(false),
  rootMaxTokens: z.natural().default(32_000),
  leafMaxTokens: z.natural().default(16_000),
  temperature: z.number(),
  maxRootIterations: z.natural().default(64),
  maxWallClockMs: z.natural().default(900_000),
  maxTotalQueries: z.natural().default(4096),
  maxConcurrentQueries: z.natural().default(16),
  maxTotalKernels: z.natural().default(8),
  maxConcurrentKernels: z.natural().default(8),
  maxInputTokens: z.natural().default(8_000_000),
  maxOutputTokens: z.natural().default(1_000_000),
  maxContextReadBytes: z.natural().default(512 * 1024 * 1024),
  maxQueryPromptChars: z.natural().default(2_000_000),
  maxInlineOutputBytes: z.natural().default(1024 * 1024),
  maxArtifactBytes: z.natural().default(1024 * 1024 * 1024),
  maxCostUsd: z.natural().default(25),
  maxPromptChars: z.natural().default(100_000_000),
  maxRootHistoryChars: z.natural().default(400_000),
  maxParseRetries: z.natural().default(2),
  maxCodePreviewChars: z.natural().default(4_000),
  maxStdoutPreviewChars: z.natural().default(2_000),
  maxErrorPreviewChars: z.natural().default(1_500),
  maxRenderedChars: z.natural().default(20_000),
  maxTaskChars: z.natural().default(8_000),
  finalVariableName: z.string().default('Final'),
  contextStoreRoot: z.string().default('.dsh/rlm-artifacts'),
  writeSessionEvents: z.boolean().default(true),
  pricing: z.dict(
    z.object({
      inputUsdPerMTok: z.number().default(0),
      outputUsdPerMTok: z.number().default(0),
    }),
  ),
})

/** A configuration with every default materialized. */
export interface ResolvedRlmPluginConfig extends Config {
  rootRoute: RlmRoute
  leafRoute: RlmRoute
  childRoute?: RlmRoute
  budgets: RlmBudgetLimits
  promptProfile: RlmModelProfile
  /** Hook for the one-time notice that session events go to the sidecar instead. */
  onSessionEventSkipped?: (type: string, reason: string) => void
}

/**
 * Apply defaults and derive the route/profile/budget set.
 *
 * Called by `apply()` as well as by the loader-normalized path, so a programmatic
 * `ctx.plugin(rlm, { ... })` cannot silently get `undefined` limits.
 */
export function resolvePluginConfig(config: Partial<Config> = {}): ResolvedRlmPluginConfig {
  const base = PAPER_PROFILES[config.promptProfileId ?? 'paper-gpt5']
  if (!base) {
    throw new Error(
      `unknown promptProfileId ${JSON.stringify(config.promptProfileId)}; known profiles: ${Object.keys(PAPER_PROFILES).join(', ')}`,
    )
  }
  const rootRoute: RlmRoute = {
    provider: config.rootProvider ?? base.root.provider,
    model: config.rootModel ?? base.root.model,
  }
  const leafRoute: RlmRoute = {
    provider: config.leafProvider ?? config.rootProvider ?? base.leaf.provider,
    model: config.leafModel ?? config.rootModel ?? base.leaf.model,
  }
  const childRoute: RlmRoute | undefined =
    config.childProvider !== undefined || config.childModel !== undefined
      ? {
          provider: config.childProvider ?? leafRoute.provider,
          model: config.childModel ?? leafRoute.model,
        }
      : undefined
  const promptProfile: RlmModelProfile = {
    ...base,
    root: rootRoute,
    leaf: leafRoute,
    ...(childRoute === undefined ? {} : { child: childRoute }),
    ...(config.rootContextWindowTokens === undefined ? {} : { contextWindowTokens: config.rootContextWindowTokens }),
    ...(config.subCallChars === undefined ? {} : { subCallChars: config.subCallChars }),
  }
  const budgets: RlmBudgetLimits = {
    ...DEFAULT_BUDGETS,
    maxRootIterations: config.maxRootIterations ?? DEFAULT_BUDGETS.maxRootIterations,
    maxWallClockMs: config.maxWallClockMs ?? DEFAULT_BUDGETS.maxWallClockMs,
    maxTotalQueries: config.maxTotalQueries ?? DEFAULT_BUDGETS.maxTotalQueries,
    maxConcurrentQueries: config.maxConcurrentQueries ?? DEFAULT_BUDGETS.maxConcurrentQueries,
    maxTotalKernels: config.maxTotalKernels ?? DEFAULT_BUDGETS.maxTotalKernels,
    maxConcurrentKernels: config.maxConcurrentKernels ?? DEFAULT_BUDGETS.maxConcurrentKernels,
    maxInputTokens: config.maxInputTokens ?? DEFAULT_BUDGETS.maxInputTokens,
    maxOutputTokens: config.maxOutputTokens ?? DEFAULT_BUDGETS.maxOutputTokens,
    maxContextReadBytes: config.maxContextReadBytes ?? DEFAULT_BUDGETS.maxContextReadBytes,
    maxQueryPromptChars: config.maxQueryPromptChars ?? DEFAULT_BUDGETS.maxQueryPromptChars,
    maxInlineOutputBytes: config.maxInlineOutputBytes ?? DEFAULT_BUDGETS.maxInlineOutputBytes,
    maxArtifactBytes: config.maxArtifactBytes ?? DEFAULT_BUDGETS.maxArtifactBytes,
    maxCostUsd: config.maxCostUsd ?? DEFAULT_BUDGETS.maxCostUsd,
  }
  const maxDepth = Math.max(0, config.maxDepth ?? 3)
  const defaultDepth = Math.min(Math.max(0, config.defaultDepth ?? 1), maxDepth)
  const pricing: Record<string, RlmPricing> = {}
  for (const [key, value] of Object.entries(config.pricing ?? {})) {
    pricing[key] = {
      inputUsdPerMTok: value?.inputUsdPerMTok ?? 0,
      outputUsdPerMTok: value?.outputUsdPerMTok ?? 0,
    }
  }
  const resolved: ResolvedRlmPluginConfig = {
    toolName: config.toolName ?? 'run_rlm',
    defaultDepth,
    maxDepth,
    rootProvider: rootRoute.provider,
    rootModel: rootRoute.model,
    leafProvider: leafRoute.provider,
    leafModel: leafRoute.model,
    ...(childRoute === undefined ? {} : { childProvider: childRoute.provider, childModel: childRoute.model }),
    promptProfileId: config.promptProfileId ?? 'paper-gpt5',
    ...(config.rootContextWindowTokens === undefined
      ? {}
      : { rootContextWindowTokens: config.rootContextWindowTokens }),
    ...(config.subCallChars === undefined ? {} : { subCallChars: config.subCallChars }),
    requireQualifiedProfile: config.requireQualifiedProfile ?? false,
    rootMaxTokens: config.rootMaxTokens ?? 32_000,
    leafMaxTokens: config.leafMaxTokens ?? 16_000,
    ...(config.temperature === undefined ? {} : { temperature: config.temperature }),
    maxRootIterations: budgets.maxRootIterations,
    maxWallClockMs: budgets.maxWallClockMs,
    maxTotalQueries: budgets.maxTotalQueries,
    maxConcurrentQueries: budgets.maxConcurrentQueries,
    maxTotalKernels: budgets.maxTotalKernels,
    maxConcurrentKernels: budgets.maxConcurrentKernels,
    maxInputTokens: budgets.maxInputTokens,
    maxOutputTokens: budgets.maxOutputTokens,
    maxContextReadBytes: budgets.maxContextReadBytes,
    maxQueryPromptChars: budgets.maxQueryPromptChars,
    maxInlineOutputBytes: budgets.maxInlineOutputBytes,
    maxArtifactBytes: budgets.maxArtifactBytes,
    maxCostUsd: budgets.maxCostUsd,
    maxPromptChars: config.maxPromptChars ?? 100_000_000,
    maxRootHistoryChars: config.maxRootHistoryChars ?? 400_000,
    maxParseRetries: config.maxParseRetries ?? 2,
    maxCodePreviewChars: config.maxCodePreviewChars ?? 4_000,
    maxStdoutPreviewChars: config.maxStdoutPreviewChars ?? 2_000,
    maxErrorPreviewChars: config.maxErrorPreviewChars ?? 1_500,
    maxRenderedChars: config.maxRenderedChars ?? 20_000,
    maxTaskChars: config.maxTaskChars ?? 8_000,
    finalVariableName: config.finalVariableName ?? 'Final',
    contextStoreRoot: config.contextStoreRoot ?? '.dsh/rlm-artifacts',
    writeSessionEvents: config.writeSessionEvents ?? true,
    pricing,
    rootRoute,
    leafRoute,
    ...(childRoute === undefined ? {} : { childRoute }),
    budgets,
    promptProfile,
  }
  return resolved
}

/** Assemble the service configuration from the resolved plugin configuration. */
export function createDefaultServiceConfig(config: ResolvedRlmPluginConfig): RlmServiceConfig {
  return {
    defaultDepth: config.defaultDepth,
    maxDepth: config.maxDepth,
    budgets: config.budgets,
    maxPromptChars: config.maxPromptChars,
    maxRootHistoryChars: config.maxRootHistoryChars,
    maxParseRetries: config.maxParseRetries,
    maxCodePreviewChars: config.maxCodePreviewChars,
    maxStdoutPreviewChars: config.maxStdoutPreviewChars,
    maxQueryPromptChars: config.budgets.maxQueryPromptChars,
    maxErrorPreviewChars: config.maxErrorPreviewChars,
    finalVariableName: config.finalVariableName,
    rootMaxTokens: config.rootMaxTokens,
    leafMaxTokens: config.leafMaxTokens,
    ...(config.temperature === undefined ? {} : { temperature: config.temperature }),
    promptProfile: config.promptProfile,
    pricing: config.pricing,
    requireQualifiedProfile: config.requireQualifiedProfile,
    isolationPolicy: DEFAULT_ISOLATION_POLICY,
  }
}
