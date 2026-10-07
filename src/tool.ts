/**
 * The model-facing `run_rlm` tool.
 *
 * Exactly one tool is registered, and it is the *only* way a model starts an
 * RLM. There is deliberately no generic `run_code` here: `run_code` is a PTC
 * consumer with different semantics (one-shot program, no persistent REPL, no
 * recursive sub-calls), and naming both "RLM" is how telemetry and users end up
 * comparing two different algorithms (README §3.3).
 *
 * The tool is also where host authority is finalized for one call: routes,
 * depth, budgets and the event sink all come from this layer, never from the
 * model's arguments beyond the narrow, validated knobs in the parameter schema.
 *
 * @module dsh-rlm/tool
 */

import { KNOWN_SESSION_EVENT_TYPES } from '@deepseek-ai/dsh-session'
import { defineTool, type ToolRunContext } from '@deepseek-ai/dsh-tools'
import type { Context } from '@deepseek-ai/cordis'
import { CompositeEventSink, NullEventSink, SessionEventSink, type RlmEventSink } from './events.ts'
import type { RlmBudgetLimits, RlmContextInput, RlmResult } from './types.ts'
import type { RlmRoute, RlmService } from './service.ts'
import { RlmAdmissionError } from './service.ts'

/** Structural view of the tool's canonical output value. */
export interface RunRlmOutput {
  status: 'ok' | 'failed' | 'cancelled' | 'budget-exceeded'
  stopReason: string
  answer?: string
  artifact?: { id: string; bytes: number; chars: number; preview: string; truncated: boolean }
  steps: number
  depth: number
  usage: {
    rootIterations: number
    queries: number
    kernels: number
    inputTokens: number
    outputTokens: number
    contextReadBytes: number
    costUsd: number
    wallClockMs: number
    maxObservedConcurrency: number
  }
  error?: { kind: string; message: string }
}

export interface RunRlmToolConfig {
  toolName: string
  description: string
  /** Deployment ceiling for one call, enforced by the tool-call timeout policy. */
  timeoutMs: number
  rootRoute: RlmRoute
  leafRoute: RlmRoute
  childRoute?: RlmRoute
  /** Fail an oversized `prompt` when material should travel as `context`. */
  maxTaskChars: number
  /** Longest result text rendered back into the model's window. */
  maxRenderedChars: number
  writeSessionEvents: boolean
  /** Sink used when a tool call has no session (or session writes are off). */
  baseSink: RlmEventSink
  onSkipSessionEvent?: (type: string, reason: string) => void
}

const OUTPUT_PROPERTIES = {
  status: { type: 'string', enum: ['ok', 'failed', 'cancelled', 'budget-exceeded'] as const, required: true as const },
  stopReason: { type: 'string', required: true as const },
  answer: { type: 'string' },
  artifact: {
    type: 'object',
    additionalProperties: false,
    properties: {
      id: { type: 'string', required: true as const },
      bytes: { type: 'integer', required: true as const },
      chars: { type: 'integer', required: true as const },
      preview: { type: 'string', required: true as const },
      truncated: { type: 'boolean', required: true as const },
    },
  },
  steps: { type: 'integer', required: true as const },
  depth: { type: 'integer', required: true as const },
  usage: {
    type: 'object',
    additionalProperties: false,
    required: true as const,
    properties: {
      rootIterations: { type: 'integer', required: true as const },
      queries: { type: 'integer', required: true as const },
      kernels: { type: 'integer', required: true as const },
      inputTokens: { type: 'integer', required: true as const },
      outputTokens: { type: 'integer', required: true as const },
      contextReadBytes: { type: 'integer', required: true as const },
      costUsd: { type: 'number', required: true as const },
      wallClockMs: { type: 'integer', required: true as const },
      maxObservedConcurrency: { type: 'integer', required: true as const },
    },
  },
  error: {
    type: 'object',
    additionalProperties: false,
    properties: {
      kind: { type: 'string', required: true as const },
      message: { type: 'string', required: true as const },
    },
  },
} as const

/** Register `run_rlm`. Returns the disposer `ctx.tools.register` produced. */
export function registerRunRlmTool(
  ctx: Context,
  service: RlmService,
  config: RunRlmToolConfig,
): () => void {
  return ctx.tools.register(
    defineTool({
      name: config.toolName,
      description: config.description,
      timeoutMs: config.timeoutMs,
      isConcurrencySafe: () => true,
      parameters: {
        prompt: {
          type: 'string',
          required: true,
          description:
            'The complete user prompt or question. It is stored OUTSIDE the model context and made available to a persistent REPL as `context`; it is never copied into the root conversation. Put long material here (or in `context`) instead of in a normal message.',
        },
        context: {
          type: 'json',
          description:
            'Optional long material to externalize alongside `prompt`: a string, an array of strings (documents), or a record. When present, `prompt` is treated as the task and the material becomes the REPL `context`.',
        },
        depth: {
          type: 'integer',
          description:
            'Maximum recursion depth for this run. 0 = no sub-calls, 1 = llm_query only (paper default), 2+ = recursive rlm_query. Bounded by the deployment ceiling.',
        },
        model: {
          type: 'string',
          description: 'Optional model override for this run: "provider/model" or a bare model id using the configured provider.',
        },
        budget: {
          type: 'json',
          description:
            'Optional budget overrides (for example {"maxTotalQueries":64,"maxWallClockMs":120000}). Values are clamped to the deployment ceiling, never raised above it.',
        },
      },
      output: {
        schema: {
          type: 'object',
          additionalProperties: false,
          properties: OUTPUT_PROPERTIES,
        },
        render: (_args, value) => [{ type: 'text', text: renderRunRlmResult(value as unknown as RunRlmOutput, config.maxRenderedChars) }],
      },
      async execute(args, exec) {
        const request = args as {
          prompt: string
          context?: unknown
          depth?: number
          model?: string
          budget?: unknown
        }
        const prompt = request.prompt ?? ''
        if (prompt.trim().length === 0 && request.context === undefined) {
          throw new RlmAdmissionError('empty-prompt', 'run_rlm requires a non-empty prompt or context')
        }
        const material = toContextInput(request.context, prompt, config.maxTaskChars)
        const routeOverride = parseModelOverride(request.model, config)
        const controller = new AbortController()
        const onAbort = (): void => controller.abort('parent step aborted')
        exec.signal.addEventListener('abort', onAbort, { once: true })
        if (exec.signal.aborted) onAbort()

        const sink = buildSink(config, exec)
        const sessionHeader = readSessionHeader(exec)
        try {
          const result = await service.run({
            prompt: material.input,
            ...(material.task === undefined ? {} : { task: material.task }),
            ...(request.depth === undefined ? {} : { depth: request.depth }),
            rootRoute: routeOverride.root,
            leafRoute: routeOverride.leaf,
            ...(routeOverride.child === undefined ? {} : { childRoute: routeOverride.child }),
            ...(sessionHeader.id === undefined ? {} : { sessionId: sessionHeader.id }),
            ...(request.budget === undefined ? {} : { budgets: sanitizeBudgets(request.budget) }),
            cwd: sessionHeader.cwd ?? process.cwd(),
            signal: controller.signal,
            toolCallId: String(exec.callId),
            events: sink,
          })
          return toOutput(result)
        } finally {
          exec.signal.removeEventListener('abort', onAbort)
          await sink.flush().catch(() => {})
        }
      },
    }),
  )
}

/** Map an `RlmResult` onto the canonical tool output. */
export function toOutput(result: RlmResult): RunRlmOutput {
  return {
    status: result.status,
    stopReason: result.stopReason,
    ...(result.text === undefined ? {} : { answer: result.text }),
    ...(result.artifactRef === undefined
      ? {}
      : {
          artifact: {
            id: result.artifactRef.id,
            bytes: result.artifactRef.bytes,
            chars: result.artifactRef.chars,
            preview: result.artifactRef.preview,
            truncated: result.artifactRef.truncated,
          },
        }),
    steps: result.steps,
    depth: result.depth,
    usage: {
      rootIterations: result.usage.rootIterations,
      queries: result.usage.queries,
      kernels: result.usage.kernels,
      inputTokens: result.usage.inputTokens,
      outputTokens: result.usage.outputTokens,
      contextReadBytes: result.usage.contextReadBytes,
      costUsd: result.usage.costUsd,
      wallClockMs: result.usage.wallClockMs,
      maxObservedConcurrency: result.usage.maxObservedConcurrency,
    },
    ...(result.error === undefined ? {} : { error: { kind: result.error.kind, message: result.error.message } }),
  }
}

/**
 * Render the tool result for the model.
 *
 * A large final answer is *not* rendered: the model gets the artifact reference
 * and a preview, because pushing megabytes back into the window is precisely the
 * failure this plugin exists to avoid.
 */
export function renderRunRlmResult(value: RunRlmOutput, maxChars: number): string {
  const lines: string[] = []
  if (value.status === 'ok') {
    if (value.answer !== undefined) {
      lines.push(value.answer.length > maxChars ? `${value.answer.slice(0, maxChars)}\n... [truncated for display]` : value.answer)
    }
    if (value.artifact) {
      lines.push(
        `[final answer stored as artifact ${value.artifact.id}: ${value.artifact.chars} chars, ${value.artifact.bytes} bytes` +
          `${value.artifact.truncated ? ', preview truncated' : ''}]`,
      )
      if (value.artifact.preview.length > 0 && value.answer === undefined) {
        lines.push('')
        lines.push(value.artifact.preview)
      }
    }
  } else {
    lines.push(`run_rlm ${value.status} (${value.stopReason})`)
    if (value.error) lines.push(`${value.error.kind}: ${value.error.message}`)
    if (value.answer !== undefined && value.answer.length > 0) lines.push(value.answer)
  }
  lines.push(
    `[depth ${value.depth}, ${value.steps} root steps, ${value.usage.queries} sub-calls, ` +
      `${value.usage.inputTokens + value.usage.outputTokens} tokens, ${Math.round(value.usage.wallClockMs / 1000)}s]`,
  )
  return lines.join('\n')
}

interface MaterializedInput {
  input: RlmContextInput
  task?: string
}

/**
 * Decide what the externalized prompt is.
 *
 * With no `context`, the paper's shape is used verbatim: `prompt` *is* P, so the
 * question lives inside the artifact and the root model reads it from the REPL.
 * With a `context`, the material becomes P and the short `prompt` becomes the
 * task, which is the only way to keep a document corpus out of root history
 * while still telling the model what is being asked.
 */
export function toContextInput(context: unknown, prompt: string, maxTaskChars: number): MaterializedInput {
  if (context === undefined || context === null) {
    return { input: { kind: 'text', text: prompt } }
  }
  if (prompt.length > maxTaskChars) {
    throw new RlmAdmissionError(
      'task-too-large',
      `prompt of ${prompt.length} characters exceeds the ${maxTaskChars} character task ceiling; ` +
        'pass long material through `context` instead (it is externalized, not injected into the conversation)',
    )
  }
  const input = normalizeContext(context)
  return { input, task: prompt }
}

function normalizeContext(context: unknown): RlmContextInput {
  if (typeof context === 'string') return { kind: 'text', text: context }
  if (Array.isArray(context)) {
    const documents = context.map((item) => (typeof item === 'string' ? item : JSON.stringify(item)))
    return { kind: 'documents', documents }
  }
  if (typeof context === 'object') {
    const record: Record<string, unknown> = {}
    for (const [key, value] of Object.entries(context as Record<string, unknown>)) {
      record[key] = value
    }
    return { kind: 'record', record }
  }
  return { kind: 'text', text: String(context) }
}

function parseModelOverride(
  model: string | undefined,
  config: RunRlmToolConfig,
): { root: RlmRoute; leaf: RlmRoute; child?: RlmRoute } {
  if (model === undefined || model.length === 0) {
    return {
      root: config.rootRoute,
      leaf: config.leafRoute,
      ...(config.childRoute === undefined ? {} : { child: config.childRoute }),
    }
  }
  const [head, tail] = model.includes('/') ? (model.split('/', 2) as [string, string]) : [undefined, model]
  const provider = head ?? config.rootRoute.provider
  const route: RlmRoute = { provider, model: tail }
  return {
    root: route,
    leaf: { provider, model: tail },
    ...(config.childRoute === undefined ? {} : { child: route }),
  }
}

/** Keep only known budget dimensions with finite, non-negative values. */
export function sanitizeBudgets(budget: unknown): Partial<RlmBudgetLimits> {
  if (budget === null || typeof budget !== 'object') return {}
  const allowed = new Set<keyof RlmBudgetLimits>([
    'maxRootIterations',
    'maxWallClockMs',
    'maxTotalQueries',
    'maxConcurrentQueries',
    'maxTotalKernels',
    'maxConcurrentKernels',
    'maxInputTokens',
    'maxOutputTokens',
    'maxContextReadBytes',
    'maxQueryPromptChars',
    'maxInlineOutputBytes',
    'maxArtifactBytes',
    'maxCostUsd',
  ])
  const result: Partial<RlmBudgetLimits> = {}
  for (const [key, value] of Object.entries(budget as Record<string, unknown>)) {
    if (!allowed.has(key as keyof RlmBudgetLimits)) continue
    if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) continue
    result[key as keyof RlmBudgetLimits] = Math.floor(value)
  }
  return result
}

function buildSink(config: RunRlmToolConfig, exec: ToolRunContext): RlmEventSink {
  if (!config.writeSessionEvents) return config.baseSink
  const session = (exec.agent as { session?: { append(type: string, data: unknown): unknown } } | undefined)?.session
  if (!session || typeof session.append !== 'function') return config.baseSink
  // Two destinations, each written exactly once: the sidecar always (durability
  // and traces) and the session when — and only when — the build declares the
  // type. The session sink's fallback is therefore a null sink, not `baseSink`,
  // which the composite already fans out to.
  return new CompositeEventSink([
    config.baseSink,
    new SessionEventSink({
      session,
      knownTypes: KNOWN_SESSION_EVENT_TYPES,
      fallback: new NullEventSink(),
      ...(config.onSkipSessionEvent === undefined ? {} : { onSkip: config.onSkipSessionEvent }),
    }),
  ])
}

function readSessionHeader(exec: ToolRunContext): { id?: string; cwd?: string } {
  const agent = exec.agent as
    | { id?: string; session?: { header?: { cwd?: string; id?: string } } }
    | undefined
  const id = agent?.id ?? agent?.session?.header?.id
  return {
    ...(typeof id === 'string' ? { id } : {}),
    ...(typeof agent?.session?.header?.cwd === 'string' ? { cwd: agent.session.header.cwd } : {}),
  }
}
