/**
 * Leaf and root model calls.
 *
 * `llm_query()` in the paper is a *direct* model request that returns a value to
 * the REPL — not an Agent turn. This module is what makes that literal: it calls
 * the harness LLM seam with a one-shot, tool-free request and assembles the
 * stream itself.
 *
 * The interface exists so the RLM core can be tested against a scripted model
 * with no network, while production runs funnel through
 * {@link DshModelClient}. Hand-built requests are an explicitly supported seam:
 * they carry no session identity, so they cannot leak into a Session log by
 * accident.
 *
 * @module dsh-rlm/llm
 */

import { BlockAssembler, type GenerateOptions, type TokenUsage } from '@deepseek-ai/dsh-llm'

/** A host-assembled, identity-free input message. */
export interface RlmInputMessage {
  readonly role: 'user'
  readonly text: string
}

export interface RlmModelRequest {
  provider: string
  model: string
  /** One-shot system slot; adapters place it ahead of `messages`. */
  system?: string
  messages: readonly RlmInputMessage[]
  maxTokens?: number
  temperature?: number
  signal?: AbortSignal
  /** Auxiliary-call classification; RLM calls are ordinary, so this stays unset. */
  purpose?: 'compaction' | 'session-title'
}

export interface RlmTokenUsage {
  inputTokens: number
  outputTokens: number
  cacheReadTokens: number
  cacheWriteTokens: number
  reasoningTokens: number
}

export interface RlmModelError {
  kind: 'aborted' | 'error' | 'max-tokens' | 'no-adapter'
  message: string
}

export interface RlmModelResponse {
  text: string
  usage: RlmTokenUsage
  finishReason: string
  error?: RlmModelError
}

/** The minimal model capability the RLM core depends on. */
export interface RlmModelClient {
  complete(request: RlmModelRequest): Promise<RlmModelResponse>
}

/** Per-route pricing, used only to populate the cost counter honestly. */
export interface RlmPricing {
  inputUsdPerMTok: number
  outputUsdPerMillionTokens?: number
  outputUsdPerMTok?: number
}

export function emptyUsage(): RlmTokenUsage {
  return {
    inputTokens: 0,
    outputTokens: 0,
    cacheReadTokens: 0,
    cacheWriteTokens: 0,
    reasoningTokens: 0,
  }
}

export function toTokenUsage(usage: TokenUsage | undefined): RlmTokenUsage {
  if (!usage) return emptyUsage()
  return {
    inputTokens: usage.inputTokens ?? 0,
    outputTokens: usage.outputTokens ?? 0,
    cacheReadTokens: usage.cacheReadTokens ?? 0,
    cacheWriteTokens: usage.cacheWriteTokens ?? 0,
    reasoningTokens: usage.reasoningTokens ?? 0,
  }
}

export function addUsage(total: RlmTokenUsage, delta: RlmTokenUsage): RlmTokenUsage {
  return {
    inputTokens: total.inputTokens + delta.inputTokens,
    outputTokens: total.outputTokens + delta.outputTokens,
    cacheReadTokens: total.cacheReadTokens + delta.cacheReadTokens,
    cacheWriteTokens: total.cacheWriteTokens + delta.cacheWriteTokens,
    reasoningTokens: total.reasoningTokens + delta.reasoningTokens,
  }
}

/** Cost in USD. Cache reads/writes are priced as ordinary input when unpriced. */
export function estimateCostUsd(usage: RlmTokenUsage, pricing: RlmPricing | undefined): number {
  if (!pricing) return 0
  const outputRate = pricing.outputUsdPerMTok ?? pricing.outputUsdPerMillionTokens ?? 0
  const billedInput = usage.inputTokens + usage.cacheReadTokens + usage.cacheWriteTokens
  return (billedInput / 1_000_000) * pricing.inputUsdPerMTok + (usage.outputTokens / 1_000_000) * outputRate
}

/** The subset of `LlmRuntime` this client uses; keeps the fake trivially small. */
export interface LlmStreamRuntime {
  stream(options: GenerateOptions): AsyncIterable<unknown>
}

/**
 * Production model client over `ctx.llm`.
 *
 * Notes that matter for correctness:
 * - the adapter honors `signal`, and an abort surfaces as a terminal `finish`
 *   chunk rather than a throw (README §4.5 cancellation path);
 * - adapter failures are normalized into a terminal `finish` chunk, so this
 *   client reports them as errors instead of pretending the text was complete;
 * - a partially streamed answer is still returned as text, because a leaf call
 *   that was cut short must be visible to the program as incomplete.
 */
export class DshModelClient implements RlmModelClient {
  constructor(private readonly llm: LlmStreamRuntime) {}

  async complete(request: RlmModelRequest): Promise<RlmModelResponse> {
    const options: GenerateOptions = {
      provider: request.provider,
      model: request.model,
      messages: request.messages.map((message) => ({
        role: 'user' as const,
        content: [{ type: 'text' as const, text: message.text }],
      })),
      ...(request.system === undefined ? {} : { system: request.system }),
      ...(request.maxTokens === undefined ? {} : { maxTokens: request.maxTokens }),
      ...(request.temperature === undefined ? {} : { temperature: request.temperature }),
      ...(request.signal === undefined ? {} : { signal: request.signal }),
      ...(request.purpose === undefined ? {} : { purpose: request.purpose }),
    }

    const assembler = new BlockAssembler()
    try {
      for await (const chunk of this.llm.stream(options)) {
        assembler.push(chunk as Parameters<BlockAssembler['push']>[0])
      }
    } catch (error) {
      return {
        text: '',
        usage: emptyUsage(),
        finishReason: 'error',
        error: { kind: 'error', message: (error as Error).message },
      }
    }

    const text = assembler
      .blocks()
      .filter((block): block is { type: 'text'; text: string } => block.type === 'text')
      .map((block) => block.text)
      .join('')
    const finish = assembler.finish
    const usage = toTokenUsage(assembler.usage)

    if (finish.kind === 'error') {
      return {
        text,
        usage,
        finishReason: 'error',
        error: { kind: 'error', message: finish.failure?.message ?? 'model call failed' },
      }
    }
    if (finish.kind === 'aborted') {
      return {
        text,
        usage,
        finishReason: 'aborted',
        error: { kind: 'aborted', message: finish.failure?.message ?? 'model call aborted' },
      }
    }
    if (finish.kind === 'max-tokens') {
      return {
        text,
        usage,
        finishReason: 'max-tokens',
        error: { kind: 'max-tokens', message: 'the model hit its output token ceiling' },
      }
    }
    return { text, usage, finishReason: finish.kind }
  }
}

export default DshModelClient
