/**
 * Root-model prompt profile and the bounded feedback format.
 *
 * The paper's result that matters here is negative: one RLM prompt does not
 * transfer reliably across models (§4, Appendix C.4), and the exact prompt
 * version is part of an experiment's identity. So the prompt is built from a
 * named {@link RlmModelProfile}, never inline, and every invocation records the
 * profile id in its durable events.
 *
 * The other half of this module is the *bounded* feedback contract: stdout,
 * code and error text are truncated before they can enter root history, because
 * "only constant-size metadata is appended" is what makes the root loop
 * terminate on a prompt much larger than the model window (README §4.4).
 *
 * @module dsh-rlm/prompt
 */

import type { RlmContextRef, RlmStepError, RlmStepResult } from './types.ts'

/** Prompt variant ids; a profile change is a new id, never an edit in place. */
export type RlmPromptVariant = 'paper-gpt5' | 'paper-qwen-coder' | 'paper-qwen3-8b' | 'compact'

export interface RlmModelProfile {
  /** Stable identifier recorded in events and compared across evaluations. */
  id: string
  /** Route used for the root loop. */
  root: { provider: string; model: string }
  /** Route used for leaf `llm_query()` calls unless a program overrides it. */
  leaf: { provider: string; model: string }
  /** Route used for `rlm_query()` children. */
  child?: { provider: string; model: string }
  contextWindowTokens: number
  maxOutputTokens: number
  /** Characters a leaf sub-call can reasonably absorb, per the paper's guidance. */
  subCallChars: number
  variant: RlmPromptVariant
  /** Set once this profile has passed the RLM qualification suite (README §4.8). */
  qualified?: boolean
}

/**
 * Paper-reported profiles. `qualified` reflects *this* plugin's evaluation
 * suite, not the paper's numbers, so they start unqualified and are marked only
 * after a local run.
 */
export const PAPER_PROFILES: Record<string, RlmModelProfile> = {
  'paper-gpt5': {
    id: 'paper-gpt5',
    root: { provider: 'unset', model: 'unset' },
    leaf: { provider: 'unset', model: 'unset' },
    contextWindowTokens: 272_000,
    maxOutputTokens: 32_000,
    subCallChars: 500_000,
    variant: 'paper-gpt5',
  },
  'paper-qwen-coder': {
    id: 'paper-qwen-coder',
    root: { provider: 'unset', model: 'unset' },
    leaf: { provider: 'unset', model: 'unset' },
    contextWindowTokens: 262_000,
    maxOutputTokens: 32_000,
    subCallChars: 200_000,
    variant: 'paper-qwen-coder',
  },
  'paper-qwen3-8b': {
    id: 'paper-qwen3-8b',
    root: { provider: 'unset', model: 'unset' },
    leaf: { provider: 'unset', model: 'unset' },
    contextWindowTokens: 32_000,
    maxOutputTokens: 8_000,
    subCallChars: 10_000,
    variant: 'paper-qwen3-8b',
  },
}

export class RlmProfileNotQualifiedError extends Error {
  override readonly name = 'RlmProfileNotQualifiedError'

  constructor(profileId: string) {
    super(
      `model profile ${profileId} has not passed the RLM qualification suite; ` +
        'refusing it as a root model. Mark it qualified only after a local evaluation run.',
    )
  }
}

export interface RlmPromptOptions {
  contextRef: RlmContextRef
  /**
   * The specific question for this kernel. Present for a child RLM (its query is
   * host-supplied) and absent at the root, where the prompt itself is the task.
   */
  task?: string
  /** Remaining depth inside this kernel. */
  depth: number
  maxDepth: number
  profile: RlmModelProfile
  /** Characters included in the metadata preview of the context. */
  previewChars: number
  maxRootIterations: number
  maxTotalQueries: number
  /** Root-host limit on one eval; also the practical "cost" signal for the model. */
  evalTimeoutMs: number
  finalVarName: string
}

/**
 * Build the RLM system prompt.
 *
 * Prompts are attributed to the paper (arXiv 2512.24601, Appendix C) because
 * benchmark comparability depends on staying close to the released text; the
 * deltas are the host limits this deployment actually enforces and the
 * `FINAL`/`FINAL_VAR` protocol note.
 */
export function buildRlmSystemPrompt(options: RlmPromptOptions): string {
  const { contextRef, profile, depth, maxDepth } = options
  const lines: string[] = []
  lines.push(
    'You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs, which you are strongly encouraged to use as much as possible. You will be queried iteratively until you provide a final answer.',
  )
  if (options.task !== undefined && options.task.length > 0) {
    lines.push('')
    lines.push(`The query you must answer is: ${options.task}`)
  }
  lines.push('')
  lines.push(
    `Your context is a ${contextRef.kind} with ${contextRef.chars} total characters, and is broken up into chunks of char lengths: ${describeChunkLengths(contextRef)}.`,
  )
  lines.push('')
  lines.push('The REPL environment is initialized with:')
  lines.push(
    `1. A \`context\` variable that contains extremely important information about your query. It is a lazy handle, not a copy: use \`len(context)\`, \`context[start:end]\`, \`context.peek(start, end)\`, \`context.items(start, end)\`, \`context.get(key)\` and \`context.search(query)\` to read it. Make sure you look through it sufficiently as you answer your query.`,
  )
  if (depth >= 1) {
    lines.push(
      `2. A \`llm_query(prompt, model=None, max_tokens=None)\` function that queries a leaf LLM (it can handle around ${profile.subCallChars} characters) inside your REPL environment. Use it for straightforward sub-tasks like summarization, extraction, or answering a question about a chunk.`,
    )
    lines.push(
      `   \`llm_query_many(prompts)\` issues one bounded, concurrent batch. Every call is real spend: batch related material and prefer few large calls over many small ones.`,
    )
  } else {
    lines.push(
      '2. No model bindings are installed in this run (maximum recursion depth is 0), so `llm_query` and `rlm_query` are not defined. Do all of the work with `context` reads and code.',
    )
  }
  if (depth >= 2) {
    lines.push(
      `3. An \`rlm_query(context, query)\` function for complex sub-tasks that benefit from iterative, multi-step reasoning. It spawns a full child RLM loop with its own REPL, sub-LLM calls and code execution. Note: if the maximum recursion depth (${maxDepth}) is reached, \`rlm_query\` automatically falls back to \`llm_query\`.`,
    )
    lines.push('')
    lines.push('Choosing between `llm_query` and `rlm_query`:')
    lines.push(
      '- `llm_query(prompt)` for simple sub-tasks: summarize a chunk, extract a fact, answer a direct question. One call, fast and cheap.',
    )
    lines.push(
      '- `rlm_query(context, query)` when the sub-task itself needs chunking, aggregation or multi-step reasoning. Slower and more expensive, but more powerful.',
    )
  }
  lines.push('')
  lines.push(
    'The ability to use `print()` statements to view the output of your REPL code and continue your reasoning. Output is truncated, so never rely on printing a long value to inspect it.',
  )
  lines.push('')
  lines.push(
    'You will only be able to see truncated outputs from the REPL environment, so you should use the query LLM function on variables you want to analyze. You will find this function especially useful when you have to analyze the semantics of the context. Use variables as buffers to build up your final answer.',
  )
  lines.push(
    'Make sure to explicitly look through the entire context in REPL before answering your query. An example strategy is to first look at the context and figure out a chunking strategy, then break up the context into smart chunks, query an LLM per chunk with a particular question and save the answers to a buffer, then query an LLM with all the buffers to produce your final answer.',
  )
  if (depth >= 1) {
    lines.push('')
    lines.push(
      `Remember that your sub-LLMs are powerful -- they can fit around ${profile.subCallChars} characters in their context window, so don't be afraid to put a lot of context into them.`,
    )
  }
  lines.push('')
  lines.push(
    'When you want to execute Python code in the REPL environment, wrap it in a single fenced block with the `repl` language identifier. For example:',
  )
  lines.push('```repl')
  lines.push(
    depth >= 1
      ? 'chunk = context[:100000]\nanswer = llm_query(f"What is the magic number in the context? Here is the chunk: {chunk}")\nprint(answer)'
      : 'chunk = context[:100000]\nprint(len(chunk))',
  )
  lines.push('```')
  lines.push('')
  lines.push('IMPORTANT: When you are done with the iterative process, you MUST provide a final answer, NOT in code. Do not use the final forms unless you have completed your task. You have two options:')
  lines.push('1. Reply with FINAL(your final answer here) to provide the answer directly.')
  lines.push(
    `2. Reply with FINAL_VAR(variable_name) to return a variable you created in the REPL (for example \`${options.finalVarName}\`) as your final output.`,
  )
  lines.push('')
  lines.push(
    `Host limits for this run: at most ${options.maxRootIterations} root iterations, at most ${options.maxTotalQueries} model sub-calls in total, and at most ${Math.round(options.evalTimeoutMs / 1000)} seconds per code execution. Exceeding them ends the run.`,
  )
  lines.push(
    'Think step by step carefully, plan, and execute this plan immediately in your response -- do not just say "I will do this" or "I will do that". Output to the REPL environment and recursive LLMs as much as possible. Remember to explicitly answer the original query in your final answer.',
  )
  return lines.join('\n')
}

function describeChunkLengths(contextRef: RlmContextRef): string {
  const lengths = contextRef.chunkChars
  if (!lengths || lengths.length === 0) return `a single value of ${contextRef.chars} characters`
  const head = lengths.slice(0, 12).join(', ')
  return lengths.length > 12 ? `${head}, ... (${lengths.length} chunks)` : head
}

/** The first root message: bounded metadata only, never the prompt itself. */
export function buildMetadataMessage(options: RlmPromptOptions): string {
  const { contextRef, profile } = options
  return [
    '# RLM invocation',
    '',
    ...(options.task === undefined ? [] : [`Query: ${options.task}`, '']),
    `Invocation prompt handle: \`context\` (kind=${contextRef.kind}, chars=${contextRef.chars}, addressable units=${contextRef.kind === 'text' ? contextRef.chars : (contextRef.chunkChars?.length ?? 'unknown')}).`,
    `Content hash: ${contextRef.sha256}`,
    `Estimated tokens (heuristic): ${contextRef.estimatedTokens['heuristic:ceil(chars/4)'] ?? 'unknown'}.`,
    `Root model profile: ${profile.id} (context window ${profile.contextWindowTokens} tokens).`,
    'The full prompt is NOT in this conversation. It lives in the REPL as `context`.',
    '',
    'Start by inspecting the context and planning a chunking strategy, then execute that plan in a ```repl block.',
  ].join('\n')
}

/** Bounded feedback appended after one evaluation. */
export interface RlmFeedbackOptions {
  step: number
  code: string
  result: RlmStepResult
  maxCodeChars: number
  maxStdoutChars: number
  maxErrorChars: number
}

export function buildStepFeedback(options: RlmFeedbackOptions): string {
  const { step, code, result } = options
  const parts: string[] = []
  parts.push(`# Turn ${step} feedback`)
  parts.push('')
  parts.push('Your program (possibly truncated):')
  parts.push('```python')
  parts.push(truncate(code, options.maxCodeChars))
  parts.push('```')
  parts.push('')
  if (result.error) {
    parts.push(`Execution error (${result.error.kind}): ${truncate(result.error.message, options.maxErrorChars)}`)
    if (result.error.traceback) {
      parts.push('```')
      parts.push(truncate(result.error.traceback, options.maxErrorChars))
      parts.push('```')
    }
    parts.push('Fix the program and try again, or finish with FINAL(...)/FINAL_VAR(...).')
  } else {
    const stdout = result.stdout.length === 0 ? '(no output)' : truncate(result.stdout, options.maxStdoutChars)
    parts.push(`stdout${result.stdoutTruncated ? ' (truncated by the host)' : ''}:`)
    parts.push('```')
    parts.push(stdout)
    parts.push('```')
  }
  if (result.final) {
    parts.push('')
    parts.push(
      `Note: the REPL variable \`${result.final.name}\` is set (${result.final.typeName}, ${result.final.chars ?? 'unknown'} chars). ` +
        `Reply with FINAL_VAR(${result.final.name}) to return it, or keep working with code.`,
    )
  }
  return parts.join('\n')
}

export function truncate(text: string, maxChars: number): string {
  if (text.length <= maxChars) return text
  const omitted = text.length - maxChars
  return `${text.slice(0, maxChars)}\n... [${omitted} characters omitted by the host]`
}

/** Short, model-visible account of a fatal step error. */
export function describeStepError(error: RlmStepError | undefined): string {
  if (!error) return 'none'
  return `${error.kind}: ${error.message}`
}
