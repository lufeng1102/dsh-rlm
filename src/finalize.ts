/**
 * Structured root actions and the `FINAL` / `FINAL_VAR` compatibility parser
 * (README §4.6).
 *
 * The paper's own results show that distinguishing "a thought" from "the final
 * answer" with textual tags is brittle (16% of turns misused `FINAL`, 13%
 * misused `FINAL_VAR`). The canonical host protocol is therefore a closed union:
 * every root turn must resolve to exactly one action. Tags are still parsed,
 * because the released RLM prompts emit them and trajectories must stay
 * comparable — but they are parsed *here*, in the host, and are never Python
 * functions inside the REPL.
 *
 * A parse failure is a first-class outcome, not an exception: the root loop
 * charges a bounded retry to the same budget and tells the model exactly what
 * was ambiguous.
 *
 * @module dsh-rlm/finalize
 */

import type { RlmParseFailure, RlmRootAction } from './types.ts'

export interface RlmRootActionParserOptions {
  /**
   * Fence language tags that count as executable code, lowercased. The paper's
   * prompts use `repl`; `python`/`py` are accepted because models drift.
   */
  codeFenceLanguages?: readonly string[]
  /** What to do with several code fences in one turn. */
  multipleCodeBlocks?: 'concat' | 'last' | 'reject'
}

export type RlmRootParseResult =
  | { ok: true; action: RlmRootAction }
  | { ok: false; failure: RlmParseFailure }

const DEFAULT_CODE_LANGUAGES = ['repl', 'python', 'py'] as const

interface Fence {
  language: string
  code: string
  start: number
  end: number
}

/** Extract fenced code blocks, tracking offsets so prose can be inspected separately. */
export function extractFences(source: string): { fences: Fence[]; unterminated: boolean } {
  const fences: Fence[] = []
  const pattern = /^[ \t]*(`{3,}|~{3,})[ \t]*([A-Za-z0-9_+-]*)[ \t]*(?:\r?\n|$)/gm
  let open: { marker: string; language: string; start: number; contentStart: number } | null = null
  let match: RegExpExecArray | null
  let unterminated = false
  while ((match = pattern.exec(source)) !== null) {
    const marker = match[1]!
    const language = (match[2] ?? '').toLowerCase()
    if (!open) {
      open = { marker: marker[0]!, language, start: match.index, contentStart: match.index + match[0].length }
      continue
    }
    if (marker[0] !== open.marker) continue
    fences.push({
      language: open.language,
      code: source.slice(open.contentStart, match.index),
      start: open.start,
      end: match.index + match[0].length,
    })
    open = null
  }
  if (open) {
    // An unterminated fence still carries code; report it so the loop can ask
    // the model to close it rather than splicing the rest of the answer.
    unterminated = true
    fences.push({
      language: open.language,
      code: source.slice(open.contentStart),
      start: open.start,
      end: source.length,
    })
  }
  return { fences, unterminated }
}

/** Replace fenced regions with spaces so offsets in the prose scan stay valid. */
function maskFences(source: string, fences: readonly Fence[]): string {
  const characters = [...source]
  for (const fence of fences) {
    for (let index = fence.start; index < fence.end && index < characters.length; index += 1) {
      if (characters[index] !== '\n') characters[index] = ' '
    }
  }
  return characters.join('')
}

interface FinalMatch {
  form: 'final-text' | 'final-variable'
  raw: string
  value: string
  start: number
  end: number
}

/**
 * Find `FINAL_VAR(...)` and `FINAL(...)` calls in prose.
 *
 * Parenthesis balancing and quote awareness are required: a real answer such as
 * `FINAL(The total is (a) 12 and (b) 13)` must not be truncated at the first `)`.
 */
export function findFinalMatches(prose: string): { matches: FinalMatch[]; malformed: string | null } {
  const matches: FinalMatch[] = []
  let malformed: string | null = null
  const patterns: { form: FinalMatch['form']; regex: RegExp }[] = [
    { form: 'final-variable', regex: /(?<![A-Za-z0-9_])FINAL_VAR\s*\(/g },
    { form: 'final-text', regex: /(?<![A-Za-z0-9_])FINAL\s*\(/g },
  ]
  const spans: [number, number][] = []
  for (const { form, regex } of patterns) {
    let match: RegExpExecArray | null
    while ((match = regex.exec(prose)) !== null) {
      const openIndex = match.index + match[0].length - 1
      const balanced = readBalanced(prose, openIndex)
      if (!balanced) {
        malformed = malformed ?? `${form === 'final-text' ? 'FINAL' : 'FINAL_VAR'} is missing its closing parenthesis`
        continue
      }
      spans.push([match.index, balanced.end + 1])
      matches.push({
        form,
        raw: prose.slice(match.index, balanced.end + 1),
        value: balanced.inner,
        start: match.index,
        end: balanced.end + 1,
      })
    }
  }
  matches.sort((a, b) => a.start - b.start)
  return { matches, malformed }
}

/** Read a balanced parenthesized region starting at `openIndex`. */
function readBalanced(source: string, openIndex: number): { inner: string; end: number } | null {
  let depth = 0
  let quote: string | null = null
  for (let index = openIndex; index < source.length; index += 1) {
    const character = source[index]!
    if (quote) {
      if (character === '\\') {
        index += 1
        continue
      }
      if (character === quote) quote = null
      continue
    }
    if (character === '"' || character === "'" || character === '`') {
      quote = character
      continue
    }
    if (character === '(') depth += 1
    else if (character === ')') {
      depth -= 1
      if (depth === 0) return { inner: source.slice(openIndex + 1, index), end: index }
    }
  }
  return null
}

/** Strip one layer of matching quotes from a `FINAL(...)` argument. */
export function unquote(value: string): string {
  const trimmed = value.trim()
  if (trimmed.length >= 2) {
    const first = trimmed[0]!
    const last = trimmed[trimmed.length - 1]!
    if ((first === '"' || first === "'" || first === '`') && first === last) {
      return trimmed.slice(1, -1)
    }
  }
  return trimmed
}

const IDENTIFIER = /^[A-Za-z_][A-Za-z0-9_]*$/

/**
 * Turn one root-model response into exactly one action.
 *
 * Precedence is deliberate: a code fence means "execute this", and prose
 * containing `FINAL` next to a code fence is a protocol error rather than a
 * silent guess, because guessing is what makes tag-based finalization unusable.
 */
export function parseRootAction(
  source: string,
  options: RlmRootActionParserOptions = {},
): RlmRootParseResult {
  const languages = new Set(
    (options.codeFenceLanguages ?? DEFAULT_CODE_LANGUAGES).map((language) => language.toLowerCase()),
  )
  const multiple = options.multipleCodeBlocks ?? 'concat'
  if (!source || source.trim().length === 0) {
    return { ok: false, failure: { reason: 'empty', message: 'the response was empty' } }
  }

  const { fences, unterminated } = extractFences(source)
  const codeFences = fences.filter((fence) => languages.has(fence.language))
  const prose = maskFences(source, fences)
  const { matches, malformed } = findFinalMatches(prose)

  if (malformed) {
    return { ok: false, failure: { reason: 'unknown', message: malformed } }
  }
  if (matches.some((match) => match.form === 'final-text') && matches.some((match) => match.form === 'final-variable')) {
    return {
      ok: false,
      failure: {
        reason: 'mixed-final-forms',
        message: 'the response used both FINAL(...) and FINAL_VAR(...); pick exactly one',
      },
    }
  }
  if (codeFences.length > 0 && matches.length > 0) {
    return {
      ok: false,
      failure: {
        reason: 'mixed-code-and-final',
        message: 'the response contained both executable code and a final answer; do one per turn',
      },
    }
  }
  if (unterminated) {
    return {
      ok: false,
      failure: { reason: 'unterminated-code-fence', message: 'a code fence was never closed' },
    }
  }

  if (matches.length > 1) {
    return {
      ok: false,
      failure: {
        reason: 'mixed-final-forms',
        message: `the response contained ${matches.length} final markers; provide exactly one`,
      },
    }
  }
  if (matches.length === 1) {
    const match = matches[0]!
    if (match.form === 'final-variable') {
      const name = unquote(match.value)
      if (!IDENTIFIER.test(name)) {
        return {
          ok: false,
          failure: {
            reason: 'unknown',
            message: `FINAL_VAR expects a plain REPL variable name, received ${JSON.stringify(match.value)}`,
          },
        }
      }
      return { ok: true, action: { type: 'final-variable', name } }
    }
    return { ok: true, action: { type: 'final-text', text: unquote(match.value) } }
  }

  if (codeFences.length > 0) {
    if (codeFences.length > 1 && multiple === 'reject') {
      return {
        ok: false,
        failure: {
          reason: 'unknown',
          message: `the response contained ${codeFences.length} code blocks; send one`,
        },
      }
    }
    const code =
      multiple === 'last'
        ? codeFences[codeFences.length - 1]!.code
        : codeFences.map((fence) => fence.code).join('\n')
    return { ok: true, action: { type: 'eval', code } }
  }

  return {
    ok: false,
    failure: {
      reason: 'unknown',
      message:
        'the response had neither an executable code fence nor a final marker; answer with code or FINAL(...)/FINAL_VAR(...)',
    },
  }
}

/** The bounded correction instruction appended after a parse failure. */
export function buildParseRetryInstruction(failure: RlmParseFailure, attempt: number, maxAttempts: number): string {
  return [
    `Your previous response could not be executed: ${failure.message}.`,
    `This is correction attempt ${attempt} of ${maxAttempts}.`,
    'Reply with exactly one of:',
    '1. a single ```repl code block to run in the persistent REPL, or',
    '2. the literal text FINAL(<your answer>) or FINAL_VAR(<variable name in the REPL>).',
    'Do not combine code and a final answer in one turn.',
  ].join('\n')
}
