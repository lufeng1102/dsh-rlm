/**
 * Root-action parsing tests (README §4.6).
 *
 * The parser is the boundary between "a thought" and "the final answer", and the
 * paper documents how brittle textual tagging is. These cases pin the host's
 * behaviour so ambiguity becomes a bounded retry rather than a silent guess.
 */

import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { buildParseRetryInstruction, extractFences, findFinalMatches, parseRootAction, unquote } from '../lib/finalize.js'

describe('fenced code extraction', () => {
  it('extracts a repl fence with the paper language tag', () => {
    const { fences, unterminated } = extractFences('plan\n```repl\nprint(1)\n```\ndone')
    assert.equal(unterminated, false)
    assert.equal(fences.length, 1)
    assert.equal(fences[0].language, 'repl')
    assert.equal(fences[0].code, 'print(1)\n')
  })

  it('reports an unterminated fence instead of swallowing the rest of the answer', () => {
    const { fences, unterminated } = extractFences('```repl\nprint(1)')
    assert.equal(unterminated, true)
    assert.equal(fences.length, 1)
  })
})

describe('FINAL / FINAL_VAR scanning', () => {
  it('balances nested parentheses in a final answer', () => {
    const { matches } = findFinalMatches('FINAL(The total is (a) 12 and (b) 13)')
    assert.equal(matches.length, 1)
    assert.equal(matches[0].value, 'The total is (a) 12 and (b) 13')
  })

  it('ignores a FINAL prefix inside a longer identifier', () => {
    const { matches } = findFinalMatches('NOTFINAL(1)')
    assert.equal(matches.length, 0)
  })

  it('reports a missing closing parenthesis as malformed', () => {
    const { malformed } = findFinalMatches('FINAL(never closed')
    assert.match(malformed ?? '', /closing parenthesis/)
  })

  it('strips one layer of matching quotes', () => {
    assert.equal(unquote('"quoted answer"'), 'quoted answer')
    assert.equal(unquote("'single'"), 'single')
    assert.equal(unquote('unquoted (keep)'), 'unquoted (keep)')
  })
})

describe('structured root action parsing', () => {
  it('accepts a single repl fence as an eval action', () => {
    const parsed = parseRootAction('Let me look.\n```repl\nprint(len(context))\n```')
    assert.equal(parsed.ok, true)
    assert.equal(parsed.action.type, 'eval')
    assert.match(parsed.action.code, /len\(context\)/)
  })

  it('accepts python and bare fences as code, per the configured language set', () => {
    assert.equal(parseRootAction('```python\nx = 1\n```').ok, true)
    assert.equal(parseRootAction('```py\nx = 1\n```').ok, true)
  })

  it('concatenates several code fences in order by default', () => {
    const parsed = parseRootAction('```repl\na = 1\n```\nthen\n```repl\nb = 2\n```')
    assert.equal(parsed.ok, true)
    assert.equal(parsed.action.code, 'a = 1\n\nb = 2\n')
  })

  it('rejects a turn that mixes code and a final answer', () => {
    const parsed = parseRootAction('```repl\nx = 1\n```\nFINAL(42)')
    assert.equal(parsed.ok, false)
    assert.equal(parsed.failure.reason, 'mixed-code-and-final')
  })

  it('rejects a turn that uses both final forms', () => {
    const parsed = parseRootAction('FINAL(42)\nFINAL_VAR(answer)')
    assert.equal(parsed.ok, false)
    assert.equal(parsed.failure.reason, 'mixed-final-forms')
  })

  it('parses FINAL_VAR into a variable action', () => {
    const parsed = parseRootAction('```repl\nfinal_answer = 1\n```'.replace(/```repl[\s\S]*```/, '') + 'FINAL_VAR(final_answer)')
    assert.equal(parsed.ok, true)
    assert.deepEqual(parsed.action, { type: 'final-variable', name: 'final_answer' })
  })

  it('rejects a FINAL_VAR that is not a plain variable name', () => {
    const parsed = parseRootAction('FINAL_VAR(answers[0])')
    assert.equal(parsed.ok, false)
    assert.match(parsed.failure.message, /plain REPL variable name/)
  })

  it('rejects an empty response', () => {
    const parsed = parseRootAction('   ')
    assert.equal(parsed.ok, false)
    assert.equal(parsed.failure.reason, 'empty')
  })

  it('rejects prose with neither code nor a final marker', () => {
    const parsed = parseRootAction('I will now analyse the context.')
    assert.equal(parsed.ok, false)
    assert.match(parsed.failure.message, /neither an executable code fence nor a final marker/)
  })

  it('rejects an unterminated fence even when it looks like code', () => {
    const parsed = parseRootAction('```repl\nprint(1)')
    assert.equal(parsed.ok, false)
    assert.equal(parsed.failure.reason, 'unterminated-code-fence')
  })

  it('explains the protocol in a bounded correction instruction', () => {
    const text = buildParseRetryInstruction({ reason: 'empty', message: 'the response was empty' }, 1, 2)
    assert.match(text, /correction attempt 1 of 2/)
    assert.match(text, /FINAL_VAR/)
  })
})
