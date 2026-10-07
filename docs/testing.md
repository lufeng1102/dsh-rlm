# Testing

```bash
node scripts/link-dev-deps.mjs        # once: point node_modules at the local dsh install
node scripts/build.mjs                # tsc → lib/, and copy the Python driver
node --test                           # 149 Node tests
python3 -m unittest discover -s src/kernel/python -p 'test_*.py'   # 30 driver tests
node scripts/demo.mjs --chars 400000  # narrated end-to-end run (exits non-zero on regression)
npm run verify                        # build + both suites
```

The suite runs against **`lib/`**, not `src/`: the artifact under test is what a
deployment loads. A build that forgets to copy `rlm_kernel.py` fails a test
rather than production (`kernel-python.test.mjs › ships the driver next to the
compiled provider`).

## What each suite is for

| Suite | Cases | Proves |
|---|---:|---|
| `test/budget.test.mjs` | 11 | reservations are returned on settle, usage propagates exactly once, a child can never widen its parent's remaining capacity, exhaustion maps to the right stop reason |
| `test/scheduler.test.mjs` | 8 | the concurrency ceiling holds, queued calls are FIFO, a queue wait times out instead of hanging, an aborted call is refused without being admitted |
| `test/finalize.test.mjs` | 17 | fence extraction (including end-of-input and unterminated fences), balanced `FINAL(...)` scanning, mixed code+final rejection, `FINAL_VAR` identifier validation |
| `test/context-local.test.mjs` | 17 | content-addressed dedup, Unicode-safe character ranges across index samples, surrogate-pair-safe indexing, item paging, per-session authorization, range/artifact ceilings, retention sweep, fail-closed encryption |
| `test/protocol.test.mjs` | 13 | frames survive arbitrary chunk boundaries and newlines, an oversized declared frame is rejected without allocation, a truncated frame is an error |
| `test/root-loop.test.mjs` | 14 | the §7.4 contract: prompt externalization, 5 MB admission, looped sub-calls, symbolic results, depth 0/1/2 semantics, cancellation, soft budget stops, bounded retries, artifact finals, kernel death |
| `test/kernel-python.test.mjs` | 22 | the *real* Python driver through the *real* provider: namespace persistence, context reads, `llm_query_many` concurrency, `await` support, `rlm_query` routing, error classification, large-file export, guest guards, process-group kill, per-eval deadline, no leaks, honest capabilities |
| `test/plugin-e2e.test.mjs` | 10 | the plugin *as a dsh plugin*: real cordis, both capability plugins, the `run_rlm` tool, a real Python kernel, load/unload symmetry, durable traces with exactly one `rlm/start` per invocation |
| `test/config-tool.test.mjs` | 24 | every limit resolves to a finite number, depth clamping, route derivation, prompt-profile validation, budget-override sanitization, artifact rendering, output mapping |
| `test/events.test.mjs` | 12 | the session-vocabulary guard (undeclared types never touch the session log), sidecar ordering, sink isolation, ring capping |
| `src/kernel/python/test_rlm_kernel.py` | 30 | the driver in isolation: framing, host channel cancel/error paths, context handle protocols, persistent namespace, guards, export, output truncation |

## Deliberate test design choices

**A real interpreter, not a mock, for anything process-shaped.** The kernel
suites spawn `python3` for every case in `kernel-python.test.mjs` and the whole
plugin e2e. Mocks could not catch the two bugs these tests actually found: a
`subprocess` guard that was decorative because the module had not been imported
yet, and a provider that hung forever when `spawn` failed (Node emits `error`,
not `exit`, for a failed spawn).

**Both a fake and a real kernel in the loop tests.** `test/root-loop.test.mjs`
drives the state machine through `InMemoryKernelRuntime` so root-loop properties
are tested deterministically and in milliseconds, while
`test/plugin-e2e.test.mjs` re-runs the same contract against the real REPL.

**A scripted model through the real seam.** The e2e test fakes the model by
implementing the *actual* `ctx.llm.stream` chunk contract (`block-start`,
`text-delta`, `usage`, `finish`). That fidelity mattered: an invented
`block-start` shape initially failed the run, which is exactly what would have
happened against a real adapter.

**Negative cases are first-class.** Oversized reads, unterminated fences,
declared-frame bombs, unknown session event types, missing final variables,
hostile `repr`, missing interpreters and non-finite budget overrides all have
tests — the failure path is where this system is most likely to lie to an
operator.

**Two regression guards for bugs found during development:**

- `root-loop.test.mjs › stores a query result in the REPL…` and friends assert
  that intermediate values never appear in any root prompt, which is the
  property that silently breaks if someone "helpfully" logs a query result.
- `plugin-e2e.test.mjs › writes durable sidecar traces…` asserts exactly one
  `rlm/start` and one `rlm/end` per invocation, because a duplicated fan-out had
  doubled every event (and would have doubled every cost figure derived from
  them).

## Not covered here

| Area | Why | Where it is tracked |
|---|---|---|
| Benchmark scores (S-NIAH, OOLONG, OOLONG-Pairs, BrowseComp-Plus, LongBench-v2) | needs live model budget and the benchmark corpora | [limitations.md](limitations.md#no-benchmark-numbers-yet) |
| Container/Landlock confinement, cgroup CPU/RSS quotas | not implemented in the reference provider | [limitations.md](limitations.md#sandbox-authority) |
| Property tests for protocol fuzzing (fast-check) | the framing suite covers hostile inputs by hand, not by generator | [limitations.md](limitations.md#not-yet-done) |
| Multi-day soak tests (10M / 100M characters) | time-boxed exercise | [limitations.md](limitations.md#not-yet-done) |

## Reproducing the intercepted bugs

For a reviewer who wants to confirm the suites bite, each of these is a
one-line change that turns a green suite red:

1. Remove `import importlib` from `_install_guards` and revert to
   `sys.modules.get(...)` → `blocks the guest network and subprocess APIs` fails.
2. Remove the `child.on('error', …)` call to `handleExit` → the kernel suite hangs
   (the failed-spawn test never settles).
3. Put `config.baseSink` back as the `SessionEventSink` fallback inside the
   composite → the e2e trace test reports two `rlm/start` events.
4. Drop `payloadChars` from the store metadata and clamp reads to `ref.chars`
   again → list paging returns the wrong items.
