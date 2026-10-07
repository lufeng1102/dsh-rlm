# Paper alignment

Two things are checked here: the algorithmic contract of §1.1 (what makes an RLM
an RLM), and the acceptance list of §7.4 (the work plan's own definition of
"RLM-complete"). Every row names the code and the test that enforce it.

## §1.1 algorithmic contract

| Requirement | Where it lives | Test |
|---|---|---|
| `P` is not in root history; it is a REPL variable | `service.ts` `admitContext` + `buildMetadataMessage` | `root-loop.test.mjs › keeps the prompt out of root history…`, `› admits a prompt far larger than the root model window` |
| The REPL is persistent for the whole invocation | `kernel/python/rlm_kernel.py` `ReplState` | `kernel-python.test.mjs › keeps one namespace across evaluations…`, `test_rlm_kernel.py › ReplStateTests.test_keeps_one_namespace_across_evaluations` |
| The root model can inspect bounded previews only | `prompt.ts` `truncate`, `maxStdoutPreviewChars` | `root-loop.test.mjs › records bounded root history in every step event` |
| `llm_query()` is callable from program code, including loops | driver bindings + `service.ts` `llmQuery` | `kernel-python.test.mjs › returns llm_query results to the program…`, `› fans out llm_query_many concurrently`, `root-loop.test.mjs › lets REPL code loop over prompt slices…` |
| `rlm_query()` creates a child only when depth allows, else falls back | `service.ts` `rlmQuery` (`depth < 2` → leaf call) | `root-loop.test.mjs › downgrades rlm_query to a leaf call at depth 1`, `› creates a child RLM with its own persistent kernel at depth 2` |
| A sub-call returns a value to the REPL, not a model-visible turn | `RlmKernelHost.llmQuery` return path | `root-loop.test.mjs › stores a query result in the REPL without putting it in root history` |
| Finalization is unambiguous | `finalize.ts` closed union + `publishText`/`publishVariable` | `finalize.test.mjs` (24 cases), `root-loop.test.mjs › bounds parser retries…` |
| Execution feedback is bounded metadata only | `buildStepFeedback` + `assertRootHistoryBounded` | `root-loop.test.mjs › records bounded root history in every step event` |

## §1.2 depth semantics

| Depth | Guaranteed behaviour | Test |
|---|---|---|
| `0` | REPL + externalization, **no** model bindings in the namespace | `kernel-python.test.mjs › omits model bindings at depth 0…`, `root-loop.test.mjs › installs no model bindings at depth 0` |
| `1` | `llm_query` allowed; `rlm_query` degrades to a leaf call | `root-loop.test.mjs › downgrades rlm_query…` |
| `2+` | child RLM with its own kernel, strictly lower remaining depth | `root-loop.test.mjs › creates a child RLM…`, `› never lets a child exceed the parent remaining budget` |

`defaultDepth` defaults to `1`, matching the paper. Depth is host-derived: the
root state machine passes `depth - 1` to a child and clamps it to `maxDepth`, so
nothing the model writes can raise it.

## §7.4 acceptance tests

| # | Acceptance criterion | Status | Evidence |
|---|---|---|---|
| 1 | a prompt larger than the root window is admitted without entering history | **pass** | `root-loop.test.mjs › admits a prompt far larger than the root model window` (5 MB prompt, every root prompt < 20 kB) |
| 2 | root code can loop over arbitrary prompt ranges and issue multiple `llm_query` calls | **pass** | `root-loop.test.mjs › lets REPL code loop over prompt slices…`, `plugin-e2e.test.mjs › runs a full RLM through the tool…` (real Python REPL, 5 slice calls) |
| 3 | a query result can be stored and transformed in the REPL without entering history | **pass** | `root-loop.test.mjs › stores a query result in the REPL…` (4,000-char answer never appears in any root prompt) |
| 4 | `RLM(depth=1)` permits leaf calls and downgrades recursive calls | **pass** | `root-loop.test.mjs › downgrades rlm_query to a leaf call at depth 1` (exactly one kernel is created) |
| 5 | `RLM(depth=2)` creates a child RLM with its own persistent REPL | **pass** | `root-loop.test.mjs › creates a child RLM with its own persistent kernel at depth 2` (two kernels, child depth 1) |
| 6 | a child cannot exceed the parent's remaining depth or budget | **pass** | `budget.test.mjs` (9 conservation/clamping cases) + `root-loop.test.mjs › never lets a child exceed the parent remaining budget` |
| 7 | a child cannot increase sandbox authority, model permissions or network access | **pass (as policy)** | child accounts are `narrowLimits(requested, parent.remaining)`; routes are resolved in `service.ts`, not read from the child; the prompt never carries authority. See [limitations.md](limitations.md#sandbox-authority) for what the provider can and cannot *prove*. |
| 8 | cancellation terminates the root kernel and all admitted child queries | **pass** | `kernel-python.test.mjs › kills the process group when the caller aborts`, `› terminates a runaway program on the per-eval deadline`, `root-loop.test.mjs › cancels the kernel and all child queries…`, `plugin-e2e.test.mjs › cancels the run and kills the kernel…` |
| 9 | a kernel crash produces a durable failure event and leaks no process | **pass** | `root-loop.test.mjs › ends the run when the kernel dies, without leaking it`, `kernel-python.test.mjs › does not leak a kernel process…`, `plugin-e2e.test.mjs › leaves no kernel workspace behind after unload` |
| 10 | root history holds bounded metadata, code and output previews | **pass** | `root-loop.test.mjs › records bounded root history in every step event` (every `rlm/step` carries `rootHistoryChars`) |
| 11 | a final value above the inline limit returns as an artifact | **pass** | `root-loop.test.mjs › returns a final value above the inline limit as an artifact…`, `plugin-e2e.test.mjs › returns a large answer as an artifact reference…` |
| 12 | network, subprocess, CPU, memory, PID and disk limits fail closed under adversarial code | **partial** | guest-level network/subprocess guards are real and tested (`kernel-python.test.mjs › blocks the guest network and subprocess APIs`), a hard lifetime and eval deadline are enforced by killing the process tree, and startup fails closed when the interpreter is missing. There is **no** CPU/RSS quota or syscall confinement in the reference provider — `capabilities` says so, `evaluateIsolation` refuses the plugin when a policy demands more than declared, and the row is intentionally not claimed green. See [limitations.md](limitations.md). |
| 13 | OOLONG-Pairs improves over `depth=0` and fixed chunk fan-out | **not run** | the evaluation harness and the benchmark tasks are out of scope for this implementation; §7 needs live model budget. [docs/limitations.md](limitations.md#no-benchmark-numbers-yet) records this as an open item, and no comparative claim is made. |

## §1.3 anti-patterns the work plan forbids

| Forbidden | Status |
|---|---|
| `agent()` as the implementation of `llm_query()` | not done: `DshModelClient.complete` issues a one-shot, tool-free `ctx.llm.stream` request and assembles it itself |
| `contextRef` alone as "enough" | not done: the REPL gets `context.peek/slice/items/get/search/materialize` over a lazy handle |
| a fixed chunk fan-out inside the plugin | not done: the *model* chooses ranges; the host only bounds them |
| putting the prompt into `workflow.args` | not done: `run_rlm` stores it and passes a reference |
| claiming arbitrary REPL heap recovery | not done: [recovery semantics](limitations.md#no-heap-checkpoints) are explicit |
| a second session JSONL trajectory | not done: the sidecar trace is an artifact under the artifact root, and session writes go through the documented event types only |
| unbounded persistent processes | not done: `maxWallClockMs`, `maxLifetimeMs`, `maxConcurrentKernels`, `maxTotalKernels` and process-group kill |
| `workflow-ptc` presenting itself as this algorithm | not done: `run_rlm` is the only RLM entry point, and the tool description steers ordinary questions away from it |

## Prompt fidelity

The system prompt in `src/prompt.ts` follows Appendix C of the paper
(context type/chunk lengths, the bindings list, the truncation warning, the
"look through the entire context" strategy, the `FINAL`/`FINAL_VAR` protocol,
the cost-of-`llm_query` warning for the Qwen profile) with two deliberate
deltas, both recorded in the prompt itself: the host limits this deployment
actually enforces, and an explicit `depth=0`/`depth≥2` variant. The profile id
and variant are written into every `rlm/start` event, because the paper's own
§4 result is that one prompt does not transfer across models — so the prompt
version is part of the experiment's identity, not an implementation detail.
