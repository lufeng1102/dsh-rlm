# Evaluation

The work plan's §7 ladder has not been run — see
[limitations.md](limitations.md#no-benchmark-numbers-yet). This document is the
*procedure* it should be run with, and an explanation of why the instrumentation
to run it already exists.

## Baselines to measure, with the same root and leaf models

| # | Baseline | How to configure it here |
|---|---|---|
| 1 | base model, prompt in context | outside this plugin |
| 2 | CodeAct without sub-calls | `RLM(depth=0)`: `defaultDepth: 0` |
| 3 | CodeAct with a sub-call tool | dsh's `run_code` + `agent`/tool surface (separate consumer) |
| 4 | compaction agent | `@deepseek-ai/dsh-compaction-basic` |
| 5 | context-offloading coding agent | dsh coding agent with the corpus on disk |
| 6 | `RLM(depth=0)` | `defaultDepth: 0` |
| 7 | `RLM(depth=1)` | `defaultDepth: 1` (the paper's default) |
| 8 | `RLM(depth=2)`, `RLM(depth=3)` | `defaultDepth: 2` / `3`, `maxDepth` at least that |
| 9 | fixed chunk fan-out (non-RLM) | `@deepseek-ai/dsh-workflow-ptc` |

The critical comparisons are **not** against the base model alone: they are
`RLM(depth=1/2)` versus `RLM(depth=0)` and versus the fixed fan-out, because that
is what isolates the value of symbolic recursion.

## Task ladder

| Task | Expected scaling | Purpose |
|---|---:|---|
| S-NIAH | O(1) | retrieval sanity check |
| OOLONG | O(n) | dense linear aggregation |
| OOLONG-Pairs | O(n²) | symbolic recursion stress test |
| BrowseComp-Plus | multi-hop | realistic long research |
| LongBench-v2 CodeQA | repository reasoning | code context transfer |

Add a small deterministic smoke suite first: the demo script
(`node scripts/demo.mjs --chars 400000`) is that suite for the mechanism — a
needle buried in a long document, with assertions that the prompt never entered
root history, that sub-calls happened, and that the answer was found.

## Procedure

1. **Qualify the model profile.** Run the local smoke suite at `depth=0/1/2` on
   the candidate root route. Then set `qualified: true` for that profile in
   `src/prompt.ts` only if it passes, and turn on `requireQualifiedProfile` for
   the evaluation profile. The paper's §4 finding is that one RLM prompt does not
   transfer across models; without this step the ladder measures the prompt as
   much as the model.
2. **Pin the prompt profile.** Record `identity.promptProfileId` and
   `identity.promptProfileVariant` from `rlm/start` — they are part of the
   experimental identity and must be identical across the arms being compared.
3. **Run each arm** with identical task sets, sampling parameters and budgets.
   Budgets belong in the comparison: an arm that is cheaper only because it was
   allowed fewer sub-calls is not a better arm.
4. **Fold the traces.** Each invocation leaves
   `<contextStoreRoot>/traces/<invocationId>.jsonl`. Load every line, then:

   ```js
   import { foldRlmRuns, aggregateRlmMetrics } from '@deepseek-ai/dsh-rlm'

   const runs = foldRlmRuns(events)              // one summary per invocation
   const metrics = aggregateRlmMetrics(runs)     // counts + p50/p90/p95
   ```

   Both functions are pure and tested
   (`test/events.test.mjs › trajectory fold and metrics`), so an evaluation
   script does not need to re-derive anything from log text.
5. **Report the §7.3 metric set**, per task and per arm:

   | Metric | Where it comes from |
   |---|---|
   | task score (F1 / accuracy / native metric) | the grader |
   | maximum input size and fraction beyond the root window | `rlm/start.contextRef.chars` vs `promptProfile.contextWindowTokens` |
   | total root iterations | `rlm/end.steps`, `rlm/step` count |
   | `llm_query` / `rlm_query` counts | `rlm/query-start` (kind, depth) |
   | maximum and average concurrency | `foldRlmRuns().peakQueryConcurrency`, `usage.maxObservedConcurrency` |
   | input/output tokens and cost | `rlm/end.usage`, priced through `pricing` |
   | p50/p90/p95 latency and cost | `aggregateRlmMetrics()` (a mean alone hides the heavy tail the paper reports) |
   | queue time, reservation failures, throttling | `rlm/query-end.errorKind`, plus scheduler counters exposed by `ctx.rlm` diagnostics |
   | kernel failures, syntax failures, cancellations, timeouts | `rlm/end.errorKind` / `stopReason` |
   | process-tree leak count, cancellation completion time | `runtime.liveKernels` after unload; wall clock between abort and `rlm/end` |
   | artifact bytes and peak live kernels | `rlm/artifact` events; `rlmKernelRuntime.liveKernels` |
   | final-answer protocol failures | `rlm/step.action === 'parse-failure'` |
   | answer quality versus recursion depth | compare the `depth` arms |

## Why the instrumentation is already sufficient

The root loop charges every dimension it enforces to one hierarchical budget
account, and `rlm/end.usage` is that account's snapshot
(`rootIterations`, `queries`, `kernels`, `inputTokens`, `outputTokens`,
`contextReadBytes`, `costUsd`, `maxObservedConcurrency`). Query events carry
per-call latency, tokens and result sizes. Nothing in the metric set above needs
new instrumentation — only a run.

## Acceptance gates that are *not* benchmark-dependent

These block a release regardless of score (work plan §10), and are checkable from
the suites in this repository:

| Gate | Evidence |
|---|---|
| no hard-budget overrun | `root-loop.test.mjs` budget cases, `budget.test.mjs` conservation |
| no leaked process | `kernel-python.test.mjs › does not leak a kernel process…`, `› kills the process group…`, `plugin-e2e.test.mjs › leaves no kernel workspace behind` |
| no authority escalation | `budget.test.mjs › never lets a child widen its parent…`, `root-loop.test.mjs › never lets a child exceed the parent remaining budget` |
| exactly one terminal event per start | `events.test.mjs › flags a start with no end as dangling…`, `plugin-e2e.test.mjs › writes durable sidecar traces…` |
