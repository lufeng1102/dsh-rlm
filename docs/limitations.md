# Limitations and open items

Everything here is a deliberate, stated gap rather than an oversight. The rule
the work plan sets — *a release is blocked if it exceeds a hard budget, leaks a
process or permits authority escalation, regardless of benchmark score* — is
respected in the other direction too: no claim is made that the tests do not
support.

## Sandbox authority

**What is implemented and tested**

- one OS process per invocation, one process group, terminated as a group on
  cancel/timeout/lifetime/unload;
- a private working directory per invocation (`<sessionCwd>/rlm-kernel-*`),
  removed on dispose;
- a scrubbed child environment: `PATH`, `HOME`/`TMPDIR` inside the workspace,
  `LANG`, `PYTHON*` — no `DSH_*`, no credentials, no inherited secrets;
- a hard lifetime, a per-eval deadline, and a bounded control frame in both
  directions;
- guest-level guards that block `socket.*` connect paths, `subprocess`,
  `multiprocessing`, `ctypes`, `pty`, `asyncio` subprocess helpers and the
  `os.exec*`/`os.system`/`os.fork` family — with the tested caveat that they are
  *not* a security boundary.

**What is not implemented**

- no syscall confinement (Landlock, seccomp, containers) in the reference
  provider;
- no cgroup CPU/RSS quota, so `capabilities.processTreeLimits` is `false`;
- no network namespace, so the declared capability is `network: 'guest-guard'`,
  not `network: 'disabled'`;
- a determined program can bypass the guest guards (e.g. by reloading a module
  or issuing raw syscalls through a path the interpreter still exposes).

**Why this is safe by construction rather than by comment**:
`RlmKernelRuntime.capabilities` is a declaration, and `evaluateIsolation`
compares it against the deployment policy at load time. A policy that requires
`network: 'disabled'` **refuses to load** this provider unless the operator
either configures a real wrapper or explicitly sets
`allowUnhardenedProvider: true`. When the plugin loads with gaps, it records them
(`service.degradations`) and the system-prompt section tells the model — and the
operator's logs — that model-generated code must be treated as trusted-local.

**To harden a deployment**, set `isolationCommand` to a wrapper that actually
confines (`bwrap --unshare-net …`, a container runtime, or a platform sandbox)
and `isolationCommandDisablesNetwork: true` when that wrapper denies egress.
With that in place the provider may honestly report `network: 'disabled'` and the
policy can stop allowing unhardened providers.

## No benchmark numbers yet

§7 of the work plan asks for a full ladder — base model, CodeAct, CodeAct with a
sub-call tool, compaction agent, context-offloading coding agent,
`RLM(depth=0/1/2/3)` — plus OOLONG-Pairs as the required symbolic-recursion test,
with p50/p90/p95 cost and latency. **None of that has been run here**, because it
needs live model budget and the benchmark corpora.

The procedure to produce them, and the metric mapping that needs no new
instrumentation, is in [evaluation.md](evaluation.md).

Consequently this repository makes **no comparative performance claim**. What it
does claim is that the mechanism such an evaluation requires exists, is
measurable, and is tested: root iterations, query counts, concurrency peaks, token
usage, context bytes read, artifacts and per-invocation stop reasons are all
recorded per run in the sidecar trace, so the metrics of §7.3 can be computed
from a trajectory without instrumenting anything new.

The ladder also needs the non-RLM baselines to be measured *separately*, and the
work plan is explicit that `workflow-ptc` must not be presented as this algorithm.
It is not: `run_rlm` is the only RLM entry point, and the tool description steers
ordinary questions away from it.

## Session event vocabulary

`rlm/*` events are **always** durable in the per-invocation sidecar trace. They
are written into the *session log* only when the running harness build declares
the type. This is not caution for its own sake; it is what
`dsh-v0.2.0-rc.2` requires, and the exact mechanism was verified against that
release:

- `Session.append(type, data)` does **not** reject an unknown type, and the JSONL
  writer does not filter one — so a custom event *persists*;
- but the reader refuses an unknown type unless the stored envelope carries
  `ignorable: true`, and `Session.append()` provides no way to set that marker;
- therefore appending `rlm/*` before the type exists in the in-repo
  `SessionEventMap` vocabulary produces a session log that build can no longer
  load.

`SessionEventSink` therefore checks `KNOWN_SESSION_EVENT_TYPES` (exported by
`@deepseek-ai/dsh-session`) and routes undeclared types to the sidecar instead,
reporting the skip once per type. The guard is tested
(`events.test.mjs › appends only event types the running build declares`).

**To make `rlm/*` durable in the session itself**, a deployment must add the six
types to the in-repo `SessionEventMap` (and mark them `ignorable: true` on any
imported envelope). No code change is needed in this plugin: the same sink starts
writing session events the moment the vocabulary knows the names.

## No heap checkpoints

An arbitrary Python heap is not serializable, and this plugin does not pretend
otherwise:

- the durable facts are the events and the artifacts — **not** the live REPL;
- kernel death ends the *attempt* with a durable failure record;
- a new attempt can start from the immutable input artifact and captured
  configuration, but automatic replay is not performed, because a query or an
  optional tool binding may have external side effects;
- `invocationId`/`attemptId` distinguish attempts so duplicate charging or
  duplicate publication is detectable.

Any future checkpoint support will be limited to explicitly registered JSON-safe
variables, and will never claim to snapshot arbitrary runtime objects.

## Model qualification

The shipped profiles (`paper-gpt5`, `paper-qwen-coder`, `paper-qwen3-8b`) are
*paper-reported* shapes (context window, output ceiling, sub-call size, prompt
variant). Their `qualified` flag is `false`, meaning **this plugin has not
verified them against its own suite on a live model**. The paper's own §4 finding
is that one RLM prompt does not transfer across models, so treating an unqualified
route as equivalent is exactly the mistake the flag exists to prevent. Set
`requireQualifiedProfile: true` in a deployment that must not run unqualified
routes.

## Model-visible summary of intermediate state

Only bounded previews, code and errors enter the root transcript. This is by
design and is enforced (`maxRootHistoryChars`), but it means the root model can
occasionally be blind to something a human would consider obvious — for example
the *complete* contents of a large leaf answer it has not asked to inspect. The
prompt compensates by instructing the model to keep buffers in the REPL and to
print only what it needs; a deployment that wants different truncation can raise
the preview limits, at the cost of root-iteration headroom.

## Not yet done

| Item | Notes |
|---|---|
| Property-based protocol fuzzing | the framing suite covers hostile inputs by hand; a generator-based test is not included |
| Soak tests at 10M / 100M characters | the store's streaming paths exist and are exercised at ~200 k characters, not at soak scale |
| Artifact encryption + key management | the seam exists (`cipher`, `requireEncryption` fails closed) but no cipher ships |
| Full session `SessionProjection` registration | blocked on the vocabulary item above. The projection logic exists and is tested as a pure fold over durable events — `foldRlmRuns` / `aggregateRlmMetrics` in `src/events.ts`, covered by `events.test.mjs › trajectory fold and metrics` — so a host that patches the vocabulary (or a future release) can register it without new logic |
| Prime Agent companion features (continuable children, `sendMessage` routing) | explicitly Phase 4 in the work plan and out of scope here |
| `depth=3` soak and safe-operation verification | depth 3 is permitted by configuration; only 0/1/2 are covered by tests |
| Windows-specific process-tree termination | the provider falls back to `child.kill()` off POSIX; no Windows CI was run |

## What would block a release

Per the work plan's own gate, any of these blocks a release regardless of
benchmark score: exceeding a hard budget, leaking a process, permitting authority
escalation, or publishing a comparative claim on an unqualified profile. The
current state satisfies the first three within the process-level isolation
described above, and makes no comparative claim at all.
