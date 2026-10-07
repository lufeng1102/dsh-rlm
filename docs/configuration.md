# Configuration

Three plugin rows, all opt-in. Every numeric value is a **hard host ceiling** —
never a prompt instruction, never readable from the prompt artifact, and never
raisable by model-written code.

## Rows

```yaml
- id: rlm-context-local
  name: '@deepseek-ai/dsh-rlm/context-local'
  config: { root: '.dsh/rlm-artifacts' }

- id: rlm-kernel-python
  name: '@deepseek-ai/dsh-rlm/kernel-python'
  config: { pythonPath: python3, guestNetworkGuard: true }

- id: rlm
  name: '@deepseek-ai/dsh-rlm'
  config:
    rootProvider: apifusion
    rootModel: deepseek-v4-pro
```

The ready-to-edit version is [cordis.patch.yml](../cordis.patch.yml).

## `rlm-context-local`

| Key | Default | Meaning |
|---|---|---|
| `root` | `.dsh/rlm-artifacts` | payloads, metadata and `traces/` |
| `charIndexStep` | `65536` | characters between char→byte index samples; `0` disables the index (reads then scan from the start) |
| `maxRangeChars` | `16000000` | ceiling on one range read, in characters |
| `maxRecordBytes` | `8388608` | ceiling on one `record`-shaped context |
| `maxScanBytes` | `16777216` | bytes scanned by one `search()` |
| `requireEncryption` | `false` | when `true` and no cipher is configured, the plugin **fails to construct** instead of silently storing plaintext |

Layout: `data/<sha256>.data`, `meta/<sha256>.json`, `artifacts/art_*.data` +
`.meta.json`, `traces/<invocationId>.jsonl`. Identical prompts deduplicate by
content hash. Every ref carries its owning `sessionId`; a read from another
session raises `RlmAccessError`, and `sweep()` implements retention.

## `rlm-kernel-python`

| Key | Default | Meaning |
|---|---|---|
| `pythonPath` | `python3` (or `$DSH_RLM_PYTHON`) | interpreter for the driver |
| `driverPath` | the shipped `rlm_kernel.py` | override only when repackaging |
| `maxFrameBytes` | `67108864` | ceiling on one control frame, both directions |
| `killGraceMs` | `2000` | wait between SIGTERM and SIGKILL of the process group |
| `guestNetworkGuard` | `true` | install guest-level socket/subprocess/`os.exec*` guards |
| `isolationCommand` | — | wrapper prepended to the driver invocation (`bwrap`, `sandbox-exec`, a container runtime) |
| `isolationCommandDisablesNetwork` | `false` | declares that the wrapper really denies network access |
| `env` | `{}` | extra child environment on top of the scrubbed base |

The child environment is rebuilt from scratch (`PATH`, `HOME`/`TMPDIR` inside the
private workspace, `LANG`, `PYTHON*`), so host credentials and `DSH_*` variables
never reach model-written code. Each invocation gets a fresh
`<sessionCwd>/rlm-kernel-*` workspace, removed on dispose.

### What the provider claims

`capabilities` is compared against the deployment policy
(`DEFAULT_ISOLATION_POLICY` in `src/kernel/index.ts`) at load time:

```text
language: python              network: 'guest-guard' | 'disabled' | ...
isolation: 'process' | 'container'
processTreeLimits: false      processTreeTermination: true
hardDeadline: true            scrubbedEnvironment: true
```

`evaluateIsolation` refuses the plugin (`RlmIsolationRefusedError`) when the
policy requires more than the provider declares and
`allowUnhardenedProvider` is `false`. Until a wrapper is configured, the honest
claim is *guest-guarded*, not *network-disabled* — see
[limitations.md](limitations.md).

## `rlm`

### Depth and routes

| Key | Default | Meaning |
|---|---|---|
| `defaultDepth` | `1` | used when a call does not specify one; clamped to `maxDepth` |
| `maxDepth` | `3` | hard ceiling on any requested depth |
| `rootProvider` / `rootModel` | — | root-loop route; unset ⇒ `route-unset` failure |
| `leafProvider` / `leafModel` | root route | `llm_query` route |
| `childProvider` / `childModel` | leaf route | `rlm_query` child route |
| `promptProfileId` | `paper-gpt5` | one of `paper-gpt5`, `paper-qwen-coder`, `paper-qwen3-8b` |
| `rootContextWindowTokens` | profile value | used in prompt text and diagnostics |
| `subCallChars` | profile value | reported to the model as what a leaf call can absorb |
| `requireQualifiedProfile` | `false` | refuse to load a profile not marked `qualified` |
| `rootMaxTokens` / `leafMaxTokens` | `32000` / `16000` | per-call output ceilings |
| `temperature` | provider default | passed through when set |

### Budgets (per invocation ceiling)

| Key | Default |
|---|---|
| `maxRootIterations` | `64` |
| `maxWallClockMs` | `900000` |
| `maxTotalQueries` | `4096` |
| `maxConcurrentQueries` | `16` |
| `maxTotalKernels` | `8` |
| `maxConcurrentKernels` | `8` |
| `maxInputTokens` | `8000000` |
| `maxOutputTokens` | `1000000` |
| `maxContextReadBytes` | `536870912` |
| `maxQueryPromptChars` | `2000000` |
| `maxInlineOutputBytes` | `1048576` |
| `maxArtifactBytes` | `1073741824` |
| `maxCostUsd` | `25` |
| `pricing` | `{}` | `{ '<provider>/<model>': { inputUsdPerMTok, outputUsdPerMTok } }` |

A call may pass a `budget` object to `run_rlm`; values are sanitized (known
dimensions, finite, non-negative) and then **intersected** with the deployment
ceiling, so a caller can only ever narrow its own budget.

### Storage, prompt and output

| Key | Default | Meaning |
|---|---|---|
| `maxPromptChars` | `100000000` | admission ceiling for one prompt (storage, not model window) |
| `maxRootHistoryChars` | `400000` | bound on the root transcript; breaching it ends the run |
| `maxParseRetries` | `2` | bounded protocol corrections, charged as ordinary turns |
| `maxCodePreviewChars` | `4000` | code echoed back per turn |
| `maxStdoutPreviewChars` | `2000` | stdout echoed back per turn |
| `maxErrorPreviewChars` | `1500` | traceback excerpt echoed back |
| `maxRenderedChars` | `20000` | longest inline answer rendered to the *calling* model |
| `maxTaskChars` | `8000` | ceiling on `prompt` when material travels as `context` |
| `finalVariableName` | `Final` | the paper's `state[Final]` variable |
| `contextStoreRoot` | `.dsh/rlm-artifacts` | where the trace sink writes |
| `writeSessionEvents` | `true` | allow session writes for declared `rlm/*` types |
| `toolName` | `run_rlm` | model-facing tool name |

## `run_rlm` arguments

| Argument | Notes |
|---|---|
| `prompt` (required) | The question, or the whole prompt when no `context` is given (the paper's shape: the question then lives inside the artifact and is read from the REPL). |
| `context` | Optional long material: string → text, array of strings → documents, object → record. When present, `prompt` becomes the bounded *task* and the material is what gets externalized. |
| `depth` | Requested recursion depth; clamped to `maxDepth`. |
| `model` | `provider/model` or a bare model id (provider taken from configuration). |
| `budget` | Partial budget override, intersected with the ceiling. |

The result is a canonical object (`status`, `stopReason`, `answer` *or*
`artifact`, `steps`, `depth`, `usage`, optional `error`). A large answer is
rendered as an artifact reference plus a bounded preview — never inlined.

## Session events

`rlm/start`, `rlm/step`, `rlm/query-start`, `rlm/query-end`, `rlm/artifact`,
`rlm/end` are written to a per-invocation sidecar trace under
`<contextStoreRoot>/traces/`. They are written into the *session log* as well
only for types the running harness build declares — see
[session events](limitations.md#session-event-vocabulary) for the constraint and
what a deployment must do to make `rlm/*` durable in the session itself.

## Failure modes an operator should expect

| Symptom | Cause | Remedy |
|---|---|---|
| `route-unset` | `rootProvider`/`rootModel` not configured | set the route in the profile patch |
| plugin never loads, no error | a required service is missing (`inject` PENDING) | check that both capability plugins are in the bundle list *before* the core row |
| `RlmIsolationRefusedError` at load | policy demands more isolation than the provider declares | configure `isolationCommand` (+ `isolationCommandDisablesNetwork`), or set `allowUnhardenedProvider: true` for trusted local use |
| `budget-exceeded` with few steps | a budget dimension was exhausted (often `maxTotalQueries` or `maxCostUsd`) | inspect `rlm/end.usage`; raise the specific ceiling deliberately |
| `kernel-failure` with `worker-exit` | the child died (OOM, SIGKILL, interpreter crash) | check the trace's `stderr` observations; the run is intentionally not retried automatically |
