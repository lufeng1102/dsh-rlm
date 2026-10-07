# Architecture

## Component map

```text
                     ┌─────────────────────── harness ───────────────────────┐
  model ── run_rlm ─▶│ tool.ts  (defineTool: parameters + output schema)     │
                     │   └─ builds per-run sink, abort bridge, routes        │
                     └───────────────┬───────────────────────────────────────┘
                                     ▼
      ctx.rlm ── service.ts (RlmService): the root state machine
        │  admit prompt            │  root loop                    │  publish
        ▼                          ▼                               ▼
  context-store.ts seam     kernel/index.ts seam            context-store.ts
  context-local.ts          kernel/python/provider.ts       artifacts + traces
  (content-addressed,       (process, framed JSON,          (writeArtifactFromFile
   Unicode-safe reads)       process-tree kill)              streams the file)
                                     │
                                     ▼
                        kernel/python/rlm_kernel.py   ← the persistent REPL
                        context · llm_query · rlm_query · Final
```

Three plugins, three seams, one dependency direction:

| Plugin | Mounts | Depends on |
|---|---|---|
| `@deepseek-ai/dsh-rlm/context-local` | `ctx.rlmContextStore` | node fs only |
| `@deepseek-ai/dsh-rlm/kernel-python` | `ctx.rlmKernelRuntime` | node child_process, python3 |
| `@deepseek-ai/dsh-rlm` | `ctx.rlm`, the `run_rlm` tool, a prompt section | `tools`, `llm`, `rlmContextStore`, `rlmKernelRuntime`, `systemPrompt` |

The core plugin declares those five as `inject`. Cordis therefore holds it
PENDING until a store and a kernel provider are mounted, and unloads it if either
disappears. **There is no fallback**: a missing dependency means no RLM, never an
unbounded or unsandboxed one. `subagents` and durable continuable children are
deliberately *not* injected — the paper does not need them.

## One invocation, end to end

1. **Admission.** `RlmService.run` clamps the requested depth to
   `min(request.depth ?? defaultDepth, maxDepth)`, checks the route is configured
   (a missing route is `route-unset`, not a silent default), enforces
   `maxConcurrentKernels`, and builds the root `RlmBudgetAccount` from the
   deployment ceilings intersected with any per-call overrides.
2. **Externalization.** The prompt goes into the content-addressed store. The
   store returns `{id, sha256, kind, bytes, chars, chunkChars, estimatedTokens}`.
   The model never receives the text.
3. **Kernel start.** The provider creates a private workspace under the session
   cwd, spawns `python3 -u rlm_kernel.py` with a scrubbed environment
   (`HOME`/`TMPDIR` inside that workspace, no `DSH_*`, no credentials), and
   performs a handshake: `ready` → `init{bindings, depth, guardNetwork, …}` →
   `ready{phase: 'initialized'}`. Binding names come from `bindingsForDepth`, so
   `depth=0` genuinely has no `llm_query` in the namespace.
4. **Root loop.** The root model is called with the profile system prompt plus a
   growing, *bounded* user-role transcript: turn 0 is metadata, each later turn is
   `bounded(code)` + `bounded(stdout|error)`. Every response must parse to exactly
   one action; a mismatch is a bounded correction charged to the same iteration
   budget.
5. **Evaluation.** `kernel.eval(program)` runs the program in the persistent
   namespace on its own thread while a *separate* reader thread serves frames —
   that split is what lets a program blocked on `llm_query` still receive
   `binding-result` and `cancel`. Bindings dispatch to the host:
   `llm_query`/`rlm_query` → the model seam, `context.*` → the store through a
   byte-metering wrapper.
6. **Finalization.** Either the REPL sets `Final` (detected after each eval), or
   the root replies `FINAL(...)` / `FINAL_VAR(...)`. A variable final is
   *exported by the kernel* into its own workspace, and the host either reads that
   file (under the inline ceiling) or streams it into managed artifact storage —
   so a 1 GB answer crosses the control channel as bounded metadata only.
7. **Teardown.** The kernel process group is SIGTERMed, then SIGKILLed after a
   grace period, the read loop is drained, the private workspace is removed, and
   exactly one `rlm/end` is emitted for the one `rlm/start`.

## Kernel wire protocol

Both directions are length-framed JSON: `<decimal byte length>\n<utf-8 JSON>`.
Framing (rather than newline-delimited JSON) is required because programs print
multi-megabyte strings that legitimately contain newlines; a newline protocol
would corrupt them silently.

| Host → driver | Meaning |
|---|---|
| `{op:'init', config}` | install bindings, guards, limits |
| `{op:'eval', id, code, timeoutMs}` | run one program in the persistent namespace |
| `{op:'export', id, name, path, previewChars, maxBytes}` | serialize a variable to a file |
| `{op:'binding-result', callId, ok, value|error}` | answer a program-initiated call |
| `{op:'cancel', callId}` / `{op:'shutdown'}` | per-call cancel; orderly exit |

| Driver → host | Meaning |
|---|---|
| `{type:'ready', protocol, phase?, bindings?}` | handshake |
| `{type:'result', id, stdout, stdoutTruncated, error?, final?, elapsedMs}` | one step |
| `{type:'binding', callId, name, args}` | a program called the host |
| `{type:'export', id, ok, bytes, chars, sha256, preview, truncated}` | export outcome |
| `{type:'fatal'}` | protocol/sandbox failure; the host reaps the process |

Failures are *fields*, not exceptions: a syntax error is `kind:'syntax'`, a
missing variable is `kind:'name'`, a wedged eval is `fatal:true, kind:'timeout'`
followed by `os._exit(3)` so the host can reap the process tree. The host turns
each into a truthful step error and, when fatal, a durable `rlm/end`.

## Why these seams exist

- **`RlmContextStore` as a service**: a deployment may need encryption at rest,
  tenant isolation or remote storage. `LocalRlmContextStore` implements the
  reference behaviour — content addressing, per-session authorization,
  character-range reads that never split a code point, retention and GC — and the
  RLM core depends only on the seam.
- **`RlmKernelRuntime` as a service, not PTC**: `@deepseek-ai/dsh-ptc-runtime`
  intentionally starts a fresh substrate per run; persisting a REPL by changing
  that contract would break an existing capability. This seam adds persistence
  beside PTC and reuses PTC's defensive *patterns* (managed child, framed
  messages, cancellation, byte ceilings), not its one-shot semantics.
- **`ctx.rlm` owns policy, providers own mechanism**: depth, budgets, routes,
  prompt profiles and admission control live in the core; the provider only
  executes and reports. That inversion is why a model-written program cannot
  widen its own authority.

## Load and unload

```text
load   plugins mount → capabilities compared against the isolation policy
       → ctx.rlm registers the service, run_rlm tool and prompt section
run    externalize → start kernel → rlm/start → loop → rlm/end → dispose kernel
unload unregister tool + prompt section → reject new runs → abort live kernels
       → disposeAll (process-tree kill) → flush traces → remove workspaces
```

`test/plugin-e2e.test.mjs` asserts the unload half: services disappear together,
contributions are unregistered, and no `rlm-kernel-*` workspace survives.
