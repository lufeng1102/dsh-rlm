# `@deepseek-ai/dsh-rlm` — a paper-aligned Recursive Language Model plugin for DeepSeek Harness

This is the implementation of `RLM-work/README.md` as a loadable DeepSeek
Harness plugin: a **persistent-REPL, prompt-externalizing, recursively
sub-calling** inference scaffold for `dsh-v0.2.0-rc.2`, matching Algorithm 1 of
*Recursive Language Models* (arXiv [2512.24601](https://arxiv.org/abs/2512.24601)).

The distinction the whole project turns on:

| | `workflow-ptc` (already shipped by dsh) | this plugin |
|---|---|---|
| prompt location | workflow arguments | a symbolic REPL handle (`context`) |
| root history | carries the material | bounded metadata, code and stdout previews only |
| sub-calls | `agent()` on pre-chosen chunks | `llm_query()` callable from inside program loops |
| recursion | none | `rlm_query()` with host-derived, strictly decreasing depth |
| intermediate values | host-chosen | symbolic, in the REPL |

`workflow-ptc` remains the fastest way to do a fixed-chunk fan-out. It is a
separate algorithm and this plugin does not claim otherwise; §7 of the work plan
lists it as a baseline to be measured *against*, not as a mode of this plugin.

## Quick start

```bash
cd courses/核心/CS329Z/readings/assignments/RLM-work/dsh-rlm

node scripts/link-dev-deps.mjs   # point node_modules at the local dsh install
node scripts/build.mjs           # tsc → lib/, and copy the Python driver
node --test                      # 149 Node tests (unit + real-kernel + plugin e2e)
python3 -m unittest discover -s src/kernel/python -p 'test_*.py'   # 30 driver tests
node scripts/demo.mjs --chars 400000   # narrated end-to-end run
```

`npm run verify` does the build, both test suites, and nothing else.

The demo prints what the paper actually promises, on a 400k-character prompt:

```text
  1. prompt externalized   root history is 1,400 chars for a 400,102 char prompt
     needle in root history: no
  2. persistent REPL       2 root steps in one namespace (step 1 built `answer`, step 2 returned it)
  3. symbolic sub-calls    21 leaf calls returned into variables, not into history
  4. large final output    returned inline (below the ceiling)
```

## Layout

```text
dsh-rlm/
  src/
    index.ts                loadable plugin `rlm`: service + run_rlm tool + prompt policy
    plugins/
      context-local.ts      loadable plugin `rlm-context-local`: ctx.rlmContextStore
      kernel-python.ts      loadable plugin `rlm-kernel-python`: ctx.rlmKernelRuntime
    service.ts              the root state machine (Algorithm 1) and ctx.rlm
    types.ts                the host-side contract vocabulary
    budget.ts               hierarchical reserve-then-settle budget accounts
    scheduler.ts            bounded, cancellation-aware query admission control
    finalize.ts             structured root actions + FINAL / FINAL_VAR parsing
    prompt.ts               prompt profiles, metadata message, bounded feedback
    llm.ts                  leaf/root model calls over ctx.llm.stream
    context-store.ts        the context/artifact service seam
    context-local.ts        content-addressed local store (Unicode-safe range reads)
    kernel/index.ts         the persistent-kernel service seam + isolation policy
    kernel/protocol.ts      length-framed JSON framing
    kernel/python/
      provider.ts           process lifecycle, binding dispatch, process-tree kill
      rlm_kernel.py         the persistent Python REPL driver
      test_rlm_kernel.py    driver unit tests (python3 -m unittest)
    events.ts               rlm/* vocabulary, sinks, session-vocabulary guard
    tool.ts                 the model-facing `run_rlm` tool definition
    config.ts               the YAML surface and its resolution into host policy
    testing.ts              shipped test doubles (scripted model, in-memory kernel)
  cordis.patch.yml          bundle patch to mount the three plugins in a profile
  docs/                     architecture, paper alignment, configuration, tests, limits
```

## Installing into a profile

The plugin is a bundle package: it declares `dsh.bundle.patch` and ships
`cordis.patch.yml` (see [docs/configuration.md](docs/configuration.md) for the
full row). For a local checkout:

```jsonc
// <profile>/package.json
{
  "dependencies": { "dsh-rlm": "link:/abs/path/to/dsh-rlm" },
  "dsh": { "profile": { "bundles": ["@deepseek-ai/dsh-base", "dsh-rlm"] } }
}
```

Then set `rootProvider` / `rootModel` (and optionally `leafProvider` /
`leafModel` / `childProvider` / `childModel`) in the profile's
`cordis.patch.yml`. An unset route makes `run_rlm` fail with `route-unset`
rather than quietly running on a different model.

## The contract, in one screen

```text
state   = InitREPL(prompt = P)        # P is a REPL handle, never history
history = [Metadata(context, limits)] # bytes, hash, chunk lengths, bindings
while within budget:
    action = root_model(history)      # exactly one of eval | final-text | final-variable
    if action is eval:
        step = kernel.eval(code)      # persistent namespace, bindings dispatch to the host
        history += bounded(code) + bounded(stdout) + bounded(error)
        if the REPL set `Final`: publish it and stop
    else:
        validate the variable/answer and publish it
```

Everything else — depth, budgets, routes, sandbox policy — is host-owned and
never readable from the prompt artifact.

## Documentation

| Document | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | components, seams, kernel protocol, load/unload lifecycle |
| [docs/paper-alignment.md](docs/paper-alignment.md) | §7.4 acceptance matrix: each paper property → code → test |
| [docs/configuration.md](docs/configuration.md) | every configuration key, defaults and what it protects |
| [docs/testing.md](docs/testing.md) | test inventory, what each suite proves, how to run it |
| [docs/evaluation.md](docs/evaluation.md) | the §7 baseline ladder, task set, metrics and how to fold traces into them |
| [docs/limitations.md](docs/limitations.md) | what is *not* production-hardened, and what that means operationally |

## Status

Experimental. The paper semantics pass, `depth=0/1/2` works, the kernel is
isolated at the process level with a scrubbed environment and process-tree
termination, and the plugin's own suites are green. It is **not** a hardened
multi-tenant sandbox, it does **not** attempt arbitrary REPL heap recovery, and
no benchmark numbers from §7 have been produced yet — see
[docs/limitations.md](docs/limitations.md) before pointing it at untrusted code
or claiming comparative results.
