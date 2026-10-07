#!/usr/bin/env python3
"""Persistent RLM REPL driver.

This process is the *environment* ``E`` of Algorithm 1 in the Recursive Language
Models paper (arXiv 2512.24601). It owns exactly one long-lived Python namespace
for the whole lifetime of one RLM invocation:

* the user prompt lives in this namespace as the symbolic handle ``context``;
* intermediate values produced by model-written code stay here and never travel
  back to the root model unless the program asks for a bounded preview;
* ``llm_query`` / ``rlm_query`` are synchronous callables that proxy to the host
  process, so code can invoke them from inside loops and comprehensions.

Wire protocol (both directions, on stdin/stdout):

    <decimal byte length> "\\n" <utf-8 JSON payload>

stdout carries ONLY framed protocol messages. Anything the program prints is
captured by the driver and returned inside the ``result`` frame. Diagnostics go
to stderr.

Host -> driver frames
---------------------
``{"op": "init", "config": {...}}``
``{"op": "eval", "id": 1, "code": "...", "timeoutMs": 30000}``
``{"op": "binding-result", "callId": 7, "ok": true, "value": ...}``
``{"op": "binding-result", "callId": 7, "ok": false, "error": {...}}``
``{"op": "cancel", "callId": 7}``
``{"op": "shutdown"}``

Driver -> host frames
---------------------
``{"type": "ready", "protocol": 1, ...}``
``{"type": "binding", "callId": 7, "name": "llm_query", "args": {...}}``
``{"type": "result", "id": 1, "stdout": ..., "error": ..., "final": ..., "usage": {...}}``
``{"type": "fatal", "kind": "...", "message": "..."}``

The driver is deliberately dumb about policy: depth, budgets, model routes,
sandboxing and admission control are host-owned. The host decides which binding
names exist (``depth=0`` installs no model bindings at all) and enforces every
hard limit by killing this process.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import inspect
import io
import json
import os
import queue
import sys
import threading
import time
import traceback
from typing import Any, Callable

PROTOCOL_VERSION = 1

MAX_FRAME_BYTES = 64 * 1024 * 1024
"""Hard ceiling on one wire frame; a larger frame is a protocol violation."""


# --------------------------------------------------------------------------- #
# framing
# --------------------------------------------------------------------------- #


class Wire:
    """Length-framed JSON transport over binary stdio."""

    def __init__(self, rfile: io.BufferedReader, wfile: io.BufferedWriter) -> None:
        self._rfile = rfile
        self._wfile = wfile
        self._write_lock = threading.Lock()

    def read(self) -> dict[str, Any] | None:
        header = self._rfile.readline()
        if not header:
            return None
        try:
            length = int(header.strip())
        except ValueError:
            raise ProtocolError(f"invalid frame header: {header[:32]!r}")
        if length < 0 or length > MAX_FRAME_BYTES:
            raise ProtocolError(f"frame length out of range: {length}")
        payload = self._rfile.read(length)
        if payload is None or len(payload) != length:
            return None
        return json.loads(payload.decode("utf-8"))

    def write(self, message: dict[str, Any]) -> None:
        payload = json.dumps(message, ensure_ascii=False, allow_nan=False).encode("utf-8")
        if len(payload) > MAX_FRAME_BYTES:
            payload = json.dumps(
                {
                    "type": "fatal",
                    "kind": "protocol",
                    "message": f"outbound frame exceeds {MAX_FRAME_BYTES} bytes",
                }
            ).encode("utf-8")
        with self._write_lock:
            self._wfile.write(str(len(payload)).encode("ascii") + b"\n")
            self._wfile.write(payload)
            self._wfile.flush()


class ProtocolError(RuntimeError):
    """Raised when the peer violates the framed-JSON contract."""


# --------------------------------------------------------------------------- #
# host binding proxy
# --------------------------------------------------------------------------- #


class BindingFailure(RuntimeError):
    """A host-side binding call failed; message text is model-visible."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


class HostChannel:
    """Dispatches program-visible calls to the host and waits for their replies.

    The channel is safe to use from several program threads at once (that is how
    ``llm_query_many`` fans out), and it keeps a per-call cancellation set so a
    host-side abort resolves a pending call with a truthful error instead of
    wedging the program.
    """

    def __init__(self, wire: Wire) -> None:
        self._wire = wire
        self._lock = threading.Lock()
        self._next_call_id = 1
        self._pending: dict[int, queue.Queue[dict[str, Any]]] = {}
        self._cancelled: set[int] = set()
        self._closed = False

    def call(self, name: str, args: dict[str, Any]) -> Any:
        with self._lock:
            if self._closed:
                raise BindingFailure("cancelled", "kernel is shutting down")
            call_id = self._next_call_id
            self._next_call_id += 1
            box: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
            self._pending[call_id] = box
        self._wire.write({"type": "binding", "callId": call_id, "name": name, "args": args})
        reply = box.get()
        with self._lock:
            self._pending.pop(call_id, None)
            self._cancelled.discard(call_id)
        if reply.get("ok"):
            return reply.get("value")
        error = reply.get("error") or {}
        raise BindingFailure(str(error.get("kind") or "error"), str(error.get("message") or "binding failed"))

    def deliver(self, call_id: int, reply: dict[str, Any]) -> None:
        with self._lock:
            box = self._pending.get(call_id)
        if box is None:
            # Late reply for a cancelled or already-settled call: discard.
            return
        try:
            box.put_nowait(reply)
        except queue.Full:  # pragma: no cover - one reply per call id
            pass

    def cancel(self, call_id: int) -> None:
        with self._lock:
            box = self._pending.get(call_id)
            self._cancelled.add(call_id)
        if box is not None:
            try:
                box.put_nowait({"ok": False, "error": {"kind": "cancelled", "message": "cancelled by host"}})
            except queue.Full:  # pragma: no cover
                pass

    def cancel_all(self) -> None:
        with self._lock:
            call_ids = list(self._pending)
        for call_id in call_ids:
            self.cancel(call_id)

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self.cancel_all()


# --------------------------------------------------------------------------- #
# program-visible context handle
# --------------------------------------------------------------------------- #


class AwaitableList(list):
    """A list that may also be awaited, so both paper- and async-style code work."""

    def __await__(self):  # type: ignore[no-untyped-def]
        if False:  # pragma: no cover - makes this a generator function
            yield None
        return self


class RLMContext:
    """Immutable lazy handle over the externalized user prompt.

    Every read is range-bounded and becomes a *metered* host binding call, which
    is what keeps a 100 MB prompt addressable without ever entering the root
    model's history.
    """

    def __init__(self, channel: HostChannel) -> None:
        self._channel = channel

    # -- metadata --------------------------------------------------------- #
    @property
    def ref(self) -> dict[str, Any]:
        return self._call("context.ref", {})

    @property
    def kind(self) -> str:
        return str(self._call("context.meta", {}).get("kind", "text"))

    @property
    def length(self) -> int:
        return int(self._call("context.meta", {}).get("length", 0))

    @property
    def chars(self) -> int:
        return int(self._call("context.meta", {}).get("chars", 0))

    def __len__(self) -> int:
        return self.length

    def __repr__(self) -> str:
        meta = self._call("context.meta", {})
        return f"<RLMContext kind={meta.get('kind')} length={meta.get('length')} chars={meta.get('chars')}>"

    # -- range reads ------------------------------------------------------ #
    def peek(self, start: int = 0, end: int | None = None) -> str:
        args: dict[str, Any] = {"start": start}
        if end is not None:
            args["end"] = end
        return str(self._call("context.peek", args))

    def slice(self, start: int, end: int) -> str:
        return str(self._call("context.slice", {"start": start, "end": end}))

    def items(self, start: int = 0, end: int | None = None) -> list[Any]:
        args: dict[str, Any] = {"start": start}
        if end is not None:
            args["end"] = end
        return list(self._call("context.items", args))

    def get(self, key: str) -> Any:
        return self._call("context.get", {"key": key})

    def search(self, query: str, max_hits: int = 20, regex: bool = False) -> list[dict[str, Any]]:
        return list(self._call("context.search", {"query": query, "maxHits": max_hits, "regex": regex}))

    def materialize(self, max_bytes: int = 1_000_000) -> Any:
        return self._call("context.materialize", {"maxBytes": max_bytes})

    def _call(self, name: str, args: dict[str, Any]) -> Any:
        return self._channel.call(name, args)

    # -- python protocols ------------------------------------------------- #
    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, slice):
            start = 0 if key.start is None else int(key.start)
            if key.stop is None:
                return self.slice(start, self.length)
            return self.slice(start, int(key.stop))
        if isinstance(key, int):
            meta = self._call("context.meta", {})
            if meta.get("kind") == "text":
                if key < 0:
                    key += int(meta.get("length", 0))
                return self.slice(key, key + 1)
            return self.items(key, key + 1)[0]
        return self.get(str(key))

    def __iter__(self):  # type: ignore[no-untyped-def]
        total = self.length
        step = 1000
        index = 0
        while index < total:
            chunk = self.items(index, min(index + step, total))
            for item in chunk:
                yield item
            index += step


class RLMRecord(dict):
    """A record-shaped context: ``context["content"]`` also works via get/set."""

    def __init__(self, context: RLMContext) -> None:
        super().__init__()
        self._context = context

    def __missing__(self, key: str) -> Any:  # pragma: no cover - convenience path
        return self._context.get(key)


# --------------------------------------------------------------------------- #
# REPL state
# --------------------------------------------------------------------------- #


class OutputCapture(io.TextIOBase):
    """Collects program output with a hard cap so a `while True: print(...)` loop
    cannot by itself exhaust host memory."""

    def __init__(self, limit: int) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._size = 0
        self._limit = limit
        self._truncated = False

    def write(self, text: str) -> int:  # type: ignore[override]
        if self._size < self._limit:
            remaining = self._limit - self._size
            chunk = text[:remaining]
            self._chunks.append(chunk)
            self._size += len(chunk)
        if self._size >= self._limit:
            self._truncated = True
        return len(text)

    def flush(self) -> None:  # pragma: no cover - nothing buffered beyond chunks
        pass

    def value(self) -> tuple[str, bool]:
        return "".join(self._chunks), self._truncated


class ReplState:
    """One persistent namespace plus the bindings installed for this invocation."""

    def __init__(self, wire: Wire, channel: HostChannel, config: dict[str, Any]) -> None:
        self.wire = wire
        self.channel = channel
        self.config = config
        self.namespace: dict[str, Any] = {"__name__": "__rlm_repl__"}
        self.final_var_name = str(config.get("finalVarName") or "Final")
        self.output_limit = int(config.get("maxStdoutChars") or 200_000)
        self._install()

    # -- binding installation --------------------------------------------- #
    def _install_guards(self) -> None:
        """Block the guest's own network and subprocess APIs.

        This is defense in depth, NOT a security boundary: a determined program
        can still reach the syscalls this interpreter exposes. The kernel
        provider reports `network: 'guest-guard'` precisely so no durable event
        over-claims what is enforced here.
        """
        import importlib
        import socket

        def _blocked(*_args: Any, **_kwargs: Any) -> Any:
            raise PermissionError("network and subprocess access are disabled in the RLM kernel")

        for name in ("connect", "connect_ex", "bind", "listen", "sendto"):
            if hasattr(socket.socket, name):
                setattr(socket.socket, name, _blocked)
        socket.create_connection = _blocked  # type: ignore[assignment]
        socket.getaddrinfo = _blocked  # type: ignore[assignment]
        socket.gethostbyname = _blocked  # type: ignore[assignment]

        # The modules must be IMPORTED here, not looked up in sys.modules: a
        # program's later `import subprocess` would otherwise bind a fresh,
        # unpatched module and the guard would be decorative.
        targets = {
            "subprocess": ("Popen", "run", "call", "check_output", "check_call", "getoutput", "getstatusoutput"),
            "multiprocessing": ("Process",),
            "ctypes": ("CDLL", "cdll", "PyDLL"),
            "pty": ("spawn",),
            "asyncio": ("create_subprocess_shell", "create_subprocess_exec"),
        }
        for module_name, names in targets.items():
            try:
                module = importlib.import_module(module_name)
            except ImportError:  # pragma: no cover - platform dependent
                continue
            for name in names:
                if hasattr(module, name):
                    setattr(module, name, _blocked)

        for name in ("fork", "forkpty", "system", "popen", "execv", "execve", "execvp", "posix_spawn", "spawnv", "spawnl"):
            if hasattr(os, name):
                setattr(os, name, _blocked)

    def _install(self) -> None:
        if self.config.get("guardNetwork"):
            self._install_guards()
        bindings = self.config.get("bindings")
        if bindings is None:
            depth = int(self.config.get("depth") or 0)
            bindings = []
            if depth >= 1:
                # `rlm_query` is present at depth 1 by design: the prompt promises
                # a fallback to `llm_query`, and the host enforces it.
                bindings = ["llm_query", "llm_query_many", "rlm_query", "rlm_query_many"]
        allowed = {str(name) for name in bindings}

        context = RLMContext(self.channel)
        self.namespace["context"] = context
        if str(self.config.get("contextKind") or "text") == "record":
            self.namespace["context_record"] = RLMRecord(context)

        # The context handle itself is always available: externalization is what
        # makes depth=0 a meaningful ablation rather than a trivial LLM call.
        self.bindings_installed = ["context"]
        for name in ("llm_query", "llm_query_many", "rlm_query", "rlm_query_many"):
            if name in allowed:
                self.namespace[name] = getattr(self, f"_{name}")
                self.bindings_installed.append(name)

    # -- program-visible model bindings ----------------------------------- #
    def _llm_query(self, prompt: Any, model: str | None = None, max_tokens: int | None = None) -> Any:
        args: dict[str, Any] = {"prompt": _as_text(prompt)}
        if model is not None:
            args["model"] = str(model)
        if max_tokens is not None:
            args["maxTokens"] = int(max_tokens)
        return self.channel.call("llm_query", args)

    def _llm_query_many(
        self,
        prompts: Any,
        model: str | None = None,
        max_tokens: int | None = None,
        max_concurrency: int | None = None,
    ) -> AwaitableList:
        items = [_as_text(p) for p in prompts]
        if not items:
            return AwaitableList()
        worker_cap = int(max_concurrency or self.config.get("maxConcurrency") or 8)
        worker_cap = max(1, min(worker_cap, len(items)))
        results: list[Any] = [None] * len(items)
        errors: list[BaseException] = []
        error_lock = threading.Lock()
        index_queue: queue.Queue[int] = queue.Queue()
        for index in range(len(items)):
            index_queue.put(index)

        def worker() -> None:
            while True:
                try:
                    index = index_queue.get_nowait()
                except queue.Empty:
                    return
                try:
                    results[index] = self._llm_query(items[index], model=model, max_tokens=max_tokens)
                except BaseException as exc:  # noqa: BLE001 - propagated below
                    with error_lock:
                        errors.append(exc)
                    return

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(worker_cap)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise errors[0]
        return AwaitableList(results)

    def _rlm_query(self, context: Any, query: Any, model: str | None = None) -> Any:
        args: dict[str, Any] = {"context": _serialize_context_arg(context), "query": _as_text(query)}
        if model is not None:
            args["model"] = str(model)
        return self.channel.call("rlm_query", args)

    def _rlm_query_many(self, contexts: Any, queries: Any, model: str | None = None) -> AwaitableList:
        pairs = list(zip(list(contexts), list(queries)))
        if not pairs:
            return AwaitableList()
        results: list[Any] = [None] * len(pairs)
        errors: list[BaseException] = []
        error_lock = threading.Lock()
        worker_cap = max(1, min(int(self.config.get("maxConcurrency") or 8), len(pairs)))
        index_queue: queue.Queue[int] = queue.Queue()
        for index in range(len(pairs)):
            index_queue.put(index)

        def worker() -> None:
            while True:
                try:
                    index = index_queue.get_nowait()
                except queue.Empty:
                    return
                context, query = pairs[index]
                try:
                    results[index] = self._rlm_query(context, query, model=model)
                except BaseException as exc:  # noqa: BLE001
                    with error_lock:
                        errors.append(exc)
                    return

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(worker_cap)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise errors[0]
        return AwaitableList(results)

    def _context_call_many(self, name: str, arg_list: list[dict[str, Any]]) -> AwaitableList:
        results: list[Any] = [None] * len(arg_list)
        for index, args in enumerate(arg_list):
            results[index] = self.channel.call(name, args)
        return AwaitableList(results)

    # -- evaluation -------------------------------------------------------- #
    def eval(self, code: str, eval_id: int, timeout_ms: int | None) -> dict[str, Any]:
        capture = OutputCapture(self.output_limit)
        box: dict[str, Any] = {}
        started = time.monotonic()

        def run() -> None:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = capture  # type: ignore[assignment]
            sys.stderr = capture  # type: ignore[assignment]
            try:
                try:
                    compiled = compile(code, "<rlm>", "exec")
                except SyntaxError:
                    # Second chance with top-level `await`, so both paper-style
                    # synchronous code and `await llm_query_many(...)` work. The
                    # bindings are blocking calls, so driving the coroutine on a
                    # private loop is exactly as synchronous as it looks.
                    compiled = compile(code, "<rlm>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
                    pending = eval(compiled, self.namespace, self.namespace)  # noqa: S307 - by design
                    if inspect.isawaitable(pending):
                        asyncio.run(pending)
                else:
                    exec(compiled, self.namespace, self.namespace)  # noqa: S102 - by design
                box["ok"] = True
            except BaseException as exc:  # noqa: BLE001 - reported to the host
                box["ok"] = False
                box["error"] = _error_payload(exc)
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        thread.join(None if not timeout_ms else max(1, int(timeout_ms)) / 1000.0)

        stdout, truncated = capture.value()
        result: dict[str, Any] = {
            "type": "result",
            "id": eval_id,
            "stdout": stdout,
            "stdoutTruncated": truncated,
            "elapsedMs": int((time.monotonic() - started) * 1000),
        }
        if thread.is_alive():
            # A Python thread cannot be pre-empted safely. The host must terminate
            # the kernel process; this frame tells it why.
            result["error"] = {
                "kind": "timeout",
                "message": f"eval exceeded {timeout_ms} ms; kernel must be terminated",
                "traceback": "",
            }
            result["fatal"] = True
            return result
        if box.get("ok"):
            result["error"] = None
        else:
            result["error"] = box.get("error") or {
                "kind": "exception",
                "message": "unknown execution failure",
                "traceback": "",
            }
        final = self._final_snapshot()
        if final is not None:
            result["final"] = final
        return result

    def _final_snapshot(self) -> dict[str, Any] | None:
        if self.final_var_name not in self.namespace:
            return None
        value = self.namespace[self.final_var_name]
        preview, truncated = _preview(value, int(self.config.get("finalPreviewChars") or 2000))
        return {
            "name": self.final_var_name,
            "type": type(value).__name__,
            "preview": preview,
            "truncated": truncated,
            "chars": len(preview) if isinstance(value, str) else None,
        }

    # -- large-value export ------------------------------------------------- #
    def export(self, export_id: int, name: str, path: str, preview_chars: int, max_bytes: int) -> dict[str, Any]:
        """Serialize one REPL variable to a file and report bounded metadata.

        The host, not model code, chose `path`. Nothing but size/hash/preview
        travels back over the control channel, which is what lets a final answer
        exceed the root model's window without becoming one unbounded frame.
        """
        reply: dict[str, Any] = {"type": "export", "id": export_id}
        if name not in self.namespace:
            reply["ok"] = False
            reply["error"] = {"kind": "name", "message": f"the REPL has no variable named {name!r}"}
            return reply
        value = self.namespace[name]
        digest = hashlib.sha256()
        chars = 0
        total = 0
        preview = ""
        try:
            with open(path, "wb") as sink:
                for chunk in _iter_serialized(value, self.output_limit):
                    data = chunk.encode("utf-8")
                    total += len(data)
                    if total > max_bytes:
                        raise ValueError(f"serialized value exceeds the {max_bytes} byte ceiling")
                    digest.update(data)
                    chars += len(chunk)
                    if len(preview) < preview_chars:
                        preview += chunk[: preview_chars - len(preview)]
                    sink.write(data)
        except BaseException as exc:  # noqa: BLE001 - reported to the host
            try:
                os.unlink(path)
            except OSError:
                pass
            reply["ok"] = False
            reply["error"] = _error_payload(exc)
            return reply
        reply.update(
            {
                "ok": True,
                "bytes": total,
                "chars": chars,
                "sha256": digest.hexdigest(),
                "typeName": type(value).__name__,
                "preview": preview,
                "truncated": chars > preview_chars,
            }
        )
        return reply


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)


def _serialize_context_arg(value: Any) -> Any:
    """Normalize an ``rlm_query`` first argument into a lossless JSON shape."""
    if isinstance(value, RLMContext):
        return {"kind": "handle", "ref": value.ref}
    if isinstance(value, str):
        return {"kind": "inline", "text": value}
    if isinstance(value, (list, tuple)):
        return {"kind": "inline", "items": [str(item) for item in value]}
    if isinstance(value, dict):
        return {"kind": "inline", "record": {str(k): str(v) for k, v in value.items()}}
    return {"kind": "inline", "text": _as_text(value)}


def _iter_serialized(value: Any, chunk_chars: int):
    """Yield the serialized form of a REPL value in bounded chunks.

    A `str` is passed through verbatim; anything else is JSON-encoded, falling
    back to `repr` so a hostile or exotic object still produces something the
    host can store and the operator can inspect.
    """
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=repr)
        except BaseException:  # noqa: BLE001 - last-resort serialization
            text = repr(value)
    step = max(1, int(chunk_chars))
    for start in range(0, len(text), step):
        yield text[start : start + step]
    if len(text) == 0:
        yield ""


def _preview(value: Any, limit: int) -> tuple[str, bool]:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = repr(value)
        except BaseException:  # noqa: BLE001 - hostile __repr__
            text = f"<unrepresentable {type(value).__name__}>"
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def _error_payload(exc: BaseException) -> dict[str, Any]:
    """Classify a program failure for the host.

    ``syntax`` and ``name`` are separated because they call for different model
    repairs: a syntax error means "rewrite the program", a NameError usually
    means "that binding does not exist at this depth".
    """
    if isinstance(exc, SyntaxError):
        kind = "syntax"
    elif isinstance(exc, BindingFailure):
        kind = exc.kind
    elif isinstance(exc, NameError):
        kind = "name"
    elif isinstance(exc, TimeoutError):
        kind = "timeout"
    elif isinstance(exc, KeyboardInterrupt):
        kind = "cancelled"
    else:
        kind = "exception"
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    # Only the last frames matter to a model repairing its own code.
    trimmed = "".join(tb[-4:]) if len(tb) > 4 else "".join(tb)
    return {"kind": kind, "message": f"{type(exc).__name__}: {exc}", "traceback": trimmed[:8000]}


# --------------------------------------------------------------------------- #
# main loop
# --------------------------------------------------------------------------- #


def _log(message: str) -> None:
    sys.stderr.write(f"[rlm-kernel] {message}\n")
    sys.stderr.flush()


def main() -> int:
    """Run the persistent REPL.

    Frame reading happens on its own thread because a program that is blocked on
    ``llm_query`` must still be reachable by ``binding-result`` and ``cancel``
    frames. Evaluations are serialized through a single worker so the namespace
    has exactly one writer, matching REPL semantics.
    """
    wire = Wire(sys.stdin.buffer, sys.stdout.buffer)
    channel = HostChannel(wire)
    box: dict[str, ReplState | None] = {"state": None}
    jobs: queue.Queue[dict[str, Any] | None] = queue.Queue()
    draining = threading.Event()
    exit_code = {"value": 0}

    def worker() -> None:
        while True:
            message = jobs.get()
            if message is None:
                return
            state = box["state"]
            if state is None:
                wire.write({"type": "fatal", "kind": "protocol", "message": "eval before init"})
                exit_code["value"] = 2
                return
            if message.get("op") == "export":
                wire.write(
                    state.export(
                        int(message.get("id") or 0),
                        str(message.get("name") or ""),
                        str(message.get("path") or ""),
                        int(message.get("previewChars") or 2000),
                        int(message.get("maxBytes") or 1_000_000_000),
                    )
                )
                continue
            code = message.get("code")
            if not isinstance(code, str):
                wire.write(
                    {
                        "type": "result",
                        "id": message.get("id"),
                        "stdout": "",
                        "stdoutTruncated": False,
                        "elapsedMs": 0,
                        "error": {
                            "kind": "protocol",
                            "message": "eval requires string code",
                            "traceback": "",
                        },
                    }
                )
                continue
            result = state.eval(code, int(message.get("id") or 0), message.get("timeoutMs"))
            wire.write(result)
            if result.get("fatal"):
                _log("fatal eval outcome; exiting so the host can reap the process tree")
                # os._exit is deliberate: the runaway program thread cannot be
                # pre-empted, and the host owns process-tree termination.
                sys.stderr.flush()
                os._exit(3)

    eval_thread = threading.Thread(target=worker, name="rlm-eval", daemon=True)
    eval_thread.start()

    wire.write(
        {
            "type": "ready",
            "protocol": PROTOCOL_VERSION,
            "python": sys.version.split()[0],
            "pid": os.getpid(),
        }
    )

    while True:
        try:
            message = wire.read()
        except ProtocolError as exc:
            wire.write({"type": "fatal", "kind": "protocol", "message": str(exc)})
            exit_code["value"] = 2
            break
        except json.JSONDecodeError as exc:
            wire.write({"type": "fatal", "kind": "protocol", "message": f"invalid JSON frame: {exc}"})
            exit_code["value"] = 2
            break
        if message is None:
            break

        op = message.get("op")
        if op == "init":
            config = message.get("config") or {}
            state = ReplState(wire, channel, config)
            box["state"] = state
            wire.write(
                {
                    "type": "ready",
                    "protocol": PROTOCOL_VERSION,
                    "phase": "initialized",
                    "bindings": state.bindings_installed,
                    "finalVarName": state.final_var_name,
                }
            )
        elif op == "eval":
            jobs.put(message)
        elif op == "export":
            if box["state"] is None:
                wire.write({"type": "fatal", "kind": "protocol", "message": "export before init"})
                exit_code["value"] = 2
                break
            jobs.put(message)
        elif op == "binding-result":
            channel.deliver(int(message.get("callId") or 0), message)
        elif op == "cancel":
            channel.cancel(int(message.get("callId") or 0))
        elif op == "shutdown":
            draining.set()
            channel.close()
            jobs.put(None)
            wire.write({"type": "bye", "protocol": PROTOCOL_VERSION})
            exit_code["value"] = 0
            break
        else:
            wire.write({"type": "fatal", "kind": "protocol", "message": f"unknown op: {op!r}"})
            exit_code["value"] = 2
            break

    channel.close()
    jobs.put(None)
    # Give an in-flight evaluation a bounded grace period to settle its frame;
    # never let the driver outlive the host's view of a closed stdin.
    eval_thread.join(timeout=2.0)
    return exit_code["value"]


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:  # pragma: no cover - host-initiated
        sys.exit(130)
