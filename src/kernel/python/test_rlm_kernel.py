"""Unit tests for the persistent RLM kernel driver.

Run with::

    python3 -m unittest discover -s src/kernel/python -p 'test_*.py' -v

These tests exercise the driver in-process (no subprocess) so a failure points at
a specific REPL behaviour rather than at process plumbing; the Node suite covers
the provider, framing and process lifecycle end to end.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

DRIVER = Path(__file__).with_name("rlm_kernel.py")
_spec = spec_from_file_location("rlm_kernel", DRIVER)
assert _spec is not None and _spec.loader is not None
rlm_kernel = module_from_spec(_spec)
_spec.loader.exec_module(rlm_kernel)


class FakeChannel:
    """Host stand-in that answers binding calls without a real wire."""

    def __init__(self, responder=None):
        self.calls = []
        self.responder = responder or (lambda name, args: f"answer({json.dumps(args)[:40]})")
        self.cancelled = set()

    def call(self, name, args):
        self.calls.append((name, args))
        if name == "context.meta":
            return {"kind": "text", "length": 26, "chars": 26, "bytes": 26, "sha256": "abc"}
        if name == "context.slice":
            text = "abcdefghijklmnopqrstuvwxyz"
            return text[int(args.get("start", 0)) : int(args.get("end", len(text)))]
        if name == "context.peek":
            text = "abcdefghijklmnopqrstuvwxyz"
            return text[int(args.get("start", 0)) : int(args.get("end", 10))]
        if name == "context.items":
            return ["a", "b"]
        if name == "context.get":
            return {"key": "value"}
        if name == "context.search":
            return [{"start": 0, "end": 3, "text": "abc"}]
        if name == "context.materialize":
            return "abcdefghijklmnopqrstuvwxyz"
        if name == "context.ref":
            return {"id": "ctx"}
        return self.responder(name, args)

    def cancel(self, call_id):
        self.cancelled.add(call_id)

    def cancel_all(self):
        pass

    def close(self):
        pass


def make_state(config=None):
    wire = rlm_kernel.Wire(io.BytesIO(b""), io.BytesIO())
    channel = FakeChannel()
    state = rlm_kernel.ReplState(wire, channel, config or {"depth": 2, "finalVarName": "Final"})
    return state, channel


class WireTests(unittest.TestCase):
    def test_round_trips_a_frame(self):
        out = io.BytesIO()
        wire = rlm_kernel.Wire(io.BytesIO(b""), out)
        wire.write({"type": "ready", "protocol": 1})
        payload = out.getvalue()
        header, _, body = payload.partition(b"\n")
        self.assertEqual(int(header), len(body))
        reader = rlm_kernel.Wire(io.BytesIO(payload), io.BytesIO())
        self.assertEqual(reader.read(), {"type": "ready", "protocol": 1})

    def test_preserves_newlines_inside_a_payload(self):
        out = io.BytesIO()
        wire = rlm_kernel.Wire(io.BytesIO(b""), out)
        wire.write({"stdout": "a\nb\nc"})
        reader = rlm_kernel.Wire(io.BytesIO(out.getvalue()), io.BytesIO())
        self.assertEqual(reader.read()["stdout"], "a\nb\nc")

    def test_rejects_a_bad_header(self):
        reader = rlm_kernel.Wire(io.BytesIO(b"not-a-number\n"), io.BytesIO())
        with self.assertRaises(rlm_kernel.ProtocolError):
            reader.read()

    def test_rejects_an_oversized_frame(self):
        reader = rlm_kernel.Wire(io.BytesIO(b"99999999999\n{}"), io.BytesIO())
        with self.assertRaises(rlm_kernel.ProtocolError):
            reader.read()

    def test_returns_none_at_end_of_stream(self):
        reader = rlm_kernel.Wire(io.BytesIO(b""), io.BytesIO())
        self.assertIsNone(reader.read())


class HostChannelTests(unittest.TestCase):
    def test_delivers_a_reply_to_the_waiting_caller(self):
        out = io.BytesIO()
        wire = rlm_kernel.Wire(io.BytesIO(b""), out)
        channel = rlm_kernel.HostChannel(wire)
        result = {}

        def caller():
            result["value"] = channel.call("llm_query", {"prompt": "hi"})

        thread = threading.Thread(target=caller)
        thread.start()
        # Read the outbound binding frame, then answer it as a host would.
        header, _, body = out.getvalue().partition(b"\n")
        while not header:
            header, _, body = out.getvalue().partition(b"\n")
        frame = json.loads(body)
        self.assertEqual(frame["type"], "binding")
        self.assertEqual(frame["name"], "llm_query")
        channel.deliver(frame["callId"], {"ok": True, "value": "leaf"})
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["value"], "leaf")

    def test_surfaces_a_host_error_as_a_binding_failure(self):
        out = io.BytesIO()
        channel = rlm_kernel.HostChannel(rlm_kernel.Wire(io.BytesIO(b""), out))
        errors = []

        def caller():
            try:
                channel.call("llm_query", {"prompt": "hi"})
            except rlm_kernel.BindingFailure as exc:
                errors.append(exc.kind)

        thread = threading.Thread(target=caller)
        thread.start()
        frame = None
        while frame is None:
            header, _, body = out.getvalue().partition(b"\n")
            if header:
                frame = json.loads(body)
        channel.deliver(frame["callId"], {"ok": False, "error": {"kind": "budget", "message": "no room"}})
        thread.join(timeout=5)
        self.assertEqual(errors, ["budget"])

    def test_discards_a_reply_for_an_unknown_call(self):
        channel = rlm_kernel.HostChannel(rlm_kernel.Wire(io.BytesIO(b""), io.BytesIO()))
        channel.deliver(999, {"ok": True, "value": "late"})  # must not raise


class ContextHandleTests(unittest.TestCase):
    def setUp(self):
        self.state, self.channel = make_state()

    def test_reports_metadata_and_length(self):
        context = self.state.namespace["context"]
        self.assertEqual(context.kind, "text")
        self.assertEqual(len(context), 26)
        self.assertIn("RLMContext", repr(context))

    def test_reads_ranges_and_items(self):
        context = self.state.namespace["context"]
        self.assertEqual(context[0:3], "abc")
        self.assertEqual(context.peek(0, 3), "abc")
        self.assertEqual(context.items(0, 2), ["a", "b"])
        self.assertEqual(context.get("key"), {"key": "value"})
        self.assertEqual(context.search("abc")[0]["text"], "abc")

    def test_supports_python_iteration_protocols(self):
        context = self.state.namespace["context"]
        self.assertEqual(list(context)[:2], ["a", "b"])


class ReplStateTests(unittest.TestCase):
    def test_keeps_one_namespace_across_evaluations(self):
        state, _ = make_state()
        state.eval("counter = 1", 1, None)
        result = state.eval("counter += 41\nprint(counter)", 2, None)
        self.assertEqual(result["stdout"], "42\n")
        self.assertIsNone(result["error"])

    def test_reports_a_final_variable_without_serializing_it(self):
        state, _ = make_state()
        result = state.eval('Final = "x" * 10', 1, None)
        self.assertEqual(result["final"]["name"], "Final")
        self.assertEqual(result["final"]["chars"], 10)

    def test_classifies_syntax_and_name_errors(self):
        state, _ = make_state()
        self.assertEqual(state.eval("x = 1 +", 1, None)["error"]["kind"], "syntax")
        self.assertEqual(state.eval("print(nope)", 2, None)["error"]["kind"], "name")
        self.assertEqual(state.eval("raise ValueError('boom')", 3, None)["error"]["kind"], "exception")

    def test_installs_bindings_by_depth(self):
        deep, _ = make_state({"depth": 2})
        self.assertIn("rlm_query", deep.namespace)
        self.assertIn("llm_query", deep.namespace)
        shallow, _ = make_state({"depth": 0})
        self.assertNotIn("llm_query", shallow.namespace)
        self.assertNotIn("rlm_query", shallow.namespace)
        # The context handle exists even with no model bindings: depth 0 is an
        # ablation of sub-calls, not of externalization.
        self.assertIn("context", shallow.namespace)

    def test_honours_an_explicit_binding_list(self):
        state, _ = make_state({"depth": 2, "bindings": ["llm_query"]})
        self.assertIn("llm_query", state.namespace)
        self.assertNotIn("rlm_query", state.namespace)

    def test_truncates_program_output(self):
        state, _ = make_state({"depth": 0, "maxStdoutChars": 50})
        result = state.eval("print('x' * 500)", 1, None)
        self.assertTrue(result["stdoutTruncated"])
        self.assertLessEqual(len(result["stdout"]), 51)

    def test_supports_top_level_await(self):
        state, _ = make_state()
        result = state.eval('answers = await llm_query_many(["a", "b"])\nprint(len(answers))', 1, None)
        self.assertEqual(result["error"], None)
        self.assertEqual(result["stdout"].strip(), "2")

    def test_fans_out_llm_query_many_concurrently(self):
        state, channel = make_state()
        peak = {"value": 0}
        active = {"value": 0}
        lock = threading.Lock()

        def responder(name, args):
            with lock:
                active["value"] += 1
                peak["value"] = max(peak["value"], active["value"])
            import time

            time.sleep(0.05)
            with lock:
                active["value"] -= 1
            return "leaf"

        channel.responder = responder
        result = state.eval('answers = llm_query_many(["a", "b", "c", "d"])', 1, None)
        self.assertEqual(result["error"], None)
        self.assertGreaterEqual(peak["value"], 2)

    def test_propagates_a_binding_failure_into_the_program(self):
        state, channel = make_state()

        def responder(name, args):
            raise rlm_kernel.BindingFailure("budget", "no room")

        channel.responder = responder
        result = state.eval('llm_query("x")', 1, None)
        self.assertEqual(result["error"]["kind"], "budget")

    def test_exports_a_variable_to_a_file_with_bounded_metadata(self):
        state, _ = make_state()
        state.eval('Final = "y" * 5000', 1, None)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "final.out")
            reply = state.export(7, "Final", path, 16, 1_000_000)
            self.assertTrue(reply["ok"])
            self.assertEqual(reply["chars"], 5000)
            self.assertEqual(reply["preview"], "y" * 16)
            self.assertTrue(reply["truncated"])
            with open(path, "r", encoding="utf-8") as handle:
                self.assertEqual(len(handle.read()), 5000)

    def test_refuses_an_export_above_the_byte_ceiling_and_removes_the_file(self):
        state, _ = make_state()
        state.eval('Final = "y" * 5000', 1, None)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "too-big.out")
            reply = state.export(7, "Final", path, 8, 100)
            self.assertFalse(reply["ok"])
            self.assertFalse(os.path.exists(path))

    def test_reports_a_missing_export_variable(self):
        state, _ = make_state()
        with tempfile.TemporaryDirectory() as directory:
            reply = state.export(1, "nope", os.path.join(directory, "x"), 8, 100)
            self.assertFalse(reply["ok"])
            self.assertEqual(reply["error"]["kind"], "name")

    def test_serializes_a_non_string_variable_as_json(self):
        state, _ = make_state()
        state.eval("Final = {'a': 1, 'b': [2, 3]}", 1, None)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "record.out")
            reply = state.export(1, "Final", path, 100, 100_000)
            self.assertTrue(reply["ok"])
            with open(path, "r", encoding="utf-8") as handle:
                self.assertEqual(json.loads(handle.read()), {"a": 1, "b": [2, 3]})

    def test_guest_guards_block_network_and_subprocess(self):
        state, _ = make_state({"depth": 0, "guardNetwork": True})
        self.assertEqual(state.eval("import socket\nsocket.create_connection(('x', 1))", 1, None)["error"]["kind"], "exception")
        self.assertEqual(state.eval("import subprocess\nsubprocess.run(['ls'])", 2, None)["error"]["kind"], "exception")
        self.assertEqual(state.eval("import os\nos.system('ls')", 3, None)["error"]["kind"], "exception")

    def test_guards_can_be_disabled(self):
        state, _ = make_state({"depth": 0, "guardNetwork": False})
        result = state.eval("import socket\nprint(callable(socket.create_connection))", 1, None)
        self.assertIsNone(result["error"])
        self.assertEqual(result["stdout"].strip(), "True")

    def test_reports_a_timeout_and_asks_to_be_terminated(self):
        state, _ = make_state()
        result = state.eval("import time\ntime.sleep(2)", 1, 50)
        self.assertEqual(result["error"]["kind"], "timeout")
        self.assertTrue(result["fatal"])

    def test_a_hanging_eval_does_not_block_the_host_channel(self):
        # The eval runs on its own thread; the reader thread must stay free to
        # deliver binding replies (the bug this structure exists to prevent).
        state, _ = make_state()
        result = state.eval('x = llm_query("hi")\nprint(x)', 1, 5000)
        self.assertEqual(result["error"], None)
        self.assertIn("answer(", result["stdout"])


class OutputCaptureTests(unittest.TestCase):
    def test_caps_captured_output(self):
        capture = rlm_kernel.OutputCapture(10)
        capture.write("abcdefghijklmno")
        value, truncated = capture.value()
        self.assertEqual(len(value), 10)
        self.assertTrue(truncated)

    def test_reports_no_truncation_below_the_cap(self):
        capture = rlm_kernel.OutputCapture(100)
        capture.write("short")
        value, truncated = capture.value()
        self.assertEqual(value, "short")
        self.assertFalse(truncated)


if __name__ == "__main__":
    unittest.main()
