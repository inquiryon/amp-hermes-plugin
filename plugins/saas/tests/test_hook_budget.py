"""A3: governed calls return their own block before a Hermes host limit fires (L11/L12, M1)."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import hermes as plugin_mod
from hermes import host_limits
from hermes.amp_client import AmpClient, AmpClientError, request_deadline
from hitl_helpers import FakeClock, complete, make_plugin, pending, quiet


def _run(plugin, clock, *, hook=None, tool=None):
    with patch.object(plugin_mod, "hook_timeout_seconds", lambda: hook), \
         patch.object(plugin_mod, "tool_deadline_seconds", lambda: tool), \
         quiet(plugin), clock.installed():
        return plugin.pre_tool_call(tool_name="terminal", args={"command": "ls"}, session_id="s1")


class HookBudgetTests(unittest.TestCase):
    def _pending_forever(self, **config):
        plugin = make_plugin(hitl_timeout_minutes=10, hitl_poll_interval_seconds=1, **config)
        plugin._client.request_hitl = Mock(return_value=pending("wi-1"))
        plugin._client.get_hitl_decision = Mock(return_value=pending("wi-1"))
        return plugin

    def test_returns_own_block_message_before_30s_hermes_timeout(self):
        plugin, clock = self._pending_forever(), FakeClock()
        result = _run(plugin, clock, hook=30.0, tool=420.0)
        self.assertEqual(result["action"], "block")
        self.assertIn("Hermes limit (plugins.hook_callback_timeout)", result["message"])
        self.assertIn("30s", result["message"])
        self.assertLessEqual(clock.elapsed, 30 - 7.5 + 0.01)  # margin = min(15, 25% of 30)
        self.assertGreater(plugin._client.get_hitl_decision.call_count, 10)

    def test_tool_deadline_binds_when_shorter_than_hook(self):
        plugin, clock = self._pending_forever(), FakeClock()
        result = _run(plugin, clock, hook=600.0, tool=420.0)
        self.assertIn("timeouts.tools.sequential_call/concurrent_batch", result["message"])
        self.assertIn("420s", result["message"])
        self.assertLessEqual(clock.elapsed, 420 - 15 + 0.01)
        self.assertGreaterEqual(clock.elapsed, 420 - 15 - 1)

    def test_unbounded_when_hermes_has_no_timeout(self):
        plugin, clock = self._pending_forever(), FakeClock()
        result = _run(plugin, clock, hook=None, tool=None)
        self.assertIn("timed out waiting for AMP HITL approval", result["message"])
        self.assertGreaterEqual(clock.elapsed, 600 - 15 - 1)
        self.assertLessEqual(clock.elapsed, 600)

    def test_budget_exhaustion_blocks_even_when_fail_open(self):
        plugin, clock = self._pending_forever(fail_closed=False), FakeClock()
        result = _run(plugin, clock, hook=30.0)
        self.assertEqual(result["action"], "block")

    def test_small_hook_budget_still_sends_the_first_request(self):
        # M1: a 10 s hook budget must not raise before the first request.
        plugin = make_plugin(hitl_poll_interval_seconds=1)
        plugin._client.request_hitl = Mock(return_value=pending("wi-1"))
        plugin._client.get_hitl_decision = Mock(side_effect=[pending("wi-1"), complete("approve", "wi-1")])
        self.assertIsNone(_run(plugin, FakeClock(), hook=10.0))
        plugin._client.request_hitl.assert_called_once()

    def test_approval_inside_budget_allows(self):
        plugin = make_plugin(hitl_poll_interval_seconds=3)
        plugin._client.request_hitl = Mock(return_value=pending("wi-1"))
        plugin._client.get_hitl_decision = Mock(side_effect=[pending("wi-1")] * 5 + [complete("approve", "wi-1")])
        self.assertIsNone(_run(plugin, FakeClock(), hook=600.0, tool=420.0))

    def test_request_failure_after_budget_blocks_even_when_fail_open(self):
        plugin = make_plugin(fail_closed=False)
        clock = FakeClock()

        def slow_failure(*_args, **_kwargs):
            clock.now += 40  # the request itself ate the whole budget
            raise AmpClientError("timed out")

        plugin._client.request_hitl = Mock(side_effect=slow_failure)
        result = _run(plugin, clock, hook=30.0)
        self.assertEqual(result["action"], "block")

    def test_http_timeout_never_exceeds_remaining_budget(self):
        client = AmpClient(SimpleNamespace(backend_url="http://amp.invalid", api_key="k"))
        seen = {}

        def spy(_req, timeout):
            seen["timeout"] = timeout
            raise AmpClientError("stop")

        clock = FakeClock()
        with clock.installed(), patch("hermes.amp_client.urlopen", side_effect=spy):
            with request_deadline(clock.now + 5):
                with self.assertRaises(AmpClientError):
                    client.get_hitl_decision("inst-1")
        self.assertLessEqual(seen["timeout"], 5)

    def test_http_timeout_floors_at_one_second_after_soft_deadline(self):
        client = AmpClient(SimpleNamespace(backend_url="http://amp.invalid", api_key="k"))
        seen = {}

        def spy(_req, timeout):
            seen["timeout"] = timeout
            return _Response()

        clock = FakeClock()
        with clock.installed(), patch("hermes.amp_client.urlopen", side_effect=spy):
            with request_deadline(clock.now - 10):
                client.get_hitl_decision("inst-1")
        self.assertEqual(seen["timeout"], 1.0)

    def test_hard_deadline_refuses_requests_after_expiry(self):
        client = AmpClient(SimpleNamespace(backend_url="http://amp.invalid", api_key="k"))
        clock = FakeClock()
        with clock.installed(), patch("hermes.amp_client.urlopen") as urlopen:
            with request_deadline(clock.now - 1, hard=True):
                with self.assertRaises(AmpClientError):
                    client.get_hitl_decision("inst-1")
        urlopen.assert_not_called()

    def test_register_logs_error_when_hook_budget_shorter_than_hitl_timeout(self):
        config = SimpleNamespace(hitl_timeout_minutes=10)
        with patch.object(host_limits, "hook_timeout_seconds", lambda: 30.0), \
             patch.object(host_limits, "tool_deadline_seconds", lambda: 420.0), \
             self.assertLogs("hermes", level="ERROR") as logs:
            plugin_mod._log_short_host_limits(config)
        text = "\n".join(logs.output)
        self.assertIn("plugins.hook_callback_timeout", text)
        self.assertIn("timeouts.tools", text)

    def test_register_quiet_when_limits_cover_hitl_timeout(self):
        config = SimpleNamespace(hitl_timeout_minutes=5)
        with patch.object(host_limits, "hook_timeout_seconds", lambda: 600.0), \
             patch.object(host_limits, "tool_deadline_seconds", lambda: 660.0), \
             patch.object(plugin_mod.logger, "error") as error:
            plugin_mod._log_short_host_limits(config)
        error.assert_not_called()


class HostLimitResolutionTests(unittest.TestCase):
    def test_margin_is_quarter_of_small_limits_and_capped_at_15s(self):
        self.assertEqual(host_limits.compute_budget(0, 600, hook_timeout=10, tool_deadline=None).deadline, 7.5)
        self.assertEqual(host_limits.compute_budget(0, 600, hook_timeout=600, tool_deadline=420).deadline, 405)
        budget = host_limits.compute_budget(0, 300, hook_timeout=600, tool_deadline=None)
        self.assertFalse(budget.host_limited)

    def test_zero_or_failing_resolvers_mean_unbounded(self):
        with patch("hermes_cli.plugins._resolve_hook_callback_timeout", return_value=0):
            self.assertIsNone(host_limits.hook_timeout_seconds())
        with patch("hermes_cli.plugins._resolve_hook_callback_timeout", side_effect=RuntimeError):
            self.assertIsNone(host_limits.hook_timeout_seconds())

    def test_real_hermes_resolvers_are_found(self):
        # Pins the private Hermes APIs this relies on (R8): defaults are 30 s hook / 420 s tool.
        with patch("hermes_cli.plugins._resolve_hook_callback_timeout", return_value=30.0):
            self.assertEqual(host_limits.hook_timeout_seconds(), 30.0)
        with patch("agent.tool_executor._resolve_sequential_tool_timeout", return_value=420.0), \
             patch("agent.tool_executor._resolve_concurrent_tool_timeout", return_value=300.0):
            self.assertEqual(host_limits.tool_deadline_seconds(), 300.0)


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return b'{"status": "pending"}'


if __name__ == "__main__":
    unittest.main()
