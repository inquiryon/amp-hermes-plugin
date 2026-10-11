"""A6 / contract K3: opt-in action_id correlation of the HITL request and decision poll."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from hermes.amp_client import AmpClient
from hermes.config import AmpConfig
from hermes.policy import NormalizedAction
from hitl_helpers import FakeClock, complete, make_plugin, pending, quiet


def _governed(plugin, **kwargs):
    with quiet(plugin), FakeClock().installed():
        return plugin.pre_tool_call(tool_name="terminal", args={"command": "ls"}, session_id="s1", **kwargs)


class ActionIdTests(unittest.TestCase):
    def test_config_defaults_off(self) -> None:
        cfg = AmpConfig("u", "k", "o", "a", "n", 10, 3, True, True, True, "observe", False, True)
        self.assertFalse(cfg.action_id_correlation)
        self.assertEqual(cfg.action_id_watchdog_seconds, 60)

    def test_flag_off_sends_no_action_id(self) -> None:
        plugin = make_plugin()
        plugin._client.request_hitl = Mock(return_value=pending("wi-1"))
        plugin._client.get_hitl_decision = Mock(return_value=complete("approve", "wi-1"))
        self.assertIsNone(_governed(plugin, tool_call_id="call_abc"))
        self.assertNotIn("action_id", plugin._client.request_hitl.call_args.kwargs)
        self.assertNotIn("action_id", plugin._client.get_hitl_decision.call_args.kwargs)

    def test_request_and_poll_carry_tool_call_id(self) -> None:
        plugin = make_plugin(action_id_correlation=True)
        plugin._client.request_hitl = Mock(return_value=pending("wi-1", action_id="call_abc"))
        plugin._client.get_hitl_decision = Mock(return_value=complete("approve", "wi-1", action_id="call_abc"))
        self.assertIsNone(_governed(plugin, tool_call_id="call_abc"))
        self.assertEqual(plugin._client.request_hitl.call_args.kwargs["action_id"], "call_abc")
        self.assertEqual(plugin._client.get_hitl_decision.call_args.kwargs["action_id"], "call_abc")

    def test_uuid_fallback_when_hermes_omits_tool_call_id(self) -> None:
        plugin = make_plugin(action_id_correlation=True)
        plugin._client.request_hitl = Mock(return_value={"status": "no-hitl"})
        self.assertIsNone(_governed(plugin))
        self.assertRegex(plugin._client.request_hitl.call_args.kwargs["action_id"], r"^[0-9a-f]{32}$")

    def test_stale_unscoped_approval_is_not_accepted(self) -> None:
        plugin = make_plugin(action_id_correlation=True)
        plugin._client.request_hitl = Mock(return_value=pending("wi-NEW"))
        plugin._client.get_hitl_decision = Mock(return_value={
            "status": "complete", "resolution": "approve", "workitem_id": "wi-NEW",
        })  # no action_id echo: backend cannot correlate
        result = _governed(plugin, tool_call_id="call_abc")
        self.assertEqual(result["action"], "block")
        self.assertIn("echo", result["message"])

    def test_missing_workitem_watchdog_blocks_after_60s_not_600s(self) -> None:
        plugin = make_plugin(action_id_correlation=True, hitl_timeout_minutes=10)
        plugin._client.request_hitl = Mock(return_value={"status": "waiting-for-response"})
        plugin._client.get_hitl_decision = Mock(return_value={"status": "pending", "action_id": "call_abc"})
        clock = FakeClock()
        with quiet(plugin), clock.installed():
            result = plugin.pre_tool_call(
                tool_name="terminal", args={"command": "ls"}, session_id="s1", tool_call_id="call_abc"
            )
        self.assertIn("action_id correlation", result["message"])
        self.assertLessEqual(clock.elapsed, 61)

    def test_client_puts_action_id_in_body_and_query(self) -> None:
        client = AmpClient(SimpleNamespace(backend_url="http://amp.invalid", api_key="k", org_id="o", agent_name="a"))
        seen = []
        client._request = Mock(side_effect=lambda *a, **k: seen.append((a, k)) or {})
        client.request_hitl("inst-1", NormalizedAction("terminal", "exec", "exec", {}), action_id="call_abc")
        client.get_hitl_decision("inst-1", action_id="call_abc")
        client.get_hitl_decision("inst-1")
        self.assertEqual(seen[0][0][2]["action_id"], "call_abc")
        self.assertEqual(seen[1][1]["query"], {"caller_id": "inst-1", "action_id": "call_abc"})
        self.assertEqual(seen[2][1]["query"], {"caller_id": "inst-1"})


if __name__ == "__main__":
    unittest.main()
