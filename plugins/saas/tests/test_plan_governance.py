"""Tests for Phase 3A: proposed execution plan governance.

Covers:
  - Plan validation (required fields only, no AMP call on failure)
  - Normalization into flat AMP policy signals + opaque payload passthrough
  - Decision outcomes (auto-approve, HITL approve/reject/timeout, no_policy, AMP unavailable)
  - approved_budget_usd initialization (and non-initialization) on ExecutionContext
  - Regression: existing Phase 1-2B behavior unaffected
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parent.parent
ROOT_PARENT = ROOT.parent
HERMES_AGENT_ROOT = Path.home() / ".hermes" / "hermes-agent"

if str(ROOT_PARENT) not in sys.path:
    sys.path.insert(0, str(ROOT_PARENT))
if HERMES_AGENT_ROOT.exists() and str(HERMES_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(HERMES_AGENT_ROOT))

from hermes import AmpGovernancePlugin, evaluate_proposed_plan
from hermes.execution_context import ExecutionContext


def _make_plugin() -> AmpGovernancePlugin:
    plugin = AmpGovernancePlugin()
    plugin._config = SimpleNamespace(
        is_configured=True,
        notifications_enabled=True,
        llm_governance_enabled=True,
        llm_governance_mode="enforce",
        llm_governance_fail_closed=False,
        llm_governance_include_subagents=True,
        fail_closed=True,
        hitl_timeout_minutes=1,
        hitl_poll_interval_seconds=0,
        username="tester",
        agent_name="hermes-test",
        org_id="O-test",
    )
    plugin._client = Mock()
    plugin._notify_user = Mock()
    return plugin


def _valid_plan(**overrides) -> dict:
    plan = {
        "plan_type": "research",
        "summary": "Research five configured topics",
        "projected_cost_usd": 4.75,
        "projected_cost_status": "estimated",
        "estimated_llm_calls": 18,
        "estimated_tool_calls": 25,
        "estimated_duration_minutes": 20,
        "work_units_total": 5,
        "payload": {"topics": ["AI agent governance"], "research_depth": "standard"},
    }
    plan.update(overrides)
    return plan


def _allow_response():
    return {"status": "no-hitl"}


def _no_policy_response():
    return {"status": "no_policy"}


def _hitl_response(workitem_id="w-1"):
    return {"status": "pending", "workitem_id": workitem_id}


def _decision(resolution="approved", info=""):
    # AMP always echoes the workitem_id; the plugin only accepts a decision for the
    # workitem its request is bound to (stale-decision guard).
    return {"status": "complete", "resolution": resolution, "information": info, "workitem_id": "w-1"}


# ---------------------------------------------------------------------------
# 1. Validation (no AMP call on failure)
# ---------------------------------------------------------------------------

class ValidationTests(unittest.TestCase):
    def test_missing_plan_type_returns_error_without_calling_amp(self):
        plugin = _make_plugin()
        result = plugin.evaluate_proposed_plan("s", _valid_plan(plan_type=""))
        self.assertEqual(result["status"], "error")
        self.assertIn("plan_type", result["reason"])
        plugin._client.request_plan_approval.assert_not_called()

    def test_missing_projected_cost_returns_error_without_calling_amp(self):
        plugin = _make_plugin()
        plan = _valid_plan()
        del plan["projected_cost_usd"]
        result = plugin.evaluate_proposed_plan("s", plan)
        self.assertEqual(result["status"], "error")
        plugin._client.request_plan_approval.assert_not_called()

    def test_negative_projected_cost_returns_error(self):
        plugin = _make_plugin()
        result = plugin.evaluate_proposed_plan("s", _valid_plan(projected_cost_usd=-1.0))
        self.assertEqual(result["status"], "error")
        plugin._client.request_plan_approval.assert_not_called()

    def test_non_numeric_projected_cost_returns_error(self):
        plugin = _make_plugin()
        result = plugin.evaluate_proposed_plan("s", _valid_plan(projected_cost_usd="lots"))
        self.assertEqual(result["status"], "error")
        plugin._client.request_plan_approval.assert_not_called()

    def test_not_configured_returns_error(self):
        plugin = _make_plugin()
        plugin._config.is_configured = False
        result = plugin.evaluate_proposed_plan("s", _valid_plan())
        self.assertEqual(result["status"], "error")
        plugin._client.request_plan_approval.assert_not_called()


# ---------------------------------------------------------------------------
# 2. Decision outcomes
# ---------------------------------------------------------------------------

class DecisionOutcomeTests(unittest.TestCase):
    def test_auto_approved_sets_budget_and_creates_context(self):
        plugin = _make_plugin()
        plugin._client.request_plan_approval = Mock(return_value=_allow_response())
        with patch.object(plugin, "_ensure_instance", return_value="inst-1"), \
             patch.object(plugin, "_safe_log"):
            result = plugin.evaluate_proposed_plan("s", _valid_plan(projected_cost_usd=4.75))

        self.assertEqual(result["status"], "approved")
        self.assertAlmostEqual(result["approved_budget_usd"], 4.75, places=5)
        exec_ctx = plugin._exec_contexts.get("s")
        self.assertIsNotNone(exec_ctx)
        self.assertAlmostEqual(exec_ctx.approved_budget_usd, 4.75, places=5)

    def test_auto_approved_notifies_user_no_review_was_needed(self):
        plugin = _make_plugin()
        plugin._client.request_plan_approval = Mock(return_value=_allow_response())
        with patch.object(plugin, "_ensure_instance", return_value="inst-1"), \
             patch.object(plugin, "_safe_log"):
            plugin.evaluate_proposed_plan("s", _valid_plan(projected_cost_usd=4.75))

        plugin._notify_user.assert_called_once()
        message = plugin._notify_user.call_args.args[0]
        self.assertIn("no human review", message.lower())
        self.assertIn("researching now", message.lower())
        self.assertIn("$4.75", message)

    def test_hitl_approved_sets_budget(self):
        plugin = _make_plugin()
        plugin._client.request_plan_approval = Mock(return_value=_hitl_response())
        plugin._client.get_hitl_decision = Mock(return_value=_decision("approved"))
        with patch.object(plugin, "_ensure_instance", return_value="inst-1"), \
             patch.object(plugin, "_safe_log"), patch("time.sleep", return_value=None):
            result = plugin.evaluate_proposed_plan("s", _valid_plan(projected_cost_usd=4.75))

        self.assertEqual(result["status"], "approved")
        self.assertAlmostEqual(result["approved_budget_usd"], 4.75, places=5)
        self.assertAlmostEqual(plugin._exec_contexts.get("s").approved_budget_usd, 4.75, places=5)

    def test_hitl_rejected_does_not_set_budget(self):
        plugin = _make_plugin()
        plugin._client.request_plan_approval = Mock(return_value=_hitl_response())
        plugin._client.get_hitl_decision = Mock(return_value=_decision("rejected", "too expensive"))
        with patch.object(plugin, "_ensure_instance", return_value="inst-1"), \
             patch.object(plugin, "_safe_log"), patch("time.sleep", return_value=None):
            result = plugin.evaluate_proposed_plan("s", _valid_plan())

        self.assertEqual(result["status"], "rejected")
        self.assertIsNone(result["approved_budget_usd"])
        self.assertIsNone(plugin._exec_contexts.get("s"))

    def test_hitl_timeout_does_not_set_budget(self):
        plugin = _make_plugin()
        plugin._config.hitl_timeout_minutes = 0
        plugin._client.request_plan_approval = Mock(return_value=_hitl_response())
        plugin._client.get_hitl_decision = Mock(return_value={"status": "pending"})
        with patch.object(plugin, "_ensure_instance", return_value="inst-1"), \
             patch.object(plugin, "_safe_log"), patch("time.sleep", return_value=None):
            result = plugin.evaluate_proposed_plan("s", _valid_plan())

        self.assertEqual(result["status"], "timed_out")
        self.assertIsNone(result["approved_budget_usd"])
        self.assertIsNone(plugin._exec_contexts.get("s"))

    def test_no_policy_returns_rejected(self):
        plugin = _make_plugin()
        plugin._client.request_plan_approval = Mock(return_value=_no_policy_response())
        with patch.object(plugin, "_ensure_instance", return_value="inst-1"), \
             patch.object(plugin, "_safe_log"):
            result = plugin.evaluate_proposed_plan("s", _valid_plan())

        self.assertEqual(result["status"], "rejected")
        self.assertIsNone(result["approved_budget_usd"])

    def test_amp_unavailable_returns_error_no_budget(self):
        plugin = _make_plugin()
        plugin._client.request_plan_approval = Mock(side_effect=Exception("connection refused"))
        with patch.object(plugin, "_ensure_instance", return_value="inst-1"), \
             patch.object(plugin, "_safe_log"):
            result = plugin.evaluate_proposed_plan("s", _valid_plan())

        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["approved_budget_usd"])
        self.assertIsNone(plugin._exec_contexts.get("s"))

    def test_unexpected_status_returns_error(self):
        plugin = _make_plugin()
        plugin._client.request_plan_approval = Mock(return_value={"status": "weird"})
        with patch.object(plugin, "_ensure_instance", return_value="inst-1"), \
             patch.object(plugin, "_safe_log"):
            result = plugin.evaluate_proposed_plan("s", _valid_plan())

        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["approved_budget_usd"])

    def test_approved_budget_equals_originally_submitted_cost(self):
        """approved_budget_usd must be the plan's own projected_cost_usd, not recomputed."""
        plugin = _make_plugin()
        plugin._client.request_plan_approval = Mock(return_value=_allow_response())
        with patch.object(plugin, "_ensure_instance", return_value="inst-1"), \
             patch.object(plugin, "_safe_log"):
            result = plugin.evaluate_proposed_plan("s", _valid_plan(projected_cost_usd=12.3456))

        self.assertAlmostEqual(result["approved_budget_usd"], 12.3456, places=5)


# ---------------------------------------------------------------------------
# 3. ExecutionContext interaction
# ---------------------------------------------------------------------------

class PlanFailClosedTests(unittest.TestCase):
    """A2, A4 and K1 on the plan-approval path."""

    def _run(self, decisions):
        plugin = _make_plugin()
        plugin._client.request_plan_approval = Mock(return_value=_hitl_response())
        # K2.1: one baseline get-decision precedes the request.
        plugin._client.get_hitl_decision = Mock(side_effect=[{"status": "pending"}, *decisions])
        with patch.object(plugin, "_ensure_instance", return_value="inst-1"), \
             patch.object(plugin, "_safe_log"), patch("time.sleep", return_value=None):
            return plugin, plugin.evaluate_proposed_plan("s", _valid_plan(projected_cost_usd=4.75))

    def test_hitl_modified_plan_is_rejected_not_approved(self):
        plugin, result = self._run([_decision("modify")])
        self.assertEqual(result["status"], "rejected")
        self.assertIn("re-submit the modified plan", result["reason"])
        self.assertIsNone(result["approved_budget_usd"])
        self.assertIsNone(plugin._exec_contexts.get("s"))

    def test_cancelled_plan_review_is_rejected_with_cancel_reason(self):
        _, result = self._run([{
            "status": "complete", "resolution": "reject", "method": "cancelled", "workitem_id": "w-1",
        }])
        self.assertEqual(result["status"], "rejected")
        self.assertIn("cancelled", result["reason"])

    def test_plan_poll_429_returns_error_in_one_poll(self):
        from hermes.amp_client import AmpClientError

        plugin, result = self._run([AmpClientError("HTTP 429", status_code=429, body="call_limit_reached")])
        self.assertEqual(result["status"], "error")
        self.assertIn("HTTP 429", result["reason"])
        self.assertEqual(plugin._client.get_hitl_decision.call_count, 2)  # baseline + 1 poll

    def test_plan_ignores_decision_for_other_workitem(self):
        plugin, result = self._run([
            {"status": "complete", "resolution": "approve", "workitem_id": "w-OLD"},
            _decision("approved"),
        ])
        self.assertEqual(result["status"], "approved")
        self.assertEqual(plugin._client.get_hitl_decision.call_count, 3)  # baseline + 2 polls


class ExecutionContextInteractionTests(unittest.TestCase):
    def test_existing_context_fields_untouched(self):
        plugin = _make_plugin()
        exec_ctx = ExecutionContext("s", "inst-1", model="m", platform="slack")
        exec_ctx.input_tokens = 500
        exec_ctx.llm_calls = 3
        plugin._exec_contexts._store["s"] = exec_ctx
        plugin._client.request_plan_approval = Mock(return_value=_allow_response())

        with patch.object(plugin, "_ensure_instance", return_value="inst-1"), \
             patch.object(plugin, "_safe_log"):
            plugin.evaluate_proposed_plan("s", _valid_plan(projected_cost_usd=2.0))

        self.assertEqual(exec_ctx.input_tokens, 500)
        self.assertEqual(exec_ctx.llm_calls, 3)
        self.assertAlmostEqual(exec_ctx.approved_budget_usd, 2.0, places=5)

    def test_context_created_when_missing(self):
        plugin = _make_plugin()
        self.assertIsNone(plugin._exec_contexts.get("s"))
        plugin._client.request_plan_approval = Mock(return_value=_allow_response())

        with patch.object(plugin, "_ensure_instance", return_value="inst-1"), \
             patch.object(plugin, "_safe_log"):
            plugin.evaluate_proposed_plan("s", _valid_plan())

        self.assertIsNotNone(plugin._exec_contexts.get("s"))


# ---------------------------------------------------------------------------
# 4. AMP normalization payload (real AmpClient, patched transport)
# ---------------------------------------------------------------------------

class AmpNormalizationTests(unittest.TestCase):
    def test_request_plan_approval_flat_field_promotion(self):
        from hermes.amp_client import AmpClient
        from hermes.config import AmpConfig

        config = AmpConfig(
            backend_url="https://amp.example.com",
            api_key="k", org_id="O-1", agent_name="hermes-test",
            username="u", hitl_timeout_minutes=10,
            hitl_poll_interval_seconds=3, fail_closed=True,
            notifications_enabled=True,
            llm_governance_enabled=True, llm_governance_mode="enforce",
            llm_governance_fail_closed=False,
            llm_governance_include_subagents=True,
        )
        client = AmpClient(config)
        captured = {}

        def fake_request(method, path, payload=None, query=None):
            captured.update(payload or {})
            return {"status": "no-hitl"}

        with patch.object(client, "_request", side_effect=fake_request):
            client.request_plan_approval(
                "inst-1",
                plan_id="plan-1",
                plan_type="research",
                summary="Research five topics",
                projected_cost_usd=4.75,
                projected_cost_status="estimated",
                estimated_llm_calls=18,
                estimated_tool_calls=25,
                estimated_duration_minutes=20,
                work_units_total=5,
                payload={"topics": ["a", "b"]},
            )

        self.assertEqual(captured["tool"], "execution_plan")
        self.assertEqual(captured["action"], "submit")
        required_flat_fields = [
            "plan_id", "plan_type",
            "plan_projected_cost_usd", "plan_projected_cost_status",
            "plan_estimated_llm_calls", "plan_estimated_tool_calls",
            "plan_estimated_duration_minutes", "plan_work_units_total",
        ]
        for field in required_flat_fields:
            self.assertIn(field, captured, f"Missing flat field: {field}")
        self.assertAlmostEqual(captured["plan_projected_cost_usd"], 4.75, places=5)
        self.assertEqual(captured["plan_estimated_llm_calls"], 18)

    def test_opaque_payload_preserved_unmodified(self):
        from hermes.amp_client import AmpClient
        from hermes.config import AmpConfig

        config = AmpConfig(
            backend_url="https://amp.example.com",
            api_key="k", org_id="O-1", agent_name="a",
            username="u", hitl_timeout_minutes=10,
            hitl_poll_interval_seconds=3, fail_closed=True,
            notifications_enabled=True,
            llm_governance_enabled=True, llm_governance_mode="enforce",
            llm_governance_fail_closed=False,
            llm_governance_include_subagents=True,
        )
        client = AmpClient(config)
        captured = {}
        research_payload = {
            "topics": ["AI agent governance", "Progressive autonomy"],
            "research_depth": "standard",
            "sources_per_topic": 5,
            "nested": {"arbitrary": ["structure", 1, True]},
        }

        with patch.object(
            client, "_request",
            side_effect=lambda m, p, payload=None, query=None: captured.update(payload or {}) or {"status": "no-hitl"},
        ):
            client.request_plan_approval("inst-1", plan_type="research", payload=research_payload)

        self.assertEqual(captured["context"]["payload"], research_payload)

    def test_save_llm_trace_payload_shape(self):
        from hermes.amp_client import AmpClient
        from hermes.config import AmpConfig

        config = AmpConfig(
            backend_url="https://amp.example.com",
            api_key="k", org_id="O-1", agent_name="hermes-test",
            username="u", hitl_timeout_minutes=10,
            hitl_poll_interval_seconds=3, fail_closed=True,
            notifications_enabled=True,
            llm_governance_enabled=True, llm_governance_mode="observe",
            llm_governance_fail_closed=False,
            llm_governance_include_subagents=True,
        )
        client = AmpClient(config)
        captured = {}
        captured_path = {}

        def fake_request(method, path, payload=None, query=None):
            captured.update(payload or {})
            captured_path["path"] = path

        with patch.object(client, "_request", side_effect=fake_request):
            client.save_llm_trace(
                "inst-1",
                call_time="2026-07-16T04:17:24+00:00",
                model="gpt-4o-mini",
                prompt={"user": "run my research topics"},
                reasoning=None,
                answer="Called: web_search(...)",
            )

        self.assertEqual(captured_path["path"], "/api/agents/inst-1/llm_trace")
        self.assertEqual(captured["call_time"], "2026-07-16T04:17:24+00:00")
        self.assertEqual(captured["model"], "gpt-4o-mini")
        self.assertEqual(captured["prompt"], {"user": "run my research topics"})
        self.assertEqual(captured["answer"], "Called: web_search(...)")
        self.assertNotIn("reasoning", captured)  # None -> omitted, not sent as null
        self.assertNotIn("trace_id", captured)

    def test_save_llm_trace_includes_reasoning_when_present(self):
        from hermes.amp_client import AmpClient
        from hermes.config import AmpConfig

        config = AmpConfig(
            backend_url="https://amp.example.com",
            api_key="k", org_id="O-1", agent_name="a",
            username="u", hitl_timeout_minutes=10,
            hitl_poll_interval_seconds=3, fail_closed=True,
            notifications_enabled=True,
            llm_governance_enabled=True, llm_governance_mode="observe",
            llm_governance_fail_closed=False,
            llm_governance_include_subagents=True,
        )
        client = AmpClient(config)
        captured = {}

        with patch.object(
            client, "_request",
            side_effect=lambda m, p, payload=None, query=None: captured.update(payload or {}),
        ):
            client.save_llm_trace(
                "inst-1", call_time="2026-07-16T04:17:24+00:00", model="deepseek-r1",
                reasoning="considered several approaches",
            )

        self.assertEqual(captured["reasoning"], "considered several approaches")


# ---------------------------------------------------------------------------
# 5. Public module-level entry point
# ---------------------------------------------------------------------------

class PublicEntryPointTests(unittest.TestCase):
    def test_module_level_function_delegates_to_singleton(self):
        import hermes as mod
        fake_plugin = Mock()
        fake_plugin.evaluate_proposed_plan = Mock(return_value={"status": "approved"})
        with patch.object(mod, "_PLUGIN", fake_plugin):
            result = evaluate_proposed_plan("s", _valid_plan())
        fake_plugin.evaluate_proposed_plan.assert_called_once_with("s", _valid_plan())
        self.assertEqual(result["status"], "approved")


# ---------------------------------------------------------------------------
# 6. Regression: existing Phase 1-2B behavior unaffected
# ---------------------------------------------------------------------------

class RegressionTests(unittest.TestCase):
    def test_pre_tool_call_still_works(self):
        plugin = _make_plugin()
        plugin._client.request_hitl = Mock(return_value={"status": "no-hitl"})
        with patch.object(plugin, "_ensure_instance", return_value="inst-1"), \
             patch.object(plugin, "_safe_log"):
            from hermes.policy import NormalizedAction
            with patch("hermes.normalize_tool_call") as mock_norm:
                mock_norm.return_value = NormalizedAction("terminal", "exec", "exec", {})
                result = plugin.pre_tool_call(tool_name="terminal", args={"command": "ls"}, session_id="s")
        self.assertIsNone(result)

    def test_llm_execution_middleware_still_noops_without_context(self):
        plugin = _make_plugin()
        next_call = Mock(return_value="resp")
        result = plugin.llm_execution_middleware(
            request={}, next_call=next_call, session_id="no-such-session", model="m"
        )
        self.assertEqual(result, "resp")
        next_call.assert_called_once_with({})


class LogPrefixTests(unittest.TestCase):
    """AmpClient.log() is the single choke point all AHP logging goes
    through (_safe_log, log_llm_event, log_execution_summary all call it),
    so prefixing there covers every AHP-issued log line uniformly."""

    def _client(self):
        from hermes.amp_client import AmpClient
        from hermes.config import AmpConfig
        config = AmpConfig(
            backend_url="https://amp.example.com",
            api_key="k", org_id="O-1", agent_name="a",
            username="u", hitl_timeout_minutes=10,
            hitl_poll_interval_seconds=3, fail_closed=True,
            notifications_enabled=True,
            llm_governance_enabled=True, llm_governance_mode="enforce",
            llm_governance_fail_closed=False,
            llm_governance_include_subagents=True,
        )
        return AmpClient(config)

    def test_log_prefixes_message_with_ahp(self):
        client = self._client()
        captured = {}
        with patch.object(
            client, "_request",
            side_effect=lambda m, p, payload=None, query=None: captured.update(payload or {}),
        ):
            client.log("inst-1", "Plan submitted | plan_id=abc")
        self.assertEqual(captured["message"], "[AHP] Plan submitted | plan_id=abc")

    def test_log_llm_event_message_is_prefixed(self):
        client = self._client()
        captured = {}
        with patch.object(
            client, "_request",
            side_effect=lambda m, p, payload=None, query=None: captured.update(payload or {}),
        ):
            client.log_llm_event("inst-1", {"api_call_number": 1, "model": "m", "total_tokens": 100})
        self.assertTrue(captured["message"].startswith("[AHP] "))


if __name__ == "__main__":
    unittest.main()
