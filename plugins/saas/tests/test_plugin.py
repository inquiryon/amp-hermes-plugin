from __future__ import annotations

from datetime import datetime, timezone
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, PropertyMock, patch

ROOT = Path(__file__).resolve().parent.parent
ROOT_PARENT = ROOT.parent
HERMES_AGENT_ROOT = Path.home() / ".hermes" / "hermes-agent"
if str(ROOT_PARENT) not in sys.path:
    sys.path.insert(0, str(ROOT_PARENT))
if HERMES_AGENT_ROOT.exists() and str(HERMES_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(HERMES_AGENT_ROOT))

from hermes import (
    _build_live_search_context,
    _build_research_skill_context,
    _calc_cost,
    _extract_reasoning_from_assistant_message,
    _infer_pricing_provider,
    _is_configured_mode_request,
    _mentions_research,
    _needs_live_web_search,
    _summarize_assistant_answer,
    _TRACE_FIELD_CHARS,
    AmpGovernancePlugin,
)
from hermes.amp_client import AmpClient, AmpClientError
from hermes.policy import NormalizedAction
from hermes.session_store import SessionStore
from hitl_helpers import BASELINE, FakeClock, complete, make_plugin, pending, quiet, terminal_action


# ---------------------------------------------------------------------------
# Freshness routing (unchanged from original tests)
# ---------------------------------------------------------------------------

class FreshnessRoutingTests(unittest.TestCase):
    def test_detects_time_sensitive_market_query(self) -> None:
        self.assertTrue(_needs_live_web_search("How did the US market perform today?"))

    def test_detects_yesterday_market_query(self) -> None:
        self.assertTrue(_needs_live_web_search("How does the market in China and HK perform yesterday?"))

    def test_detects_last_week_market_query(self) -> None:
        self.assertTrue(_needs_live_web_search("How did the US market perform last week?"))

    def test_detects_explicit_web_search_request(self) -> None:
        self.assertTrue(_needs_live_web_search("Perform a web search to find social security fraud news in the US"))

    def test_ignores_non_freshness_prompt(self) -> None:
        self.assertFalse(_needs_live_web_search("Explain what a stock market index is"))

    def test_context_mentions_web_search_and_current_date(self) -> None:
        context = _build_live_search_context("How did the US market perform today?")
        self.assertIn("Current UTC date:", context)
        self.assertIn("web_search tool", context)
        self.assertIn("Do not answer from memory", context)

    def test_pre_llm_call_returns_context_for_freshness_query(self) -> None:
        plugin = AmpGovernancePlugin()
        with patch.object(type(plugin._config), "is_configured", new_callable=PropertyMock, return_value=True), \
             patch.object(plugin, "_ensure_instance", return_value="inst-1"), \
             patch.object(plugin, "_safe_log"):
            result = plugin.pre_llm_call(
                session_id="sess-1",
                user_message="How did the US market perform today?",
                model="test-model",
                platform="slack",
            )
        self.assertIsInstance(result, dict)
        assert isinstance(result, dict)
        self.assertIn("context", result)
        self.assertIn("web_search tool", result["context"])

    def test_pre_llm_call_no_context_for_non_freshness_query(self) -> None:
        plugin = AmpGovernancePlugin()
        with patch.object(type(plugin._config), "is_configured", new_callable=PropertyMock, return_value=True), \
             patch.object(plugin, "_ensure_instance", return_value="inst-1"), \
             patch.object(plugin, "_safe_log"):
            result = plugin.pre_llm_call(
                session_id="sess-1",
                user_message="Explain what a stock market index is",
                model="test-model",
                platform="slack",
            )
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Research-word carve-out: freshness routing must not steal ad hoc research
# requests away from the amp-research-topic skill just because they also
# happen to use time-sensitive/market-y wording (e.g. "today", "market").
# ---------------------------------------------------------------------------

class ResearchCarveOutTests(unittest.TestCase):
    def test_mentions_research_true_for_research_word(self) -> None:
        self.assertTrue(_mentions_research("Research ORCL stock price"))
        self.assertTrue(_mentions_research("research the US market today"))

    def test_mentions_research_false_without_research_word(self) -> None:
        self.assertFalse(_mentions_research("What is the stock price of ORCL?"))
        self.assertFalse(_mentions_research("What is the weather in SF today?"))

    def test_pre_llm_call_injects_skill_load_directive_when_research_mentioned(self) -> None:
        plugin = AmpGovernancePlugin()
        with patch.object(type(plugin._config), "is_configured", new_callable=PropertyMock, return_value=True), \
             patch.object(plugin, "_ensure_instance", return_value="inst-1"), \
             patch.object(plugin, "_safe_log"):
            result = plugin.pre_llm_call(
                session_id="sess-1",
                user_message="research the US market today",
                model="test-model",
                platform="slack",
            )
        self.assertIsInstance(result, dict)
        assert isinstance(result, dict)
        self.assertIn("context", result)
        self.assertIn("amp-governance:research-agent", result["context"])
        self.assertNotIn("web_search tool first", result["context"])

    def test_research_directive_takes_priority_over_freshness_wording(self) -> None:
        """"research on US-Iran situation today" matches both the research
        trigger and the freshness heuristic -- research must win, since a
        governed workflow request should never be diverted into a direct,
        ungoverned web_search."""
        plugin = AmpGovernancePlugin()
        with patch.object(type(plugin._config), "is_configured", new_callable=PropertyMock, return_value=True), \
             patch.object(plugin, "_ensure_instance", return_value="inst-1"), \
             patch.object(plugin, "_safe_log") as mock_log:
            result = plugin.pre_llm_call(
                session_id="sess-1",
                user_message="research on US-Iran situation today",
                model="test-model",
                platform="slack",
            )
        assert isinstance(result, dict)
        self.assertIn("amp-governance:research-agent", result["context"])
        logged_messages = [call.args[1] for call in mock_log.call_args_list]
        self.assertTrue(any("Research routing" in msg for msg in logged_messages))
        self.assertFalse(any("Freshness routing" in msg for msg in logged_messages))

    def test_build_research_skill_context_names_the_skill(self) -> None:
        context = _build_research_skill_context("research ORCL stock price")
        self.assertIn("amp-governance:research-agent", context)
        self.assertIn("skill_view", context)

    def test_build_research_skill_context_forbids_pointer_skills(self) -> None:
        context = _build_research_skill_context("research the US market today")
        self.assertIn("amp-research", context)
        self.assertIn("amp-research-topic", context)

    def test_is_configured_mode_request_true_for_saved_topics_phrasing(self) -> None:
        self.assertTrue(_is_configured_mode_request("run my research topics"))
        self.assertTrue(_is_configured_mode_request("Research my configured topics"))
        self.assertTrue(_is_configured_mode_request("Run my daily research"))

    def test_is_configured_mode_request_false_for_named_topic(self) -> None:
        self.assertFalse(_is_configured_mode_request("research the US market today"))
        self.assertFalse(_is_configured_mode_request("research ORCL stock price"))
        self.assertFalse(_is_configured_mode_request("research on US-Iran situation today"))

    def test_build_research_skill_context_tags_configured_mode(self) -> None:
        context = _build_research_skill_context("run my research topics")
        self.assertIn("mode=CONFIGURED", context)

    def test_build_research_skill_context_tags_ad_hoc_mode(self) -> None:
        context = _build_research_skill_context("research the US market today")
        self.assertIn("mode=AD HOC", context)

    def test_build_research_skill_context_grounds_current_date(self) -> None:
        """A research request must be told the real current date the same way
        freshness routing already is -- otherwise the model falls back to a
        stale/hallucinated date for both search queries and the report
        dateline (observed: "October 2023" search queries and report header
        while the real date was July 2026)."""
        with patch("hermes.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2026, 7, 17, tzinfo=timezone.utc)
            context = _build_research_skill_context("research the US market today")
        self.assertIn("Current UTC date: July 17, 2026", context)

    def test_pre_llm_call_still_injects_context_for_plain_freshness_query(self) -> None:
        """Weather/stock-price/current-events questions with no "research"
        wording must keep using the pre-existing freshness routing path
        unchanged."""
        plugin = AmpGovernancePlugin()
        with patch.object(type(plugin._config), "is_configured", new_callable=PropertyMock, return_value=True), \
             patch.object(plugin, "_ensure_instance", return_value="inst-1"), \
             patch.object(plugin, "_safe_log"):
            result = plugin.pre_llm_call(
                session_id="sess-1",
                user_message="What is the stock price of ORCL today?",
                model="test-model",
                platform="slack",
            )
        self.assertIsInstance(result, dict)
        assert isinstance(result, dict)
        self.assertIn("web_search tool", result["context"])


# ---------------------------------------------------------------------------
# Pricing provider inference: Hermes' auth-provider config ("custom" for a
# generic OpenAI-compatible endpoint) and the pricing module's billing-route
# vocabulary ("openai") don't overlap -- config.yaml can't just say
# provider: openai (hermes doctor rejects it, breaks live LLM calls). Real
# per-token cost tracking depends on inferring the pricing route from
# base_url instead, without touching the auth-provider config at all.
# ---------------------------------------------------------------------------

class PricingProviderInferenceTests(unittest.TestCase):
    def test_custom_provider_with_openai_base_url_infers_openai(self) -> None:
        self.assertEqual(
            _infer_pricing_provider("custom", "https://api.openai.com/v1"), "openai"
        )

    def test_local_provider_with_openai_base_url_infers_openai(self) -> None:
        self.assertEqual(
            _infer_pricing_provider("local", "https://api.openai.com/v1"), "openai"
        )

    def test_empty_provider_with_openai_base_url_infers_openai(self) -> None:
        self.assertEqual(_infer_pricing_provider("", "https://api.openai.com/v1"), "openai")

    def test_custom_provider_with_non_openai_base_url_left_unchanged(self) -> None:
        self.assertEqual(
            _infer_pricing_provider("custom", "http://localhost:11434/v1"), "custom"
        )

    def test_distinct_real_provider_never_overridden(self) -> None:
        """A provider that's already something billing-specific (not
        "custom"/"local") must never be second-guessed, even against an
        openai.com base_url -- that would be a real, deliberate config."""
        self.assertEqual(
            _infer_pricing_provider("anthropic", "https://api.openai.com/v1"), "anthropic"
        )

    def test_calc_cost_resolves_real_openai_pricing_via_custom_provider(self) -> None:
        """End-to-end (no mocking of the pricing module itself): a
        provider="custom" + base_url=api.openai.com call for gpt-4o-mini
        must resolve to real, non-zero pricing -- this is the exact
        configuration this fix targets."""
        cost, status, _source = _calc_cost(
            "gpt-4o-mini",
            "custom",
            "https://api.openai.com/v1",
            {"input_tokens": 1_000_000, "output_tokens": 0},
        )
        self.assertEqual(status, "estimated")
        self.assertAlmostEqual(cost, 0.15, places=5)


# ---------------------------------------------------------------------------
# Notification bridge (updated from Slack-only to platform-neutral)
# ---------------------------------------------------------------------------

class NotificationBridgeTests(unittest.TestCase):
    """Tests for the platform-neutral notification bridge."""

    def _make_plugin_with_dispatch(self, return_value=None) -> tuple:
        plugin = AmpGovernancePlugin()
        dispatch = Mock(return_value=json.dumps({"success": True}) if return_value is None else json.dumps(return_value))
        plugin._dispatch_tool = dispatch
        return plugin, dispatch

    def test_notify_user_sends_to_slack_thread(self) -> None:
        plugin, dispatch = self._make_plugin_with_dispatch()
        with patch("hermes.notification.build_notification_target", return_value="slack:C1234567890:1712345678.100200"):
            plugin._notify_user("Waiting for approval.")
        dispatch.assert_called_once_with(
            "send_message",
            {
                "target": "slack:C1234567890:1712345678.100200",
                "message": "[AMP]\nWaiting for approval.",
            },
        )

    def test_notify_user_sends_to_telegram(self) -> None:
        plugin, dispatch = self._make_plugin_with_dispatch()
        with patch("hermes.notification.build_notification_target", return_value="telegram:-100123:17585"):
            plugin._notify_user("Approval required.")
        dispatch.assert_called_once_with(
            "send_message",
            {
                "target": "telegram:-100123:17585",
                "message": "[AMP]\nApproval required.",
            },
        )

    def test_notify_user_sends_to_discord(self) -> None:
        plugin, dispatch = self._make_plugin_with_dispatch()
        with patch("hermes.notification.build_notification_target", return_value="discord:999888"):
            plugin._notify_user("Review requested.")
        dispatch.assert_called_with("send_message", {"target": "discord:999888", "message": "[AMP]\nReview requested."})

    def test_notify_user_is_noop_in_cli_mode(self) -> None:
        plugin, dispatch = self._make_plugin_with_dispatch()
        with patch("hermes.notification.build_notification_target", return_value=""):
            plugin._notify_user("Should not send.")
        dispatch.assert_not_called()

    def test_notify_user_is_noop_when_dispatch_missing(self) -> None:
        plugin = AmpGovernancePlugin()
        plugin._dispatch_tool = None
        with patch("hermes.notification.build_notification_target", return_value="slack:C123"):
            plugin._notify_user("Should not send.")  # must not raise

    def test_notify_user_failure_does_not_raise(self) -> None:
        plugin = AmpGovernancePlugin()
        plugin._dispatch_tool = Mock(side_effect=RuntimeError("channel gone"))
        with patch("hermes.notification.build_notification_target", return_value="slack:C123"):
            plugin._notify_user("message")  # must not raise

    def test_notify_user_disabled_when_notifications_off(self) -> None:
        plugin = AmpGovernancePlugin()
        plugin._dispatch_tool = Mock(return_value=json.dumps({"success": True}))
        # Override config with notifications_enabled=False
        plugin._config = SimpleNamespace(notifications_enabled=False)
        with patch("hermes.notification.build_notification_target", return_value="slack:C123"):
            plugin._notify_user("Should not send.")
        plugin._dispatch_tool.assert_not_called()

    # Regression: existing Slack HITL notification behavior is preserved
    def test_pending_hitl_sends_waiting_notification(self) -> None:
        plugin = AmpGovernancePlugin()
        plugin._notify_user = Mock()
        plugin._client.request_hitl = Mock(return_value={"status": "pending", "workitem_id": "w1"})
        plugin._client.get_hitl_decision = Mock(
            return_value={"status": "complete", "resolution": "approved", "workitem_id": "w1"}
        )
        plugin._config = SimpleNamespace(
            hitl_timeout_minutes=1,
            hitl_poll_interval_seconds=0,
            username="tester",
            fail_closed=True,
            agent_name="amp-test",
        )
        action = NormalizedAction("web_search", "exec", "web_search", {"query": "today market"})
        with patch.object(plugin, "_safe_log"), patch("time.sleep", return_value=None):
            result = plugin._evaluate_governance("inst-1", action)
        self.assertIsNone(result)
        plugin._notify_user.assert_any_call(
            'AMP is waiting for a human reviewer to approve "web_search" before continuing. '
            "This action is paused pending review."
        )
        plugin._notify_user.assert_any_call('AMP review approved "web_search". Continuing now.')

    def test_rejected_hitl_sends_rejection_notification(self) -> None:
        plugin = AmpGovernancePlugin()
        plugin._notify_user = Mock()
        plugin._client.request_hitl = Mock(return_value={"status": "pending", "workitem_id": "w1"})
        plugin._client.get_hitl_decision = Mock(
            return_value={
                "status": "complete", "resolution": "rejected",
                "information": "Need human review", "workitem_id": "w1",
            }
        )
        plugin._config = SimpleNamespace(
            hitl_timeout_minutes=1,
            hitl_poll_interval_seconds=0,
            username="tester",
            fail_closed=True,
            agent_name="amp-test",
        )
        action = NormalizedAction("terminal", "exec", "exec", {"command": "rm -rf /tmp/x"})
        with patch.object(plugin, "_safe_log"), patch("time.sleep", return_value=None):
            result = plugin._evaluate_governance("inst-1", action)
        self.assertEqual(
            result,
            {
                "action": "block",
                "message": "terminal was rejected by AMP HITL review. Reviewer note: Need human review",
            },
        )
        plugin._notify_user.assert_any_call('AMP reviewer rejected "terminal". Reviewer note: Need human review')

    def test_timeout_hitl_sends_timeout_notification(self) -> None:
        plugin = AmpGovernancePlugin()
        plugin._notify_user = Mock()
        plugin._client.request_hitl = Mock(return_value={"status": "pending", "workitem_id": "w1"})
        # Decision never becomes complete
        plugin._client.get_hitl_decision = Mock(return_value={"status": "pending"})
        plugin._config = SimpleNamespace(
            hitl_timeout_minutes=0,  # immediate timeout
            hitl_poll_interval_seconds=0,
            username="tester",
            fail_closed=True,
            agent_name="amp-test",
        )
        action = NormalizedAction("terminal", "exec", "exec", {"command": "ls"})
        with patch.object(plugin, "_safe_log"), patch("time.sleep", return_value=None):
            result = plugin._evaluate_governance("inst-1", action)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["action"], "block")
        self.assertIn("timed out", result["message"])
        plugin._notify_user.assert_any_call(
            'AMP review timed out for "terminal". The action was blocked.'
        )


# ---------------------------------------------------------------------------
# Tool governance (existing behavior must be unchanged)
# ---------------------------------------------------------------------------

class ToolGovernanceRegressionTests(unittest.TestCase):
    def test_pre_tool_call_blocks_on_no_policy(self) -> None:
        plugin = AmpGovernancePlugin()
        plugin._client.request_hitl = Mock(return_value={"status": "no_policy"})
        plugin._config = SimpleNamespace(
            is_configured=True,
            fail_closed=True,
            agent_name="amp-test",
            username="tester",
            hitl_timeout_minutes=1,
            hitl_poll_interval_seconds=0,
            notifications_enabled=False,
            llm_governance_enabled=False,
        )
        with patch.object(plugin, "_ensure_instance", return_value="inst-1"), \
             patch.object(plugin, "_safe_log"):
            result = plugin.pre_tool_call(
                tool_name="terminal",
                args={"command": "ls"},
                session_id="sess-1",
            )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["action"], "block")

    def test_pre_tool_call_allows_on_no_hitl(self) -> None:
        plugin = AmpGovernancePlugin()
        plugin._client.request_hitl = Mock(return_value={"status": "no-hitl"})
        plugin._client.get_hitl_decision = Mock(return_value={"status": "pending"})  # K2.1 baseline
        plugin._config = SimpleNamespace(
            is_configured=True,
            fail_closed=True,
            agent_name="amp-test",
            username="tester",
            hitl_timeout_minutes=1,
            hitl_poll_interval_seconds=0,
            notifications_enabled=False,
            llm_governance_enabled=False,
        )
        with patch.object(plugin, "_ensure_instance", return_value="inst-1"), \
             patch.object(plugin, "_safe_log"):
            result = plugin.pre_tool_call(
                tool_name="terminal",
                args={"command": "ls"},
                session_id="sess-1",
            )
        self.assertIsNone(result)

    def test_ungoverned_tool_is_ignored(self) -> None:
        plugin = AmpGovernancePlugin()
        plugin._client.request_hitl = Mock()
        plugin._config = SimpleNamespace(
            is_configured=True,
            fail_closed=True,
            llm_governance_enabled=False,
        )
        result = plugin.pre_tool_call(
            tool_name="send_message",
            args={"target": "slack:C123", "message": "hi"},
            session_id="sess-1",
        )
        self.assertIsNone(result)
        plugin._client.request_hitl.assert_not_called()

    def test_policy_error_block_uses_backend_message(self) -> None:
        plugin = make_plugin()
        plugin._client.request_hitl = Mock(return_value={
            "status": "policy_error", "reason": "policy_invalid_hitl_assignee", "message": "reviewer invalid",
        })
        with quiet(plugin):
            result = plugin.pre_tool_call(tool_name="terminal", args={"command": "ls"}, session_id="s1")
        self.assertEqual(result, {"action": "block", "message": "reviewer invalid"})

    def test_http_400_reports_rejected_not_unavailable(self) -> None:
        plugin = make_plugin()
        plugin._client.request_hitl = Mock(side_effect=AmpClientError(
            "HTTP 400: bad", status_code=400, body='{"error":"missing_username_in_meta"}',
        ))
        with quiet(plugin):
            result = plugin.pre_tool_call(tool_name="terminal", args={"command": "ls"}, session_id="s1")
        assert result is not None
        self.assertEqual(result["action"], "block")
        self.assertTrue(result["message"].startswith("AMP rejected the governance request"), result["message"])
        self.assertIn("HTTP 400", result["message"])
        self.assertIn("missing_username_in_meta", result["message"])

    def test_http_4xx_blocks_even_when_fail_open(self) -> None:
        # H2: a 4xx is AMP refusing the call, not an outage; AMP_FAIL_CLOSED=false must not allow it.
        plugin = make_plugin(fail_closed=False)
        plugin._client.request_hitl = Mock(side_effect=AmpClientError(
            "HTTP 422: policy_error", status_code=422, body='{"status":"policy_error"}',
        ))
        with quiet(plugin):
            result = plugin.pre_tool_call(tool_name="terminal", args={"command": "ls"}, session_id="s1")
        assert result is not None
        self.assertEqual(result["action"], "block")
        self.assertIn("HTTP 422", result["message"])

    def test_url_error_still_reports_unavailable(self) -> None:
        plugin = make_plugin()
        plugin._client.request_hitl = Mock(side_effect=AmpClientError("connection refused"))
        with quiet(plugin):
            result = plugin.pre_tool_call(tool_name="terminal", args={"command": "ls"}, session_id="s1")
        assert result is not None
        self.assertTrue(result["message"].startswith("AMP governance is unavailable"), result["message"])

    def test_url_error_with_fail_open_still_allows(self) -> None:
        # Regression: the operator's explicit outage policy is unchanged for real outages.
        plugin = make_plugin(fail_closed=False)
        plugin._client.request_hitl = Mock(side_effect=AmpClientError("connection refused", pre_send=True))
        with quiet(plugin):
            result = plugin.pre_tool_call(tool_name="terminal", args={"command": "ls"}, session_id="s1")
        self.assertIsNone(result)

    def test_pre_tool_call_not_blocked_by_corrupt_store(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "sessions.json"
            store_path.write_text(json.dumps({
                "s1": {"session_id": "s1", "instance_id": "inst-old", "future_field": 1},
            }))
            plugin = make_plugin()
            plugin._store = SessionStore(path=store_path)
            plugin._client.init_instance = Mock(return_value="inst-9")
            plugin._client.request_hitl = Mock(return_value={"status": "no-hitl"})
            with patch.object(plugin, "_safe_log"):
                result = plugin.pre_tool_call(tool_name="terminal", args={"command": "ls"}, session_id="s1")
        self.assertIsNone(result)

    def test_transform_llm_output_replaces_blocked_turn(self) -> None:
        plugin = AmpGovernancePlugin()
        plugin._blocked_turn_messages["sess-1"] = "blocked reason"
        result = plugin.transform_llm_output(response_text="original", session_id="sess-1")
        self.assertEqual(
            result,
            "This request is blocked by AMP governance. No action was taken.\nReason: blocked reason",
        )

    def test_transform_llm_output_passthrough_when_not_blocked(self) -> None:
        plugin = AmpGovernancePlugin()
        result = plugin.transform_llm_output(response_text="normal response", session_id="sess-1")
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# LLM observation (Phase 2A)
# ---------------------------------------------------------------------------

class LlmObservationTests(unittest.TestCase):
    def _make_plugin(self, llm_governance_enabled: bool = True) -> AmpGovernancePlugin:
        plugin = AmpGovernancePlugin()
        plugin._config = SimpleNamespace(
            is_configured=True,
            llm_governance_enabled=llm_governance_enabled,
            notifications_enabled=False,
            fail_closed=True,
            agent_name="amp-test",
            username="tester",
        )
        # Silence AMP client calls
        plugin._client.log = Mock()
        plugin._client.log_llm_event = Mock()
        plugin._client.log_execution_summary = Mock()
        plugin._client.save_llm_trace = Mock()
        return plugin

    def _start_session(self, plugin: AmpGovernancePlugin, session_id: str, instance_id: str = "inst-1") -> None:
        plugin._exec_contexts.create(session_id, instance_id, model="claude-opus-4-8", platform="slack")

    def _make_usage(self, **overrides) -> dict:
        base = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "reasoning_tokens": 0,
        }
        base.update(overrides)
        return base

    def test_hook_skipped_when_disabled(self) -> None:
        plugin = self._make_plugin(llm_governance_enabled=False)
        self._start_session(plugin, "sess-1")
        plugin.post_api_request(
            session_id="sess-1",
            model="claude-opus-4-8",
            provider="anthropic",
            usage=self._make_usage(),
        )
        ctx = plugin._exec_contexts.get("sess-1")
        assert ctx is not None
        self.assertEqual(ctx.llm_calls, 0)  # nothing accumulated

    def test_single_call_accumulates_tokens(self) -> None:
        plugin = self._make_plugin()
        self._start_session(plugin, "sess-1")
        plugin.post_api_request(
            session_id="sess-1",
            api_request_id="req-1",
            model="claude-opus-4-8",
            provider="anthropic",
            base_url="",
            api_call_count=1,
            api_duration=1.2,
            usage=self._make_usage(input_tokens=200, output_tokens=80),
        )
        ctx = plugin._exec_contexts.get("sess-1")
        assert ctx is not None
        self.assertEqual(ctx.input_tokens, 200)
        self.assertEqual(ctx.output_tokens, 80)
        self.assertEqual(ctx.llm_calls, 1)

    def test_multi_turn_accumulates_across_calls(self) -> None:
        plugin = self._make_plugin()
        self._start_session(plugin, "sess-1")
        for i in range(3):
            plugin.post_api_request(
                session_id="sess-1",
                api_request_id=f"req-{i}",
                api_call_count=i + 1,
                model="claude-opus-4-8",
                provider="anthropic",
                usage=self._make_usage(input_tokens=100, output_tokens=50),
            )
        ctx = plugin._exec_contexts.get("sess-1")
        assert ctx is not None
        self.assertEqual(ctx.input_tokens, 300)
        self.assertEqual(ctx.output_tokens, 150)
        self.assertEqual(ctx.llm_calls, 3)

    def test_none_usage_records_unknown_cost(self) -> None:
        plugin = self._make_plugin()
        self._start_session(plugin, "sess-1")
        plugin.post_api_request(
            session_id="sess-1",
            model="local-model",
            provider="ollama",
            usage=None,
        )
        ctx = plugin._exec_contexts.get("sess-1")
        assert ctx is not None
        self.assertEqual(ctx.llm_calls, 1)
        self.assertEqual(ctx.input_tokens, 0)
        self.assertEqual(ctx.output_tokens, 0)
        self.assertEqual(ctx.cost_status, "unknown")
        self.assertEqual(ctx.total_cost_usd, 0.0)

    def test_cache_tokens_accumulated(self) -> None:
        plugin = self._make_plugin()
        self._start_session(plugin, "sess-1")
        plugin.post_api_request(
            session_id="sess-1",
            model="claude-opus-4-8",
            provider="anthropic",
            usage=self._make_usage(
                input_tokens=100,
                cache_read_tokens=500,
                cache_write_tokens=200,
            ),
        )
        ctx = plugin._exec_contexts.get("sess-1")
        assert ctx is not None
        self.assertEqual(ctx.cache_read_tokens, 500)
        self.assertEqual(ctx.cache_write_tokens, 200)

    def test_reasoning_tokens_accumulated(self) -> None:
        plugin = self._make_plugin()
        self._start_session(plugin, "sess-1")
        plugin.post_api_request(
            session_id="sess-1",
            model="claude-opus-4-8",
            provider="anthropic",
            usage=self._make_usage(reasoning_tokens=150),
        )
        ctx = plugin._exec_contexts.get("sess-1")
        assert ctx is not None
        self.assertEqual(ctx.reasoning_tokens, 150)

    def test_retry_records_separate_calls(self) -> None:
        plugin = self._make_plugin()
        self._start_session(plugin, "sess-1")
        for i in range(2):
            plugin.post_api_request(
                session_id="sess-1",
                api_request_id=f"req-attempt-{i}",
                api_call_count=i + 1,
                model="claude-opus-4-8",
                provider="anthropic",
                usage=self._make_usage(input_tokens=50),
            )
        ctx = plugin._exec_contexts.get("sess-1")
        assert ctx is not None
        self.assertEqual(ctx.llm_calls, 2)
        self.assertEqual(len(ctx.llm_call_records), 2)
        self.assertEqual(ctx.input_tokens, 100)

    def test_concurrent_sessions_are_isolated(self) -> None:
        plugin = self._make_plugin()
        import threading

        sessions = ["sess-a", "sess-b", "sess-c"]
        for sid in sessions:
            plugin._exec_contexts.create(sid, f"inst-{sid}")

        errors = []

        def worker(sid: str, token_count: int) -> None:
            try:
                for _ in range(5):
                    plugin.post_api_request(
                        session_id=sid,
                        model="m",
                        provider="p",
                        usage=self._make_usage(input_tokens=token_count),
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(sid, i * 10))
            for i, sid in enumerate(sessions, start=1)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertFalse(errors)
        for i, sid in enumerate(sessions, start=1):
            ctx = plugin._exec_contexts.get(sid)
            assert ctx is not None
            self.assertEqual(ctx.input_tokens, i * 10 * 5)

    def test_amp_log_called_per_llm_call(self) -> None:
        plugin = self._make_plugin()
        self._start_session(plugin, "sess-1")
        plugin.post_api_request(
            session_id="sess-1",
            model="claude-opus-4-8",
            provider="anthropic",
            usage=self._make_usage(),
        )
        plugin._client.log_llm_event.assert_called_once()

    def test_amp_log_failure_does_not_propagate(self) -> None:
        plugin = self._make_plugin()
        plugin._client.log_llm_event = Mock(side_effect=Exception("AMP unavailable"))
        self._start_session(plugin, "sess-1")
        plugin.post_api_request(
            session_id="sess-1",
            model="claude-opus-4-8",
            provider="anthropic",
            usage=self._make_usage(),
        )  # must not raise
        ctx = plugin._exec_contexts.get("sess-1")
        assert ctx is not None
        self.assertEqual(ctx.llm_calls, 1)  # accumulation still happened

    def test_no_context_for_session_is_ignored(self) -> None:
        plugin = self._make_plugin()
        # Do not create context for this session
        plugin.post_api_request(
            session_id="sess-no-context",
            model="claude-opus-4-8",
            provider="anthropic",
            usage=self._make_usage(),
        )  # must not raise

    def test_tool_call_counter_incremented_for_governed_tools(self) -> None:
        plugin = self._make_plugin()
        self._start_session(plugin, "sess-1")
        plugin._store.get = Mock(return_value=SimpleNamespace(
            instance_id="inst-1",
            session_id="sess-1",
            model="m",
            platform="slack",
        ))
        plugin._safe_log = Mock()
        plugin.post_tool_call(
            tool_name="terminal",
            args={"command": "ls"},
            session_id="sess-1",
            status="success",
        )
        ctx = plugin._exec_contexts.get("sess-1")
        assert ctx is not None
        self.assertEqual(ctx.tool_calls, 1)

    def test_tool_call_counter_not_incremented_for_ungoverned_tools(self) -> None:
        plugin = self._make_plugin()
        self._start_session(plugin, "sess-1")
        plugin.post_tool_call(
            tool_name="send_message",
            args={"target": "slack:C123", "message": "hi"},
            session_id="sess-1",
            status="success",
        )
        ctx = plugin._exec_contexts.get("sess-1")
        assert ctx is not None
        self.assertEqual(ctx.tool_calls, 0)

    def test_session_finalize_logs_summary_and_cleans_up(self) -> None:
        plugin = self._make_plugin()
        plugin._store.get = Mock(return_value=SimpleNamespace(instance_id="inst-1"))
        plugin._store.delete = Mock()
        plugin._client.set_state = Mock()
        plugin._safe_log = Mock()
        self._start_session(plugin, "sess-1")
        plugin.post_api_request(
            session_id="sess-1",
            model="m",
            provider="p",
            usage=self._make_usage(input_tokens=500),
        )
        plugin.on_session_finalize(session_id="sess-1", reason="done")
        # ExecutionContext removed
        self.assertIsNone(plugin._exec_contexts.get("sess-1"))
        # Summary logged
        plugin._client.log_execution_summary.assert_called_once()
        summary = plugin._client.log_execution_summary.call_args[0][1]
        self.assertEqual(summary["input_tokens"], 500)
        self.assertEqual(summary["status"], "finished")

    def test_session_finalize_cleans_up_even_when_summary_log_fails(self) -> None:
        plugin = self._make_plugin()
        plugin._client.log_execution_summary = Mock(side_effect=Exception("AMP down"))
        plugin._store.get = Mock(return_value=SimpleNamespace(instance_id="inst-1"))
        plugin._store.delete = Mock()
        plugin._client.set_state = Mock()
        plugin._safe_log = Mock()
        self._start_session(plugin, "sess-1")
        plugin.on_session_finalize(session_id="sess-1", reason="done")
        self.assertIsNone(plugin._exec_contexts.get("sess-1"))  # still cleaned up


class LlmTraceCaptureTests(unittest.TestCase):
    """Covers pre_api_request prompt stashing and post_api_request's best-effort
    save_llm_trace call -- the data AMP's "click LLM in the log" panel reads."""

    def _make_plugin(self, llm_governance_enabled: bool = True) -> AmpGovernancePlugin:
        plugin = AmpGovernancePlugin()
        plugin._config = SimpleNamespace(
            is_configured=True,
            llm_governance_enabled=llm_governance_enabled,
            notifications_enabled=False,
            fail_closed=True,
            agent_name="amp-test",
            username="tester",
        )
        plugin._client.log = Mock()
        plugin._client.log_llm_event = Mock()
        plugin._client.save_llm_trace = Mock()
        return plugin

    def _start_session(self, plugin: AmpGovernancePlugin, session_id: str, instance_id: str = "inst-1") -> None:
        plugin._exec_contexts.create(session_id, instance_id, model="claude-opus-4-8", platform="slack")

    # -- pre_api_request stashing -----------------------------------------

    def test_pre_api_request_stashes_prompt_by_request_id(self) -> None:
        plugin = self._make_plugin()
        plugin.pre_api_request(api_request_id="req-1", user_message="do the thing")
        self.assertEqual(plugin._pending_prompts["req-1"], "do the thing")

    def test_pre_api_request_noop_when_governance_disabled(self) -> None:
        plugin = self._make_plugin(llm_governance_enabled=False)
        plugin.pre_api_request(api_request_id="req-1", user_message="do the thing")
        self.assertNotIn("req-1", plugin._pending_prompts)

    def test_pre_api_request_noop_without_request_id(self) -> None:
        plugin = self._make_plugin()
        plugin.pre_api_request(api_request_id="", user_message="do the thing")
        self.assertEqual(plugin._pending_prompts, {})

    def test_pre_api_request_truncates_long_message(self) -> None:
        plugin = self._make_plugin()
        long_message = "x" * 5000
        plugin.pre_api_request(api_request_id="req-1", user_message=long_message)
        self.assertLessEqual(len(plugin._pending_prompts["req-1"]), _TRACE_FIELD_CHARS)

    def test_pop_pending_prompt_removes_entry(self) -> None:
        plugin = self._make_plugin()
        plugin.pre_api_request(api_request_id="req-1", user_message="hello")
        popped = plugin._pop_pending_prompt("req-1")
        self.assertEqual(popped, "hello")
        self.assertNotIn("req-1", plugin._pending_prompts)

    def test_pop_pending_prompt_missing_returns_empty(self) -> None:
        plugin = self._make_plugin()
        self.assertEqual(plugin._pop_pending_prompt("no-such-request"), "")

    # -- pre_api_request -> ExecutionContext provider/base_url capture -----
    # (needed so _project_research_cost has real pricing inputs available
    # *before* a same-turn tool call like amp_evaluate_research_plan
    # dispatches -- post_api_request's accumulate step fires too late for
    # that, since tool dispatch happens before usage is accumulated.)

    def test_pre_api_request_captures_provider_and_base_url_on_exec_ctx(self) -> None:
        plugin = self._make_plugin()
        self._start_session(plugin, "sess-1")
        plugin.pre_api_request(
            api_request_id="req-1", user_message="hi", session_id="sess-1",
            provider="custom", base_url="https://api.openai.com/v1",
        )
        exec_ctx = plugin._exec_contexts.get("sess-1")
        self.assertEqual(exec_ctx.last_provider, "custom")
        self.assertEqual(exec_ctx.last_base_url, "https://api.openai.com/v1")

    def test_pre_api_request_no_exec_context_does_not_raise(self) -> None:
        plugin = self._make_plugin()
        plugin.pre_api_request(
            api_request_id="req-1", user_message="hi", session_id="no-such-session",
            provider="custom", base_url="https://api.openai.com/v1",
        )  # must not raise

    def test_pre_api_request_empty_provider_does_not_clear_existing_value(self) -> None:
        plugin = self._make_plugin()
        self._start_session(plugin, "sess-1")
        exec_ctx = plugin._exec_contexts.get("sess-1")
        exec_ctx.last_provider = "custom"
        exec_ctx.last_base_url = "https://api.openai.com/v1"
        plugin.pre_api_request(api_request_id="req-2", user_message="hi", session_id="sess-1")
        self.assertEqual(exec_ctx.last_provider, "custom")
        self.assertEqual(exec_ctx.last_base_url, "https://api.openai.com/v1")

    # -- post_api_request -> save_llm_trace --------------------------------

    def test_post_api_request_saves_trace_with_prompt_and_answer(self) -> None:
        plugin = self._make_plugin()
        self._start_session(plugin, "sess-1")
        plugin.pre_api_request(api_request_id="req-1", user_message="run my research topics")
        assistant_message = SimpleNamespace(content="Here is the report.", tool_calls=None)
        plugin.post_api_request(
            session_id="sess-1",
            api_request_id="req-1",
            model="gpt-4o-mini",
            provider="openai",
            usage={"input_tokens": 10, "output_tokens": 5},
            assistant_message=assistant_message,
        )
        plugin._client.save_llm_trace.assert_called_once()
        _, kwargs = plugin._client.save_llm_trace.call_args
        self.assertEqual(kwargs["model"], "gpt-4o-mini")
        self.assertEqual(kwargs["prompt"], {"user": "run my research topics"})
        self.assertEqual(kwargs["answer"], "Here is the report.")
        self.assertIsNone(kwargs["reasoning"])
        self.assertIn("call_time", kwargs)

    def test_post_api_request_answer_summarizes_tool_calls_when_content_empty(self) -> None:
        plugin = self._make_plugin()
        self._start_session(plugin, "sess-1")
        tool_call = SimpleNamespace(
            function=SimpleNamespace(name="web_search", arguments='{"query": "AI governance"}')
        )
        assistant_message = SimpleNamespace(content="", tool_calls=[tool_call])
        plugin.post_api_request(
            session_id="sess-1",
            api_request_id="req-1",
            model="gpt-4o-mini",
            usage={"input_tokens": 10, "output_tokens": 5},
            assistant_message=assistant_message,
        )
        kwargs = plugin._client.save_llm_trace.call_args.kwargs
        self.assertIn("web_search", kwargs["answer"])
        self.assertIn("AI governance", kwargs["answer"])

    def test_post_api_request_no_prompt_when_nothing_stashed(self) -> None:
        plugin = self._make_plugin()
        self._start_session(plugin, "sess-1")
        plugin.post_api_request(
            session_id="sess-1",
            api_request_id="req-never-stashed",
            model="gpt-4o-mini",
            usage={"input_tokens": 10, "output_tokens": 5},
            assistant_message=SimpleNamespace(content="ok", tool_calls=None),
        )
        kwargs = plugin._client.save_llm_trace.call_args.kwargs
        self.assertIsNone(kwargs["prompt"])

    def test_post_api_request_extracts_reasoning_when_present(self) -> None:
        plugin = self._make_plugin()
        self._start_session(plugin, "sess-1")
        assistant_message = SimpleNamespace(content="answer", tool_calls=None, reasoning="thought about it")
        plugin.post_api_request(
            session_id="sess-1",
            api_request_id="req-1",
            model="deepseek-r1",
            usage={"input_tokens": 10, "output_tokens": 5},
            assistant_message=assistant_message,
        )
        kwargs = plugin._client.save_llm_trace.call_args.kwargs
        self.assertEqual(kwargs["reasoning"], "thought about it")

    def test_save_llm_trace_failure_does_not_affect_accumulation(self) -> None:
        """Regression: a broken trace save must never break Phase 2A accumulation."""
        plugin = self._make_plugin()
        plugin._client.save_llm_trace = Mock(side_effect=Exception("network down"))
        self._start_session(plugin, "sess-1")
        plugin.post_api_request(
            session_id="sess-1",
            api_request_id="req-1",
            model="gpt-4o-mini",
            usage={"input_tokens": 100, "output_tokens": 50},
            assistant_message=SimpleNamespace(content="ok", tool_calls=None),
        )
        ctx = plugin._exec_contexts.get("sess-1")
        self.assertEqual(ctx.input_tokens, 100)
        self.assertEqual(ctx.llm_calls, 1)
        plugin._client.log_llm_event.assert_called_once()  # unaffected by trace failure

    def test_no_trace_saved_when_governance_disabled(self) -> None:
        plugin = self._make_plugin(llm_governance_enabled=False)
        self._start_session(plugin, "sess-1")
        plugin.post_api_request(
            session_id="sess-1",
            api_request_id="req-1",
            model="gpt-4o-mini",
            usage={"input_tokens": 10, "output_tokens": 5},
            assistant_message=SimpleNamespace(content="ok", tool_calls=None),
        )
        plugin._client.save_llm_trace.assert_not_called()


class ReasoningExtractionHelperTests(unittest.TestCase):
    """Unit tests for the module-level helpers directly, independent of the
    plugin/hook machinery above."""

    def test_none_assistant_message_returns_none(self) -> None:
        self.assertIsNone(_extract_reasoning_from_assistant_message(None))

    def test_no_reasoning_fields_returns_none(self) -> None:
        msg = SimpleNamespace(content="just an answer")
        self.assertIsNone(_extract_reasoning_from_assistant_message(msg))

    def test_reasoning_content_field(self) -> None:
        msg = SimpleNamespace(reasoning_content="step by step thinking")
        self.assertEqual(_extract_reasoning_from_assistant_message(msg), "step by step thinking")

    def test_reasoning_details_array_format(self) -> None:
        msg = SimpleNamespace(
            reasoning_details=[{"type": "reasoning.summary", "summary": "considered options"}]
        )
        self.assertEqual(_extract_reasoning_from_assistant_message(msg), "considered options")

    def test_summarize_answer_prefers_content(self) -> None:
        msg = SimpleNamespace(content="the final answer", tool_calls=[SimpleNamespace()])
        self.assertEqual(_summarize_assistant_answer(msg), "the final answer")

    def test_summarize_answer_empty_when_nothing_available(self) -> None:
        msg = SimpleNamespace(content="", tool_calls=None)
        self.assertEqual(_summarize_assistant_answer(msg), "")

    def test_summarize_answer_none_message(self) -> None:
        self.assertEqual(_summarize_assistant_answer(None), "")


# ---------------------------------------------------------------------------
# HITL resolutions: modify (A2) and cancelled (B2 forward-compat, contract K1)
# ---------------------------------------------------------------------------

class HitlResolutionTests(unittest.TestCase):
    def _run(self, decisions, **config):
        plugin = make_plugin(**config)
        plugin._client.request_hitl = Mock(return_value=pending("w1"))
        plugin._client.get_hitl_decision = Mock(side_effect=[BASELINE, *decisions])
        clock = FakeClock()
        with quiet(plugin), clock.installed():
            result = plugin.pre_tool_call(tool_name="terminal", args={"command": "rm -rf /x"}, session_id="s1")
        return plugin, result

    def test_modify_resolution_blocks_tool_and_never_returns_none(self) -> None:
        plugin, result = self._run([
            complete("modify", "w1", modified_action={"params": {"command": "ls /safe"}}),
        ])
        assert result is not None
        self.assertEqual(result["action"], "block")
        self.assertIn("modified version", result["message"])
        plugin._notify_user.assert_any_call(result["message"])

    def test_modified_spelling_also_blocks(self) -> None:
        _, result = self._run([complete("modified", "w1")])
        assert result is not None
        self.assertEqual(result["action"], "block")

    def test_approve_still_allows(self) -> None:
        _, result = self._run([complete("approved", "w1")])
        self.assertIsNone(result)

    def test_cancelled_status_blocks_immediately(self) -> None:
        # Variant (b) shape: {status: cancelled}
        plugin, result = self._run([{"status": "cancelled", "workitem_id": "w1"}])
        assert result is not None
        self.assertEqual(result["action"], "block")
        self.assertIn("cancelled", result["message"])
        self.assertEqual(plugin._client.get_hitl_decision.call_count, 2)  # baseline + 1 poll

    def test_cancelled_method_says_cancelled_not_rejected(self) -> None:
        # Contract K1 shape.
        _, result = self._run([{
            "status": "complete", "resolution": "reject", "method": "cancelled",
            "workitem_status": "Cancelled", "workitem_id": "w1", "information": "workitem cancelled",
        }])
        assert result is not None
        self.assertEqual(result["action"], "block")
        self.assertIn("cancelled", result["message"])
        self.assertNotIn("rejected by AMP HITL review", result["message"])
        self.assertIn("not rejected by a reviewer", result["message"])

    def test_pending_body_with_cancelled_workitem_status_blocks(self) -> None:
        _, result = self._run([pending("w1", workitem_status="Cancelled")])
        assert result is not None
        self.assertIn("cancelled", result["message"])

    def test_plain_pending_keeps_polling(self) -> None:
        plugin, result = self._run([pending("w1"), pending("w1"), complete("approve", "w1")])
        self.assertIsNone(result)
        self.assertEqual(plugin._client.get_hitl_decision.call_count, 4)  # baseline + 3 polls

    def test_unknown_resolution_blocks(self) -> None:
        _, result = self._run([complete("defer", "w1")])
        assert result is not None
        self.assertEqual(result["action"], "block")


# ---------------------------------------------------------------------------
# Poll failures (A4)
# ---------------------------------------------------------------------------

class PollFailureTests(unittest.TestCase):
    def _run(self, poll_side_effect, **config):
        plugin = make_plugin(**config)
        plugin._client.request_hitl = Mock(return_value=pending("w1"))
        plugin._client.get_hitl_decision = Mock(side_effect=[BASELINE, *poll_side_effect])
        clock = FakeClock()
        with quiet(plugin), clock.installed():
            result = plugin.pre_tool_call(tool_name="terminal", args={"command": "ls"}, session_id="s1")
        return plugin, result, clock

    def test_poll_http_429_blocks_in_one_poll(self) -> None:
        plugin, result, clock = self._run([AmpClientError(
            "HTTP 429: limit", status_code=429, body='{"error":"call_limit_reached"}',
        )])
        assert result is not None
        self.assertEqual(result["action"], "block")
        self.assertIn("HTTP 429", result["message"])
        self.assertIn("call_limit_reached", result["message"])
        self.assertEqual(plugin._client.get_hitl_decision.call_count, 2)  # baseline + 1 poll
        self.assertLess(clock.elapsed, 5)

    def test_poll_http_401_403_404_block_in_one_poll(self) -> None:
        for code in (401, 403, 404):
            with self.subTest(code=code):
                plugin, result, _ = self._run([AmpClientError(f"HTTP {code}", status_code=code)])
                assert result is not None
                self.assertIn(f"HTTP {code}", result["message"])
                self.assertEqual(plugin._client.get_hitl_decision.call_count, 2)  # baseline + 1 poll

    def test_poll_http_429_blocks_even_when_fail_open(self) -> None:
        _, result, _ = self._run([AmpClientError("HTTP 429", status_code=429)], fail_closed=False)
        assert result is not None
        self.assertEqual(result["action"], "block")

    def test_poll_http_500_keeps_polling(self) -> None:
        plugin, result, _ = self._run([
            AmpClientError("HTTP 500", status_code=500),
            AmpClientError("HTTP 500", status_code=500),
            complete("approve", "w1"),
        ])
        self.assertIsNone(result)
        self.assertEqual(plugin._client.get_hitl_decision.call_count, 4)  # baseline + 3 polls

    def test_poll_unexpected_exception_keeps_polling_never_allows(self) -> None:
        plugin, result, _ = self._run([RuntimeError("boom"), complete("reject", "w1")], fail_closed=False)
        assert result is not None
        self.assertEqual(result["action"], "block")
        self.assertEqual(plugin._client.get_hitl_decision.call_count, 3)  # baseline + 2 polls

    def test_amp_client_error_carries_status_code(self) -> None:
        from io import BytesIO
        from urllib.error import HTTPError

        client = AmpClient(SimpleNamespace(backend_url="http://amp.invalid", api_key="k"))
        error = HTTPError("http://amp.invalid/x", 403, "Forbidden", {}, BytesIO(b'{"error":"forbidden"}'))
        with patch("hermes.amp_client.urlopen", side_effect=error):
            with self.assertRaises(AmpClientError) as ctx:
                client.get_hitl_decision("inst-1")
        self.assertEqual(ctx.exception.status_code, 403)
        self.assertEqual(ctx.exception.body, '{"error":"forbidden"}')

    def test_socket_timeout_becomes_amp_client_error(self) -> None:
        client = AmpClient(SimpleNamespace(backend_url="http://amp.invalid", api_key="k"))
        with patch("hermes.amp_client.urlopen", side_effect=TimeoutError("timed out")):
            with self.assertRaises(AmpClientError) as ctx:
                client.get_hitl_decision("inst-1")
        self.assertIsNone(ctx.exception.status_code)


if __name__ == "__main__":
    unittest.main()
