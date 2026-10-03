"""Tests for Phase 2B: runtime LLM token and cost enforcement.

Covers:
  - Capability detection (LLMExecutionBlocked available / unavailable)
  - Enforcement configuration gating
  - Policy outcomes (allow, block, HITL pending→approve/reject/timeout)
  - Fail-open and fail-closed behavior on AMP unavailability
  - Provider-call behavior (call count assertions)
  - Multi-turn and context behavior
  - Regression: existing tool governance, notification, and Phase 2A tests
"""

from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, call, patch

ROOT = Path(__file__).resolve().parent.parent
ROOT_PARENT = ROOT.parent
# Add Hermes repo to path so hermes_cli.middleware (LLMExecutionBlocked) is importable
HERMES_AGENT_ROOT = Path.home() / ".hermes" / "hermes-agent"

if str(ROOT_PARENT) not in sys.path:
    sys.path.insert(0, str(ROOT_PARENT))
if HERMES_AGENT_ROOT.exists() and str(HERMES_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(HERMES_AGENT_ROOT))

import hermes as _plugin_mod
from hermes import AmpGovernancePlugin, _estimate_pending_call_cost, _worst_cost_status
from hermes.execution_context import ExecutionContext, ExecutionContextStore


# Released Hermes (v2026.8/v2026.9) does not ship hermes_cli.middleware.LLMExecutionBlocked
# (upstream PR nousresearch/hermes-agent#64662 is unmerged). Use the real class when present,
# otherwise a stand-in with the same constructor/attributes so the enforcement logic still runs.
try:
    from hermes_cli.middleware import LLMExecutionBlocked as _LEB
    _REAL_LEB_AVAILABLE = True
except ImportError:
    _REAL_LEB_AVAILABLE = False

    class _LEB(Exception):  # type: ignore[no-redef]
        def __init__(self, reason, metadata=None):
            super().__init__(reason)
            self.reason = reason
            self.metadata = metadata or {}


def _use_llm_blocked(test: unittest.TestCase):
    """Patch the plugin module so enforcement raises _LEB for the duration of `test`."""
    for name, value in (("_LLMExecutionBlocked", _LEB), ("_LLM_BLOCKED_AVAILABLE", True)):
        patcher = patch.object(_plugin_mod, name, value)
        patcher.start()
        test.addCleanup(patcher.stop)
    return _LEB


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_plugin(*, llm_enabled=True, mode="enforce", fail_closed=False) -> AmpGovernancePlugin:
    plugin = AmpGovernancePlugin()
    plugin._config = SimpleNamespace(
        is_configured=True,
        notifications_enabled=True,
        llm_governance_enabled=llm_enabled,
        llm_governance_mode=mode,
        llm_governance_fail_closed=fail_closed,
        llm_governance_include_subagents=True,
        fail_closed=True,
        hitl_timeout_minutes=1,
        hitl_poll_interval_seconds=0,
        username="tester",
        agent_name="hermes-test",
        org_id="O-test",
    )
    return plugin


def _make_exec_ctx(
    session_id="sess-1",
    instance_id="inst-1",
    *,
    total_cost_usd=0.0,
    cost_status="unknown",
    llm_calls=0,
) -> ExecutionContext:
    ctx = ExecutionContext(session_id, instance_id, model="claude-opus-4-8", platform="slack")
    ctx.total_cost_usd = total_cost_usd
    ctx.cost_status = cost_status
    ctx.llm_calls = llm_calls
    return ctx


def _make_next_call(return_value="response") -> Mock:
    return Mock(return_value=return_value)


def _allow_response():
    return {"status": "no-hitl"}


def _block_response():
    return {"status": "no_policy"}


def _hitl_response(workitem_id="w-1"):
    return {"status": "pending", "workitem_id": workitem_id}


def _decision(resolution="approved", info=""):
    # AMP always echoes the workitem_id; the plugin only accepts a decision for the
    # workitem its request is bound to (stale-decision guard).
    return {"status": "complete", "resolution": resolution, "information": info, "workitem_id": "w-1"}


# ---------------------------------------------------------------------------
# 1. Capability detection
# ---------------------------------------------------------------------------

class CapabilityDetectionTests(unittest.TestCase):
    @unittest.skipUnless(_REAL_LEB_AVAILABLE, "needs hermes-agent#64662 (LLMExecutionBlocked)")
    def test_blocked_available_import_succeeds(self):
        """When Hermes has LLMExecutionBlocked, the flag is True."""
        import importlib
        import hermes as mod
        self.assertTrue(mod._LLM_BLOCKED_AVAILABLE)
        self.assertIsNotNone(mod._LLMExecutionBlocked)

    def test_blocked_unavailable_falls_back_gracefully(self):
        """When hermes_cli.middleware lacks LLMExecutionBlocked, flag is False and
        module still loads without error."""
        # Simulate missing attribute by patching the module's cached value
        import hermes as mod
        original_available = mod._LLM_BLOCKED_AVAILABLE
        original_cls = mod._LLMExecutionBlocked
        try:
            mod._LLM_BLOCKED_AVAILABLE = False
            mod._LLMExecutionBlocked = None
            # Plugin can be instantiated safely
            plugin = AmpGovernancePlugin()
            self.assertIsNotNone(plugin)
        finally:
            mod._LLM_BLOCKED_AVAILABLE = original_available
            mod._LLMExecutionBlocked = original_cls

    def test_enforcement_configured_but_unavailable_logs_error(self):
        """When mode=enforce but LLMExecutionBlocked is absent, register() logs an error."""
        import hermes as mod
        plugin = _make_plugin(llm_enabled=True, mode="enforce")

        ctx = SimpleNamespace(
            dispatch_tool=Mock(),
            register_hook=Mock(),
            register_middleware=Mock(),
            register_tool=Mock(),
            register_skill=Mock(),
            register_cli_command=Mock(),
        )

        with patch.object(mod, "_LLM_BLOCKED_AVAILABLE", False), \
             patch.object(mod, "_PLUGIN", plugin), \
             patch("logging.Logger.error") as mock_error:
            mod.register(ctx)

        # Middleware was NOT registered
        ctx.register_middleware.assert_not_called()
        # Error was logged
        mock_error.assert_called()
        logged_msg = mock_error.call_args[0][0]
        self.assertIn("enforce", logged_msg.lower())

    def test_observation_continues_when_enforcement_unavailable(self):
        """When blocked is unavailable but mode=observe, post_api_request still accumulates."""
        import hermes as mod
        plugin = _make_plugin(llm_enabled=True, mode="observe")

        exec_ctx = _make_exec_ctx()
        plugin._exec_contexts._store["sess-1"] = exec_ctx
        plugin._client = Mock()

        with patch.object(mod, "_LLM_BLOCKED_AVAILABLE", False):
            plugin.post_api_request(
                session_id="sess-1",
                api_request_id="r1",
                model="claude-opus-4-8",
                provider="anthropic",
                base_url="",
                api_call_count=1,
                api_duration=1.0,
                usage={"input_tokens": 100, "output_tokens": 50},
            )

        self.assertEqual(exec_ctx.llm_calls, 1)
        self.assertEqual(exec_ctx.input_tokens, 100)


# ---------------------------------------------------------------------------
# 2. Configuration gating
# ---------------------------------------------------------------------------

class ConfigurationGatingTests(unittest.TestCase):
    def test_middleware_noop_when_llm_governance_disabled(self):
        """llm_execution_middleware passes through immediately when disabled."""
        plugin = _make_plugin(llm_enabled=False, mode="enforce")
        next_call = _make_next_call("resp")
        result = plugin.llm_execution_middleware(
            request={}, next_call=next_call, session_id="s", model="m"
        )
        self.assertEqual(result, "resp")
        next_call.assert_called_once_with({})

    def test_middleware_noop_in_observe_mode(self):
        """llm_execution_middleware passes through when mode=observe."""
        plugin = _make_plugin(llm_enabled=True, mode="observe")
        next_call = _make_next_call("resp")
        result = plugin.llm_execution_middleware(
            request={}, next_call=next_call, session_id="s", model="m"
        )
        self.assertEqual(result, "resp")
        next_call.assert_called_once_with({})

    def test_middleware_noop_when_no_exec_context(self):
        """llm_execution_middleware allows the call if no ExecutionContext is tracked."""
        plugin = _make_plugin(llm_enabled=True, mode="enforce")
        next_call = _make_next_call("resp")
        result = plugin.llm_execution_middleware(
            request={}, next_call=next_call, session_id="no-such-session", model="m"
        )
        self.assertEqual(result, "resp")
        next_call.assert_called_once_with({})

    def test_enforce_mode_evaluates_policy(self):
        """When enforce is enabled, policy is evaluated before next_call."""
        plugin = _make_plugin(llm_enabled=True, mode="enforce")
        exec_ctx = _make_exec_ctx()
        plugin._exec_contexts._store["sess-1"] = exec_ctx

        with patch.object(plugin, "_evaluate_llm_governance", return_value=("allow", "")) as mock_eval, \
             patch.object(plugin, "_safe_log"):
            next_call = _make_next_call("response")
            plugin.llm_execution_middleware(
                request={"messages": []}, next_call=next_call,
                session_id="sess-1", model="m", provider="p", base_url="",
                api_call_count=1,
            )

        mock_eval.assert_called_once()
        next_call.assert_called_once()

    def test_invalid_mode_treated_as_observe(self):
        """AmpConfig.llm_governance_mode is validated to 'observe' for unknown values."""
        from hermes.config import load_config
        import os, tempfile
        from pathlib import Path as P
        with tempfile.TemporaryDirectory() as tmp:
            (P(tmp) / ".env").write_text(
                "AMP_BACKEND_URL=https://amp.example.com\n"
                "AMP_API_KEY=k\nAMP_ORG_ID=O\nAMP_USERNAME=u\nAMP_AGENT_NAME=a\n"
                "AMP_LLM_GOVERNANCE_MODE=invalid\n"
            )
            with patch.dict(os.environ, {"HERMES_HOME": tmp}):
                cfg = load_config()
        self.assertEqual(cfg.llm_governance_mode, "observe")


# ---------------------------------------------------------------------------
# 3. Policy outcomes
# ---------------------------------------------------------------------------

class PolicyOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.LEB = _use_llm_blocked(self)

    def _run_middleware(self, plugin, amp_response, decision_response=None):
        """Helper: run llm_execution_middleware with mocked AMP responses."""
        exec_ctx = _make_exec_ctx()
        plugin._exec_contexts._store["sess-1"] = exec_ctx

        plugin._client = Mock()
        plugin._client.request_llm_hitl = Mock(return_value=amp_response)
        if decision_response is not None:
            plugin._client.get_hitl_decision = Mock(return_value=decision_response)

        plugin._notify_user = Mock()
        next_call = _make_next_call("provider-response")

        with patch.object(plugin, "_safe_log"), \
             patch("time.sleep", return_value=None):
            try:
                result = plugin.llm_execution_middleware(
                    request={"messages": []}, next_call=next_call,
                    session_id="sess-1", model="claude-opus-4-8",
                    provider="anthropic", base_url="",
                    api_call_count=1,
                )
            except self.LEB as exc:
                return "blocked", exc, next_call
            return "allowed", result, next_call

    def test_allow_calls_provider(self):
        plugin = _make_plugin()
        outcome, result, next_call = self._run_middleware(plugin, _allow_response())
        self.assertEqual(outcome, "allowed")
        self.assertEqual(result, "provider-response")
        next_call.assert_called_once()

    def test_immediate_block_raises_leb_and_skips_provider(self):
        plugin = _make_plugin()
        outcome, exc, next_call = self._run_middleware(plugin, _block_response())
        self.assertEqual(outcome, "blocked")
        self.assertIsInstance(exc, self.LEB)
        self.assertIn("policy", exc.reason.lower())
        next_call.assert_not_called()

    def test_hitl_approve_calls_provider_exactly_once(self):
        plugin = _make_plugin()
        outcome, result, next_call = self._run_middleware(
            plugin, _hitl_response(), _decision("approved")
        )
        self.assertEqual(outcome, "allowed")
        next_call.assert_called_once()

    def test_hitl_reject_raises_leb_and_skips_provider(self):
        plugin = _make_plugin()
        outcome, exc, next_call = self._run_middleware(
            plugin, _hitl_response(), _decision("rejected", "budget exceeded")
        )
        self.assertEqual(outcome, "blocked")
        self.assertIsInstance(exc, self.LEB)
        next_call.assert_not_called()

    def test_hitl_timeout_raises_leb_and_skips_provider(self):
        plugin = _make_plugin()
        exec_ctx = _make_exec_ctx()
        plugin._exec_contexts._store["sess-1"] = exec_ctx
        plugin._config = SimpleNamespace(
            **{**vars(plugin._config),
               "hitl_timeout_minutes": 0,  # immediate
               "hitl_poll_interval_seconds": 0,
               }
        )
        plugin._client = Mock()
        plugin._client.request_llm_hitl = Mock(return_value=_hitl_response())
        plugin._client.get_hitl_decision = Mock(return_value={"status": "pending"})
        plugin._notify_user = Mock()
        next_call = _make_next_call()

        with patch.object(plugin, "_safe_log"), patch("time.sleep", return_value=None):
            try:
                plugin.llm_execution_middleware(
                    request={"messages": []}, next_call=next_call,
                    session_id="sess-1", model="m", provider="p", base_url="",
                    api_call_count=1,
                )
                outcome = "allowed"
            except self.LEB:
                outcome = "blocked"

        self.assertEqual(outcome, "blocked")
        next_call.assert_not_called()

    def test_amp_unavailable_fail_open_allows_provider(self):
        plugin = _make_plugin(fail_closed=False)
        exec_ctx = _make_exec_ctx()
        plugin._exec_contexts._store["sess-1"] = exec_ctx
        plugin._client = Mock()
        plugin._client.request_llm_hitl = Mock(side_effect=Exception("connection refused"))
        plugin._notify_user = Mock()
        next_call = _make_next_call("resp")

        with patch.object(plugin, "_safe_log"):
            result = plugin.llm_execution_middleware(
                request={"messages": []}, next_call=next_call,
                session_id="sess-1", model="m", provider="p", base_url="",
                api_call_count=1,
            )

        self.assertEqual(result, "resp")
        next_call.assert_called_once()

    def test_amp_unavailable_fail_closed_raises_leb(self):
        plugin = _make_plugin(fail_closed=True)
        exec_ctx = _make_exec_ctx()
        plugin._exec_contexts._store["sess-1"] = exec_ctx
        plugin._client = Mock()
        plugin._client.request_llm_hitl = Mock(side_effect=Exception("timeout"))
        plugin._notify_user = Mock()
        next_call = _make_next_call()

        with patch.object(plugin, "_safe_log"):
            with self.assertRaises(self.LEB):
                plugin.llm_execution_middleware(
                    request={"messages": []}, next_call=next_call,
                    session_id="sess-1", model="m", provider="p", base_url="",
                    api_call_count=1,
                )

        next_call.assert_not_called()

    def test_unexpected_amp_status_blocks_even_when_fail_open(self):
        # An unknown status (e.g. a reserved "deny") is an AMP answer, not an outage.
        plugin = _make_plugin(fail_closed=False)
        exec_ctx = _make_exec_ctx()
        plugin._exec_contexts._store["sess-1"] = exec_ctx
        plugin._client = Mock()
        plugin._client.request_llm_hitl = Mock(return_value={"status": "weird_status"})
        plugin._notify_user = Mock()
        next_call = _make_next_call("resp")

        with patch.object(plugin, "_safe_log"), self.assertRaises(self.LEB):
            plugin.llm_execution_middleware(
                request={"messages": []}, next_call=next_call,
                session_id="sess-1", model="m", provider="p", base_url="",
                api_call_count=1,
            )

        next_call.assert_not_called()

    def test_unexpected_amp_status_fail_closed_blocks(self):
        plugin = _make_plugin(fail_closed=True)
        exec_ctx = _make_exec_ctx()
        plugin._exec_contexts._store["sess-1"] = exec_ctx
        plugin._client = Mock()
        plugin._client.request_llm_hitl = Mock(return_value={"status": "weird_status"})
        plugin._notify_user = Mock()
        next_call = _make_next_call()

        with patch.object(plugin, "_safe_log"):
            with self.assertRaises(self.LEB):
                plugin.llm_execution_middleware(
                    request={"messages": []}, next_call=next_call,
                    session_id="sess-1", model="m", provider="p", base_url="",
                    api_call_count=1,
                )

        next_call.assert_not_called()


# ---------------------------------------------------------------------------
# 4. Provider-call behavior
# ---------------------------------------------------------------------------

class LlmFailClosedTests(unittest.TestCase):
    """A1 (LLM half), A2, A4, K1 and H2 on the LLM middleware path."""

    def setUp(self):
        self.LEB = _use_llm_blocked(self)

    def _run(self, plugin, amp_response=None, decisions=None, request_error=None):
        exec_ctx = _make_exec_ctx()
        plugin._exec_contexts._store["sess-1"] = exec_ctx
        plugin._client = Mock()
        if request_error is not None:
            plugin._client.request_llm_hitl = Mock(side_effect=request_error)
        else:
            plugin._client.request_llm_hitl = Mock(return_value=amp_response)
        if decisions is not None:
            # K2.1: one baseline get-decision precedes the request.
            plugin._client.get_hitl_decision = Mock(side_effect=[{"status": "pending"}, *decisions])
        plugin._notify_user = Mock()
        with patch.object(plugin, "_safe_log"), patch("time.sleep", return_value=None):
            return plugin._evaluate_llm_governance(
                exec_ctx, "m", "p", "", "sess-1", {"messages": []}, 1
            )

    def test_policy_error_blocks_llm_call_even_when_fail_open(self):
        plugin = _make_plugin(fail_closed=False)
        decision = self._run(plugin, {"status": "policy_error", "message": "bad route"})
        self.assertEqual(decision, ("block", "bad route"))

    def test_config_error_without_message_blocks(self):
        plugin = _make_plugin(fail_closed=False)
        decision, reason = self._run(plugin, {"status": "config_error"})
        self.assertEqual(decision, "block")
        self.assertTrue(reason)

    def test_hitl_modify_blocks_llm_call(self):
        plugin = _make_plugin()
        decision, reason = self._run(plugin, _hitl_response(), [_decision("modify")])
        self.assertEqual(decision, "block")
        self.assertIn("reviewer requested modifications", reason)

    def test_cancelled_decision_blocks_llm_call(self):
        plugin = _make_plugin()
        decision, reason = self._run(
            plugin, _hitl_response(),
            [{"status": "complete", "resolution": "reject", "method": "cancelled", "workitem_id": "w-1"}],
        )
        self.assertEqual(decision, "block")
        self.assertIn("cancelled", reason)

    def test_poll_403_blocks_llm_call_in_one_poll(self):
        from hermes.amp_client import AmpClientError

        plugin = _make_plugin()
        decision, reason = self._run(
            plugin, _hitl_response(), [AmpClientError("HTTP 403", status_code=403, body="forbidden")]
        )
        self.assertEqual(decision, "block")
        self.assertIn("HTTP 403", reason)
        self.assertEqual(plugin._client.get_hitl_decision.call_count, 2)  # baseline + 1 poll

    def test_request_4xx_blocks_llm_call_even_when_fail_open(self):
        from hermes.amp_client import AmpClientError

        plugin = _make_plugin(fail_closed=False)
        decision, reason = self._run(
            plugin, request_error=AmpClientError("HTTP 429", status_code=429, body="call_limit_reached")
        )
        self.assertEqual(decision, "block")
        self.assertIn("HTTP 429", reason)

    def test_stale_decision_for_other_workitem_is_ignored(self):
        plugin = _make_plugin()
        decision, _ = self._run(
            plugin, _hitl_response("w-2"),
            [{"status": "complete", "resolution": "approve", "workitem_id": "w-OLD"},
             {"status": "complete", "resolution": "reject", "workitem_id": "w-2"}],
        )
        self.assertEqual(decision, "block")
        self.assertEqual(plugin._client.get_hitl_decision.call_count, 3)  # baseline + 2 polls

    def test_reviewer_note_is_labelled_on_reject(self):
        plugin = _make_plugin()
        _, reason = self._run(plugin, _hitl_response(), [_decision("reject", "too costly")])
        self.assertIn("Reviewer note: too costly", reason)


class ProviderCallBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.LEB = _use_llm_blocked(self)

    def test_allow_calls_provider_exactly_once(self):
        plugin = _make_plugin()
        exec_ctx = _make_exec_ctx()
        plugin._exec_contexts._store["s"] = exec_ctx
        plugin._client = Mock()
        plugin._client.request_llm_hitl = Mock(return_value=_allow_response())
        plugin._notify_user = Mock()
        next_call = _make_next_call()

        with patch.object(plugin, "_safe_log"):
            plugin.llm_execution_middleware(
                request={"messages": []}, next_call=next_call,
                session_id="s", model="m", provider="p", base_url="", api_call_count=1,
            )

        next_call.assert_called_once()

    def test_block_never_calls_provider(self):
        plugin = _make_plugin()
        exec_ctx = _make_exec_ctx()
        plugin._exec_contexts._store["s"] = exec_ctx
        plugin._client = Mock()
        plugin._client.request_llm_hitl = Mock(return_value=_block_response())
        plugin._notify_user = Mock()
        next_call = _make_next_call()

        with patch.object(plugin, "_safe_log"):
            with self.assertRaises(self.LEB):
                plugin.llm_execution_middleware(
                    request={"messages": []}, next_call=next_call,
                    session_id="s", model="m", provider="p", base_url="", api_call_count=1,
                )

        next_call.assert_not_called()

    def test_rejected_hitl_never_calls_provider(self):
        plugin = _make_plugin()
        exec_ctx = _make_exec_ctx()
        plugin._exec_contexts._store["s"] = exec_ctx
        plugin._client = Mock()
        plugin._client.request_llm_hitl = Mock(return_value=_hitl_response())
        plugin._client.get_hitl_decision = Mock(return_value=_decision("rejected"))
        plugin._notify_user = Mock()
        next_call = _make_next_call()

        with patch.object(plugin, "_safe_log"), patch("time.sleep", return_value=None):
            with self.assertRaises(self.LEB):
                plugin.llm_execution_middleware(
                    request={"messages": []}, next_call=next_call,
                    session_id="s", model="m", provider="p", base_url="", api_call_count=1,
                )

        next_call.assert_not_called()

    def test_approved_hitl_calls_provider_exactly_once(self):
        plugin = _make_plugin()
        exec_ctx = _make_exec_ctx()
        plugin._exec_contexts._store["s"] = exec_ctx
        plugin._client = Mock()
        plugin._client.request_llm_hitl = Mock(return_value=_hitl_response())
        plugin._client.get_hitl_decision = Mock(return_value=_decision("approved"))
        plugin._notify_user = Mock()
        next_call = _make_next_call()

        with patch.object(plugin, "_safe_log"), patch("time.sleep", return_value=None):
            plugin.llm_execution_middleware(
                request={"messages": []}, next_call=next_call,
                session_id="s", model="m", provider="p", base_url="", api_call_count=1,
            )

        next_call.assert_called_once()

    def test_provider_exception_propagates_normally(self):
        """LLMExecutionBlocked is NOT raised when the provider itself errors."""
        plugin = _make_plugin()
        exec_ctx = _make_exec_ctx()
        plugin._exec_contexts._store["s"] = exec_ctx
        plugin._client = Mock()
        plugin._client.request_llm_hitl = Mock(return_value=_allow_response())
        plugin._notify_user = Mock()

        def failing_provider(req):
            raise ConnectionError("provider down")

        with patch.object(plugin, "_safe_log"):
            with self.assertRaises(ConnectionError):
                plugin.llm_execution_middleware(
                    request={"messages": []}, next_call=failing_provider,
                    session_id="s", model="m", provider="p", base_url="", api_call_count=1,
                )

    def test_phase2a_observation_still_runs_after_allow(self):
        """post_api_request accumulation is unaffected by enforcement."""
        plugin = _make_plugin(llm_enabled=True, mode="enforce")
        exec_ctx = _make_exec_ctx()
        plugin._exec_contexts._store["s"] = exec_ctx
        plugin._client = Mock()

        # Phase 2B: allow
        plugin._client.request_llm_hitl = Mock(return_value=_allow_response())
        plugin._notify_user = Mock()
        next_call = _make_next_call()

        with patch.object(plugin, "_safe_log"):
            plugin.llm_execution_middleware(
                request={"messages": []}, next_call=next_call,
                session_id="s", model="claude-opus-4-8", provider="anthropic",
                base_url="", api_call_count=1,
            )

        # Phase 2A: observe
        plugin.post_api_request(
            session_id="s",
            api_request_id="r1",
            model="claude-opus-4-8",
            provider="anthropic",
            base_url="",
            api_call_count=1,
            api_duration=1.0,
            usage={"input_tokens": 200, "output_tokens": 80},
        )

        self.assertEqual(exec_ctx.llm_calls, 1)
        self.assertEqual(exec_ctx.input_tokens, 200)


# ---------------------------------------------------------------------------
# 5. Multi-turn and context behavior
# ---------------------------------------------------------------------------

class MultiTurnContextTests(unittest.TestCase):
    def setUp(self):
        self.LEB = _use_llm_blocked(self)

    def test_three_allowed_calls_accumulate_correctly(self):
        plugin = _make_plugin(llm_enabled=True, mode="observe")
        exec_ctx = _make_exec_ctx()
        plugin._exec_contexts._store["s"] = exec_ctx
        plugin._client = Mock()

        for i in range(3):
            plugin.post_api_request(
                session_id="s",
                api_request_id=f"r{i}",
                model="claude-opus-4-8",
                provider="anthropic",
                base_url="",
                api_call_count=i + 1,
                api_duration=0.5,
                usage={"input_tokens": 100, "output_tokens": 50},
            )

        self.assertEqual(exec_ctx.llm_calls, 3)
        self.assertEqual(exec_ctx.input_tokens, 300)
        self.assertEqual(exec_ctx.output_tokens, 150)

    def test_later_call_blocked_after_earlier_calls_allowed(self):
        """Simulate: calls 1 and 2 allowed (observe), call 3 blocked (enforce)."""
        plugin = _make_plugin(llm_enabled=True, mode="enforce")
        exec_ctx = _make_exec_ctx()
        plugin._exec_contexts._store["s"] = exec_ctx
        plugin._client = Mock()
        plugin._notify_user = Mock()

        # Accumulate 2 calls via post_api_request
        for i in range(2):
            plugin.post_api_request(
                session_id="s",
                api_request_id=f"r{i}",
                model="m", provider="p", base_url="",
                api_call_count=i + 1, api_duration=0.5,
                usage={"input_tokens": 100, "output_tokens": 50},
            )

        # 3rd call: block
        plugin._client.request_llm_hitl = Mock(return_value=_block_response())
        next_call = _make_next_call()

        with patch.object(plugin, "_safe_log"):
            with self.assertRaises(self.LEB):
                plugin.llm_execution_middleware(
                    request={"messages": []}, next_call=next_call,
                    session_id="s", model="m", provider="p", base_url="", api_call_count=3,
                )

        self.assertEqual(exec_ctx.llm_calls, 2)  # 3rd call was blocked before accumulation
        next_call.assert_not_called()

    def test_concurrent_sessions_remain_isolated(self):
        """Two sessions running concurrently do not share ExecutionContext."""
        plugin = _make_plugin(llm_enabled=True, mode="observe")
        plugin._client = Mock()

        ctx_a = ExecutionContext("sess-a", "inst-a", model="m", platform="")
        ctx_b = ExecutionContext("sess-b", "inst-b", model="m", platform="")
        plugin._exec_contexts._store["sess-a"] = ctx_a
        plugin._exec_contexts._store["sess-b"] = ctx_b

        errors = []

        def accumulate_session(session_id, n):
            for i in range(n):
                try:
                    plugin.post_api_request(
                        session_id=session_id,
                        api_request_id=f"{session_id}-r{i}",
                        model="m", provider="p", base_url="",
                        api_call_count=i + 1, api_duration=0.1,
                        usage={"input_tokens": 10, "output_tokens": 5},
                    )
                except Exception as e:
                    errors.append(e)

        t1 = threading.Thread(target=accumulate_session, args=("sess-a", 5))
        t2 = threading.Thread(target=accumulate_session, args=("sess-b", 3))
        t1.start(); t2.start()
        t1.join(); t2.join()

        self.assertEqual(errors, [])
        self.assertEqual(ctx_a.llm_calls, 5)
        self.assertEqual(ctx_b.llm_calls, 3)
        self.assertEqual(ctx_a.input_tokens, 50)
        self.assertEqual(ctx_b.input_tokens, 30)

    def test_status_transitions_allow(self):
        plugin = _make_plugin()
        exec_ctx = _make_exec_ctx()
        plugin._exec_contexts._store["s"] = exec_ctx
        plugin._client = Mock()
        plugin._client.request_llm_hitl = Mock(return_value=_allow_response())
        plugin._notify_user = Mock()
        next_call = _make_next_call()

        with patch.object(plugin, "_safe_log"):
            plugin.llm_execution_middleware(
                request={"messages": []}, next_call=next_call,
                session_id="s", model="m", provider="p", base_url="", api_call_count=1,
            )

        self.assertEqual(exec_ctx.status, "running")

    def test_status_transitions_hitl_approve(self):
        plugin = _make_plugin()
        exec_ctx = _make_exec_ctx()
        plugin._exec_contexts._store["s"] = exec_ctx
        plugin._client = Mock()
        plugin._client.request_llm_hitl = Mock(return_value=_hitl_response())
        plugin._client.get_hitl_decision = Mock(return_value=_decision("approved"))
        plugin._notify_user = Mock()
        next_call = _make_next_call()

        with patch.object(plugin, "_safe_log"), patch("time.sleep", return_value=None):
            plugin.llm_execution_middleware(
                request={"messages": []}, next_call=next_call,
                session_id="s", model="m", provider="p", base_url="", api_call_count=1,
            )

        # Status should transition back to running after approval
        self.assertEqual(exec_ctx.status, "running")
        self.assertEqual(exec_ctx.llm_hitl_approved, 1)

    def test_context_cleanup_on_finalize(self):
        plugin = _make_plugin(llm_enabled=True, mode="observe")
        exec_ctx = _make_exec_ctx()
        plugin._exec_contexts._store["s"] = exec_ctx
        plugin._client = Mock()
        plugin._store = Mock()
        plugin._store.get = Mock(return_value=SimpleNamespace(instance_id="inst-1"))
        plugin._store.delete = Mock()
        plugin._client.log_execution_summary = Mock()
        plugin._client.set_state = Mock()

        with patch.object(plugin, "_safe_log"):
            plugin.on_session_finalize(session_id="s", reason="complete")

        self.assertIsNone(plugin._exec_contexts.get("s"))

    def test_phase2a_observation_still_accumulates(self):
        """Regression: Phase 2A accumulation tests still pass with Phase 2B code."""
        plugin = _make_plugin(llm_enabled=True, mode="observe")
        exec_ctx = _make_exec_ctx()
        plugin._exec_contexts._store["s"] = exec_ctx
        plugin._client = Mock()

        plugin.post_api_request(
            session_id="s",
            api_request_id="r1",
            model="claude-opus-4-8",
            provider="anthropic",
            base_url="",
            api_call_count=1,
            api_duration=2.0,
            usage={
                "input_tokens": 500,
                "output_tokens": 200,
                "cache_read_tokens": 1000,
                "cache_write_tokens": 50,
                "reasoning_tokens": 30,
            },
        )

        self.assertEqual(exec_ctx.llm_calls, 1)
        self.assertEqual(exec_ctx.input_tokens, 500)
        self.assertEqual(exec_ctx.cache_read_tokens, 1000)


# ---------------------------------------------------------------------------
# 6. HITL notifications
# ---------------------------------------------------------------------------

class HitlNotificationTests(unittest.TestCase):
    def setUp(self):
        self.LEB = _use_llm_blocked(self)

    def _run_hitl(self, plugin, decision_resolution):
        exec_ctx = _make_exec_ctx(total_cost_usd=1.0, cost_status="estimated")
        plugin._exec_contexts._store["s"] = exec_ctx
        plugin._client = Mock()
        plugin._client.request_llm_hitl = Mock(return_value=_hitl_response())
        plugin._client.get_hitl_decision = Mock(
            return_value=_decision(decision_resolution, "reviewer note")
        )
        plugin._notify_user = Mock()
        next_call = _make_next_call()

        with patch.object(plugin, "_safe_log"), patch("time.sleep", return_value=None):
            try:
                plugin.llm_execution_middleware(
                    request={"messages": [{"role": "user", "content": "hello"}]},
                    next_call=next_call,
                    session_id="s", model="claude-opus-4-8",
                    provider="anthropic", base_url="",
                    api_call_count=2,
                )
            except self.LEB:
                pass
        return plugin._notify_user.call_args_list

    def test_hitl_pause_notification_includes_cost_context(self):
        plugin = _make_plugin()
        notifications = self._run_hitl(plugin, "approved")
        # First notification is the pause
        first_msg = notifications[0][0][0]
        self.assertIn("paused pending approval", first_msg)
        self.assertIn("Current cost", first_msg)

    def test_hitl_approve_notification_sent(self):
        plugin = _make_plugin()
        notifications = self._run_hitl(plugin, "approved")
        all_msgs = [n[0][0] for n in notifications]
        self.assertTrue(any("Approval received" in m for m in all_msgs))

    def test_hitl_reject_notification_sent(self):
        plugin = _make_plugin()
        notifications = self._run_hitl(plugin, "rejected")
        all_msgs = [n[0][0] for n in notifications]
        self.assertTrue(any("rejected" in m.lower() for m in all_msgs))

    def test_hitl_timeout_notification_sent(self):
        plugin = _make_plugin()
        exec_ctx = _make_exec_ctx()
        plugin._exec_contexts._store["s"] = exec_ctx
        plugin._config = SimpleNamespace(
            **{**vars(plugin._config),
               "hitl_timeout_minutes": 0,
               "hitl_poll_interval_seconds": 0,
               }
        )
        plugin._client = Mock()
        plugin._client.request_llm_hitl = Mock(return_value=_hitl_response())
        plugin._client.get_hitl_decision = Mock(return_value={"status": "pending"})
        plugin._notify_user = Mock()
        next_call = _make_next_call()

        with patch.object(plugin, "_safe_log"), patch("time.sleep", return_value=None):
            try:
                plugin.llm_execution_middleware(
                    request={"messages": []}, next_call=next_call,
                    session_id="s", model="m", provider="p", base_url="", api_call_count=1,
                )
            except self.LEB:
                pass

        all_msgs = [n[0][0] for n in plugin._notify_user.call_args_list]
        self.assertTrue(any("timed out" in m.lower() for m in all_msgs))

    def test_no_notification_on_plain_allow(self):
        plugin = _make_plugin()
        exec_ctx = _make_exec_ctx()
        plugin._exec_contexts._store["s"] = exec_ctx
        plugin._client = Mock()
        plugin._client.request_llm_hitl = Mock(return_value=_allow_response())
        plugin._notify_user = Mock()
        next_call = _make_next_call()

        with patch.object(plugin, "_safe_log"):
            plugin.llm_execution_middleware(
                request={"messages": []}, next_call=next_call,
                session_id="s", model="m", provider="p", base_url="", api_call_count=1,
            )

        plugin._notify_user.assert_not_called()


# ---------------------------------------------------------------------------
# 7. AMP signal payload
# ---------------------------------------------------------------------------

class AmpSignalPayloadTests(unittest.TestCase):
    def test_request_llm_hitl_includes_required_fields(self):
        from hermes.amp_client import AmpClient
        from hermes.config import AmpConfig
        config = AmpConfig(
            backend_url="https://amp.example.com",
            api_key="k",
            org_id="O-1",
            agent_name="hermes-test",
            username="u",
            hitl_timeout_minutes=10,
            hitl_poll_interval_seconds=3,
            fail_closed=True,
            notifications_enabled=True,
            llm_governance_enabled=True,
            llm_governance_mode="enforce",
            llm_governance_fail_closed=False,
            llm_governance_include_subagents=True,
        )
        client = AmpClient(config)
        captured = {}

        def fake_request(method, path, payload=None, query=None):
            captured.update(payload or {})
            return {"status": "no-hitl"}

        with patch.object(client, "_request", side_effect=fake_request):
            client.request_llm_hitl(
                "inst-1",
                execution_id="exec-1",
                model="claude-opus-4-8",
                provider="anthropic",
                session_id="sess-1",
                current_cost_usd=1.23,
                current_cost_status="estimated",
                estimated_next_call_cost_usd=0.05,
                estimated_next_call_cost_status="estimated",
                projected_total_cost_usd=1.28,
                projected_total_cost_status="estimated",
                input_tokens_total=1000,
                output_tokens_total=400,
                total_tokens=1400,
                llm_calls_total=2,
            )

        required = [
            "tool", "action", "execution_id", "model", "provider",
            "current_cost_usd", "current_cost_status",
            "estimated_next_call_cost_usd", "estimated_next_call_cost_status",
            "projected_total_cost_usd", "projected_total_cost_status",
            "input_tokens_total", "output_tokens_total", "total_tokens",
            "llm_calls_total",
        ]
        for field in required:
            self.assertIn(field, captured, f"Missing field: {field}")

        self.assertEqual(captured["tool"], "llm")
        self.assertEqual(captured["action"], "invoke")
        self.assertEqual(captured["execution_id"], "exec-1")
        self.assertAlmostEqual(captured["current_cost_usd"], 1.23, places=5)

    def test_approved_budget_omitted_when_none(self):
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

        with patch.object(client, "_request", side_effect=lambda m, p, payload=None, query=None: captured.update(payload or {}) or {"status": "no-hitl"}):
            client.request_llm_hitl("inst-1", approved_budget_usd=None)

        self.assertNotIn("approved_budget_usd", captured)


# ---------------------------------------------------------------------------
# 8. Cost estimation and projection utilities
# ---------------------------------------------------------------------------

class CostEstimationTests(unittest.TestCase):
    def test_worst_cost_status_unknown_wins(self):
        self.assertEqual(_worst_cost_status("estimated", "unknown"), "unknown")
        self.assertEqual(_worst_cost_status("unknown", "estimated"), "unknown")

    def test_worst_cost_status_estimated_over_actual(self):
        self.assertEqual(_worst_cost_status("actual", "estimated"), "estimated")

    def test_worst_cost_status_same(self):
        self.assertEqual(_worst_cost_status("estimated", "estimated"), "estimated")

    def test_estimate_returns_float_and_status(self):
        cost, status = _estimate_pending_call_cost("unknown-model", "ollama", "http://localhost:11434", {})
        self.assertIsInstance(cost, float)
        self.assertIsInstance(status, str)
        self.assertGreaterEqual(cost, 0.0)

    def test_estimate_with_messages(self):
        request = {
            "messages": [
                {"role": "user", "content": "A" * 400},  # ~100 tokens
                {"role": "assistant", "content": "B" * 200},
            ]
        }
        cost, status = _estimate_pending_call_cost("unknown-model", "anthropic", "", request)
        self.assertIsInstance(cost, float)

    def test_format_cost_context_known_costs(self):
        from hermes import AmpGovernancePlugin
        msg = AmpGovernancePlugin._format_cost_context(
            1.82, "estimated", 0.24, "estimated", 2.06, "estimated"
        )
        self.assertIn("$1.82", msg)
        self.assertIn("$0.24", msg)
        self.assertIn("$2.06", msg)

    def test_format_cost_context_unknown_costs(self):
        from hermes import AmpGovernancePlugin
        msg = AmpGovernancePlugin._format_cost_context(
            0.0, "unknown", 0.0, "unknown", 0.0, "unknown"
        )
        self.assertIn("unknown", msg)
        self.assertNotIn("$", msg)


# ---------------------------------------------------------------------------
# 9. ExecutionContext Phase 2B fields
# ---------------------------------------------------------------------------

class ExecutionContextEnforcementFieldsTests(unittest.TestCase):
    def test_new_fields_initialized(self):
        ctx = _make_exec_ctx()
        self.assertIsNone(ctx.approved_budget_usd)
        self.assertIsNone(ctx.last_policy_eval_at)
        self.assertIsNone(ctx.last_policy_result)
        self.assertIsNone(ctx.hitl_workitem_id)
        self.assertEqual(ctx.policy_eval_count, 0)
        self.assertEqual(ctx.llm_hitl_approved, 0)
        self.assertEqual(ctx.llm_blocked, 0)

    def test_summary_dict_includes_enforcement_fields(self):
        ctx = _make_exec_ctx()
        ctx.policy_eval_count = 3
        ctx.llm_hitl_approved = 1
        ctx.llm_blocked = 0
        d = ctx.to_summary_dict()
        self.assertIn("policy_eval_count", d)
        self.assertIn("llm_hitl_approved", d)
        self.assertIn("llm_blocked", d)
        self.assertNotIn("approved_budget_usd", d)  # None → omitted

    def test_summary_dict_includes_budget_when_set(self):
        ctx = _make_exec_ctx()
        ctx.approved_budget_usd = 5.0
        d = ctx.to_summary_dict()
        self.assertIn("approved_budget_usd", d)
        self.assertAlmostEqual(d["approved_budget_usd"], 5.0, places=5)

    def test_blocked_count_increments(self):
        plugin = _make_plugin()
        exec_ctx = _make_exec_ctx()
        plugin._exec_contexts._store["s"] = exec_ctx
        plugin._client = Mock()
        plugin._client.request_llm_hitl = Mock(return_value=_block_response())
        plugin._notify_user = Mock()

        LLMExecutionBlocked = _use_llm_blocked(self)
        with patch.object(plugin, "_safe_log"):
            with self.assertRaises(LLMExecutionBlocked):
                plugin.llm_execution_middleware(
                    request={"messages": []}, next_call=_make_next_call(),
                    session_id="s", model="m", provider="p", base_url="", api_call_count=1,
                )

        self.assertEqual(exec_ctx.llm_blocked, 1)


# ---------------------------------------------------------------------------
# 10. Regression: existing tests unchanged
# ---------------------------------------------------------------------------

class RegressionToolGovernanceTests(unittest.TestCase):
    """Verify that tool-governance behavior is unaffected by Phase 2B code."""

    def test_pre_tool_call_blocks_on_no_policy(self):
        plugin = _make_plugin()
        plugin._client = Mock()
        plugin._client.request_hitl = Mock(return_value={"status": "no_policy"})
        with patch.object(plugin, "_ensure_instance", return_value="inst-1"), \
             patch.object(plugin, "_safe_log"):
            from hermes.policy import NormalizedAction
            # Patch normalize_tool_call to return a governed action
            with patch("hermes.normalize_tool_call") as mock_norm:
                mock_norm.return_value = NormalizedAction("terminal", "exec", "exec", {})
                result = plugin.pre_tool_call(
                    tool_name="terminal", args={"command": "ls"}, session_id="s"
                )
        self.assertIsNotNone(result)
        self.assertEqual(result.get("action"), "block")

    def test_pre_tool_call_allows_on_no_hitl(self):
        plugin = _make_plugin()
        plugin._client = Mock()
        plugin._client.request_hitl = Mock(return_value={"status": "no-hitl"})
        with patch.object(plugin, "_ensure_instance", return_value="inst-1"), \
             patch.object(plugin, "_safe_log"):
            with patch("hermes.normalize_tool_call") as mock_norm:
                from hermes.policy import NormalizedAction
                mock_norm.return_value = NormalizedAction("terminal", "exec", "exec", {})
                result = plugin.pre_tool_call(
                    tool_name="terminal", args={"command": "ls"}, session_id="s"
                )
        self.assertIsNone(result)

    def test_ungoverned_tool_passthrough(self):
        plugin = _make_plugin()
        # normalize_tool_call returns None for ungoverned tools (e.g., send_message)
        result = plugin.pre_tool_call(
            tool_name="send_message",
            args={"target": "slack:C1", "message": "hi"},
            session_id="s",
        )
        self.assertIsNone(result)

    def test_tool_fail_closed_is_separate_from_llm_fail_closed(self):
        """AMP_FAIL_CLOSED (tool) and AMP_LLM_GOVERNANCE_FAIL_CLOSED (LLM) are independent."""
        plugin = _make_plugin(fail_closed=False)  # LLM fail-open
        plugin._config = SimpleNamespace(
            **{**vars(plugin._config),
               "fail_closed": True,       # tool governance: fail closed
               "llm_governance_fail_closed": False,  # LLM governance: fail open
               }
        )
        exec_ctx = _make_exec_ctx()
        plugin._exec_contexts._store["s"] = exec_ctx
        plugin._client = Mock()
        plugin._client.request_llm_hitl = Mock(side_effect=Exception("down"))
        plugin._notify_user = Mock()
        next_call = _make_next_call("resp")

        with patch.object(plugin, "_safe_log"):
            result = plugin.llm_execution_middleware(
                request={"messages": []}, next_call=next_call,
                session_id="s", model="m", provider="p", base_url="", api_call_count=1,
            )

        # LLM governance is fail-open → provider called
        self.assertEqual(result, "resp")


if __name__ == "__main__":
    unittest.main()
