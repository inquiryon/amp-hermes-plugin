"""Tests for Phase 3B registered tools: amp_evaluate_research_plan,
amp_load_research_topics, amp_governance_summary.

These test the deterministic Python surface only — whether a model actually
loads the research-agent skill and calls these tools in the right order is
model behavior, not covered here (see the manual test procedure in the
final report).
"""

from __future__ import annotations

import json
import sys
import tempfile
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

import hermes as mod
from hermes.execution_context import ExecutionContext


def _write_topics(tmp: str, content: str) -> str:
    path = Path(tmp) / "research_topics.yaml"
    path.write_text(content)
    return str(path)


# ---------------------------------------------------------------------------
# 1. Registration
# ---------------------------------------------------------------------------

class ToolRegistrationTests(unittest.TestCase):
    def test_register_wires_all_three_tools_and_the_skill(self):
        ctx = SimpleNamespace(
            dispatch_tool=Mock(),
            register_hook=Mock(),
            register_middleware=Mock(),
            register_tool=Mock(),
            register_skill=Mock(),
            register_cli_command=Mock(),
        )
        fresh_plugin = mod.AmpGovernancePlugin()
        with patch.object(mod, "_PLUGIN", fresh_plugin):
            mod.register(ctx)

        tool_names = {call.kwargs.get("name") for call in ctx.register_tool.call_args_list}
        self.assertEqual(
            tool_names,
            {"amp_evaluate_research_plan", "amp_load_research_topics", "amp_governance_summary"},
        )
        for call in ctx.register_tool.call_args_list:
            self.assertEqual(call.kwargs.get("toolset"), "amp_governance")
            self.assertTrue(callable(call.kwargs.get("handler")))
            self.assertIn("parameters", call.kwargs.get("schema"))

        ctx.register_skill.assert_called_once()
        skill_args = ctx.register_skill.call_args
        self.assertEqual(skill_args.args[0], "research-agent")
        self.assertTrue(str(skill_args.args[1]).endswith("skills/research-agent/SKILL.md"))

    def test_skill_registration_failure_does_not_crash_register(self):
        ctx = SimpleNamespace(
            dispatch_tool=Mock(),
            register_hook=Mock(),
            register_middleware=Mock(),
            register_tool=Mock(),
            register_skill=Mock(side_effect=FileNotFoundError("no SKILL.md")),
            register_cli_command=Mock(),
        )
        fresh_plugin = mod.AmpGovernancePlugin()
        with patch.object(mod, "_PLUGIN", fresh_plugin):
            mod.register(ctx)  # must not raise


# ---------------------------------------------------------------------------
# 2. amp_evaluate_research_plan tool
# ---------------------------------------------------------------------------

class EvaluateResearchPlanToolTests(unittest.TestCase):
    def test_missing_plan_arg_returns_error_without_calling_plugin(self):
        fake_plugin = Mock()
        with patch.object(mod, "_PLUGIN", fake_plugin):
            raw = mod._tool_amp_evaluate_research_plan({})
        result = json.loads(raw)
        self.assertEqual(result["status"], "error")
        fake_plugin.evaluate_proposed_plan.assert_not_called()

    def test_session_id_sourced_from_session_context_not_args(self):
        fake_plugin = Mock()
        fake_plugin.evaluate_proposed_plan = Mock(return_value={"status": "approved"})
        fake_plugin._exec_contexts = Mock()
        fake_plugin._exec_contexts.get = Mock(return_value=None)
        plan = {"plan_type": "research", "projected_cost_usd": 4.75, "estimated_llm_calls": 5}
        args = {"plan": plan, "session_id": "should-be-ignored"}

        with patch.object(mod, "_PLUGIN", fake_plugin), \
             patch.object(mod, "_current_session_id", return_value="real-session-123"):
            raw = mod._tool_amp_evaluate_research_plan(args)

        fake_plugin.evaluate_proposed_plan.assert_called_once_with("real-session-123", plan)
        self.assertEqual(json.loads(raw)["status"], "approved")

    def test_deeply_nested_payload_preserved_unmodified(self):
        fake_plugin = Mock()
        fake_plugin.evaluate_proposed_plan = Mock(return_value={"status": "approved"})
        fake_plugin._exec_contexts = Mock()
        fake_plugin._exec_contexts.get = Mock(return_value=None)
        nested_payload = {
            "topics": ["AI agent governance", "Progressive autonomy"],
            "research_depth": "standard",
            "sources_per_topic": 5,
            "nested": {"arbitrary": ["structure", 1, True, {"deep": None}]},
        }
        plan = {
            "plan_type": "research",
            "projected_cost_usd": 4.75,
            "estimated_llm_calls": 5,
            "payload": nested_payload,
        }

        with patch.object(mod, "_PLUGIN", fake_plugin), \
             patch.object(mod, "_current_session_id", return_value="s"):
            mod._tool_amp_evaluate_research_plan({"plan": plan})

        forwarded_plan = fake_plugin.evaluate_proposed_plan.call_args[0][1]
        self.assertEqual(forwarded_plan["payload"], nested_payload)

    def test_all_four_decision_statuses_pass_through_unchanged(self):
        for status in ("approved", "rejected", "timed_out", "error"):
            fake_plugin = Mock()
            decision = {
                "status": status, "plan_id": "p1", "approved_budget_usd": 4.75 if status == "approved" else None,
                "reason": "", "workitem_id": None,
            }
            fake_plugin.evaluate_proposed_plan = Mock(return_value=decision)
            fake_plugin._exec_contexts = Mock()
            fake_plugin._exec_contexts.get = Mock(return_value=None)
            with patch.object(mod, "_PLUGIN", fake_plugin), \
                 patch.object(mod, "_current_session_id", return_value="s"):
                raw = mod._tool_amp_evaluate_research_plan(
                    {"plan": {"plan_type": "research", "projected_cost_usd": 1.0, "estimated_llm_calls": 5}}
                )
            self.assertEqual(json.loads(raw), decision)


# ---------------------------------------------------------------------------
# 3. amp_load_research_topics tool
# ---------------------------------------------------------------------------

class LoadResearchTopicsToolTests(unittest.TestCase):
    def test_valid_path_returns_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_topics(tmp, "topics:\n  - AI agent governance\n")
            raw = mod._tool_amp_load_research_topics({"path": path})
        result = json.loads(raw)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["topics"], ["AI agent governance"])

    def test_missing_file_returns_error_not_exception(self):
        raw = mod._tool_amp_load_research_topics({"path": "/nonexistent/research_topics.yaml"})
        result = json.loads(raw)
        self.assertEqual(result["status"], "error")

    def test_default_path_used_when_omitted(self):
        with patch.object(mod, "load_research_topics") as mock_load:
            mock_load.return_value = {"topics": ["x"], "research_depth": "standard",
                                       "sources_per_topic": 5, "lookback_days": 7}
            mod._tool_amp_load_research_topics({})
        mock_load.assert_called_once_with(mod._DEFAULT_RESEARCH_TOPICS_PATH)


# ---------------------------------------------------------------------------
# 4. amp_governance_summary tool
# ---------------------------------------------------------------------------

class GovernanceSummaryToolTests(unittest.TestCase):
    def test_no_context_returns_explicit_status(self):
        with patch.object(mod, "_current_session_id", return_value="no-such-session"):
            raw = mod._tool_amp_governance_summary({})
        self.assertEqual(json.loads(raw)["status"], "no_context")

    def test_no_session_id_returns_no_context(self):
        with patch.object(mod, "_current_session_id", return_value=""):
            raw = mod._tool_amp_governance_summary({})
        self.assertEqual(json.loads(raw)["status"], "no_context")

    def test_existing_context_returns_summary_fields(self):
        exec_ctx = ExecutionContext("s", "inst-1", model="m", platform="slack")
        exec_ctx.total_cost_usd = 3.92
        exec_ctx.llm_calls = 16
        exec_ctx.tool_calls = 21
        exec_ctx.llm_hitl_approved = 0

        with patch.object(mod, "_current_session_id", return_value="s"), \
             patch.object(mod._PLUGIN._exec_contexts, "get", return_value=exec_ctx):
            raw = mod._tool_amp_governance_summary({})

        result = json.loads(raw)
        self.assertEqual(result["status"], "running")
        self.assertAlmostEqual(result["total_cost_usd"], 3.92, places=5)
        self.assertEqual(result["llm_calls"], 16)
        self.assertEqual(result["tool_calls"], 21)
        self.assertEqual(result["llm_hitl_approved"], 0)

    def test_does_not_invent_values_not_in_summary_dict(self):
        """No 'approved_budget_usd' key when it was never set (None -> omitted
        by ExecutionContext.to_summary_dict(), matching existing Phase 2B behavior)."""
        exec_ctx = ExecutionContext("s", "inst-1", model="m", platform="slack")
        with patch.object(mod, "_current_session_id", return_value="s"), \
             patch.object(mod._PLUGIN._exec_contexts, "get", return_value=exec_ctx):
            raw = mod._tool_amp_governance_summary({})
        self.assertNotIn("approved_budget_usd", json.loads(raw))

    def test_includes_ready_to_paste_report_text(self):
        """The response always carries a report_text field so research-agent's
        step 8 has nothing left to compose or recall from memory."""
        exec_ctx = ExecutionContext("s", "inst-1", model="m", platform="slack")
        exec_ctx.total_cost_usd = 0.006
        exec_ctx.cost_status = "estimated"
        exec_ctx.llm_calls = 3
        exec_ctx.tool_calls = 9
        exec_ctx.llm_hitl_approved = 0
        exec_ctx.approved_budget_usd = 0.03
        exec_ctx.last_plan_projected_cost_usd = 0.03
        exec_ctx.last_plan_projected_cost_status = "estimated"
        exec_ctx.plan_approvals_count = 1

        with patch.object(mod, "_current_session_id", return_value="s"), \
             patch.object(mod._PLUGIN._exec_contexts, "get", return_value=exec_ctx):
            raw = mod._tool_amp_governance_summary({})

        result = json.loads(raw)
        self.assertIn("report_text", result)
        self.assertEqual(
            result["report_text"],
            "Governance summary\n"
            "Plan projected cost: $0.03 (estimated)\n"
            "Approved budget: $0.03\n"
            "Actual cost: $0.01\n"
            "LLM calls: 3\n"
            "Tool calls: 9\n"
            "AMP plan approvals: 1\n"
            "Runtime HITL approvals: 0",
        )


# ---------------------------------------------------------------------------
# 5a. _format_governance_report_text (deterministic rendering)
# ---------------------------------------------------------------------------

class FormatGovernanceReportTextTests(unittest.TestCase):
    def test_unknown_cost_status_never_prints_a_dollar_amount(self):
        text = mod._format_governance_report_text({
            "cost_status": "unknown", "total_cost_usd": 0.0,
            "llm_calls": 3, "tool_calls": 9, "llm_hitl_approved": 0,
        })
        self.assertIn("Actual cost: cost tracking unavailable for this model (cost_status=unknown)", text)
        self.assertNotIn("Actual cost: $", text)

    def test_actual_cost_status_prints_dollar_amount(self):
        text = mod._format_governance_report_text({
            "cost_status": "actual", "total_cost_usd": 1.2345,
            "llm_calls": 1, "tool_calls": 1, "llm_hitl_approved": 0,
        })
        self.assertIn("Actual cost: $1.23", text)

    def test_missing_plan_projected_cost_prints_unavailable(self):
        text = mod._format_governance_report_text({
            "cost_status": "unknown", "llm_calls": 0, "tool_calls": 0, "llm_hitl_approved": 0,
        })
        self.assertIn("Plan projected cost: unavailable", text)

    def test_missing_approved_budget_prints_unavailable(self):
        text = mod._format_governance_report_text({
            "cost_status": "unknown", "llm_calls": 0, "tool_calls": 0, "llm_hitl_approved": 0,
        })
        self.assertIn("Approved budget: unavailable", text)

    def test_present_plan_projected_cost_includes_status(self):
        text = mod._format_governance_report_text({
            "plan_projected_cost_usd": 4.5, "plan_projected_cost_status": "estimated",
            "cost_status": "unknown", "llm_calls": 0, "tool_calls": 0, "llm_hitl_approved": 0,
        })
        self.assertIn("Plan projected cost: $4.50 (estimated)", text)

    def test_defaults_missing_call_counts_to_zero(self):
        text = mod._format_governance_report_text({"cost_status": "unknown"})
        self.assertIn("LLM calls: 0", text)
        self.assertIn("Tool calls: 0", text)
        self.assertIn("AMP plan approvals: 0", text)
        self.assertIn("Runtime HITL approvals: 0", text)


# ---------------------------------------------------------------------------
# 5. Deterministic cost projection (_project_research_cost)
# ---------------------------------------------------------------------------

class ProjectResearchCostTests(unittest.TestCase):
    def test_zero_estimated_calls_returns_unknown(self):
        exec_ctx = ExecutionContext("s", "inst-1", model="gpt-4o-mini", platform="slack")
        exec_ctx.last_provider = "openai"
        self.assertEqual(mod._project_research_cost(exec_ctx, 0), (0.0, "unknown"))

    def test_negative_estimated_calls_returns_unknown(self):
        exec_ctx = ExecutionContext("s", "inst-1", model="gpt-4o-mini", platform="slack")
        self.assertEqual(mod._project_research_cost(exec_ctx, -1), (0.0, "unknown"))

    def test_no_exec_context_returns_unknown(self):
        self.assertEqual(mod._project_research_cost(None, 5), (0.0, "unknown"))

    def test_no_model_on_context_returns_unknown(self):
        exec_ctx = ExecutionContext("s", "inst-1", model="", platform="slack")
        self.assertEqual(mod._project_research_cost(exec_ctx, 5), (0.0, "unknown"))

    def test_known_model_and_provider_yields_grounded_estimate(self):
        exec_ctx = ExecutionContext("s", "inst-1", model="gpt-4o-mini", platform="slack")
        exec_ctx.last_provider = "openai"
        exec_ctx.last_base_url = "https://api.openai.com/v1"
        with patch.object(mod, "_calc_cost", return_value=(0.0221, "estimated", "official_docs_snapshot")) as mock_calc:
            cost, status = mod._project_research_cost(exec_ctx, 5)
        called_model, called_provider, called_base_url, called_usage = mock_calc.call_args[0]
        self.assertEqual(called_model, "gpt-4o-mini")
        self.assertEqual(called_provider, "openai")
        self.assertEqual(called_base_url, "https://api.openai.com/v1")
        self.assertEqual(called_usage["input_tokens"], mod._RESEARCH_ASSUMED_INPUT_TOKENS_PER_CALL * 5)
        self.assertEqual(called_usage["output_tokens"], mod._RESEARCH_ASSUMED_OUTPUT_TOKENS_PER_CALL * 5)
        self.assertEqual(status, "estimated")
        self.assertEqual(cost, 0.03)  # rounded up to the cent from 0.0221

    def test_unknown_pricing_lookup_yields_unknown_status(self):
        exec_ctx = ExecutionContext("s", "inst-1", model="some-unpriced-model", platform="slack")
        with patch.object(mod, "_calc_cost", return_value=(0.0, "unknown", "")):
            cost, status = mod._project_research_cost(exec_ctx, 5)
        self.assertEqual((cost, status), (0.0, "unknown"))

    def test_result_always_rounds_up_to_the_cent(self):
        exec_ctx = ExecutionContext("s", "inst-1", model="gpt-4o-mini", platform="slack")
        with patch.object(mod, "_calc_cost", return_value=(1.001, "estimated", "src")):
            cost, status = mod._project_research_cost(exec_ctx, 3)
        self.assertEqual(cost, 1.01)
        self.assertEqual(status, "estimated")

    def test_result_is_estimated_even_when_price_lookup_was_actual(self):
        """The assumed token counts are never measured, so the projection is
        always a projection regardless of how confident the underlying price
        lookup was."""
        exec_ctx = ExecutionContext("s", "inst-1", model="gpt-4o-mini", platform="slack")
        with patch.object(mod, "_calc_cost", return_value=(0.05, "actual", "src")):
            _cost, status = mod._project_research_cost(exec_ctx, 2)
        self.assertEqual(status, "estimated")


# ---------------------------------------------------------------------------
# 6. amp_evaluate_research_plan tool always overrides the LLM-supplied cost
# ---------------------------------------------------------------------------

class EvaluateResearchPlanCostOverrideTests(unittest.TestCase):
    def test_llm_supplied_cost_is_always_overwritten(self):
        """The model might still put a number in projected_cost_usd (an old
        habit, a stale cached prompt, etc.) -- it must never survive into the
        plan AMP actually evaluates."""
        exec_ctx = ExecutionContext("s", "inst-1", model="gpt-4o-mini", platform="slack")
        fake_plugin = Mock()
        fake_plugin.evaluate_proposed_plan = Mock(return_value={"status": "approved"})
        fake_plugin._exec_contexts = Mock()
        fake_plugin._exec_contexts.get = Mock(return_value=exec_ctx)
        plan = {
            "plan_type": "research",
            "projected_cost_usd": 100.00,
            "projected_cost_status": "estimated",
            "estimated_llm_calls": 5,
        }

        with patch.object(mod, "_PLUGIN", fake_plugin), \
             patch.object(mod, "_current_session_id", return_value="s"), \
             patch.object(mod, "_project_research_cost", return_value=(0.03, "estimated")) as mock_project:
            mod._tool_amp_evaluate_research_plan({"plan": plan})

        mock_project.assert_called_once_with(exec_ctx, 5)
        forwarded_plan = fake_plugin.evaluate_proposed_plan.call_args[0][1]
        self.assertEqual(forwarded_plan["projected_cost_usd"], 0.03)
        self.assertEqual(forwarded_plan["projected_cost_status"], "estimated")
        self.assertEqual(exec_ctx.last_plan_projected_cost_usd, 0.03)
        self.assertEqual(exec_ctx.last_plan_projected_cost_status, "estimated")
        self.assertEqual(exec_ctx.plan_approvals_count, 1)

    def test_plan_approvals_count_not_incremented_on_rejection(self):
        exec_ctx = ExecutionContext("s", "inst-1", model="gpt-4o-mini", platform="slack")
        fake_plugin = Mock()
        fake_plugin.evaluate_proposed_plan = Mock(return_value={"status": "rejected"})
        fake_plugin._exec_contexts = Mock()
        fake_plugin._exec_contexts.get = Mock(return_value=exec_ctx)
        plan = {"plan_type": "research", "estimated_llm_calls": 5}

        with patch.object(mod, "_PLUGIN", fake_plugin), \
             patch.object(mod, "_current_session_id", return_value="s"), \
             patch.object(mod, "_project_research_cost", return_value=(0.03, "estimated")):
            mod._tool_amp_evaluate_research_plan({"plan": plan})

        # Projected cost is still stashed (a rejected plan still had a real
        # projection), but approvals count must not move for a non-approval.
        self.assertEqual(exec_ctx.last_plan_projected_cost_usd, 0.03)
        self.assertEqual(exec_ctx.plan_approvals_count, 0)

    def test_plan_approvals_count_increments_across_multiple_approved_plans(self):
        exec_ctx = ExecutionContext("s", "inst-1", model="gpt-4o-mini", platform="slack")
        fake_plugin = Mock()
        fake_plugin.evaluate_proposed_plan = Mock(return_value={"status": "approved"})
        fake_plugin._exec_contexts = Mock()
        fake_plugin._exec_contexts.get = Mock(return_value=exec_ctx)
        plan = {"plan_type": "research", "estimated_llm_calls": 5}

        with patch.object(mod, "_PLUGIN", fake_plugin), \
             patch.object(mod, "_current_session_id", return_value="s"), \
             patch.object(mod, "_project_research_cost", return_value=(0.03, "estimated")):
            mod._tool_amp_evaluate_research_plan({"plan": dict(plan)})
            mod._tool_amp_evaluate_research_plan({"plan": dict(plan)})

        self.assertEqual(exec_ctx.plan_approvals_count, 2)

    def test_missing_estimated_llm_calls_rejected(self):
        fake_plugin = Mock()
        fake_plugin.evaluate_proposed_plan = Mock(return_value={"status": "approved"})
        fake_plugin._exec_contexts = Mock()
        fake_plugin._exec_contexts.get = Mock(return_value=None)
        plan = {"plan_type": "research"}

        with patch.object(mod, "_PLUGIN", fake_plugin), \
             patch.object(mod, "_current_session_id", return_value="s"), \
             patch.object(mod, "_project_research_cost") as mock_project:
            result = json.loads(mod._tool_amp_evaluate_research_plan({"plan": plan}))

        self.assertEqual(result["status"], "error")
        mock_project.assert_not_called()
        fake_plugin.evaluate_proposed_plan.assert_not_called()

    def test_non_integer_estimated_llm_calls_rejected(self):
        fake_plugin = Mock()
        fake_plugin.evaluate_proposed_plan = Mock(return_value={"status": "approved"})
        fake_plugin._exec_contexts = Mock()
        fake_plugin._exec_contexts.get = Mock(return_value=None)
        plan = {"plan_type": "research", "estimated_llm_calls": "not-a-number"}

        with patch.object(mod, "_PLUGIN", fake_plugin), \
             patch.object(mod, "_current_session_id", return_value="s"), \
             patch.object(mod, "_project_research_cost") as mock_project:
            result = json.loads(mod._tool_amp_evaluate_research_plan({"plan": plan}))

        self.assertEqual(result["status"], "error")
        mock_project.assert_not_called()
        fake_plugin.evaluate_proposed_plan.assert_not_called()


if __name__ == "__main__":
    unittest.main()
