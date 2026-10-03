from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import re
import threading
import time
import uuid
from typing import Any, Dict, Optional, Tuple

from .amp_client import AmpClient, AmpClientError
from .config import AmpConfig, load_config
from .execution_context import ExecutionContext, ExecutionContextStore, LlmCallRecord
from .notification import notify_user
from .policy import NormalizedAction, normalize_tool_call
from .research_config import load_research_topics
from .session_store import SessionRecord, SessionStore, agent_fingerprint

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Runtime capability detection for Phase 2B LLM enforcement.
#
# LLMExecutionBlocked is available only when Hermes includes the upstream
# change from nousresearch/hermes-agent#64662. AHP detects this at import
# time and registers the enforcement middleware only when the class is present.
# ---------------------------------------------------------------------------
try:
    from hermes_cli.middleware import LLMExecutionBlocked as _LLMExecutionBlocked
    _LLM_BLOCKED_AVAILABLE: bool = True
except ImportError:
    _LLMExecutionBlocked = None  # type: ignore[assignment,misc]
    _LLM_BLOCKED_AVAILABLE = False


def _truncate(value: str, limit: int = 220) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit - 3]}..."


# ---------------------------------------------------------------------------
# LLM trace capture (AMP's "click LLM in the log" panel) — see
# amp_client.save_llm_trace and _save_llm_trace below.
# ---------------------------------------------------------------------------

_MAX_PENDING_PROMPTS = 200
_TRACE_FIELD_CHARS = 2000


def _extract_reasoning_from_assistant_message(assistant_message: Any) -> Optional[str]:
    """Best-effort reasoning/thinking extraction from a provider response.

    Adapted from Hermes' own agent/agent_runtime_helpers.py::extract_reasoning
    (same attribute names/precedence) but kept self-contained here rather than
    imported, since it's a handful of attribute checks, not Hermes-internal
    logic AHP needs to stay coupled to. Most models (e.g. gpt-4o-mini) expose
    none of these fields — that's expected, not an error; callers should
    treat None as "no reasoning available", not "extraction failed".
    """
    if assistant_message is None:
        return None
    parts = []
    reasoning = getattr(assistant_message, "reasoning", None)
    if reasoning:
        parts.append(str(reasoning))
    reasoning_content = getattr(assistant_message, "reasoning_content", None)
    if reasoning_content and str(reasoning_content) not in parts:
        parts.append(str(reasoning_content))
    reasoning_details = getattr(assistant_message, "reasoning_details", None)
    if reasoning_details:
        for detail in reasoning_details:
            if not isinstance(detail, dict):
                continue
            summary = (
                detail.get("summary")
                or detail.get("thinking")
                or detail.get("content")
                or detail.get("text")
            )
            if summary and summary not in parts:
                parts.append(str(summary))
    if not parts:
        return None
    return _truncate("\n".join(parts), limit=_TRACE_FIELD_CHARS)


def _summarize_assistant_answer(assistant_message: Any) -> str:
    """Human-readable summary of what the assistant actually did on this
    call: its response text if it wrote one, otherwise which tools it called
    and with what arguments (a tool-calls-only turn has empty .content)."""
    if assistant_message is None:
        return ""
    content = (getattr(assistant_message, "content", None) or "").strip()
    if content:
        return _truncate(content, limit=_TRACE_FIELD_CHARS)
    tool_calls = getattr(assistant_message, "tool_calls", None) or []
    if not tool_calls:
        return ""
    described = []
    for call in tool_calls:
        fn = getattr(call, "function", None)
        name = getattr(fn, "name", None) or getattr(call, "name", None) or "unknown_tool"
        args = getattr(fn, "arguments", None) or getattr(call, "arguments", None) or ""
        described.append(f"{name}({_truncate(str(args), limit=200)})")
    return _truncate("Called: " + ", ".join(described), limit=_TRACE_FIELD_CHARS)


_FRESHNESS_PATTERNS = [
    re.compile(r"\b(today|latest|current|currently|recent|now|right now)\b", re.IGNORECASE),
    re.compile(
        r"\b(yesterday|tomorrow|tonight|this morning|this afternoon|this evening|last night)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(this week|this month|this year|last week|last month|last year)\b", re.IGNORECASE),
    re.compile(r"\b(live|up-to-date|up to date|as of)\b", re.IGNORECASE),
]

_LIVE_DATA_DOMAIN_PATTERNS = [
    re.compile(r"\b(stock|market|markets|share price|price|prices|earnings|index|indices)\b", re.IGNORECASE),
    re.compile(r"\b(news|headline|headlines|weather|forecast|sports|score|scores)\b", re.IGNORECASE),
]

_EXPLICIT_SEARCH_PATTERNS = [
    re.compile(r"\b(web search|search the web|look up|lookup|find online|search online)\b", re.IGNORECASE),
]

_RESEARCH_TRIGGER_PATTERN = re.compile(r"\bresearch\b", re.IGNORECASE)

# Same configured-vs-ad-hoc distinction documented in the two pointer skills'
# own descriptions (skills-pointer/amp-research{,-topic}/SKILL.md). Resolved
# here, once, instead of leaving it to the model to pick the right pointer
# skill by NL matching -- that was observed to pick the wrong one for a
# clearly ad hoc message ("research US market today" loaded amp-research,
# the configured-mode pointer, and ran unrelated saved topics).
_CONFIGURED_MODE_PATTERN = re.compile(
    r"\bresearch\b.{0,20}\btopics?\b|\btopics?\b.{0,20}\bresearch\b|"
    r"\bmy\b.{0,20}\bresearch\b|\bresearch\b.{0,20}\bmy\b",
    re.IGNORECASE,
)


def _needs_live_web_search(user_message: str) -> bool:
    text = str(user_message or "").strip()
    if not text:
        return False
    has_freshness = any(pattern.search(text) for pattern in _FRESHNESS_PATTERNS)
    has_live_domain = any(pattern.search(text) for pattern in _LIVE_DATA_DOMAIN_PATTERNS)
    has_explicit_search = any(pattern.search(text) for pattern in _EXPLICIT_SEARCH_PATTERNS)
    return has_explicit_search or (has_freshness and has_live_domain)


def _mentions_research(user_message: str) -> bool:
    """True if the message uses the word "research" -- the trigger word for
    the AMP-governed research workflow. When true, pre_llm_call injects a
    directive routing straight to the governed skill (see
    _build_research_skill_context) instead of the freshness-search context.
    """
    return bool(_RESEARCH_TRIGGER_PATTERN.search(str(user_message or "")))


def _is_configured_mode_request(user_message: str) -> bool:
    """True for "run my saved/configured research topics" phrasing, false
    for a message naming a specific topic (e.g. "research the US market").
    """
    return bool(_CONFIGURED_MODE_PATTERN.search(str(user_message or "")))


def _build_live_search_context(user_message: str) -> str:
    now = datetime.now(timezone.utc)
    today_utc = f"{now.strftime('%B')} {now.day}, {now.year}"
    return (
        f"Current UTC date: {today_utc}.\n"
        "The user's request appears time-sensitive or explicitly asks for live information.\n"
        "Do not answer from memory for this request.\n"
        "Use the web_search tool first to verify current facts before answering.\n"
        "If live search is unavailable, state that you cannot verify current information."
    )


def _build_research_skill_context(user_message: str) -> str:
    mode = "CONFIGURED" if _is_configured_mode_request(user_message) else "AD HOC"
    now = datetime.now(timezone.utc)
    today_utc = f"{now.strftime('%B')} {now.day}, {now.year}"
    return (
        f"Current UTC date: {today_utc}. Use this as \"today\" for every part of "
        "this workflow -- search queries, lookback windows, and the final "
        "report's dateline -- never your own belief about the current date.\n"
        f"This request is for the AMP-governed research workflow, mode={mode}.\n"
        "Do not call web_search, and do not call skill_view on amp-research "
        "or amp-research-topic -- go straight to the real skill: call "
        'skill_view(name="amp-governance:research-agent") now and follow it '
        f"exactly. Its step 0 is already decided for you: mode={mode}."
    )


def _infer_pricing_provider(provider: str, base_url: str) -> str:
    """Resolve the provider string used for *pricing lookups only* -- separate
    from whatever Hermes' own auth-provider config says.

    Hermes' auth-provider config often uses "custom" for a generic
    OpenAI-compatible endpoint (its auth-provider vocabulary, validated at
    startup against a fixed list, doesn't include "openai" as a value even
    though the pricing module's billing-route vocabulary does -- confirmed by
    `hermes doctor` rejecting `provider: openai` outright). So config.yaml
    can't simply say `provider: openai` to get real per-token OpenAI rates;
    doing so breaks live LLM calls entirely. Infer the real provider from
    base_url instead, only when Hermes' own provider tag isn't already
    something billing-specific (i.e. don't override a real, distinct
    provider like "anthropic" that's already correct).
    """
    if provider and provider.lower() not in {"custom", "local"}:
        return provider
    try:
        from utils import base_url_host_matches
    except Exception:
        return provider
    if base_url_host_matches(base_url, "openai.com"):
        return "openai"
    return provider


def _calc_cost(
    model: str,
    provider: str,
    base_url: str,
    usage: dict,
) -> tuple[float, str, str]:
    """Estimate cost from a usage dict using Hermes' pricing table.

    Returns (cost_usd, cost_status, cost_source).
    Falls back to (0.0, "unknown", "") when pricing is unavailable.
    """
    try:
        from agent.usage_pricing import CanonicalUsage, estimate_usage_cost
    except Exception:
        return 0.0, "unknown", ""
    try:
        cu = CanonicalUsage(
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            cache_read_tokens=int(usage.get("cache_read_tokens") or 0),
            cache_write_tokens=int(usage.get("cache_write_tokens") or 0),
            reasoning_tokens=int(usage.get("reasoning_tokens") or 0),
        )
        effective_provider = _infer_pricing_provider(provider, base_url)
        result = estimate_usage_cost(model, cu, provider=effective_provider or None, base_url=base_url or None)
        cost = float(result.amount_usd or 0.0)
        return cost, str(result.status), str(result.source)
    except Exception:
        return 0.0, "unknown", ""


# Calibrated from observed production token usage across research-agent runs
# (post_api_request "LLM call" log data): individual calls in this workflow
# ranged from ~16K to ~55K total tokens, driven mostly by the growing
# conversation history each turn resends as input. Rounded up per the
# "be conservative" principle used elsewhere in plan estimation -- this
# feeds a HITL budget gate, not a bill, so overstating is the safe error.
_RESEARCH_ASSUMED_INPUT_TOKENS_PER_CALL = 35_000
_RESEARCH_ASSUMED_OUTPUT_TOKENS_PER_CALL = 1_500


def _project_research_cost(exec_ctx: Optional[ExecutionContext], estimated_llm_calls: int) -> tuple[float, str]:
    """Deterministically project a research plan's dollar cost from a call-count
    estimate x real per-token pricing, instead of asking the LLM to invent a
    dollar figure outright. The model is reasonably placed to judge how many
    LLM calls a run needs (that's what estimated_llm_calls is); it has no
    grounded way to know actual $/token rates, and asking it to guess both at
    once routinely produced numbers wildly disconnected from reality (e.g.
    $100 projected for a run that actually costs a few cents). Uses the
    session's own most-recently-observed model/provider/base_url so the
    projection reflects real pricing for whatever's actually configured.

    Returns (projected_cost_usd, projected_cost_status). Falls back to
    (0.0, "unknown") when there's nothing to price against yet (e.g. no LLM
    call has happened in this session) or pricing data isn't available for
    the resolved route -- callers should treat "unknown" the same way AMP's
    policy criteria already treat any other unknown-cost plan.
    """
    if estimated_llm_calls <= 0 or exec_ctx is None or not exec_ctx.model:
        return 0.0, "unknown"
    usage = {
        "input_tokens": _RESEARCH_ASSUMED_INPUT_TOKENS_PER_CALL * estimated_llm_calls,
        "output_tokens": _RESEARCH_ASSUMED_OUTPUT_TOKENS_PER_CALL * estimated_llm_calls,
    }
    cost_usd, cost_status, _source = _calc_cost(
        exec_ctx.model, exec_ctx.last_provider, exec_ctx.last_base_url, usage
    )
    if cost_status == "unknown":
        return 0.0, "unknown"
    # The token counts above are assumed, not measured, so the result is
    # always a projection regardless of how confident the price lookup
    # itself was -- round up to the cent, never under-quote the approver.
    return math.ceil(cost_usd * 100) / 100.0, "estimated"


# ---------------------------------------------------------------------------
# Cost status precedence (used by both Phase 2A accumulation and 2B projection)
# ---------------------------------------------------------------------------

_COST_STATUS_RANK: Dict[str, int] = {
    "actual": 0,
    "estimated": 1,
    "included": 2,
    "unknown": 3,
}


def _worst_cost_status(a: str, b: str) -> str:
    """Return the cost status with the higher uncertainty rank."""
    return a if _COST_STATUS_RANK.get(a, 3) >= _COST_STATUS_RANK.get(b, 3) else b


def _estimate_pending_call_cost(
    model: str,
    provider: str,
    base_url: str,
    request: dict,
) -> Tuple[float, str]:
    """Conservatively estimate the cost of a pending LLM provider call.

    Uses message character count (~4 chars/token) for input estimation and
    1024 tokens as a default output estimate. This is intentionally conservative
    (may over-estimate) to avoid under-estimating budget impact.

    Returns (cost_usd, cost_status). Returns (0.0, "unknown") when the model
    is not in Hermes' pricing table (e.g. local/Ollama models).
    """
    try:
        from agent.usage_pricing import CanonicalUsage, estimate_usage_cost
    except Exception:
        return 0.0, "unknown"
    try:
        messages = request.get("messages") or []
        total_chars = sum(
            len(str(m.get("content") or ""))
            for m in messages
            if isinstance(m, dict)
        )
        # Rough input estimate: 4 chars per token, minimum 100
        estimated_input = max(total_chars // 4, 100)
        # Output estimate: 1024 tokens as a conservative default
        estimated_output = 1024
        cu = CanonicalUsage(
            input_tokens=estimated_input,
            output_tokens=estimated_output,
        )
        result = estimate_usage_cost(model, cu, provider=provider or None, base_url=base_url or None)
        cost = float(result.amount_usd or 0.0)
        return cost, str(result.status)
    except Exception:
        return 0.0, "unknown"


class AmpGovernancePlugin:
    def __init__(self) -> None:
        self._config: AmpConfig = load_config()
        self._client = AmpClient(self._config)
        self._store = SessionStore()
        self._exec_contexts = ExecutionContextStore()
        self._warned_unconfigured = False
        self._blocked_turn_messages: Dict[str, str] = {}
        self._dispatch_tool = None
        # api_request_id -> truncated triggering user message, stashed by
        # pre_api_request and consumed by post_api_request to build LLM trace
        # entries (see _save_llm_trace). Capped so a request that never gets
        # a matching post_api_request (e.g. an error path) can't leak memory.
        self._pending_prompts: Dict[str, str] = {}
        self._pending_prompts_lock = threading.Lock()

    def _warn_unconfigured(self) -> None:
        if self._warned_unconfigured:
            return
        self._warned_unconfigured = True
        logger.warning(
            "amp-governance is not fully configured. Required env vars: "
            "AMP_BACKEND_URL, AMP_API_KEY, AMP_ORG_ID, AMP_USERNAME, AMP_AGENT_NAME or AGENT_NAME"
        )

    def _config_fingerprint(self) -> str:
        return agent_fingerprint(
            self._config.backend_url,
            self._config.org_id,
            self._config.agent_name,
            self._config.api_key,
        )

    def _ensure_instance(
        self,
        session_id: str,
        *,
        model: str = "",
        platform: str = "",
    ) -> str:
        fingerprint = self._config_fingerprint()
        record = self._store.get(session_id)
        if record and record.agent_fingerprint == fingerprint:
            return record.instance_id
        reconnected = record is not None
        instance_id = self._client.init_instance(session_id, model, platform)
        self._store.put(
            SessionRecord(
                session_id=session_id,
                instance_id=instance_id,
                model=model,
                platform=platform,
                agent_fingerprint=fingerprint,
            )
        )
        if reconnected:
            self._safe_log(instance_id, f"AMP-governed Hermes session re-initialized after config change | platform={platform} | model={model}")
        else:
            self._safe_log(instance_id, f"AMP-governed Hermes session started | platform={platform} | model={model}")
        return instance_id

    def _safe_log(self, instance_id: str, message: str, *, level: str = "INFO") -> None:
        try:
            self._client.log(instance_id, message, level=level)
        except Exception as exc:
            logger.warning("amp-governance log failed: %s", exc)

    def _block_message(self, reason: str) -> Dict[str, str]:
        return {"action": "block", "message": reason}

    def _record_blocked_turn(self, session_id: str, reason: str) -> None:
        if not session_id:
            return
        self._blocked_turn_messages[session_id] = reason

    def _consume_blocked_turn_message(self, session_id: str) -> str:
        if not session_id:
            return ""
        return self._blocked_turn_messages.pop(session_id, "")

    def attach_context(self, ctx: Any) -> None:
        self._dispatch_tool = getattr(ctx, "dispatch_tool", None)

    def _notify_user(self, message: str) -> None:
        """Send a best-effort governance notification to the originating channel."""
        notify_user(
            self._dispatch_tool,
            message,
            notifications_enabled=self._config.notifications_enabled,
        )

    @staticmethod
    def _format_cost_context(
        current_cost: float,
        current_status: str,
        est_next: float,
        est_next_status: str,
        projected_total: float,
        projected_status: str,
    ) -> str:
        """Format a concise cost summary for HITL pause notifications."""
        unknown = "unknown"

        def _fmt(cost: float, status: str) -> str:
            if status == unknown:
                return "unknown"
            return f"${cost:.2f}"

        lines = [
            f"Current cost: {_fmt(current_cost, current_status)}",
            f"Estimated next call: {_fmt(est_next, est_next_status)}",
            f"Projected total after next call: {_fmt(projected_total, projected_status)}",
        ]
        return "\n".join(lines)

    def _evaluate_llm_governance(
        self,
        exec_ctx: ExecutionContext,
        model: str,
        provider: str,
        base_url: str,
        session_id: str,
        request: dict,
        api_call_count: int,
    ) -> Tuple[str, str]:
        """Evaluate LLM governance policy before a provider call.

        Returns (decision, reason) where decision is one of "allow" or "block".
        Handles HITL polling internally, blocking the current thread until a
        decision is received or the timeout expires.
        """
        from datetime import datetime, timezone

        instance_id = exec_ctx.instance_id
        exec_ctx.status = "policy_check"
        exec_ctx.policy_eval_count += 1
        exec_ctx.last_policy_eval_at = datetime.now(timezone.utc).isoformat()

        # Cost snapshot for this evaluation
        current_cost = exec_ctx.total_cost_usd
        current_status = exec_ctx.cost_status

        # Estimate pending call cost
        est_next, est_next_status = _estimate_pending_call_cost(model, provider, base_url, request)

        # Project total = accumulated + estimated next call
        projected_total = current_cost + est_next
        projected_status = _worst_cost_status(current_status, est_next_status)

        self._safe_log(
            instance_id,
            f"Pre-LLM policy eval | execution_id={exec_ctx.execution_id} | "
            f"model={model} | llm_call=#{api_call_count} | "
            f"current_cost=${current_cost:.2f} ({current_status}) | "
            f"est_next=${est_next:.2f} ({est_next_status}) | "
            f"projected=${projected_total:.2f} ({projected_status})",
        )

        try:
            response = self._client.request_llm_hitl(
                instance_id,
                execution_id=exec_ctx.execution_id,
                model=model,
                provider=provider,
                session_id=session_id,
                current_cost_usd=current_cost,
                current_cost_status=current_status,
                approved_budget_usd=exec_ctx.approved_budget_usd,
                estimated_next_call_cost_usd=est_next,
                estimated_next_call_cost_status=est_next_status,
                projected_total_cost_usd=projected_total,
                projected_total_cost_status=projected_status,
                input_tokens_total=exec_ctx.input_tokens,
                output_tokens_total=exec_ctx.output_tokens,
                cache_read_tokens_total=exec_ctx.cache_read_tokens,
                cache_write_tokens_total=exec_ctx.cache_write_tokens,
                reasoning_tokens_total=exec_ctx.reasoning_tokens,
                total_tokens=exec_ctx.total_tokens,
                llm_calls_total=exec_ctx.llm_calls,
            )
        except Exception as exc:
            self._safe_log(
                instance_id,
                f"AMP unavailable for LLM policy check: {exc}",
                level="WARN",
            )
            exec_ctx.status = "running"
            if self._config.llm_governance_fail_closed:
                self._safe_log(
                    instance_id,
                    "Fail-closed: blocking LLM call due to AMP unavailability",
                )
                exec_ctx.llm_blocked += 1
                exec_ctx.status = "blocked"
                return "block", "AMP governance unavailable; LLM call blocked (fail-closed)."
            self._safe_log(
                instance_id,
                "Fail-open: allowing LLM call despite AMP unavailability",
            )
            return "allow", ""

        status = str(response.get("status") or "").strip().lower()
        reason = str(response.get("reason") or response.get("information") or "").strip()
        exec_ctx.last_policy_result = status

        self._safe_log(
            instance_id,
            f"LLM policy decision | status={status or 'unknown'}"
            + (f" | {reason}" if reason else ""),
        )

        if status == "no_policy":
            exec_ctx.llm_blocked += 1
            exec_ctx.status = "blocked"
            return "block", f'No active AMP governance policy for agent "{self._config.agent_name}".'

        if status in {"no-hitl", "allow", "allowed", "approved"}:
            exec_ctx.status = "running"
            self._safe_log(
                instance_id,
                f"LLM call allowed | execution_id={exec_ctx.execution_id} | call=#{api_call_count}",
            )
            return "allow", ""

        if status in {"pending", "waiting-for-response"} or response.get("workitem_id"):
            workitem_id = str(response.get("workitem_id") or "").strip()
            exec_ctx.hitl_workitem_id = workitem_id
            exec_ctx.status = "hitl_pending"
            cost_ctx = self._format_cost_context(
                current_cost, current_status,
                est_next, est_next_status,
                projected_total, projected_status,
            )
            self._notify_user(
                f"Execution paused pending approval in AMP.\n{cost_ctx}"
            )
            self._safe_log(
                instance_id,
                f"LLM HITL requested | workitem_id={workitem_id} | waiting_for={self._config.username}",
                level="WARN",
            )
            deadline = time.time() + (self._config.hitl_timeout_minutes * 60)
            while time.time() < deadline:
                time.sleep(max(self._config.hitl_poll_interval_seconds, 1))
                try:
                    decision = self._client.get_hitl_decision(instance_id)
                except AmpClientError as exc:
                    logger.warning("amp-governance LLM decision poll failed: %s", exc)
                    continue
                if str(decision.get("status") or "").strip().lower() != "complete":
                    continue
                resolution = str(decision.get("resolution") or "").strip().lower()
                info = str(decision.get("information") or "").strip()
                self._safe_log(
                    instance_id,
                    f"LLM HITL resolution | resolution={resolution}"
                    + (f" | {info}" if info else ""),
                    level="WARN" if resolution not in {"approve", "approved", "modify", "modified"} else "INFO",
                )
                if resolution in {"approve", "approved", "modify", "modified"}:
                    exec_ctx.llm_hitl_approved += 1
                    exec_ctx.status = "resuming"
                    self._notify_user(
                        "Approval received. Resuming LLM execution."
                        + (f"\n{info}" if info else "")
                    )
                    self._safe_log(
                        instance_id,
                        f"LLM HITL approved | execution_id={exec_ctx.execution_id} | call=#{api_call_count}",
                    )
                    return "allow", ""
                exec_ctx.llm_blocked += 1
                exec_ctx.status = "blocked"
                self._notify_user(
                    f"AMP reviewer rejected LLM execution."
                    + (f" {info}" if info else "")
                )
                self._safe_log(
                    instance_id,
                    f"LLM HITL rejected | execution_id={exec_ctx.execution_id} | call=#{api_call_count}"
                    + (f" | {info}" if info else ""),
                    level="WARN",
                )
                return "block", f"LLM execution rejected by AMP HITL review.{' ' + info if info else ''}"
            # Timeout
            exec_ctx.llm_blocked += 1
            exec_ctx.status = "blocked"
            self._notify_user("AMP review timed out. LLM execution was blocked.")
            self._safe_log(
                instance_id,
                f"LLM HITL timed out | execution_id={exec_ctx.execution_id} | call=#{api_call_count}",
                level="WARN",
            )
            return "block", "LLM execution timed out waiting for AMP HITL approval."

        # Unexpected status
        if self._config.llm_governance_fail_closed:
            exec_ctx.llm_blocked += 1
            exec_ctx.status = "blocked"
            self._safe_log(
                instance_id,
                f"LLM unexpected AMP status '{status}'; fail-closed → block",
                level="WARN",
            )
            return "block", f'AMP returned unexpected status "{status or "unknown"}"; LLM call blocked (fail-closed).'
        exec_ctx.status = "running"
        self._safe_log(
            instance_id,
            f"LLM unexpected AMP status '{status}'; fail-open → allow",
            level="WARN",
        )
        return "allow", ""

    def llm_execution_middleware(
        self,
        request: dict,
        next_call: Any,
        session_id: str = "",
        model: str = "",
        provider: str = "",
        base_url: str = "",
        api_call_count: int = 0,
        **_: Any,
    ) -> Any:
        """llm_execution middleware: evaluate LLM governance before every provider
        call made through Hermes' normal conversational agent loop (the
        run_llm_execution_middleware() call site in agent/conversation_loop.py).

        This does NOT cover LLM calls a plugin makes via ctx.llm.complete() /
        complete_structured() — that facade calls agent/auxiliary_client.py::
        call_llm() directly and never reaches this middleware or pre_api_request/
        post_api_request. See README.md "Scope: what Phase 2A/2B actually covers".

        Only active when AMP_LLM_GOVERNANCE_ENABLED=true and
        AMP_LLM_GOVERNANCE_MODE=enforce, and when the installed Hermes version
        provides LLMExecutionBlocked (nousresearch/hermes-agent#64662).
        """
        if (
            not self._config.llm_governance_enabled
            or self._config.llm_governance_mode != "enforce"
        ):
            return next_call(request)

        exec_ctx = self._exec_contexts.get(session_id)
        if exec_ctx is None:
            # Session context not tracked (session_start failed or governance disabled at start)
            return next_call(request)

        decision, reason = self._evaluate_llm_governance(
            exec_ctx, model, provider, base_url, session_id, request, api_call_count
        )

        if decision == "block":
            if _LLMExecutionBlocked is None:
                # Should not happen — middleware is only registered when available
                logger.error(
                    "amp-governance: LLMExecutionBlocked is unavailable but enforcement "
                    "was triggered. Allowing call to avoid crash. Check plugin registration."
                )
                return next_call(request)
            raise _LLMExecutionBlocked(
                reason,
                metadata={
                    "execution_id": exec_ctx.execution_id,
                    "current_cost_usd": exec_ctx.total_cost_usd,
                    "cost_status": exec_ctx.cost_status,
                    "policy_result": exec_ctx.last_policy_result or "unknown",
                    "session_id": session_id,
                },
            )

        exec_ctx.status = "running"
        return next_call(request)

    def _evaluate_governance(
        self,
        instance_id: str,
        action: NormalizedAction,
    ) -> Optional[Dict[str, str]]:
        self._safe_log(
            instance_id,
            f"Policy check | raw_tool={action.raw_tool_name} | tool={action.tool} | action={action.action} | context={action.context}",
        )
        try:
            response = self._client.request_hitl(instance_id, action)
        except Exception as exc:
            message = f"AMP governance is unavailable; blocked {action.raw_tool_name}. Error: {exc}"
            self._safe_log(instance_id, message, level="ERROR")
            if self._config.fail_closed:
                return self._block_message(message)
            return None

        status = str(response.get("status") or "").strip().lower()
        reason = str(response.get("reason") or response.get("information") or "").strip()
        self._safe_log(
            instance_id,
            f"Policy decision | raw_tool={action.raw_tool_name} | status={status or 'unknown'}{f' | {reason}' if reason else ''}",
        )

        if status == "no_policy":
            return self._block_message(
                f'No active AMP governance policy for agent "{self._config.agent_name}".'
            )

        if status in {"policy_error", "config_error", "error"}:
            return self._block_message(
                reason or (
                    f'AMP policy for agent "{self._config.agent_name}" is misconfigured; '
                    f'"{action.raw_tool_name}" was blocked.'
                )
            )

        if status in {"no-hitl", "allow", "allowed", "approved"}:
            return None

        if status in {"pending", "waiting-for-response"} or response.get("workitem_id"):
            deadline = time.time() + (self._config.hitl_timeout_minutes * 60)
            self._safe_log(
                instance_id,
                f"HITL requested | raw_tool={action.raw_tool_name} | waiting_for={self._config.username}",
                level="WARN",
            )
            self._notify_user(
                f'AMP is waiting for a human reviewer to approve "{action.raw_tool_name}" before continuing. '
                "This action is paused pending review."
            )
            while time.time() < deadline:
                time.sleep(max(self._config.hitl_poll_interval_seconds, 1))
                try:
                    decision = self._client.get_hitl_decision(instance_id)
                except AmpClientError as exc:
                    logger.warning("amp-governance decision poll failed: %s", exc)
                    continue
                if str(decision.get("status") or "").strip().lower() != "complete":
                    continue
                resolution = str(decision.get("resolution") or "").strip().lower()
                info = str(decision.get("information") or "").strip()
                self._safe_log(
                    instance_id,
                    f"HITL resolution | raw_tool={action.raw_tool_name} | resolution={resolution}{f' | {info}' if info else ''}",
                    level="WARN" if resolution not in {"approve", "approved", "modify", "modified"} else "INFO",
                )
                if resolution in {"approve", "approved", "modify", "modified"}:
                    approval_message = (
                        f'AMP review approved "{action.raw_tool_name}" with modifications. Continuing now.'
                        if resolution in {"modify", "modified"}
                        else f'AMP review approved "{action.raw_tool_name}". Continuing now.'
                    )
                    self._notify_user(approval_message)
                    return None
                self._notify_user(
                    f'AMP reviewer rejected "{action.raw_tool_name}".{f" {info}" if info else ""}'
                )
                return self._block_message(
                    f'{action.raw_tool_name} was rejected by AMP HITL review.{f" {info}" if info else ""}'
                )
            self._notify_user(
                f'AMP review timed out for "{action.raw_tool_name}". The action was blocked.'
            )
            return self._block_message(
                f'{action.raw_tool_name} timed out waiting for AMP HITL approval.'
            )

        if self._config.fail_closed:
            return self._block_message(
                f'AMP returned unexpected status "{status or "unknown"}" for {action.raw_tool_name}; blocked.'
            )
        return None

    @staticmethod
    def _plan_result(
        status: str,
        *,
        plan_id: Optional[str] = None,
        approved_budget_usd: Optional[float] = None,
        reason: str = "",
        workitem_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "status": status,
            "plan_id": plan_id,
            "approved_budget_usd": approved_budget_usd,
            "reason": reason,
            "workitem_id": workitem_id,
        }

    def _apply_approved_budget(self, session_id: str, instance_id: str, approved_budget_usd: float) -> None:
        """Initialize the session's runtime budget from an approved plan.

        Creates the ExecutionContext if on_session_start hadn't already (Phase 2B's
        llm_execution_middleware silently no-ops without one), so an approved plan
        always results in an enforceable budget.
        """
        exec_ctx = self._exec_contexts.get(session_id)
        if exec_ctx is None:
            exec_ctx = self._exec_contexts.create(session_id, instance_id)
        exec_ctx.approved_budget_usd = approved_budget_usd

    def evaluate_proposed_plan(self, session_id: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a proposed execution plan to AMP for governance approval (Phase 3A).

        This is the public plan-governance interface: it accepts a plan dict,
        normalizes it into AMP policy signals, submits it through the existing
        /api/hitl/request path, and — on approval — initializes the session's
        runtime budget (ExecutionContext.approved_budget_usd) that Phase 2B
        enforces on every subsequent LLM call.

        AHP does not interpret plan['payload'] or decide what counts as a valid
        plan beyond the two governance-relevant fields (plan_type, projected_cost_usd);
        composing the plan (reading config, calling a planning LLM, reacting to a
        chat trigger) is the caller's responsibility, not AHP's.
        """
        plan = plan if isinstance(plan, dict) else {}
        plan_type = str(plan.get("plan_type") or "").strip()
        projected_cost_usd = plan.get("projected_cost_usd")

        if not plan_type:
            return self._plan_result("error", reason="plan_type is required")
        if (
            not isinstance(projected_cost_usd, (int, float))
            or isinstance(projected_cost_usd, bool)
            or projected_cost_usd < 0
        ):
            return self._plan_result("error", reason="projected_cost_usd must be a non-negative number")

        if not self._config.is_configured:
            self._warn_unconfigured()
            return self._plan_result("error", reason="AMP governance is not configured")

        plan_id = str(uuid.uuid4())
        summary = str(plan.get("summary") or "")
        projected_cost_status = str(plan.get("projected_cost_status") or "estimated")
        estimated_llm_calls = int(plan.get("estimated_llm_calls") or 0)
        estimated_tool_calls = int(plan.get("estimated_tool_calls") or 0)
        estimated_duration_minutes = int(plan.get("estimated_duration_minutes") or 0)
        work_units_total = int(plan.get("work_units_total") or 0)
        payload = plan.get("payload") or {}
        cost = float(projected_cost_usd)

        try:
            instance_id = self._ensure_instance(session_id)
        except Exception as exc:
            return self._plan_result("error", plan_id=plan_id, reason=f"AMP instance init failed: {exc}")

        self._safe_log(
            instance_id,
            f"Plan submitted | plan_id={plan_id} | plan_type={plan_type} | "
            f"projected_cost=${cost:.2f} ({projected_cost_status})",
        )

        try:
            response = self._client.request_plan_approval(
                instance_id,
                plan_id=plan_id,
                plan_type=plan_type,
                summary=summary,
                projected_cost_usd=cost,
                projected_cost_status=projected_cost_status,
                estimated_llm_calls=estimated_llm_calls,
                estimated_tool_calls=estimated_tool_calls,
                estimated_duration_minutes=estimated_duration_minutes,
                work_units_total=work_units_total,
                payload=payload,
            )
        except Exception as exc:
            self._safe_log(instance_id, f"AMP unavailable for plan approval: {exc}", level="ERROR")
            return self._plan_result("error", plan_id=plan_id, reason=f"AMP governance unavailable: {exc}")

        status = str(response.get("status") or "").strip().lower()
        reason = str(response.get("reason") or response.get("information") or "").strip()

        if status == "no_policy":
            self._safe_log(instance_id, f"Plan decision | status=no_policy | plan_id={plan_id}", level="WARN")
            return self._plan_result(
                "rejected",
                plan_id=plan_id,
                reason=reason or f'No active AMP governance policy for agent "{self._config.agent_name}".',
            )

        if status in {"no-hitl", "allow", "allowed", "approved"}:
            self._apply_approved_budget(session_id, instance_id, cost)
            self._safe_log(instance_id, f"Plan approved | plan_id={plan_id} | budget=${cost:.2f}")
            # Distinct from the pending/HITL branch below: this plan never
            # reached a human reviewer at all -- say so explicitly, so
            # "why wasn't I asked to approve this" isn't a silent question.
            self._notify_user(
                "Execution plan approved automatically -- no human review "
                f"required under AMP's current policy.\nBudget: ${cost:.2f}\n"
                "Researching now -- the report will follow shortly."
            )
            return self._plan_result("approved", plan_id=plan_id, approved_budget_usd=cost, reason=reason)

        if status in {"pending", "waiting-for-response"} or response.get("workitem_id"):
            # workitem_id is always empty at this point -- AMP's synchronous
            # /api/hitl/request response for a still-pending decision doesn't
            # include it yet (it's created moments later by a separate
            # internal service call); a log line for it here would always
            # show workitem_id= with nothing after it, so we don't write one.
            # AMP's own "[HITL] Created work item..." log line (a different
            # service) is the one with the real id.
            workitem_id = str(response.get("workitem_id") or "").strip()
            self._notify_user(
                f"Execution plan submitted for AMP approval.\n"
                f"Projected cost: ${cost:.2f}\n"
                "Awaiting approval in AMP..."
            )

            deadline = time.time() + (self._config.hitl_timeout_minutes * 60)
            while time.time() < deadline:
                time.sleep(max(self._config.hitl_poll_interval_seconds, 1))
                try:
                    decision = self._client.get_hitl_decision(instance_id)
                except AmpClientError as exc:
                    logger.warning("amp-governance plan decision poll failed: %s", exc)
                    continue
                if str(decision.get("status") or "").strip().lower() != "complete":
                    continue
                resolution = str(decision.get("resolution") or "").strip().lower()
                info = str(decision.get("information") or "").strip()
                if resolution in {"approve", "approved", "modify", "modified"}:
                    self._apply_approved_budget(session_id, instance_id, cost)
                    self._notify_user(
                        "Execution plan approved by AMP." + (f"\n{info}" if info else "")
                        + "\nResearching now -- the report will follow shortly."
                    )
                    self._safe_log(
                        instance_id,
                        f"Plan HITL approved | plan_id={plan_id} | budget=${cost:.2f}",
                    )
                    return self._plan_result(
                        "approved", plan_id=plan_id, approved_budget_usd=cost,
                        reason=info, workitem_id=workitem_id,
                    )
                self._notify_user(
                    "AMP reviewer rejected the execution plan." + (f" {info}" if info else "")
                )
                self._safe_log(
                    instance_id,
                    f"Plan HITL rejected | plan_id={plan_id}" + (f" | {info}" if info else ""),
                    level="WARN",
                )
                return self._plan_result(
                    "rejected", plan_id=plan_id,
                    reason=info or "Execution plan rejected by AMP HITL review.",
                    workitem_id=workitem_id,
                )
            self._notify_user("AMP review of the execution plan timed out.")
            self._safe_log(instance_id, f"Plan HITL timed out | plan_id={plan_id}", level="WARN")
            return self._plan_result(
                "timed_out", plan_id=plan_id,
                reason="Timed out waiting for AMP HITL approval.",
                workitem_id=workitem_id,
            )

        self._safe_log(instance_id, f"Plan unexpected AMP status '{status}' | plan_id={plan_id}", level="WARN")
        return self._plan_result(
            "error", plan_id=plan_id,
            reason=f'AMP returned unexpected status "{status or "unknown"}".',
        )

    def on_session_start(self, session_id: str = "", model: str = "", platform: str = "", **_: Any) -> None:
        if not session_id:
            return
        if not self._config.is_configured:
            self._warn_unconfigured()
            return
        try:
            instance_id = self._ensure_instance(session_id, model=model, platform=platform)
            if self._config.llm_governance_enabled:
                self._exec_contexts.create(
                    session_id,
                    instance_id,
                    model=model,
                    platform=platform,
                )
                # Warn once per session when enforcement is configured but unavailable
                if (
                    self._config.llm_governance_mode == "enforce"
                    and not _LLM_BLOCKED_AVAILABLE
                ):
                    logger.error(
                        "amp-governance: AMP_LLM_GOVERNANCE_MODE=enforce is set but the "
                        "installed Hermes does not provide LLMExecutionBlocked "
                        "(nousresearch/hermes-agent#64662). Enforcement is unavailable — "
                        "LLM calls will proceed without enforcement. "
                        "Set AMP_LLM_GOVERNANCE_MODE=observe to suppress this error, "
                        "or upgrade Hermes to a version that includes the upstream fix."
                    )
                    self._notify_user(
                        "AMP LLM enforcement is configured but cannot be activated: "
                        "the installed Hermes version does not support LLMExecutionBlocked "
                        "(see nousresearch/hermes-agent#64662). "
                        "LLM calls will proceed without enforcement."
                    )
        except Exception as exc:
            logger.warning("amp-governance session init failed: %s", exc)

    def on_session_finalize(self, session_id: str = "", reason: str = "", **_: Any) -> None:
        if not session_id:
            return
        self._blocked_turn_messages.pop(session_id, None)

        # Log execution summary before closing the session
        if self._config.llm_governance_enabled:
            exec_ctx = self._exec_contexts.remove(session_id)
            if exec_ctx is not None:
                exec_ctx.status = "finished"
                record = self._store.get(session_id)
                instance_id = record.instance_id if record else exec_ctx.instance_id
                try:
                    self._client.log_execution_summary(instance_id, exec_ctx.to_summary_dict())
                except Exception as exc:
                    logger.warning("amp-governance execution summary log failed: %s", exc)

        record = self._store.get(session_id)
        if not record:
            return
        try:
            self._safe_log(record.instance_id, f"Finalizing session | reason={reason or 'unknown'}")
            self._client.set_state(record.instance_id, "finished")
        except Exception as exc:
            logger.warning("amp-governance finalize failed: %s", exc)
        finally:
            self._store.delete(session_id)

    def pre_api_request(
        self,
        api_request_id: str = "",
        user_message: str = "",
        session_id: str = "",
        provider: str = "",
        base_url: str = "",
        **_: Any,
    ) -> None:
        """Stash the triggering user message for this specific provider call,
        keyed by api_request_id, so post_api_request can attach it to the
        LLM trace entry it saves. Only active when LLM governance is enabled
        (same gate as post_api_request) — this hook exists purely to feed
        that one.

        Also records provider/base_url onto the session's ExecutionContext
        immediately (not just in post_api_request's accumulate step). This
        call hasn't completed yet, so there's no usage/cost to accumulate --
        but a plan-evaluation tool call requested by *this same* completion
        (e.g. amp_evaluate_research_plan) dispatches before post_api_request
        for it ever fires, since tool dispatch happens between "response
        received" and "usage accumulated". _project_research_cost needs
        provider/base_url to project a plan's cost before that ordering
        would otherwise make them available.
        """
        if not self._config.llm_governance_enabled or not api_request_id:
            return
        if session_id:
            exec_ctx = self._exec_contexts.get(session_id)
            if exec_ctx is not None:
                exec_ctx.last_provider = provider or exec_ctx.last_provider
                exec_ctx.last_base_url = base_url or exec_ctx.last_base_url
        with self._pending_prompts_lock:
            if len(self._pending_prompts) >= _MAX_PENDING_PROMPTS:
                self._pending_prompts.pop(next(iter(self._pending_prompts)), None)
            self._pending_prompts[api_request_id] = _truncate(user_message, limit=_TRACE_FIELD_CHARS)

    def _pop_pending_prompt(self, api_request_id: str) -> str:
        if not api_request_id:
            return ""
        with self._pending_prompts_lock:
            return self._pending_prompts.pop(api_request_id, "")

    def pre_llm_call(self, session_id: str = "", user_message: str = "", model: str = "", platform: str = "", **_: Any) -> None:
        if not session_id or not self._config.is_configured:
            if not self._config.is_configured:
                self._warn_unconfigured()
            return None
        try:
            self._blocked_turn_messages.pop(session_id, None)
            instance_id = self._ensure_instance(session_id, model=model, platform=platform)
            if user_message:
                self._safe_log(instance_id, f"User prompt: {_truncate(user_message)}")
            if _mentions_research(user_message):
                context = _build_research_skill_context(user_message)
                self._safe_log(instance_id, "Research routing | injected skill-load directive")
                return {"context": context}
            if _needs_live_web_search(user_message):
                context = _build_live_search_context(user_message)
                self._safe_log(instance_id, "Freshness routing | injected live-search context")
                return {"context": context}
        except Exception as exc:
            logger.warning("amp-governance pre_llm_call failed: %s", exc)
        return None

    def pre_tool_call(
        self,
        tool_name: str = "",
        args: Any = None,
        session_id: str = "",
        model: str = "",
        platform: str = "",
        **_: Any,
    ) -> Optional[Dict[str, str]]:
        action = normalize_tool_call(tool_name, args)
        if action is None:
            return None
        if not self._config.is_configured:
            self._warn_unconfigured()
            if self._config.fail_closed:
                return self._block_message(
                    f'AMP governance is not configured; blocked governed tool "{tool_name}".'
                )
            return None
        try:
            instance_id = self._ensure_instance(session_id, model=model, platform=platform)
            decision = self._evaluate_governance(instance_id, action)
            if decision and decision.get("action") == "block":
                self._record_blocked_turn(session_id, str(decision.get("message") or ""))
            return decision
        except Exception as exc:
            logger.warning("amp-governance pre_tool_call failed: %s", exc)
            if self._config.fail_closed:
                message = (
                    f'AMP governance failed while checking "{tool_name}": {exc}'
                )
                self._record_blocked_turn(session_id, message)
                return self._block_message(message)
            return None

    def post_tool_call(
        self,
        tool_name: str = "",
        args: Any = None,
        session_id: str = "",
        status: str = "",
        duration_ms: Any = None,
        error_message: str = "",
        result: Any = None,
        **_: Any,
    ) -> None:
        action = normalize_tool_call(tool_name, args)
        if action is None:
            return
        if not self._config.is_configured:
            return
        # Count this governed tool call in the execution context
        if self._config.llm_governance_enabled:
            exec_ctx = self._exec_contexts.get(session_id)
            if exec_ctx is not None:
                exec_ctx.tool_calls += 1
        record = self._store.get(session_id)
        if not record:
            return
        summary = f"Tool result | raw_tool={tool_name} | normalized={action.tool}/{action.action} | status={status or 'unknown'}"
        if duration_ms is not None:
            summary += f" | duration_ms={duration_ms}"
        if error_message:
            summary += f" | error={_truncate(error_message, 180)}"
        elif result:
            summary += f" | result={_truncate(str(result), 180)}"
        self._safe_log(
            record.instance_id,
            summary,
            level="ERROR" if (status or "").lower() in {"error", "blocked"} else "INFO",
        )

    def post_api_request(
        self,
        session_id: str = "",
        api_request_id: str = "",
        model: str = "",
        provider: str = "",
        base_url: str = "",
        api_call_count: int = 0,
        api_duration: float = 0.0,
        usage: Optional[Dict] = None,
        assistant_message: Any = None,
        **_: Any,
    ) -> None:
        """
        Observer hook fired after every LLM API call made through Hermes' normal
        conversational agent loop, with normalized usage data. Plugin LLM calls
        via ctx.llm.complete()/complete_structured() do not fire this hook —
        see README.md "Scope: what Phase 2A/2B actually covers".

        Accumulates token and cost metrics into the session's ExecutionContext,
        then logs the event to AMP. Enabled only when AMP_LLM_GOVERNANCE_ENABLED=true.
        Does not block or modify LLM calls.
        """
        if not self._config.llm_governance_enabled:
            return
        if not session_id or not self._config.is_configured:
            return
        exec_ctx = self._exec_contexts.get(session_id)
        if exec_ctx is None:
            return

        # Extract token counts (usage may be None for providers that don't report)
        u = usage or {}
        input_tokens = int(u.get("input_tokens") or 0)
        output_tokens = int(u.get("output_tokens") or 0)
        cache_read = int(u.get("cache_read_tokens") or 0)
        cache_write = int(u.get("cache_write_tokens") or 0)
        reasoning = int(u.get("reasoning_tokens") or 0)

        # Calculate cost (best-effort; unknown if pricing table lacks this model)
        if usage:
            cost_usd, cost_status, cost_source = _calc_cost(model, provider, base_url, u)
        else:
            cost_usd, cost_status, cost_source = 0.0, "unknown", ""

        record = LlmCallRecord(
            api_request_id=api_request_id or "",
            api_call_number=api_call_count,
            model=model or exec_ctx.model,
            provider=provider or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            reasoning_tokens=reasoning,
            cost_usd=cost_usd,
            cost_status=cost_status,
            cost_source=cost_source,
            api_duration=api_duration,
        )
        exec_ctx.accumulate(record)
        exec_ctx.last_provider = provider or exec_ctx.last_provider
        exec_ctx.last_base_url = base_url or exec_ctx.last_base_url

        # Log to AMP (best-effort, non-blocking)
        try:
            self._client.log_llm_event(
                exec_ctx.instance_id,
                record.to_dict(),
                execution_id=exec_ctx.execution_id,
            )
        except Exception as exc:
            logger.warning("amp-governance llm event log failed: %s", exc)

        self._save_llm_trace(exec_ctx.instance_id, api_request_id, model, assistant_message)

    def _save_llm_trace(
        self,
        instance_id: str,
        api_request_id: str,
        model: str,
        assistant_message: Any,
    ) -> None:
        """Best-effort: save what this LLM call was asked and what it decided
        so AMP's "click LLM in the log" panel has something real to show.
        Never raises — a failure here must never affect the LLM call itself
        or the cost/token accumulation above, which already succeeded."""
        try:
            user_message = self._pop_pending_prompt(api_request_id)
            prompt = {"user": user_message} if user_message else None
            reasoning = _extract_reasoning_from_assistant_message(assistant_message)
            answer = _summarize_assistant_answer(assistant_message)
            self._client.save_llm_trace(
                instance_id,
                call_time=datetime.now(timezone.utc).isoformat(),
                model=model,
                prompt=prompt,
                reasoning=reasoning,
                answer=answer or None,
            )
        except Exception as exc:
            logger.warning("amp-governance llm trace save failed: %s", exc)

    def transform_llm_output(
        self,
        response_text: str = "",
        session_id: str = "",
        **_: Any,
    ) -> Optional[str]:
        blocked_reason = self._consume_blocked_turn_message(session_id)
        if not blocked_reason:
            return None
        return "This request is blocked by AMP governance. No action was taken."


_PLUGIN = AmpGovernancePlugin()


def evaluate_proposed_plan(session_id: str, plan: Dict[str, Any]) -> Dict[str, Any]:
    """Public Phase 3A entry point: submit a proposed execution plan for AMP approval.

    Intended for a future caller (e.g. a research-agent skill) to call directly —
    see AmpGovernancePlugin.evaluate_proposed_plan for the full contract.
    """
    return _PLUGIN.evaluate_proposed_plan(session_id, plan)


# ---------------------------------------------------------------------------
# Phase 3B: registered tools for the research-agent skill.
#
# These run as normal tool calls inside the model's own conversational turn —
# deliberately not via ctx.llm, which calls agent/auxiliary_client.py::call_llm()
# and does not fire pre_api_request/post_api_request/llm_execution, so anything
# routed through it would be invisible to Phase 2A/2B governance. Keeping the
# research skill entirely inside normal turns/tool-calls is what makes "AHP
# Phase 2B governs every LLM invocation" actually true for this sample.
# ---------------------------------------------------------------------------

def _current_session_id() -> str:
    """Best-effort current-turn session_id, same pattern as notification.py's
    build_notification_target() — a plain function import failure (no gateway
    context, e.g. CLI/test) degrades to an empty string rather than raising."""
    try:
        from gateway.session_context import get_session_env
    except Exception:
        return ""
    return str(get_session_env("HERMES_SESSION_ID", "") or "").strip()


_EVALUATE_RESEARCH_PLAN_SCHEMA = {
    "name": "amp_evaluate_research_plan",
    "description": (
        "Submit a proposed research execution plan to AMP for governance approval "
        "before starting any research. Blocks until a decision is reached (which may "
        "involve a human reviewer in AMP) and returns the structured decision. Do not "
        "begin researching any topic until this returns status=\"approved\". Do not "
        "include a dollar cost anywhere in the plan -- AHP computes it from "
        "estimated_llm_calls and real pricing, and overwrites anything you put there."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "plan": {
                "type": "object",
                "description": "The proposed execution plan.",
                "properties": {
                    "plan_type": {"type": "string", "description": 'Always "research" for this workflow.'},
                    "summary": {"type": "string"},
                    "estimated_llm_calls": {
                        "type": "integer",
                        "description": "Must be a positive integer -- a reviewer's budget decision depends on it.",
                    },
                    "estimated_tool_calls": {"type": "integer"},
                    "estimated_duration_minutes": {"type": "integer"},
                    "work_units_total": {"type": "integer"},
                    "payload": {
                        "type": "object",
                        "description": (
                            "Opaque caller data (e.g. topics, research_depth). "
                            "Not interpreted by AMP governance."
                        ),
                    },
                },
                "required": ["plan_type", "estimated_llm_calls"],
            },
        },
        "required": ["plan"],
    },
}

_LOAD_RESEARCH_TOPICS_SCHEMA = {
    "name": "amp_load_research_topics",
    "description": "Load and validate the local research topics configuration file (1-5 topics).",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Optional path override. Defaults to ~/.hermes/research_topics.yaml.",
            },
        },
        "required": [],
    },
}

_GOVERNANCE_SUMMARY_SCHEMA = {
    "name": "amp_governance_summary",
    "description": (
        "Return the current session's accumulated AMP governance totals (cost, token, "
        "and call counts) for reporting. Takes no arguments."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

_DEFAULT_RESEARCH_TOPICS_PATH = str(Path.home() / ".hermes" / "research_topics.yaml")


def _tool_amp_evaluate_research_plan(args: dict, **_: Any) -> str:
    plan = (args or {}).get("plan")
    if not isinstance(plan, dict):
        return json.dumps({
            "status": "error", "plan_id": None, "approved_budget_usd": None,
            "reason": "plan must be an object", "workitem_id": None,
        })
    session_id = _current_session_id()
    # projected_cost_usd/projected_cost_status are always computed here, not
    # taken from whatever the model put in the plan -- see
    # _project_research_cost for why an LLM-invented dollar figure isn't
    # trustworthy enough to gate a HITL budget decision on.
    try:
        estimated_llm_calls = int(plan.get("estimated_llm_calls") or 0)
    except (TypeError, ValueError):
        estimated_llm_calls = 0
    if estimated_llm_calls <= 0:
        # A plan with no call estimate can only ever project to $0.00
        # (unknown) -- reject it here rather than letting a hollow plan
        # through to an approval decision no reviewer could meaningfully
        # make. Forces the model to actually produce the full plan schema
        # instead of skipping straight to submission.
        return json.dumps({
            "status": "error", "plan_id": None, "approved_budget_usd": None,
            "reason": "plan.estimated_llm_calls must be a positive integer",
            "workitem_id": None,
        })
    exec_ctx = _PLUGIN._exec_contexts.get(session_id) if session_id else None
    cost_usd, cost_status = _project_research_cost(exec_ctx, estimated_llm_calls)
    plan["projected_cost_usd"] = cost_usd
    plan["projected_cost_status"] = cost_status
    if exec_ctx is not None:
        exec_ctx.last_plan_projected_cost_usd = cost_usd
        exec_ctx.last_plan_projected_cost_status = cost_status
    result = _PLUGIN.evaluate_proposed_plan(session_id, plan)
    if result.get("status") == "approved" and session_id:
        # Re-fetch: evaluate_proposed_plan's _apply_approved_budget lazily
        # creates the ExecutionContext on approval if on_session_start
        # somehow hadn't already, so exec_ctx above may be stale/None.
        approved_exec_ctx = _PLUGIN._exec_contexts.get(session_id)
        if approved_exec_ctx is not None:
            approved_exec_ctx.plan_approvals_count += 1
    return json.dumps(result)


def _tool_amp_load_research_topics(args: dict, **_: Any) -> str:
    path = (args or {}).get("path") or _DEFAULT_RESEARCH_TOPICS_PATH
    try:
        topics = load_research_topics(path)
    except ValueError as exc:
        return json.dumps({"status": "error", "reason": str(exc)})
    return json.dumps({"status": "ok", **topics})


def _format_governance_report_text(summary: dict) -> str:
    """Deterministically render the exact "Governance summary" text block
    research-agent's step 8 pastes verbatim into its final reply.

    This exists because asking the model to *compose* this section from
    numbers it read across several earlier tool calls (and, worse, from
    memory of an earlier turn's report in the same conversation) proved
    unreliable in practice: a long-running session was observed reproducing
    an earlier turn's fabricated figures verbatim in every subsequent
    report, rather than calling amp_governance_summary fresh each time.
    Returning ready-made text removes the opportunity to substitute,
    "improve", or reuse a stale number -- every value here comes straight
    from the session's real accumulated ExecutionContext.
    """
    plan_cost = summary.get("plan_projected_cost_usd")
    plan_cost_status = summary.get("plan_projected_cost_status", "unknown")
    plan_cost_line = f"${plan_cost:.2f} ({plan_cost_status})" if plan_cost is not None else "unavailable"

    approved_budget = summary.get("approved_budget_usd")
    approved_budget_line = f"${approved_budget:.2f}" if approved_budget is not None else "unavailable"

    cost_status = summary.get("cost_status", "unknown")
    if cost_status in ("actual", "estimated"):
        actual_cost_line = f"${float(summary.get('total_cost_usd') or 0.0):.2f}"
    else:
        actual_cost_line = f"cost tracking unavailable for this model (cost_status={cost_status})"

    return (
        "Governance summary\n"
        f"Plan projected cost: {plan_cost_line}\n"
        f"Approved budget: {approved_budget_line}\n"
        f"Actual cost: {actual_cost_line}\n"
        f"LLM calls: {summary.get('llm_calls', 0)}\n"
        f"Tool calls: {summary.get('tool_calls', 0)}\n"
        f"AMP plan approvals: {summary.get('plan_approvals_count', 0)}\n"
        f"Runtime HITL approvals: {summary.get('llm_hitl_approved', 0)}"
    )


def _tool_amp_governance_summary(args: dict, **_: Any) -> str:
    """Returns ExecutionContext.to_summary_dict() plus a ready-to-paste
    'report_text' field (see _format_governance_report_text). The only
    sentinel this adds is status="no_context", which never collides with a
    real execution status, for when nothing has been tracked yet."""
    session_id = _current_session_id()
    exec_ctx = _PLUGIN._exec_contexts.get(session_id) if session_id else None
    if exec_ctx is None:
        return json.dumps({
            "status": "no_context",
            "reason": "No active AMP execution context for this session yet.",
        })
    summary = exec_ctx.to_summary_dict()
    summary["report_text"] = _format_governance_report_text(summary)
    return json.dumps(summary)


def register(ctx) -> None:
    _PLUGIN.attach_context(ctx)
    ctx.register_hook("on_session_start", _PLUGIN.on_session_start)
    ctx.register_hook("on_session_finalize", _PLUGIN.on_session_finalize)
    ctx.register_hook("pre_llm_call", _PLUGIN.pre_llm_call)
    ctx.register_hook("pre_tool_call", _PLUGIN.pre_tool_call)
    ctx.register_hook("post_tool_call", _PLUGIN.post_tool_call)
    ctx.register_hook("pre_api_request", _PLUGIN.pre_api_request)
    ctx.register_hook("post_api_request", _PLUGIN.post_api_request)
    ctx.register_hook("transform_llm_output", _PLUGIN.transform_llm_output)

    from .cli import register_cli, amp_command
    ctx.register_cli_command(
        name="amp",
        help="AMP governance setup commands",
        setup_fn=register_cli,
        handler_fn=amp_command,
        description=(
            "Connect this Hermes agent to AMP in one step: writes AMP_* keys "
            "to .env and restarts the gateway, printing each step as it "
            "happens. Run after `hermes plugins install ... --enable`."
        ),
    )

    # Phase 3B: tools + plugin-bundled skill for the research-agent sample.
    # These are ordinary registered tools dispatched inside the model's own
    # conversational turn — see the comment above _current_session_id() for why.
    ctx.register_tool(
        name="amp_evaluate_research_plan",
        toolset="amp_governance",
        schema=_EVALUATE_RESEARCH_PLAN_SCHEMA,
        handler=_tool_amp_evaluate_research_plan,
        description=_EVALUATE_RESEARCH_PLAN_SCHEMA["description"],
    )
    ctx.register_tool(
        name="amp_load_research_topics",
        toolset="amp_governance",
        schema=_LOAD_RESEARCH_TOPICS_SCHEMA,
        handler=_tool_amp_load_research_topics,
        description=_LOAD_RESEARCH_TOPICS_SCHEMA["description"],
    )
    ctx.register_tool(
        name="amp_governance_summary",
        toolset="amp_governance",
        schema=_GOVERNANCE_SUMMARY_SCHEMA,
        handler=_tool_amp_governance_summary,
        description=_GOVERNANCE_SUMMARY_SCHEMA["description"],
    )
    try:
        ctx.register_skill(
            "research-agent",
            Path(__file__).parent / "skills" / "research-agent" / "SKILL.md",
            description=(
                "Governed research workflow: reads configured topics, gets an "
                "AMP-approved plan, researches, and reports."
            ),
        )
    except Exception as exc:
        logger.warning("amp-governance: research-agent skill registration failed: %s", exc)

    # Phase 2B: register llm_execution middleware only when both enforcement is
    # enabled and the required Hermes capability (LLMExecutionBlocked) is present.
    if (
        _PLUGIN._config.llm_governance_enabled
        and _PLUGIN._config.llm_governance_mode == "enforce"
    ):
        if _LLM_BLOCKED_AVAILABLE:
            ctx.register_middleware("llm_execution", _PLUGIN.llm_execution_middleware)
            logger.info(
                "amp-governance: LLM enforcement enabled (AMP_LLM_GOVERNANCE_MODE=enforce); "
                "llm_execution middleware registered."
            )
        else:
            logger.error(
                "amp-governance: AMP_LLM_GOVERNANCE_MODE=enforce is configured but "
                "LLMExecutionBlocked is not available in the installed Hermes "
                "(nousresearch/hermes-agent#64662). "
                "Enforcement middleware was NOT registered. "
                "LLM calls will proceed without enforcement."
            )
