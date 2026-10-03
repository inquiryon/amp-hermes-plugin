from __future__ import annotations

import atexit
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
import uuid
from typing import Any, Dict, FrozenSet, Iterator, List, Optional, Set, Tuple

from . import hitl_poll
from .amp_client import AmpClient, AmpClientError, request_deadline
from .config import AmpConfig, load_config
from .execution_context import ExecutionContext, ExecutionContextStore, LlmCallRecord
from .hitl_poll import DecisionLedger, PollOutcome, poll_for_decision, reviewer_note
from .host_limits import WaitBudget, compute_budget, hook_timeout_seconds, limits_shorter_than, tool_deadline_seconds
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


# Total time the exit-time finalize (single-query runs) may spend talking to AMP.
_EXIT_FINALIZE_BUDGET_SECONDS = 3.0


def _http_client_error_code(exc: BaseException) -> Optional[int]:
    """HTTP status when AMP answered but refused (4xx) or sent an unusable body (2xx/3xx
    non-JSON); None for outages (network error, timeout, 5xx). A refusal is a decision,
    so AMP_FAIL_CLOSED=false never turns it into an allow."""
    code = getattr(exc, "status_code", None)
    if isinstance(code, int) and code < 500:
        return code
    return None


def _call_fingerprint(action: NormalizedAction, args: Any) -> str:
    """Identify one exact governed call (tool, action, full args) without storing the args."""
    try:
        canonical = json.dumps(args, sort_keys=True, default=str, separators=(",", ":"))
    except (TypeError, ValueError):
        canonical = repr(args)
    digest = hashlib.sha256(canonical.encode("utf-8", "replace")).hexdigest()
    return f"{action.raw_tool_name}|{action.tool}|{action.action}|{digest}"


def _is_single_query_process() -> bool:
    """True under `hermes -z` / `hermes chat -q` (finite, one-turn runs)."""
    return os.environ.get("HERMES_SINGLE_QUERY_SESSION") == "1"


try:  # cross-process review lock (POSIX); other platforms keep the in-process lock only
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows
    _fcntl = None  # type: ignore[assignment]

# How long a request whose workitem never showed up keeps the next call on the same
# instance from starting its own review (contract K2.1, late-workitem guard).
_UNREGISTERED_REVIEW_TTL_SECONDS = 300.0
# Slot-lock polling step while another call (thread or process) holds the review.
_SLOT_POLL_SECONDS = 0.05


class _InstanceFileLock:
    """flock-based lock so two Hermes processes on one AMP instance never interleave
    their baseline/request/poll. Filesystems without flock fall back to no file lock."""

    _warned = False

    def __init__(self, path: Optional[Path]) -> None:
        self._path = path
        self._fd: Optional[int] = None

    def try_acquire(self) -> bool:
        if self._path is None or _fcntl is None:
            return True
        try:
            self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o600)
        except OSError as exc:
            self._warn(exc)
            return True
        try:
            _fcntl.flock(fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False
        except OSError as exc:
            os.close(fd)
            self._warn(exc)
            return True
        self._fd = fd
        return True

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            _fcntl.flock(self._fd, _fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None

    @classmethod
    def _warn(cls, exc: OSError) -> None:
        if not cls._warned:
            cls._warned = True
            logger.warning("amp-governance: cross-process review lock unavailable (%s); "
                           "run one Hermes process per session", exc)


@dataclass(frozen=True)
class _ReviewSlot:
    """Result of entering a per-instance review slot (contract K2.1, action_id off).

    acquired: the per-instance lock was obtained within the time allowed.
    foreign: workitem AMP reported before this call's request (never ours).
    accept_unbound: False when the baseline poll failed, so only a workitem seen
        pending (bound) may decide this call.
    error: the baseline poll's AmpClientError, if any.
    """

    acquired: bool
    foreign: FrozenSet[str] = frozenset()
    accept_unbound: bool = True
    error: Optional[AmpClientError] = None
    # True when the previous call's request never produced a visible workitem: a new
    # review now could bind that late workitem, so the call must not start one.
    unregistered: bool = False
    baseline_workitem: str = ""
    baselined: bool = False


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
        # Workitems already consumed/abandoned per AMP instance (stale-decision guard)
        # and still-pending reviews an identical retry may reuse.
        self._ledger = DecisionLedger()
        # Sessions this process initialised and has not finalised yet (exit-time finalize).
        self._open_sessions: Set[str] = set()
        self._open_sessions_lock = threading.Lock()
        # K2.1: one HITL review at a time per AMP instance when action_id is off, and
        # consumed/abandoned workitems reloaded from the session store after a restart.
        self._instance_locks: Dict[str, threading.Lock] = {}
        self._instance_locks_guard = threading.Lock()
        self._seeded_instances: Set[str] = set()
        # instance -> (newest workitem at that call's baseline, time): a request whose
        # workitem was never seen; see _review_slot.
        self._unregistered: Dict[str, Tuple[str, float]] = {}

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
        if session_id:
            with self._open_sessions_lock:
                self._open_sessions.add(session_id)
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

    def _safe_set_state(self, instance_id: str, state: str) -> None:
        try:
            self._client.set_state(instance_id, state)
        except Exception as exc:
            logger.warning("amp-governance set_state(%s) failed: %s", state, exc)

    def _ensure_ledger_seeded(self, instance_id: str) -> None:
        """Load workitems an earlier process consumed/abandoned for this instance (K2.1)."""
        with self._instance_locks_guard:
            if instance_id in self._seeded_instances:
                return
        try:
            spent = self._store.spent_for_instance(instance_id)
        except Exception as exc:
            logger.warning("amp-governance: could not load consumed workitems: %s", exc)
            return  # retried on the next call
        for workitem_id in spent:
            self._ledger.mark_spent(instance_id, workitem_id)
        with self._instance_locks_guard:
            self._seeded_instances.add(instance_id)

    def _record_spent(self, instance_id: str, workitem_id: str) -> None:
        """Never act on this workitem again, in this process or after a restart."""
        if not workitem_id:
            return
        self._ledger.mark_spent(instance_id, workitem_id)
        try:
            self._store.remember_spent(instance_id, workitem_id)
        except Exception as exc:
            logger.warning("amp-governance: could not persist consumed workitem: %s", exc)

    def _instance_lock(self, instance_id: str) -> threading.Lock:
        with self._instance_locks_guard:
            return self._instance_locks.setdefault(instance_id, threading.Lock())

    def _instance_lock_path(self, instance_id: str) -> Optional[Path]:
        store_path = getattr(self._store, "_path", None)
        if not isinstance(store_path, Path):
            return None
        digest = hashlib.sha256(instance_id.encode("utf-8")).hexdigest()[:16]
        return store_path.parent / "locks" / f"{digest}.lock"

    def _acquire_review_locks(self, instance_id: str, deadline: float) -> Optional[_InstanceFileLock]:
        """Take the in-process and cross-process review locks, or None by ``deadline``."""
        thread_lock = self._instance_lock(instance_id)
        file_lock = _InstanceFileLock(self._instance_lock_path(instance_id))
        while True:
            if thread_lock.acquire(blocking=False):
                if file_lock.try_acquire():
                    return file_lock
                thread_lock.release()
            if time.time() >= deadline:
                return None
            time.sleep(_SLOT_POLL_SECONDS)

    def _note_unregistered(self, instance_id: str, slot: _ReviewSlot) -> None:
        """A request went out but its workitem was never seen (K2.1 late workitem)."""
        if slot.baselined or not slot.accept_unbound:
            self._unregistered[instance_id] = (slot.baseline_workitem, time.time())

    def _take_baseline(self, instance_id: str) -> _ReviewSlot:
        try:
            baseline = self._client.get_hitl_decision(instance_id)
        except AmpClientError as exc:
            return _ReviewSlot(acquired=True, accept_unbound=False, error=exc)
        except Exception as exc:  # local failure: never bind from a pending poll
            return _ReviewSlot(acquired=True, accept_unbound=False, error=AmpClientError(str(exc)))
        newest = ""
        if isinstance(baseline, dict):
            newest = str(baseline.get("workitem_id") or "").strip()
        marker = self._unregistered.get(instance_id)
        if marker is not None:
            marker_newest, noted_at = marker
            if newest == marker_newest and time.time() - noted_at < _UNREGISTERED_REVIEW_TTL_SECONDS:
                return _ReviewSlot(acquired=True, unregistered=True, baseline_workitem=newest, baselined=True)
            self._unregistered.pop(instance_id, None)
            if newest != marker_newest:
                self._record_spent(instance_id, newest)  # most likely the late workitem
        return _ReviewSlot(
            acquired=True,
            foreign=frozenset({newest}) if newest else frozenset(),
            baseline_workitem=newest,
            baselined=True,
        )

    @contextmanager
    def _review_slot(self, instance_id: str, *, deadline: float, baseline: bool = True) -> Iterator[_ReviewSlot]:
        """Hold the instance's review locks and, unless action_id correlates the call,
        take the pre-request baseline (K2.1).

        AMP's get-decision returns the instance's newest workitem regardless of status
        (ORDER BY created DESC LIMIT 1). Polling once before the request, under locks
        that keep other calls (threads and processes) on this instance from creating
        workitems meanwhile, identifies the workitem that is *not* ours.
        """
        self._ensure_ledger_seeded(instance_id)
        file_lock = self._acquire_review_locks(instance_id, deadline)
        if file_lock is None:
            yield _ReviewSlot(acquired=False)
            return
        try:
            slot = self._take_baseline(instance_id) if baseline else _ReviewSlot(acquired=True)
            yield slot
        finally:
            file_lock.release()
            self._instance_lock(instance_id).release()

    def _unregistered_message(self, subject: str) -> str:
        return (
            f"The AMP review requested by the previous governed call has not been registered yet; "
            f"{subject} was blocked so the two decisions cannot be mixed up. Retry shortly."
        )

    def _wait_budget(self, started_at: float, *, include_hook: bool) -> WaitBudget:
        """Time this call may wait for AMP: the shortest of AMP_HITL_TIMEOUT_MINUTES, the
        Hermes hook timeout (hooks only) and the Hermes tool deadline, minus a margin."""
        return compute_budget(
            started_at,
            self._config.hitl_timeout_minutes * 60,
            hook_timeout=hook_timeout_seconds() if include_hook else None,
            tool_deadline=tool_deadline_seconds(),
        )

    @staticmethod
    def _announce_wait(subject: str, workitem_id: str) -> None:
        """One console/log line per review so a blocked-looking CLI run is explained (L5)."""
        line = f"AMP: waiting for approval of {subject} (workitem {workitem_id or 'not yet assigned'})"
        logger.warning(line)
        try:
            sys.stderr.write(line + "\n")
            sys.stderr.flush()
        except Exception:
            pass

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
        """Run the LLM policy check inside the instance's review slot (K2.1)."""
        if not exec_ctx.instance_id:
            exec_ctx.llm_blocked += 1
            exec_ctx.status = "blocked"
            return "block", "No AMP instance for this session; LLM call blocked."
        # One deadline for the lock wait and the review (LLM calls have no Hermes limit).
        review_deadline = time.time() + (self._config.hitl_timeout_minutes * 60)
        with self._review_slot(exec_ctx.instance_id, deadline=review_deadline) as slot:
            if not slot.acquired:
                exec_ctx.llm_blocked += 1
                exec_ctx.status = "blocked"
                return "block", "Another governed call is still waiting for AMP review; LLM call blocked."
            if slot.unregistered:
                exec_ctx.llm_blocked += 1
                exec_ctx.status = "blocked"
                return "block", self._unregistered_message("the LLM call")
            if slot.error is not None:
                refused_code = _http_client_error_code(slot.error)
                if refused_code is not None or self._config.llm_governance_fail_closed:
                    exec_ctx.llm_blocked += 1
                    exec_ctx.status = "blocked"
                    if refused_code is not None:
                        return "block", f"AMP rejected the LLM governance request (HTTP {refused_code}); LLM call blocked."
                    return "block", "AMP governance unavailable; LLM call blocked (fail-closed)."
            return self._evaluate_llm_governance_in_slot(
                exec_ctx, model, provider, base_url, session_id, request, api_call_count, slot,
                review_deadline,
            )

    def _evaluate_llm_governance_in_slot(
        self,
        exec_ctx: ExecutionContext,
        model: str,
        provider: str,
        base_url: str,
        session_id: str,
        request: dict,
        api_call_count: int,
        slot: _ReviewSlot,
        review_deadline: float,
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
            rejected_code = _http_client_error_code(exc)
            if rejected_code is not None:
                # AMP answered and refused (bad key, org, call limit, payload): never an
                # outage, so fail-open does not apply (H2).
                message = (
                    f"AMP rejected the LLM governance request (HTTP {rejected_code}); LLM call blocked. "
                    f"{_truncate(getattr(exc, 'body', '') or str(exc), 200)}"
                ).strip()
                self._safe_log(instance_id, message, level="ERROR")
                exec_ctx.llm_blocked += 1
                exec_ctx.status = "blocked"
                return "block", message
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

        if not isinstance(response, dict):
            response = {}  # falls through to the unexpected-status block below
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

        if status in {"policy_error", "config_error", "error"}:
            exec_ctx.llm_blocked += 1
            exec_ctx.status = "blocked"
            server_message = _truncate(" ".join(str(response.get("message") or reason).split()), 300)
            return "block", server_message or "AMP policy is misconfigured; LLM call blocked."

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
            self._announce_wait("the next LLM call", workitem_id)
            # LLM middleware is not a Hermes hook or tool, so only the AMP timeout applies.
            with request_deadline(review_deadline):
                outcome = poll_for_decision(
                    lambda: self._client.get_hitl_decision(instance_id),
                    instance_id=instance_id,
                    ledger=self._ledger,
                    deadline=review_deadline,
                    interval_seconds=self._config.hitl_poll_interval_seconds,
                    bound_workitem_id=workitem_id,
                    label="LLM ",
                    foreign_workitem_ids=slot.foreign,
                    accept_unbound=slot.accept_unbound,
                )
                self._record_spent(instance_id, outcome.workitem_id)
                if not outcome.workitem_id and outcome.kind not in hitl_poll.TERMINAL_KINDS:
                    self._note_unregistered(instance_id, slot)
                self._safe_set_state(instance_id, "active")
            return self._llm_outcome(exec_ctx, instance_id, api_call_count, outcome)

        # Unexpected status (e.g. a reserved "deny"): AMP answered, so this is not an
        # outage and AMP_LLM_GOVERNANCE_FAIL_CLOSED=false does not apply.
        exec_ctx.llm_blocked += 1
        exec_ctx.status = "blocked"
        self._safe_log(
            instance_id,
            f"LLM unexpected AMP status '{status}'; blocked",
            level="WARN",
        )
        return "block", f'AMP returned unexpected status "{status or "unknown"}"; LLM call blocked.'

    def _llm_outcome(
        self,
        exec_ctx: ExecutionContext,
        instance_id: str,
        api_call_count: int,
        outcome: PollOutcome,
    ) -> Tuple[str, str]:
        """Map a HITL poll outcome to ("allow"|"block", reason). Only APPROVED allows."""
        note = reviewer_note(outcome.information)
        if outcome.kind != hitl_poll.TIMED_OUT:
            self._safe_log(
                instance_id,
                f"LLM HITL resolution | outcome={outcome.kind}"
                + (f" | workitem_id={outcome.workitem_id}" if outcome.workitem_id else "")
                + (f" | {_truncate(outcome.information)}" if outcome.information else ""),
                level="INFO" if outcome.kind == hitl_poll.APPROVED else "WARN",
            )
        if outcome.kind == hitl_poll.APPROVED:
            exec_ctx.llm_hitl_approved += 1
            exec_ctx.status = "resuming"
            self._notify_user("Approval received. Resuming LLM execution." + (f"\n{note}" if note else ""))
            self._safe_log(
                instance_id,
                f"LLM HITL approved | execution_id={exec_ctx.execution_id} | call=#{api_call_count}",
            )
            return "allow", ""

        exec_ctx.llm_blocked += 1
        exec_ctx.status = "blocked"
        suffix = f" {note}" if note else ""
        if outcome.kind == hitl_poll.MODIFIED:
            reason = "LLM call not executed: reviewer requested modifications." + suffix
            self._notify_user(reason)
        elif outcome.kind == hitl_poll.CANCELLED:
            reason = (
                "LLM call blocked: the AMP review was cancelled before a reviewer decided "
                "(not a reviewer rejection)."
            )
            self._notify_user(reason)
        elif outcome.kind == hitl_poll.POLL_REJECTED:
            reason = (
                f"AMP rejected the HITL decision poll for the LLM call (HTTP {outcome.status_code}); "
                f"blocked. {_truncate(outcome.detail, 160)}"
            ).strip()
            self._notify_user(reason)
        elif outcome.kind == hitl_poll.UNVERIFIABLE:
            reason = (
                f"AMP returned an LLM decision not tied to a workitem (method {outcome.detail}); "
                "unverifiable decision, LLM call blocked."
            )
            self._notify_user(reason)
        elif outcome.kind == hitl_poll.UNAVAILABLE:
            reason = f"AMP was unreachable while waiting for the LLM review; blocked. {_truncate(outcome.detail, 160)}"
            self._notify_user(reason)
        elif outcome.kind in {hitl_poll.TIMED_OUT, hitl_poll.BUDGET_EXHAUSTED}:
            self._notify_user("AMP review timed out. LLM execution was blocked.")
            self._safe_log(
                instance_id,
                f"LLM HITL timed out | execution_id={exec_ctx.execution_id} | call=#{api_call_count}",
                level="WARN",
            )
            reason = "LLM execution timed out waiting for AMP HITL approval."
        else:  # REJECTED and anything unrecognised
            self._notify_user("AMP reviewer rejected LLM execution." + suffix)
            self._safe_log(
                instance_id,
                f"LLM HITL rejected | execution_id={exec_ctx.execution_id} | call=#{api_call_count}",
                level="WARN",
            )
            reason = "LLM execution rejected by AMP HITL review." + suffix
        return "block", reason

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
        *,
        budget: Optional[WaitBudget] = None,
        action_id: Optional[str] = None,
        call_fingerprint: str = "",
    ) -> Optional[Dict[str, str]]:
        """Ask AMP whether a governed tool call may run. None = allow, else a block dict.

        ``budget`` bounds the whole check (see host_limits); ``action_id`` is sent only
        when AMP_ACTION_ID_CORRELATION is on; ``call_fingerprint`` lets an identical
        retry reuse a review that is still pending (A19). Without action_id the check
        runs inside a per-instance review slot with a baseline poll (K2.1).
        """
        if budget is None:
            budget = self._wait_budget(time.time(), include_hook=True)
        # With action_id, AMP scopes the decision to this call: no baseline, but the
        # lock still keeps unscoped LLM/plan polls on this instance from interleaving.
        with self._review_slot(instance_id, deadline=budget.deadline, baseline=not action_id) as slot:
            if not slot.acquired:
                return self._block_message(
                    f"Another governed call in this session is still waiting for AMP review; "
                    f"{action.raw_tool_name} was blocked. One approval at a time: retry after that review."
                )
            if slot.unregistered:
                return self._block_message(self._unregistered_message(action.raw_tool_name))
            return self._evaluate_governance_in_slot(
                instance_id, action, budget=budget, action_id=action_id,
                call_fingerprint=call_fingerprint, slot=slot,
            )

    def _evaluate_governance_in_slot(
        self,
        instance_id: str,
        action: NormalizedAction,
        *,
        budget: WaitBudget,
        action_id: Optional[str],
        call_fingerprint: str,
        slot: _ReviewSlot,
    ) -> Optional[Dict[str, str]]:
        self._ensure_ledger_seeded(instance_id)
        reused = self._ledger.take_pending(instance_id, call_fingerprint)
        if reused is not None:
            workitem_id, reused_action_id = reused
            self._safe_log(
                instance_id,
                f"Retry of a call still awaiting review | raw_tool={action.raw_tool_name} | "
                f"reusing workitem_id={workitem_id}",
            )
            return self._await_tool_review(
                instance_id, action, budget=budget, workitem_id=workitem_id,
                action_id=reused_action_id, call_fingerprint=call_fingerprint,
            )

        if slot.error is not None:
            refused = self._baseline_refusal(action.raw_tool_name, slot.error, budget.deadline)
            if refused is not None:
                return refused

        if budget.deadline - time.time() < max(float(self._config.hitl_poll_interval_seconds), 1.0) + 1.0:
            # Too little time left to send a request and see its workitem: posting now
            # would only leave an orphan review behind.
            self._notify_user(f'AMP review timed out for "{action.raw_tool_name}". The action was blocked.')
            return self._block_message(
                f"{action.raw_tool_name} timed out: the AMP review window ended before the request "
                "could be sent; blocked."
            )
        self._safe_log(
            instance_id,
            f"Policy check | raw_tool={action.raw_tool_name} | tool={action.tool} | action={action.action} | context={action.context}",
        )
        try:
            if action_id:
                response = self._client.request_hitl(instance_id, action, action_id=action_id)
            else:
                response = self._client.request_hitl(instance_id, action)
        except Exception as exc:
            rejected_code = _http_client_error_code(exc)
            if rejected_code is not None:
                # AMP answered and refused (bad key, org, call limit, payload). That is a
                # decision, not an outage, so AMP_FAIL_CLOSED=false must not allow it (H2).
                message = (
                    f"AMP rejected the governance request for {action.raw_tool_name} "
                    f"(HTTP {rejected_code}); blocked. "
                    f"{_truncate(getattr(exc, 'body', '') or str(exc), 200)}"
                ).strip()
                self._safe_log(instance_id, message, level="ERROR")
                return self._block_message(message)
            message = (
                f"AMP governance is unavailable; blocked {action.raw_tool_name}. "
                f"Error: {_truncate(str(exc), 200)}"
            )
            self._safe_log(instance_id, message, level="ERROR")
            if not action_id:
                self._note_unregistered(instance_id, slot)  # AMP may have accepted it anyway
            if self._config.fail_closed or time.time() >= budget.deadline:
                return self._block_message(message)
            return None

        if not isinstance(response, dict):
            message = f"AMP returned an unreadable governance response for {action.raw_tool_name}; blocked."
            self._safe_log(instance_id, message, level="ERROR")
            return self._block_message(message)
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
            # Prefer AMP's human-readable message; `reason` is a code such as
            # policy_invalid_hitl_assignee. Server text: one line, bounded.
            return self._block_message(
                _truncate(" ".join(str(response.get("message") or "").split()), 300)
                or _truncate(reason, 300) or (
                    f'AMP policy for agent "{self._config.agent_name}" is misconfigured; '
                    f'"{action.raw_tool_name}" was blocked.'
                )
            )

        if status in {"no-hitl", "allow", "allowed", "approved"}:
            return None

        if status in {"pending", "waiting-for-response"} or response.get("workitem_id"):
            return self._await_tool_review(
                instance_id, action, budget=budget,
                workitem_id=str(response.get("workitem_id") or "").strip(),
                action_id=action_id, call_fingerprint=call_fingerprint,
                slot=slot,
            )

        # AMP answered with something that is not an allow (e.g. a reserved "deny" or an
        # unknown status). That is not an outage, so AMP_FAIL_CLOSED=false does not apply.
        return self._block_message(
            f'AMP returned unexpected status "{status or "unknown"}" for {action.raw_tool_name}; blocked.'
        )

    def _await_tool_review(
        self,
        instance_id: str,
        action: NormalizedAction,
        *,
        budget: WaitBudget,
        workitem_id: str,
        action_id: Optional[str],
        call_fingerprint: str,
        slot: _ReviewSlot = _ReviewSlot(acquired=True),
    ) -> Optional[Dict[str, str]]:
        """Wait for the reviewer's decision on this call's workitem and map it to allow/block."""
        tool = action.raw_tool_name
        self._safe_log(
            instance_id,
            f"HITL requested | raw_tool={tool} | waiting_for={self._config.username}",
            level="WARN",
        )
        self._notify_user(
            f'AMP is waiting for a human reviewer to approve "{tool}" before continuing. '
            "This action is paused pending review."
        )
        self._announce_wait(tool, workitem_id)
        watchdog = getattr(self._config, "action_id_watchdog_seconds", 60) if action_id else None
        outcome = poll_for_decision(
            (lambda: self._client.get_hitl_decision(instance_id, action_id=action_id))
            if action_id
            else (lambda: self._client.get_hitl_decision(instance_id)),
            instance_id=instance_id,
            ledger=self._ledger,
            deadline=budget.deadline,
            interval_seconds=self._config.hitl_poll_interval_seconds,
            budget_limited=budget.host_limited,
            bound_workitem_id=workitem_id,
            action_id=action_id,
            watchdog_seconds=watchdog,
            foreign_workitem_ids=slot.foreign,
            accept_unbound=slot.accept_unbound,
        )
        bound = outcome.workitem_id
        # Consumed or abandoned: no later call may act on this workitem's decision ...
        self._record_spent(instance_id, bound)
        if not bound and not action_id and outcome.kind not in hitl_poll.TERMINAL_KINDS:
            self._note_unregistered(instance_id, slot)
        if outcome.kind not in hitl_poll.TERMINAL_KINDS:
            # ... except an identical retry, which waits on the still-open review (A19).
            self._ledger.remember_pending(instance_id, call_fingerprint, bound, action_id)
        if outcome.kind != hitl_poll.TIMED_OUT:
            self._safe_log(
                instance_id,
                f"HITL resolution | raw_tool={tool} | outcome={outcome.kind}"
                + (f" | workitem_id={bound}" if bound else "")
                + (f" | {_truncate(outcome.information)}" if outcome.information else ""),
                level="INFO" if outcome.kind == hitl_poll.APPROVED else "WARN",
            )
        self._safe_set_state(instance_id, "active")
        return self._tool_outcome(action, outcome, budget)

    def _baseline_refusal(self, subject: str, exc: AmpClientError, deadline: float) -> Optional[Dict[str, str]]:
        """Block for a failed baseline poll, or None to continue with accept_unbound=False."""
        refused_code = _http_client_error_code(exc)
        if refused_code is not None:
            return self._block_message(
                f"AMP rejected the governance request for {subject} (HTTP {refused_code}); blocked. "
                f"{_truncate(getattr(exc, 'body', '') or str(exc), 200)}".strip()
            )
        if self._config.fail_closed or time.time() >= deadline:
            return self._block_message(
                f"AMP governance is unavailable; blocked {subject}. Error: {_truncate(str(exc), 200)}"
            )
        return None

    def _tool_outcome(
        self, action: NormalizedAction, outcome: PollOutcome, budget: WaitBudget
    ) -> Optional[Dict[str, str]]:
        """Map a HITL poll outcome to None (allow) or a block. Only APPROVED allows."""
        tool = action.raw_tool_name
        note = reviewer_note(outcome.information)
        suffix = f" {note}" if note else ""
        if outcome.kind == hitl_poll.APPROVED:
            self._notify_user(f'AMP review approved "{tool}". Continuing now.')
            return None
        if outcome.kind == hitl_poll.MODIFIED:
            message = (
                f'AMP reviewer approved a modified version of "{tool}". '
                "The modified version was not applied, so the original call was blocked; "
                "re-submit using the reviewer's changes." + suffix
            )
            self._notify_user(message)
            return self._block_message(message)
        if outcome.kind == hitl_poll.CANCELLED:
            message = (
                f'The AMP review of "{tool}" was cancelled before a reviewer decided'
                + (f" (workitem {outcome.workitem_id})" if outcome.workitem_id else "")
                + "; the action was blocked. It was not rejected by a reviewer."
            )
            self._notify_user(message)
            return self._block_message(message)
        if outcome.kind == hitl_poll.POLL_REJECTED:
            message = (
                f'AMP rejected the HITL decision poll for "{tool}" (HTTP {outcome.status_code}); '
                f"blocked. {_truncate(outcome.detail, 160)}"
            ).strip()
            self._notify_user(message)
            return self._block_message(message)
        if outcome.kind == hitl_poll.CORRELATION_FAILED:
            if outcome.detail == "watchdog":
                seconds = int(getattr(self._config, "action_id_watchdog_seconds", 60))
                message = (
                    f"AMP has not created a review for {tool} after {seconds}s "
                    "(action_id correlation may be unsupported by the HITL service); blocked."
                )
            else:
                message = (
                    f"AMP did not echo action_id for {tool} — the backend does not support "
                    "decision correlation; blocked."
                )
            return self._block_message(message)
        if outcome.kind == hitl_poll.UNVERIFIABLE:
            message = (
                f"AMP returned a decision for {tool} that is not tied to a workitem "
                f"(method {outcome.detail}), so it cannot be verified as this call's decision "
                "(unverifiable decision); blocked."
            )
            self._notify_user(message)
            return self._block_message(message)
        if outcome.kind == hitl_poll.UNAVAILABLE:
            message = (
                f"AMP was unreachable while waiting for the review of {tool}; blocked. "
                f"Last error: {_truncate(outcome.detail, 160)}"
            ) + (
                f" The review may still be open in AMP (workitem {outcome.workitem_id}); "
                "do not retry until a reviewer approves it."
                if outcome.workitem_id else ""
            )
            self._notify_user(f'AMP was unreachable while "{tool}" waited for review. The action was blocked.')
            return self._block_message(message)
        if outcome.kind in {hitl_poll.TIMED_OUT, hitl_poll.BUDGET_EXHAUSTED}:
            retry_hint = (
                f" The review is still open in AMP (workitem {outcome.workitem_id}). Do not retry this "
                "call until a reviewer approves it; an identical retry in this session waits on the "
                "same review instead of opening a new one."
                if outcome.workitem_id
                else " Do not retry this call until a reviewer approves it in AMP."
            )
            if outcome.kind == hitl_poll.BUDGET_EXHAUSTED:
                self._notify_user(
                    f'AMP review for "{tool}" did not finish within the Hermes time limit. '
                    "The action was blocked."
                )
                return self._block_message(
                    f'AMP review for "{tool}" did not complete within the {int(budget.limit_seconds)}s '
                    f"Hermes limit ({budget.host_limit_name}); the action was blocked. Raise "
                    "plugins.hook_callback_timeout (max 600) and timeouts.tools.sequential_call / "
                    "concurrent_batch in Hermes config.yaml for longer reviews." + retry_hint
                )
            self._notify_user(f'AMP review timed out for "{tool}". The action was blocked.')
            return self._block_message(f"{tool} timed out waiting for AMP HITL approval." + retry_hint)
        # REJECTED and anything unrecognised
        self._notify_user(f'AMP reviewer rejected "{tool}".' + suffix)
        return self._block_message(f"{tool} was rejected by AMP HITL review." + suffix)

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
        started_at = time.time()
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

        # Plan approval runs inside a tool call, so Hermes's tool deadline applies (not the
        # hook timeout); one budget covers the lock wait and the review.
        budget = self._wait_budget(started_at, include_hook=False)
        with self._review_slot(instance_id, deadline=budget.deadline) as slot:
            if not slot.acquired:
                return self._plan_result(
                    "error", plan_id=plan_id,
                    reason="Another governed call is still waiting for AMP review; plan not submitted.",
                )
            if slot.unregistered:
                return self._plan_result("error", plan_id=plan_id, reason=self._unregistered_message("the plan"))
            if budget.deadline - time.time() < max(float(self._config.hitl_poll_interval_seconds), 1.0) + 1.0:
                return self._plan_result(
                    "timed_out", plan_id=plan_id,
                    reason="The AMP review window ended before the plan could be submitted.",
                )
            if slot.error is not None:
                # Plans have no fail-open mode: any baseline failure is an error result.
                refused_code = _http_client_error_code(slot.error)
                reason = (
                    f"AMP rejected the plan approval request (HTTP {refused_code})."
                    if refused_code is not None
                    else f"AMP governance unavailable: {_truncate(str(slot.error), 200)}"
                )
                return self._plan_result("error", plan_id=plan_id, reason=reason)
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

                self._announce_wait("the execution plan", workitem_id)
                with request_deadline(budget.deadline):
                    outcome = poll_for_decision(
                        lambda: self._client.get_hitl_decision(instance_id),
                        instance_id=instance_id,
                        ledger=self._ledger,
                        deadline=budget.deadline,
                        interval_seconds=self._config.hitl_poll_interval_seconds,
                        budget_limited=budget.host_limited,
                        bound_workitem_id=workitem_id,
                        label="plan ",
                        foreign_workitem_ids=slot.foreign,
                        accept_unbound=slot.accept_unbound,
                    )
                    self._record_spent(instance_id, outcome.workitem_id)
                    if not outcome.workitem_id and outcome.kind not in hitl_poll.TERMINAL_KINDS:
                        self._note_unregistered(instance_id, slot)
                    self._safe_set_state(instance_id, "active")
                return self._plan_outcome(session_id, instance_id, plan_id, cost, outcome)

            self._safe_log(instance_id, f"Plan unexpected AMP status '{status}' | plan_id={plan_id}", level="WARN")
            return self._plan_result(
                "error", plan_id=plan_id,
                reason=f'AMP returned unexpected status "{status or "unknown"}".',
            )

    def _plan_outcome(
        self,
        session_id: str,
        instance_id: str,
        plan_id: str,
        cost: float,
        outcome: PollOutcome,
    ) -> Dict[str, Any]:
        """Map a HITL poll outcome to a plan result. Only APPROVED sets a budget."""
        workitem_id = outcome.workitem_id or None
        note = reviewer_note(outcome.information)
        if outcome.kind == hitl_poll.APPROVED:
            self._apply_approved_budget(session_id, instance_id, cost)
            self._notify_user(
                "Execution plan approved by AMP." + (f"\n{note}" if note else "")
                + "\nResearching now -- the report will follow shortly."
            )
            self._safe_log(instance_id, f"Plan HITL approved | plan_id={plan_id} | budget=${cost:.2f}")
            return self._plan_result(
                "approved", plan_id=plan_id, approved_budget_usd=cost,
                reason=note, workitem_id=workitem_id,
            )
        if outcome.kind in {hitl_poll.TIMED_OUT, hitl_poll.BUDGET_EXHAUSTED}:
            self._notify_user("AMP review of the execution plan timed out.")
            self._safe_log(instance_id, f"Plan HITL timed out | plan_id={plan_id}", level="WARN")
            return self._plan_result(
                "timed_out", plan_id=plan_id,
                reason="Timed out waiting for AMP HITL approval.",
                workitem_id=workitem_id,
            )
        if outcome.kind in {hitl_poll.UNVERIFIABLE, hitl_poll.UNAVAILABLE}:
            reason = (
                f"AMP returned a plan decision not tied to a workitem (method {outcome.detail}); unverifiable decision."
                if outcome.kind == hitl_poll.UNVERIFIABLE
                else f"AMP was unreachable while waiting for the plan review: {_truncate(outcome.detail, 160)}"
            )
            self._safe_log(instance_id, f"Plan HITL {outcome.kind} | plan_id={plan_id} | {reason}", level="WARN")
            return self._plan_result("error", plan_id=plan_id, reason=reason, workitem_id=workitem_id)
        if outcome.kind == hitl_poll.POLL_REJECTED:
            reason = (
                f"AMP rejected the HITL decision poll (HTTP {outcome.status_code}); plan not approved. "
                f"{_truncate(outcome.detail, 160)}"
            ).strip()
            self._safe_log(instance_id, f"Plan HITL poll rejected | plan_id={plan_id} | {reason}", level="WARN")
            return self._plan_result("error", plan_id=plan_id, reason=reason, workitem_id=workitem_id)
        if outcome.kind == hitl_poll.MODIFIED:
            reason = "Reviewer modified the plan; re-submit the modified plan." + (f" {note}" if note else "")
        elif outcome.kind == hitl_poll.CANCELLED:
            reason = "The AMP review of the plan was cancelled before a reviewer decided; plan not approved."
        else:
            reason = note or "Execution plan rejected by AMP HITL review."
        self._notify_user("AMP did not approve the execution plan. " + reason)
        self._safe_log(
            instance_id,
            f"Plan HITL not approved | plan_id={plan_id} | outcome={outcome.kind}"
            + (f" | {_truncate(outcome.information)}" if outcome.information else ""),
            level="WARN",
        )
        return self._plan_result("rejected", plan_id=plan_id, reason=reason, workitem_id=workitem_id)

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

        with self._open_sessions_lock:
            self._open_sessions.discard(session_id)
        record = self._store.get(session_id)
        if not record:
            return
        self._ledger.forget_instance(record.instance_id)
        try:
            self._safe_log(record.instance_id, f"Finalizing session | reason={reason or 'unknown'}")
            self._client.set_state(record.instance_id, "finished")
        except Exception as exc:
            logger.warning("amp-governance finalize failed: %s", exc)
        finally:
            self._store.delete(session_id)

    def on_session_end(self, session_id: str = "", **_: Any) -> None:
        """Finalize the AMP session at the end of a single-query run (A16b).

        `hermes -z` / `hermes chat -q` never dispatch on_session_finalize and leave
        via os._exit (no atexit), so the AMP instance would stay open and the
        session record would leak. Interactive and gateway sessions span many
        turns and are finalized by Hermes itself, so they are left alone.
        """
        if not session_id or not _is_single_query_process():
            return
        self._finalize_within_budget([session_id], reason="single_query_end")

    def finalize_open_sessions_at_exit(self) -> None:
        """atexit fallback for single-query runs that exit normally (best-effort, <= 3 s)."""
        if not _is_single_query_process():
            return
        with self._open_sessions_lock:
            open_sessions = sorted(self._open_sessions)
        if open_sessions:
            self._finalize_within_budget(open_sessions, reason="process_exit")

    def _finalize_within_budget(self, session_ids: List[str], *, reason: str) -> None:
        deadline = time.time() + _EXIT_FINALIZE_BUDGET_SECONDS
        with request_deadline(deadline, hard=True):
            for session_id in session_ids:
                if time.time() >= deadline:
                    logger.warning("amp-governance: exit finalize budget exhausted; %s left open", session_id)
                    continue
                try:
                    self.on_session_finalize(session_id=session_id, reason=reason)
                except Exception as exc:
                    logger.warning("amp-governance exit finalize failed for %s: %s", session_id, exc)

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
        tool_call_id: str = "",
        **_: Any,
    ) -> Optional[Dict[str, str]]:
        started_at = time.time()
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
        # Return our own decision before Hermes's hook timeout / tool deadline fires (A3).
        budget = self._wait_budget(started_at, include_hook=True)
        action_id = None
        if getattr(self._config, "action_id_correlation", False):
            action_id = str(tool_call_id or "").strip() or uuid.uuid4().hex
        try:
            with request_deadline(budget.deadline):
                instance_id = self._ensure_instance(session_id, model=model, platform=platform)
                decision = self._evaluate_governance(
                    instance_id, action, budget=budget, action_id=action_id,
                    call_fingerprint=_call_fingerprint(action, args),
                )
            if decision and decision.get("action") == "block":
                self._record_blocked_turn(session_id, str(decision.get("message") or ""))
            return decision
        except Exception as exc:
            logger.warning("amp-governance pre_tool_call failed: %s", exc)
            refused_code = _http_client_error_code(exc)
            if refused_code is not None:
                # e.g. init_instance with a revoked key: a refusal, never fail-open.
                message = (
                    f'AMP rejected the governance request for "{tool_name}" (HTTP {refused_code}); blocked. '
                    f"{_truncate(getattr(exc, 'body', '') or str(exc), 200)}"
                ).strip()
                self._record_blocked_turn(session_id, message)
                return self._block_message(message)
            # AMP_FAIL_CLOSED=false only covers AMP being unreachable. A local error
            # (store, notification, bug) after or before AMP answered is never an allow.
            is_outage = isinstance(exc, AmpClientError) and exc.status_code is None
            if self._config.fail_closed or not is_outage or time.time() >= budget.deadline:
                message = (
                    f'AMP governance failed while checking "{tool_name}": {_truncate(str(exc), 200)}'
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
        # Keep the cause visible (rejected vs cancelled vs timed out vs AMP error vs
        # outage) instead of one generic line for every block (L15).
        return (
            "This request is blocked by AMP governance. No action was taken.\n"
            f"Reason: {_truncate(blocked_reason, 700)}"
        )


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


_EXIT_FINALIZER_REGISTERED = False


def _finalize_open_sessions_at_exit() -> None:
    _PLUGIN.finalize_open_sessions_at_exit()


def _register_exit_finalizer() -> None:
    global _EXIT_FINALIZER_REGISTERED
    if not _EXIT_FINALIZER_REGISTERED:
        atexit.register(_finalize_open_sessions_at_exit)
        _EXIT_FINALIZER_REGISTERED = True


def _log_short_host_limits(config: AmpConfig) -> None:
    """Say at startup when Hermes will cut HITL reviews short (L11)."""
    try:
        hitl_seconds = int(config.hitl_timeout_minutes) * 60
        short = limits_shorter_than(hitl_seconds)
    except Exception:
        return
    for name, seconds in short:
        logger.error(
            "amp-governance: Hermes %s=%gs is shorter than AMP_HITL_TIMEOUT_MINUTES=%d (%ds); "
            "governed calls stop waiting for a reviewer after ~%gs and are blocked. Set "
            "plugins.hook_callback_timeout: 600 and timeouts.tools.sequential_call / concurrent_batch: 660 "
            "in Hermes config.yaml (or lower AMP_HITL_TIMEOUT_MINUTES).",
            name, seconds, config.hitl_timeout_minutes, hitl_seconds,
            compute_budget(0.0, hitl_seconds, hook_timeout=seconds, tool_deadline=None).deadline,
        )


def register(ctx) -> None:
    _PLUGIN.attach_context(ctx)
    ctx.register_hook("on_session_start", _PLUGIN.on_session_start)
    ctx.register_hook("on_session_finalize", _PLUGIN.on_session_finalize)
    ctx.register_hook("on_session_end", _PLUGIN.on_session_end)
    ctx.register_hook("pre_llm_call", _PLUGIN.pre_llm_call)
    ctx.register_hook("pre_tool_call", _PLUGIN.pre_tool_call)
    ctx.register_hook("post_tool_call", _PLUGIN.post_tool_call)
    ctx.register_hook("pre_api_request", _PLUGIN.pre_api_request)
    ctx.register_hook("post_api_request", _PLUGIN.post_api_request)
    ctx.register_hook("transform_llm_output", _PLUGIN.transform_llm_output)
    _register_exit_finalizer()
    _log_short_host_limits(_PLUGIN._config)

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
