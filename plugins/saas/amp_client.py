from __future__ import annotations

from contextlib import contextmanager
import json
import threading
import time
from typing import Any, Dict, Iterator, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .config import AmpConfig
from .policy import NormalizedAction


_DEFAULT_HTTP_TIMEOUT_SECONDS = 30.0
_MIN_HTTP_TIMEOUT_SECONDS = 1.0

# Per-thread wall-clock deadline for AMP HTTP calls made inside one Hermes hook
# or tool invocation (see request_deadline). Thread-local because Hermes runs
# hooks for concurrent tool calls on separate worker threads.
_deadline_state = threading.local()


class AmpClientError(RuntimeError):
    """An AMP call failed. ``status_code`` is the HTTP status when AMP answered
    (None for network errors, timeouts and exhausted deadlines); ``body`` is
    the raw response body AMP sent with that status, if any."""

    def __init__(self, message: str, *, status_code: Optional[int] = None, body: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


@contextmanager
def request_deadline(at: Optional[float], *, hard: bool = False) -> Iterator[None]:
    """Bound every AMP HTTP call made on this thread by the absolute time ``at``.

    Each request's socket timeout becomes ``min(30 s, at - now)``. With
    ``hard=False`` (governance checks) a request is never refused: once the
    deadline has passed the timeout floors at 1 s, so the first governance
    request is always attempted and the caller decides what an exhausted budget
    means. With ``hard=True`` (best-effort cleanup at exit) a request made after
    the deadline raises ``AmpClientError`` immediately instead.
    ``at=None`` leaves the default 30 s timeout in place.
    """
    previous = getattr(_deadline_state, "value", None)
    _deadline_state.value = (at, hard) if at is not None else None
    try:
        yield
    finally:
        _deadline_state.value = previous


def _http_timeout() -> float:
    state = getattr(_deadline_state, "value", None)
    if state is None:
        return _DEFAULT_HTTP_TIMEOUT_SECONDS
    at, hard = state
    remaining = at - time.time()
    if remaining <= 0 and hard:
        raise AmpClientError("AMP request deadline exhausted")
    return min(_DEFAULT_HTTP_TIMEOUT_SECONDS, max(remaining, _MIN_HTTP_TIMEOUT_SECONDS))


class AmpClient:
    def __init__(self, config: AmpConfig) -> None:
        self._config = config

    def _headers(self) -> Dict[str, str]:
        return {
            "X-API-Key": self._config.api_key,
            "Content-Type": "application/json",
        }

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        query: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        url = f"{self._config.backend_url}{path}"
        if query:
            url = f"{url}?{urlencode(query)}"
        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
        req = Request(url, data=data, headers=self._headers(), method=method)
        timeout = _http_timeout()
        try:
            with urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace") or "{}"
                status = getattr(resp, "status", None)
            try:
                return json.loads(body)
            except ValueError as exc:
                # AMP (or a proxy/captive portal in front of it) answered with something
                # that is not JSON. Carry the status so callers treat it as an answer.
                raise AmpClientError(
                    f"AMP returned a non-JSON response (HTTP {status})",
                    status_code=status if isinstance(status, int) else 200,
                    body=body[:500],
                ) from exc
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise AmpClientError(
                f"HTTP {exc.code}: {body or exc.reason}", status_code=exc.code, body=body
            ) from exc
        except URLError as exc:
            raise AmpClientError(str(exc.reason)) from exc
        except (TimeoutError, OSError) as exc:
            # A read timeout or reset after the connection was established is
            # not wrapped in URLError; surface it as an AMP failure so callers
            # treat it like any other outage instead of an unexpected crash.
            raise AmpClientError(f"AMP request failed: {exc}") from exc

    def init_instance(self, session_id: str, model: str, platform: str) -> str:
        data = self._request(
            "POST",
            "/api/agent/init",
            {
                "agent_name": self._config.agent_name,
                "org_id": self._config.org_id,
                "username": self._config.username,
                "prompt": "Hermes governance session",
                "auto_start": True,
                "config": {"agent_mode": "polling"},
                "metadata": {
                    "source": "hermes-amp-plugin",
                    "session_id": session_id,
                    "platform": platform,
                    "model": model,
                },
            },
        )
        instance_id = str(data.get("instance_id") or "").strip()
        if not instance_id:
            raise AmpClientError("AMP did not return instance_id")
        return instance_id

    def log(
        self,
        instance_id: str,
        message: str,
        *,
        level: str = "INFO",
    ) -> None:
        """Every AHP-issued log line goes through here, so prefixing here
        (rather than at each call site) covers all of them uniformly —
        mirrors AMP's own "[HITL]" convention for its internal log lines."""
        self._request(
            "POST",
            "/api/log",
            {
                "instance_id": instance_id,
                "service": self._config.agent_name,
                "level": level,
                "message": f"[AHP] {message}",
                "org_id": self._config.org_id,
                "username": self._config.username,
            },
        )

    def request_hitl(
        self,
        instance_id: str,
        action: NormalizedAction,
        *,
        action_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Submit a governed tool action for eval-policy/HITL review.

        The payload uses the normalized Hermes tool/action names plus the
        tool-specific context extracted by the policy normalizer. AMP then
        decides allow/no-hitl, pending HITL, or a policy/config error.
        ``action_id`` (opt-in, AMP_ACTION_ID_CORRELATION) is sent top-level so
        AMP can scope the later decision poll to this one call.
        """
        payload: Dict[str, Any] = {
            "caller_id": instance_id,
            "instance_id": instance_id,
            "org_id": self._config.org_id,
            "agent_name": self._config.agent_name,
            "tool": action.tool,
            "action": action.action,
            "context": action.context,
            "hitl": {
                "enable": True,
                "when": "policy",
            },
        }
        if action_id:
            payload["action_id"] = action_id
        return self._request("POST", "/api/hitl/request", payload)

    def get_hitl_decision(self, instance_id: str, *, action_id: Optional[str] = None) -> Dict[str, Any]:
        """Poll the normalized decision endpoint for a caller instance.

        Returns a compact decision state (`pending` or `complete`) without
        exposing the internal workitem-stage routing details AMP uses behind
        the scenes. With ``action_id`` AMP only matches that call's workitem
        and echoes ``action_id`` back.
        """
        query = {"caller_id": instance_id}
        if action_id:
            query["action_id"] = action_id
        return self._request("GET", "/api/hitl/get-decision", query=query)

    def request_llm_hitl(
        self,
        instance_id: str,
        *,
        execution_id: str = "",
        model: str = "",
        provider: str = "",
        session_id: str = "",
        current_cost_usd: float = 0.0,
        current_cost_status: str = "unknown",
        approved_budget_usd: Optional[float] = None,
        estimated_next_call_cost_usd: float = 0.0,
        estimated_next_call_cost_status: str = "unknown",
        projected_total_cost_usd: float = 0.0,
        projected_total_cost_status: str = "unknown",
        input_tokens_total: int = 0,
        output_tokens_total: int = 0,
        cache_read_tokens_total: int = 0,
        cache_write_tokens_total: int = 0,
        reasoning_tokens_total: int = 0,
        total_tokens: int = 0,
        llm_calls_total: int = 0,
    ) -> Dict[str, Any]:
        """Submit a pre-LLM-call governance signal to AMP and return the policy decision.

        All cost and token fields are promoted to policy params by AMP's flat-field
        promotion mechanism (no AMP schema change required).
        """
        payload: Dict[str, Any] = {
            "caller_id": instance_id,
            "instance_id": instance_id,
            "org_id": self._config.org_id,
            "agent_name": self._config.agent_name,
            "tool": "llm",
            "action": "invoke",
            "context": {
                "model": model,
                "provider": provider,
                "session_id": session_id,
                "execution_id": execution_id,
            },
            "hitl": {"enable": True, "when": "policy"},
            # Flat governance signals — AMP promotes these to policy params automatically
            "execution_id": execution_id,
            "model": model,
            "provider": provider,
            "current_cost_usd": round(current_cost_usd, 8),
            "current_cost_status": current_cost_status,
            "estimated_next_call_cost_usd": round(estimated_next_call_cost_usd, 8),
            "estimated_next_call_cost_status": estimated_next_call_cost_status,
            "projected_total_cost_usd": round(projected_total_cost_usd, 8),
            "projected_total_cost_status": projected_total_cost_status,
            "input_tokens_total": input_tokens_total,
            "output_tokens_total": output_tokens_total,
            "cache_read_tokens_total": cache_read_tokens_total,
            "cache_write_tokens_total": cache_write_tokens_total,
            "reasoning_tokens_total": reasoning_tokens_total,
            "total_tokens": total_tokens,
            "llm_calls_total": llm_calls_total,
        }
        if approved_budget_usd is not None:
            payload["approved_budget_usd"] = round(approved_budget_usd, 8)
        return self._request("POST", "/api/hitl/request", payload)

    def request_plan_approval(
        self,
        instance_id: str,
        *,
        plan_id: str = "",
        plan_type: str = "",
        summary: str = "",
        projected_cost_usd: float = 0.0,
        projected_cost_status: str = "unknown",
        estimated_llm_calls: int = 0,
        estimated_tool_calls: int = 0,
        estimated_duration_minutes: int = 0,
        work_units_total: int = 0,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Submit a proposed execution plan to AMP for governance approval.

        Reuses the same /api/hitl/request path as tool and LLM governance.
        All plan fields are promoted to flat top-level payload fields so AMP's
        eval-policy engine can reference them (e.g. plan_projected_cost_usd)
        without any AMP schema change. `payload` carries caller-specific data
        (e.g. research topics) and is passed through unmodified — AHP does not
        interpret it.
        """
        return self._request(
            "POST",
            "/api/hitl/request",
            {
                "caller_id": instance_id,
                "instance_id": instance_id,
                "org_id": self._config.org_id,
                "agent_name": self._config.agent_name,
                "tool": "execution_plan",
                "action": "submit",
                "context": {
                    "plan_id": plan_id,
                    "summary": summary,
                    "payload": payload or {},
                },
                "hitl": {"enable": True, "when": "policy"},
                # Flat governance signals — AMP promotes these to policy params automatically
                "plan_id": plan_id,
                "plan_type": plan_type,
                "plan_projected_cost_usd": round(float(projected_cost_usd), 8),
                "plan_projected_cost_status": projected_cost_status,
                "plan_estimated_llm_calls": estimated_llm_calls,
                "plan_estimated_tool_calls": estimated_tool_calls,
                "plan_estimated_duration_minutes": estimated_duration_minutes,
                "plan_work_units_total": work_units_total,
            },
        )

    def log_llm_event(
        self,
        instance_id: str,
        event: Dict[str, Any],
        *,
        execution_id: str = "",
    ) -> None:
        """Log a structured LLM call event to AMP."""
        model = event.get("model", "")
        call_num = event.get("api_call_number", "")
        total_tokens = int(event.get("total_tokens") or 0)
        cost_usd = float(event.get("cost_usd") or 0.0)
        cost_status = event.get("cost_status", "unknown")
        parts = []
        if execution_id:
            parts.append(f"execution_id={execution_id}")
        parts.append(
            f"LLM call #{call_num} | model={model} | tokens={total_tokens:,} | "
            f"cost_usd={cost_usd:.2f} | cost_status={cost_status}"
        )
        self.log(instance_id, " | ".join(parts))

    def log_execution_summary(self, instance_id: str, summary: Dict[str, Any]) -> None:
        """Log a final execution cost/usage summary to AMP."""
        execution_id = summary.get("execution_id", "")
        llm_calls = summary.get("llm_calls", 0)
        total_tokens = int(summary.get("total_tokens") or 0)
        total_cost = float(summary.get("total_cost_usd") or 0.0)
        cost_status = summary.get("cost_status", "unknown")
        self.log(
            instance_id,
            f"Execution summary | execution_id={execution_id} | "
            f"llm_calls={llm_calls} | total_tokens={total_tokens:,} | "
            f"total_cost_usd={total_cost:.2f} | cost_status={cost_status}",
        )

    def save_llm_trace(
        self,
        instance_id: str,
        *,
        call_time: str,
        model: str,
        prompt: Optional[Dict[str, Any]] = None,
        reasoning: Optional[Any] = None,
        answer: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> None:
        """Save what an LLM call was actually asked and what it decided, so
        AMP's "click LLM in the log" panel has something real to show instead
        of "No saved LLM reasoning was found for this entry."

        Best-effort by design: callers should catch/log-and-continue on
        failure the same way they already do for _safe_log, never let this
        block or fail the LLM call it's describing.
        """
        payload: Dict[str, Any] = {"call_time": call_time, "model": model}
        if prompt is not None:
            payload["prompt"] = prompt
        if reasoning is not None:
            payload["reasoning"] = reasoning
        if answer is not None:
            payload["answer"] = answer
        if trace_id is not None:
            payload["trace_id"] = trace_id
        self._request("POST", f"/api/agents/{instance_id}/llm_trace", payload)

    def set_state(self, instance_id: str, state: str) -> None:
        self._request(
            "POST",
            "/api/agent/setState",
            {
                "agent_name": self._config.agent_name,
                "instance_id": instance_id,
                "state": state,
            },
        )
