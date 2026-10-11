"""Hermes host time limits that bound how long a governed call may wait for AMP.

A governed tool call waits for a human reviewer inside Hermes's ``pre_tool_call``
hook. Three limits apply, and the shortest one wins:

* ``AMP_HITL_TIMEOUT_MINUTES`` — the plugin's own review window;
* ``plugins.hook_callback_timeout`` — Hermes >= v2026.9 abandons a hook after
  this many seconds (default 30, max 600, ``0`` = unbounded) and suppresses the
  plugin for 60 s;
* ``timeouts.tools.sequential_call`` / ``timeouts.tools.concurrent_batch`` —
  Hermes's per-call / per-batch tool deadline (default 420 s), which also covers
  time spent in ``pre_tool_call``.

The plugin returns its own block message shortly before the binding limit so
Hermes never has to time the hook or tool out. Both Hermes resolvers are private
APIs; any failure to resolve them means "no limit known" (None).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

# Stop waiting this long before the binding limit, capped at a quarter of the
# limit so small budgets (e.g. a 10 s hook timeout) still leave time to poll.
MARGIN_CAP_SECONDS = 15.0
MARGIN_FRACTION = 0.25

HOOK_LIMIT_NAME = "plugins.hook_callback_timeout"
TOOL_LIMIT_NAME = "timeouts.tools.sequential_call/concurrent_batch"


def _positive(value: object) -> Optional[float]:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def hook_timeout_seconds() -> Optional[float]:
    """Hermes ``pre_tool_call`` budget in seconds; None when unbounded or unknown."""
    try:
        from hermes_cli.plugins import _resolve_hook_callback_timeout
    except Exception:
        return None
    try:
        return _positive(_resolve_hook_callback_timeout())
    except Exception:
        return None


def tool_deadline_seconds() -> Optional[float]:
    """Shortest Hermes tool deadline (sequential or concurrent); None when unbounded or unknown.

    The hook cannot tell which executor runs the call, so it assumes the stricter one.
    """
    try:
        from agent import tool_executor
    except Exception:
        return None
    limits: List[float] = []
    for resolver_name in ("_resolve_sequential_tool_timeout", "_resolve_concurrent_tool_timeout"):
        resolver = getattr(tool_executor, resolver_name, None)
        if resolver is None:
            continue
        try:
            value = _positive(resolver())
        except Exception:
            continue
        if value is not None:
            limits.append(value)
    return min(limits) if limits else None


@dataclass(frozen=True)
class WaitBudget:
    """How long one governed call may wait for an AMP decision."""

    deadline: float          # absolute time.time() value
    limit_seconds: float     # the binding limit before the safety margin
    host_limit_name: str     # Hermes setting that binds; "" when AMP_HITL_TIMEOUT_MINUTES binds
    margin_seconds: float = 0.0  # deadline = start + limit - margin

    @property
    def grace_seconds(self) -> float:
        """Extra time a post-deadline bookkeeping write (e.g. the late-workitem marker) may
        wait for the session-store lock: at most half the margin, so it stays inside the
        Hermes limit even for small hook budgets."""
        return min(1.0, 0.5 * self.margin_seconds)

    @property
    def host_limited(self) -> bool:
        return bool(self.host_limit_name)


def compute_budget(
    started_at: float,
    hitl_timeout_seconds: float,
    *,
    hook_timeout: Optional[float],
    tool_deadline: Optional[float],
) -> WaitBudget:
    candidates: List[Tuple[float, str]] = [(max(float(hitl_timeout_seconds), 0.0), "")]
    if hook_timeout is not None:
        candidates.append((hook_timeout, HOOK_LIMIT_NAME))
    if tool_deadline is not None:
        candidates.append((tool_deadline, TOOL_LIMIT_NAME))
    # Ties go to the AMP timeout (listed first): its message is the familiar one.
    limit, name = min(candidates, key=lambda item: item[0])
    margin = min(MARGIN_CAP_SECONDS, MARGIN_FRACTION * limit)
    return WaitBudget(
        deadline=started_at + limit - margin, limit_seconds=limit, host_limit_name=name, margin_seconds=margin
    )


def limits_shorter_than(seconds: float) -> List[Tuple[str, float]]:
    """Hermes limits that would cut an AMP review shorter than ``seconds``."""
    found: List[Tuple[str, float]] = []
    for name, value in ((HOOK_LIMIT_NAME, hook_timeout_seconds()), (TOOL_LIMIT_NAME, tool_deadline_seconds())):
        if value is not None and value < seconds:
            found.append((name, value))
    return found
