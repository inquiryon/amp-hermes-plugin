"""Poll AMP's HITL decision endpoint for one governed call.

Shared by the tool, LLM and plan governance paths. Every outcome except
``APPROVED`` must end in a block (fix spec §2 rule 2).

Workitem guard (stale-decision race, H3/L16): ``GET /api/hitl/get-decision`` is
scoped to the AMP *instance*, not the call — it returns the newest workitem of
the instance. A call therefore binds to the first ``workitem_id`` it sees (the
request response, else its first pending poll) and only accepts a ``complete``
decision for that workitem. Before a workitem is bound, decisions for
workitems this process already consumed or abandoned are ignored.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import logging
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

from .amp_client import AmpClientError

logger = logging.getLogger(__name__)

# Poll failures that will not fix themselves within one review window.
POLL_FATAL_STATUS = frozenset({401, 403, 404, 429})

_APPROVE = frozenset({"approve", "approved"})
_MODIFY = frozenset({"modify", "modified"})
_CANCELLED = frozenset({"cancelled", "canceled"})

REVIEWER_NOTE_LIMIT = 500
_LEDGER_MAX_ENTRIES = 1000

# Outcome kinds.
APPROVED = "approved"
MODIFIED = "modified"
REJECTED = "rejected"
CANCELLED = "cancelled"
TIMED_OUT = "timed_out"
BUDGET_EXHAUSTED = "budget_exhausted"
POLL_REJECTED = "poll_rejected"
CORRELATION_FAILED = "correlation_failed"

TERMINAL_KINDS = frozenset({APPROVED, MODIFIED, REJECTED, CANCELLED, POLL_REJECTED, CORRELATION_FAILED})


@dataclass(frozen=True)
class PollOutcome:
    kind: str
    workitem_id: str = ""
    information: str = ""
    status_code: Optional[int] = None
    detail: str = ""
    polls: int = 0


def reviewer_note(information: str) -> str:
    """Label and bound reviewer free text before it reaches the model (L9).

    Whitespace (including newlines) is collapsed so the note stays one line.
    """
    text = " ".join(str(information or "").split())
    if not text:
        return ""
    if len(text) > REVIEWER_NOTE_LIMIT:
        text = text[: REVIEWER_NOTE_LIMIT - 3] + "..."
    return f"Reviewer note: {text}"


def _lower(value: Any) -> str:
    return str(value or "").strip().lower()


class DecisionLedger:
    """Per-instance memory of AMP workitems (thread-safe, bounded).

    * spent: workitems whose decision was consumed, or whose call gave up
      waiting. A later call must never act on their decision unless it is
      explicitly re-bound to them (A19 reuse).
    * pending reviews: workitems still awaiting a reviewer when the call gave
      up, keyed by the call's (tool, action, args) fingerprint, so an identical
      retry waits on the same review instead of opening a new one (A19).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._spent: Dict[str, "OrderedDict[str, None]"] = {}
        self._pending: Dict[str, "OrderedDict[str, Tuple[str, Optional[str]]]"] = {}

    @staticmethod
    def _bounded_put(store: "OrderedDict[str, Any]", key: str, value: Any) -> None:
        store.pop(key, None)
        store[key] = value
        while len(store) > _LEDGER_MAX_ENTRIES:
            store.popitem(last=False)

    def mark_spent(self, instance_id: str, workitem_id: str) -> None:
        if not workitem_id:
            return
        with self._lock:
            self._bounded_put(self._spent.setdefault(instance_id, OrderedDict()), workitem_id, None)

    def is_spent(self, instance_id: str, workitem_id: str) -> bool:
        with self._lock:
            return workitem_id in self._spent.get(instance_id, ())

    def remember_pending(
        self, instance_id: str, fingerprint: str, workitem_id: str, action_id: Optional[str]
    ) -> None:
        if not fingerprint or not workitem_id:
            return
        with self._lock:
            self._bounded_put(
                self._pending.setdefault(instance_id, OrderedDict()), fingerprint, (workitem_id, action_id)
            )

    def take_pending(self, instance_id: str, fingerprint: str) -> Optional[Tuple[str, Optional[str]]]:
        if not fingerprint:
            return None
        with self._lock:
            return self._pending.get(instance_id, OrderedDict()).pop(fingerprint, None)

    def forget_instance(self, instance_id: str) -> None:
        with self._lock:
            self._spent.pop(instance_id, None)
            self._pending.pop(instance_id, None)


def poll_for_decision(
    fetch: Callable[[], Dict[str, Any]],
    *,
    instance_id: str,
    ledger: DecisionLedger,
    deadline: float,
    interval_seconds: float,
    budget_limited: bool = False,
    bound_workitem_id: str = "",
    action_id: Optional[str] = None,
    watchdog_seconds: Optional[float] = None,
    label: str = "",
) -> PollOutcome:
    """Poll until a decision for this call arrives or ``deadline`` passes.

    ``budget_limited`` selects BUDGET_EXHAUSTED over TIMED_OUT when the deadline
    passes (a Hermes limit, not the AMP timeout, was binding).
    """
    bound = str(bound_workitem_id or "").strip()
    started = time.time()
    polls = 0
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        time.sleep(min(max(float(interval_seconds), 1.0), remaining))
        polls += 1
        try:
            decision = fetch()
        except AmpClientError as exc:
            if exc.status_code in POLL_FATAL_STATUS:
                return PollOutcome(
                    POLL_REJECTED, bound, status_code=exc.status_code,
                    detail=exc.body or str(exc), polls=polls,
                )
            logger.warning("amp-governance %sdecision poll failed: %s", label, exc)
            continue
        except Exception as exc:  # never let a poll error turn into an allow
            logger.warning("amp-governance %sdecision poll failed: %s", label, exc)
            continue
        if not isinstance(decision, dict):
            continue

        if action_id is not None and str(decision.get("action_id") or "") != action_id:
            return PollOutcome(CORRELATION_FAILED, bound, detail="echo", polls=polls)

        status = _lower(decision.get("status"))
        workitem_id = str(decision.get("workitem_id") or "").strip()
        if workitem_id and workitem_id != bound:
            if bound or ledger.is_spent(instance_id, workitem_id):
                continue  # another call's workitem (H3): never act on it

        if status == "complete":
            if bound and not workitem_id:
                continue  # cannot prove the decision is ours
            ledger.mark_spent(instance_id, workitem_id)
            information = str(decision.get("information") or "").strip()
            cancelled = _lower(decision.get("method")) == "cancelled" or _lower(
                decision.get("workitem_status")
            ) in _CANCELLED
            resolution = _lower(decision.get("resolution"))
            if cancelled:
                kind = CANCELLED
            elif resolution in _APPROVE:
                kind = APPROVED
            elif resolution in _MODIFY:
                kind = MODIFIED
            else:
                kind = REJECTED  # reject, defer, unknown, missing: never an allow
            return PollOutcome(kind, workitem_id or bound, information=information, polls=polls)

        if status in _CANCELLED or _lower(decision.get("workitem_status")) in _CANCELLED:
            ledger.mark_spent(instance_id, workitem_id)
            information = str(decision.get("information") or "").strip()
            return PollOutcome(CANCELLED, workitem_id or bound, information=information, polls=polls)

        # pending (or an unrecognised status): keep waiting on this call's workitem
        if not bound and workitem_id:
            bound = workitem_id
        if (
            action_id is not None
            and watchdog_seconds
            and not workitem_id
            and time.time() - started >= watchdog_seconds
        ):
            return PollOutcome(CORRELATION_FAILED, bound, detail="watchdog", polls=polls)

    return PollOutcome(BUDGET_EXHAUSTED if budget_limited else TIMED_OUT, bound, polls=polls)
