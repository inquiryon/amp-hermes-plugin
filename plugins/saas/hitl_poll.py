"""Poll AMP's HITL decision endpoint for one governed call.

Shared by the tool, LLM and plan governance paths. Every outcome except
``APPROVED`` must end in a block (fix spec §2 rule 2).

Workitem guard (stale-decision race, H3/L16): ``GET /api/hitl/get-decision`` is
scoped to the AMP *instance*, not the call — it returns the newest workitem of
the instance. A call therefore binds to the first ``workitem_id`` it sees (the
request response, else its first pending poll) and only accepts a ``complete``
decision for that workitem. Before a workitem is bound, decisions for
workitems this process already consumed or abandoned, or that another live
call has claimed, are ignored. An approval without a workitem_id is only
accepted when action_id correlation proves it belongs to this call.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import logging
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple
import uuid

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
    * claimed: workitems a live poll is bound to (owner token). Other polls on
      the same instance never bind to or consume them.
    * pending reviews: workitems still awaiting a reviewer when the call gave
      up, keyed by the call's (tool, action, args) fingerprint, so an identical
      retry waits on the same review instead of opening a new one (A19).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._spent: Dict[str, "OrderedDict[str, None]"] = {}
        self._pending: Dict[str, "OrderedDict[str, Tuple[str, Optional[str]]]"] = {}
        self._claims: Dict[str, Dict[str, str]] = {}

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

    def claim(self, instance_id: str, workitem_id: str, owner: str, *, force: bool = False) -> bool:
        """Bind ``workitem_id`` to one live poll. False if it is spent or another poll holds it.

        ``force`` is for a workitem AMP returned for this very request (or an A19 reuse):
        that binding is authoritative, so it takes the claim and clears "spent".
        """
        if not workitem_id:
            return False
        with self._lock:
            claims = self._claims.setdefault(instance_id, {})
            if force:
                self._spent.get(instance_id, OrderedDict()).pop(workitem_id, None)
                claims[workitem_id] = owner
                return True
            if workitem_id in self._spent.get(instance_id, ()):
                return False
            holder = claims.get(workitem_id)
            if holder is not None and holder != owner:
                return False
            claims[workitem_id] = owner
            return True

    def usable_by(self, instance_id: str, workitem_id: str, owner: str) -> bool:
        """True when ``owner`` may act on ``workitem_id`` (not spent, not claimed by another poll)."""
        with self._lock:
            if workitem_id in self._spent.get(instance_id, ()):
                return False
            holder = self._claims.get(instance_id, {}).get(workitem_id)
            return holder is None or holder == owner

    def try_consume(self, instance_id: str, workitem_id: str, owner: str) -> bool:
        """Atomically check-and-mark a decision as consumed by ``owner``."""
        with self._lock:
            if workitem_id in self._spent.get(instance_id, ()):
                return False
            claims = self._claims.get(instance_id, {})
            holder = claims.get(workitem_id)
            if holder is not None and holder != owner:
                return False
            claims.pop(workitem_id, None)
            self._bounded_put(self._spent.setdefault(instance_id, OrderedDict()), workitem_id, None)
            return True

    def release(self, instance_id: str, owner: str) -> None:
        """Drop every claim ``owner`` still holds (the poll ended)."""
        with self._lock:
            claims = self._claims.get(instance_id, {})
            for workitem_id in [w for w, holder in claims.items() if holder == owner]:
                del claims[workitem_id]

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
            self._claims.pop(instance_id, None)


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

    ``bound_workitem_id`` is the workitem AMP returned for this call's request
    (authoritative). ``budget_limited`` selects BUDGET_EXHAUSTED over TIMED_OUT
    when the deadline passes (a Hermes limit, not the AMP timeout, was binding).
    """
    owner = uuid.uuid4().hex
    bound = str(bound_workitem_id or "").strip()
    if bound:
        ledger.claim(instance_id, bound, owner, force=True)
    try:
        return _poll(fetch, instance_id, ledger, owner, bound, deadline, interval_seconds,
                     budget_limited, action_id, watchdog_seconds, label)
    finally:
        ledger.release(instance_id, owner)


def _poll(
    fetch: Callable[[], Dict[str, Any]],
    instance_id: str,
    ledger: DecisionLedger,
    owner: str,
    bound: str,
    deadline: float,
    interval_seconds: float,
    budget_limited: bool,
    action_id: Optional[str],
    watchdog_seconds: Optional[float],
    label: str,
) -> PollOutcome:
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
            if bound or not ledger.usable_by(instance_id, workitem_id, owner):
                continue  # another call's workitem (H3): never act on it

        if status == "complete":
            resolution = _lower(decision.get("resolution"))
            if not workitem_id:
                # Cannot prove the decision is ours. Only a correlated (action_id-echoed)
                # or tighten-only decision may be taken without a workitem id.
                if bound or (resolution in _APPROVE | _MODIFY and action_id is None):
                    continue
            elif not ledger.try_consume(instance_id, workitem_id, owner):
                continue  # consumed by another poll in the meantime
            information = str(decision.get("information") or "").strip()
            cancelled = _lower(decision.get("method")) == "cancelled" or _lower(
                decision.get("workitem_status")
            ) in _CANCELLED
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
        if not bound and workitem_id and ledger.claim(instance_id, workitem_id, owner):
            bound = workitem_id
        if (
            action_id is not None
            and watchdog_seconds
            and not workitem_id
            and time.time() - started >= watchdog_seconds
        ):
            return PollOutcome(CORRELATION_FAILED, bound, detail="watchdog", polls=polls)

    return PollOutcome(BUDGET_EXHAUSTED if budget_limited else TIMED_OUT, bound, polls=polls)
