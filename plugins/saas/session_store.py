from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

try:  # cross-process lock for read-modify-write of sessions.json (POSIX)
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows
    _fcntl = None  # type: ignore[assignment]

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# Length of the API-key digest kept in a fingerprint. The fingerprint only has
# to tell "same key" from "different key"; it must never hold the key itself.
_KEY_DIGEST_CHARS = 16
# Consumed/abandoned AMP workitems remembered per record, so a restarted process that
# reuses the instance_id cannot act on an old decision (contract K2.1).
SPENT_WORKITEMS_LIMIT = 50
_KEY_DIGEST_RE = re.compile(r"^[0-9a-f]{%d}$" % _KEY_DIGEST_CHARS)


# How long a store operation waits for another Hermes process holding sessions.json.
_STORE_LOCK_TIMEOUT_SECONDS = 5.0
_STORE_LOCK_POLL_SECONDS = 0.02
_STORE_LOCK_GRACE_SECONDS = 1.0


def _governance_deadline() -> Optional[float]:
    try:
        from .amp_client import current_deadline
    except ImportError:  # imported standalone (tests of this module alone)
        return None
    return current_deadline()


class StoreLockError(OSError):
    """sessions.json could not be locked in time (or locking is unavailable).

    Callers on a governance path must treat this as a block, never as an outage."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state_path() -> Path:
    # Uses the real hermes_constants.get_hermes_home() (env var + context-var
    # override + profile awareness + platform-native default) rather than a
    # local HERMES_HOME-env-var-only reimplementation, which silently wrote
    # session state to the wrong place under a non-default profile or on
    # Windows.
    state_dir = get_hermes_home() / "plugin_state" / "amp-governance"
    state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    return state_dir / "sessions.json"


def _key_digest(api_key: str) -> str:
    if not api_key:
        return ""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:_KEY_DIGEST_CHARS]


def agent_fingerprint(backend_url: str, org_id: str, agent_name: str, api_key: str) -> str:
    """Identify the AMP agent a cached instance belongs to without storing the key.

    sessions.json lives under HERMES_HOME, which can be world-readable (e.g. an
    NTFS mount under WSL), so only a short SHA-256 digest of the API key is kept.
    """
    return "|".join([backend_url, org_id, agent_name, _key_digest(api_key)])


def _scrub_fingerprint(fingerprint: str) -> str:
    """Rewrite a pre-0.5.3 fingerprint that ended in the raw API key.

    The result equals ``agent_fingerprint`` for the same key, so a migrated
    session keeps its AMP instance instead of being re-initialised.
    """
    parts = fingerprint.split("|")
    if len(parts) < 4:
        return fingerprint
    key = "|".join(parts[3:])
    if not key or _KEY_DIGEST_RE.match(key):
        return fingerprint
    return "|".join(parts[:3] + [_key_digest(key)])


@dataclass
class SessionRecord:
    session_id: str
    instance_id: str
    model: str = ""
    platform: str = ""
    created_at: str = ""
    last_seen_at: str = ""
    # Fingerprint of the AMP config (backend/org/agent/API-key digest) active
    # when this record was created. A cached instance_id is only safe to reuse
    # if the plugin is still talking to the same AMP agent — without this, a
    # Hermes session that outlives an .env change (e.g. re-pointing at a new
    # agent during QSJ setup) keeps reusing the old agent's instance_id and
    # never calls init_instance() again, so AMP never records a fresh
    # connection for the new agent. See agent_fingerprint().
    agent_fingerprint: str = ""
    # Newest last, at most SPENT_WORKITEMS_LIMIT entries.
    spent_workitem_ids: List[str] = field(default_factory=list)
    # Late-workitem marker (K2.1): newest workitem seen when a request went out whose own
    # workitem never appeared ("?" = unknown), and when (epoch seconds, as text).
    unregistered_after: str = ""
    unregistered_since: str = ""

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> Optional["SessionRecord"]:
        """Build a record from stored JSON; None if it cannot be trusted.

        Unknown keys (written by a newer plugin) are ignored; a record without
        session_id/instance_id, or with values of the wrong type, is dropped.
        """
        known = {key: value for key, value in payload.items() if key in _RECORD_FIELDS}
        if not known.get("session_id") or not known.get("instance_id"):
            return None
        spent = known.get("spent_workitem_ids", [])
        if not isinstance(spent, list) or not all(isinstance(item, str) for item in spent):
            return None
        if not all(isinstance(value, str) for key, value in known.items() if key != "spent_workitem_ids"):
            return None
        return cls(**known)


_RECORD_FIELDS = frozenset(field.name for field in fields(SessionRecord))


class SessionStore:
    def __init__(self, path: Optional[Path] = None, *, lock_timeout: float = _STORE_LOCK_TIMEOUT_SECONDS) -> None:
        self._path = path or _state_path()
        self._thread_lock = threading.Lock()
        self._lock_timeout = lock_timeout

    @contextmanager
    def _lock_ctx(self) -> Iterator[None]:
        """Thread lock plus a non-blocking, deadline-bound flock on sessions.json.lock, so
        another Hermes process cannot interleave a load/save and drop consumed workitems
        or review markers. Raises StoreLockError when the lock is not obtained in time."""
        deadline = time.time() + self._lock_timeout
        governance_deadline = _governance_deadline()
        if governance_deadline is not None:
            # Never outlast the hook's window; a short grace still lets the marker/spent
            # write that follows a timed-out review land inside the safety margin.
            deadline = min(deadline, governance_deadline + _STORE_LOCK_GRACE_SECONDS)
        if not self._thread_lock.acquire(timeout=max(deadline - time.time(), 0.0)):
            raise StoreLockError("session store busy (thread lock)")
        fd = None
        try:
            if _fcntl is not None:
                try:
                    fd = os.open(f"{self._path}.lock", os.O_RDWR | os.O_CREAT, 0o600)
                except OSError as exc:
                    raise StoreLockError(f"session store lock unavailable: {exc}") from exc
                while True:
                    try:
                        _fcntl.flock(fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.time() >= deadline:
                            raise StoreLockError("session store busy (another Hermes process)") from None
                        time.sleep(_STORE_LOCK_POLL_SECONDS)
                    except OSError as exc:
                        raise StoreLockError(f"session store lock unavailable: {exc}") from exc
            yield
        finally:
            if fd is not None:
                try:
                    _fcntl.flock(fd, _fcntl.LOCK_UN)
                except OSError:
                    pass
                os.close(fd)
            self._thread_lock.release()

    def _load_all(self) -> Tuple[Dict[str, SessionRecord], bool]:
        """Return (records, needs_rewrite). needs_rewrite is True when a record
        was dropped or migrated, so the caller persists the cleaned file."""
        if not self._path.exists():
            return {}, False
        try:
            raw = json.loads(self._path.read_text())
        except Exception as exc:
            logger.warning("amp-governance: ignoring unreadable session store %s: %s", self._path, exc)
            return {}, False
        if not isinstance(raw, dict):
            return {}, False
        out: Dict[str, SessionRecord] = {}
        dirty = False
        for session_id, payload in raw.items():
            record = SessionRecord.from_dict(payload) if isinstance(payload, dict) else None
            if record is None:
                logger.warning("amp-governance: dropping unreadable session record %r", session_id)
                dirty = True
                continue
            scrubbed = _scrub_fingerprint(record.agent_fingerprint)
            if scrubbed != record.agent_fingerprint:
                record.agent_fingerprint = scrubbed
                dirty = True
            out[session_id] = record
        return out, dirty

    def _save_all(self, data: Dict[str, SessionRecord]) -> None:
        serializable = {key: asdict(value) for key, value in data.items()}
        text = json.dumps(serializable, indent=2) + "\n"
        # mkstemp: unpredictable name, O_EXCL (no symlink following), mode 0600 from creation.
        fd, tmp_name = tempfile.mkstemp(prefix=f".{self._path.name}.", suffix=".tmp", dir=self._path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self._path)
        finally:
            if os.path.exists(tmp_name):
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass

    def _load_clean(self) -> Dict[str, SessionRecord]:
        data, dirty = self._load_all()
        if dirty:
            try:
                self._save_all(data)
            except OSError as exc:
                logger.warning("amp-governance: could not rewrite session store: %s", exc)
        return data

    def get(self, session_id: str) -> Optional[SessionRecord]:
        with self._lock_ctx():
            return self._load_clean().get(session_id)

    def put(self, record: SessionRecord) -> None:
        with self._lock_ctx():
            data, _ = self._load_all()
            record.last_seen_at = _utc_now()
            if not record.created_at:
                record.created_at = record.last_seen_at
            data[record.session_id] = record
            self._save_all(data)

    def remember_spent(self, instance_id: str, workitem_id: str) -> None:
        """Record a consumed/abandoned workitem on every record of ``instance_id`` (bounded)."""
        if not instance_id or not workitem_id:
            return
        with self._lock_ctx():
            data, dirty = self._load_all()
            for record in data.values():
                if record.instance_id != instance_id:
                    continue
                ids = [item for item in record.spent_workitem_ids if item != workitem_id]
                ids.append(workitem_id)
                record.spent_workitem_ids = ids[-SPENT_WORKITEMS_LIMIT:]
                dirty = True
            if dirty:
                self._save_all(data)

    def set_unregistered(self, instance_id: str, newest: str, since: float) -> bool:
        """Persist the late-workitem marker on every record of the instance; False if none."""
        return self._update_instance(instance_id, newest, f"{since:.3f}")

    def clear_unregistered(self, instance_id: str) -> None:
        self._update_instance(instance_id, "", "")

    def get_unregistered(self, instance_id: str) -> Optional[Tuple[str, float]]:
        with self._lock_ctx():
            for record in self._load_clean().values():
                if record.instance_id == instance_id and record.unregistered_since:
                    try:
                        return record.unregistered_after, float(record.unregistered_since)
                    except ValueError:
                        return record.unregistered_after, time.time()  # unreadable: fail closed
        return None

    def _update_instance(self, instance_id: str, newest: str, since: str) -> bool:
        with self._lock_ctx():
            data, dirty = self._load_all()
            found = False
            for record in data.values():
                if record.instance_id == instance_id:
                    found = True
                    if (record.unregistered_after, record.unregistered_since) != (newest, since):
                        record.unregistered_after, record.unregistered_since = newest, since
                        dirty = True
            if dirty:
                self._save_all(data)
            return found

    def spent_for_instance(self, instance_id: str) -> List[str]:
        with self._lock_ctx():
            ids: List[str] = []
            for record in self._load_clean().values():
                if record.instance_id == instance_id:
                    ids.extend(item for item in record.spent_workitem_ids if item not in ids)
            return ids

    def delete(self, session_id: str) -> None:
        with self._lock_ctx():
            data, dirty = self._load_all()
            if session_id in data:
                del data[session_id]
                dirty = True
            if dirty:
                self._save_all(data)
