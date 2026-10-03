from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict

from hermes_constants import get_hermes_home, get_env_path


def _hermes_home() -> Path:
    """Delegates to hermes_constants.get_hermes_home() — the real, single
    source of truth (env var override + context-var override + profile
    awareness + platform-native default). The plugin previously reimplemented
    a simplified version of this locally (HERMES_HOME env var only, POSIX-only
    default), which silently diverged from core on Windows and under
    `hermes profile use` without HERMES_HOME set."""
    return get_hermes_home()


def _read_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (
            (value.startswith('"') and value.endswith('"'))
            or (value.startswith("'") and value.endswith("'"))
        ):
            value = value[1:-1]
        values[key] = value
    return values


def _env_value(key: str, env_file: Dict[str, str], *aliases: str) -> str:
    for name in (key, *aliases):
        value = os.environ.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
        if name in env_file and str(env_file[name]).strip():
            return str(env_file[name]).strip()
    return ""


def _int_value(env_file: Dict[str, str], default: int, *names: str) -> int:
    for name in names:
        raw = _env_value(name, env_file)
        if not raw:
            continue
        try:
            return int(raw)
        except ValueError:
            continue
    return default


def _bool_value(env_file: Dict[str, str], default: bool, *names: str) -> bool:
    truthy = {"1", "true", "yes", "on"}
    falsy = {"0", "false", "no", "off"}
    for name in names:
        raw = _env_value(name, env_file).lower()
        if raw in truthy:
            return True
        if raw in falsy:
            return False
    return default


@dataclass(frozen=True)
class AmpConfig:
    backend_url: str
    api_key: str
    org_id: str
    agent_name: str
    username: str
    hitl_timeout_minutes: int
    hitl_poll_interval_seconds: int
    fail_closed: bool
    # Phase 1: Notification bridge
    notifications_enabled: bool
    # Phase 2A: LLM usage observation
    llm_governance_enabled: bool
    llm_governance_mode: str        # "observe" | "enforce"
    llm_governance_fail_closed: bool
    llm_governance_include_subagents: bool
    # Fields below are optional (defaults) so existing AmpConfig(...) callers keep working.
    # Opt-in action_id correlation of HITL request + decision poll (A6); needs AMP's HITL
    # service to persist the top-level action_id on the workitem.
    action_id_correlation: bool = False
    # Block when an action_id-scoped review still has no workitem after this many seconds.
    action_id_watchdog_seconds: int = 60

    @property
    def is_configured(self) -> bool:
        return all(
            [
                self.backend_url,
                self.api_key,
                self.org_id,
                self.agent_name,
                self.username,
            ]
        )


def load_config() -> AmpConfig:
    env_file = _read_env_file(get_env_path())
    backend_url = _env_value("AMP_BACKEND_URL", env_file)
    api_key = _env_value("AMP_API_KEY", env_file)
    org_id = _env_value("AMP_ORG_ID", env_file)
    username = _env_value("AMP_USERNAME", env_file)
    agent_name = _env_value("AMP_AGENT_NAME", env_file)
    llm_governance_mode_raw = _env_value("AMP_LLM_GOVERNANCE_MODE", env_file).strip().lower()
    llm_governance_mode = llm_governance_mode_raw if llm_governance_mode_raw in {"observe", "enforce"} else "observe"
    return AmpConfig(
        backend_url=backend_url.rstrip("/"),
        api_key=api_key.removeprefix("Bearer ").removeprefix("bearer ").strip(),
        org_id=org_id,
        agent_name=agent_name,
        username=username,
        hitl_timeout_minutes=_int_value(
            env_file,
            10,
            "AMP_HITL_TIMEOUT_MINUTES",
            "HITL_TIMEOUT_MINUTES",
        ),
        hitl_poll_interval_seconds=_int_value(
            env_file,
            3,
            "AMP_HITL_POLL_INTERVAL_SECONDS",
        ),
        fail_closed=_bool_value(env_file, True, "AMP_FAIL_CLOSED"),
        notifications_enabled=_bool_value(env_file, True, "AMP_NOTIFICATIONS_ENABLED"),
        llm_governance_enabled=_bool_value(env_file, True, "AMP_LLM_GOVERNANCE_ENABLED"),
        llm_governance_mode=llm_governance_mode,
        llm_governance_fail_closed=_bool_value(env_file, False, "AMP_LLM_GOVERNANCE_FAIL_CLOSED"),
        llm_governance_include_subagents=_bool_value(env_file, True, "AMP_LLM_GOVERNANCE_INCLUDE_SUBAGENTS"),
        action_id_correlation=_bool_value(env_file, False, "AMP_ACTION_ID_CORRELATION"),
        action_id_watchdog_seconds=_int_value(env_file, 60, "AMP_ACTION_ID_WATCHDOG_SECONDS"),
    )
