"""Shared fakes for the HITL governance tests (fake clock, plugin factory)."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import Mock, patch

from hermes import AmpGovernancePlugin
from hermes.policy import NormalizedAction


class FakeClock:
    """Deterministic time: time.time() returns ``now``; time.sleep(s) advances it."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.start = start
        self.now = start
        self.sleeps: list = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += max(float(seconds), 0.0)

    @property
    def elapsed(self) -> float:
        return self.now - self.start

    @contextmanager
    def installed(self) -> Iterator["FakeClock"]:
        with patch("time.time", self.time), patch("time.sleep", self.sleep):
            yield self


def make_config(**overrides: Any) -> SimpleNamespace:
    values = dict(
        is_configured=True,
        backend_url="http://amp.invalid",
        api_key="amp_k_test_secret_value",
        fail_closed=True,
        agent_name="amp-test",
        username="tester",
        org_id="O-test",
        hitl_timeout_minutes=10,
        hitl_poll_interval_seconds=1,
        notifications_enabled=False,
        llm_governance_enabled=False,
        llm_governance_mode="observe",
        llm_governance_fail_closed=False,
        llm_governance_include_subagents=True,
        action_id_correlation=False,
        action_id_watchdog_seconds=60,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def make_plugin(**config_overrides: Any) -> AmpGovernancePlugin:
    """Plugin with a Mock AMP client, a SimpleNamespace config and no notifications."""
    plugin = AmpGovernancePlugin()
    plugin._config = make_config(**config_overrides)
    plugin._client = Mock()
    plugin._notify_user = Mock()
    return plugin


def terminal_action(command: str = "ls") -> NormalizedAction:
    return NormalizedAction("terminal", "exec", "exec", {"command": command})


@contextmanager
def quiet(plugin: AmpGovernancePlugin, instance_id: str = "inst-1") -> Iterator[ExitStack]:
    """Silence AMP log calls and pin the AMP instance id."""
    with ExitStack() as stack:
        stack.enter_context(patch.object(plugin, "_safe_log"))
        stack.enter_context(patch.object(plugin, "_ensure_instance", return_value=instance_id))
        yield stack


def pending(workitem_id: str = "wi-1", **extra: Any) -> dict:
    return {"status": "pending", "workitem_id": workitem_id, **extra}


def complete(resolution: str, workitem_id: str = "wi-1", information: str = "", **extra: Any) -> dict:
    return {
        "status": "complete",
        "resolution": resolution,
        "method": "human",
        "workitem_id": workitem_id,
        "information": information,
        **extra,
    }
