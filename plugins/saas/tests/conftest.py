"""Makes ``from hermes import ...`` work in the test suite regardless of what
this checkout's directory is actually named on disk.

The tests import this plugin's own code as ``hermes`` by convention (not to
be confused with the Hermes agent framework itself, which installs as
``hermes_cli`` -- no naming collision). That convention only worked by
accident on a checkout whose directory happens to be named ``hermes``
(e.g. via ``~/.hermes/plugins/amp-governance`` symlinked to a dev checkout
at .../hermes); a plain ``git clone`` of this repo lands in a directory
named ``amp-hermes-plugin``, where the bare import fails. Registering the
package under ``sys.modules["hermes"]`` here, before test collection,
makes the import work either way.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from urllib.error import URLError

import pytest

_PLUGIN_ROOT = Path(__file__).resolve().parent.parent

# Keep the suite hermetic: plugin state (sessions.json) goes to a throwaway
# HERMES_HOME, never the developer's real one, and set before the plugin module
# (and its module-level _PLUGIN) is imported below. An inherited HERMES_HOME
# (e.g. from .tools/env.sh) could point at a .env with production credentials.
os.environ["HERMES_HOME"] = tempfile.mkdtemp(prefix="amp-hermes-plugin-tests-")

if "hermes" not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        "hermes",
        _PLUGIN_ROOT / "__init__.py",
        submodule_search_locations=[str(_PLUGIN_ROOT)],
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["hermes"] = _mod
    _spec.loader.exec_module(_mod)


@pytest.fixture(autouse=True)
def _hermetic_amp(monkeypatch):
    """Default every test to: no Hermes host time limits, no real network.

    Tests that exercise host limits or HTTP patch these names themselves
    (``hermes.hook_timeout_seconds`` / ``hermes.tool_deadline_seconds`` /
    ``hermes.amp_client.urlopen``); their patches apply on top of these.
    """
    plugin = sys.modules["hermes"]
    monkeypatch.setattr(plugin, "hook_timeout_seconds", lambda: None)
    monkeypatch.setattr(plugin, "tool_deadline_seconds", lambda: None)

    def _no_network(*_args, **_kwargs):
        raise URLError("network disabled in unit tests")

    monkeypatch.setattr(sys.modules["hermes.amp_client"], "urlopen", _no_network)
    monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION", raising=False)
