"""A5 (robust session store) and A17 (no raw API key at rest)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from hermes.session_store import SessionRecord, SessionStore, agent_fingerprint
from hitl_helpers import make_plugin

RAW_KEY = "amp_k_test_0123456789abcdefSECRET"


class _TempStore(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "sessions.json"

    def seed(self, payload) -> SessionStore:
        self.path.write_text(json.dumps(payload))
        return SessionStore(path=self.path)

    def on_disk(self) -> dict:
        return json.loads(self.path.read_text())


class SessionStoreSchemaTests(_TempStore):
    def test_extra_key_is_ignored(self) -> None:
        store = self.seed({"s1": {"session_id": "s1", "instance_id": "i1", "future_field": 1}})
        self.assertEqual(store.get("s1").instance_id, "i1")

    def test_missing_required_key_drops_record(self) -> None:
        store = self.seed({"s1": {"session_id": "s1"}, "s2": {"session_id": "s2", "instance_id": "i2"}})
        self.assertIsNone(store.get("s1"))
        self.assertEqual(store.get("s2").instance_id, "i2")

    def test_wrong_value_type_drops_record(self) -> None:
        store = self.seed({"s1": {"session_id": "s1", "instance_id": ["not", "a", "string"]}})
        self.assertIsNone(store.get("s1"))

    def test_put_after_corrupt_record_rewrites_file(self) -> None:
        store = self.seed({"s1": {"session_id": "s1"}, "s2": {"session_id": "s2", "instance_id": "i2"}})
        store.put(SessionRecord("s3", "i3"))
        self.assertEqual(sorted(self.on_disk()), ["s2", "s3"])

    def test_unparseable_file_is_treated_as_empty(self) -> None:
        self.path.write_text("{not json")
        store = SessionStore(path=self.path)
        self.assertIsNone(store.get("s1"))
        store.put(SessionRecord("s1", "i1"))
        self.assertEqual(store.get("s1").instance_id, "i1")

    def test_save_leaves_no_temp_file(self) -> None:
        store = SessionStore(path=self.path)
        store.put(SessionRecord("s1", "i1"))
        self.assertEqual([p.name for p in self.path.parent.iterdir()], ["sessions.json"])

    @unittest.skipIf(os.name == "nt", "POSIX modes only")
    def test_file_is_private_where_the_filesystem_supports_it(self) -> None:
        store = SessionStore(path=self.path)
        store.put(SessionRecord("s1", "i1"))
        mode = self.path.stat().st_mode & 0o777
        # NTFS mounts (WSL /mnt/c) report 0o777 whatever chmod does; ext4 honours 0o600.
        self.assertIn(mode, {0o600, 0o777})


class ApiKeyAtRestTests(_TempStore):
    def test_fingerprint_never_contains_the_raw_key(self) -> None:
        fp = agent_fingerprint("https://amp", "O-1", "agent", RAW_KEY)
        self.assertNotIn(RAW_KEY, fp)
        self.assertNotIn("amp_k_", fp)
        self.assertRegex(fp.split("|")[-1], r"^[0-9a-f]{16}$")
        self.assertNotEqual(fp, agent_fingerprint("https://amp", "O-1", "agent", RAW_KEY + "x"))

    def test_plugin_writes_hashed_fingerprint_only(self) -> None:
        plugin = make_plugin(api_key=RAW_KEY)
        plugin._store = SessionStore(path=self.path)
        plugin._client.init_instance = Mock(return_value="inst-1")
        with patch.object(plugin, "_safe_log"):
            plugin._ensure_instance("s1")
        self.assertNotIn(RAW_KEY, self.path.read_text())
        self.assertNotIn("amp_k_", self.path.read_text())

    def test_legacy_raw_key_record_is_migrated_on_load_and_keeps_instance(self) -> None:
        legacy_fp = f"http://amp.invalid|O-test|amp-test|{RAW_KEY}"
        self.seed({"s1": {"session_id": "s1", "instance_id": "inst-old", "agent_fingerprint": legacy_fp}})
        plugin = make_plugin(api_key=RAW_KEY)
        plugin._store = SessionStore(path=self.path)
        plugin._client.init_instance = Mock(return_value="inst-new")
        with patch.object(plugin, "_safe_log"):
            instance_id = plugin._ensure_instance("s1")
        self.assertEqual(instance_id, "inst-old")  # same agent: no re-init
        plugin._client.init_instance.assert_not_called()
        self.assertNotIn(RAW_KEY, self.path.read_text())

    def test_legacy_records_of_other_sessions_are_scrubbed_too(self) -> None:
        legacy_fp = f"https://amp|O-1|agent|{RAW_KEY}"
        store = self.seed({
            "s1": {"session_id": "s1", "instance_id": "i1", "agent_fingerprint": legacy_fp},
            "s2": {"session_id": "s2", "instance_id": "i2", "agent_fingerprint": legacy_fp},
        })
        store.get("s1")
        text = self.path.read_text()
        self.assertNotIn(RAW_KEY, text)
        self.assertEqual(len(self.on_disk()), 2)


if __name__ == "__main__":
    unittest.main()
