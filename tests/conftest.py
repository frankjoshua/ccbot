"""Global test isolation: never let tests touch the real ~/.ccbot state.

The production ``Config`` singleton resolves state paths at call time
(``session.py`` save/load use ``config.state_file`` etc.), so pointing those
attributes at a per-test ``tmp_path`` guarantees no test can write production
state — regardless of how individual tests patch or construct managers.

Regression guard for the 2026-06-09 incident: running pytest wiped
``~/.ccbot/state.json`` (test fixtures patched config only during
``SessionManager()`` construction, so later mutating calls saved through the
real config), silently killing every Telegram thread binding.
"""

import pytest

from ccbot.config import config


@pytest.fixture(autouse=True)
def isolate_ccbot_state(tmp_path, monkeypatch):
    """Redirect every state path on the shared config singleton to tmp_path."""
    monkeypatch.setattr(config, "config_dir", tmp_path)
    monkeypatch.setattr(config, "state_file", tmp_path / "state.json")
    monkeypatch.setattr(config, "session_map_file", tmp_path / "session_map.json")
    monkeypatch.setattr(
        config, "monitor_state_file", tmp_path / "monitor_state.json"
    )
    monkeypatch.setattr(config, "claude_projects_path", tmp_path / "projects")
