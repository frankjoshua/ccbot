"""Tests for ccbot.hook — session_map key format and hook processing."""

import json
import os
from io import StringIO
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from ccbot.hook import hook_main


VALID_UUID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


@pytest.fixture
def tmp_ccbot_dir(tmp_path):
    """Point ccbot_dir() at a temp directory."""
    with patch.dict(os.environ, {"CCBOT_DIR": str(tmp_path)}):
        yield tmp_path


def _run_hook(pane_id, tmux_output, session_id=VALID_UUID, cwd="/home/josh/dev"):
    """Helper: invoke hook_main with mocked stdin, env, and tmux subprocess."""
    payload = {
        "session_id": session_id,
        "cwd": cwd,
        "hook_event_name": "SessionStart",
    }
    mock_result = MagicMock()
    mock_result.stdout = tmux_output + "\n"

    with (
        patch.dict(os.environ, {"TMUX_PANE": pane_id}),
        patch("sys.stdin", StringIO(json.dumps(payload))),
        patch("sys.argv", ["ccbot", "hook"]),
        patch("subprocess.run", return_value=mock_result) as mock_run,
    ):
        hook_main()

    return mock_run


def test_key_uses_real_tmux_session_name(tmp_ccbot_dir):
    """The session_map key must be '{real_session_name}:{window_id}', not hardcoded."""
    _run_hook(pane_id="%5", tmux_output="my-custom-session:@12:window-name")

    map_file = tmp_ccbot_dir / "session_map.json"
    session_map = json.loads(map_file.read_text())

    assert "my-custom-session:@12" in session_map
    entry = session_map["my-custom-session:@12"]
    assert entry["session_id"] == VALID_UUID
    assert entry["cwd"] == "/home/josh/dev"
    assert entry["window_name"] == "window-name"


def test_different_sessions_produce_different_keys(tmp_ccbot_dir):
    """Two windows in different tmux sessions get distinct keys."""
    _run_hook(pane_id="%1", tmux_output="ccbot:@3:agent1")
    _run_hook(pane_id="%2", tmux_output="gtd-agents:@5:gtd")

    session_map = json.loads((tmp_ccbot_dir / "session_map.json").read_text())

    assert "ccbot:@3" in session_map
    assert "gtd-agents:@5" in session_map
    assert session_map["ccbot:@3"]["window_name"] == "agent1"
    assert session_map["gtd-agents:@5"]["window_name"] == "gtd"


def test_same_session_different_windows(tmp_ccbot_dir):
    """Two windows in the SAME tmux session get separate keys."""
    _run_hook(pane_id="%1", tmux_output="dev:@10:editor")
    _run_hook(
        pane_id="%2",
        tmux_output="dev:@11:tests",
        session_id="bbbbbbbb-cccc-dddd-eeee-ffffffffffff",
    )

    session_map = json.loads((tmp_ccbot_dir / "session_map.json").read_text())

    assert "dev:@10" in session_map
    assert "dev:@11" in session_map
    assert session_map["dev:@10"]["session_id"] == VALID_UUID
    assert session_map["dev:@11"]["session_id"] == "bbbbbbbb-cccc-dddd-eeee-ffffffffffff"


def test_no_tmux_pane_skips_silently(tmp_ccbot_dir):
    """Without TMUX_PANE, the hook should exit without writing anything."""
    payload = {
        "session_id": VALID_UUID,
        "cwd": "/tmp",
        "hook_event_name": "SessionStart",
    }
    with (
        patch.dict(os.environ, {}, clear=False),
        patch("sys.stdin", StringIO(json.dumps(payload))),
        patch("sys.argv", ["ccbot", "hook"]),
    ):
        # Make sure TMUX_PANE is not set
        os.environ.pop("TMUX_PANE", None)
        hook_main()

    map_file = tmp_ccbot_dir / "session_map.json"
    assert not map_file.exists()


def test_old_format_key_cleaned_up(tmp_ccbot_dir):
    """If an old-format key (session:window_name) exists, it gets removed."""
    # Pre-populate with old-format key
    old_map = {
        "my-session:my-window": {
            "session_id": "old-sess",
            "cwd": "/old",
            "window_name": "my-window",
        }
    }
    map_file = tmp_ccbot_dir / "session_map.json"
    map_file.write_text(json.dumps(old_map))

    # Hook fires for same session + window but with proper @id key
    _run_hook(pane_id="%3", tmux_output="my-session:@7:my-window")

    session_map = json.loads(map_file.read_text())

    # New key present, old key removed
    assert "my-session:@7" in session_map
    assert "my-session:my-window" not in session_map
    assert session_map["my-session:@7"]["session_id"] == VALID_UUID
