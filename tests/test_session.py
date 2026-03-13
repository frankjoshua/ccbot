import pytest
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from ccbot.session import SessionManager, WindowState


@pytest.fixture
def sm(tmp_path):
    """SessionManager with temp state files, bypassing Config."""
    # Patch config to avoid needing env vars
    mock_config = MagicMock()
    mock_config.state_file = tmp_path / "state.json"
    mock_config.session_map_file = tmp_path / "session_map.json"
    mock_config.claude_projects_path = tmp_path / "projects"

    with patch("ccbot.session.config", mock_config):
        mgr = SessionManager()
    # Re-assign after init so tests can access
    mgr._config = mock_config
    return mgr


def test_bind_thread_composite(sm):
    sm.bind_thread(user_id=123, thread_id=456, window_id="gtd-agents:@12")
    result = sm.resolve_window_for_thread(123, 456)
    assert result == "gtd-agents:@12"


def test_window_states_composite_key(sm):
    sm.window_states["gtd-agents:@12"] = WindowState(
        session_id="abc-123",
        cwd="/home/josh/dev/gtd",
        window_name="gtd",
    )
    assert "gtd-agents:@12" in sm.window_states
    assert sm.window_states["gtd-agents:@12"].session_id == "abc-123"


def test_iter_thread_bindings_composite(sm):
    sm.bind_thread(user_id=100, thread_id=200, window_id="ccbot:@5")
    sm.bind_thread(user_id=100, thread_id=300, window_id="gtd-agents:@12")
    results = list(sm.iter_thread_bindings())
    assert len(results) == 2
    wids = {wid for _, _, wid in results}
    assert "ccbot:@5" in wids
    assert "gtd-agents:@12" in wids


def test_unbind_thread_returns_composite(sm):
    sm.bind_thread(user_id=100, thread_id=200, window_id="gtd-agents:@7")
    result = sm.unbind_thread(100, 200)
    assert result == "gtd-agents:@7"


def test_get_display_name_composite(sm):
    sm.window_display_names["gtd-agents:@12"] = "my-window"
    assert sm.get_display_name("gtd-agents:@12") == "my-window"
    # Fallback: returns the ref itself
    assert sm.get_display_name("ccbot:@99") == "ccbot:@99"


def test_update_display_name_composite(sm):
    sm.window_states["gtd-agents:@12"] = WindowState(
        session_id="abc", cwd="/tmp", window_name="old"
    )
    with patch("ccbot.session.config", sm._config):
        sm.update_display_name("gtd-agents:@12", "new-name")
    assert sm.window_display_names["gtd-agents:@12"] == "new-name"
    assert sm.window_states["gtd-agents:@12"].window_name == "new-name"


def test_user_window_offsets_composite(sm):
    sm.update_user_window_offset(user_id=100, window_id="gtd-agents:@12", offset=4096)
    assert sm.user_window_offsets[100]["gtd-agents:@12"] == 4096


def test_is_composite_ref(sm):
    assert sm._is_composite_ref("gtd-agents:@12") is True
    assert sm._is_composite_ref("ccbot:@0") is True
    assert sm._is_composite_ref("my-session:@99") is True
    # Bare window IDs are not composite refs
    assert sm._is_composite_ref("@12") is False
    # Plain names are not composite refs
    assert sm._is_composite_ref("my-window") is False


def test_is_window_id_still_works(sm):
    """_is_window_id should still detect bare window IDs."""
    assert sm._is_window_id("@12") is True
    assert sm._is_window_id("@0") is True
    assert sm._is_window_id("gtd-agents:@12") is False
    assert sm._is_window_id("window-name") is False


@pytest.mark.asyncio
async def test_load_session_map_composite_keys(sm, tmp_path):
    """load_session_map should store composite keys in window_states."""
    session_map = {
        "ccbot:@18": {
            "session_id": "sess-aaa",
            "cwd": "/home/josh/dev/ccbot",
            "window_name": "ccbot-dev",
        },
        "gtd-agents:@5": {
            "session_id": "sess-bbb",
            "cwd": "/home/josh/dev/gtd",
            "window_name": "gtd",
        },
    }
    sm._config.session_map_file.write_text(json.dumps(session_map))

    with patch("ccbot.session.config", sm._config):
        await sm.load_session_map()

    # Both should be present with composite keys
    assert "ccbot:@18" in sm.window_states
    assert "gtd-agents:@5" in sm.window_states
    assert sm.window_states["ccbot:@18"].session_id == "sess-aaa"
    assert sm.window_states["gtd-agents:@5"].session_id == "sess-bbb"
    # Display names should be set
    assert sm.window_display_names.get("ccbot:@18") == "ccbot-dev"
    assert sm.window_display_names.get("gtd-agents:@5") == "gtd"


@pytest.mark.asyncio
async def test_load_session_map_no_filter_by_session(sm, tmp_path):
    """load_session_map should load entries from ALL sessions."""
    session_map = {
        "other-session:@1": {
            "session_id": "sess-other",
            "cwd": "/tmp",
            "window_name": "other",
        },
    }
    sm._config.session_map_file.write_text(json.dumps(session_map))

    with patch("ccbot.session.config", sm._config):
        await sm.load_session_map()

    assert "other-session:@1" in sm.window_states
    assert sm.window_states["other-session:@1"].session_id == "sess-other"


@pytest.mark.asyncio
async def test_find_users_for_session_composite(sm):
    """find_users_for_session should work with composite keys in thread_bindings."""
    sm.window_states["gtd-agents:@12"] = WindowState(
        session_id="abc-123",
        cwd="/home/josh/dev/gtd",
        window_name="gtd",
    )
    sm.thread_bindings = {123: {456: "gtd-agents:@12"}}

    # Mock resolve_session_for_window to avoid file I/O
    from ccbot.session import ClaudeSession

    mock_session = ClaudeSession(
        session_id="abc-123",
        summary="test",
        message_count=5,
        file_path="/tmp/abc-123.jsonl",
    )
    with patch.object(sm, "resolve_session_for_window", return_value=mock_session):
        users = await sm.find_users_for_session("abc-123")

    assert len(users) >= 1
    user_id, window_id, thread_id = users[0]
    assert user_id == 123
    assert window_id == "gtd-agents:@12"
    assert thread_id == 456


@pytest.mark.asyncio
async def test_resolve_stale_ids_composite(sm):
    """resolve_stale_ids should match by session_name + display_name."""
    # Set up stale composite ref
    sm.window_states["gtd-agents:@99"] = WindowState(
        session_id="old-sess",
        cwd="/home/josh/dev/gtd",
        window_name="gtd",
    )
    sm.window_display_names["gtd-agents:@99"] = "gtd"
    sm.thread_bindings = {100: {200: "gtd-agents:@99"}}
    sm.user_window_offsets = {100: {"gtd-agents:@99": 1024}}

    # Mock tmux_manager.list_windows to return new ref for same window name
    from ccbot.tmux_manager import TmuxWindow

    mock_window = TmuxWindow(
        ref="gtd-agents:@42",
        window_id="@42",
        session_name="gtd-agents",
        window_name="gtd",
        cwd="/home/josh/dev/gtd",
    )
    mock_list = AsyncMock(return_value=[mock_window])

    with (
        patch("ccbot.session.tmux_manager") as mock_tm,
        patch("ccbot.session.config", sm._config),
    ):
        mock_tm.list_windows = mock_list
        await sm.resolve_stale_ids()

    # Should have re-resolved to new composite ref
    assert "gtd-agents:@42" in sm.window_states
    assert "gtd-agents:@99" not in sm.window_states
    assert sm.thread_bindings[100][200] == "gtd-agents:@42"
    assert sm.user_window_offsets[100].get("gtd-agents:@42") == 1024
    assert "gtd-agents:@99" not in sm.user_window_offsets[100]


@pytest.mark.asyncio
async def test_resolve_stale_ids_drops_dead_refs(sm):
    """resolve_stale_ids should drop composite refs for windows no longer alive."""
    sm.window_states["ccbot:@50"] = WindowState(
        session_id="dead-sess",
        cwd="/tmp",
        window_name="dead",
    )
    sm.window_display_names["ccbot:@50"] = "dead"
    sm.thread_bindings = {100: {300: "ccbot:@50"}}

    # No live windows at all
    mock_list = AsyncMock(return_value=[])

    with (
        patch("ccbot.session.tmux_manager") as mock_tm,
        patch("ccbot.session.config", sm._config),
    ):
        mock_tm.list_windows = mock_list
        await sm.resolve_stale_ids()

    assert "ccbot:@50" not in sm.window_states
    assert 100 not in sm.thread_bindings  # empty user removed


def test_save_and_load_state_composite(sm, tmp_path):
    """State round-trip should preserve composite keys."""
    sm.window_states["gtd-agents:@12"] = WindowState(
        session_id="abc", cwd="/tmp/gtd", window_name="gtd"
    )
    sm.thread_bindings = {100: {200: "gtd-agents:@12"}}
    sm.user_window_offsets = {100: {"gtd-agents:@12": 512}}
    sm.window_display_names = {"gtd-agents:@12": "gtd"}

    with patch("ccbot.session.config", sm._config):
        sm._save_state()

    # Create fresh manager and load
    with patch("ccbot.session.config", sm._config):
        sm2 = SessionManager()

    assert "gtd-agents:@12" in sm2.window_states
    assert sm2.window_states["gtd-agents:@12"].session_id == "abc"
    assert sm2.thread_bindings[100][200] == "gtd-agents:@12"
    assert sm2.user_window_offsets[100]["gtd-agents:@12"] == 512
    assert sm2.window_display_names["gtd-agents:@12"] == "gtd"


def test_load_state_migrates_bare_ids(sm, tmp_path):
    """Loading state with bare window IDs should detect them as needing migration."""
    state = {
        "window_states": {
            "@12": {"session_id": "old", "cwd": "/tmp"},
        },
        "thread_bindings": {
            "100": {"200": "@12"},
        },
        "user_window_offsets": {},
        "window_display_names": {},
        "group_chat_ids": {},
    }
    sm._config.state_file.write_text(json.dumps(state))

    with patch("ccbot.session.config", sm._config):
        sm2 = SessionManager()

    # Bare IDs should be loaded but flagged for migration
    # (migration happens in resolve_stale_ids, not _load_state)
    assert "@12" in sm2.window_states
    assert sm2.thread_bindings[100][200] == "@12"
