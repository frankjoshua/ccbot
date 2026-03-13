import pytest
from ccbot.tmux_manager import parse_window_ref


def test_parse_window_ref_composite():
    session, window = parse_window_ref("gtd-agents:@12")
    assert session == "gtd-agents"
    assert window == "@12"


def test_parse_window_ref_bare_id_raises():
    """Bare window IDs should raise — we always need the session."""
    with pytest.raises(ValueError):
        parse_window_ref("@12")


def test_parse_window_ref_empty_raises():
    with pytest.raises(ValueError):
        parse_window_ref("")


def test_parse_window_ref_no_window_raises():
    with pytest.raises(ValueError):
        parse_window_ref("session:")


def test_parse_window_ref_no_session_raises():
    with pytest.raises(ValueError):
        parse_window_ref(":@12")
