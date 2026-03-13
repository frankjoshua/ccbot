import pytest
from ccbot.tmux_manager import make_session_name


def test_make_session_name_basic():
    name = make_session_name("/home/josh/development/workspace/agents/gtd")
    assert name == "gtd-agents"


def test_make_session_name_sanitize():
    name = make_session_name("/home/josh/My Projects/Test App")
    assert name == "test-app-my-projects"


def test_make_session_name_length_cap():
    long_path = "/home/josh/" + "a" * 100 + "/" + "b" * 100
    name = make_session_name(long_path)
    assert len(name) <= 60


def test_make_session_name_root():
    name = make_session_name("/")
    assert name == "root-root"
