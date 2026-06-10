"""Tests for sender tagging of forwarded messages."""

from ccbot.sender_tag import tag_sender_text

PRIMARY = 7716121391
OTHER = 8364672384


def test_primary_user_is_never_tagged():
    assert tag_sender_text(PRIMARY, "Josh", "hello", PRIMARY) == "hello"


def test_non_primary_user_is_tagged_with_name():
    assert (
        tag_sender_text(OTHER, "Laura", "show me the game", PRIMARY)
        == "[Laura via Telegram] show me the game"
    )


def test_slash_commands_are_not_tagged():
    assert tag_sender_text(OTHER, "Laura", "/clear", PRIMARY) == "/clear"


def test_bash_commands_are_not_tagged():
    assert tag_sender_text(OTHER, "Laura", "!ls -la", PRIMARY) == "!ls -la"


def test_missing_name_falls_back_to_id():
    assert (
        tag_sender_text(OTHER, None, "hi", PRIMARY)
        == f"[{OTHER} via Telegram] hi"
    )


def test_blank_name_falls_back_to_id():
    assert (
        tag_sender_text(OTHER, "  ", "hi", PRIMARY)
        == f"[{OTHER} via Telegram] hi"
    )
