"""Tag forwarded messages with the sender's identity.

Multiple Telegram users (ALLOWED_USERS) can share one bridge. Claude receives
forwarded text as anonymous keystrokes, so without help it cannot tell who is
talking. This module prefixes messages from non-primary users with the sender's
name (e.g. "[Laura via Telegram] hi") while leaving the primary user's
messages untouched.

Slash commands and "!" bash commands are never tagged: a prefix would break
their special handling (command routing, bash output capture).
"""


def tag_sender_text(
    sender_id: int,
    sender_name: str | None,
    text: str,
    primary_user_id: int,
) -> str:
    """Prefix ``text`` with the sender's name unless it is the primary user.

    ``sender_name`` should be the sender's Telegram first name or username;
    falls back to the numeric id when absent.
    """
    if sender_id == primary_user_id:
        return text
    if text.startswith(("/", "!")):
        return text
    name = (sender_name or "").strip() or str(sender_id)
    return f"[{name} via Telegram] {text}"
