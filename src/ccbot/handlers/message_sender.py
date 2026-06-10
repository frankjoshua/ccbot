"""Safe message sending helpers with MarkdownV2 fallback.

Provides utility functions for sending Telegram messages with automatic
format conversion and fallback to plain text on failure.

Uses telegramify-markdown for MarkdownV2 formatting.

Functions:
  - send_with_fallback: Send with formatting → plain text fallback
  - send_photo: Photo sending (single or media group)
  - safe_reply: Reply with formatting, fallback to plain text
  - safe_edit: Edit message with formatting, fallback to plain text
  - safe_send: Send message with formatting, fallback to plain text

Rate limiting is handled globally by AIORateLimiter on the Application.
RetryAfter exceptions are re-raised so callers (queue worker) can handle them.
"""

import io
import logging
import time
from typing import Any

from telegram import Bot, InputMediaPhoto, LinkPreviewOptions, Message
from telegram.error import RetryAfter

from ..markdown_v2 import convert_markdown
from ..transcript_parser import TranscriptParser

logger = logging.getLogger(__name__)

# Circuit breaker: suppress sends to threads that fail repeatedly.
# Key: (chat_id, thread_id_or_0) -> (consecutive_failures, suppressed_until_monotonic)
_thread_failures: dict[tuple[int, int], tuple[int, float]] = {}
CIRCUIT_BREAKER_THRESHOLD = 3  # failures before suppressing
CIRCUIT_BREAKER_COOLDOWN = 60.0  # seconds to suppress after tripping


def _circuit_open(chat_id: int, thread_id: int | None) -> bool:
    """Return True if sends to this (chat, thread) are suppressed."""
    key = (chat_id, thread_id or 0)
    info = _thread_failures.get(key)
    if not info:
        return False
    failures, suppressed_until = info
    if failures < CIRCUIT_BREAKER_THRESHOLD:
        return False
    if time.monotonic() >= suppressed_until:
        # Cooldown expired — reset and allow retry
        _thread_failures.pop(key, None)
        logger.info(
            "Circuit breaker recovered: chat_id=%d thread_id=%s",
            chat_id,
            thread_id,
        )
        return False
    return True


def _record_failure(chat_id: int, thread_id: int | None) -> None:
    """Record a send failure. Trips circuit breaker after threshold."""
    key = (chat_id, thread_id or 0)
    info = _thread_failures.get(key)
    failures = (info[0] if info else 0) + 1
    suppressed_until = time.monotonic() + CIRCUIT_BREAKER_COOLDOWN
    _thread_failures[key] = (failures, suppressed_until)
    if failures == CIRCUIT_BREAKER_THRESHOLD:
        logger.warning(
            "Circuit breaker tripped: chat_id=%d thread_id=%s — "
            "suppressing sends for %ds after %d consecutive failures",
            chat_id,
            thread_id,
            int(CIRCUIT_BREAKER_COOLDOWN),
            failures,
        )


def _record_success(chat_id: int, thread_id: int | None) -> None:
    """Reset failure tracking on successful send."""
    key = (chat_id, thread_id or 0)
    if key in _thread_failures:
        _thread_failures.pop(key, None)


def strip_sentinels(text: str) -> str:
    """Strip expandable quote sentinel markers for plain text fallback."""
    for s in (
        TranscriptParser.EXPANDABLE_QUOTE_START,
        TranscriptParser.EXPANDABLE_QUOTE_END,
    ):
        text = text.replace(s, "")
    return text


def _ensure_formatted(text: str) -> str:
    """Convert markdown to MarkdownV2."""
    return convert_markdown(text)


PARSE_MODE = "MarkdownV2"


# Disable link previews in all messages to reduce visual noise
NO_LINK_PREVIEW = LinkPreviewOptions(is_disabled=True)


async def send_with_fallback(
    bot: Bot,
    chat_id: int,
    text: str,
    **kwargs: Any,
) -> Message | None:
    """Send message with MarkdownV2, falling back to plain text on failure.

    Returns the sent Message on success, None on failure.
    RetryAfter is re-raised for caller handling.
    """
    thread_id = kwargs.get("message_thread_id")
    if _circuit_open(chat_id, thread_id):
        return None
    kwargs.setdefault("link_preview_options", NO_LINK_PREVIEW)
    try:
        msg = await bot.send_message(
            chat_id=chat_id,
            text=_ensure_formatted(text),
            parse_mode=PARSE_MODE,
            **kwargs,
        )
        _record_success(chat_id, thread_id)
        return msg
    except RetryAfter:
        raise
    except Exception:
        try:
            msg = await bot.send_message(
                chat_id=chat_id, text=strip_sentinels(text), **kwargs
            )
            _record_success(chat_id, thread_id)
            return msg
        except RetryAfter:
            raise
        except Exception as e:
            _record_failure(chat_id, thread_id)
            logger.error(
                "Failed to send message to chat_id=%d thread_id=%s: %s",
                chat_id,
                thread_id,
                e,
            )
            return None


async def send_photo(
    bot: Bot,
    chat_id: int,
    image_data: list[tuple[str, bytes]],
    **kwargs: Any,
) -> None:
    """Send photo(s) to chat. Sends as media group if multiple images.

    Rate limiting is handled globally by AIORateLimiter on the Application.

    Args:
        bot: Telegram Bot instance
        chat_id: Target chat ID
        image_data: List of (media_type, raw_bytes) tuples
        **kwargs: Extra kwargs passed to send_photo/send_media_group
    """
    if not image_data:
        return
    thread_id = kwargs.get("message_thread_id")
    if _circuit_open(chat_id, thread_id):
        return
    try:
        if len(image_data) == 1:
            _media_type, raw_bytes = image_data[0]
            await bot.send_photo(
                chat_id=chat_id,
                photo=io.BytesIO(raw_bytes),
                **kwargs,
            )
        else:
            media = [
                InputMediaPhoto(media=io.BytesIO(raw_bytes))
                for _media_type, raw_bytes in image_data
            ]
            await bot.send_media_group(
                chat_id=chat_id,
                media=media,
                **kwargs,
            )
        _record_success(chat_id, thread_id)
    except RetryAfter:
        raise
    except Exception as e:
        _record_failure(chat_id, thread_id)
        logger.error(
            "Failed to send photo to chat_id=%d thread_id=%s: %s",
            chat_id,
            thread_id,
            e,
        )


async def safe_reply(message: Message, text: str, **kwargs: Any) -> Message:
    """Reply with formatting, falling back to plain text on failure."""
    kwargs.setdefault("link_preview_options", NO_LINK_PREVIEW)
    try:
        return await message.reply_text(
            _ensure_formatted(text),
            parse_mode=PARSE_MODE,
            **kwargs,
        )
    except RetryAfter:
        raise
    except Exception:
        try:
            return await message.reply_text(strip_sentinels(text), **kwargs)
        except RetryAfter:
            raise
        except Exception as e:
            logger.error(f"Failed to reply: {e}")
            raise


async def safe_edit(target: Any, text: str, **kwargs: Any) -> None:
    """Edit message with formatting, falling back to plain text on failure."""
    kwargs.setdefault("link_preview_options", NO_LINK_PREVIEW)
    try:
        await target.edit_message_text(
            _ensure_formatted(text),
            parse_mode=PARSE_MODE,
            **kwargs,
        )
    except RetryAfter:
        raise
    except Exception:
        try:
            await target.edit_message_text(strip_sentinels(text), **kwargs)
        except RetryAfter:
            raise
        except Exception as e:
            logger.error("Failed to edit message: %s", e)


async def safe_send(
    bot: Bot,
    chat_id: int,
    text: str,
    message_thread_id: int | None = None,
    **kwargs: Any,
) -> None:
    """Send message with formatting, falling back to plain text on failure."""
    if _circuit_open(chat_id, message_thread_id):
        return
    kwargs.setdefault("link_preview_options", NO_LINK_PREVIEW)
    if message_thread_id is not None:
        kwargs.setdefault("message_thread_id", message_thread_id)
    try:
        await bot.send_message(
            chat_id=chat_id,
            text=_ensure_formatted(text),
            parse_mode=PARSE_MODE,
            **kwargs,
        )
        _record_success(chat_id, message_thread_id)
    except RetryAfter:
        raise
    except Exception:
        try:
            await bot.send_message(
                chat_id=chat_id, text=strip_sentinels(text), **kwargs
            )
            _record_success(chat_id, message_thread_id)
        except RetryAfter:
            raise
        except Exception as e:
            _record_failure(chat_id, message_thread_id)
            logger.error(
                "Failed to send message to chat_id=%d thread_id=%s: %s",
                chat_id,
                message_thread_id,
                e,
            )
