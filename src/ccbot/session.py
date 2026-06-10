"""Claude Code session management — the core state hub.

Manages the key mappings:
  Window→Session (window_states): which Claude session_id a window holds
    (keyed by composite ref like "session_name:@window_id").
  User→Thread→Window (thread_bindings): topic-to-window bindings
    (1 topic = 1 composite ref).

Responsibilities:
  - Persist/load state to ~/.ccbot/state.json.
  - Sync window↔session bindings from session_map.json (written by hook).
  - Resolve composite refs to ClaudeSession objects (JSONL file reading).
  - Track per-user read offsets for unread-message detection.
  - Manage thread↔window bindings for Telegram topic routing.
  - Send keystrokes to tmux windows and retrieve message history.
  - Maintain composite_ref→display name mapping for UI display.
  - Re-resolve stale refs on startup (tmux server restart recovery).
  - Migrate old-format keys (bare @id or window_name) to composite refs.

Key class: SessionManager (singleton instantiated as `session_manager`).
Key methods for thread binding access:
  - resolve_window_for_thread: Get composite ref for a user's thread
  - iter_thread_bindings: Generator for iterating all (user_id, thread_id, ref)
  - find_users_for_session: Find all users bound to a session_id
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Iterator
from typing import Any

import aiofiles

from .config import config
from .tmux_manager import tmux_manager
from .transcript_parser import TranscriptParser
from .utils import atomic_write_json

logger = logging.getLogger(__name__)


@dataclass
class WindowState:
    """Persistent state for a tmux window.

    Attributes:
        session_id: Associated Claude session ID (empty if not yet detected)
        cwd: Working directory for direct file path construction
        window_name: Display name of the window
    """

    session_id: str = ""
    cwd: str = ""
    window_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "session_id": self.session_id,
            "cwd": self.cwd,
        }
        if self.window_name:
            d["window_name"] = self.window_name
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WindowState":
        return cls(
            session_id=data.get("session_id", ""),
            cwd=data.get("cwd", ""),
            window_name=data.get("window_name", ""),
        )


@dataclass
class ClaudeSession:
    """Information about a Claude Code session."""

    session_id: str
    summary: str
    message_count: int
    file_path: str


@dataclass
class SessionManager:
    """Manages session state for Claude Code.

    All internal keys use composite refs (e.g. 'ccbot:@0', 'gtd-agents:@12')
    for uniqueness across multiple tmux sessions.
    Display names (window_name) are stored separately for UI presentation.

    window_states: composite_ref -> WindowState (session_id, cwd, window_name)
    user_window_offsets: user_id -> {composite_ref -> byte_offset}
    thread_bindings: user_id -> {thread_id -> composite_ref}
    window_display_names: composite_ref -> window_name (for display)
    group_chat_ids: "user_id:thread_id" -> group chat_id (for supergroup routing)
    """

    window_states: dict[str, WindowState] = field(default_factory=dict)
    user_window_offsets: dict[int, dict[str, int]] = field(default_factory=dict)
    thread_bindings: dict[int, dict[int, str]] = field(default_factory=dict)
    # window_id -> display name (window_name)
    window_display_names: dict[str, str] = field(default_factory=dict)
    # "user_id:thread_id" -> group chat_id (for supergroup forum topic routing)
    # IMPORTANT: This mapping is essential for supergroup/forum topic support.
    # Telegram Bot API requires group chat_id (negative number like -100xxx)
    # as the chat_id parameter when sending messages to forum topics.
    # Using user_id as chat_id will fail with "Message thread not found".
    # See: https://core.telegram.org/bots/api#sendmessage
    # History: originally added in 5afc111, erroneously removed in 26cb81f,
    # restored in PR #23.
    group_chat_ids: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._load_state()

    def _save_state(self) -> None:
        state: dict[str, Any] = {
            "window_states": {k: v.to_dict() for k, v in self.window_states.items()},
            "user_window_offsets": {
                str(uid): offsets for uid, offsets in self.user_window_offsets.items()
            },
            "thread_bindings": {
                str(uid): {str(tid): wid for tid, wid in bindings.items()}
                for uid, bindings in self.thread_bindings.items()
            },
            "window_display_names": self.window_display_names,
            "group_chat_ids": self.group_chat_ids,
        }
        atomic_write_json(config.state_file, state)
        logger.debug("State saved to %s", config.state_file)

    def _is_window_id(self, key: str) -> bool:
        """Check if a key looks like a bare tmux window ID (e.g. '@0', '@12')."""
        return key.startswith("@") and len(key) > 1 and key[1:].isdigit()

    def _is_composite_ref(self, key: str) -> bool:
        """Check if a key looks like a composite ref (e.g. 'ccbot:@12', 'gtd-agents:@0')."""
        if ":" not in key:
            return False
        session_name, _, window_id = key.partition(":")
        return bool(session_name) and self._is_window_id(window_id)

    def _load_state(self) -> None:
        """Load state synchronously during initialization.

        Detects old-format state (window_name keys without '@' prefix) and
        marks for migration on next startup re-resolution.
        """
        if config.state_file.exists():
            try:
                state = json.loads(config.state_file.read_text())
                self.window_states = {
                    k: WindowState.from_dict(v)
                    for k, v in state.get("window_states", {}).items()
                }
                self.user_window_offsets = {
                    int(uid): offsets
                    for uid, offsets in state.get("user_window_offsets", {}).items()
                }
                self.thread_bindings = {
                    int(uid): {int(tid): wid for tid, wid in bindings.items()}
                    for uid, bindings in state.get("thread_bindings", {}).items()
                }
                self.window_display_names = state.get("window_display_names", {})
                self.group_chat_ids = {
                    k: int(v) for k, v in state.get("group_chat_ids", {}).items()
                }

                # Detect old format: keys that are neither composite refs
                # nor bare window IDs (e.g. window_name strings), or bare
                # window IDs that need upgrading to composite refs.
                needs_migration = False
                for k in self.window_states:
                    if not self._is_composite_ref(k):
                        needs_migration = True
                        break
                if not needs_migration:
                    for bindings in self.thread_bindings.values():
                        for wid in bindings.values():
                            if not self._is_composite_ref(wid):
                                needs_migration = True
                                break
                        if needs_migration:
                            break

                if needs_migration:
                    logger.info(
                        "Detected old-format state (non-composite keys), "
                        "will re-resolve on startup"
                    )
                    pass

            except (json.JSONDecodeError, ValueError) as e:
                logger.warning("Failed to load state: %s", e)
                self.window_states = {}
                self.user_window_offsets = {}
                self.thread_bindings = {}
                self.window_display_names = {}
                self.group_chat_ids = {}
                pass

    async def resolve_stale_ids(self) -> None:
        """Re-resolve persisted refs against live tmux windows.

        Called on startup. Handles three cases:
        1. Composite refs (current format): 'session:@id' — validate against live windows
        2. Bare window IDs (old format): '@id' — upgrade to composite refs
        3. Window name keys (oldest format): 'name' — resolve to composite refs

        Builds lookup maps from live windows, then remaps or drops entries.
        """
        windows = await tmux_manager.list_windows()
        live_refs: set[str] = set()  # set of composite refs
        # Map (session_name, window_name) -> composite ref for re-resolution
        live_by_session_and_name: dict[tuple[str, str], str] = {}
        # Map bare window_id -> composite ref (for bare ID migration)
        live_bare_to_ref: dict[str, str] = {}
        # Map window_name -> composite ref (for oldest format migration)
        live_by_name: dict[str, str] = {}

        for w in windows:
            live_refs.add(w.ref)
            live_by_session_and_name[(w.session_name, w.window_name)] = w.ref
            live_bare_to_ref[w.window_id] = w.ref
            live_by_name[w.window_name] = w.ref

        changed = False
        # Track old_key -> new_ref remapping so thread_bindings and offsets
        # can re-resolve even after display names are updated in window_states phase.
        ref_remap: dict[str, str] = {}

        # --- Migrate window_states ---
        new_window_states: dict[str, WindowState] = {}
        for key, ws in self.window_states.items():
            if self._is_composite_ref(key):
                if key in live_refs:
                    new_window_states[key] = ws
                else:
                    # Stale composite ref — try re-resolve by session + display name
                    session_name, _, bare_id = key.partition(":")
                    display = self.window_display_names.get(
                        key, ws.window_name or bare_id
                    )
                    new_ref = live_by_session_and_name.get((session_name, display))
                    if new_ref:
                        logger.info(
                            "Re-resolved stale ref %s -> %s (name=%s)",
                            key,
                            new_ref,
                            display,
                        )
                        new_window_states[new_ref] = ws
                        ws.window_name = display
                        self.window_display_names[new_ref] = display
                        self.window_display_names.pop(key, None)
                        ref_remap[key] = new_ref
                        changed = True
                    else:
                        logger.info(
                            "Dropping stale window_state: %s (name=%s)", key, display
                        )
                        changed = True
            elif self._is_window_id(key):
                # Old format: bare window ID — upgrade to composite ref
                new_ref = live_bare_to_ref.get(key)
                if new_ref:
                    logger.info("Migrating bare window_id %s -> %s", key, new_ref)
                    new_window_states[new_ref] = ws
                    if ws.window_name:
                        self.window_display_names[new_ref] = ws.window_name
                    # Migrate display name from old bare key
                    old_display = self.window_display_names.pop(key, None)
                    if old_display:
                        self.window_display_names[new_ref] = old_display
                    ref_remap[key] = new_ref
                    changed = True
                else:
                    logger.info(
                        "Dropping stale bare window_id: %s (no live window)", key
                    )
                    changed = True
            else:
                # Oldest format: key is window_name
                new_ref = live_by_name.get(key)
                if new_ref:
                    logger.info("Migrating window_name key %s -> %s", key, new_ref)
                    ws.window_name = key
                    new_window_states[new_ref] = ws
                    self.window_display_names[new_ref] = key
                    ref_remap[key] = new_ref
                    changed = True
                else:
                    logger.info(
                        "Dropping old-format window_state: %s (no live window)", key
                    )
                    changed = True
        self.window_states = new_window_states

        # Helper to remap a ref/id/name using remap table or live lookups
        def _remap(val: str) -> str | None:
            """Remap a stale or old-format ref to a live composite ref."""
            # Already remapped during window_states phase
            if val in ref_remap:
                return ref_remap[val]
            # Already live
            if val in live_refs:
                return val
            # Stale composite ref — try session + display name
            if self._is_composite_ref(val):
                session_name, _, bare_id = val.partition(":")
                display = self.window_display_names.get(val, bare_id)
                return live_by_session_and_name.get((session_name, display))
            # Bare window ID
            if self._is_window_id(val):
                return live_bare_to_ref.get(val)
            # Window name
            return live_by_name.get(val)

        # --- Migrate thread_bindings ---
        for uid, bindings in self.thread_bindings.items():
            new_bindings: dict[int, str] = {}
            for tid, val in bindings.items():
                new_ref = _remap(val)
                if new_ref:
                    if new_ref != val:
                        logger.info(
                            "Re-resolved thread binding %s -> %s", val, new_ref
                        )
                        changed = True
                    new_bindings[tid] = new_ref
                else:
                    logger.info(
                        "Dropping stale thread binding: "
                        "user=%d, thread=%d, ref=%s",
                        uid,
                        tid,
                        val,
                    )
                    changed = True
            self.thread_bindings[uid] = new_bindings

        # Remove empty user entries
        empty_users = [uid for uid, b in self.thread_bindings.items() if not b]
        for uid in empty_users:
            del self.thread_bindings[uid]

        # --- Migrate user_window_offsets ---
        for uid, offsets in self.user_window_offsets.items():
            new_offsets: dict[str, int] = {}
            for key, offset in offsets.items():
                new_ref = _remap(key)
                if new_ref:
                    if new_ref != key:
                        changed = True
                    new_offsets[new_ref] = offset
                else:
                    changed = True
            self.user_window_offsets[uid] = new_offsets

        if changed:
            self._save_state()
            logger.info("Startup re-resolution complete")

        # Clean up session_map.json: stale entries and old-format keys
        await self._cleanup_stale_session_map_entries(live_refs)
        await self._cleanup_old_format_session_map_keys()

    async def _cleanup_old_format_session_map_keys(self) -> None:
        """Remove old-format keys (not composite refs) from session_map.json."""
        if not config.session_map_file.exists():
            return
        try:
            async with aiofiles.open(config.session_map_file, "r") as f:
                content = await f.read()
            session_map = json.loads(content)
        except (json.JSONDecodeError, OSError):
            return

        old_keys = [
            key
            for key in session_map
            if not self._is_composite_ref(key)
        ]
        if not old_keys:
            return

        for key in old_keys:
            del session_map[key]
        atomic_write_json(config.session_map_file, session_map)
        logger.info(
            "Cleaned up %d old-format session_map keys: %s", len(old_keys), old_keys
        )

    async def _cleanup_stale_session_map_entries(self, live_refs: set[str]) -> None:
        """Remove entries for tmux windows that no longer exist.

        When windows are closed externally (outside ccbot), session_map.json
        retains orphan references. This cleanup removes entries whose composite
        ref is not in the current set of live tmux windows.
        """
        if not config.session_map_file.exists():
            return
        try:
            async with aiofiles.open(config.session_map_file, "r") as f:
                content = await f.read()
            session_map = json.loads(content)
        except (json.JSONDecodeError, OSError):
            return

        stale_keys = [
            key
            for key in session_map
            if self._is_composite_ref(key) and key not in live_refs
        ]
        if not stale_keys:
            return

        for key in stale_keys:
            del session_map[key]
            logger.info("Removed stale session_map entry: %s", key)

        atomic_write_json(config.session_map_file, session_map)
        logger.info(
            "Cleaned up %d stale session_map entries (windows no longer in tmux)",
            len(stale_keys),
        )

    # --- Display name management ---

    def get_display_name(self, window_id: str) -> str:
        """Get display name for a window_id, fallback to window_id itself."""
        return self.window_display_names.get(window_id, window_id)

    def update_display_name(self, window_id: str, new_name: str) -> None:
        """Update the display name for a window and persist state."""
        self.window_display_names[window_id] = new_name
        # Also update WindowState.window_name if it exists
        if window_id in self.window_states:
            self.window_states[window_id].window_name = new_name
        self._save_state()
        logger.info("Updated display name: window_id %s -> '%s'", window_id, new_name)

    # --- Group chat ID management (supergroup forum topic routing) ---

    def set_group_chat_id(
        self, user_id: int, thread_id: int | None, chat_id: int
    ) -> None:
        """Store the group chat_id for a user+thread combination.

        In supergroups with forum topics, messages must be sent to the group's
        chat_id (negative number like -100xxx) rather than the user's personal ID.
        Telegram's Bot API rejects message_thread_id when chat_id is a private
        user ID — the thread only exists within the group context.

        DO NOT REMOVE this method or the group_chat_ids mapping.
        Without it, all outbound messages in forum topics fail with
        "Message thread not found". See commit history: 5afc111 → 26cb81f → PR #23.
        """
        tid = thread_id or 0
        key = f"{user_id}:{tid}"
        if self.group_chat_ids.get(key) != chat_id:
            self.group_chat_ids[key] = chat_id
            self._save_state()
            logger.debug(
                "Stored group chat_id: user=%d, thread=%s, chat_id=%d",
                user_id,
                thread_id,
                chat_id,
            )

    def resolve_chat_id(self, user_id: int, thread_id: int | None = None) -> int:
        """Resolve the correct chat_id for sending messages.

        Returns the stored group chat_id when a thread_id is present and a
        mapping exists, otherwise falls back to user_id (for private chats).

        Every outbound Telegram API call (send_message, edit_message_text,
        delete_message, send_chat_action, edit_forum_topic, etc.) MUST use
        this method instead of raw user_id. Using user_id directly breaks
        supergroup forum topic routing.
        """
        if thread_id is not None:
            key = f"{user_id}:{thread_id}"
            group_id = self.group_chat_ids.get(key)
            if group_id is not None:
                return group_id
        return user_id

    async def wait_for_session_map_entry(
        self, window_ref: str, timeout: float = 5.0, interval: float = 0.5
    ) -> bool:
        """Poll session_map.json until an entry for window_ref appears.

        Args:
            window_ref: Composite ref (e.g. 'gtd-agents:@12') or bare window ID.
                        Bare IDs (e.g. '@12') are matched against any session
                        by scanning all keys.

        Returns True if the entry was found within timeout, False otherwise.
        """
        is_composite = self._is_composite_ref(window_ref)
        logger.debug(
            "Waiting for session_map entry: ref=%s, timeout=%.1f",
            window_ref,
            timeout,
        )
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            try:
                if config.session_map_file.exists():
                    async with aiofiles.open(config.session_map_file, "r") as f:
                        content = await f.read()
                    session_map = json.loads(content)
                    if is_composite:
                        info = session_map.get(window_ref, {})
                        found = bool(info.get("session_id"))
                    else:
                        # Bare window ID — scan all keys for a matching suffix
                        found = any(
                            k.endswith(f":{window_ref}")
                            and v.get("session_id")
                            for k, v in session_map.items()
                        )
                    if found:
                        logger.debug(
                            "session_map entry found for ref %s", window_ref
                        )
                        await self.load_session_map()
                        return True
            except (json.JSONDecodeError, OSError):
                pass
            await asyncio.sleep(interval)
        logger.warning(
            "Timed out waiting for session_map entry: ref=%s", window_ref
        )
        return False

    async def load_session_map(self) -> None:
        """Read session_map.json and update window_states with new session associations.

        Keys in session_map are composite refs: "session_name:@window_id"
        (e.g. "ccbot:@12", "gtd-agents:@5"). All valid entries are processed
        regardless of which tmux session they belong to.
        Also cleans up window_states entries not in current session_map.
        Updates window_display_names from the "window_name" field in values.
        """
        if not config.session_map_file.exists():
            return
        try:
            async with aiofiles.open(config.session_map_file, "r") as f:
                content = await f.read()
            session_map = json.loads(content)
        except (json.JSONDecodeError, OSError):
            return

        valid_refs: set[str] = set()
        changed = False

        for key, info in session_map.items():
            # Only process entries that are valid composite refs
            if not self._is_composite_ref(key):
                continue
            valid_refs.add(key)
            new_sid = info.get("session_id", "")
            new_cwd = info.get("cwd", "")
            new_wname = info.get("window_name", "")
            if not new_sid:
                continue
            state = self.get_window_state(key)
            if state.session_id != new_sid or state.cwd != new_cwd:
                logger.info(
                    "Session map: %s updated sid=%s, cwd=%s",
                    key,
                    new_sid,
                    new_cwd,
                )
                state.session_id = new_sid
                state.cwd = new_cwd
                changed = True
            # Update display name
            if new_wname:
                state.window_name = new_wname
                if self.window_display_names.get(key) != new_wname:
                    self.window_display_names[key] = new_wname
                    changed = True

        # Clean up window_states entries not in current session_map.
        stale_refs = [r for r in self.window_states if r and r not in valid_refs]
        for ref in stale_refs:
            logger.info("Removing stale window_state: %s", ref)
            del self.window_states[ref]
            changed = True

        if changed:
            self._save_state()

    # --- Window state management ---

    def get_window_state(self, window_id: str) -> WindowState:
        """Get or create window state."""
        if window_id not in self.window_states:
            self.window_states[window_id] = WindowState()
        return self.window_states[window_id]

    def clear_window_session(self, window_id: str) -> None:
        """Clear session association for a window (e.g., after /clear command)."""
        state = self.get_window_state(window_id)
        state.session_id = ""
        self._save_state()
        logger.info("Cleared session for window_id %s", window_id)

    @staticmethod
    def _encode_cwd(cwd: str) -> str:
        """Encode a cwd path to match Claude Code's project directory naming.

        Replaces all non-alphanumeric characters (except dash) with dashes.
        E.g. /home/user_name/Code/project -> -home-user-name-Code-project
        """
        return re.sub(r"[^a-zA-Z0-9-]", "-", cwd)

    def _build_session_file_path(self, session_id: str, cwd: str) -> Path | None:
        """Build the direct file path for a session from session_id and cwd."""
        if not session_id or not cwd:
            return None
        encoded_cwd = self._encode_cwd(cwd)
        return config.claude_projects_path / encoded_cwd / f"{session_id}.jsonl"

    def file_path_for_window(self, window_id: str) -> Path | None:
        """Fast path: return the session JSONL path for a window with zero I/O.

        Uses only in-memory window_state; does not read the file or validate
        existence. Callers that need to stat/read should handle OSError.
        Returns None if the window has no associated session.
        """
        state = self.get_window_state(window_id)
        if not state.session_id or not state.cwd:
            return None
        return self._build_session_file_path(state.session_id, state.cwd)

    async def _get_session_direct(
        self, session_id: str, cwd: str
    ) -> ClaudeSession | None:
        """Get a ClaudeSession directly from session_id and cwd (no scanning)."""
        file_path = self._build_session_file_path(session_id, cwd)

        # Fallback: glob search if direct path doesn't exist
        if not file_path or not file_path.exists():
            pattern = f"*/{session_id}.jsonl"
            matches = list(config.claude_projects_path.glob(pattern))
            if matches:
                file_path = matches[0]
                logger.debug("Found session via glob: %s", file_path)
            else:
                return None

        # Single pass: read file once, extract summary + count messages
        summary = ""
        last_user_msg = ""
        message_count = 0
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                async for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    message_count += 1
                    try:
                        data = json.loads(line)
                        # Check for summary
                        if data.get("type") == "summary":
                            s = data.get("summary", "")
                            if s:
                                summary = s
                        # Track last user message as fallback
                        elif TranscriptParser.is_user_message(data):
                            parsed = TranscriptParser.parse_message(data)
                            if parsed and parsed.text.strip():
                                last_user_msg = parsed.text.strip()
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return None

        if not summary:
            summary = last_user_msg[:50] if last_user_msg else "Untitled"

        return ClaudeSession(
            session_id=session_id,
            summary=summary,
            message_count=message_count,
            file_path=str(file_path),
        )

    # --- Directory session listing ---

    async def list_sessions_for_directory(self, cwd: str) -> list[ClaudeSession]:
        """List existing Claude sessions for a directory.

        Encodes the cwd path to find the project directory under
        ~/.claude/projects/{encoded_cwd}/, globs *.jsonl files, and
        extracts summary info from each.

        Returns a list sorted by mtime (most recent first), capped at 10.
        """
        encoded_cwd = self._encode_cwd(cwd)
        project_dir = config.claude_projects_path / encoded_cwd
        if not project_dir.is_dir():
            return []

        # Collect JSONL files sorted by mtime (newest first)
        jsonl_files = sorted(
            project_dir.glob("*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        # Skip sessions-index and cap at 10
        sessions: list[ClaudeSession] = []
        for f in jsonl_files:
            if f.stem == "sessions-index":
                continue
            if len(sessions) >= 10:
                break
            session_id = f.stem
            session = await self._get_session_direct(session_id, cwd)
            if session and session.message_count > 0:
                sessions.append(session)
        return sessions

    # --- Window → Session resolution ---

    async def resolve_session_for_window(self, window_id: str) -> ClaudeSession | None:
        """Resolve a tmux window to the best matching Claude session.

        Uses persisted session_id + cwd to construct file path directly.
        Returns None if no session is associated with this window.
        """
        state = self.get_window_state(window_id)

        if not state.session_id or not state.cwd:
            return None

        session = await self._get_session_direct(state.session_id, state.cwd)
        if session:
            return session

        # File no longer exists, clear state
        logger.warning(
            "Session file no longer exists for window_id %s (sid=%s, cwd=%s)",
            window_id,
            state.session_id,
            state.cwd,
        )
        state.session_id = ""
        state.cwd = ""
        self._save_state()
        return None

    # --- User window offset management ---

    def update_user_window_offset(
        self, user_id: int, window_id: str, offset: int
    ) -> None:
        """Update the user's last read offset for a window."""
        if user_id not in self.user_window_offsets:
            self.user_window_offsets[user_id] = {}
        self.user_window_offsets[user_id][window_id] = offset
        self._save_state()

    # --- Thread binding management ---

    def bind_thread(
        self, user_id: int, thread_id: int, window_id: str, window_name: str = ""
    ) -> None:
        """Bind a Telegram topic thread to a tmux window.

        Args:
            user_id: Telegram user ID
            thread_id: Telegram topic thread ID
            window_id: Tmux window ID (e.g. '@0')
            window_name: Display name for the window (optional)
        """
        if user_id not in self.thread_bindings:
            self.thread_bindings[user_id] = {}
        self.thread_bindings[user_id][thread_id] = window_id
        if window_name:
            self.window_display_names[window_id] = window_name
        self._save_state()
        display = window_name or self.get_display_name(window_id)
        logger.info(
            "Bound thread %d -> window_id %s (%s) for user %d",
            thread_id,
            window_id,
            display,
            user_id,
        )

    def unbind_thread(self, user_id: int, thread_id: int) -> str | None:
        """Remove a thread binding. Returns the previously bound window_id, or None."""
        bindings = self.thread_bindings.get(user_id)
        if not bindings or thread_id not in bindings:
            return None
        window_id = bindings.pop(thread_id)
        if not bindings:
            del self.thread_bindings[user_id]
        self._save_state()
        logger.info(
            "Unbound thread %d (was %s) for user %d",
            thread_id,
            window_id,
            user_id,
        )
        return window_id

    def get_window_for_thread(self, user_id: int, thread_id: int) -> str | None:
        """Look up the window_id bound to a thread."""
        bindings = self.thread_bindings.get(user_id)
        if not bindings:
            return None
        return bindings.get(thread_id)

    def resolve_window_for_thread(
        self,
        user_id: int,
        thread_id: int | None,
    ) -> str | None:
        """Resolve the tmux window_id for a user's thread.

        Returns None if thread_id is None or the thread is not bound.
        """
        if thread_id is None:
            return None
        return self.get_window_for_thread(user_id, thread_id)

    def iter_thread_bindings(self) -> Iterator[tuple[int, int, str]]:
        """Iterate all thread bindings as (user_id, thread_id, window_id).

        Provides encapsulated access to thread_bindings without exposing
        the internal data structure directly.
        """
        for user_id, bindings in self.thread_bindings.items():
            for thread_id, window_id in bindings.items():
                yield user_id, thread_id, window_id

    async def find_users_for_session(
        self,
        session_id: str,
    ) -> list[tuple[int, str, int]]:
        """Find all users whose thread-bound window maps to the given session_id.

        Hot path: called for every message from every active Claude session.
        Uses in-memory window_state lookups only — no JSONL reads, no file I/O.
        Stale bindings (where state.session_id doesn't match any live session)
        simply don't match and are skipped; cleanup happens elsewhere.

        Returns list of (user_id, window_id, thread_id) tuples.
        """
        result: list[tuple[int, str, int]] = []
        for user_id, thread_id, window_id in self.iter_thread_bindings():
            state = self.get_window_state(window_id)
            if state.session_id == session_id:
                result.append((user_id, window_id, thread_id))
        return result

    # --- Tmux helpers ---

    async def send_to_window(self, window_id: str, text: str) -> tuple[bool, str]:
        """Send text to a tmux window by ID."""
        display = self.get_display_name(window_id)
        logger.debug(
            "send_to_window: window_id=%s (%s), text_len=%d",
            window_id,
            display,
            len(text),
        )
        window = await tmux_manager.find_window_by_id(window_id)
        if not window:
            return False, "Window not found (may have been closed)"
        success = await tmux_manager.send_keys(window.ref, text)
        if success:
            return True, f"Sent to {display}"
        return False, "Failed to send keys"

    # --- Message history ---

    async def get_recent_messages(
        self,
        window_id: str,
        *,
        start_byte: int = 0,
        end_byte: int | None = None,
    ) -> tuple[list[dict], int]:
        """Get user/assistant messages for a window's session.

        Resolves window → session, then reads the JSONL.
        Supports byte range filtering via start_byte/end_byte.
        Returns (messages, total_count).
        """
        session = await self.resolve_session_for_window(window_id)
        if not session or not session.file_path:
            return [], 0

        file_path = Path(session.file_path)
        if not file_path.exists():
            return [], 0

        # Read JSONL entries (optionally filtered by byte range)
        entries: list[dict] = []
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                if start_byte > 0:
                    await f.seek(start_byte)

                while True:
                    # Check byte limit before reading
                    if end_byte is not None:
                        current_pos = await f.tell()
                        if current_pos >= end_byte:
                            break

                    line = await f.readline()
                    if not line:
                        break

                    data = TranscriptParser.parse_line(line)
                    if data:
                        entries.append(data)
        except OSError as e:
            logger.error("Error reading session file %s: %s", file_path, e)
            return [], 0

        parsed_entries, _ = TranscriptParser.parse_entries(entries)
        all_messages = [
            {
                "role": e.role,
                "text": e.text,
                "content_type": e.content_type,
                "timestamp": e.timestamp,
            }
            for e in parsed_entries
        ]

        return all_messages, len(all_messages)


session_manager = SessionManager()
