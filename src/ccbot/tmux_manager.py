"""Tmux session/window management via libtmux.

Wraps libtmux to provide async-friendly operations across ALL tmux sessions:
  - list_windows / find_window_by_name: discover Claude Code windows.
  - capture_pane: read terminal content (plain or with ANSI colors).
  - send_keys: forward user input or control keys to a window.
  - create_session / kill_window: lifecycle management.

Windows are identified by composite refs: "session_name:@window_id".

All blocking libtmux calls are wrapped in asyncio.to_thread().

Key class: TmuxManager (singleton instantiated as `tmux_manager`).
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import libtmux

from .config import SENSITIVE_ENV_VARS, config

logger = logging.getLogger(__name__)


def parse_window_ref(ref: str) -> tuple[str, str]:
    """Parse 'session_name:@window_id' into (session_name, window_id).

    Raises ValueError if format is invalid.
    """
    if not ref or ":" not in ref:
        raise ValueError(f"Invalid window ref '{ref}': expected 'session:@id'")
    session, _, window_id = ref.partition(":")
    if not session or not window_id:
        raise ValueError(f"Invalid window ref '{ref}': expected 'session:@id'")
    return session, window_id


def make_session_name(directory: str) -> str:
    """Generate tmux session name from directory, matching cmux convention.

    Convention: {child}-{parent}, lowercased, sanitized, max 60 chars.
    Example: /home/josh/workspace/agents/gtd -> gtd-agents
    """
    path = Path(directory).resolve()
    child = path.name or "root"
    parent = path.parent.name or "root"
    name = f"{child}-{parent}".lower()
    name = re.sub(r"[^a-z0-9_-]", "-", name)
    name = name.strip("-")
    return name[:60]


@dataclass
class TmuxWindow:
    """Information about a tmux window."""

    ref: str  # Composite ref: "session_name:@window_id"
    window_id: str
    session_name: str
    window_name: str
    cwd: str  # Current working directory
    pane_current_command: str = ""  # Process running in active pane


class TmuxManager:
    """Manages tmux windows for Claude Code sessions across all tmux sessions."""

    # TTL for cached list_windows results (seconds)
    _CACHE_TTL = 1.0

    def __init__(self) -> None:
        """Initialize tmux manager."""
        self._cached_windows: list[TmuxWindow] | None = None
        self._cache_time: float = 0.0

    def _find_pane_sync(self, ref: str) -> libtmux.Pane | None:
        """Resolve composite ref to a libtmux Pane (sync, for use in to_thread)."""
        session_name, window_id = parse_window_ref(ref)
        server = libtmux.Server()
        session = server.sessions.get(session_name=session_name)
        if not session:
            return None
        window = next((w for w in session.windows if w.window_id == window_id), None)
        if not window:
            return None
        return window.panes[0]

    def _find_window_sync(self, ref: str) -> libtmux.Window | None:
        """Resolve composite ref to a libtmux Window (sync, for use in to_thread)."""
        session_name, window_id = parse_window_ref(ref)
        server = libtmux.Server()
        session = server.sessions.get(session_name=session_name)
        if not session:
            return None
        return next((w for w in session.windows if w.window_id == window_id), None)

    @staticmethod
    def _scrub_session_env(session: libtmux.Session) -> None:
        """Remove sensitive env vars from the tmux session environment.

        Prevents new windows (and their child processes like Claude Code)
        from inheriting secrets such as TELEGRAM_BOT_TOKEN.
        """
        for var in SENSITIVE_ENV_VARS:
            try:
                session.unset_environment(var)
            except Exception:
                pass  # var not set in session env — nothing to remove

    async def list_windows(self) -> list[TmuxWindow]:
        """List all windows across ALL tmux sessions.

        Results are cached for _CACHE_TTL seconds to avoid redundant
        libtmux Server queries from concurrent callers (status poll,
        session monitor, message queue).

        Returns:
            List of TmuxWindow with window info and cwd
        """
        now = time.monotonic()
        if self._cached_windows is not None and (now - self._cache_time) < self._CACHE_TTL:
            return self._cached_windows

        def _sync_list_windows() -> list[TmuxWindow]:
            server = libtmux.Server()
            result: list[TmuxWindow] = []
            for session in server.sessions:
                for window in session.windows:
                    name = window.window_name or ""
                    try:
                        # Get the active pane's current path and command
                        pane = window.active_pane
                        if pane:
                            cwd = pane.pane_current_path or ""
                            pane_cmd = pane.pane_current_command or ""
                        else:
                            cwd = ""
                            pane_cmd = ""

                        wid = window.window_id or ""
                        sname = session.session_name or ""
                        ref = f"{sname}:{wid}"

                        result.append(
                            TmuxWindow(
                                ref=ref,
                                window_id=wid,
                                session_name=sname,
                                window_name=name,
                                cwd=cwd,
                                pane_current_command=pane_cmd,
                            )
                        )
                    except Exception as e:
                        logger.debug(f"Error getting window info: {e}")

            return result

        windows = await asyncio.to_thread(_sync_list_windows)
        self._cached_windows = windows
        self._cache_time = time.monotonic()
        return windows

    async def find_window_by_name(self, window_name: str) -> TmuxWindow | None:
        """Find a window by its name.

        Args:
            window_name: The window name to match

        Returns:
            TmuxWindow if found, None otherwise
        """
        windows = await self.list_windows()
        for window in windows:
            if window.window_name == window_name:
                return window
        logger.debug("Window not found by name: %s", window_name)
        return None

    async def find_window_by_id(self, window_id: str) -> TmuxWindow | None:
        """Find a window by composite ref or bare window ID.

        Accepts either a composite ref like 'session:@12' or a bare
        window ID like '@12' (searches all sessions for backward compat).

        Args:
            window_id: Composite ref or bare tmux window ID

        Returns:
            TmuxWindow if found, None otherwise
        """
        windows = await self.list_windows()
        # If it's a composite ref, match on ref
        if ":" in window_id:
            for window in windows:
                if window.ref == window_id:
                    return window
        else:
            # Bare window ID — search across all sessions
            for window in windows:
                if window.window_id == window_id:
                    return window
        logger.debug("Window not found by id: %s", window_id)
        return None

    async def capture_pane(self, window_id: str, with_ansi: bool = False) -> str | None:
        """Capture the visible text content of a window's active pane.

        Args:
            window_id: Composite ref (session:@id) or bare window ID
            with_ansi: If True, capture with ANSI color codes

        Returns:
            The captured text, or None on failure.
        """
        # Resolve the target for tmux CLI: if composite ref, use it as-is for -t
        # tmux CLI understands "session:@id" natively
        if with_ansi:
            # Use async subprocess to call tmux capture-pane -e for ANSI colors
            try:
                proc = await asyncio.create_subprocess_exec(
                    "tmux",
                    "capture-pane",
                    "-e",
                    "-p",
                    "-t",
                    window_id,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await proc.communicate()
                if proc.returncode == 0:
                    return stdout.decode("utf-8")
                logger.error(
                    f"Failed to capture pane {window_id}: {stderr.decode('utf-8')}"
                )
                return None
            except Exception as e:
                logger.error(f"Unexpected error capturing pane {window_id}: {e}")
                return None

        # Original implementation for plain text - wrap in thread
        def _sync_capture() -> str | None:
            pane = self._find_pane_sync(window_id) if ":" in window_id else None
            if not pane:
                # Fallback: bare window ID, search all sessions
                server = libtmux.Server()
                for session in server.sessions:
                    try:
                        window = session.windows.get(window_id=window_id)
                        if window:
                            pane = window.active_pane
                            break
                    except Exception:
                        continue
            if not pane:
                return None
            try:
                lines = pane.capture_pane()
                return "\n".join(lines) if isinstance(lines, list) else str(lines)
            except Exception as e:
                logger.error(f"Failed to capture pane {window_id}: {e}")
                return None

        return await asyncio.to_thread(_sync_capture)

    async def send_keys(
        self, window_id: str, text: str, enter: bool = True, literal: bool = True
    ) -> bool:
        """Send keys to a specific window.

        Args:
            window_id: Composite ref (session:@id) or bare window ID
            text: Text to send
            enter: Whether to press enter after the text
            literal: If True, send text literally. If False, interpret special keys
                     like "Up", "Down", "Left", "Right", "Escape", "Enter".

        Returns:
            True if successful, False otherwise
        """
        if literal and enter:
            # Split into text + delay + Enter via libtmux.
            # Claude Code's TUI sometimes interprets a rapid-fire Enter
            # (arriving in the same input batch as the text) as a newline
            # rather than submit.  A 500ms gap lets the TUI process the
            # text before receiving Enter.
            def _send_literal(chars: str) -> bool:
                pane = self._resolve_pane_sync(window_id)
                if not pane:
                    logger.error(f"Window {window_id} not found")
                    return False
                try:
                    pane.send_keys(chars, enter=False, literal=True)
                    return True
                except Exception as e:
                    logger.error(f"Failed to send keys to window {window_id}: {e}")
                    return False

            def _send_enter() -> bool:
                pane = self._resolve_pane_sync(window_id)
                if not pane:
                    return False
                try:
                    pane.send_keys("", enter=True, literal=False)
                    return True
                except Exception as e:
                    logger.error(f"Failed to send Enter to window {window_id}: {e}")
                    return False

            # Claude Code's ! command mode: send "!" first so the TUI
            # switches to bash mode, wait 1s, then send the rest.
            if text.startswith("!"):
                if not await asyncio.to_thread(_send_literal, "!"):
                    return False
                rest = text[1:]
                if rest:
                    await asyncio.sleep(1.0)
                    if not await asyncio.to_thread(_send_literal, rest):
                        return False
            else:
                if not await asyncio.to_thread(_send_literal, text):
                    return False
            await asyncio.sleep(0.5)
            return await asyncio.to_thread(_send_enter)

        # Other cases: special keys (literal=False) or no-enter
        def _sync_send_keys() -> bool:
            pane = self._resolve_pane_sync(window_id)
            if not pane:
                logger.error(f"Window {window_id} not found")
                return False
            try:
                pane.send_keys(text, enter=enter, literal=literal)
                return True
            except Exception as e:
                logger.error(f"Failed to send keys to window {window_id}: {e}")
                return False

        return await asyncio.to_thread(_sync_send_keys)

    def _resolve_pane_sync(self, window_id: str) -> libtmux.Pane | None:
        """Resolve window_id (composite ref or bare) to a libtmux Pane."""
        if ":" in window_id:
            return self._find_pane_sync(window_id)
        # Bare window ID — search all sessions
        server = libtmux.Server()
        for session in server.sessions:
            try:
                window = session.windows.get(window_id=window_id)
                if window:
                    return window.active_pane
            except Exception:
                continue
        return None

    async def rename_window(self, window_id: str, new_name: str) -> bool:
        """Rename a tmux window by composite ref or bare window ID."""

        def _sync_rename() -> bool:
            window = self._find_window_sync(window_id) if ":" in window_id else None
            if not window:
                # Fallback: bare window ID, search all sessions
                server = libtmux.Server()
                for session in server.sessions:
                    try:
                        window = session.windows.get(window_id=window_id)
                        if window:
                            break
                    except Exception:
                        continue
            if not window:
                return False
            try:
                window.rename_window(new_name)
                logger.info("Renamed window %s to '%s'", window_id, new_name)
                return True
            except Exception as e:
                logger.error(f"Failed to rename window {window_id}: {e}")
                return False

        return await asyncio.to_thread(_sync_rename)

    async def kill_window(self, window_id: str) -> bool:
        """Kill a tmux window by composite ref or bare window ID."""

        def _sync_kill() -> bool:
            window = self._find_window_sync(window_id) if ":" in window_id else None
            if not window:
                # Fallback: bare window ID, search all sessions
                server = libtmux.Server()
                for session in server.sessions:
                    try:
                        window = session.windows.get(window_id=window_id)
                        if window:
                            break
                    except Exception:
                        continue
            if not window:
                return False
            try:
                window.kill()
                logger.info("Killed window %s", window_id)
                return True
            except Exception as e:
                logger.error(f"Failed to kill window {window_id}: {e}")
                return False

        return await asyncio.to_thread(_sync_kill)

    async def create_session(
        self, work_dir: str, resume_session_id: str | None = None
    ) -> tuple[bool, str, str]:
        """Create a new tmux session with Claude Code, cmux-style.

        Args:
            work_dir: Working directory for the new session.
            resume_session_id: If set, append --resume <id> to claude command.

        Returns:
            Tuple of (success, message, composite_ref) where composite_ref
            is "session_name:@window_id".
        """
        # Validate directory first
        path = Path(work_dir).expanduser().resolve()
        if not path.exists():
            return False, f"Directory does not exist: {work_dir}", ""
        if not path.is_dir():
            return False, f"Not a directory: {work_dir}", ""

        session_name = make_session_name(str(path))

        def _create() -> tuple[bool, str, str]:
            server = libtmux.Server()

            # Check if session already exists
            existing = server.sessions.get(session_name=session_name)
            if existing:
                window = existing.windows[0]
                wid = window.window_id or ""
                ref = f"{session_name}:{wid}"
                return True, f"Session '{session_name}' already exists", ref

            # Build claude command
            cmd = config.claude_command
            if resume_session_id:
                cmd += f" --resume {resume_session_id}"

            try:
                session = server.new_session(
                    session_name=session_name,
                    start_directory=str(path),
                    window_command=cmd,
                )
                self._scrub_session_env(session)
                window = session.windows[0]
                wid = window.window_id or ""
                ref = f"{session_name}:{wid}"
                logger.info(
                    "Created session '%s' (ref=%s) at %s",
                    session_name,
                    ref,
                    path,
                )
                return True, f"Created session '{session_name}'", ref
            except Exception as e:
                logger.error(f"Failed to create session: {e}")
                return False, f"Failed to create session: {e}", ""

        return await asyncio.to_thread(_create)


# Global instance — no longer tied to a single session
tmux_manager = TmuxManager()
