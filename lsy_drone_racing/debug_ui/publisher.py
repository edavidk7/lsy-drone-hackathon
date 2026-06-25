"""Bidirectional debug bridge for controllers.

Features:
- publish telemetry (obs + action + prev_action) to UI
- poll UI commands (e.g., stop/resume) without blocking control loop
- keep optional behavior: disabled unless DEBUG_UI_ENABLE is set
"""

from __future__ import annotations

import logging
import os
from typing import Any

from lsy_drone_racing.debug_ui.protocol import (
    DEFAULT_ADDR,
    DEFAULT_CMD_ADDR,
    decode_cmd,
    encode,
)

logger = logging.getLogger(__name__)


def _enabled() -> bool:
    """Return True iff the debug UI is enabled via the DEBUG_UI_ENABLE env var."""
    return os.environ.get("DEBUG_UI_ENABLE", "").strip().lower() not in ("", "0", "false", "no")


class DebugBridge:
    """Non-blocking telemetry publisher + command subscriber."""

    def __init__(self, addr: str | None = None, cmd_addr: str | None = None):
        import zmq  # Lazy import: only reached when the debug UI is enabled.

        self._addr = addr or os.environ.get("DEBUG_UI_ADDR", DEFAULT_ADDR)
        self._cmd_addr = cmd_addr or os.environ.get("DEBUG_UI_CMD_ADDR", DEFAULT_CMD_ADDR)
        self._ctx = zmq.Context.instance()

        # Controller is stable endpoint for telemetry: bind PUB, server connects SUB.
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.setsockopt(zmq.SNDHWM, 1)
        self._pub.setsockopt(zmq.LINGER, 0)
        self._pub.bind(self._addr)

        # Server is stable endpoint for commands: bind PUB on server, controllers connect SUB.
        self._cmd_sub = self._ctx.socket(zmq.SUB)
        self._cmd_sub.setsockopt(zmq.SUBSCRIBE, b"")
        self._cmd_sub.setsockopt(zmq.CONFLATE, 1)
        self._cmd_sub.setsockopt(zmq.RCVHWM, 1)
        self._cmd_sub.setsockopt(zmq.LINGER, 0)
        self._cmd_sub.connect(self._cmd_addr)

        self._again = zmq.Again
        self._stop_latched = False
        logger.info("Debug bridge active telemetry=%s commands=%s", self._addr, self._cmd_addr)

    def publish(self, t: int, obs: dict, action: Any, prev_action: Any) -> None:
        """Publish control-step telemetry. Never blocks and never raises."""
        try:
            import zmq

            self._pub.send(encode(t, obs, action, prev_action), flags=zmq.NOBLOCK)
        except self._again:
            pass
        except Exception:  # noqa: BLE001
            logger.debug("Debug bridge publish failed", exc_info=True)

    def poll_request(self) -> dict | None:
        """Poll one latest command frame. Returns None if no command available."""
        try:
            import zmq

            msg = self._cmd_sub.recv(flags=zmq.NOBLOCK)
            cmd = decode_cmd(msg)
            ctype = str(cmd.get("type", "")).lower()
            enabled = bool(cmd.get("enabled", True))
            if ctype == "stop":
                self._stop_latched = enabled
            elif ctype in ("resume", "continue"):
                self._stop_latched = False
            return cmd
        except self._again:
            return None
        except Exception:  # noqa: BLE001
            logger.debug("Debug bridge poll failed", exc_info=True)
            return None

    def stop_requested(self) -> bool:
        """Return stop latch state after polling latest command."""
        self.poll_request()
        return self._stop_latched

    def close(self) -> None:
        """Close sockets (best-effort)."""
        for sock in (self._pub, self._cmd_sub):
            try:
                sock.close(0)
            except Exception:  # noqa: BLE001
                pass


# Backward compatibility alias.
DebugPublisher = DebugBridge


def get_bridge(addr: str | None = None, cmd_addr: str | None = None) -> DebugBridge | None:
    """Return a :class:`DebugBridge` if enabled, else None."""
    if not _enabled():
        return None
    try:
        return DebugBridge(addr=addr, cmd_addr=cmd_addr)
    except Exception:  # noqa: BLE001
        logger.warning("Debug UI enabled but bridge could not start", exc_info=True)
        return None


def get_publisher(addr: str | None = None) -> DebugBridge | None:
    """Compatibility helper for existing controllers."""
    return get_bridge(addr=addr)
