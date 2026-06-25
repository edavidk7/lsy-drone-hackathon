"""Non-blocking ZMQ publisher used by the live controller to feed the debug dashboard.

Design goals:
- **Zero impact when off.** ``get_publisher`` returns ``None`` unless ``DEBUG_UI_ENABLE`` is set,
  and ``zmq`` is imported lazily, so a deployment without ``pyzmq`` installed imports fine.
- **Never block the control loop.** Sends use ``zmq.NOBLOCK`` with ``SNDHWM=1``; if no subscriber
  is connected (or it is slow) the frame is dropped silently. Any error is swallowed.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from lsy_drone_racing.debug_ui.protocol import DEFAULT_ADDR, encode

logger = logging.getLogger(__name__)


def _enabled() -> bool:
    """Return True iff the debug UI is enabled via the DEBUG_UI_ENABLE env var."""
    return os.environ.get("DEBUG_UI_ENABLE", "").strip().lower() not in ("", "0", "false", "no")


class DebugPublisher:
    """Fire-and-forget ZMQ PUB wrapper. All operations are best-effort and never raise."""

    def __init__(self, addr: str | None = None):
        import zmq  # Lazy import: only reached when the debug UI is enabled.

        self._addr = addr or os.environ.get("DEBUG_UI_ADDR", DEFAULT_ADDR)
        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, 1)  # Keep at most one queued frame.
        self._socket.setsockopt(zmq.LINGER, 0)
        # The controller is the stable endpoint, so it binds; the dashboard connects.
        self._socket.bind(self._addr)
        self._again = zmq.Again
        logger.info("Debug UI publisher bound on %s", self._addr)

    def publish(self, t: int, obs: dict, action: Any, prev_action: Any) -> None:
        """Publish a control-step snapshot. Drops the frame if no subscriber is ready."""
        try:
            import zmq

            self._socket.send(encode(t, obs, action, prev_action), flags=zmq.NOBLOCK)
        except self._again:
            pass  # No subscriber ready / HWM reached -> drop, never block.
        except Exception:  # noqa: BLE001 - publishing must never crash the controller.
            logger.debug("Debug UI publish failed", exc_info=True)

    def close(self) -> None:
        """Close the socket (best-effort)."""
        try:
            self._socket.close(0)
        except Exception:  # noqa: BLE001
            pass


def get_publisher(addr: str | None = None) -> DebugPublisher | None:
    """Return a :class:`DebugPublisher` if the debug UI is enabled, else ``None``.

    Safe to call unconditionally from a controller constructor: when disabled, or when ``pyzmq`` is
    not installed, this returns ``None`` without raising.
    """
    if not _enabled():
        return None
    try:
        return DebugPublisher(addr)
    except Exception:  # noqa: BLE001 - e.g. pyzmq missing or address in use.
        logger.warning("Debug UI enabled but publisher could not start", exc_info=True)
        return None
