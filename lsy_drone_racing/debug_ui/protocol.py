"""Shared wire format for the debug UI (publisher <-> server).

Messages are single-frame JSON so ZMQ ``CONFLATE`` (keep-only-latest) works on the subscriber.
Payloads are tiny (a few hundred floats), so JSON keeps the publisher dependency-free on the
control-loop side -- no numpy serialization library needed.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

# Default ZMQ endpoint. Override via the DEBUG_UI_ADDR env var on both ends.
DEFAULT_ADDR = "tcp://127.0.0.1:5599"

# Observation keys forwarded to the dashboard. Mirrors the env observation space.
OBS_KEYS = (
    "pos",
    "quat",
    "vel",
    "ang_vel",
    "target_gate",
    "gates_pos",
    "gates_quat",
    "gates_visited",
    "obstacles_pos",
    "obstacles_visited",
)


def _to_jsonable(value: Any) -> Any:
    """Convert numpy scalars/arrays (and nested containers) into JSON-serializable Python."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def encode(t: int, obs: dict, action: Any, prev_action: Any) -> bytes:
    """Encode a control-step snapshot into a JSON frame."""
    payload = {
        "t": int(t),
        "obs": {k: _to_jsonable(obs[k]) for k in OBS_KEYS if k in obs},
        "action": _to_jsonable(action),
        "prev_action": _to_jsonable(prev_action),
    }
    return json.dumps(payload).encode("utf-8")


def decode(frame: bytes) -> dict:
    """Decode a JSON frame produced by :func:`encode`."""
    return json.loads(frame.decode("utf-8"))
