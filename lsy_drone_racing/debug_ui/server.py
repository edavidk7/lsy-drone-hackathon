"""Debug UI server: subscribe to the controller's ZMQ stream and serve a live dashboard.

Run::

    python -m lsy_drone_racing.debug_ui.server --config level2.toml

Then start a publishing controller (in another terminal)::

    DEBUG_UI_ENABLE=1 python scripts/sim.py --config level2.toml --controller nav_rl_controller.py

and open http://localhost:8000.

Architecture: a background thread owns the ZMQ SUB socket (``CONFLATE`` -> always the latest obs),
the :class:`ShadowSim`, and the rolling history. For each received observation it runs the forward
rollout and publishes a JSON "frame" to a shared slot. The FastAPI websocket endpoint streams that
slot to browsers at a fixed rate. Keeping the (CPU-bound) rollout off the asyncio event loop keeps
the websocket responsive.
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from lsy_drone_racing.debug_ui.protocol import DEFAULT_ADDR, decode
from lsy_drone_racing.debug_ui.forward_sim import ShadowSim
from lsy_drone_racing.utils import load_config

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
HISTORY_SECONDS = 5.0
PREDICT_HORIZON_STEPS = 10  # 10 steps @ 50 Hz = 200 ms


class _SharedFrame:
    """Thread-safe holder for the latest dashboard frame (a JSON string + sequence number)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seq = 0
        self._json = "{}"

    def set(self, payload: dict) -> None:
        text = json.dumps(payload)
        with self._lock:
            self._seq += 1
            payload["seq"] = self._seq
            self._json = json.dumps(payload)

    def get(self) -> tuple[int, str]:
        with self._lock:
            return self._seq, self._json


class Receiver(threading.Thread):
    """Background thread: ZMQ SUB -> ShadowSim rollout -> shared frame + rolling history."""

    def __init__(
        self,
        config,
        addr: str,
        shared: _SharedFrame,
        controller_file: str = "nav_rl_controller.py",
        max_rate_hz: float = 25.0,
    ):
        super().__init__(daemon=True)
        self._config = config
        self._addr = addr
        self._shared = shared
        self._controller_file = controller_file
        self._min_period = 1.0 / max_rate_hz
        self._stop = threading.Event()
        self.freq = int(config.env.freq)
        maxlen = int(HISTORY_SECONDS * self.freq)
        self._hist_t: deque[float] = deque(maxlen=maxlen)
        self._hist_pos: deque[list] = deque(maxlen=maxlen)
        self._hist_vel: deque[list] = deque(maxlen=maxlen)
        self._hist_action: deque[list] = deque(maxlen=maxlen)
        self._hist_gate: deque[int] = deque(maxlen=maxlen)

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:  # noqa: C901 - linear receive loop, fine to keep together.
        import zmq

        shadow = ShadowSim(self._config, self._controller_file)
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.setsockopt(zmq.CONFLATE, 1)  # Keep only the most recent frame.
        sock.setsockopt(zmq.RCVHWM, 1)
        sock.connect(self._addr)
        logger.info("Debug UI server subscribed to %s", self._addr)

        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        last_process = 0.0

        while not self._stop.is_set():
            events = dict(poller.poll(timeout=200))
            if sock not in events:
                continue
            try:
                msg = sock.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                continue
            now = time.time()
            if now - last_process < self._min_period:
                continue  # Throttle: drop frames so the rollout keeps up with the 50 Hz stream.
            last_process = now
            try:
                self._process(shadow, decode(msg))
            except Exception:  # noqa: BLE001 - keep the loop alive on bad frames.
                logger.warning("Failed to process frame", exc_info=True)

        sock.close(0)
        shadow.close()

    def _process(self, shadow: ShadowSim, packet: dict) -> None:
        obs = {k: np.asarray(v) for k, v in packet["obs"].items()}
        prev_action = np.asarray(packet.get("prev_action", []), dtype=np.float32)
        action = np.asarray(packet.get("action", []), dtype=np.float32)
        t_step = int(packet.get("t", 0))
        t_sec = t_step / self.freq

        pred = shadow.predict(obs, prev_action, tick=t_step, n_steps=PREDICT_HORIZON_STEPS)

        self._hist_t.append(t_sec)
        self._hist_pos.append(np.asarray(obs["pos"], dtype=float).tolist())
        self._hist_vel.append(np.asarray(obs["vel"], dtype=float).tolist())
        self._hist_action.append(action.astype(float).tolist())
        self._hist_gate.append(int(np.asarray(obs["target_gate"])))

        vel = np.asarray(obs["vel"], dtype=float)
        frame = {
            "control_mode": shadow.control_mode,
            "freq": self.freq,
            "t": t_sec,
            "history": {
                "t": list(self._hist_t),
                "pos": list(self._hist_pos),
                "vel": list(self._hist_vel),
                "action": list(self._hist_action),
                "target_gate": list(self._hist_gate),
            },
            "prediction": {
                "xyz": pred["xyz"].astype(float).tolist(),
                "actions": pred["actions"].astype(float).tolist(),
            },
            "current": {
                "pos": np.asarray(obs["pos"], dtype=float).tolist(),
                "vel": vel.tolist(),
                "speed": float(np.linalg.norm(vel)),
                "action": action.astype(float).tolist(),
                "target_gate": int(np.asarray(obs["target_gate"])),
            },
            "gates": {
                "pos": np.asarray(obs["gates_pos"], dtype=float).tolist(),
                "quat": np.asarray(obs["gates_quat"], dtype=float).tolist(),
            },
            "obstacles": {"pos": np.asarray(obs["obstacles_pos"], dtype=float).tolist()},
        }
        self._shared.set(frame)


def build_app(config, addr: str, controller_file: str = "nav_rl_controller.py"):
    """Build the FastAPI app and start the background receiver."""
    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    shared = _SharedFrame()
    receiver = Receiver(config, addr, shared, controller_file)
    receiver.start()

    app = FastAPI(title="Drone Controller Debug UI")
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.middleware("http")
    async def log_http_requests(request: Request, call_next):
        response = await call_next(request)
        logger.info("HTTP %s %s -> %d", request.method, request.url.path, response.status_code)
        return response

    @app.get("/")
    def index():
        return FileResponse(str(STATIC_DIR / "index.html"))

    @app.get("/healthz")
    def healthz():
        seq, _ = shared.get()
        return JSONResponse({"ok": True, "latest_seq": seq, "addr": addr})

    @app.websocket("/ws")
    async def ws(socket: WebSocket):
        import asyncio

        await socket.accept()
        logger.info("Websocket connected from %s", getattr(socket.client, "host", "unknown"))
        last_seq = -1
        try:
            while True:
                seq, payload = shared.get()
                if seq != last_seq:
                    last_seq = seq
                    await socket.send_text(payload)
                await asyncio.sleep(0.04)  # ~25 Hz push cap.
        except WebSocketDisconnect:
            logger.info("Websocket disconnected from %s", getattr(socket.client, "host", "unknown"))
        except Exception:  # noqa: BLE001
            logger.debug("Websocket closed", exc_info=True)

    @app.on_event("shutdown")
    def _shutdown():
        receiver.stop()

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Drone controller debug dashboard server.")
    parser.add_argument("--config", default="level2.toml", help="Config file in config/.")
    parser.add_argument(
        "--controller",
        default="nav_rl_controller.py",
        help="Controller file in lsy_drone_racing/control/ used for the forward-sim rollout. "
        "Match this to the publishing controller (e.g. attitude_mpc.py).",
    )
    parser.add_argument("--addr", default=None, help=f"ZMQ address (default {DEFAULT_ADDR}).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = load_config(Path(__file__).parents[2] / "config" / args.config)
    import os

    addr = args.addr or os.environ.get("DEBUG_UI_ADDR", DEFAULT_ADDR)

    import uvicorn
    import importlib.util

    if importlib.util.find_spec("websockets") is None and importlib.util.find_spec("wsproto") is None:
        logger.warning(
            "No websocket backend detected. Install 'websockets' or 'wsproto' "
            "(for example: pip install \"uvicorn[standard]\" or pip install websockets)."
        )

    app = build_app(config, addr, args.controller)
    logger.info("Open http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
