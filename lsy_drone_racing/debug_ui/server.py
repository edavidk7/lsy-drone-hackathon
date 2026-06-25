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
import asyncio
import io
import json
import logging
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from lsy_drone_racing.debug_ui.protocol import (
    DEFAULT_ADDR,
    DEFAULT_CMD_ADDR,
    decode,
    encode_cmd,
)
from lsy_drone_racing.debug_ui.forward_sim import ShadowSim
from lsy_drone_racing.utils import load_config

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
HISTORY_SECONDS = 5.0
PREDICT_HORIZON_STEPS = 50  # steps @ control frequency (e.g. 50 @ 50 Hz = 1.0 s)
DEFAULT_VIDEO_CAMERA = "fpv_cam:0"
DEFAULT_VIDEO_WIDTH = 640
DEFAULT_VIDEO_HEIGHT = 360
DEFAULT_VIDEO_RATE_HZ = 15.0
DEFAULT_VIDEO_QUALITY = 75
DEFAULT_VIDEO_RECYCLE_S = 12.0


async def _self_test_websocket(url: str) -> None:
    """Try a loopback websocket connection and log the result."""
    try:
        import websockets

        async with websockets.connect(url, open_timeout=2.0, close_timeout=1.0):
            logger.info("Self-test websocket connected to %s", url)
    except Exception:
        logger.warning("Self-test websocket failed for %s", url, exc_info=True)


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


class _SharedBytesFrame:
    """Thread-safe holder for latest binary frame (JPEG bytes + sequence number)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seq = 0
        self._bytes = b""

    def set(self, payload: bytes) -> None:
        with self._lock:
            self._seq += 1
            self._bytes = payload

    def get(self) -> tuple[int, bytes]:
        with self._lock:
            return self._seq, self._bytes


class _SharedObsFrame:
    """Thread-safe holder for latest observation packet for video rendering."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seq = 0
        self._t_step = -1
        self._obs: dict | None = None

    def set(self, t_step: int, obs: dict) -> None:
        with self._lock:
            self._seq += 1
            self._t_step = int(t_step)
            self._obs = obs

    def get(self) -> tuple[int, int, dict | None]:
        with self._lock:
            return self._seq, self._t_step, self._obs


class _CommandPublisher:
    """UI-side command publisher toward controller debug bridges."""

    def __init__(self, cmd_addr: str):
        import zmq

        self._cmd_addr = cmd_addr
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.SNDHWM, 1)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind(cmd_addr)
        self._again = zmq.Again
        logger.info("Debug UI command publisher bound on %s", cmd_addr)

    def send(self, command: dict) -> bool:
        try:
            import zmq

            self._sock.send(encode_cmd(command), flags=zmq.NOBLOCK)
            return True
        except self._again:
            return False
        except Exception:  # noqa: BLE001
            logger.warning("Failed to send debug UI command", exc_info=True)
            return False

    def close(self) -> None:
        try:
            self._sock.close(0)
        except Exception:  # noqa: BLE001
            pass


class Receiver(threading.Thread):
    """Background thread: ZMQ SUB -> ShadowSim rollout -> shared frame + rolling history."""

    def __init__(
        self,
        config,
        addr: str,
        shared: _SharedFrame,
        shared_obs: _SharedObsFrame | None = None,
        controller_file: str = "nav_rl_controller.py",
        max_rate_hz: float = 25.0,
    ):
        super().__init__(daemon=True)
        self._config = config
        self._addr = addr
        self._shared = shared
        self._controller_file = controller_file
        self._shared_obs = shared_obs
        self._min_period = 1.0 / max_rate_hz
        self._stop = threading.Event()
        self.freq = int(config.env.freq)
        maxlen = int(HISTORY_SECONDS * self.freq)
        self._hist_t: deque[float] = deque(maxlen=maxlen)
        self._hist_pos: deque[list] = deque(maxlen=maxlen)
        self._hist_vel: deque[list] = deque(maxlen=maxlen)
        self._hist_action: deque[list] = deque(maxlen=maxlen)
        self._hist_gate: deque[int] = deque(maxlen=maxlen)
        self._last_t_step: int | None = None

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

    def _clear_history(self) -> None:
        self._hist_t.clear()
        self._hist_pos.clear()
        self._hist_vel.clear()
        self._hist_action.clear()
        self._hist_gate.clear()

    def _process(self, shadow: ShadowSim, packet: dict) -> None:
        if self._shared_obs is not None:
            self._shared_obs.set(int(packet.get("t", 0)), packet["obs"])
        obs = {k: np.asarray(v) for k, v in packet["obs"].items()}
        prev_action = np.asarray(packet.get("prev_action", []), dtype=np.float32)
        action = np.asarray(packet.get("action", []), dtype=np.float32)
        t_step = int(packet.get("t", 0))
        if self._last_t_step is not None and t_step < self._last_t_step:
            logger.info("Mission restart detected (t %d -> %d). Clearing debug UI history.", self._last_t_step, t_step)
            self._clear_history()
        self._last_t_step = t_step
        t_sec = t_step / self.freq

        pred = shadow.predict(
            obs,
            prev_action,
            current_action=action,
            tick=t_step,
            n_steps=PREDICT_HORIZON_STEPS,
        )
        pred_xyz = np.asarray(pred.get("xyz", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        if pred_xyz.ndim != 2 or pred_xyz.shape[1] != 3:
            pred_xyz = np.asarray(obs["pos"], dtype=np.float32).reshape(1, 3)
        target_len = int(PREDICT_HORIZON_STEPS) + 1
        if pred_xyz.shape[0] < target_len:
            tail = np.repeat(pred_xyz[-1:].copy(), target_len - pred_xyz.shape[0], axis=0)
            pred_xyz = np.vstack([pred_xyz, tail])

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
                "xyz": pred_xyz.astype(float).tolist(),
                "actions": pred["actions"].astype(float).tolist(),
                "horizon_steps": int(PREDICT_HORIZON_STEPS),
                "horizon_s": float(PREDICT_HORIZON_STEPS / self.freq),
                "motor": pred.get("motor", {}),
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


def build_app(
    config,
    addr: str,
    controller_file: str = "nav_rl_controller.py",
    video_camera: str = DEFAULT_VIDEO_CAMERA,
    video_width: int = DEFAULT_VIDEO_WIDTH,
    video_height: int = DEFAULT_VIDEO_HEIGHT,
    video_rate_hz: float = DEFAULT_VIDEO_RATE_HZ,
    video_quality: int = DEFAULT_VIDEO_QUALITY,
    video_recycle_s: float = DEFAULT_VIDEO_RECYCLE_S,
    cmd_addr: str = DEFAULT_CMD_ADDR,
):
    """Build the FastAPI app and start the background receiver."""
    shared = _SharedFrame()
    shared_video = _SharedBytesFrame()
    shared_obs = _SharedObsFrame()
    cmd_pub = _CommandPublisher(cmd_addr)
    receiver = Receiver(
        config,
        addr,
        shared,
        shared_obs=shared_obs,
        controller_file=controller_file,
    )
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
        video_seq, _ = shared_video.get()
        return JSONResponse(
            {
                "ok": True,
                "latest_seq": seq,
                "latest_video_seq": video_seq,
                "addr": addr,
                "cmd_addr": cmd_addr,
            }
        )

    @app.post("/api/control/stop")
    def api_control_stop():
        sent = cmd_pub.send({"type": "stop", "enabled": True, "ts": time.time()})
        return JSONResponse({"ok": sent, "command": "stop"})

    @app.post("/api/control/resume")
    def api_control_resume():
        sent = cmd_pub.send({"type": "resume", "enabled": False, "ts": time.time()})
        return JSONResponse({"ok": sent, "command": "resume"})

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        import asyncio

        logger.info("Websocket handler entered from %s", getattr(websocket.client, "host", "unknown"))
        await websocket.accept()
        logger.info("Websocket connected from %s", getattr(websocket.client, "host", "unknown"))
        last_seq = -1
        try:
            while True:
                seq, payload = shared.get()
                if seq != last_seq:
                    last_seq = seq
                    await websocket.send_text(payload)
                await asyncio.sleep(0.04)  # ~25 Hz push cap.
        except WebSocketDisconnect:
            logger.info(
                "Websocket disconnected from %s", getattr(websocket.client, "host", "unknown")
            )
        except Exception:  # noqa: BLE001
            logger.debug("Websocket closed", exc_info=True)

    @app.websocket("/ws/video")
    async def ws_video(websocket: WebSocket):
        logger.info("Video websocket handler entered from %s", getattr(websocket.client, "host", "unknown"))
        await websocket.accept()
        logger.info("Video websocket connected from %s", getattr(websocket.client, "host", "unknown"))
        frame_interval = 1.0 / max(video_rate_hz, 1e-3)
        jpeg_quality = int(np.clip(video_quality, 10, 95))
        last_obs_seq = -1

        def _spawn_shadow() -> tuple[ShadowSim, list[str]]:
            shadow = ShadowSim(config, controller_file)
            available = set(shadow._available_cameras())  # noqa: SLF001
            cands = [video_camera, "fpv_cam:0", "track_cam:0"]
            cands = [c for i, c in enumerate(cands) if c and c not in cands[:i] and c in available]
            if not cands:
                cands = [video_camera]
            return shadow, cands

        video_shadow, candidates = _spawn_shadow()
        last_shadow_reset_t = time.monotonic()
        camera_idx = 0
        dark_frames = 0
        same_payload_count = 0
        prev_payload: bytes | None = None
        last_good_payload: bytes | None = None
        try:
            while True:
                if time.monotonic() - last_shadow_reset_t >= max(float(video_recycle_s), 1.0):
                    logger.info("Periodic video renderer recycle (%.1fs)", float(video_recycle_s))
                    video_shadow.close()
                    video_shadow, candidates = _spawn_shadow()
                    last_shadow_reset_t = time.monotonic()
                    camera_idx = 0
                    dark_frames = 0
                    same_payload_count = 0
                    prev_payload = None
                obs_seq, _, obs_packet = shared_obs.get()
                if obs_seq != last_obs_seq and obs_packet is not None:
                    last_obs_seq = obs_seq
                    active_camera = candidates[camera_idx]
                    frame_rgb = video_shadow.render_ego_frame(
                        obs_packet,
                        camera_name=active_camera,
                        width=video_width,
                        height=video_height,
                    )
                    if frame_rgb is not None:
                        is_dark = float(np.mean(frame_rgb)) < 3.0
                        dark_frames = dark_frames + 1 if is_dark else 0

                        if dark_frames >= 5 and camera_idx + 1 < len(candidates):
                            camera_idx += 1
                            dark_frames = 0
                            logger.warning(
                                "Video feed appears black on camera '%s'; switching to '%s'.",
                                active_camera,
                                candidates[camera_idx],
                            )
                            continue

                        image = Image.fromarray(frame_rgb)
                        buff = io.BytesIO()
                        image.save(buff, format="JPEG", quality=jpeg_quality, optimize=True)
                        payload = buff.getvalue()

                        if not is_dark:
                            last_good_payload = payload
                            same_payload_count = same_payload_count + 1 if payload == prev_payload else 0
                            prev_payload = payload
                            shared_video.set(payload)
                            await websocket.send_bytes(payload)
                        elif last_good_payload is not None:
                            shared_video.set(last_good_payload)
                            await websocket.send_bytes(last_good_payload)

                        if dark_frames >= 12 or same_payload_count >= 40:
                            reason = "dark" if dark_frames >= 12 else "stuck"
                            logger.warning("Resetting video renderer (%s frames) for %s camera=%s", dark_frames if reason == "dark" else same_payload_count, reason, active_camera)
                            video_shadow.close()
                            video_shadow, candidates = _spawn_shadow()
                            last_shadow_reset_t = time.monotonic()
                            camera_idx = 0
                            dark_frames = 0
                            same_payload_count = 0
                            prev_payload = None
                await asyncio.sleep(frame_interval)
        except WebSocketDisconnect:
            logger.info(
                "Video websocket disconnected from %s",
                getattr(websocket.client, "host", "unknown"),
            )
        except Exception:  # noqa: BLE001
            logger.debug("Video websocket closed", exc_info=True)
        finally:
            video_shadow.close()

    @app.on_event("shutdown")
    def _shutdown():
        receiver.stop()
        cmd_pub.close()

    @app.on_event("startup")
    def _startup():
        for route in app.routes:
            path = getattr(route, "path", None)
            name = getattr(route, "name", None)
            logger.info("Route registered: %s %s (%s)", path, name, type(route).__name__)

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
    parser.add_argument("--video-camera", default=DEFAULT_VIDEO_CAMERA)
    parser.add_argument("--video-width", type=int, default=DEFAULT_VIDEO_WIDTH)
    parser.add_argument("--video-height", type=int, default=DEFAULT_VIDEO_HEIGHT)
    parser.add_argument("--video-rate", type=float, default=DEFAULT_VIDEO_RATE_HZ)
    parser.add_argument("--video-quality", type=int, default=DEFAULT_VIDEO_QUALITY)
    parser.add_argument("--video-recycle-s", type=float, default=DEFAULT_VIDEO_RECYCLE_S)
    parser.add_argument("--cmd-addr", default=None, help=f"Command ZMQ addr (default {DEFAULT_CMD_ADDR}).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = load_config(Path(__file__).parents[2] / "config" / args.config)
    import os

    addr = args.addr or os.environ.get("DEBUG_UI_ADDR", DEFAULT_ADDR)
    cmd_addr = args.cmd_addr or os.environ.get("DEBUG_UI_CMD_ADDR", DEFAULT_CMD_ADDR)

    import uvicorn
    import importlib.util

    if importlib.util.find_spec("websockets") is None and importlib.util.find_spec("wsproto") is None:
        logger.warning(
            "No websocket backend detected. Install 'websockets' or 'wsproto' "
            "(for example: pip install \"uvicorn[standard]\" or pip install websockets)."
        )

    app = build_app(
        config,
        addr,
        args.controller,
        video_camera=args.video_camera,
        video_width=args.video_width,
        video_height=args.video_height,
        video_rate_hz=args.video_rate,
        video_quality=args.video_quality,
        video_recycle_s=args.video_recycle_s,
        cmd_addr=cmd_addr,
    )
    logger.info("Open http://%s:%d", args.host, args.port)
    async def _run_self_test():
        await asyncio.sleep(1.0)
        await _self_test_websocket(f"ws://{args.host}:{args.port}/ws")

    threading.Thread(target=lambda: asyncio.run(_run_self_test()), daemon=True).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
