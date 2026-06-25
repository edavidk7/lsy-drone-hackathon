# Controller Debug UI

A web dashboard to **diagnose the controller** live: see the observation, the controller's output,
rolling histories, and a **200 ms forward-simulated prediction** of where the drone is heading.

It connects to a running controller over ZMQ/TCP. The controller only does a non-blocking
fire-and-forget publish; all the heavy lifting (re-running the controller + a shadow simulation for
the prediction) happens in the dashboard server, so the 50 Hz control loop is never blocked.

## Toggle (deployment safety)

Everything is gated behind the `DEBUG_UI_ENABLE` environment variable.

- **Unset (default):** the controller never imports `zmq`; runs identically to before. Safe for the
  real drone even if `pyzmq`/`fastapi` are not installed.
- **Set (e.g. `DEBUG_UI_ENABLE=1`):** the controller binds a ZMQ PUB socket and publishes each step.
  If no dashboard is listening, frames are dropped silently (`zmq.NOBLOCK`) — still zero latency.

Optional `DEBUG_UI_ADDR` (default `tcp://127.0.0.1:5599`) sets the endpoint on both ends.

## Install

```bash
pip install -e ".[debug-ui]"      # adds pyzmq, fastapi, uvicorn
# or: pixi add pyzmq fastapi uvicorn
```

## Run (two terminals)

```bash
# Terminal A — dashboard server (builds its own shadow sim from the config)
python -m lsy_drone_racing.debug_ui.server --config level2.toml
# open http://localhost:8000

# Terminal B — the controller, with publishing enabled
DEBUG_UI_ENABLE=1 python scripts/sim.py --config level2.toml --controller nav_rl_controller.py
```

### Using a different controller

Publishing is wired into `nav_rl_controller.py` and `attitude_mpc.py`. Point the server's
forward-sim rollout at the **same** controller you run, so the 200 ms prediction matches:

```bash
# Terminal A
python -m lsy_drone_racing.debug_ui.server --config level2.toml --controller attitude_mpc.py
# Terminal B
DEBUG_UI_ENABLE=1 python scripts/sim.py --config level2.toml --controller attitude_mpc.py
```

Note: in `control_mode="attitude"` both controllers (and the env action space) use the layout
`[roll, pitch, yaw, thrust]`, which is what the dashboard's action labels assume.

The dashboard shows a 3D trajectory (history + the orange 200 ms prediction + gates/obstacles),
position and action time-series, and a status sidebar (target gate, speed, current action, update
rate).

## How the prediction works

The server keeps a private `DroneRaceEnv` and its own `NavRLController`. For each received
observation it injects the live drone kinematics **and** the observed gate/obstacle layout into the
env, then rolls the controller forward 10 env steps (50 Hz → 200 ms), recording the predicted path.

**Limitations (initial demo):** collision/contact checks in the rollout read MuJoCo state that is
not re-synced after injection, so predicted *termination* flags may be stale — only the integrated
positions/actions are used. The prediction assumes the gate layout stays as currently observed over
the short horizon.
