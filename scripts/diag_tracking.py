"""Diagnostic: is each crash a tracking failure (drone leaves a valid path) or a planner failure
(the planned path itself is bad)? Runs fixed-seed episodes and logs, per episode:

* max/mean tracking error ||planned_setpoint - actual_pos||
* planned-path min clearance to gates & poles at plan time (validity of the path itself)
* termination cause (terminated = collision / out-of-bounds, truncated = timeout)
"""

from __future__ import annotations

from pathlib import Path

import fire
import gymnasium
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.utils import load_config, load_controller


def _path_clearance(ctrl):
    """Min clearance (negative = violation) of the controller's planned path to poles & gates."""
    p = ctrl.planner
    ts = np.linspace(0.0, p.t_total, 1500)
    pts = p.spline(ts)
    pole_clr = np.inf
    for cx, cy in p._poles_xy:
        pole_clr = min(pole_clr, float((np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - p._pole_r).min()))
    gate_clr = np.inf
    for i in range(len(p._gates_pos)):
        local = p._gates_rot[i].apply(pts - p._gates_pos[i], inverse=True)
        gate_clr = min(gate_clr, float(p._frame_clearance(local).min()))
    return pole_clr, gate_clr


def main(config: str = "levelcompetition.toml", n_runs: int = 8, seed: int = 7):
    cfg = load_config(Path(__file__).parents[1] / "config" / config)
    cfg.sim.render = False
    cfg.env.seed = seed
    ctrl_path = Path(__file__).parents[1] / "lsy_drone_racing/control/min_snap_tracking_controller.py"
    controller_cls = load_controller(ctrl_path)

    env = gymnasium.make(
        cfg.env.id, freq=cfg.env.freq, sim_config=cfg.sim, sensor_range=cfg.env.sensor_range,
        control_mode=cfg.env.control_mode, track=cfg.env.track,
        disturbances=cfg.env.get("disturbances"), randomizations=cfg.env.get("randomizations"),
        seed=cfg.env.seed,
    )
    env = JaxToNumpy(env)

    for run in range(n_runs):
        obs, info = env.reset()
        ctrl = controller_cls(obs, info, cfg)
        pole_clr, gate_clr = _path_clearance(ctrl)
        errs, i, cause = [], 0, "controller_finished"
        while True:
            action = ctrl.compute_control(obs, info)
            t = min(i / cfg.env.freq, ctrl._t_total)
            errs.append(float(np.linalg.norm(ctrl._pos_spl(t) - obs["pos"])))
            obs, reward, terminated, truncated, info = env.step(action)
            fin = ctrl.step_callback(action, obs, reward, terminated, truncated, info)
            if terminated:
                cause = "TERMINATED (collision/bounds)"
            elif truncated:
                cause = "truncated (timeout)"
            if terminated or truncated or fin:
                break
            i += 1
        ctrl.episode_callback()
        gp = obs["target_gate"]
        gp = len(cfg.env.track.gates) if gp == -1 else gp
        print(
            f"run {run}: gates={gp}/{len(cfg.env.track.gates)} cause={cause:30s} "
            f"track_err mean={np.mean(errs):.3f} max={np.max(errs):.3f} | "
            f"planned pole_clr={pole_clr:+.3f} gate_clr={gate_clr:+.3f}"
        )
    env.close()


if __name__ == "__main__":
    fire.Fire(main)
