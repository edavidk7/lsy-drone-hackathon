"""Evaluate controller performance over multiple random trajectories.

Runs N headless episodes, each with a fresh randomly generated trajectory
(seeded for reproducibility), and reports per-episode metrics plus an
aggregate summary.

Metrics per episode
-------------------
  IAE   — Integral Absolute Error: integral of ||pos - ref|| dt
  RMSE  — Root Mean Squared position tracking error
  U²    — Integral of squared control effort ||u||² dt  (energy proxy)
  time  — Episode duration in seconds

Usage
-----
    python scripts/eval_randtraj.py --n 10
    python scripts/eval_randtraj.py --n 20 --config level_open_attitude.toml --seed 42
    python scripts/eval_randtraj.py --n 5 --controller attitude_mpc_lin_potential_randtraj.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ["SCIPY_ARRAY_API"] = "1"

import numpy as np

# Make the repo root importable regardless of working directory.
sys.path.insert(0, str(Path(__file__).parents[1]))

import gymnasium
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.utils import load_config, load_controller


def run_episode(
    env: gymnasium.Env,
    ctrl_cls,
    config,
    seed: int,
) -> dict:
    """Run one episode and return a metrics dict."""
    obs, info = env.reset(seed=seed)
    ctrl = ctrl_cls(obs, info, config, seed=seed)

    dt = 1.0 / config.env.freq
    t = 0.0
    sq_errs: list[float] = []
    abs_errs: list[float] = []
    u2_acc = 0.0
    status = "timeout"

    while True:
        pos = np.asarray(obs["pos"]).squeeze()
        i_ref = min(ctrl._tick, ctrl._tick_max)
        err = float(np.linalg.norm(pos - ctrl._waypoints_pos[i_ref]))
        abs_errs.append(err)
        sq_errs.append(err**2)

        action = ctrl.compute_control(obs, info)
        u2_acc += float(np.sum(np.asarray(action) ** 2)) * dt

        obs, reward, terminated, truncated, info = env.step(action)
        done = ctrl.step_callback(action, obs, reward, terminated, truncated, info)
        t += dt

        if terminated:
            status = "crash"
            break
        if done:
            status = "completed"
            break
        if truncated:
            status = "timeout"
            break

    ctrl.episode_callback()

    n = len(abs_errs) or 1
    return {
        "status": status,
        "flight_time": t,
        "iae": float(np.sum(abs_errs) * dt),
        "rmse": float(np.sqrt(np.mean(sq_errs))),
        "u2": u2_acc,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n", type=int, default=10, help="Number of episodes to run")
    parser.add_argument(
        "--config",
        type=str,
        default="level_open_attitude.toml",
        help="Config filename in config/",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="attitude_mpc_lin_potential_randtraj.py",
        help="Controller filename in lsy_drone_racing/control/",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed (episode i uses seed+i)")
    args = parser.parse_args()

    config_path = Path(__file__).parents[1] / "config" / args.config
    config = load_config(config_path)
    config.sim.render = False

    control_path = Path(__file__).parents[1] / "lsy_drone_racing" / "control"
    ctrl_cls = load_controller(control_path / args.controller)

    env = gymnasium.make(
        config.env.id,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
    )
    env = JaxToNumpy(env)

    print(f"Config    : {args.config}")
    print(f"Controller: {args.controller}")
    print(f"Episodes  : {args.n}  (seeds {args.seed}–{args.seed + args.n - 1})")
    print("-" * 60)

    results = []
    wall_times = []
    for ep in range(args.n):
        ep_seed = args.seed + ep
        t_wall = time.perf_counter()
        metrics = run_episode(env, ctrl_cls, config, ep_seed)
        wall_times.append(time.perf_counter() - t_wall)
        results.append(metrics)

        status_tag = {"completed": "DONE   ", "crash": "CRASH  ", "timeout": "TIMEOUT"}[metrics["status"]]
        print(
            f"  ep {ep + 1:3d}  [{status_tag}]"
            f"  t={metrics['flight_time']:6.2f}s"
            f"  IAE={metrics['iae']:.3f}"
            f"  RMSE={metrics['rmse']:.3f} m"
            f"  U²={metrics['u2']:.1f}"
            f"  wall={wall_times[-1]:.1f}s"
        )

    env.close()

    # --- Summary ---
    n_done = sum(r["status"] == "completed" for r in results)
    n_crash = sum(r["status"] == "crash" for r in results)
    n_timeout = sum(r["status"] == "timeout" for r in results)

    def _stats(key: str) -> str:
        vals = [r[key] for r in results]
        return f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"

    print()
    print("=" * 60)
    print(f"Summary over {args.n} episodes")
    print("=" * 60)
    print(f"  Completed : {n_done}/{args.n}")
    print(f"  Crashed   : {n_crash}/{args.n}")
    print(f"  Timeout   : {n_timeout}/{args.n}")
    print(f"  Flight time: {_stats('flight_time')} s")
    print(f"  IAE        : {_stats('iae')}")
    print(f"  RMSE       : {_stats('rmse')} m")
    print(f"  U²         : {_stats('u2')}")
    print(f"  Wall time  : {np.mean(wall_times):.1f} ± {np.std(wall_times):.1f} s/ep")


if __name__ == "__main__":
    main()
