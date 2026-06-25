"""Generate arbitrary random trajectories via cubic spline interpolation.

Produces one or more smooth 3-D trajectories suitable for use as reference
paths in drone-racing controllers or RL environments.  Each trajectory is
built from randomly sampled waypoints inside a bounded arena, with a fixed
takeoff ramp at the start so every trajectory begins from the same position
with a consistent upward climb.

Run as:

    python scripts/generate_trajectory.py                                   # print shape info
    python scripts/generate_trajectory.py --n 4 --plot                     # matplotlib 3-D plot
    python scripts/generate_trajectory.py --config level0_attitude.toml    # show in sim viewer
    python scripts/generate_trajectory.py --seed 7 --save traj.npy
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Must be set before scipy (or any package importing scipy) is loaded.
# crazyflow requires this mode; without it the two imports collide mid-session.
os.environ["SCIPY_ARRAY_API"] = "1"

import numpy as np


# ---------------------------------------------------------------------------
# Constants that mirror the training environment defaults
# ---------------------------------------------------------------------------
_TAKEOFF_POS = np.array([-1.5, 1.0, 0.07])
_TAKEOFF_RAMP = np.array([[-1.5, 1.0, 0.07], [-1.0, 0.55, 0.4], [0.3, 0.35, 0.7]])
_TAKEOFF_VEL = np.array([0.0, 0.0, 0.4])       # initial velocity (upward climb)
_ARENA_SCALE = np.array([1.2, 1.2, 0.5])        # half-widths of the random waypoint box
_ARENA_OFFSET = 0.3 * _TAKEOFF_POS + np.array([0.0, 0.0, 0.7])  # box centre


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------

def generate_trajectories(
    n: int = 1,
    num_waypoints: int = 10,
    trajectory_time: float = 15.0,
    freq: int = 500,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate *n* random cubic-spline trajectories.

    The first three waypoints of every trajectory are fixed to a short takeoff
    ramp; the remaining waypoints are drawn uniformly inside the arena box.
    The spline is conditioned on a gentle upward velocity at t=0 so take-off
    is smooth regardless of the random suffix.

    Args:
        n: Number of trajectories to generate.
        num_waypoints: Total number of knots (including the 3 fixed ramp knots).
        trajectory_time: Total duration of the trajectory in seconds.
        freq: Sampling frequency in Hz — output has ``trajectory_time * freq`` rows.
        rng: Optional numpy random generator for reproducibility.

    Returns:
        Array of shape ``(n, trajectory_time * freq, 3)`` containing xyz positions.
    """
    if num_waypoints < 4:
        raise ValueError("num_waypoints must be >= 4 (3 fixed ramp + at least 1 random)")

    from scipy.interpolate import CubicSpline

    if rng is None:
        rng = np.random.default_rng()

    n_steps = int(np.ceil(trajectory_time * freq))
    t_knots = np.linspace(0, trajectory_time, num_waypoints)
    t_eval = np.linspace(0, trajectory_time, n_steps)

    # Random portion of the waypoints
    random_wps = rng.uniform(-1, 1, size=(n, num_waypoints, 3)) * _ARENA_SCALE + _ARENA_OFFSET

    # Override first three knots with the fixed takeoff ramp (same for every trajectory)
    random_wps[:, :3, :] = _TAKEOFF_RAMP

    # Boundary condition: pin initial velocity to the upward-climb vector
    v0 = np.tile(_TAKEOFF_VEL, (n, 1))  # (n, 3)

    spline = CubicSpline(t_knots, random_wps, axis=1, bc_type=((1, v0), "not-a-knot"))
    return spline(t_eval)  # (n, n_steps, 3)


# ---------------------------------------------------------------------------
# Sim visualisation (matches the pattern used by sim.py / render_callback)
# ---------------------------------------------------------------------------

def visualise_in_sim(
    config_name: str,
    traj: np.ndarray,
    seed: int | None = None,
    controller: str | None = None,
) -> None:
    """Render *traj* as a green line in the Crazyflow simulation viewer.

    Loads the environment from *config_name* exactly as sim.py does.  If
    *controller* is given the drone flies under that controller's control;
    otherwise it hovers at the start position.  The generated trajectory is
    always drawn as a white line for reference.

    Args:
        config_name: Filename inside ``config/`` (e.g. ``"level_open_attitude.toml"``).
        traj: (T, 3) array of positions to draw.
        seed: Optional random seed forwarded to env.reset().
        controller: Controller filename inside ``lsy_drone_racing/control/``, or None to hover.
    """
    import gymnasium
    from crazyflow.sim.visualize import draw_line
    from drone_models.core import load_params
    from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

    from lsy_drone_racing.utils import load_config, load_controller

    config = load_config(Path(__file__).parents[1] / "config" / config_name)
    config.sim.render = True

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
    obs, info = env.reset(seed=seed)

    # Subsample to ~500 points for draw_line performance
    subsample = max(1, len(traj) // 500)
    traj_display = traj[::subsample]

    fps = 60
    i = 0

    if controller is not None:
        control_path = Path(__file__).parents[1] / "lsy_drone_racing" / "control"
        ctrl = load_controller(control_path / controller)(obs, info, config)

        while True:
            action = ctrl.compute_control(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            controller_finished = ctrl.step_callback(action, obs, reward, terminated, truncated, info)
            if terminated or truncated or controller_finished:
                break
            if ((i * fps) % config.env.freq) < fps:
                ctrl.render_callback(env.unwrapped.sim)
                env.render()
            i += 1
    else:
        params = load_params("so_rpy", config.sim.drone_model)
        hover_thrust = float(params["mass"] * abs(params["gravity_vec"][-1]))
        hover_action = np.array([0.0, 0.0, 0.0, hover_thrust])

        while True:
            obs, _, terminated, truncated, _ = env.step(hover_action)
            if terminated or truncated:
                break
            if ((i * fps) % config.env.freq) < fps:
                draw_line(env.unwrapped.sim, traj_display, rgba=(1.0, 1.0, 1.0, 0.6))
                env.render()
            i += 1

    env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n", type=int, default=1, help="Number of trajectories to generate")
    parser.add_argument("--waypoints", type=int, default=10, help="Number of spline knots")
    parser.add_argument("--time", type=float, default=15.0, help="Trajectory duration in seconds")
    parser.add_argument("--freq", type=int, default=500, help="Sampling frequency in Hz")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--save", type=str, default=None, help="Save trajectories to this .npy file")
    parser.add_argument("--plot", action="store_true", help="Plot the trajectories with matplotlib")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config filename in config/ — show trajectory in the Crazyflow sim viewer",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Controller filename in lsy_drone_racing/control/ — drone flies under this controller (requires --config)",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    trajs = generate_trajectories(
        n=args.n,
        num_waypoints=args.waypoints,
        trajectory_time=args.time,
        freq=args.freq,
        rng=rng,
    )

    print(f"Generated {args.n} trajectory/ies — shape: {trajs.shape}  (n, timesteps, xyz)")
    print(f"  x range: [{trajs[..., 0].min():.2f}, {trajs[..., 0].max():.2f}]")
    print(f"  y range: [{trajs[..., 1].min():.2f}, {trajs[..., 1].max():.2f}]")
    print(f"  z range: [{trajs[..., 2].min():.2f}, {trajs[..., 2].max():.2f}]")

    if args.save:
        out = Path(args.save)
        np.save(out, trajs)
        print(f"Saved to {out.resolve()}")

    if args.config:
        print(f"Opening Crazyflow viewer with config: {args.config}")
        visualise_in_sim(args.config, trajs[0], seed=args.seed, controller=args.controller)

    if args.plot:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        for i, traj in enumerate(trajs):
            subsample = max(1, len(traj) // 500)
            ax.plot(
                traj[::subsample, 0],
                traj[::subsample, 1],
                traj[::subsample, 2],
                label=f"traj {i}",
            )
        ax.scatter(*_TAKEOFF_POS, color="green", s=60, zorder=5, label="takeoff")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
