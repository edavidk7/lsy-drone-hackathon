"""Collision-aware min-snap path tracked by the trained PPO attitude policy, with re-planning.

This combines two pieces:

* **Planning** -- :class:`MinSnapController` builds a smooth, dynamically feasible, collision-aware
  trajectory through the gates (perpendicular gate crossings, pole + gate-frame keep-outs, snap
  minimization, acceleration-limited time scaling). See ``min_snap_controller.py``.
* **Tracking** -- the trained PPO policy (``ppo_drone_racing.ckpt``) consumes a *path* as a set of
  look-ahead points and outputs collective-thrust + attitude commands. It learned the drone
  dynamics, so it tracks tight turns far more gracefully than the built-in onboard state
  controller.

**Re-planning.** The initial plan uses the (possibly nominal) gate poses from the first
observation. As the drone flies and a gate's true pose becomes visible within ``sensor_range``,
its observed pose changes; we then re-plan from the *current* position through the remaining gates.
The re-plan uses the planner's fast path (seeded via points + clearance repair, skipping the slow
L-BFGS pass) so it runs in ~1 ms -- safe to do inside the control loop. This is what lets the
controller race the perturbed (level 2) and randomized (level 3) tracks, not just the static one.

.. note::
    This controller emits **attitude** commands, so it must run with ``control_mode = "attitude"``
    (see ``config/level0_attitude.toml`` etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from drone_models.core import load_params

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.min_snap_controller import MinSnapController
from lsy_drone_racing.control.train_rl import Agent

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MinSnapRLController(Controller):
    """Min-snap planner + PPO attitude policy, re-planning as gate poses are revealed."""

    PAD_SECONDS = 3.0  # extra hover samples appended so the policy's look-ahead stays valid at the end
    # The PPO policy was trained tracking a ~15 s reference. Plan a gentler path (slower cruise,
    # lower acceleration cap) than the state-mode default so the look-ahead spacing stays in the
    # policy's training distribution -- this is what lets it clear the final gate cleanly.
    PLAN_SPEED = 0.6
    PLAN_A_LIMIT = 6.0
    REPLAN = True             # re-plan when sensed gate poses change; harmless under perfect knowledge
    REPLAN_POS_THRESH = 0.04  # re-plan when a remaining gate's observed pose moves more than this [m]

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Plan the initial path and load the tracking policy.

        Args:
            obs: Initial observation (uses ``pos``, ``quat``, ``vel``, ``ang_vel``, ``gates_pos``,
                ``gates_quat``, ``obstacles_pos``, ``target_gate``).
            info: Reset info.
            config: Race config; ``config.env.freq`` is the control frequency.
        """
        super().__init__(obs, info, config)
        self.freq = config.env.freq
        self._traj_i = 0
        self._finished = False

        # --- Planning: reuse the collision-aware min-snap planner (full optimize on the first plan).
        MinSnapController.SPEED = self.PLAN_SPEED
        MinSnapController.A_LIMIT = self.PLAN_A_LIMIT
        self.planner = MinSnapController(obs, info, config)
        self._plan_gates = np.asarray(obs["gates_pos"], dtype=float).copy()
        self._build_trajectory()

        # --- Tracking: PPO policy setup, identical to AttitudeRL. -------------------------------
        drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self.drone_mass = drone_params["mass"]
        self.thrust_min = drone_params["thrust_min"] * 4
        self.thrust_max = drone_params["thrust_max"] * 4

        self.n_obs = 2
        self.n_samples = 10
        self.samples_dt = 0.1
        self.sample_offsets = np.array(
            np.arange(self.n_samples) * self.freq * self.samples_dt, dtype=int
        )

        self.agent = Agent((13 + 3 * self.n_samples + self.n_obs * 13 + 4,), (4,)).to("cpu")
        # Prefer the policy retrained on our min-snap gate courses (train_rl.py); fall back to the
        # stock checkpoint if it has not been trained yet.
        here = Path(__file__).parent
        model_path = here / "ppo_min_snap.ckpt"
        if not model_path.exists():
            model_path = here / "ppo_drone_racing.ckpt"
        self.agent.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        self.last_action = np.array([0.0, 0.0, 0.0, self.drone_mass * 9.81], dtype=np.float32)
        self.basic_obs_key = ["pos", "quat", "vel", "ang_vel"]
        basic_obs = np.concatenate([obs[k] for k in self.basic_obs_key], axis=-1)
        self.prev_obs = np.tile(basic_obs[None, :], (self.n_obs, 1))

    def _build_trajectory(self):
        """Sample the planner's current spline densely at the control frequency."""
        n_steps = int(self.freq * self.planner.t_total)
        ts = np.linspace(0.0, self.planner.t_total, max(n_steps, 2))
        traj = self.planner.spline(ts)
        pad = np.tile(traj[-1], (int(self.freq * self.PAD_SECONDS), 1))
        self.trajectory = np.concatenate([traj, pad], axis=0).astype(np.float32)  # (n, 3)
        self._traj_end = traj.shape[0] - 1

    def _maybe_replan(self, obs: dict[str, NDArray[np.floating]]):
        """Re-plan from the current state if a not-yet-passed gate's observed pose has changed."""
        if not self.REPLAN:  # plan-once mode (valid under perfect knowledge)
            return
        tg = int(obs["target_gate"])
        if tg < 0:  # all gates passed
            return
        gates_pos = np.asarray(obs["gates_pos"], dtype=float)
        # Largest pose change among the remaining gates vs. what the current plan assumed.
        moved = np.linalg.norm(gates_pos[tg:] - self._plan_gates[tg:], axis=1)
        if moved.size and moved.max() > self.REPLAN_POS_THRESH:
            self.planner.replan(
                start_pos=obs["pos"],
                first_gate=tg,
                gates_pos=gates_pos,
                gates_quat=obs["gates_quat"],
                obstacles_pos=obs["obstacles_pos"],
                optimize=False,  # fast seed + clearance repair (~1 ms), safe in the loop
            )
            self._plan_gates = gates_pos.copy()
            self._build_trajectory()
            self._traj_i = 0  # restart trajectory indexing from the new path's origin

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Return collective thrust + roll/pitch/yaw from the PPO policy tracking the path."""
        self._maybe_replan(obs)
        if self._traj_i >= self._traj_end and int(obs["target_gate"]) < 0:
            self._finished = True

        obs_rl = torch.tensor(self._obs_rl(obs), dtype=torch.float32).unsqueeze(0).to("cpu")
        with torch.no_grad():
            act, _, _, _ = self.agent.get_action_and_value(obs_rl, deterministic=True)
            self.last_action = np.asarray(torch.asarray(act.squeeze(0))).copy()
            act[..., 2] = 0.0  # zero yaw command, as in training
        return self._scale_actions(act.squeeze(0).numpy()).astype(np.float32)

    def _obs_rl(self, obs: dict[str, NDArray[np.floating]]) -> NDArray[np.floating]:
        """Build the policy observation: basic obs, path look-ahead, stacked history, last action."""
        basic_obs = np.concatenate([obs[k] for k in self.basic_obs_key], axis=-1)
        idx = np.clip(self._traj_i + self.sample_offsets, 0, self.trajectory.shape[0] - 1)
        local_samples = (self.trajectory[idx] - obs["pos"]).reshape(-1)
        out = np.concatenate(
            [basic_obs, local_samples, self.prev_obs.reshape(-1), self.last_action], axis=-1
        ).astype(np.float32)
        self.prev_obs = np.concatenate([self.prev_obs[1:, :], basic_obs[None, :]], axis=0)
        return out

    def _scale_actions(self, actions: NDArray) -> NDArray:
        """Rescale and clip policy actions from [-1, 1] to the simulator's command range."""
        scale = np.array(
            [np.pi / 2, np.pi / 2, np.pi / 2, (self.thrust_max - self.thrust_min) / 2.0],
            dtype=np.float32,
        )
        mean = np.array([0.0, 0.0, 0.0, (self.thrust_max + self.thrust_min) / 2.0], dtype=np.float32)
        return np.clip(actions, -1.0, 1.0) * scale + mean

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Advance the trajectory index. Returns True when the path is finished."""
        self._traj_i += 1
        return self._finished

    def episode_callback(self):
        """Reset indices for a new episode."""
        self._traj_i = 0
        self._finished = False
