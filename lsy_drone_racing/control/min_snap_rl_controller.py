"""Collision-aware min-snap path tracked by the trained PPO attitude policy.

This combines two pieces:

* **Planning** -- :class:`MinSnapController` builds a smooth, dynamically feasible, collision-aware
  trajectory through the gates (perpendicular gate crossings, pole + gate-frame keep-outs, snap
  minimization, acceleration-limited time scaling). See ``min_snap_controller.py``.
* **Tracking** -- the trained PPO policy (``ppo_drone_racing.ckpt``) consumes a *path* as a set of
  look-ahead points and outputs collective-thrust + attitude commands. It learned the drone
  dynamics, so it tracks tight turns far more gracefully than the built-in onboard state
  controller (which overshoots when the feedforward velocity spikes at a corner).

The policy was trained tracking a hand-tuned cubic spline; here we feed it our planned trajectory
instead, sampled densely at the control frequency. The observation layout (basic obs, look-ahead
samples, stacked previous obs, last action) and action scaling are identical to ``attitude_rl.py``.

.. note::
    This controller emits **attitude** commands, so it must run with ``control_mode = "attitude"``
    (see ``config/level0_attitude.toml``). The path itself is identical to the state-mode
    ``min_snap_controller`` -- only the tracking layer changes.
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
    """Min-snap planner + PPO attitude policy for robust tracking."""

    PAD_SECONDS = 3.0  # extra hover samples appended so the policy's look-ahead stays valid at the end
    # The PPO policy was trained tracking a ~15 s reference. Plan a gentler path (slower cruise,
    # lower acceleration cap) than the state-mode default so the look-ahead spacing stays in the
    # policy's training distribution -- this is what lets it clear the final gate cleanly.
    PLAN_SPEED = 0.6
    PLAN_A_LIMIT = 6.0

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Plan the path and load the tracking policy.

        Args:
            obs: Initial observation (uses ``pos``, ``quat``, ``vel``, ``ang_vel``, ``gates_pos``,
                ``gates_quat``, ``obstacles_pos``).
            info: Reset info.
            config: Race config; ``config.env.freq`` is the control frequency.
        """
        super().__init__(obs, info, config)
        self.freq = config.env.freq
        self._tick = 0
        self._finished = False

        # --- Planning: reuse the collision-aware min-snap planner, then sample it densely. -------
        MinSnapController.SPEED = self.PLAN_SPEED
        MinSnapController.A_LIMIT = self.PLAN_A_LIMIT
        planner = MinSnapController(obs, info, config)
        n_steps = int(self.freq * planner.t_total)
        ts = np.linspace(0.0, planner.t_total, max(n_steps, 2))
        traj = planner.spline(ts)
        pad = np.tile(traj[-1], (int(self.freq * self.PAD_SECONDS), 1))
        self.trajectory = np.concatenate([traj, pad], axis=0).astype(np.float32)  # (n, 3)
        self._traj_end = traj.shape[0] - 1  # index where the racing path ends

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
        model_path = Path(__file__).parent / "ppo_drone_racing.ckpt"
        self.agent.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        self.last_action = np.array([0.0, 0.0, 0.0, self.drone_mass * 9.81], dtype=np.float32)
        self.basic_obs_key = ["pos", "quat", "vel", "ang_vel"]
        basic_obs = np.concatenate([obs[k] for k in self.basic_obs_key], axis=-1)
        self.prev_obs = np.tile(basic_obs[None, :], (self.n_obs, 1))

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Return collective thrust + roll/pitch/yaw from the PPO policy tracking the path."""
        if self._tick >= self._traj_end:
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
        idx = np.clip(self._tick + self.sample_offsets, 0, self.trajectory.shape[0] - 1)
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
        """Advance the tick. Returns True when the path is finished."""
        self._tick += 1
        return self._finished

    def episode_callback(self):
        """Reset the tick counter for a new episode."""
        self._tick = 0
        self._finished = False
