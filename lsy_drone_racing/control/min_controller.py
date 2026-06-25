"""Min-snap path tracked by a geometric (differential-flatness) attitude controller.

This is the *careful, deterministic* counterpart to :class:`MinSnapRLController`. It reuses the same
collision-aware min-snap planner (:class:`MinSnapController`) but replaces the learned PPO tracker
with a classic cascaded controller:

1. **Planning** -- :class:`MinSnapController` builds a smooth, collision-aware trajectory through the
   gates and exposes it as a spline (position + velocity + acceleration are all available as
   analytic derivatives). See ``min_snap_controller.py``.
2. **Tracking** -- at each control step we evaluate the planned position/velocity/**acceleration** at
   the current time and run a PID law *around the acceleration feedforward*::

       a_cmd = a_ff + Kp (p_des - p) + Kd (v_des - v) + Ki ∫(p_des - p) + g ẑ
       F_des = m · a_cmd                          (desired thrust vector, world frame)
       thrust = F_des · z_body                    (project onto the current body-z axis)
       R_des  = attitude whose body-z aligns with F_des, at the desired yaw

   The acceleration feedforward is the key difference from the stock ``AttitudeController``: the PID
   only has to correct small residual errors instead of synthesizing the whole maneuver, so tracking
   stays tight and stable even at modest gains. Speed is irrelevant to stability here -- plan as slow
   as you like and the drone follows the curve faithfully.

It emits **attitude** commands ``[roll, pitch, yaw, collective_thrust]`` and so must run with
``control_mode = "attitude"`` (e.g. ``config/levelcompetition.toml``).

**Re-planning** mirrors :class:`MinSnapRLController`: when a not-yet-passed gate's observed pose
changes we re-plan from the current position via the planner's fast path. Under the competition's
perfect-knowledge setting no re-plan fires, but it keeps the controller usable on levels 2/3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from drone_models.core import load_params
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.min_snap_controller import MinSnapController

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


class MinSnapTrackingController(Controller):
    """Min-snap planner + geometric attitude tracker (no learning). A careful RL baseline."""

    # Plan a gentle trajectory: tracking accuracy, not lap time, is the goal of this baseline.
    PLAN_SPEED = 0.6
    PLAN_A_LIMIT = 6.0

    # Cascaded controller gains, in acceleration space (m/s^2 per m, per m/s). Conservative and
    # well-damped -- with the acceleration feedforward the loop only corrects residual error.
    KP = np.array([8.0, 8.0, 12.0])
    KD = np.array([5.0, 5.0, 7.0])
    KI = np.array([1.0, 1.0, 2.0])
    I_LIMIT = np.array([1.0, 1.0, 1.0])  # integral clamp [m·s]
    G = 9.81

    REPLAN = True
    REPLAN_POS_THRESH = 0.04  # re-plan when a remaining gate's observed pose moves more than this [m]

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Plan the initial path and set up the tracker.

        Args:
            obs: Initial observation (uses ``pos``, ``quat``, ``vel``, ``gates_pos``, ``gates_quat``,
                ``obstacles_pos``, ``target_gate``).
            info: Reset info (unused).
            config: Race config; ``config.env.freq`` is the control frequency.
        """
        super().__init__(obs, info, config)
        self.freq = config.env.freq
        self._tick = 0
        self._finished = False
        self._i_error = np.zeros(3)

        # --- Planning: reuse the collision-aware min-snap planner (full optimize on the first plan).
        MinSnapController.SPEED = self.PLAN_SPEED
        MinSnapController.A_LIMIT = self.PLAN_A_LIMIT
        self.planner = MinSnapController(obs, info, config)
        self._plan_gates = np.asarray(obs["gates_pos"], dtype=float).copy()
        self._refresh_splines()

        # --- Tracking: drone mass for the force law and thrust limits for clipping.
        drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self.drone_mass = drone_params["mass"]
        self.thrust_min = drone_params["thrust_min"] * 4
        self.thrust_max = drone_params["thrust_max"] * 4

    def _refresh_splines(self):
        """Cache the planner's position spline and its velocity/acceleration derivatives."""
        self._pos_spl = self.planner.spline
        self._vel_spl = self._pos_spl.derivative(1)
        self._acc_spl = self._pos_spl.derivative(2)
        self._t_total = float(self.planner.t_total)

    def _maybe_replan(self, obs: dict[str, NDArray[np.floating]]):
        """Re-plan from the current state if a not-yet-passed gate's observed pose has changed."""
        if not self.REPLAN:
            return
        tg = int(obs["target_gate"])
        if tg < 0:  # all gates passed
            return
        gates_pos = np.asarray(obs["gates_pos"], dtype=float)
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
            self._refresh_splines()
            self._tick = 0  # the new path starts at the current position at t=0
            self._i_error[:] = 0.0

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Return ``[roll, pitch, yaw, collective_thrust]`` tracking the planned spline."""
        self._maybe_replan(obs)

        t = self._tick / self.freq
        past_end = t >= self._t_total
        t = min(t, self._t_total)
        if past_end and int(obs["target_gate"]) < 0:
            self._finished = True

        des_pos = self._pos_spl(t)
        # Hold position (zero feedforward velocity/acceleration) once the trajectory is exhausted.
        des_vel = np.zeros(3) if past_end else self._vel_spl(t)
        des_acc = np.zeros(3) if past_end else self._acc_spl(t)

        pos_err = des_pos - obs["pos"]
        vel_err = des_vel - obs["vel"]
        self._i_error = np.clip(
            self._i_error + pos_err / self.freq, -self.I_LIMIT, self.I_LIMIT
        )

        # Desired acceleration -> desired thrust vector (world frame), gravity-compensated.
        a_cmd = des_acc + self.KP * pos_err + self.KD * vel_err + self.KI * self._i_error
        a_cmd[2] += self.G
        f_des = self.drone_mass * a_cmd  # [N], world frame

        # Collective thrust = projection of the desired force onto the current body-z axis.
        z_body = R.from_quat(obs["quat"]).as_matrix()[:, 2]
        thrust = float(np.clip(f_des @ z_body, self.thrust_min, self.thrust_max))

        # Desired attitude: body-z aligned with f_des, body-x in the desired-yaw direction.
        des_yaw = 0.0
        z_axis = f_des / (np.linalg.norm(f_des) + 1e-9)
        x_c = np.array([np.cos(des_yaw), np.sin(des_yaw), 0.0])
        y_axis = np.cross(z_axis, x_c)
        y_axis /= np.linalg.norm(y_axis) + 1e-9
        x_axis = np.cross(y_axis, z_axis)
        rpy = R.from_matrix(np.vstack([x_axis, y_axis, z_axis]).T).as_euler("xyz")

        return np.concatenate([rpy, [thrust]], dtype=np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Advance the time index. Returns True when the path is finished."""
        self._tick += 1
        return self._finished

    def episode_callback(self):
        """Reset indices and integral state for a new episode."""
        self._tick = 0
        self._finished = False
        self._i_error[:] = 0.0

    def render_callback(self, sim: Sim):
        """Draw the planned path (green), its waypoints (blue), and the current setpoint (red)."""
        from crazyflow.sim.visualize import draw_line, draw_points

        traj = self._pos_spl(np.linspace(0.0, self._t_total, 200))
        draw_line(sim, traj, rgba=(0.0, 1.0, 0.0, 1.0))
        draw_points(sim, self.planner._waypoints, rgba=(0.0, 0.0, 1.0, 1.0), size=0.02)
        setpoint = self._pos_spl(min(self._tick / self.freq, self._t_total)).reshape(1, -1)
        draw_points(sim, setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.03)
