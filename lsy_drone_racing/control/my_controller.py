"""Your controller — START HERE.

This is the only file you need to write for the challenge. Implement a subclass of ``Controller``
that, given the current observation, returns the next command for the drone. The same controller
runs in simulation and on the real drone.

This template just takes off and hovers (so it runs out of the box). Replace ``compute_control``
with your racing logic. Good starting points to copy from:
    - ``state_controller.py``  : trajectory tracking with state setpoints (easiest)
    - ``attitude_mpc.py``      : model predictive control with attitude/thrust commands
    - ``attitude_rl.py`` + ``train_rl.py`` : a trained RL policy

Key rule: the demo track is HELD OUT. Read the gate poses from ``obs`` at runtime
(``gates_pos``, ``gates_quat``, ``target_gate``) — do NOT hardcode gate coordinates, or you will
fail on the unseen track. See CHALLENGE.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MyController(Controller):
    """A minimal example controller (takes off and hovers). Replace with your racing logic."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the controller.

        Args:
            obs: Initial observation. Keys include ``pos``, ``quat`` (xyzw), ``vel``, ``ang_vel``,
                ``target_gate``, ``gates_pos``, ``gates_quat``, ``gates_visited``,
                ``obstacles_pos``, ``obstacles_visited``. Gate/obstacle poses are exact only within
                the sensor range, otherwise their nominal (config) pose is reported.
            info: Reset info.
            config: Race configuration (``config.env.freq`` is the control frequency, etc.).
        """
        super().__init__(obs, info, config)
        self.freq = config.env.freq
        # Hover target: 1 m above the start position. TODO: replace with a plan through the gates.
        self._hover = np.asarray(obs["pos"], dtype=np.float32).copy()
        self._hover[2] = 1.0

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Return the next command.

        With ``control_mode = "state"`` (default) return a 13-D state setpoint
        ``[x, y, z, vx, vy, vz, ax, ay, az, yaw, roll_rate, pitch_rate, yaw_rate]``.
        With ``control_mode = "attitude"`` return ``[collective_thrust, roll, pitch, yaw]``.

        Args:
            obs: Current observation (see ``__init__`` for the keys).
            info: Optional additional info.

        Returns:
            The command as a numpy array.
        """
        # TODO: your racing logic here. Use obs["target_gate"] and obs["gates_pos"]/["gates_quat"]
        #       to fly through the gates. For now we just hover at the start position.
        return np.concatenate((self._hover, np.zeros(10)), dtype=np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Called once after each step. Return True to signal the controller is finished."""
        return False
