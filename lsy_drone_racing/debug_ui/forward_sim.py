"""Shadow simulation that predicts the drone's upcoming trajectory from a received observation.

The dashboard server owns one :class:`ShadowSim`. For every observation received from the live
controller it:

1. **Injects** the live drone kinematics and gate/obstacle layout into a private ``DroneRaceEnv``
   (so the prediction matches the live, possibly randomized, track), and
2. **Rolls out** ~200 ms by repeatedly running its own copy of the controller and stepping the env.

The live controller and this shadow controller are independent instances, so rolling out here has
no effect on the drone.

Note on fidelity: state injection writes the analytical drone state and the gate/obstacle layout.
Collision/contact checks inside ``env.step`` read MuJoCo/MJX state that is not re-synced here, so
predicted ``terminated`` flags may be stale -- we only use the integrated positions/actions, which
come from the analytical dynamics and are accurate for a short horizon.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import gymnasium
import jax.numpy as jp
import mujoco
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.utils import load_controller

if TYPE_CHECKING:
    from ml_collections import ConfigDict

logger = logging.getLogger(__name__)


class ShadowSim:
    """A private env + controller used for predictions and ego-camera rendering."""

    def __init__(self, config: ConfigDict, controller_file: str = "nav_rl_controller.py"):
        self._config = config
        self.control_mode = str(config.env.control_mode)
        # Build the env exactly like scripts/sim.py so dynamics/config match the live run.
        self._env = gymnasium.make(
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
        self._env = JaxToNumpy(self._env)
        obs, info = self._env.reset()

        control_path = Path(__file__).parents[1] / "control"
        controller_cls = load_controller(control_path / controller_file)
        self._controller = controller_cls(obs, info, config)
        logger.info("ShadowSim ready (control_mode=%s)", self.control_mode)

    @property
    def freq(self) -> int:
        """Environment step frequency in Hz."""
        return int(self._config.env.freq)

    def _inject(self, obs: dict) -> None:
        """Overwrite the env's data with the live observation (drone state + track layout)."""
        data = self._env.unwrapped.data
        states = data.sim_data.states

        def set00(arr, value):  # set the (env=0, drone=0) slot
            return arr.at[0, 0].set(jp.asarray(value, dtype=arr.dtype))

        def set0(arr, value):  # set the (env=0) slot
            return arr.at[0].set(jp.asarray(value, dtype=arr.dtype))

        states = states.replace(
            pos=set00(states.pos, obs["pos"]),
            quat=set00(states.quat, obs["quat"]),
            vel=set00(states.vel, obs["vel"]),
            ang_vel=set00(states.ang_vel, obs["ang_vel"]),
        )
        sim_data = data.sim_data.replace(states=states)

        gates_pos = np.asarray(obs["gates_pos"], dtype=np.float32)
        gates_quat = np.asarray(obs["gates_quat"], dtype=np.float32)
        obstacles_pos = np.asarray(obs["obstacles_pos"], dtype=np.float32)
        # Set both true and nominal layout to the observed values so the sensor-mask in obs() always
        # reports what the live controller actually saw, regardless of the visited flags.
        data = data.replace(
            sim_data=sim_data,
            gates_pos=set0(data.gates_pos, gates_pos),
            nominal_gates_pos=set0(data.nominal_gates_pos, gates_pos),
            gates_quat=set0(data.gates_quat, gates_quat),
            nominal_gates_quat=set0(data.nominal_gates_quat, gates_quat),
            obstacles_pos=set0(data.obstacles_pos, obstacles_pos),
            nominal_obstacles_pos=set0(data.nominal_obstacles_pos, obstacles_pos),
            gates_visited=set00(data.gates_visited, obs["gates_visited"]),
            obstacles_visited=set00(data.obstacles_visited, obs["obstacles_visited"]),
            target_gate=set00(data.target_gate, obs["target_gate"]),
            last_drone_pos=set00(data.last_drone_pos, obs["pos"]),
            disabled_drones=set0(data.disabled_drones, np.zeros(data.disabled_drones.shape[1:], dtype=bool)),
            marked_for_reset=set0(data.marked_for_reset, False),
            steps=set0(data.steps, 0),
        )
        self._env.unwrapped.data = data

    def _available_cameras(self) -> list[str]:
        """List available MuJoCo camera names in the model."""
        model = self._env.unwrapped.sim.mj_model
        names: list[str] = []
        for cam_id in range(int(model.ncam)):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id)
            if name:
                names.append(str(name))
        return names

    def _resolve_camera(self, preferred: str = "fpv_cam:0") -> int | str:
        """Resolve camera name with robust fallbacks."""
        names = self._available_cameras()
        if preferred in names:
            return preferred
        for fallback in ("fpv_cam:0", "track_cam:0"):
            if fallback in names:
                return fallback
        if names:
            return names[0]
        return -1

    def render_ego_frame(
        self,
        obs: dict,
        camera_name: str = "fpv_cam:0",
        width: int = 640,
        height: int = 360,
    ) -> np.ndarray | None:
        """Render one RGB frame from the requested ego camera."""
        self._inject(obs)
        env = self._env.unwrapped
        try:
            # Force sync every frame after state injection; otherwise camera can appear frozen.
            env.data, env.sim.mjx_data = env._render_sync(env.data, env.sim.mjx_data)
            frame = env.sim.render(
                mode="rgb_array",
                camera=self._resolve_camera(camera_name),
                width=int(width),
                height=int(height),
            )
            if frame is None:
                return None
            return np.asarray(frame, dtype=np.uint8)
        except Exception:  # noqa: BLE001 - video failure must not kill telemetry.
            logger.warning("Ego-camera render failed", exc_info=True)
            return None

    def _motor_telemetry(self) -> dict:
        """Extract per-motor RPM and estimated torque from current sim state."""
        sim_data = self._env.unwrapped.data.sim_data
        rpm = np.abs(np.asarray(sim_data.states.rotor_vel, dtype=np.float32)[0, 0])
        peak = float(np.max(rpm)) if rpm.size else 0.0
        intensity = rpm / (peak + 1e-6) if peak > 0 else np.zeros_like(rpm)

        torque_est = np.zeros_like(rpm)
        coeff = getattr(sim_data.params, "rpm2torque", None)
        if coeff is not None:
            try:
                torque_est = np.asarray(np.polyval(np.asarray(coeff, dtype=np.float32), rpm), dtype=np.float32)
            except Exception:  # noqa: BLE001
                torque_est = np.zeros_like(rpm)

        return {
            "rpm": rpm.astype(float).tolist(),
            "torque_est": torque_est.astype(float).tolist(),
            "intensity": intensity.astype(float).tolist(),
        }

    def predict(
        self,
        obs: dict,
        prev_action: np.ndarray,
        current_action: np.ndarray | None = None,
        tick: int | None = None,
        n_steps: int = 10,
    ) -> dict:
        """Inject the observation and roll the controller forward ``n_steps`` env steps.

        ``tick`` is the live controller's step counter (published as ``t``). It seeds time-indexed
        controllers (e.g. the MPC, whose reference trajectory is indexed by ``_tick``) so the rolled
        reference matches the injected state; without it the MPC would track its start waypoints from
        a mid-track state and the QP would go infeasible.

        Returns a dict with predicted positions ``xyz`` (shape (n_steps+1, 3), including the current
        position as the first point) and the predicted ``actions`` (shape (n_steps, act_dim)).

        If ``current_action`` is provided, it is stepped first (representing the already-published
        live controller command), then subsequent steps use this shadow controller.
        """
        self._inject(obs)

        # Fork controller state so successive predictions don't accumulate state: snapshot the
        # mutable bits, seed them to match the live controller, restore in `finally`.
        ctrl = self._controller
        saved_prev = getattr(ctrl, "_prev_action", None)
        saved_finished = getattr(ctrl, "_finished", None)
        saved_tick = getattr(ctrl, "_tick", None)
        if prev_action is not None and hasattr(ctrl, "_prev_action"):
            ctrl._prev_action = np.asarray(prev_action, dtype=np.float32)
        if tick is not None and hasattr(ctrl, "_tick"):
            ctrl._tick = int(tick)

        xyz = [np.asarray(obs["pos"], dtype=np.float32)]
        actions: list[np.ndarray] = []
        obs_k = obs
        horizon = max(int(n_steps), 0)
        steps_done = 0
        motor = self._motor_telemetry()
        try:
            if current_action is not None and np.asarray(current_action).size > 0 and horizon > 0:
                action0 = np.asarray(current_action, dtype=np.float32)
                actions.append(action0)
                obs_k, reward, terminated, truncated, info = self._env.step(action0)
                ctrl.step_callback(action0, obs_k, reward, terminated, truncated, info)
                xyz.append(np.asarray(obs_k["pos"], dtype=np.float32))
                motor = self._motor_telemetry()
                steps_done = 1
                if terminated or truncated:
                    horizon = steps_done

            for k in range(steps_done, horizon):
                action = np.asarray(ctrl.compute_control(obs_k), dtype=np.float32)
                actions.append(action)
                obs_k, reward, terminated, truncated, info = self._env.step(action)
                # Advance controller internal state exactly like scripts/sim.py's loop, so e.g. the
                # MPC's trajectory index keeps pace with the rolled-out state.
                ctrl.step_callback(action, obs_k, reward, terminated, truncated, info)
                xyz.append(np.asarray(obs_k["pos"], dtype=np.float32))
                if k == 0 and steps_done == 0:
                    motor = self._motor_telemetry()
                if terminated or truncated:
                    break
        except Exception:  # noqa: BLE001 - a failed rollout must not kill the server loop.
            logger.warning("Forward rollout failed", exc_info=True)
        finally:
            if saved_prev is not None:
                ctrl._prev_action = saved_prev
            if saved_finished is not None:
                ctrl._finished = saved_finished
            if saved_tick is not None:
                ctrl._tick = saved_tick

        return {
            "xyz": np.asarray(xyz, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32) if actions else np.zeros((0,)),
            "motor": motor,
        }

    def close(self) -> None:
        """Close the underlying env."""
        try:
            self._env.close()
        except Exception:  # noqa: BLE001
            pass
