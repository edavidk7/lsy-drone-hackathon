"""Linearized attitude MPC for the drone racing task.

This controller uses the attitude interface ``[roll, pitch, yaw, collective_thrust]``.  It
re-plans a short path through the currently observed remaining gates on every control step, then
tracks that path with a dense linear MPC.  The linear model is rebuilt from the same
``drone_models.so_rpy.symbolic_dynamics_euler`` dynamics used by ``attitude_mpc.py`` by evaluating
the continuous-time Jacobians along the current reference and discretizing them with forward Euler.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as ca
import numpy as np
from drone_models.core import load_params
from drone_models.so_rpy import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class LinearizedAttitudeMPC(Controller):
    """A lightweight linear MPC using collective thrust and attitude commands."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the controller."""
        super().__init__(obs, info, config)

        mpc_config = config.controller.get("linearized_mpc", {})
        self._N = int(mpc_config.get("horizon", 25))
        self._dt = 1.0 / float(config.env.freq)
        self._regularization = float(mpc_config.get("regularization", 1e-7))

        self._target_speed = float(mpc_config.get("target_speed", 0.85))
        self._gate_offset = float(mpc_config.get("gate_offset", 0.20))
        self._previous_gate_exit_offset = float(mpc_config.get("previous_gate_exit_offset", 0.40))
        self._previous_gate_exit_steps = int(mpc_config.get("previous_gate_exit_steps", 24))
        self._avoid_radius = float(mpc_config.get("avoid_radius", 0.55))
        self._insert_detours = bool(mpc_config.get("insert_obstacle_detours", False))
        self._brake_dist = float(mpc_config.get("brake_dist", 0.55))
        self._min_path_spacing = float(mpc_config.get("min_path_spacing", 1e-6))

        self.drone_params = load_params("so_rpy", config.sim.drone_model)
        self._mass = float(self.drone_params["mass"])
        self._g = float(-np.asarray(self.drone_params["gravity_vec"], dtype=float)[2])
        self._hover_thrust = self._mass * self._g
        self._thrust_min = float(self.drone_params["thrust_min"]) * 4.0
        self._thrust_max = float(self.drone_params["thrust_max"]) * 4.0
        self._nx = 12
        self._nu = 4
        self._build_linearization_function()

        self._max_tilt = float(mpc_config.get("max_tilt", 0.50))
        self._max_yaw = float(mpc_config.get("max_yaw", 0.50))
        self._max_tilt_step = float(mpc_config.get("max_tilt_rate", 3.0)) * self._dt
        self._max_thrust_step = float(mpc_config.get("max_thrust_rate", 4.0)) * self._dt

        self._q = np.diag(
            np.asarray(
                mpc_config.get(
                    "state_weights",
                    [
                        55.0,
                        55.0,
                        220.0,
                        1.0,
                        1.0,
                        0.5,
                        8.0,
                        8.0,
                        18.0,
                        0.5,
                        0.5,
                        0.4,
                    ],
                ),
                dtype=float,
            )
        )
        terminal_scale = float(mpc_config.get("terminal_weight_scale", 3.0))
        self._p = self._q * terminal_scale
        self._r = np.diag(
            np.asarray(mpc_config.get("input_weights", [1.0, 1.0, 0.4, 35.0]), dtype=float)
        )
        self._r_rate = np.diag(
            np.asarray(mpc_config.get("input_rate_weights", [20.0, 20.0, 4.0, 50.0]), dtype=float)
        )

        self._last_action = np.array([0.0, 0.0, 0.0, self._hover_thrust], dtype=float)
        self._last_target_gate = int(np.asarray(obs["target_gate"]))
        self._gate_switch_ticks = self._previous_gate_exit_steps
        self._last_log: dict = {}
        self._finished = False

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next ``[roll, pitch, yaw, collective_thrust]`` action."""
        if int(np.asarray(obs["target_gate"])) < 0:
            self._finished = True
        current_target_gate = int(np.asarray(obs["target_gate"]))
        if current_target_gate != self._last_target_gate:
            self._last_target_gate = current_target_gate
            self._gate_switch_ticks = 0

        pos_ref, vel_ref = self._plan_reference(obs)
        self._gate_switch_ticks += 1
        x0 = self._current_state(obs)
        x_ref = self._build_reference_states(pos_ref, vel_ref)
        u_ref = self._build_reference_inputs()

        a_seq, b_seq, c_seq = self._build_dynamics_sequence(x_ref, u_ref)
        sx, su, soff = self._build_prediction_matrices(a_seq, b_seq, c_seq)
        q_bar = self._build_block_diagonal_state_weights()
        r_bar = self._build_block_diagonal_input_weights(self._r)
        r_rate_bar = self._build_block_diagonal_input_weights(self._r_rate)
        x_ref_vec = x_ref[1 : self._N + 1].reshape(-1)
        u_ref_vec = u_ref.reshape(-1)
        d_rate, d_rate_offset = self._build_input_rate_matrix()

        nominal = sx @ x0 + soff
        hessian = (
            su.T @ q_bar @ su
            + r_bar
            + d_rate.T @ r_rate_bar @ d_rate
            + self._regularization * np.eye(self._N * self._nu)
        )
        gradient = (
            su.T @ q_bar @ (nominal - x_ref_vec)
            + r_bar @ (-u_ref_vec)
            + d_rate.T @ r_rate_bar @ d_rate_offset
        )

        try:
            cholesky = np.linalg.cholesky(hessian)
            du = np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, -gradient))
            solver = "cholesky"
        except np.linalg.LinAlgError:
            solution = np.linalg.solve(hessian, -gradient)
            solver = "solve"
        else:
            solution = du

        if not np.all(np.isfinite(solution)):
            action = self._pd_fallback(obs, pos_ref[1], vel_ref[1], "non_finite_solution")
            return action.astype(np.float32)

        action = self._clip_action_step(solution[: self._nu])
        self._last_action = action.copy()

        predicted = (nominal + su @ solution).reshape(self._N, -1)
        self._last_log = {
            "controller": "linearized_attitude_mpc",
            "solver": solver,
            "model": "so_rpy_jacobian",
            "state": x0,
            "reference_pos": pos_ref,
            "reference_vel": vel_ref,
            "predicted_states": predicted,
            "predicted_inputs": solution.reshape(self._N, self._nu),
            "action": action.copy(),
        }
        return action.astype(np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Report whether the controller has finished the race."""
        return self._finished

    def episode_callback(self):
        """Reset controller state between episodes."""
        self._finished = False
        self._last_action[:] = np.array([0.0, 0.0, 0.0, self._hover_thrust], dtype=float)
        self._last_target_gate = -999
        self._gate_switch_ticks = self._previous_gate_exit_steps
        self._last_log = {}

    def _plan_reference(
        self, obs: dict[str, NDArray[np.floating]]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Plan a short reference from the current state through live remaining gate poses."""
        pos = np.asarray(obs["pos"], dtype=float)
        gates_pos = np.asarray(obs["gates_pos"], dtype=float)
        gates_quat = np.asarray(obs["gates_quat"], dtype=float)
        obstacles_pos = np.asarray(obs["obstacles_pos"], dtype=float).reshape(-1, 3)

        target = int(np.asarray(obs["target_gate"]))
        remaining = range(target, len(gates_pos)) if target >= 0 else range(0)

        points = [pos]
        is_gate_center = [False]
        prev = pos
        if target > 0 and self._gate_switch_ticks < self._previous_gate_exit_steps:
            prev_center = gates_pos[target - 1]
            prev_normal = self._gate_exit_normal(target - 1, gates_pos, gates_quat, pos)
            prev_exit = prev_center + prev_normal * self._previous_gate_exit_offset
            if np.dot(pos - prev_center, prev_normal) < self._previous_gate_exit_offset:
                points.append(prev_exit)
                is_gate_center.append(False)
                prev = prev_exit

        for gate_idx in remaining:
            center = gates_pos[gate_idx]
            normal = self._gate_exit_normal(gate_idx, gates_pos, gates_quat, prev)

            points.extend([center, center + normal * self._gate_offset])
            is_gate_center.extend([True, False])
            prev = center + normal * self._gate_offset

        points = np.asarray(points, dtype=float)
        is_gate_center = np.asarray(is_gate_center, dtype=bool)

        if obstacles_pos.size:
            points = self._push_points_away_from_obstacles(points, is_gate_center, obstacles_pos)
            if self._insert_detours:
                points = self._insert_obstacle_detours(points, obstacles_pos)

        if len(points) > 1:
            deltas = np.linalg.norm(np.diff(points, axis=0), axis=1)
            keep = np.concatenate(([True], deltas > self._min_path_spacing))
            points = points[keep]

        pos_ref = np.tile(pos, (self._N + 1, 1))
        vel_ref = np.zeros((self._N + 1, 3), dtype=float)
        if len(points) < 2:
            return pos_ref, vel_ref

        arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
        length = float(arc[-1])
        if length <= self._min_path_spacing:
            return pos_ref, vel_ref

        query = np.minimum(
            np.arange(self._N + 1, dtype=float) * self._target_speed * self._dt,
            length,
        )

        pos_ref = np.column_stack([np.interp(query, arc, points[:, dim]) for dim in range(3)])
        segment_idx = np.searchsorted(arc, query, side="right") - 1
        segment_idx = np.clip(segment_idx, 0, len(points) - 2)
        segment_vec = points[segment_idx + 1] - points[segment_idx]
        segment_len = np.linalg.norm(segment_vec, axis=1, keepdims=True)
        unit_tangents = segment_vec / np.clip(segment_len, 1e-9, None)
        speed = self._target_speed * np.clip((length - query) / self._brake_dist, 0.0, 1.0)
        vel_ref = unit_tangents * speed[:, None]
        return pos_ref, vel_ref

    def _gate_exit_normal(
        self,
        gate_idx: int,
        gates_pos: NDArray[np.floating],
        gates_quat: NDArray[np.floating],
        fallback_prev: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Orient a gate normal so the exit side points toward the next checkpoint."""
        center = gates_pos[gate_idx]
        normal = R.from_quat(gates_quat[gate_idx]).apply([1.0, 0.0, 0.0])
        normal = normal / np.clip(np.linalg.norm(normal), 1e-9, None)

        if gate_idx + 1 < len(gates_pos):
            desired = gates_pos[gate_idx + 1] - center
        else:
            desired = center - fallback_prev
        if np.dot(desired, normal) < 0.0:
            normal = -normal
        return normal

    def _push_points_away_from_obstacles(
        self,
        points: NDArray[np.floating],
        is_gate_center: NDArray[np.bool_],
        obstacles_pos: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Move non-gate path knots out of obstacle clearance disks in the xy plane."""
        points = points.copy()
        for point_idx in range(1, len(points)):
            if is_gate_center[point_idx]:
                continue
            for obstacle in obstacles_pos:
                delta = points[point_idx, :2] - obstacle[:2]
                dist = float(np.linalg.norm(delta))
                if 1e-6 < dist < self._avoid_radius:
                    points[point_idx, :2] += delta / dist * (self._avoid_radius - dist)
        return points

    def _insert_obstacle_detours(
        self, points: NDArray[np.floating], obstacles_pos: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Insert lateral knots where a segment would skim an observed obstacle."""
        detoured = [points[0]]
        for start, end in zip(points[:-1], points[1:]):
            segment_xy = end[:2] - start[:2]
            segment_len_sq = float(segment_xy @ segment_xy)
            segment_detours = []
            if segment_len_sq > 1e-12:
                for obstacle in obstacles_pos:
                    t = float(
                        np.clip(
                            ((obstacle[:2] - start[:2]) @ segment_xy) / segment_len_sq,
                            0.0,
                            1.0,
                        )
                    )
                    if t <= 0.08 or t >= 0.92:
                        continue
                    closest_xy = start[:2] + t * segment_xy
                    offset = closest_xy - obstacle[:2]
                    dist = float(np.linalg.norm(offset))
                    if dist >= self._avoid_radius:
                        continue
                    if dist < 1e-6:
                        offset = np.array([-segment_xy[1], segment_xy[0]], dtype=float)
                        dist = float(np.linalg.norm(offset))
                    detour = start + t * (end - start)
                    detour[:2] = (
                        obstacle[:2] + offset / np.clip(dist, 1e-9, None) * self._avoid_radius
                    )
                    segment_detours.append((t, detour))

            for _, detour in sorted(segment_detours, key=lambda item: item[0]):
                if np.linalg.norm(detour - detoured[-1]) > self._min_path_spacing:
                    detoured.append(detour)
            if np.linalg.norm(end - detoured[-1]) > self._min_path_spacing:
                detoured.append(end)
        return np.asarray(detoured, dtype=float)

    def _build_linearization_function(self) -> None:
        """Create CasADi functions for the same continuous model used by the nonlinear MPC."""
        x_dot, x, u, _ = symbolic_dynamics_euler(
            mass=self.drone_params["mass"],
            gravity_vec=self.drone_params["gravity_vec"],
            J=self.drone_params["J"],
            J_inv=self.drone_params["J_inv"],
            acc_coef=self.drone_params["acc_coef"],
            cmd_f_coef=self.drone_params["cmd_f_coef"],
            rpy_coef=self.drone_params["rpy_coef"],
            rpy_rates_coef=self.drone_params["rpy_rates_coef"],
            cmd_rpy_coef=self.drone_params["cmd_rpy_coef"],
        )
        a = ca.jacobian(x_dot, x)
        b = ca.jacobian(x_dot, u)
        self._linearization_fun = ca.Function("so_rpy_linearization", [x, u], [x_dot, a, b])

    def _current_state(self, obs: dict[str, NDArray[np.floating]]) -> NDArray[np.floating]:
        """Return the current 12D ``so_rpy`` state: pos, rpy, vel, rpy rates."""
        quat = np.asarray(obs["quat"], dtype=float)
        rpy = R.from_quat(quat).as_euler("xyz")
        drpy = ang_vel2rpy_rates(quat, np.asarray(obs["ang_vel"], dtype=float))
        return np.concatenate(
            (
                np.asarray(obs["pos"], dtype=float),
                rpy,
                np.asarray(obs["vel"], dtype=float),
                np.asarray(drpy, dtype=float),
            )
        )

    def _build_reference_states(
        self, pos_ref: NDArray[np.floating], vel_ref: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Build the ``so_rpy`` state reference for all horizon knots."""
        x_ref = np.zeros((self._N + 1, self._nx), dtype=float)
        x_ref[:, 0:3] = pos_ref
        x_ref[:, 6:9] = vel_ref
        return x_ref

    def _build_reference_inputs(self) -> NDArray[np.floating]:
        """Use upright attitude and hover thrust as the input linearization/reference."""
        u_ref = np.zeros((self._N, self._nu), dtype=float)
        u_ref[:, 3] = self._hover_thrust
        return u_ref

    def _build_dynamics_sequence(
        self, x_ref: NDArray[np.floating], u_ref: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """Linearize and discretize the ``so_rpy`` model along the reference."""
        a_seq = np.zeros((self._N, self._nx, self._nx), dtype=float)
        b_seq = np.zeros((self._N, self._nx, self._nu), dtype=float)
        c_seq = np.zeros((self._N, self._nx), dtype=float)

        for step in range(self._N):
            f_cont, a_cont, b_cont = self._linearization_fun(x_ref[step], u_ref[step])
            f_cont = np.asarray(f_cont, dtype=float).reshape(self._nx)
            a_cont = np.asarray(a_cont, dtype=float)
            b_cont = np.asarray(b_cont, dtype=float)

            a_d = np.eye(self._nx, dtype=float) + self._dt * a_cont
            b_d = self._dt * b_cont
            c_d = self._dt * (f_cont - a_cont @ x_ref[step] - b_cont @ u_ref[step])

            a_seq[step] = a_d
            b_seq[step] = b_d
            c_seq[step] = c_d
        return a_seq, b_seq, c_seq

    def _build_prediction_matrices(
        self,
        a_seq: NDArray[np.floating],
        b_seq: NDArray[np.floating],
        c_seq: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """Build stacked prediction matrices for affine time-varying linear dynamics."""
        nz = a_seq.shape[1]
        nu = b_seq.shape[2]
        sx = np.zeros((self._N * nz, nz), dtype=float)
        su = np.zeros((self._N * nz, self._N * nu), dtype=float)
        soff = np.zeros(self._N * nz, dtype=float)

        sx_step = np.eye(nz, dtype=float)
        su_step = np.zeros((nz, self._N * nu), dtype=float)
        offset = np.zeros(nz, dtype=float)
        for step in range(self._N):
            sx_step = a_seq[step] @ sx_step
            su_step = a_seq[step] @ su_step
            su_step[:, step * nu : (step + 1) * nu] += b_seq[step]
            offset = a_seq[step] @ offset + c_seq[step]

            row = slice(step * nz, (step + 1) * nz)
            sx[row, :] = sx_step
            su[row, :] = su_step
            soff[row] = offset
        return sx, su, soff

    def _build_block_diagonal_state_weights(self) -> NDArray[np.floating]:
        nz = self._q.shape[0]
        q_bar = np.zeros((self._N * nz, self._N * nz), dtype=float)
        for step in range(self._N):
            row = slice(step * nz, (step + 1) * nz)
            q_bar[row, row] = self._p if step == self._N - 1 else self._q
        return q_bar

    def _build_block_diagonal_input_weights(
        self, weights: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        nu = weights.shape[0]
        r_bar = np.zeros((self._N * nu, self._N * nu), dtype=float)
        for step in range(self._N):
            row = slice(step * nu, (step + 1) * nu)
            r_bar[row, row] = weights
        return r_bar

    def _build_input_rate_matrix(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Return ``D, d`` so ``D @ U + d`` is the stacked input-rate vector."""
        size = self._N * self._nu
        d_rate = np.eye(size, dtype=float)
        for step in range(1, self._N):
            row = slice(step * self._nu, (step + 1) * self._nu)
            prev_col = slice((step - 1) * self._nu, step * self._nu)
            d_rate[row, prev_col] = -np.eye(self._nu)
        d_rate_offset = np.zeros(size, dtype=float)
        d_rate_offset[: self._nu] = -self._last_action
        return d_rate, d_rate_offset

    def _clip_action_step(self, action: NDArray[np.floating]) -> NDArray[np.floating]:
        """Clip solver output to attitude/thrust limits and per-step slew limits."""
        action = np.asarray(action, dtype=float).copy()
        delta = action - self._last_action
        delta[0:2] = np.clip(delta[0:2], -self._max_tilt_step, self._max_tilt_step)
        delta[2] = np.clip(delta[2], -self._max_tilt_step, self._max_tilt_step)
        delta[3] = np.clip(delta[3], -self._max_thrust_step, self._max_thrust_step)
        action = self._last_action + delta
        action[0:2] = np.clip(action[0:2], -self._max_tilt, self._max_tilt)
        action[2] = np.clip(action[2], -self._max_yaw, self._max_yaw)
        action[3] = np.clip(action[3], self._thrust_min, self._thrust_max)
        return action

    def _pd_fallback(
        self,
        obs: dict[str, NDArray[np.floating]],
        pos_ref: NDArray[np.floating],
        vel_ref: NDArray[np.floating],
        reason: str,
    ) -> NDArray[np.floating]:
        pos_error = np.asarray(pos_ref, dtype=float) - np.asarray(obs["pos"], dtype=float)
        vel_error = np.asarray(vel_ref, dtype=float) - np.asarray(obs["vel"], dtype=float)
        accel_cmd = np.array([1.2, 1.2, 2.2]) * pos_error + np.array([0.45, 0.45, 0.8]) * vel_error

        action = np.array(
            [
                -accel_cmd[1] / self._g,
                accel_cmd[0] / self._g,
                0.0,
                self._hover_thrust + self._mass * accel_cmd[2],
            ],
            dtype=float,
        )
        action = self._clip_action_step(action)
        self._last_action = action.copy()
        self._last_log = {
            "controller": "linearized_attitude_mpc",
            "solver": "fallback",
            "fallback_reason": reason,
            "action": action.copy(),
        }
        return action
