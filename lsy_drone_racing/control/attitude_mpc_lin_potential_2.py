"""MPC using attitude control with successive linearization, APF obstacle avoidance,
and a collision-aware min-snap reference trajectory.

Identical to attitude_mpc_lin_potential.py except the reference trajectory is generated
by _MinSnapPlanner from threeb_controller.py rather than three_b_one_s_controller.py.
The newer planner runs at PLAN_SPEED = 1.5 m/s, supports return_to_start, and
re-plans at runtime (fast path, no L-BFGS) whenever a not-yet-passed gate's observed
pose changes by more than REPLAN_POS_THRESH metres.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import time

import casadi as cs
import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points
from drone_models.core import load_params
from drone_models.so_rpy import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.threeb_controller import _MinSnapPlanner

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


def rk4_step(f: cs.Function, x: cs.MX, u: cs.MX, dt: float) -> cs.MX:
    k1 = f(x, u)
    k2 = f(x + (dt / 2) * k1, u)
    k3 = f(x + (dt / 2) * k2, u)
    k4 = f(x + dt * k3, u)
    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


class AttitudeMPC(Controller):
    """Attitude MPC with RTI, APF obstacle avoidance, and min-snap reference trajectory.

    Each call to compute_control:
      1. Checks if a not-yet-passed gate has moved; re-plans if so (fast, no L-BFGS).
      2. Linearises the RK4 step around the current warm-start (X_ws, U_ws).
      3. Perturbs the N-step reference horizon with APF repulsion from obstacles.
      4. Solves the resulting QP.
      5. Shifts the solution forward as the next warm-start.
    """

    # --- Planner parameters ---------------------------------------------------
    PLAN_SPEED: float = 1.5          # cruise speed for time allocation [m/s]
    PLAN_A_LIMIT: float = 12.0       # peak acceleration cap [m/s²]
    PLAN_SAFE_MARGIN: float = 0.05   # planner safety margin [m]; smaller than threeb default (0.12)
    PLAN_GATE_LEAD: float = 0.40     # pre/post waypoint distance from gate center [m]
    REPLAN_POS_THRESH: float = 0.04  # re-plan when a remaining gate moves more than this [m]

    # --- APF parameters -------------------------------------------------------
    _APF_POTENTIAL: str = "khatib"
    _APF_RADIUS: float = 0.1
    _APF_GATE_RADIUS: float = 0.2
    _APF_K_REP: float = 0.08
    _APF_MAX_DISP: float = 0.3

    # --- Gate geometry --------------------------------------------------------
    _GATE_OUTER_HALF: float = 0.36
    _GATE_INNER_HALF: float = 0.20
    _GATE_SAMPLES_PER_SIDE: int = 8

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self._N = 23
        self._dt = 1 / config.env.freq
        self._freq = config.env.freq

        # --- Trajectory planning ----------------------------------------------
        # SAFE_MARGIN is reduced vs the threeb default (0.12 m): the planner's gate-opening
        # constraint allows only (GATE_OPENING_HALF - DRONE_RADIUS - SAFE_MARGIN) off-centre
        # deviation. At 0.12 m margin the allowance is 2 cm — unreachably tight for the spline
        # optimizer. At 0.05 m it is 9 cm, which the optimizer can actually achieve. The MPC's
        # APF layer handles fine-grained frame repulsion at runtime so tight planner margins are
        # not needed here.
        _MinSnapPlanner.SPEED = self.PLAN_SPEED
        _MinSnapPlanner.A_LIMIT = self.PLAN_A_LIMIT
        _MinSnapPlanner.SAFE_MARGIN = self.PLAN_SAFE_MARGIN
        _MinSnapPlanner.GATE_LEAD = self.PLAN_GATE_LEAD
        self._planner = _MinSnapPlanner(obs)
        self._plan_gates = np.asarray(obs["gates_pos"], dtype=float).copy()
        self._refresh_waypoints()

        # --- Drone model & CasADi dynamics ------------------------------------
        params = load_params("so_rpy", config.sim.drone_model)
        self.drone_params = params

        X_dot, X_sym, U_sym, _ = symbolic_dynamics_euler(
            mass=params["mass"],
            gravity_vec=params["gravity_vec"],
            J=params["J"],
            J_inv=params["J_inv"],
            acc_coef=params["acc_coef"],
            cmd_f_coef=params["cmd_f_coef"],
            rpy_coef=params["rpy_coef"],
            rpy_rates_coef=params["rpy_rates_coef"],
            cmd_rpy_coef=params["cmd_rpy_coef"],
        )
        self._nx = X_sym.shape[0]  # 12
        self._nu = U_sym.shape[0]  # 4

        f_dyn = cs.Function("f_dyn", [X_sym, U_sym], [X_dot])

        x_s = cs.MX.sym("x", self._nx)
        u_s = cs.MX.sym("u", self._nu)
        rk4_expr = rk4_step(f_dyn, x_s, u_s, self._dt)
        self._f_rk4 = cs.Function("f_rk4", [x_s, u_s], [rk4_expr])
        self._jac_x = cs.Function("jac_x", [x_s, u_s], [cs.jacobian(rk4_expr, x_s)])
        self._jac_u = cs.Function("jac_u", [x_s, u_s], [cs.jacobian(rk4_expr, u_s)])

        self._f_rk4_map = self._f_rk4.map(self._N)
        self._jac_x_map = self._jac_x.map(self._N)
        self._jac_u_map = self._jac_u.map(self._N)

        Q  = np.diag([126.68, 126.68, 55.47, 2.71, 2.71, 2.71, 4.42, 4.42, 4.42, 0.10, 0.10, 0.10])
        Rw = np.diag([1.66, 1.66, 1.66, 41.25])
        W = np.block(
            [[Q, np.zeros((self._nx, self._nu))], [np.zeros((self._nu, self._nx)), Rw]]
        )

        self._hover_thrust = float(params["mass"] * abs(params["gravity_vec"][-1]))
        self._u_lb = np.array([-0.8, -0.8, -0.8, params["thrust_min"] * 4])
        self._u_ub = np.array([0.8, 0.8, 0.8, params["thrust_max"] * 4])

        (
            self._opti,
            self._X_var,
            self._U_var,
            self._x0_p,
            self._yref_p,
            self._yref_e_p,
            self._A_p,
            self._B_p,
            self._c_p,
        ) = self._build_opti(W, Q)

        self._X_ws = np.zeros((self._nx, self._N + 1))
        self._U_ws = np.zeros((self._nu, self._N))
        self._U_ws[3, :] = self._hover_thrust
        self._C_ws = np.zeros((self._nx, self._N))

        self._tick = 0
        self._finished = False

        self._debug_apf: bool = False
        self._compute_times: list[float] = []

        self._obstacles_pos: np.ndarray = np.zeros((0, 3))
        self._gate_frame_points: np.ndarray = np.zeros((0, 3))
        self._last_gates_pos: np.ndarray | None = None
        self._last_gates_quat: np.ndarray | None = None
        self._horizon_displaced: np.ndarray = np.zeros((self._N, 3))

    # --------------------------------------------------------------------------
    # Trajectory management
    # --------------------------------------------------------------------------

    def _refresh_waypoints(self) -> None:
        """Sample the planner spline at env frequency and update _tick_max."""
        n_steps = int(np.ceil(self._planner.t_total * self._freq))
        t_eval = np.linspace(0, self._planner.t_total, n_steps)
        self._waypoints_pos = self._planner.spline(t_eval)
        self._waypoints_vel = self._planner.spline.derivative(1)(t_eval)
        self._waypoints_yaw = np.zeros(n_steps)
        self._tick_max = n_steps - 1 - self._N

    def _maybe_replan(self, obs: dict[str, NDArray[np.floating]]) -> None:
        """Re-plan from the current position if a not-yet-passed gate has moved."""
        tg = int(obs["target_gate"])
        if tg < 0:
            return
        gates_pos = np.asarray(obs["gates_pos"], dtype=float)
        moved = np.linalg.norm(gates_pos[tg:] - self._plan_gates[tg:], axis=1)
        if moved.size == 0 or moved.max() <= self.REPLAN_POS_THRESH:
            return

        self._planner.replan(
            start_pos=np.asarray(obs["pos"]).squeeze(),
            first_gate=tg,
            gates_pos=gates_pos,
            gates_quat=np.asarray(obs["gates_quat"]),
            obstacles_pos=np.asarray(obs["obstacles_pos"]),
            optimize=False,  # fast seed + clearance repair; safe inside the 50 Hz loop
        )
        self._plan_gates = gates_pos.copy()
        self._refresh_waypoints()
        # Restart tick and reset warm-start to hover so the MPC re-converges cleanly.
        self._tick = 0
        self._X_ws[:] = 0.0
        self._U_ws[:] = 0.0
        self._U_ws[3, :] = self._hover_thrust
        # Invalidate gate frame cache so APF recomputes against updated positions.
        self._last_gates_pos = None
        self._last_gates_quat = None

    # --------------------------------------------------------------------------
    # Opti setup
    # --------------------------------------------------------------------------

    def _build_opti(self, W: np.ndarray, Q_e: np.ndarray):
        N, nx, nu = self._N, self._nx, self._nu
        ny = nx + nu
        rpy_limit = 0.5

        opti = cs.Opti()
        X = opti.variable(nx, N + 1)
        U = opti.variable(nu, N)
        x0_p = opti.parameter(nx)
        yref_p = opti.parameter(ny, N)
        yref_e_p = opti.parameter(nx)

        A_p = opti.parameter(nx, nx * N)
        B_p = opti.parameter(nx, nu * N)
        c_p = opti.parameter(nx, N)

        W_dm = cs.DM(W)
        Qe_dm = cs.DM(Q_e)

        cost = 0
        for j in range(N):
            e = cs.vertcat(X[:, j], U[:, j]) - yref_p[:, j]
            cost += cs.dot(e, cs.mtimes(W_dm, e))
        e_e = X[:, N] - yref_e_p
        cost += cs.dot(e_e, cs.mtimes(Qe_dm, e_e))
        opti.minimize(cost)

        opti.subject_to(X[:, 0] == x0_p)
        for j in range(N):
            A_j = A_p[:, j * nx : (j + 1) * nx]
            B_j = B_p[:, j * nu : (j + 1) * nu]
            opti.subject_to(
                X[:, j + 1] == cs.mtimes(A_j, X[:, j]) + cs.mtimes(B_j, U[:, j]) + c_p[:, j]
            )

        opti.subject_to(cs.vec(X[3:6, 1:N]) >= -rpy_limit)
        opti.subject_to(cs.vec(X[3:6, 1:N]) <= rpy_limit)

        u_lb_dm = cs.repmat(cs.DM(self._u_lb.reshape(-1, 1)), 1, N)
        u_ub_dm = cs.repmat(cs.DM(self._u_ub.reshape(-1, 1)), 1, N)
        opti.subject_to(cs.vec(U) >= cs.vec(u_lb_dm))
        opti.subject_to(cs.vec(U) <= cs.vec(u_ub_dm))

        p_opts = {"print_time": 0}
        s_opts = {"max_iter": 20, "tol": 1e-4, "print_level": 0, "warm_start_init_point": "yes"}
        opti.solver("ipopt", p_opts, s_opts)

        return opti, X, U, x0_p, yref_p, yref_e_p, A_p, B_p, c_p

    # --------------------------------------------------------------------------
    # Potential field
    # --------------------------------------------------------------------------

    def _generate_gate_frame_points(
        self, gate_centers: np.ndarray, gate_quats: np.ndarray
    ) -> np.ndarray:
        half = self._GATE_OUTER_HALF
        n = self._GATE_SAMPLES_PER_SIDE
        t = np.linspace(-half, half, n)
        half_in = self._GATE_INNER_HALF
        t_in = np.linspace(-half_in, half_in, n)
        blocks: list[np.ndarray] = []

        for center, quat in zip(gate_centers, gate_quats):
            rot = R.from_quat(quat)
            h_axis = rot.apply([0.0, 1.0, 0.0])
            h_axis[2] = 0.0
            h_norm = float(np.linalg.norm(h_axis))
            if h_norm < 1e-6:
                continue
            h_axis /= h_norm
            v_axis = np.array([0.0, 0.0, 1.0])

            outer = np.vstack([
                center + t[:, None] * h_axis + half * v_axis,
                center + t[:, None] * h_axis - half * v_axis,
                center - half * h_axis + t[1:-1, None] * v_axis,
                center + half * h_axis + t[1:-1, None] * v_axis,
            ])
            inner = np.vstack([
                center + t_in[:, None] * h_axis + half_in * v_axis,
                center + t_in[:, None] * h_axis - half_in * v_axis,
                center - half_in * h_axis + t_in[1:-1, None] * v_axis,
                center + half_in * h_axis + t_in[1:-1, None] * v_axis,
            ])
            blocks.append(outer)
            blocks.append(inner)

        return np.vstack(blocks) if blocks else np.zeros((0, 3))

    def _apply_apf(
        self,
        yref: np.ndarray,
        yref_e: np.ndarray,
        cyl_obstacles: np.ndarray,
        gate_points: np.ndarray,
    ) -> None:
        refs = np.vstack([yref[0:3, :].T, yref_e[0:3][None]])
        disp = np.zeros_like(refs)

        if cyl_obstacles.shape[0] > 0:
            diff_xy = refs[:, None, :2] - cyl_obstacles[None, :, :2]
            dists = np.linalg.norm(diff_xy, axis=-1)
            in_range = (dists > 1e-6) & (dists < self._APF_RADIUS)
            safe_d = np.where(in_range, dists, 1.0)
            if self._APF_POTENTIAL == "log_barrier":
                mags = np.where(in_range, self._APF_K_REP / safe_d, 0.0)
            else:
                mags = np.where(
                    in_range,
                    self._APF_K_REP * (1.0 / safe_d - 1.0 / self._APF_RADIUS) / safe_d**2,
                    0.0,
                )
            disp[:, :2] += (mags[:, :, None] * diff_xy / safe_d[:, :, None]).sum(axis=1)

        if gate_points.shape[0] > 0:
            diff = refs[:, None, :] - gate_points[None, :, :]
            dists = np.linalg.norm(diff, axis=-1)
            in_range = (dists > 1e-6) & (dists < self._APF_GATE_RADIUS)
            safe_d = np.where(in_range, dists, 1.0)
            if self._APF_POTENTIAL == "log_barrier":
                mags = np.where(in_range, self._APF_K_REP / safe_d, 0.0)
            else:
                mags = np.where(
                    in_range,
                    self._APF_K_REP * (1.0 / safe_d - 1.0 / self._APF_GATE_RADIUS) / safe_d**2,
                    0.0,
                )
            disp += (mags[:, :, None] * diff / safe_d[:, :, None]).sum(axis=1)

        d_total = np.linalg.norm(disp, axis=-1, keepdims=True)
        scale = np.where(d_total > self._APF_MAX_DISP, self._APF_MAX_DISP / d_total, 1.0)
        disp *= scale

        yref[0:3, :] += disp[: self._N, :].T
        yref_e[0:3] += disp[self._N, :]

    # --------------------------------------------------------------------------

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        _t0 = time.perf_counter()
        result = self._compute_control_impl(obs, info)
        self._compute_times.append(time.perf_counter() - _t0)
        return result

    def _compute_control_impl(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        self._maybe_replan(obs)

        i = min(self._tick, self._tick_max)
        if self._tick >= self._tick_max:
            self._finished = True

        quat = np.asarray(obs["quat"]).squeeze()
        ang_vel = np.asarray(obs["ang_vel"]).squeeze()
        pos = np.asarray(obs["pos"]).squeeze()
        vel = np.asarray(obs["vel"]).squeeze()

        rpy = R.from_quat(quat).as_euler("xyz")
        drpy = ang_vel2rpy_rates(quat, ang_vel)
        x0 = np.concatenate((pos, rpy, vel, drpy))

        N, nx, nu = self._N, self._nx, self._nu
        X_bar = self._X_ws[:, :N]

        A_all = np.asarray(self._jac_x_map(X_bar, self._U_ws))
        B_all = np.asarray(self._jac_u_map(X_bar, self._U_ws))
        F_all = np.asarray(self._f_rk4_map(X_bar, self._U_ws))

        for j in range(N):
            A_j = A_all[:, j * nx : (j + 1) * nx]
            B_j = B_all[:, j * nu : (j + 1) * nu]
            self._C_ws[:, j] = F_all[:, j] - A_j @ X_bar[:, j] - B_j @ self._U_ws[:, j]

        self._opti.set_value(self._A_p, A_all)
        self._opti.set_value(self._B_p, B_all)
        self._opti.set_value(self._c_p, self._C_ws)
        self._opti.set_value(self._x0_p, x0)

        ny = nx + nu
        yref = np.zeros((ny, N))
        yref[0:3, :] = self._waypoints_pos[i : i + N].T
        yref[5, :] = self._waypoints_yaw[i : i + N]
        yref[6:9, :] = self._waypoints_vel[i : i + N].T
        yref[nx + 3, :] = self._hover_thrust

        yref_e = np.zeros(nx)
        yref_e[0:3] = self._waypoints_pos[i + N]
        yref_e[5] = self._waypoints_yaw[i + N]
        yref_e[6:9] = self._waypoints_vel[i + N]

        raw_obs_pos = obs.get("obstacles_pos")
        if raw_obs_pos is not None:
            self._obstacles_pos = np.asarray(raw_obs_pos).reshape(-1, 3)

        raw_gates_pos = obs.get("gates_pos")
        raw_gates_quat = obs.get("gates_quat")
        if raw_gates_pos is not None and raw_gates_quat is not None:
            new_pos = np.asarray(raw_gates_pos).reshape(-1, 3)
            new_quat = np.asarray(raw_gates_quat).reshape(-1, 4)
            if (
                self._last_gates_pos is None
                or not np.array_equal(new_pos, self._last_gates_pos)
                or not np.array_equal(new_quat, self._last_gates_quat)
            ):
                self._gate_frame_points = self._generate_gate_frame_points(new_pos, new_quat)
                self._last_gates_pos = new_pos
                self._last_gates_quat = new_quat

        yref_pos_before = yref[0:3, :].copy()
        if self._obstacles_pos.shape[0] > 0 or self._gate_frame_points.shape[0] > 0:
            self._apply_apf(yref, yref_e, self._obstacles_pos, self._gate_frame_points)

        if self._debug_apf:
            disp_norms = np.linalg.norm(yref[0:3, :] - yref_pos_before, axis=0)
            max_disp = float(disp_norms.max())
            if max_disp > 1e-4:
                print(
                    f"[APF] tick={self._tick:4d}  max_disp={max_disp:.4f} m  "
                    f"at horizon step {int(disp_norms.argmax())}  "
                    f"n_obs={self._obstacles_pos.shape[0]}  "
                    f"n_gate_pts={self._gate_frame_points.shape[0]}"
                )

        self._horizon_displaced = yref[0:3, :].T.copy()

        self._opti.set_value(self._yref_p, yref)
        self._opti.set_value(self._yref_e_p, yref_e)

        self._opti.set_initial(self._X_var, self._X_ws)
        self._opti.set_initial(self._U_var, self._U_ws)

        try:
            sol = self._opti.solve()
            X_sol = np.asarray(sol.value(self._X_var))
            U_sol = np.asarray(sol.value(self._U_var))
            self._X_ws[:, :-1] = X_sol[:, 1:]
            self._X_ws[:, -1] = X_sol[:, -1]
            self._U_ws[:, :-1] = U_sol[:, 1:]
            self._U_ws[:, -1] = U_sol[:, -1]
            return U_sol[:, 0].copy()
        except RuntimeError:
            return self._U_ws[:, 0].copy()

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        self._tick += 1
        return self._finished

    def render_callback(self, sim: Sim):
        i = min(self._tick, self._tick_max)
        draw_line(sim, self._waypoints_pos[::3], rgba=(0.0, 1.0, 0.0, 1.0))
        if self._horizon_displaced.shape[0] > 0:
            draw_points(sim, self._horizon_displaced[::3], rgba=(1.0, 0.5, 0.0, 1.0), size=0.015)
        if self._gate_frame_points.shape[0] > 0:
            draw_points(sim, self._gate_frame_points, rgba=(1.0, 1.0, 0.0, 0.8), size=0.012)
        setpoint = self._waypoints_pos[i].reshape(1, -1)
        draw_points(sim, setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)

    def episode_callback(self):
        if self._compute_times:
            t = np.array(self._compute_times) * 1e3
            budget = 1000.0 / 50
            over = np.sum(t > budget)
            print(
                f"[timing] compute_control over {len(t)} calls: "
                f"mean={t.mean():.2f} ms  "
                f"p95={np.percentile(t, 95):.2f} ms  "
                f"max={t.max():.2f} ms  "
                f"over_budget={over}/{len(t)} (>{budget:.0f} ms)"
            )
            self._compute_times.clear()
        self._tick = 0
