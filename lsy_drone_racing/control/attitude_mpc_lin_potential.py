"""MPC using attitude control with successive linearization and local APF obstacle avoidance.

Identical to attitude_mpc_lin.py except that before each MPC solve the N-step reference
horizon is perturbed by an Artificial Potential Field (APF) repulsion computed from the
known obstacle positions.  The MPC formulation itself is unchanged — only the reference
passed to the QP is modified, so there is zero solver overhead.

APF repulsion acts in the horizontal (x-y) plane only, reflecting the fact that obstacles
are vertical cylinders.  A displacement is added to every column of yref[0:3, :] and to
the terminal reference yref_e[0:3].  The magnitude is clamped to _apf_max_disp so the
reference never moves so far that the QP becomes infeasible.
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
from lsy_drone_racing.control.three_b_one_s_controller import make_planner

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
    """Attitude MPC with RTI and local APF obstacle avoidance.

    Each call to compute_control:
      1. Linearises the RK4 step around the current warm-start (X_ws, U_ws).
      2. Perturbs the N-step reference horizon with APF repulsion from obstacles.
      3. Solves the resulting QP.
      4. Shifts the solution forward as the next warm-start.
    """

    # --- APF parameters -------------------------------------------------------
    # Potential function to use: "khatib" or "log_barrier".
    #   "khatib"      — mag = k*(1/d - 1/d₀)/d²  (smooth onset, zero at d₀, grows as 1/d³)
    #   "log_barrier" — mag = k/d                 (always nonzero inside d₀, grows as 1/d)
    _APF_POTENTIAL: str = "khatib"
    # Cylindrical obstacles — influence radius (horizontal distance only).
    _APF_RADIUS: float = 0.1      # metres
    # Gate frame points — influence radius (3-D distance).
    # Smaller than cylinder radius: gate bars are thin and we only want local repulsion.
    _APF_GATE_RADIUS: float = 0.15  # metres
    # Repulsive gain shared by both obstacle types.
    _APF_K_REP: float = 0.08
    # Maximum displacement applied to any single reference point.
    _APF_MAX_DISP: float = 0.3    # metres

    # VASEK VERSION: RE-ADJUST WEIGHTS
    _APF_RADIUS: float = 0.1
    _APF_GATE_RADIUS: float = 0.1
    _APF_K_REP: float = 0.03
    _APF_MAX_DISP: float = 0.1

    # --- Gate geometry --------------------------------------------------------
    # Half-width of the outer gate frame (full outer size = 0.72 m).
    _GATE_OUTER_HALF: float = 0.36   # metres
    # Half-width of the inner gate opening (full opening = 0.40 m).
    _GATE_INNER_HALF: float = 0.20   # metres
    # Number of sample points per edge of the gate frame (both outer and inner).
    _GATE_SAMPLES_PER_SIDE: int = 8

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self._N = 23
        self._dt = 1 / config.env.freq

        self._planner = make_planner(freq=config.env.freq)
        self._planner.replan(
            start_pos=np.asarray(obs["pos"]).squeeze(),
            first_gate=0,
            gates_pos=np.asarray(obs["gates_pos"]),
            gates_quat=np.asarray(obs["gates_quat"]),
            obstacles_pos=np.asarray(obs["obstacles_pos"]),
            optimize=True,
        )
        self._t_total = self._planner.t_total
        n_steps = int(np.ceil(self._t_total * config.env.freq))
        t_eval = np.linspace(0, self._t_total, n_steps)
        self._waypoints_pos = self._planner.spline(t_eval)
        self._waypoints_vel = self._planner.spline.derivative(1)(t_eval)
        self._waypoints_yaw = np.zeros(n_steps)

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

        # Batched versions: evaluate all N warm-start points in one CasADi call.
        # map(N) output layout: (nx, nx*N) / (nx, nu*N) / (nx, N) — j-th block at [:, j*k:(j+1)*k]
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
        self._C_ws = np.zeros((self._nx, self._N))  # affine offsets c_j = f_j - A_j x_j - B_j u_j

        self._tick = 0
        self._tick_max = len(self._waypoints_pos) - 1 - self._N
        self._finished = False

        self._debug_apf: bool = False  # set True to print displacement info each step
        self._compute_times: list[float] = []

        # Obstacle positions updated every compute_control step; used by render_callback.
        self._obstacles_pos: np.ndarray = np.zeros((0, 3))
        # Gate frame sample points; only regenerated when gate obs changes.
        self._gate_frame_points: np.ndarray = np.zeros((0, 3))
        self._last_gates_pos: np.ndarray | None = None
        self._last_gates_quat: np.ndarray | None = None
        # Displaced horizon reference positions saved for visualisation.
        self._horizon_displaced: np.ndarray = np.zeros((self._N, 3))

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

        # Single wide parameter matrices instead of N separate parameters.
        # Layout: A_p[:, j*nx:(j+1)*nx] is the j-th linearised A matrix.
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
        """Sample points along the outer and inner edges of each gate's square frame.

        The drone crosses each gate along its local x-axis (R.apply([1,0,0])).
        The gate width therefore spans the local y-axis (R.apply([0,1,0])) and
        height spans z.  Two concentric square rings are sampled per gate:
          - Outer ring (0.72 m): repels from outside the solid frame.
          - Inner ring (0.40 m): acts as a funnel toward the opening centre.

        Args:
            gate_centers: (n, 3) gate centre positions in world frame.
            gate_quats:   (n, 4) gate quaternions in xyzw order.

        Returns:
            (m, 3) array of sample points, or empty (0, 3) if no gates given.
        """
        half = self._GATE_OUTER_HALF
        n = self._GATE_SAMPLES_PER_SIDE
        t = np.linspace(-half, half, n)          # (n,)
        half_in = self._GATE_INNER_HALF
        t_in = np.linspace(-half_in, half_in, n) # (n,)
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

            # Outer perimeter — vectorised (no Python element loop)
            outer = np.vstack([
                center + t[:, None] * h_axis + half * v_axis,      # top bar   (n, 3)
                center + t[:, None] * h_axis - half * v_axis,      # bottom bar
                center - half * h_axis + t[1:-1, None] * v_axis,   # left bar (no corners)
                center + half * h_axis + t[1:-1, None] * v_axis,   # right bar
            ])
            # Inner perimeter — same pattern
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
        """Perturb the position columns of yref and yref_e in-place.

        All N+1 reference points are processed in a single vectorised numpy pass
        rather than a Python loop, eliminating per-point interpreter overhead.
        """
        # Stack horizon + terminal into one batch: (N+1, 3)
        refs = np.vstack([yref[0:3, :].T, yref_e[0:3][None]])
        disp = np.zeros_like(refs)  # (N+1, 3)

        # --- Cylindrical obstacles: horizontal (x-y) distance only -----------
        if cyl_obstacles.shape[0] > 0:
            diff_xy = refs[:, None, :2] - cyl_obstacles[None, :, :2]  # (N+1, n_obs, 2)
            dists = np.linalg.norm(diff_xy, axis=-1)                   # (N+1, n_obs)
            in_range = (dists > 1e-6) & (dists < self._APF_RADIUS)
            safe_d = np.where(in_range, dists, 1.0)
            if self._APF_POTENTIAL == "log_barrier":
                # mag = k/d  →  force = k/d * unit_vec = k*(x-obs)/d²
                mags = np.where(in_range, self._APF_K_REP / safe_d, 0.0)
            else:  # "khatib"
                # mag = k*(1/d - 1/d₀)/d²  →  zero at d₀, grows as 1/d³
                mags = np.where(
                    in_range,
                    self._APF_K_REP * (1.0 / safe_d - 1.0 / self._APF_RADIUS) / safe_d**2,
                    0.0,
                )
            disp[:, :2] += (mags[:, :, None] * diff_xy / safe_d[:, :, None]).sum(axis=1)

        # --- Gate frame points: full 3-D distance ----------------------------
        if gate_points.shape[0] > 0:
            diff = refs[:, None, :] - gate_points[None, :, :]         # (N+1, m, 3)
            dists = np.linalg.norm(diff, axis=-1)                      # (N+1, m)
            in_range = (dists > 1e-6) & (dists < self._APF_GATE_RADIUS)
            safe_d = np.where(in_range, dists, 1.0)
            if self._APF_POTENTIAL == "log_barrier":
                mags = np.where(in_range, self._APF_K_REP / safe_d, 0.0)
            else:  # "khatib"
                mags = np.where(
                    in_range,
                    self._APF_K_REP * (1.0 / safe_d - 1.0 / self._APF_GATE_RADIUS) / safe_d**2,
                    0.0,
                )
            disp += (mags[:, :, None] * diff / safe_d[:, :, None]).sum(axis=1)

        # Clamp each displacement vector independently
        d_total = np.linalg.norm(disp, axis=-1, keepdims=True)      # (N+1, 1)
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
        X_bar = self._X_ws[:, :N]                                    # (nx, N)

        # Three batched CasADi calls instead of 3×N sequential ones.
        A_all = np.asarray(self._jac_x_map(X_bar, self._U_ws))      # (nx, nx*N)
        B_all = np.asarray(self._jac_u_map(X_bar, self._U_ws))      # (nx, nu*N)
        F_all = np.asarray(self._f_rk4_map(X_bar, self._U_ws))      # (nx, N)

        # Affine offsets: c_j = f_j - A_j x_j - B_j u_j  (pure numpy, no CasADi)
        for j in range(N):
            A_j = A_all[:, j * nx : (j + 1) * nx]
            B_j = B_all[:, j * nu : (j + 1) * nu]
            self._C_ws[:, j] = F_all[:, j] - A_j @ X_bar[:, j] - B_j @ self._U_ws[:, j]

        # Three set_value calls instead of 3×N.
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

        # --- APF: perturb reference away from obstacles and gate frames -------
        raw_obs_pos = obs.get("obstacles_pos")
        if raw_obs_pos is not None:
            self._obstacles_pos = np.asarray(raw_obs_pos).reshape(-1, 3)

        raw_gates_pos = obs.get("gates_pos")
        raw_gates_quat = obs.get("gates_quat")  # xyzw quaternions
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
            disp_norms = np.linalg.norm(yref[0:3, :] - yref_pos_before, axis=0)  # (N,)
            max_disp = float(disp_norms.max())
            if max_disp > 1e-4:
                max_j = int(disp_norms.argmax())
                print(
                    f"[APF] tick={self._tick:4d}  "
                    f"max_disp={max_disp:.4f} m  at horizon step {max_j}  "
                    f"n_obs={self._obstacles_pos.shape[0]}  "
                    f"n_gate_pts={self._gate_frame_points.shape[0]}"
                )

        self._horizon_displaced = yref[0:3, :].T.copy()  # (N, 3) for render
        # ----------------------------------------------------------------------

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
        """Visualize trajectories, APF-displaced horizon, and gate frame samples.

        Green line   — planned spline (original, no APF).
        Orange dots  — APF-displaced reference horizon (what the MPC actually tracks).
        Yellow dots  — gate frame sample points used for APF repulsion.
        Red dot      — current setpoint.
        """
        i = min(self._tick, self._tick_max)
        # Original spline
        draw_line(sim, self._waypoints_pos[::3], rgba=(0.0, 1.0, 0.0, 1.0))
        # APF-displaced horizon positions
        if self._horizon_displaced.shape[0] > 0:
            draw_points(sim, self._horizon_displaced[::3], rgba=(1.0, 0.5, 0.0, 1.0), size=0.015)
        # Gate frame APF sample points
        if self._gate_frame_points.shape[0] > 0:
            draw_points(sim, self._gate_frame_points, rgba=(1.0, 1.0, 0.0, 0.8), size=0.012)
        # Current setpoint
        setpoint = self._waypoints_pos[i].reshape(1, -1)
        draw_points(sim, setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)

    def episode_callback(self):
        if self._compute_times:
            t = np.array(self._compute_times) * 1e3  # ms
            budget = 1000.0 / 50  # 20 ms at 50 Hz
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
