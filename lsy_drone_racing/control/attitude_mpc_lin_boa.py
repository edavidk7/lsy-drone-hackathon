"""MPC using attitude control for a quadrotor, with successive linearization (RTI scheme).

Copy of attitude_mpc_lin.py modified so that Q and Rw diagonal weights can be passed as
constructor arguments. Used by boa_tune.py for automated weight optimisation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as cs
import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points
from drone_models.core import load_params
from drone_models.so_rpy import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


def rk4_step(f: cs.Function, x: cs.MX, u: cs.MX, dt: float) -> cs.MX:
    k1 = f(x, u)
    k2 = f(x + (dt / 2) * k1, u)
    k3 = f(x + (dt / 2) * k2, u)
    k4 = f(x + dt * k3, u)
    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


_DEFAULT_Q_DIAG = [100.0, 100.0, 200.0, 1.0, 1.0, 1.0, 40.0, 40.0, 40.0, 5.0, 5.0, 5.0]
_DEFAULT_RW_DIAG = [1.0, 1.0, 1.0, 50.0]


class AttitudeMPCTunable(Controller):
    """Attitude MPC with successive linearization (RTI scheme).

    Accepts optional ``q_diag`` and ``rw_diag`` constructor arguments so that an
    external optimiser (e.g. BOA) can sweep weight configurations without editing
    this file.  When omitted the defaults match attitude_mpc_lin.py.
    """

    def __init__(
        self,
        obs: dict[str, NDArray[np.floating]],
        info: dict,
        config: dict,
        q_diag: list[float] | None = None,
        rw_diag: list[float] | None = None,
        t_total: float | None = None,
    ):
        super().__init__(obs, info, config)
        self._N = 30
        self._dt = 1 / config.env.freq

        start_pos = obs["pos"]
        waypoints = np.array(
            [
                start_pos,
                [-1.0, 0.75, 0.4],
                [0.3, 0.35, 0.7],
                [1.3, -0.15, 0.9],
                [0.85, 0.85, 1.2],
                [-0.5, -0.05, 0.7],
                [-1.2, -0.2, 0.8],
                [-1.2, -0.2, 1.2],
                [-0.0, -0.7, 1.2],
                [0.5, -0.75, 1.2],
            ]
        )
        self._t_total = t_total if t_total is not None else 9.5
        t = np.linspace(0, self._t_total, len(waypoints))
        pos_spline = CubicSpline(t, waypoints)
        vel_spline = pos_spline.derivative()
        t_eval = np.linspace(0, self._t_total, int(config.env.freq * self._t_total))
        self._waypoints_pos = pos_spline(t_eval)
        self._waypoints_vel = vel_spline(t_eval)
        self._waypoints_yaw = np.zeros(len(self._waypoints_pos))

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

        q_d = q_diag if q_diag is not None else _DEFAULT_Q_DIAG
        rw_d = rw_diag if rw_diag is not None else _DEFAULT_RW_DIAG
        Q = np.diag(q_d)
        Rw = np.diag(rw_d)
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

        self._tick = 0
        self._tick_max = len(self._waypoints_pos) - 1 - self._N
        self._finished = False

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

        A_p = [opti.parameter(nx, nx) for _ in range(N)]
        B_p = [opti.parameter(nx, nu) for _ in range(N)]
        c_p = [opti.parameter(nx) for _ in range(N)]

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
            opti.subject_to(
                X[:, j + 1] == cs.mtimes(A_p[j], X[:, j]) + cs.mtimes(B_p[j], U[:, j]) + c_p[j]
            )

        opti.subject_to(cs.vec(X[3:6, 1:N]) >= -rpy_limit)
        opti.subject_to(cs.vec(X[3:6, 1:N]) <= rpy_limit)

        u_lb_dm = cs.repmat(cs.DM(self._u_lb.reshape(-1, 1)), 1, N)
        u_ub_dm = cs.repmat(cs.DM(self._u_ub.reshape(-1, 1)), 1, N)
        opti.subject_to(cs.vec(U) >= cs.vec(u_lb_dm))
        opti.subject_to(cs.vec(U) <= cs.vec(u_ub_dm))

        p_opts = {"print_time": 0}
        s_opts = {"max_iter": 50, "tol": 1e-4, "print_level": 0, "warm_start_init_point": "yes"}
        opti.solver("ipopt", p_opts, s_opts)

        return opti, X, U, x0_p, yref_p, yref_e_p, A_p, B_p, c_p

    def compute_control(
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

        for j in range(self._N):
            x_bar = self._X_ws[:, j]
            u_bar = self._U_ws[:, j]
            A_j = np.asarray(self._jac_x(x_bar, u_bar))
            B_j = np.asarray(self._jac_u(x_bar, u_bar))
            f_bar = np.asarray(self._f_rk4(x_bar, u_bar)).ravel()
            c_j = f_bar - A_j @ x_bar - B_j @ u_bar
            self._opti.set_value(self._A_p[j], A_j)
            self._opti.set_value(self._B_p[j], B_j)
            self._opti.set_value(self._c_p[j], c_j)

        self._opti.set_value(self._x0_p, x0)

        N, nx, nu = self._N, self._nx, self._nu
        ny = nx + nu
        yref = np.zeros((ny, N))
        yref[0:3, :] = self._waypoints_pos[i : i + N].T
        yref[5, :] = self._waypoints_yaw[i : i + N]
        yref[6:9, :] = self._waypoints_vel[i : i + N].T
        yref[nx + 3, :] = self._hover_thrust
        self._opti.set_value(self._yref_p, yref)

        yref_e = np.zeros(nx)
        yref_e[0:3] = self._waypoints_pos[i + N]
        yref_e[5] = self._waypoints_yaw[i + N]
        yref_e[6:9] = self._waypoints_vel[i + N]
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
        setpoint = self._waypoints_pos[i].reshape(1, -1)
        draw_points(sim, setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)
        trajectory = self._waypoints_pos[::3]
        draw_line(sim, trajectory, rgba=(0.0, 1.0, 0.0, 1.0))

    def episode_callback(self):
        self._tick = 0
