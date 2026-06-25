"""Self-contained min-snap planner + geometric (differential-flatness) attitude controller.

This file is fully self-contained: it bundles its own collision-aware min-snap *planner*
(:class:`_MinSnapPlanner`) together with a classic cascaded *tracker*
(:class:`MinSnapTrackingController`). It does not depend on any other controller module.

1. **Planning** -- :class:`_MinSnapPlanner` builds a smooth, collision-aware trajectory through the
   gates and exposes it as a spline (position + velocity + acceleration are all available as
   analytic derivatives). With ``return_to_start=True`` the final waypoint is the takeoff point, so
   the drone finishes where it started.
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
``control_mode = "attitude"``.

**Re-planning**: when a not-yet-passed gate's observed pose changes we re-plan from the current
position via the planner's fast path. Under a perfect-knowledge setting no re-plan fires, but it
keeps the controller usable on levels with moving gates.

**Config knobs** (read from ``[controller]`` in the track toml):
    ``return_to_start = true``  -- end the trajectory back at the takeoff point.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from drone_models.core import load_params
from scipy.interpolate import make_interp_spline
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


class _MinSnapPlanner:
    """Plans a collision-aware min-snap trajectory through the gates and exposes it as a spline.

    Plain planner (not a Controller): the tracker drives it and reads ``spline`` / ``t_total``.
    """

    # --- Tunables -----------------------------------------------------------------------------
    SPEED = 2.9           # nominal cruise speed [m/s] used for time allocation. Lower = safer.
    A_LIMIT = 12.0        # peak acceleration cap [m/s^2]; trajectory is time-stretched to respect it
    GATE_LEAD = 0.25      # distance of pre/post points along the gate normal [m]
    N_VIA = 2             # free via points inserted between each pair of fixed waypoints
    # Collision model (all in meters). Inflate for the real drone (Lighthouse + tracking error).
    DRONE_RADIUS = 0.06
    SAFE_MARGIN = 0.12    # extra clearance buffer
    POLE_RADIUS = 0.015   # obstacle cylinder radius (0.03 m diameter)
    GATE_OPENING_HALF = 0.20   # 0.4 m opening
    GATE_OUTER_HALF = 0.36     # 0.72 m outer frame
    GATE_HALF_THICK = 0.10     # frame half-thickness along the normal (inflated)
    # Optimization weights
    W_SNAP = 1.0
    W_COLLISION = 5.0e4

    def __init__(self, obs: dict[str, NDArray[np.floating]], *, return_to_start: bool = False,
                 plan: bool = True):
        """Plan the trajectory from the initial observation.

        Args:
            obs: Initial observation. Uses ``pos``, ``gates_pos``, ``gates_quat``, ``obstacles_pos``.
            return_to_start: If ``True``, the final waypoint is the takeoff point (loop back home).
            plan: If ``False``, only set up state without computing the (expensive) initial plan.
        """
        self._return_to_start = return_to_start
        # Keep-out geometry, derived from the observation.
        self._poles_xy = np.asarray(obs["obstacles_pos"], dtype=float)[:, :2]
        self._pole_r = self.POLE_RADIUS + self.DRONE_RADIUS + self.SAFE_MARGIN
        self._gates_pos = np.asarray(obs["gates_pos"], dtype=float)
        self._gates_rot = R.from_quat(np.asarray(obs["gates_quat"], dtype=float))
        self._first_gate = 0  # plan through gates [_first_gate:]; advanced on re-plan
        self._start_pos = np.asarray(obs["pos"], dtype=float)

        if plan:
            self._plan(self._start_pos)

    def replan(self, start_pos, first_gate, gates_pos, gates_quat, obstacles_pos, optimize=False):
        """Re-plan from ``start_pos`` through the remaining gates with updated sensed poses.

        ``optimize=False`` skips the slow L-BFGS pass (keeping only the cheap seed + clearance
        repair) so the re-plan is fast enough to run between control steps.
        """
        self._poles_xy = np.asarray(obstacles_pos, dtype=float)[:, :2]
        self._gates_pos = np.asarray(gates_pos, dtype=float)
        self._gates_rot = R.from_quat(np.asarray(gates_quat, dtype=float))
        self._first_gate = int(first_gate)
        self._plan(np.asarray(start_pos, dtype=float), optimize=optimize)

    @property
    def spline(self):
        """The planned position trajectory as a callable spline ``t -> [x, y, z]``."""
        return self._spline

    @property
    def t_total(self) -> float:
        """Total duration of the planned trajectory in seconds."""
        return self._t_total

    # --- Planning -----------------------------------------------------------------------------
    def _plan(self, start_pos: NDArray[np.floating], optimize: bool = True):
        """Build waypoints, then optimize the free via points for min snap + clearance."""
        fixed, free_idx = self._build_waypoints(start_pos)
        # Time allocation: proportional to chord length, then global speed scaling.
        seg = np.linalg.norm(np.diff(fixed, axis=0), axis=1)
        knots = np.concatenate([[0.0], np.cumsum(np.maximum(seg, 1e-3))]) / self.SPEED
        self._t_total = float(knots[-1])
        self._knots = knots

        free_mask = np.zeros(len(fixed), dtype=bool)
        free_mask[free_idx] = True
        x0 = fixed[free_mask].ravel().copy()

        def assemble(x: NDArray) -> NDArray:
            wp = fixed.copy()
            wp[free_mask] = x.reshape(-1, 3)
            return wp

        def objective(x: NDArray) -> float:
            wp = assemble(x)
            spl = make_interp_spline(knots, wp, k=5)
            return self.W_SNAP * self._snap_cost(spl) + self.W_COLLISION * self._collision_cost(spl)

        if optimize:
            try:
                res = minimize(objective, x0, method="L-BFGS-B", options={"maxiter": 80})
                wp = assemble(res.x)
            except Exception:  # planning must never crash the controller; fall back to the guess
                wp = fixed
        else:  # fast path: skip the L-BFGS pass, rely on seeded via points + clearance repair
            wp = fixed

        # Guarantee clearance: the snap+penalty optimization gives a smooth path but can leave
        # residual pole violations in tight spots. Repair them deterministically by inserting an
        # interpolation waypoint at the worst violation, pushed radially out to the keep-out
        # boundary -- this pins the curve away from the pole. Repeat until clear.
        wp, knots = self._repair_clearance(wp, knots)

        # Dynamic feasibility: uniformly stretch time until peak acceleration is under A_LIMIT.
        # Time-scaling by factor s leaves the path (and thus clearance) unchanged but scales
        # velocity by 1/s and acceleration by 1/s^2, so it is always safe to apply.
        spl = make_interp_spline(knots, wp, k=5)
        ts = np.linspace(0.0, knots[-1], 1500)
        amax = float(np.linalg.norm(spl.derivative(2)(ts), axis=1).max())
        if amax > self.A_LIMIT:
            knots = knots * np.sqrt(amax / self.A_LIMIT)

        self._t_total = float(knots[-1])
        self._knots = knots
        self._spline = make_interp_spline(knots, wp, k=5)
        self._vel_spline = self._spline.derivative(1)
        self._acc_spline = self._spline.derivative(2)
        self._waypoints = wp

    def _repair_clearance(self, wp: NDArray, knots: NDArray, max_iters: int = 25):
        """Insert waypoints to remove residual pole and gate-frame violations along the spline.

        At each iteration it finds the single worst keep-out violation -- pole (push the curve
        radially out) or gate frame (pull the curve back to the opening center) -- pins a waypoint
        there, and repeats until clear.
        """
        for _ in range(max_iters):
            spl = make_interp_spline(knots, wp, k=5)
            ts = np.linspace(0.0, knots[-1], 1200)
            p = spl(ts)
            # Worst (most negative) clearance to any pole or gate frame across all samples.
            worst_pen, worst_t, worst_fix = 0.0, None, None
            for cx, cy in self._poles_xy:
                d = np.hypot(p[:, 0] - cx, p[:, 1] - cy) - self._pole_r
                k = int(np.argmin(d))
                if d[k] < worst_pen:
                    worst_pen = d[k]
                    worst_t = ts[k]
                    direction = np.array([p[k, 0] - cx, p[k, 1] - cy])
                    direction = direction / (np.linalg.norm(direction) + 1e-9)
                    fix = p[k].copy()
                    fix[:2] = np.array([cx, cy]) + direction * (self._pole_r + 0.05)
                    worst_fix = fix
            for i in range(len(self._gates_pos)):
                local = self._gates_rot[i].apply(p - self._gates_pos[i], inverse=True)
                clr = self._frame_clearance(local)
                k = int(np.argmin(clr))
                if clr[k] < worst_pen:
                    worst_pen = clr[k]
                    worst_t = ts[k]
                    worst_fix = self._gate_center_fix(p[k], i)
            if worst_t is None:  # all clear
                break
            i = int(np.searchsorted(knots, worst_t))
            i = max(1, min(i, len(wp) - 1))
            # Guard against degenerate splines: if the correction lands on top of an existing
            # waypoint, move that waypoint instead of inserting a near-duplicate (closely spaced
            # interpolation knots make a quintic spline blow up).
            if min(np.linalg.norm(worst_fix - wp[i - 1]), np.linalg.norm(worst_fix - wp[i])) < 0.08:
                j = i - 1 if np.linalg.norm(worst_fix - wp[i - 1]) < np.linalg.norm(worst_fix - wp[i]) else i
                wp = wp.copy()
                wp[j] = worst_fix
            else:
                wp = np.insert(wp, i, worst_fix, axis=0)
            seg = np.linalg.norm(np.diff(wp, axis=0), axis=1)
            knots = np.concatenate([[0.0], np.cumsum(np.maximum(seg, 0.05))]) / self.SPEED
        return wp, knots

    def _gate_center_fix(self, point: NDArray[np.floating], i: int) -> NDArray[np.floating]:
        """World point pulled onto gate ``i``'s opening axis, keeping its along-normal depth.

        A frame violation means the curve reached the gate plane off-center. The drone must pass
        *through* the opening (not around it), so the only safe repair is to pin that point to the
        opening center in the gate's y-z plane. We keep the local x (depth along the normal) so the
        correction sits at the same point along the crossing, just re-centered.
        """
        local = self._gates_rot[i].apply(point - self._gates_pos[i], inverse=True)
        local[1] = 0.0  # centre in the in-plane axes; the safe zone is only ~2 cm wide anyway
        local[2] = 0.0
        return self._gates_rot[i].apply(local) + self._gates_pos[i]

    def _build_waypoints(self, start_pos: NDArray[np.floating]):
        """Return (waypoints, indices_of_free_via_points)."""
        fixed = [start_pos]
        last = max(self._first_gate, len(self._gates_pos) - 1)
        for i in range(self._first_gate, len(self._gates_pos)):
            normal = self._gates_rot[i].apply([1.0, 0.0, 0.0])
            center = self._gates_pos[i]
            fixed.append(center - self.GATE_LEAD * normal)  # pre
            fixed.append(center)                            # center
            fixed.append(center + self.GATE_LEAD * normal)  # post
        # End: loop back to the takeoff point, or continue a bit past the last remaining gate.
        if self._return_to_start:
            fixed.append(np.asarray(start_pos, dtype=float))
        else:
            last_normal = self._gates_rot[last].apply([1.0, 0.0, 0.0])
            fixed.append(self._gates_pos[last] + (self.GATE_LEAD + 0.5) * last_normal)

        # Insert N_VIA free via points between every pair of fixed points, seeded clear of the
        # poles. Multiple points give the spline local control to bow around a pole that sits close
        # to the pinned pre/post points; a single midpoint-seeded point falls into a local minimum
        # where snap cost resists the detour.
        wp, free_idx = [], []
        for a, b in zip(fixed[:-1], fixed[1:]):
            a, b = np.asarray(a), np.asarray(b)
            wp.append(a)
            for s in np.linspace(0.0, 1.0, self.N_VIA + 2)[1:-1]:
                free_idx.append(len(wp))
                wp.append(self._seed_via(a + s * (b - a)))
        wp.append(np.asarray(fixed[-1]))
        return np.asarray(wp, dtype=float), free_idx

    def _seed_via(self, via: NDArray) -> NDArray:
        """Nudge a seed point in x-y out of any pole keep-out it falls inside."""
        via = via.copy()
        for cx, cy in self._poles_xy:
            d = np.hypot(via[0] - cx, via[1] - cy)
            if d < self._pole_r + 0.05:
                # Push radially away from the pole to the keep-out boundary plus a small buffer.
                direction = np.array([via[0] - cx, via[1] - cy])
                direction = direction / (np.linalg.norm(direction) + 1e-9)
                via[0], via[1] = np.array([cx, cy]) + direction * (self._pole_r + 0.08)
        return via

    def _snap_cost(self, spl) -> float:
        """Integral of squared 4th derivative (snap) along the trajectory."""
        d4 = spl.derivative(4)
        ts = np.linspace(0.0, self._t_total, 300)
        a = d4(ts)
        return float(np.sum(a * a) * (self._t_total / len(ts)))

    def _collision_cost(self, spl) -> float:
        """Soft hinge penalty on clearance to poles and gate frames, sampled along the path."""
        ts = np.linspace(0.0, self._t_total, 400)
        p = spl(ts)
        cost = 0.0
        # Poles: horizontal distance to each cylinder axis.
        for cx, cy in self._poles_xy:
            d = np.hypot(p[:, 0] - cx, p[:, 1] - cy) - self._pole_r
            cost += np.sum(np.clip(-d, 0.0, None) ** 2)
        # Gate frames: clearance to the square ring in each gate's local y-z plane.
        for i in range(len(self._gates_pos)):
            local = self._gates_rot[i].apply(p - self._gates_pos[i], inverse=True)
            cost += np.sum(np.clip(-self._frame_clearance(local), 0.0, None) ** 2)
        return float(cost)

    def _frame_clearance(self, local: NDArray[np.floating]) -> NDArray[np.floating]:
        """Signed clearance (positive = safe) to a gate frame, given points in gate-local coords.

        Gate material occupies |x| <= half_thick and opening_half <= max(|y|,|z|) <= outer_half.
        """
        x, y, z = local[:, 0], local[:, 1], local[:, 2]
        m = np.maximum(np.abs(y), np.abs(z))  # Chebyshev radius in the gate plane
        dx = np.abs(x) - self.GATE_HALF_THICK  # >0 when past the frame plane
        margin = self.DRONE_RADIUS + self.SAFE_MARGIN
        # In-plane signed distance to the ring material.
        d_in = np.where(
            m < self.GATE_OPENING_HALF,
            self.GATE_OPENING_HALF - m,           # inside the opening -> distance to inner edge
            np.where(m > self.GATE_OUTER_HALF, m - self.GATE_OUTER_HALF, 0.0),  # outside / inside
        )
        # Combine plane offset and in-plane distance, then subtract the required margin.
        clear = np.where(dx > 0.0, np.hypot(dx, d_in), d_in)
        return clear - margin


class MinSnapTrackingController(Controller):
    """Min-snap planner + geometric attitude tracker (no learning). A careful RL baseline."""

    # Trajectory pace. PLAN_SPEED sets cruise speed (time allocation); PLAN_A_LIMIT caps peak
    # acceleration -- raise BOTH to go faster, or the planner time-stretches the path back down to
    # respect the accel cap and undoes the speed bump. Push higher for more pace, but watch tracking
    # error and that collective thrust stays under thrust_max in the corners.
    PLAN_SPEED = 1.5
    PLAN_A_LIMIT = 12.0

    # Cascaded controller gains, in acceleration space (m/s^2 per m, per m/s). Conservative and
    # well-damped -- with the acceleration feedforward the loop only corrects residual error.
    KP = np.array([8.0, 8.0, 12.0])
    KD = np.array([5.0, 5.0, 7.0])
    KI = np.array([1.0, 1.0, 2.0])
    I_LIMIT = np.array([1.0, 1.0, 1.0])  # integral clamp [m·s]
    G = 9.81

    REPLAN = False
    REPLAN_POS_THRESH = 0.04  # re-plan when a remaining gate's observed pose moves more than this [m]

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Plan the initial path and set up the tracker.

        Args:
            obs: Initial observation (uses ``pos``, ``quat``, ``vel``, ``gates_pos``, ``gates_quat``,
                ``obstacles_pos``, ``target_gate``).
            info: Reset info (unused).
            config: Race config; ``config.env.freq`` is the control frequency. ``config.controller``
                may set ``return_to_start = true`` to finish back at the takeoff point.
        """
        super().__init__(obs, info, config)
        self.freq = config.env.freq
        self._tick = 0
        self._finished = False
        self._i_error = np.zeros(3)

        # --- Planning: collision-aware min-snap planner (full optimize on the first plan).
        _MinSnapPlanner.SPEED = self.PLAN_SPEED
        _MinSnapPlanner.A_LIMIT = self.PLAN_A_LIMIT
        return_to_start = bool(config.controller.get("return_to_start", False))
        self.planner = _MinSnapPlanner(obs, return_to_start=return_to_start)
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
