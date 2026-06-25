"""Min-snap trajectory controller with a collision penalty.

The controller plans **once** at construction time from the gate poses in ``obs`` (never hardcoded):

1. Build waypoints. For every gate we insert a *pre*, *center*, and *post* point along the gate's
   local +x axis (the direction gates must be crossed, see ``envs/utils.gate_passed``). This forces
   a perpendicular, centered crossing -- the main cause of frame clips on the real drone.
2. Insert one free *via* point between consecutive fixed points.
3. Optimize the free via points to minimize the integral of squared snap (4th derivative) plus a
   soft collision penalty that keeps the path away from the pole obstacles (vertical cylinders) and
   the gate frames (square rings in each gate's y-z plane). Gate pre/center/post points stay fixed
   so the crossing geometry is preserved.

At runtime we just evaluate the optimized quintic spline and feed position + velocity + acceleration
feedforward as a state setpoint. The trajectory is deliberately speed-scaled (``speed`` knob) so you
can start slow, verify sim-to-real transfer, and tighten later.

.. note::
    Planning happens in ``__init__`` (and on ``episode_callback``). For Level 0 the nominal gate
    poses equal the true poses, so planning from the initial ``obs`` is exact. For levels with
    moving gates you would re-plan when observed poses change -- the planner here is reusable for
    that, but optimization takes ~1 s so it should not run inside the 50 Hz control loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


class MinSnapController(Controller):
    """Plans a collision-aware min-snap trajectory through the gates and tracks it."""

    # --- Tunables -----------------------------------------------------------------------------
    SPEED = 0.9           # nominal cruise speed [m/s] used for time allocation. Lower = safer.
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

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict, *,
                 plan: bool = True):
        """Plan the trajectory from the initial observation.

        Args:
            obs: Initial observation. Uses ``pos``, ``gates_pos``, ``gates_quat``, ``obstacles_pos``.
            info: Reset info (unused).
            config: Race config; ``config.env.freq`` is the control frequency.
            plan: If ``False``, only set up state without computing the (expensive) initial plan.
                Used when the object is reused purely as a planner (e.g. in training) and the path
                is produced via :meth:`replan` instead.
        """
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self._tick = 0
        self._finished = False

        # Keep-out geometry, derived from the observation.
        self._poles_xy = np.asarray(obs["obstacles_pos"], dtype=float)[:, :2]
        self._pole_r = self.POLE_RADIUS + self.DRONE_RADIUS + self.SAFE_MARGIN
        self._gates_pos = np.asarray(obs["gates_pos"], dtype=float)
        self._gates_rot = R.from_quat(np.asarray(obs["gates_quat"], dtype=float))
        self._first_gate = 0  # plan through gates [_first_gate:]; advanced on re-plan

        if plan:
            self._plan(np.asarray(obs["pos"], dtype=float))

    def replan(self, start_pos, first_gate, gates_pos, gates_quat, obstacles_pos, optimize=False):
        """Re-plan from ``start_pos`` through the remaining gates with updated sensed poses.

        Used by re-planning controllers when a gate's observed pose changes. ``optimize=False``
        skips the slow L-BFGS pass (keeping only the cheap seed + clearance repair) so the re-plan
        is fast enough to run between control steps.
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

        The soft collision penalty only runs inside the (optional) L-BFGS pass, and even then the
        pole-repair below can re-insert waypoints that bow the quintic into a gate frame. This pass
        is the deterministic, always-on guarantee for *both* obstacle types: at each iteration it
        finds the single worst keep-out violation -- pole (push the curve radially out) or gate frame
        (pull the curve back to the opening center) -- pins a waypoint there, and repeats until clear.
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
        # End: continue a bit past the last remaining gate along its normal.
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

    # --- Runtime ------------------------------------------------------------------------------
    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Evaluate the planned spline and return a state setpoint with vel/acc feedforward."""
        t = self._tick / self._freq
        if t >= self._t_total:
            t = self._t_total
            self._finished = True
        pos = self._spline(t)
        vel = self._vel_spline(t) if not self._finished else np.zeros(3)
        acc = self._acc_spline(t) if not self._finished else np.zeros(3)
        return np.concatenate((pos, vel, acc, np.zeros(4)), dtype=np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Advance the time index. Returns True when the trajectory is finished."""
        self._tick += 1
        return self._finished

    def episode_callback(self):
        """Reset the time index for a new episode."""
        self._tick = 0
        self._finished = False

    def render_callback(self, sim: Sim):
        """Draw the planned trajectory, waypoints, and current setpoint for debugging."""
        from crazyflow.sim.visualize import draw_line, draw_points

        traj = self._spline(np.linspace(0.0, self._t_total, 200))
        draw_line(sim, traj, rgba=(0.0, 1.0, 0.0, 1.0))
        draw_points(sim, self._waypoints, rgba=(0.0, 0.0, 1.0, 1.0), size=0.02)
        setpoint = self._spline(min(self._tick / self._freq, self._t_total)).reshape(1, -1)
        draw_points(sim, setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.03)


# region Training helpers
def sample_random_track(rng, takeoff_pos, n_gates: int = 4, n_obstacles: int = 4):
    """Sample a random gate/obstacle layout for training, matching the competition distribution.

    Gates are scattered in x-y within the arena, alternating the two height tiers (0.7 m short /
    1.2 m tall, as in the nominal track) with random yaw. Obstacles (poles) are placed at random
    x-y positions, kept away from the gate centers and the takeoff spot so the track stays solvable.

    Args:
        rng: A ``numpy.random.Generator``.
        takeoff_pos: Drone start position ``[x, y, z]`` (gates/obstacles are kept clear of it).
        n_gates: Number of gates.
        n_obstacles: Number of obstacles.

    Returns:
        ``(gates_pos (n_gates, 3), gates_quat (n_gates, 4), obstacles_pos (n_obstacles, 3))``.
    """
    x_lo, x_hi, y_lo, y_hi = -2.0, 2.0, -1.3, 1.3
    heights = np.array([0.7, 1.2])
    start_xy = np.asarray(takeoff_pos, dtype=float)[:2]

    gates_pos = np.zeros((n_gates, 3))
    for i in range(n_gates):
        for _ in range(50):  # rejection-sample a spot far enough from prior gates and the start
            xy = np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)])
            ok = np.linalg.norm(xy - start_xy) > 0.6
            ok &= all(np.linalg.norm(xy - gates_pos[j, :2]) > 0.8 for j in range(i))
            if ok:
                break
        gates_pos[i, :2] = xy
        gates_pos[i, 2] = heights[i % 2]
    yaws = rng.uniform(-np.pi, np.pi, size=n_gates)
    gates_quat = np.zeros((n_gates, 4))  # xyzw quaternion for a yaw (z-axis) rotation
    gates_quat[:, 2] = np.sin(yaws / 2.0)
    gates_quat[:, 3] = np.cos(yaws / 2.0)

    obstacles_pos = np.zeros((n_obstacles, 3))
    for i in range(n_obstacles):
        for _ in range(50):  # keep poles off the gate centers and the start
            xy = np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)])
            ok = np.linalg.norm(xy - start_xy) > 0.5
            ok &= all(np.linalg.norm(xy - gates_pos[j, :2]) > 0.35 for j in range(n_gates))
            ok &= all(np.linalg.norm(xy - obstacles_pos[j, :2]) > 0.5 for j in range(i))
            if ok:
                break
        obstacles_pos[i, :2] = xy
        obstacles_pos[i, 2] = 1.55
    return gates_pos, gates_quat, obstacles_pos


def make_planner(freq: float = 50.0):
    """Create a reusable :class:`MinSnapController` set up purely as a planner (no initial plan).

    Call :meth:`MinSnapController.replan` on the returned object to generate a path, then read
    ``planner.spline`` / ``planner.t_total``. Used by the training pipeline so training trajectories
    are produced by the exact same planner as inference.
    """
    from types import SimpleNamespace

    dummy = {
        "pos": np.zeros(3),
        "gates_pos": np.zeros((1, 3)),
        "gates_quat": np.array([[0.0, 0.0, 0.0, 1.0]]),
        "obstacles_pos": np.zeros((1, 3)),
    }
    config = SimpleNamespace(env=SimpleNamespace(freq=freq))
    return MinSnapController(dummy, {}, config, plan=False)
