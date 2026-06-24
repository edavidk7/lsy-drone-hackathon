"""Your controller — START HERE.

Online-replanning, full-state controller for the LSY racing env.

Builds a time-parameterized cubic spline through, for each remaining gate, an approach point / the
gate center / an exit point placed along the gate's normal (so the drone crosses straight through the
opening). The spline is tracked open-loop in time, emitting position + velocity + acceleration
feed-forward as a state reference; the drone's onboard low-level controller does the rest. When a
gate's true pose is revealed within sensor range, the estimate is updated and the path is replanned.

Designed for ``control_mode = "state"``. Nothing about the track is hardcoded — gate poses are read
from ``obs`` at runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MyController(Controller):
    """Online-replanning, full-state spline controller."""

    GATE_AXIS = np.array([1.0, 0.0, 0.0])  # gates are crossed along their local +x axis
    APPROACH_OFFSET = 0.15  # m, approach/exit points in front of / behind each gate
    V_NOMINAL = 0.4  # m/s; raise only after it's reliable
    LOOKAHEAD = 0.01  # s; compensates tracker latency
    REPLAN_THRESH = 0.02  # m, replan when a revealed gate/obstacle moves more than this
    ESTIMATE_BLEND = 0.3  # EMA weight for newly revealed poses
    DENSIFY_STEP = 0.1  # m, spacing the path is resampled to before obstacle avoidance
    OBSTACLE_CLEARANCE = 0.4  # m, keep the path at least this far from an obstacle (xy)
    GATE_PROTECT = 0.25  # m, never move path points this close to a gate center

    def __init__(self, obs: dict, info: dict, config: dict) -> None:
        """Initialize the controller and plan the initial path.

        Args:
            obs: Initial observation (``pos``, ``gates_pos``, ``gates_quat``, ``target_gate``, ...).
            info: Reset info.
            config: Race configuration (``config.env.freq`` is the control frequency).
        """
        super().__init__(obs, info, config)
        self._freq = float(config.env.freq)
        self._tick = 0
        self._finished = False
        self._gate_pos = np.asarray(obs["gates_pos"], dtype=np.float64)
        self._gate_quat = np.asarray(obs["gates_quat"], dtype=np.float64)
        self._obstacle_pos = np.asarray(obs["obstacles_pos"], dtype=np.float64).reshape(-1, 3)
        self._spline, self._duration = self._plan(obs)

    def _gate_normals(self, quat: np.ndarray) -> np.ndarray:
        """World-frame unit normals (flight axis) for the given gate quaternions."""
        return Rotation.from_quat(quat).apply(self.GATE_AXIS)

    def _waypoints(self, obs: dict) -> np.ndarray:
        """Approach / center / exit waypoints from the current position through remaining gates."""
        start = np.asarray(obs["pos"], dtype=np.float64)
        target = max(int(obs["target_gate"]), 0)
        normals = self._gate_normals(self._gate_quat)
        points: list[np.ndarray] = [start]
        for i in range(target, len(self._gate_pos)):
            center, n = self._gate_pos[i], normals[i]
            points.append(center - self.APPROACH_OFFSET * n)
            points.append(center)
            points.append(center + self.APPROACH_OFFSET * n)
        points = np.asarray(points, dtype=np.float64)
        # Drop waypoints that coincide with their predecessor (CubicSpline needs increasing t).
        keep = np.concatenate(([True], np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-6))
        return points[keep]

    def _densify(self, pts: np.ndarray) -> np.ndarray:
        """Resample the polyline to ``DENSIFY_STEP`` spacing so avoidance bends it smoothly."""
        dense = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            seg = b - a
            n = max(1, int(np.ceil(np.linalg.norm(seg) / self.DENSIFY_STEP)))
            dense.extend(a + seg * (k / n) for k in range(1, n + 1))
        return np.asarray(dense)

    def _avoid_obstacles(self, pts: np.ndarray) -> np.ndarray:
        """Push path points out of each obstacle's clearance disk (xy), leaving gate crossings intact."""
        if self._obstacle_pos.size == 0:
            return pts
        pts = pts.copy()
        for k in range(1, len(pts)):  # keep the start anchored at the drone
            if np.min(np.linalg.norm(self._gate_pos[:, :2] - pts[k, :2], axis=1)) < self.GATE_PROTECT:
                continue  # don't disturb points near a gate center
            for o in self._obstacle_pos:
                d = pts[k, :2] - o[:2]
                dist = float(np.linalg.norm(d))
                if dist < self.OBSTACLE_CLEARANCE:
                    if dist < 1e-6:  # exactly on the obstacle: push perpendicular to travel
                        seg = pts[k] - pts[k - 1]
                        d = np.array([-seg[1], seg[0]])
                        dist = float(np.linalg.norm(d)) + 1e-9
                    pts[k, :2] = o[:2] + d / dist * self.OBSTACLE_CLEARANCE
        return pts

    def _plan(self, obs: dict) -> tuple[CubicSpline, float]:
        """Fit a constant-speed, clamped cubic spline through obstacle-avoiding waypoints."""
        pts = self._avoid_obstacles(self._densify(self._waypoints(obs)))
        # Drop points coincident with their predecessor (CubicSpline needs strictly increasing t).
        keep = np.concatenate(([True], np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-6))
        pts = pts[keep]
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        t = arc / self.V_NOMINAL  # constant-speed parameterization
        return CubicSpline(t, pts, bc_type="clamped"), float(t[-1])

    def _update_estimates(self, obs: dict) -> bool:
        """Blend in the true pose of any gate/obstacle now revealed within sensor range.

        Returns True if anything moved enough to warrant a replan.
        """
        changed = False
        for i, seen in enumerate(np.asarray(obs["gates_visited"], dtype=bool)):
            if seen:  # true gate pose now revealed
                new = np.asarray(obs["gates_pos"][i], dtype=np.float64)
                if np.linalg.norm(new - self._gate_pos[i]) > self.REPLAN_THRESH:
                    self._gate_pos[i] = (
                        (1 - self.ESTIMATE_BLEND) * self._gate_pos[i] + self.ESTIMATE_BLEND * new
                    )
                    self._gate_quat[i] = np.asarray(obs["gates_quat"][i], dtype=np.float64)
                    changed = True
        for i, seen in enumerate(np.asarray(obs["obstacles_visited"], dtype=bool)):
            if seen:  # true obstacle pose now revealed
                new = np.asarray(obs["obstacles_pos"][i], dtype=np.float64)
                if np.linalg.norm(new - self._obstacle_pos[i]) > self.REPLAN_THRESH:
                    self._obstacle_pos[i] = (
                        (1 - self.ESTIMATE_BLEND) * self._obstacle_pos[i] + self.ESTIMATE_BLEND * new
                    )
                    changed = True
        return changed

    def compute_control(self, obs: dict, info: dict | None = None) -> NDArray[np.floating]:
        """Return the state reference ``[x, y, z, vx, vy, vz, ax, ay, az, yaw, rr, pr, yr]``."""
        if int(np.asarray(obs["target_gate"])) < 0:
            self._finished = True

        if self._update_estimates(obs):
            self._spline, self._duration = self._plan(obs)
            self._tick = 0

        t = min(self._tick / self._freq + self.LOOKAHEAD, self._duration)
        pos = self._spline(t)
        vel = self._spline(t, 1)
        acc = self._spline(t, 2)
        yaw = float(np.arctan2(vel[1], vel[0])) if np.linalg.norm(vel[:2]) > 0.1 else 0.0

        self._tick += 1
        return np.concatenate([pos, vel, acc, [yaw], np.zeros(3)]).astype(np.float64)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Called once after each step. Return True to signal the controller is finished."""
        return self._finished

    def episode_callback(self):
        """Reset the internal state between episodes."""
        self._tick = 0
        self._finished = False
