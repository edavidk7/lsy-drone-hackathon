"""Standalone PPO experiment for gate-to-gate drone navigation."""

import random
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import fire
import gymnasium as gym
import jax
import jax.numpy as jp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
from gymnasium.vector import VectorActionWrapper, VectorEnv, VectorObservationWrapper, VectorRewardWrapper
from gymnasium.vector.utils import batch_space
from gymnasium.wrappers.vector.jax_to_torch import JaxToTorch
from jax import Array
from torch import Tensor
from torch.distributions.normal import Normal

import wandb
from drone_models.core import load_params
from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv
from lsy_drone_racing.envs.race_core import obs as race_obs
from lsy_drone_racing.envs.utils import gate_passed
from lsy_drone_racing.utils import load_config


@dataclass
class Args:
    """Configuration for the standalone navigation experiment."""

    seed: int = 42
    cuda: bool = True
    jax_device: str = "cpu"
    wandb_project_name: str = "ADR-PPO-Navigation"
    wandb_entity: str | None = None
    config: str = "level2.toml"

    total_timesteps: int = 10_000_000
    learning_rate: float = 1.5e-3
    num_envs: int = 2048
    num_steps: int = 16
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 8
    update_epochs: int = 8
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.007
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    target_kl: float | None = None

    gate_progress_coef: float = 8.0
    gate_alignment_coef: float = 3.0
    gate_plane_progress_coef: float = 4.0
    gate_proximity_coef: float = 2.0  # dense, always-positive proximity reward (removes hover fixed point)
    gate_proximity_sharpness: float = 2.0  # exp(-sharpness * dist) shaping
    backward_progress_coef: float = 4.0
    gate_pass_bonus: float = 20.0
    success_bonus: float = 20.0
    crash_penalty: float = 3.0
    wrong_gate_pass_penalty: float = 8.0
    step_penalty: float = 1.0
    rpy_coef: float = 0.06  # penalty on drone tilt (roll/pitch) magnitude
    act_coef: float = 0.01  # energy penalty on the collective-thrust action channel only
    d_act_main_coef: float = 0.25  # jerk penalty on the roll/pitch/yaw action channels
    d_act_aux_coef: float = 0.1  # jerk penalty on the collective-thrust action channel
    max_angle: float = float(np.pi / 2)  # rad, max commanded roll/pitch
    max_yaw: float = 0.0  # rad, max commanded yaw (0 disables yaw control, like the original RL)
    n_nearest_obstacles: int = 2
    history_steps: int = 2
    checkpoint_every_iterations: int = 10

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

    @staticmethod
    def create(**kwargs: Any) -> "Args":
        args = Args(**kwargs)
        args.batch_size = int(args.num_envs * args.num_steps)
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
        args.num_iterations = max(1, args.total_timesteps // args.batch_size)
        return args


def set_seeds(seed: int) -> None:
    """Seed Python, numpy, torch, and JAX-adjacent paths."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _safe_run_name(name: str) -> str:
    """Convert a run name into a filesystem-safe directory name."""
    keep = []
    for ch in name:
        keep.append(ch if ch.isalnum() or ch in {"-", "_", "."} else "_")
    cleaned = "".join(keep).strip("._")
    return cleaned or "run"


def _gather_by_index(values: Array, indices: Array) -> Array:
    """Gather the per-env value at indices from a batched array."""
    env_ids = jp.arange(values.shape[0])
    safe_indices = jp.clip(indices, 0, values.shape[1] - 1)
    return values[env_ids, safe_indices]


def _normalize_obs(obs: dict[str, Array]) -> dict[str, Array]:
    """Normalize observations to VecDroneRaceEnv's squeezed shape convention."""
    # Drop the singleton n_drones axis (axis 1) for the single-drone vector env. The rank that
    # indicates the axis is present differs per key, so each group has its own threshold.
    normalized = {}
    for key, value in obs.items():
        if key in {"pos", "quat", "vel", "ang_vel"} and value.ndim >= 3:
            normalized[key] = value[:, 0]
        elif key in {"gates_pos", "gates_quat", "obstacles_pos"} and value.ndim >= 4:
            normalized[key] = value[:, 0]
        elif key in {"gates_visited", "obstacles_visited"} and value.ndim >= 3:
            normalized[key] = value[:, 0]
        elif key == "target_gate" and value.ndim >= 2:
            normalized[key] = value[:, 0]
        else:
            normalized[key] = value
    return normalized


def _quat_conjugate(quat: Array) -> Array:
    """Return xyzw quaternion conjugate."""
    return jp.concatenate([-quat[..., :3], quat[..., 3:4]], axis=-1)


def _quat_multiply(lhs: Array, rhs: Array) -> Array:
    """Multiply xyzw quaternions."""
    lx, ly, lz, lw = [lhs[..., i] for i in range(4)]
    rx, ry, rz, rw = [rhs[..., i] for i in range(4)]
    return jp.stack(
        [lw * rx + lx * rw + ly * rz - lz * ry, lw * ry - lx * rz + ly * rw + lz * rx, lw * rz + lx * ry - ly * rx + lz * rw, lw * rw - lx * rx - ly * ry - lz * rz], axis=-1
    )


def _rotate_world_to_body(vec: Array, quat: Array) -> Array:
    """Rotate vectors from world frame into body frame using xyzw quaternions."""
    quat_inv = _quat_conjugate(quat)
    vec_quat = jp.concatenate([vec, jp.zeros((*vec.shape[:-1], 1), dtype=vec.dtype)], axis=-1)
    return _quat_multiply(_quat_multiply(quat_inv, vec_quat), quat)[..., :3]


def _rotate_body_to_world(vec: Array, quat: Array) -> Array:
    """Rotate vectors from body frame into world frame using xyzw quaternions."""
    vec_quat = jp.concatenate([vec, jp.zeros((*vec.shape[:-1], 1), dtype=vec.dtype)], axis=-1)
    return _quat_multiply(_quat_multiply(quat, vec_quat), _quat_conjugate(quat))[..., :3]


def _target_gate_metrics(obs: dict[str, Array]) -> tuple[Array, Array, Array]:
    """Compute target-gate-relative features from an observation dict."""
    target_gate = obs["target_gate"]
    gates_pos = obs["gates_pos"]
    gates_quat = obs["gates_quat"]
    target_gate_pos = _gather_by_index(gates_pos, target_gate)
    target_gate_quat = _gather_by_index(gates_quat, target_gate)
    gate_delta_world = target_gate_pos - obs["pos"]
    gate_delta_body = _rotate_world_to_body(gate_delta_world, obs["quat"])
    gate_quat_rel = _quat_multiply(_quat_conjugate(obs["quat"]), target_gate_quat)
    return gate_delta_world, gate_delta_body, gate_quat_rel


def _nearest_obstacle_features(obs: dict[str, Array], n_nearest: int) -> Array:
    """Return nearest obstacle offsets in the drone body frame."""
    obstacle_delta_world = obs["obstacles_pos"] - obs["pos"][:, None, :]
    obstacle_delta_body = _rotate_world_to_body(obstacle_delta_world.reshape(-1, 3), jp.repeat(obs["quat"], obstacle_delta_world.shape[1], axis=0)).reshape(
        obstacle_delta_world.shape
    )
    dists = jp.linalg.norm(obstacle_delta_body, axis=-1)
    nearest_ids = jp.argsort(dists, axis=-1)[..., :n_nearest]
    env_ids = jp.arange(obstacle_delta_body.shape[0])[:, None]
    nearest = obstacle_delta_body[env_ids, nearest_ids]
    return nearest.reshape(obstacle_delta_body.shape[0], -1)


@partial(jax.jit, static_argnames=("n_nearest_obstacles",))
def build_navigation_features(obs: dict[str, Array], n_nearest_obstacles: int) -> Array:
    """Build a compact navigation observation from the privileged environment state."""
    obs = _normalize_obs(obs)
    gate_delta_world, gate_delta_body, gate_quat_rel = _target_gate_metrics(obs)
    nearest_obstacles = _nearest_obstacle_features(obs, n_nearest_obstacles)
    # Progress is encoded by visited_ratio; the raw target-gate index is dropped (poor as a feature).
    visited_ratio = jp.mean(obs["gates_visited"].astype(jp.float32), axis=-1, keepdims=True)
    return jp.concatenate([gate_delta_world, gate_delta_body, gate_quat_rel, obs["vel"], obs["ang_vel"], nearest_obstacles, visited_ratio], axis=-1)


@jax.jit
def compute_navigation_reward(
    prev_obs: dict[str, Array],
    next_obs: dict[str, Array],
    reward: Array,
    terminated: Array,
    gate_progress_coef: float,
    gate_alignment_coef: float,
    gate_plane_progress_coef: float,
    gate_proximity_coef: float,
    gate_proximity_sharpness: float,
    backward_progress_coef: float,
    gate_pass_bonus: float,
    success_bonus: float,
    crash_penalty: float,
    wrong_gate_pass_penalty: float,
    step_penalty: float,
    rpy_coef: float,
) -> tuple[Array, Array, Array]:
    """Shape reward with gate progress, gate passes, success, and crash penalties."""
    prev_obs = _normalize_obs(prev_obs)
    next_obs = _normalize_obs(next_obs)
    prev_target = prev_obs["target_gate"]
    next_target = next_obs["target_gate"]
    n_gates = prev_obs["gates_pos"].shape[1]

    prev_gate_pos = _gather_by_index(prev_obs["gates_pos"], prev_target)
    prev_dist = jp.linalg.norm(prev_gate_pos - prev_obs["pos"], axis=-1)
    next_dist_to_prev = jp.linalg.norm(prev_gate_pos - next_obs["pos"], axis=-1)
    progress_reward = gate_progress_coef * (prev_dist - next_dist_to_prev)
    _, prev_gate_delta_body, _ = _target_gate_metrics(prev_obs)
    next_gate_delta_world, next_gate_delta_body, _ = _target_gate_metrics(next_obs)
    prev_plane_dist = jp.abs(prev_gate_delta_body[:, 0])
    next_plane_dist = jp.abs(next_gate_delta_body[:, 0])
    plane_progress_reward = gate_plane_progress_coef * (prev_plane_dist - next_plane_dist)
    next_lateral_error = jp.linalg.norm(next_gate_delta_body[:, 1:3], axis=-1)
    forward_progress = (prev_plane_dist - next_plane_dist) > 0
    alignment_reward = gate_alignment_coef * jp.exp(-4.0 * next_lateral_error) * forward_progress.astype(jp.float32)
    backward_penalty = backward_progress_coef * jp.maximum(next_dist_to_prev - prev_dist, 0.0)
    # Dense, always-positive proximity to the current target gate (mirrors the original RL's
    # exp(-dist) tracking reward): hovering far away earns little, so doing nothing is not a free
    # local optimum, and the gradient pulls the drone toward the gate.
    next_dist_to_target = jp.linalg.norm(next_gate_delta_world, axis=-1)
    proximity_reward = gate_proximity_coef * jp.exp(-gate_proximity_sharpness * next_dist_to_target)
    # Tilt penalty: how far the drone's body-up axis leans from world-up (penalizes extreme roll/pitch).
    up = jp.broadcast_to(jp.array([0.0, 0.0, 1.0]), next_obs["quat"].shape[:-1] + (3,))
    body_up = _rotate_body_to_world(up, next_obs["quat"])
    tilt = jp.linalg.norm(body_up[..., :2], axis=-1)

    gate_advanced = next_target > prev_target
    success = (prev_target == (n_gates - 1)) & (next_target == -1)
    crash = terminated & ~success
    all_gate_passes = gate_passed(next_obs["pos"][:, None, :], prev_obs["pos"][:, None, :], prev_obs["gates_pos"], prev_obs["gates_quat"], (0.45, 0.45))
    target_gate_oh = jp.arange(n_gates)[None, :] == jp.clip(prev_target[:, None], 0, n_gates - 1)
    wrong_gate_passes = jp.any(all_gate_passes & ~target_gate_oh, axis=-1)

    shaped = reward + progress_reward + plane_progress_reward + alignment_reward + proximity_reward
    shaped += gate_pass_bonus * gate_advanced.astype(jp.float32)
    shaped += success_bonus * success.astype(jp.float32)
    shaped -= crash_penalty * crash.astype(jp.float32)
    shaped -= backward_penalty
    shaped -= wrong_gate_pass_penalty * wrong_gate_passes.astype(jp.float32)
    shaped -= rpy_coef * tilt
    shaped -= step_penalty
    return shaped, wrong_gate_passes, success


class NavigationObservation(VectorObservationWrapper):
    """Reduce the full environment observation to a compact navigation feature vector."""

    def __init__(self, env: VectorEnv, n_nearest_obstacles: int = 2):
        super().__init__(env)
        self.n_nearest_obstacles = n_nearest_obstacles
        sample = build_navigation_features(race_obs(env.unwrapped.data), self.n_nearest_obstacles)
        feature_dim = int(sample.shape[-1])
        self.single_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(feature_dim,), dtype=np.float32)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)

    def observations(self, observations: dict[str, Array]) -> Array:
        return build_navigation_features(observations, self.n_nearest_obstacles)


def _update_history(obs_hist: Array, act_hist: Array, obs_vec: Array, action: Array, history_steps: int) -> tuple[Array, Array]:
    """Append the latest observation and action to fixed-length histories."""
    if history_steps <= 0:
        return obs_hist, act_hist
    next_obs_hist = jp.concatenate([obs_hist[:, 1:, :], obs_vec[:, None, :]], axis=1)
    next_act_hist = jp.concatenate([act_hist[:, 1:, :], action[:, None, :]], axis=1)
    return next_obs_hist, next_act_hist


class StackObsAct(VectorObservationWrapper):
    """Append a fixed number of past observation/action pairs to each observation."""

    def __init__(self, env: VectorEnv, history_steps: int):
        super().__init__(env)
        self.history_steps = history_steps
        obs_dim = int(np.prod(self.single_observation_space.shape))
        act_dim = int(np.prod(self.single_action_space.shape))
        self._obs_dim = obs_dim
        self._act_dim = act_dim
        self._obs_hist = jp.zeros((self.num_envs, history_steps, obs_dim), dtype=jp.float32)
        self._act_hist = jp.zeros((self.num_envs, history_steps, act_dim), dtype=jp.float32)
        total_dim = obs_dim + history_steps * (obs_dim + act_dim)
        self.single_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)

    def _augment(self, obs_vec: Array) -> Array:
        if self.history_steps <= 0:
            return obs_vec
        return jp.concatenate([obs_vec, self._obs_hist.reshape(self.num_envs, -1), self._act_hist.reshape(self.num_envs, -1)], axis=-1)

    def reset(self, *, seed: int | list[int] | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._obs_hist = jp.zeros((self.num_envs, self.history_steps, self._obs_dim), dtype=jp.float32)
        self._act_hist = jp.zeros((self.num_envs, self.history_steps, self._act_dim), dtype=jp.float32)
        return self._augment(obs), info

    def step(self, action: Array):
        obs, reward, terminated, truncated, info = self.env.step(action)
        augmented_obs = self._augment(obs)
        self._obs_hist, self._act_hist = _update_history(self._obs_hist, self._act_hist, obs, action, self.history_steps)
        if self.history_steps > 0:
            # Clear history for envs that just ended so the next (autoreset) episode is not
            # contaminated with stale observations/actions from the previous one.
            keep = (~(terminated | truncated)).astype(jp.float32)[:, None, None]
            self._obs_hist = self._obs_hist * keep
            self._act_hist = self._act_hist * keep
        return augmented_obs, reward, terminated, truncated, info


class NavigationReward(VectorRewardWrapper):
    """Add dense gate-to-gate shaping on top of the sparse base environment reward."""

    def __init__(
        self,
        env: VectorEnv,
        gate_progress_coef: float,
        gate_alignment_coef: float,
        gate_plane_progress_coef: float,
        gate_proximity_coef: float,
        gate_proximity_sharpness: float,
        backward_progress_coef: float,
        gate_pass_bonus: float,
        success_bonus: float,
        crash_penalty: float,
        wrong_gate_pass_penalty: float,
        step_penalty: float,
        rpy_coef: float,
    ):
        super().__init__(env)
        self.gate_progress_coef = gate_progress_coef
        self.gate_alignment_coef = gate_alignment_coef
        self.gate_plane_progress_coef = gate_plane_progress_coef
        self.gate_proximity_coef = gate_proximity_coef
        self.gate_proximity_sharpness = gate_proximity_sharpness
        self.backward_progress_coef = backward_progress_coef
        self.gate_pass_bonus = gate_pass_bonus
        self.success_bonus = success_bonus
        self.crash_penalty = crash_penalty
        self.wrong_gate_pass_penalty = wrong_gate_pass_penalty
        self.step_penalty = step_penalty
        self.rpy_coef = rpy_coef
        self._prev_obs: dict[str, Array] | None = None
        self._prev_done: Array | None = None

    def reset(self, *, seed: int | list[int] | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._prev_obs = obs
        self._prev_done = jp.zeros(self.num_envs, dtype=bool)
        return obs, info

    def step(self, action: Array):
        observations, rewards, terminations, truncations, infos = self.env.step(action)
        assert self._prev_obs is not None and self._prev_done is not None
        shaped_rewards, wrong_gate_passes, success = compute_navigation_reward(
            self._prev_obs,
            observations,
            rewards,
            terminations,
            self.gate_progress_coef,
            self.gate_alignment_coef,
            self.gate_plane_progress_coef,
            self.gate_proximity_coef,
            self.gate_proximity_sharpness,
            self.backward_progress_coef,
            self.gate_pass_bonus,
            self.success_bonus,
            self.crash_penalty,
            self.wrong_gate_pass_penalty,
            self.step_penalty,
            self.rpy_coef,
        )
        # On the first transition of a new episode (the env autoreset on the previous step), the
        # prev/next observations straddle two episodes; the shaped terms (progress, gate passes,
        # etc.) are meaningless and would otherwise inject a spurious gate-pass bonus. Fall back to
        # the raw base reward for those transitions.
        valid = ~self._prev_done
        shaped_rewards = jp.where(valid, shaped_rewards, rewards)
        wrong_gate_passes = wrong_gate_passes & valid
        success = success & valid
        infos = dict(infos)
        infos["wrong_gate_passes"] = wrong_gate_passes
        infos["success"] = success
        self._prev_obs = observations
        self._prev_done = terminations | truncations
        return observations, shaped_rewards, terminations, truncations, infos


class ActionPenalty(VectorRewardWrapper):
    """Penalize action magnitude and jerkiness.

    The policy action is ``[roll, pitch, yaw, thrust]`` in roughly ``[-1, 1]``. The energy penalty is
    applied to the **thrust channel only** (penalizing roll/pitch magnitude would discourage the very
    tilting needed to fly toward a gate). Smoothness (jerk) is penalized per channel: the "main"
    coefficient on roll/pitch/yaw and the "aux" coefficient on thrust.
    """

    def __init__(self, env: VectorEnv, act_coef: float = 0.01, d_act_main_coef: float = 0.25, d_act_aux_coef: float = 0.1):
        super().__init__(env)
        self.act_coef = act_coef
        self.d_act_main_coef = d_act_main_coef
        self.d_act_aux_coef = d_act_aux_coef
        self._last_action = jp.zeros((self.num_envs, self.single_action_space.shape[0]))

    def reset(self, *, seed: int | list[int] | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_action = jp.zeros((self.num_envs, self.single_action_space.shape[0]))
        return obs, info

    def step(self, action: Array):
        obs, reward, terminated, truncated, info = self.env.step(action)
        action_diff = action - self._last_action
        reward -= self.act_coef * action[:, 3] ** 2  # thrust energy only
        reward -= self.d_act_main_coef * jp.sum(action_diff[:, :3] ** 2, axis=-1)
        reward -= self.d_act_aux_coef * jp.sum(action_diff[:, 3:] ** 2, axis=-1)
        # Forget the previous action for envs that just reset, so the next episode's first delta
        # penalty is not computed across an episode boundary.
        self._last_action = jp.where((terminated | truncated)[:, None], jp.zeros_like(action), action)
        return obs, reward, terminated, truncated, info


NAV_ACTION_DIM = 4  # policy action: [roll, pitch, yaw, thrust]


def attitude_setpoint_from_action(action, thrust_min: float, thrust_max: float, max_angle: float, max_yaw: float = 0.0, xp=jp):
    """Map a ``[-1, 1]`` policy action to a 4-D attitude command ``[roll, pitch, yaw, thrust]``.

    Shared by the training wrapper (``xp=jp``) and the inference controller (``xp=np``) so the two
    cannot drift. Affine ``clip(a) * scale + mean`` mapping (same convention as the original attitude
    RL): roll / pitch span ``[-max_angle, max_angle]``, yaw spans ``[-max_yaw, max_yaw]`` (``max_yaw=0``
    disables yaw, as in the original), collective thrust spans ``[thrust_min, thrust_max]``. Works for
    a single sample or a batch (broadcasts on ``...``).
    """
    a = xp.clip(action, -1.0, 1.0)
    angle_scale = xp.asarray([max_angle, max_angle, max_yaw], dtype=a.dtype)
    rpy = a[..., :3] * angle_scale
    thrust = (thrust_max + thrust_min) / 2.0 + a[..., 3:4] * (thrust_max - thrust_min) / 2.0
    return xp.concatenate([rpy, thrust], axis=-1)


class AttitudeAction(VectorActionWrapper):
    """Map a bounded 4-D policy action to a collective-thrust + roll/pitch/yaw attitude command.

    The policy outputs a 4-D action in ``[-1, 1]``; this wrapper affine-maps it (see
    :func:`attitude_setpoint_from_action`) to the env's attitude action ``[roll, pitch, yaw, thrust]``.
    Thrust bounds come from the drone model (per-rotor bounds x 4 for the collective).
    """

    def __init__(self, env: VectorEnv, drone_model: str, max_angle: float = float(np.pi / 2), max_yaw: float = 0.0):
        super().__init__(env)
        self.max_yaw = max_yaw
        params = load_params("so_rpy", drone_model)
        self.thrust_min = float(params["thrust_min"]) * 4.0
        self.thrust_max = float(params["thrust_max"]) * 4.0
        self.max_angle = max_angle
        self.single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(NAV_ACTION_DIM,), dtype=np.float32)
        self.action_space = batch_space(self.single_action_space, self.num_envs)

    def actions(self, actions: Array) -> Array:
        return attitude_setpoint_from_action(actions, self.thrust_min, self.thrust_max, self.max_angle, self.max_yaw, xp=jp)


def make_envs(config: str = "level2.toml", num_envs: int = 256, jax_device: str = "cpu", torch_device: torch.device = torch.device("cpu"), args: Args | None = None) -> VectorEnv:
    """Create the standalone gate-navigation training environment."""
    if args is None:
        args = Args.create()
    config_data = load_config(Path(__file__).parents[2] / "config" / config)
    assert config_data.env.control_mode == "attitude", f"train_nav_rl is attitude-only; set control_mode='attitude' in {config}"
    env = VecDroneRaceEnv(
        num_envs=num_envs,
        freq=config_data.env.freq,
        sim_config=config_data.sim,
        track=config_data.env.track,
        sensor_range=config_data.env.sensor_range,
        control_mode=config_data.env.control_mode,
        disturbances=getattr(config_data.env, "disturbances", None),
        randomizations=getattr(config_data.env, "randomizations", None),
        seed=config_data.env.seed,
        max_episode_steps=1500,
        device=jax_device,
    )
    env = AttitudeAction(env, config_data.sim.drone_model, max_angle=args.max_angle, max_yaw=args.max_yaw)
    env = NavigationReward(
        env,
        gate_progress_coef=args.gate_progress_coef,
        gate_alignment_coef=args.gate_alignment_coef,
        gate_plane_progress_coef=args.gate_plane_progress_coef,
        gate_proximity_coef=args.gate_proximity_coef,
        gate_proximity_sharpness=args.gate_proximity_sharpness,
        backward_progress_coef=args.backward_progress_coef,
        gate_pass_bonus=args.gate_pass_bonus,
        success_bonus=args.success_bonus,
        crash_penalty=args.crash_penalty,
        wrong_gate_pass_penalty=args.wrong_gate_pass_penalty,
        step_penalty=args.step_penalty,
        rpy_coef=args.rpy_coef,
    )
    env = ActionPenalty(env, act_coef=args.act_coef, d_act_main_coef=args.d_act_main_coef, d_act_aux_coef=args.d_act_aux_coef)
    env = NavigationObservation(env, n_nearest_obstacles=args.n_nearest_obstacles)
    env = StackObsAct(env, history_steps=args.history_steps)
    env = JaxToTorch(env, torch_device)
    return env


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class RunningMeanStd(nn.Module):
    """Running mean/variance (Welford) for observation normalization, stored as module buffers."""

    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-4):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(epsilon))

    @torch.no_grad()
    def update(self, x: Tensor) -> None:
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x: Tensor) -> Tensor:
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-8), -10.0, 10.0)


class Agent(nn.Module):
    """Simple MLP actor-critic with running observation normalization."""

    def __init__(self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...], hdim: int = 256):
        super().__init__()
        obs_dim = int(torch.tensor(obs_shape).prod())
        act_dim = int(torch.tensor(action_shape).prod())
        self.obs_rms = RunningMeanStd((obs_dim,))
        self.critic = nn.Sequential(layer_init(nn.Linear(obs_dim, hdim)), nn.Tanh(), layer_init(nn.Linear(hdim, hdim)), nn.Tanh(), layer_init(nn.Linear(hdim, 1), std=1.0))
        # Final Tanh keeps the action mean in (-1, 1) with live gradients, so the policy does not
        # saturate against the wrapper's hard clip (which has zero gradient outside [-1, 1]).
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hdim)), nn.Tanh(), layer_init(nn.Linear(hdim, hdim)), nn.Tanh(), layer_init(nn.Linear(hdim, act_dim), std=0.01), nn.Tanh()
        )
        # Asymmetric exploration: small std on roll/pitch/yaw, large std on thrust (last channel) so
        # the drone explores lift and learns to take off instead of collapsing to a hover.
        init_logstd = torch.full((1, act_dim), -1.0)
        init_logstd[0, -1] = 1.0
        self.actor_logstd = nn.Parameter(init_logstd)

    @torch.compile
    def get_value(self, x: Tensor) -> Tensor:
        return self.critic(self.obs_rms.normalize(x))

    @torch.compile
    def get_action_and_value(self, x: Tensor, action: Tensor | None = None, deterministic: bool = False) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x = self.obs_rms.normalize(x)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = action_mean if deterministic else probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


def select_torch_device(use_cuda: bool) -> torch.device:
    """Select torch device with CPU-safe fallback."""
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_ppo(args: Args, model_path: Path | None, device: torch.device, jax_device: str, wandb_enabled: bool = False):
    """Train PPO on gate-to-gate navigation."""
    if wandb_enabled and wandb.run is None:
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, config=vars(args))

    set_seeds(args.seed)
    print("Training on device:", device, "| Environment device:", jax_device)
    envs = make_envs(config=args.config, num_envs=args.num_envs, jax_device=jax_device, torch_device=device, args=args)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs.single_observation_space.shape, envs.single_action_space.shape).to(device)
    optimizer = optim.AdamW(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    base_dir = model_path.parent if model_path is not None else Path(__file__).parent
    if wandb_enabled and wandb.run is not None and wandb.run.name:
        run_dir_name = _safe_run_name(wandb.run.name)
    else:
        run_dir_name = f"local_{time.strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = base_dir / run_dir_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    train_start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)
    next_done = torch.zeros(args.num_envs, device=device)  # term | trunc, for episode bookkeeping
    next_termination = torch.zeros(args.num_envs, device=device)  # terminations only, for GAE
    sum_rewards = torch.zeros(args.num_envs, device=device)
    episode_gate_hits = torch.zeros(args.num_envs, device=device)
    episode_obstacle_hits = torch.zeros(args.num_envs, device=device)
    episode_wrong_gate_passes = torch.zeros(args.num_envs, device=device)
    episode_success = torch.zeros(args.num_envs, device=device)
    reward_hist: list[float] = []

    for iteration in range(1, args.num_iterations + 1):
        iter_start = time.time()
        iter_gate_hits = 0.0
        iter_obstacle_hits = 0.0
        iter_gate_hit_episodes = 0.0
        iter_obstacle_hit_episodes = 0.0
        iter_wrong_gate_passes = 0.0
        iter_wrong_gate_pass_episodes = 0.0
        iter_success_episodes = 0.0
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_termination  # GAE cuts bootstrap only on true terminations

            agent.obs_rms.update(next_obs)
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action)
            next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)
            reward = torch.as_tensor(reward, device=device, dtype=torch.float32)
            terminations = torch.as_tensor(terminations, device=device)
            truncations = torch.as_tensor(truncations, device=device)
            gate_hits = torch.as_tensor(infos["gate_hits"], device=device).float()
            obstacle_hits = torch.as_tensor(infos["obstacle_hits"], device=device).float()
            wrong_gate_passes = torch.as_tensor(infos["wrong_gate_passes"], device=device).float()
            success = torch.as_tensor(infos["success"], device=device).float()

            rewards[step] = reward
            sum_rewards += reward
            episode_gate_hits += gate_hits
            episode_obstacle_hits += obstacle_hits
            episode_wrong_gate_passes += wrong_gate_passes
            episode_success = torch.maximum(episode_success, success)
            iter_gate_hits += gate_hits.sum().item()
            iter_obstacle_hits += obstacle_hits.sum().item()
            iter_wrong_gate_passes += wrong_gate_passes.sum().item()
            next_termination = terminations
            next_done = terminations | truncations

            if wandb_enabled and next_done.any():
                done_rewards = sum_rewards[next_done.bool()]
                done_gate_hits = episode_gate_hits[next_done.bool()]
                done_obstacle_hits = episode_obstacle_hits[next_done.bool()]
                done_wrong_gate_passes = episode_wrong_gate_passes[next_done.bool()]
                done_success = episode_success[next_done.bool()]
                iter_gate_hit_episodes += (done_gate_hits > 0).float().sum().item()
                iter_obstacle_hit_episodes += (done_obstacle_hits > 0).float().sum().item()
                iter_wrong_gate_pass_episodes += (done_wrong_gate_passes > 0).float().sum().item()
                iter_success_episodes += done_success.sum().item()
                for r, g, o, w, s in zip(done_rewards, done_gate_hits, done_obstacle_hits, done_wrong_gate_passes, done_success):
                    reward_hist.append(r.item())
                    wandb.log(
                        {"train/reward": r.item(), "train/gate_hits": g.item(), "train/obstacle_hits": o.item(), "train/wrong_gate_passes": w.item(), "train/success": s.item()},
                        step=global_step,
                    )
                sum_rewards[next_done.bool()] = 0
                episode_gate_hits[next_done.bool()] = 0
                episode_obstacle_hits[next_done.bool()] = 0
                episode_wrong_gate_passes[next_done.bool()] = 0
                episode_success[next_done.bool()] = 0
            elif next_done.any():
                done_gate_hits = episode_gate_hits[next_done.bool()]
                done_obstacle_hits = episode_obstacle_hits[next_done.bool()]
                done_wrong_gate_passes = episode_wrong_gate_passes[next_done.bool()]
                done_success = episode_success[next_done.bool()]
                iter_gate_hit_episodes += (done_gate_hits > 0).float().sum().item()
                iter_obstacle_hit_episodes += (done_obstacle_hits > 0).float().sum().item()
                iter_wrong_gate_pass_episodes += (done_wrong_gate_passes > 0).float().sum().item()
                iter_success_episodes += done_success.sum().item()
                sum_rewards[next_done.bool()] = 0
                episode_gate_hits[next_done.bool()] = 0
                episode_obstacle_hits[next_done.bool()] = 0
                episode_wrong_gate_passes[next_done.bool()] = 0
                episode_success[next_done.bool()] = 0

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_termination.float()
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for _epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.detach().cpu().numpy(), b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if wandb_enabled:
            wandb.log(
                {
                    "charts/learning_rate": optimizer.param_groups[0]["lr"],
                    "losses/value_loss": v_loss.item(),
                    "losses/policy_loss": pg_loss.item(),
                    "losses/entropy": entropy_loss.item(),
                    "losses/old_approx_kl": old_approx_kl.item(),
                    "losses/approx_kl": approx_kl.item(),
                    "losses/clipfrac": float(np.mean(clipfracs)),
                    "losses/explained_variance": explained_var,
                    "charts/SPS": int(global_step / max(1e-6, (time.time() - train_start_time))),
                    "charts/gate_hit_events": iter_gate_hits,
                    "charts/obstacle_hit_events": iter_obstacle_hits,
                    "charts/gate_hit_episodes": iter_gate_hit_episodes,
                    "charts/obstacle_hit_episodes": iter_obstacle_hit_episodes,
                    "charts/wrong_gate_pass_events": iter_wrong_gate_passes,
                    "charts/wrong_gate_pass_episodes": iter_wrong_gate_pass_episodes,
                    "charts/success_episodes": iter_success_episodes,
                },
                step=global_step,
            )

        print(
            f"Iter {iteration}/{args.num_iterations} took {time.time() - iter_start:.2f} seconds"
            f" | gate_hits={iter_gate_hits:.0f} obstacle_hits={iter_obstacle_hits:.0f}"
            f" | wrong_gate_passes={iter_wrong_gate_passes:.0f}"
            f" | gate_hit_eps={iter_gate_hit_episodes:.0f} obstacle_hit_eps={iter_obstacle_hit_episodes:.0f}"
            f" wrong_gate_pass_eps={iter_wrong_gate_pass_episodes:.0f} success_eps={iter_success_episodes:.0f}"
        )
        if model_path is not None and args.checkpoint_every_iterations > 0:
            if iteration % args.checkpoint_every_iterations == 0:
                ckpt_path = checkpoint_dir / f"{model_path.stem}_iter{iteration:03d}.ckpt"
                torch.save(agent.state_dict(), ckpt_path)

    print(f"Training for {global_step} steps took {time.time() - train_start_time:.2f} seconds.")
    if model_path is not None:
        final_model_path = checkpoint_dir / model_path.name
        torch.save(agent.state_dict(), final_model_path)
        print(f"model saved to {final_model_path}")
    envs.close()
    return reward_hist, []


def main(wandb_enabled: bool = True, train: bool = True, **kwargs: Any) -> None:
    """Entry point."""
    args = Args.create(**kwargs)
    model_path = Path(__file__).parent / "ppo_nav_drone_racing.ckpt"
    device = select_torch_device(args.cuda)
    if train:
        train_ppo(args, model_path, device, args.jax_device, wandb_enabled)
    if wandb_enabled and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
