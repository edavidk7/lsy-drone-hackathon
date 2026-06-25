"""Standalone PPO experiment for gate-to-gate drone navigation."""

import random
import time
from dataclasses import asdict, dataclass
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
from crazyflow.envs.norm_actions_wrapper import NormalizeActions
from drone_models.core import load_params
from gymnasium import spaces
from gymnasium.vector import VectorActionWrapper, VectorEnv, VectorObservationWrapper, VectorRewardWrapper
from gymnasium.vector.utils import batch_space
from gymnasium.wrappers.vector.jax_to_torch import JaxToTorch
from jax import Array
from torch import Tensor
from torch.distributions.normal import Normal

import wandb
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
    resume_from: str | None = None  # path to a checkpoint (.ckpt) to resume model weights from

    total_timesteps: int = 40_000_000
    learning_rate: float = 3e-4  # 1.5e-3 works on this problem too
    num_envs: int = 2048
    num_steps: int = 16
    anneal_lr: bool = True
    min_learning_rate: float = 5e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 8
    update_epochs: int = 8
    norm_adv: bool = True
    clip_coef: float = 0.23  # 0.23 also worked well
    clip_vloss: bool = True
    ent_coef: float = 0.02
    vf_coef: float = 0.7
    max_grad_norm: float = 1.0
    target_kl: float | None = None  # can be none

    actor_hdim: int = 256
    critic_hdim: int = 256
    gate_progress_coef: float = 5.0
    gate_pass_bonus: float = 10.0
    success_bonus: float = 15.0
    crash_penalty: float = 5.0
    init_logstd: float = -1
    init_logstd_last: float = 1.0
    act_coef: float = 0.01  # energy penalty on the collective-thrust action channel only
    d_act_main_coef: float = 0.1  # jerk penalty on the roll/pitch/yaw action channels
    d_act_aux_coef: float = 0.2  # jerk penalty on the collective-thrust action channel
    max_angle: float = float(np.pi / 2)  # rad, max commanded roll/pitch
    max_yaw: float = float(np.pi / 2)  # rad, max commanded yaw 
    n_nearest_obstacles: int = 2
    progress_obs: bool = True  # append normalized passed-gate fraction (target_gate / n_gates) to the observation
    lookahead_gates: int = 1  # number of upcoming gates (after the target) to include, in the drone body frame
    gravity_obs: bool = True  # append the gravity direction in the drone body frame (attitude/tilt cue)
    checkpoint_every_iterations: int = 10
    episode_step_limit: int = 1500

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

    @staticmethod
    def create(**kwargs: Any) -> "Args":
        args = Args(**kwargs)
        args.batch_size = int(args.num_envs * args.num_steps)
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
        args.num_iterations = max(1, args.total_timesteps // args.batch_size)
        print(f"Created args with {args.batch_size=}, {args.minibatch_size=}, {args.num_iterations=}")
        return args


def set_seeds(seed: int) -> None:
    """Seed Python, numpy, torch, and JAX-adjacent paths."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('high')


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


@jax.jit
def _quat_conjugate(quat: Array) -> Array:
    """Return xyzw quaternion conjugate."""
    return jp.concatenate([-quat[..., :3], quat[..., 3:4]], axis=-1)


@jax.jit
def _quat_multiply(lhs: Array, rhs: Array) -> Array:
    """Multiply xyzw quaternions."""
    lx, ly, lz, lw = [lhs[..., i] for i in range(4)]
    rx, ry, rz, rw = [rhs[..., i] for i in range(4)]
    return jp.stack(
        [lw * rx + lx * rw + ly * rz - lz * ry, lw * ry - lx * rz + ly * rw + lz * rx, lw * rz + lx * ry - ly * rx + lz * rw, lw * rw - lx * rx - ly * ry - lz * rz], axis=-1
    )


@jax.jit
def _quat_to_rotmat(quat: Array) -> Array:
    """Convert an xyzw quaternion to a flattened 3x3 rotation matrix, shape ``(..., 9)``.

    A rotation matrix is a continuous attitude representation that avoids the quaternion
    double-cover ambiguity (``q`` and ``-q`` are the same rotation), which is the convention used
    by the Swift paper and is known to be friendlier to neural-network inputs than quaternions.
    """
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    r00 = 1.0 - 2.0 * (yy + zz)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)
    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (xx + zz)
    r12 = 2.0 * (yz - wx)
    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)
    return jp.stack([r00, r01, r02, r10, r11, r12, r20, r21, r22], axis=-1)


@jax.jit
def _rotate_world_to_body(vec: Array, quat: Array) -> Array:
    """Rotate vectors from world frame into body frame using xyzw quaternions."""
    quat_inv = _quat_conjugate(quat)
    vec_quat = jp.concatenate([vec, jp.zeros((*vec.shape[:-1], 1), dtype=vec.dtype)], axis=-1)
    return _quat_multiply(_quat_multiply(quat_inv, vec_quat), quat)[..., :3]


@jax.jit
def _rotate_body_to_world(vec: Array, quat: Array) -> Array:
    """Rotate vectors from body frame into world frame using xyzw quaternions."""
    vec_quat = jp.concatenate([vec, jp.zeros((*vec.shape[:-1], 1), dtype=vec.dtype)], axis=-1)
    return _quat_multiply(_quat_multiply(quat, vec_quat), _quat_conjugate(quat))[..., :3]


@jax.jit
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
    """Return the ``n_nearest`` obstacles' body-frame offsets, each tagged with its sensed flag.

    Two deliberate choices:
    - Membership is the ``n_nearest`` closest obstacles, but the selected slots are then ordered by
      **track index** (not by distance). This removes the slot-swap discontinuity that pure
      distance-sorting introduces when two selected obstacles are near-equidistant: with a fixed
      index order their feature 3-vectors never swap places. (The membership boundary - a third
      obstacle becoming nearer than a selected one - remains discontinuous; that is the inherent
      k<N wart and is unavoidable without feeding all obstacles.)
    - Each obstacle's ``obstacles_visited`` flag is appended. Until an obstacle is within
      sensor_range its reported position is the nominal (pre-randomization) guess and snaps to the
      true position once sensed; the flag lets the policy distinguish a confirmed pole from a guess.
    """
    obstacle_delta_world = obs["obstacles_pos"] - obs["pos"][:, None, :]
    obstacle_delta_body = _rotate_world_to_body(obstacle_delta_world.reshape(-1, 3), jp.repeat(obs["quat"], obstacle_delta_world.shape[1], axis=0)).reshape(
        obstacle_delta_world.shape
    )
    dists = jp.linalg.norm(obstacle_delta_body, axis=-1)
    nearest_ids = jp.argsort(dists, axis=-1)[..., :n_nearest]
    # Re-order the selected indices by track index so each slot is tied to a fixed obstacle ordering.
    nearest_ids = jp.sort(nearest_ids, axis=-1)
    env_ids = jp.arange(obstacle_delta_body.shape[0])[:, None]
    nearest = obstacle_delta_body[env_ids, nearest_ids]
    visited = obs["obstacles_visited"].astype(jp.float32)[env_ids, nearest_ids][..., None]
    return jp.concatenate([nearest, visited], axis=-1).reshape(obstacle_delta_body.shape[0], -1)


@jax.jit
def _gate_facing_body(quat_rel: Array) -> Array:
    """Gate pass-through normal (gate local +x) expressed in the drone body frame, shape ``(..., 3)``.

    A gate is a ~1-DOF (yaw) oriented plane, so its facing direction is all the policy needs to plan
    the crossing - far less than a full 3x3 relative rotation, which would also redundantly re-encode
    the drone's own attitude (fed separately as gravity-in-body). As a body-frame unit vector it is
    egocentric and continuous (no yaw wraparound), and it still captures small gate roll/pitch
    perturbations. Equals the first column of the drone->gate relative rotation matrix.
    """
    rotmat_rel = _quat_to_rotmat(quat_rel)
    return rotmat_rel[..., jp.asarray([0, 3, 6])]


def _lookahead_gate_features(obs: dict[str, Array], n_lookahead: int) -> Array:
    """Body-frame relative position + relative orientation of the next ``n_lookahead`` gates.

    The features are egocentric (drone-local), like the target-gate terms, so they are rotation
    invariant and directly actionable. On a randomized track the policy cannot memorize the layout,
    so seeing the upcoming gate(s) is what lets it plan a line through the current gate that sets up
    the next one. Gates past the last (no successor) are zeroed; the policy reads all-zeros as "no
    further gate". The upcoming gate's nominal pose is available from t=0 even under a finite
    sensor_range, so this is real planning information, not a privileged leak.
    """
    target_gate = obs["target_gate"]
    n_gates = obs["gates_pos"].shape[1]
    feats = []
    for k in range(1, n_lookahead + 1):
        idx = target_gate + k
        valid = ((target_gate >= 0) & (idx < n_gates))[:, None].astype(jp.float32)
        gate_pos = _gather_by_index(obs["gates_pos"], idx)
        gate_quat = _gather_by_index(obs["gates_quat"], idx)
        delta_body = _rotate_world_to_body(gate_pos - obs["pos"], obs["quat"])
        facing_body = _gate_facing_body(_quat_multiply(_quat_conjugate(obs["quat"]), gate_quat))
        feats.append(delta_body * valid)
        feats.append(facing_body * valid)
    return jp.concatenate(feats, axis=-1)


@partial(jax.jit, static_argnames=("n_nearest_obstacles", "include_progress", "lookahead_gates", "gravity_obs"))
def build_navigation_features(
    obs: dict[str, Array], n_nearest_obstacles: int, include_progress: bool = False, lookahead_gates: int = 0, gravity_obs: bool = False
) -> Array:
    """Build a compact navigation observation from the privileged environment state."""
    obs = _normalize_obs(obs)
    gate_delta_world, gate_delta_body, gate_quat_rel = _target_gate_metrics(obs)
    # The target gate's facing normal in the body frame (3-D), not a full 9-D relative rotation: a
    # gate is a ~1-DOF (yaw) plane, so its pass-through direction is the sufficient, non-redundant cue.
    gate_facing_body = _gate_facing_body(gate_quat_rel)
    nearest_obstacles = _nearest_obstacle_features(obs, n_nearest_obstacles)
    # gates_visited is a *sensed* fraction (within sensor_range), not a passed fraction.
    visited_ratio = jp.mean(obs["gates_visited"].astype(jp.float32), axis=-1, keepdims=True)
    feats = [gate_delta_world, gate_delta_body, gate_facing_body]
    if lookahead_gates > 0:
        feats.append(_lookahead_gate_features(obs, lookahead_gates))
    # Velocity in the body frame (ang_vel already is), so the drone's motion is expressed egocentrically
    # and needs no absolute world heading to interpret.
    vel_body = _rotate_world_to_body(obs["vel"], obs["quat"])
    feats.extend([vel_body, obs["ang_vel"]])
    if gravity_obs:
        # Gravity direction expressed in the body frame: a compact attitude cue telling the policy
        # which way is down and how tilted it is (the drone's absolute attitude is otherwise only
        # implicit in the gate-relative terms). World down is (0, 0, -1).
        gravity_world = jp.broadcast_to(jp.asarray([0.0, 0.0, -1.0], dtype=obs["pos"].dtype), obs["pos"].shape)
        feats.append(_rotate_world_to_body(gravity_world, obs["quat"]))
    # Absolute altitude (world z): the one genuinely global cue the egocentric encoding drops. The
    # symmetry the body frame exploits (yaw + horizontal translation) does not include the vertical
    # axis (gravity breaks it), so height above the ground - relevant for the z<0 ground crash and the
    # absolute gate heights - is not recoverable from the gate-relative features alone.
    feats.append(obs["pos"][..., 2:3])
    feats.extend([nearest_obstacles, visited_ratio])
    if include_progress:
        # Fraction of gates already passed: target_gate is the index of the gate being flown to (=
        # number passed), normalized to [0, 1]; the post-finish sentinel (-1) maps to 1.0. Unlike
        # visited_ratio this is a true completion-progress signal, which helps the critic predict the
        # terminal success bonus and lets the actor modulate behavior near the finish.
        target_gate = obs["target_gate"]
        n_gates = obs["gates_pos"].shape[1]
        progress = jp.where(target_gate < 0, n_gates, target_gate).astype(jp.float32) / n_gates
        feats.append(progress[:, None])
    return jp.concatenate(feats, axis=-1)


@jax.jit
def compute_navigation_reward(
    prev_obs: dict[str, Array],
    next_obs: dict[str, Array],
    reward: Array,
    terminated: Array,
    gate_progress_coef: float,
    gate_pass_bonus: float,
    success_bonus: float,
    crash_penalty: float,
) -> tuple[Array, Array, Array, Array]:
    """Shape reward with simple gate progress, gate passes, success, and crash penalties."""
    prev_obs = _normalize_obs(prev_obs)
    next_obs = _normalize_obs(next_obs)
    prev_target = prev_obs["target_gate"]
    next_target = next_obs["target_gate"]
    n_gates = prev_obs["gates_pos"].shape[1]

    prev_gate_pos = _gather_by_index(prev_obs["gates_pos"], prev_target)
    prev_dist = jp.linalg.norm(prev_gate_pos - prev_obs["pos"], axis=-1)
    next_dist_to_prev = jp.linalg.norm(prev_gate_pos - next_obs["pos"], axis=-1)
    progress_reward = gate_progress_coef * (prev_dist - next_dist_to_prev)

    gate_advanced = next_target > prev_target
    success = (prev_target == (n_gates - 1)) & (next_target == -1)
    crash = terminated & ~success
    all_gate_passes = gate_passed(next_obs["pos"][:, None, :], prev_obs["pos"][:, None, :], prev_obs["gates_pos"], prev_obs["gates_quat"], (0.45, 0.45))
    target_gate_oh = jp.arange(n_gates)[None, :] == jp.clip(prev_target[:, None], 0, n_gates - 1)
    wrong_gate_passes = jp.any(all_gate_passes & ~target_gate_oh, axis=-1)

    shaped = reward + progress_reward
    shaped += gate_pass_bonus * gate_advanced.astype(jp.float32)
    shaped += success_bonus * success.astype(jp.float32)
    shaped -= crash_penalty * crash.astype(jp.float32)
    return shaped, gate_advanced, wrong_gate_passes, success


class NavigationObservation(VectorObservationWrapper):
    """Reduce the full environment observation to a compact navigation feature vector."""

    def __init__(self, env: VectorEnv, n_nearest_obstacles: int = 2, include_progress: bool = False, lookahead_gates: int = 0, gravity_obs: bool = False):
        super().__init__(env)
        self.n_nearest_obstacles = n_nearest_obstacles
        self.include_progress = include_progress
        self.lookahead_gates = lookahead_gates
        self.gravity_obs = gravity_obs
        sample = build_navigation_features(
            race_obs(env.unwrapped.data), self.n_nearest_obstacles, self.include_progress, self.lookahead_gates, self.gravity_obs
        )
        feature_dim = int(sample.shape[-1])
        self.single_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(feature_dim,), dtype=np.float32)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)

    def observations(self, observations: dict[str, Array]) -> Array:
        return build_navigation_features(observations, self.n_nearest_obstacles, self.include_progress, self.lookahead_gates, self.gravity_obs)


class PrevAction(VectorObservationWrapper):
    """Append the single previous action ``a_{t-1}`` to each observation.

    Matches the Swift paper, whose only temporal feedback into the policy is the action applied in
    the previous step (no observation/action history stack). The action attached to the observation
    returned at step ``t`` is exactly the action just applied (``a_t``), so that the observation
    consumed at the next decision carries ``a_{t-1}``. On the autoreset step (NEXT_STEP autoreset,
    where the env ignores the passed action), the previous action is zeroed so the first observation
    of a new episode carries no stale action.
    """

    def __init__(self, env: VectorEnv):
        super().__init__(env)
        obs_dim = int(np.prod(self.single_observation_space.shape))
        act_dim = int(np.prod(self.single_action_space.shape))
        self._act_dim = act_dim
        self._prev_done = jp.zeros(self.num_envs, dtype=bool)
        total_dim = obs_dim + act_dim
        self.single_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)

    def reset(self, *, seed: int | list[int] | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._prev_done = jp.zeros(self.num_envs, dtype=bool)
        zeros = jp.zeros((self.num_envs, self._act_dim), dtype=jp.float32)
        return jp.concatenate([obs, zeros], axis=-1), info

    def step(self, action: Array):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # If the previous step ended the episode, this is an autoreset step: the env ignored
        # ``action``, so the fresh observation carries a zero previous action. Otherwise attach the
        # just-applied action, which becomes ``a_{t-1}`` at the next decision.
        prev_action = jp.where(self._prev_done[:, None], jp.zeros_like(action), action)
        augmented_obs = jp.concatenate([obs, prev_action], axis=-1)
        self._prev_done = terminated | truncated
        return augmented_obs, reward, terminated, truncated, info


class NavigationReward(VectorRewardWrapper):
    """Add dense gate-to-gate shaping on top of the sparse base environment reward."""

    def __init__(self, env: VectorEnv, gate_progress_coef: float, gate_pass_bonus: float, success_bonus: float, crash_penalty: float):
        super().__init__(env)
        self.gate_progress_coef = gate_progress_coef
        self.gate_pass_bonus = gate_pass_bonus
        self.success_bonus = success_bonus
        self.crash_penalty = crash_penalty

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
        shaped_rewards, gate_passed, wrong_gate_passes, success = compute_navigation_reward(
            self._prev_obs, observations, rewards, terminations, self.gate_progress_coef, self.gate_pass_bonus, self.success_bonus, self.crash_penalty
        )
        # On the first transition of a new episode (the env autoreset on the previous step), the
        # prev/next observations straddle two episodes; the shaped terms (progress, gate passes,
        # etc.) are meaningless and would otherwise inject a spurious gate-pass bonus. Fall back to
        # the raw base reward for those transitions.
        valid = ~self._prev_done
        shaped_rewards = jp.where(valid, shaped_rewards, rewards)
        gate_passed = gate_passed & valid
        wrong_gate_passes = wrong_gate_passes & valid
        success = success & valid
        infos = dict(infos)
        infos["gate_passed"] = gate_passed
        infos["wrong_gate_passes"] = wrong_gate_passes
        infos["success"] = success
        self._prev_obs = observations
        self._prev_done = terminations | truncations
        return observations, shaped_rewards, terminations, truncations, infos


@partial(jax.jit, static_argnames=("control_mode",))
def _action_penalty(
    reward: Array, action: Array, last_action: Array, done: Array, act_coef: float, d_act_main_coef: float, d_act_aux_coef: float, control_mode: str
) -> tuple[Array, Array]:
    """Apply the energy + jerk penalties and return ``(reward, next_last_action)`` in one fused call.

    ``act_coef`` (collective-thrust energy) is only applied in attitude mode. ``next_last_action`` is
    zeroed for envs that just ended so the next episode's first jerk term is not computed across an
    episode boundary.
    """
    action_diff = action - last_action
    if control_mode == "attitude":
        reward = reward - act_coef * action[:, 3] ** 2  # thrust energy only
    reward = reward - d_act_main_coef * jp.sum(action_diff[:, :3] ** 2, axis=-1)
    reward = reward - d_act_aux_coef * jp.sum(action_diff[:, 3:] ** 2, axis=-1)
    next_last_action = jp.where(done[:, None], jp.zeros_like(action), action)
    return reward, next_last_action


class ActionPenalty(VectorRewardWrapper):
    """Penalize action magnitude and jerkiness.

    The policy action is ``[roll, pitch, yaw, thrust]`` in roughly ``[-1, 1]``. The energy penalty is
    applied to the **thrust channel only** (penalizing roll/pitch magnitude would discourage the very
    tilting needed to fly toward a gate). Smoothness (jerk) is penalized per channel: the "main"
    coefficient on roll/pitch/yaw and the "aux" coefficient on thrust.
    """

    def __init__(self, env: VectorEnv, act_coef: float = 0.01, d_act_main_coef: float = 0.25, d_act_aux_coef: float = 0.1, control_mode: str = "attitude"):
        super().__init__(env)
        self.act_coef = act_coef
        self.d_act_main_coef = d_act_main_coef
        self.d_act_aux_coef = d_act_aux_coef
        self.control_mode = control_mode
        self._last_action = jp.zeros((self.num_envs, self.single_action_space.shape[0]))

    def reset(self, *, seed: int | list[int] | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_action = jp.zeros((self.num_envs, self.single_action_space.shape[0]))
        return obs, info

    def step(self, action: Array):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward, self._last_action = _action_penalty(
            reward, action, self._last_action, terminated | truncated, self.act_coef, self.d_act_main_coef, self.d_act_aux_coef, self.control_mode
        )
        return obs, reward, terminated, truncated, info


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
        self.single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        # Jit the mapping with the (constant) thrust/angle bounds bound in, so the per-step call fuses
        # clip/scale/concat into one dispatch and never rebuilds ``angle_scale`` on the host. Reuses
        # attitude_setpoint_from_action so the training and inference paths cannot drift.
        self._actions_jit = jax.jit(
            partial(attitude_setpoint_from_action, thrust_min=self.thrust_min, thrust_max=self.thrust_max, max_angle=self.max_angle, max_yaw=self.max_yaw, xp=jp)
        )

    def actions(self, actions: Array) -> Array:
        return self._actions_jit(actions)


def make_envs(config: str = "level2.toml", num_envs: int = 256, jax_device: str = "cpu", torch_device: torch.device = torch.device("cpu"), args: Args | None = None) -> VectorEnv:
    """Create the standalone gate-navigation training environment."""
    if args is None:
        args = Args.create()
    config_data = load_config(Path(__file__).parents[2] / "config" / config)
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
        max_episode_steps=args.episode_step_limit,
        device=jax_device,
    )
    control_mode = str(config_data.env.control_mode)
    if control_mode == "attitude":
        env = AttitudeAction(env, config_data.sim.drone_model, max_angle=args.max_angle, max_yaw=args.max_yaw)
    elif control_mode == "state":
        env = NormalizeActions(env)
    else:
        raise ValueError(f"Unsupported control_mode: {control_mode}")
    env = NavigationReward(
        env, gate_progress_coef=args.gate_progress_coef, gate_pass_bonus=args.gate_pass_bonus, success_bonus=args.success_bonus, crash_penalty=args.crash_penalty
    )
    env = ActionPenalty(env, act_coef=args.act_coef, d_act_main_coef=args.d_act_main_coef, d_act_aux_coef=args.d_act_aux_coef, control_mode=control_mode)
    env = NavigationObservation(
        env, n_nearest_obstacles=args.n_nearest_obstacles, include_progress=args.progress_obs, lookahead_gates=args.lookahead_gates, gravity_obs=args.gravity_obs
    )
    env = PrevAction(env)
    env = JaxToTorch(env, torch_device)
    return env


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def grad_norm(parameters: list[nn.Parameter]) -> float:
    """Return the L2 norm of the current gradients for a parameter list."""
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    return float(torch.sqrt(sum(torch.sum(g * g) for g in grads)).item())


def clip_grad_group(parameters: list[nn.Parameter], max_norm: float) -> tuple[float, float]:
    """Clip a parameter group and return its pre/post-clip gradient norms."""
    pre_clip = grad_norm(parameters)
    if parameters:
        nn.utils.clip_grad_norm_(parameters, max_norm)
    post_clip = grad_norm(parameters)
    return pre_clip, post_clip


def build_checkpoint_payload(agent: nn.Module, args: Args, obs_shape: tuple[int, ...], action_shape: tuple[int, ...]) -> dict[str, Any]:
    """Serialize model weights together with the training args and architecture metadata."""
    return {
        "state_dict": agent.state_dict(),
        "args": asdict(args),
        "arch": {
            "obs_shape": tuple(int(x) for x in obs_shape),
            "action_shape": tuple(int(x) for x in action_shape),
            "actor_hdim": int(args.actor_hdim),
            "critic_hdim": int(args.critic_hdim),
            "init_logstd": float(args.init_logstd),
            "init_logstd_last": None if args.init_logstd_last is None else float(args.init_logstd_last),
        },
    }


class RunningMeanStd(nn.Module):
    """Running mean/variance (Welford) for observation normalization, stored as module buffers."""

    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-4):
        super().__init__()
        self._epsilon = epsilon
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(epsilon))

    @torch.no_grad()
    def reset(self) -> None:
        """Reset the running statistics to their initial state (mean 0, var 1, count epsilon).

        Used when warm-starting from a checkpoint: the restored statistics carry a large sample
        count, which would otherwise pin the normalization to the source run's observation
        distribution and prevent it from adapting to the env now being trained on.
        """
        self.mean.zero_()
        self.var.fill_(1.0)
        self.count.fill_(self._epsilon)

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

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        actor_hdim: int = 128,
        critic_hdim: int = 256,
        init_logstd: float = -1.0,
        init_logstd_last: float | None = 1.0,
    ):
        super().__init__()
        obs_dim = int(torch.tensor(obs_shape).prod())
        act_dim = int(torch.tensor(action_shape).prod())
        self.obs_rms = RunningMeanStd((obs_dim,))
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, critic_hdim)), nn.Tanh(), layer_init(nn.Linear(critic_hdim, critic_hdim)), nn.Tanh(), layer_init(nn.Linear(critic_hdim, 1), std=1.0)
        )
        # Final Tanh keeps the action mean in (-1, 1) with live gradients, so the policy does not
        # saturate against the wrapper's hard clip (which has zero gradient outside [-1, 1]).
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, actor_hdim)),
            nn.Tanh(),
            layer_init(nn.Linear(actor_hdim, actor_hdim)),
            nn.Tanh(),
            layer_init(nn.Linear(actor_hdim, act_dim), std=0.01),
            nn.Tanh(),
        )
        # Asymmetric exploration: small std on roll/pitch/yaw, large std on thrust (last channel) so
        # the drone explores lift and learns to take off instead of collapsing to a hover.
        actor_logstd = torch.full((1, act_dim), float(init_logstd))
        if init_logstd_last is not None:
            actor_logstd[0, -1] = float(init_logstd_last)
        self.actor_logstd = nn.Parameter(actor_logstd)

    @torch.no_grad()
    def reset_logstd(self, init_logstd: float, init_logstd_last: float | None) -> None:
        """Reset the action log-std to the initial exploration level (same layout as ``__init__``).

        Used when warm-starting from a checkpoint: ``load_state_dict`` restores the source policy's
        (often converged, near-deterministic) log-std, which leaves a resumed run on a harder task
        exploration-starved and prone to collapsing to a deterministic local optimum. Resetting
        re-opens exploration while keeping the loaded actor/critic weights.
        """
        self.actor_logstd.fill_(float(init_logstd))
        if init_logstd_last is not None:
            self.actor_logstd[0, -1] = float(init_logstd_last)

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
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, config=vars(args), save_code=True, monitor_gym=True)

    set_seeds(args.seed)
    print("Training on device:", device, "| Environment device:", jax_device)
    envs = make_envs(config=args.config, num_envs=args.num_envs, jax_device=jax_device, torch_device=device, args=args)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    checkpoint_payload = lambda: build_checkpoint_payload(agent, args, envs.single_observation_space.shape, envs.single_action_space.shape)

    agent = Agent(
        envs.single_observation_space.shape,
        envs.single_action_space.shape,
        actor_hdim=args.actor_hdim,
        critic_hdim=args.critic_hdim,
        init_logstd=args.init_logstd,
        init_logstd_last=args.init_logstd_last,
    ).to(device)
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.is_absolute():
            resume_path = Path(__file__).parent / resume_path
        ckpt = torch.load(resume_path, map_location=device)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        try:
            agent.load_state_dict(state)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to resume from '{resume_path}'. The checkpoint must match the current "
                f"obs/action dims and architecture (actor_hdim={args.actor_hdim}, "
                f"critic_hdim={args.critic_hdim})."
            ) from exc
        # load_state_dict also restored obs_rms (mean/var/count). Its count carries the source run's
        # full sample size, which would freeze observation normalization at the source-env statistics
        # and stop it adapting to the env now being trained on (e.g. level0 -> level2, where gate and
        # obstacle features have a wider, shifted distribution). Reset it so the normalizer
        # recalibrates from the current env's observations (update() runs before the first forward
        # pass each step, so there is no normalization shock).
        agent.obs_rms.reset()
        # load_state_dict also restored actor_logstd, i.e. the source policy's (often converged,
        # near-deterministic) exploration level. Inheriting it leaves a resume on a harder task
        # exploration-starved, which collapses to a deterministic local optimum (grad norm and
        # clipfrac -> 0). Reset it to the configured initial exploration so the warm-started weights
        # can still explore.
        agent.reset_logstd(args.init_logstd, args.init_logstd_last)
        print(
            f"Resumed agent weights from {resume_path} "
            f"(obs normalization + action log-std reset to re-adapt and re-explore)"
        )
    optimizer = optim.AdamW(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    base_dir = model_path.parent if model_path is not None else Path(__file__).parent
    if wandb_enabled and wandb.run is not None and wandb.run.name:
        run_dir_name = _safe_run_name(wandb.run.name)
    else:
        run_dir_name = f"local_{time.strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = base_dir / f"ppo-nav-training-{run_dir_name}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)  # terminations only, cuts the bootstrap
    dones_full = torch.zeros((args.num_steps, args.num_envs)).to(device)  # term | trunc, cuts GAE accumulation
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    train_start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)
    next_done = torch.zeros(args.num_envs, device=device)  # term | trunc, for episode bookkeeping
    next_termination = torch.zeros(args.num_envs, device=device)  # terminations only, for GAE
    sum_rewards = torch.zeros(args.num_envs, device=device)
    episode_gate_passes = torch.zeros(args.num_envs, device=device)
    episode_gate_hits = torch.zeros(args.num_envs, device=device)
    episode_obstacle_hits = torch.zeros(args.num_envs, device=device)
    episode_wrong_gate_passes = torch.zeros(args.num_envs, device=device)
    episode_success = torch.zeros(args.num_envs, device=device)
    reward_hist: list[float] = []
    actor_params = list(agent.actor_mean.parameters()) + [agent.actor_logstd]
    critic_params = list(agent.critic.parameters())

    for iteration in range(1, args.num_iterations + 1):
        iter_start = time.time()
        iter_gate_passes = 0.0
        iter_gate_hits = 0.0
        iter_obstacle_hits = 0.0
        iter_gate_pass_episodes = 0.0
        iter_gate_hit_episodes = 0.0
        iter_obstacle_hit_episodes = 0.0
        iter_wrong_gate_passes = 0.0
        iter_wrong_gate_pass_episodes = 0.0
        iter_success_episodes = 0.0
        actor_grad_norm_pre = 0.0
        actor_grad_norm_post = 0.0
        critic_grad_norm_pre = 0.0
        critic_grad_norm_post = 0.0
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = max(args.min_learning_rate, frac * args.learning_rate)

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_termination  # GAE cuts bootstrap only on true terminations
            dones_full[step] = next_done  # post-episode (autoreset) state; cuts GAE accumulation + masks loss

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
            gate_passes = torch.as_tensor(infos["gate_passed"], device=device).float()
            gate_hits = torch.as_tensor(infos["gate_hits"], device=device).float()
            obstacle_hits = torch.as_tensor(infos["obstacle_hits"], device=device).float()
            wrong_gate_passes = torch.as_tensor(infos["wrong_gate_passes"], device=device).float()
            success = torch.as_tensor(infos["success"], device=device).float()

            rewards[step] = reward
            sum_rewards += reward
            episode_gate_passes += gate_passes
            episode_gate_hits += gate_hits
            episode_obstacle_hits += obstacle_hits
            episode_wrong_gate_passes += wrong_gate_passes
            episode_success = torch.maximum(episode_success, success)
            iter_gate_passes += gate_passes.sum().item()
            iter_gate_hits += gate_hits.sum().item()
            iter_obstacle_hits += obstacle_hits.sum().item()
            iter_wrong_gate_passes += wrong_gate_passes.sum().item()
            next_termination = terminations
            next_done = terminations | truncations

            if wandb_enabled and next_done.any():
                done_rewards = sum_rewards[next_done.bool()]
                done_gate_passes = episode_gate_passes[next_done.bool()]
                done_gate_hits = episode_gate_hits[next_done.bool()]
                done_obstacle_hits = episode_obstacle_hits[next_done.bool()]
                done_wrong_gate_passes = episode_wrong_gate_passes[next_done.bool()]
                done_success = episode_success[next_done.bool()]
                iter_gate_pass_episodes += (done_gate_passes > 0).float().sum().item()
                iter_gate_hit_episodes += (done_gate_hits > 0).float().sum().item()
                iter_obstacle_hit_episodes += (done_obstacle_hits > 0).float().sum().item()
                iter_wrong_gate_pass_episodes += (done_wrong_gate_passes > 0).float().sum().item()
                iter_success_episodes += done_success.sum().item()
                for r, gp, g, o, w, s in zip(done_rewards, done_gate_passes, done_gate_hits, done_obstacle_hits, done_wrong_gate_passes, done_success):
                    reward_hist.append(r.item())
                    wandb.log(
                        {
                            "train/reward": r.item(),
                            "train/gate_passes": gp.item(),
                            "train/gate_hits": g.item(),
                            "train/obstacle_hits": o.item(),
                            "train/wrong_gate_passes": w.item(),
                            "train/success": s.item(),
                        },
                        step=global_step,
                    )
                sum_rewards[next_done.bool()] = 0
                episode_gate_passes[next_done.bool()] = 0
                episode_gate_hits[next_done.bool()] = 0
                episode_obstacle_hits[next_done.bool()] = 0
                episode_wrong_gate_passes[next_done.bool()] = 0
                episode_success[next_done.bool()] = 0
            elif next_done.any():
                done_gate_passes = episode_gate_passes[next_done.bool()]
                done_gate_hits = episode_gate_hits[next_done.bool()]
                done_obstacle_hits = episode_obstacle_hits[next_done.bool()]
                done_wrong_gate_passes = episode_wrong_gate_passes[next_done.bool()]
                done_success = episode_success[next_done.bool()]
                iter_gate_pass_episodes += (done_gate_passes > 0).float().sum().item()
                iter_gate_hit_episodes += (done_gate_hits > 0).float().sum().item()
                iter_obstacle_hit_episodes += (done_obstacle_hits > 0).float().sum().item()
                iter_wrong_gate_pass_episodes += (done_wrong_gate_passes > 0).float().sum().item()
                iter_success_episodes += done_success.sum().item()
                sum_rewards[next_done.bool()] = 0
                episode_gate_passes[next_done.bool()] = 0
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
                    next_nonterm = 1.0 - next_termination.float()  # cuts bootstrap on termination
                    next_nondone = 1.0 - next_done.float()  # cuts GAE accumulation on any episode end
                    nextvalues = next_value
                else:
                    next_nonterm = 1.0 - dones[t + 1]
                    next_nondone = 1.0 - dones_full[t + 1]
                    nextvalues = values[t + 1]
                # Under NEXT_STEP autoreset the obs at the done step is the true terminal state, so a
                # truncation keeps its bootstrap (next_nonterm=1) while still ending the GAE chain
                # (next_nondone=0), so advantage never leaks across an episode boundary.
                delta = rewards[t] + args.gamma * nextvalues * next_nonterm - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_nondone * lastgaelam
            returns = advantages + values
            valids = 1.0 - dones_full  # 0 for autoreset pseudo-transitions, which carry no real signal

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_valids = valids.reshape(-1)

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
                actor_grad_norm_pre, actor_grad_norm_post = clip_grad_group(actor_params, args.max_grad_norm)
                critic_grad_norm_pre, critic_grad_norm_post = clip_grad_group(critic_params, args.max_grad_norm)
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
                    "gradients/actor_norm": actor_grad_norm_pre,
                    "gradients/actor_norm_clipped": actor_grad_norm_post,
                    "gradients/critic_norm": critic_grad_norm_pre,
                    "gradients/critic_norm_clipped": critic_grad_norm_post,
                    "charts/SPS": int(global_step / max(1e-6, (time.time() - train_start_time))),
                    "charts/gate_pass_events": iter_gate_passes,
                    "charts/gate_hit_events": iter_gate_hits,
                    "charts/obstacle_hit_events": iter_obstacle_hits,
                    "charts/gate_pass_episodes": iter_gate_pass_episodes,
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
            f" | gate_passes={iter_gate_passes:.0f} gate_hits={iter_gate_hits:.0f} obstacle_hits={iter_obstacle_hits:.0f}"
            f" | wrong_gate_passes={iter_wrong_gate_passes:.0f}"
            f" | gate_pass_eps={iter_gate_pass_episodes:.0f} gate_hit_eps={iter_gate_hit_episodes:.0f} obstacle_hit_eps={iter_obstacle_hit_episodes:.0f}"
            f" wrong_gate_pass_eps={iter_wrong_gate_pass_episodes:.0f} success_eps={iter_success_episodes:.0f}"
        )
        if model_path is not None and args.checkpoint_every_iterations > 0:
            if iteration % args.checkpoint_every_iterations == 0:
                ckpt_path = checkpoint_dir / f"{model_path.stem}_iter{iteration:03d}.ckpt"
                torch.save(checkpoint_payload(), ckpt_path)

    print(f"Training for {global_step} steps took {time.time() - train_start_time:.2f} seconds.")
    if model_path is not None:
        final_model_path = checkpoint_dir / model_path.name
        torch.save(checkpoint_payload(), final_model_path)
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
