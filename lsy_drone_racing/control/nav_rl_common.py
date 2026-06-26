"""Shared, low-dependency helpers for the navigation PPO policy.

This module holds everything both the training script (``train_nav_rl.py``) and the inference
controller (``nav_rl_controller.py``) need: the training-args dataclass, the JAX observation-feature
builder (and its quaternion/rotation helpers), the action mapping, and the Torch actor-critic. It is
deliberately kept to a minimal dependency set (``numpy``/``jax``/``torch`` only) so importing it for
**deployment** does not drag in the training-only stack (wandb, fire, optimizer, the crazyflow
training env, the gymnasium vector wrappers).
"""

from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jp
import numpy as np
import torch
import torch.nn as nn
from jax import Array
from torch import Tensor
from torch.distributions.normal import Normal


@dataclass
class Args:
    """Configuration for the standalone navigation experiment."""

    seed: int = 42
    cuda: bool = True
    jax_device: str = "gpu"
    wandb_project_name: str = "ADR-PPO-Navigation"
    wandb_entity: str | None = None
    config: str = "level_competition_train.toml"
    resume_from: str | None = None  # path to a checkpoint (.ckpt) to resume model weights from
    continue_run: bool = False  # with resume_from: true-continue (keep obs_rms, log-std, optimizer state) instead of a transfer warm-start (which resets them)

    total_timesteps: int = 50_000_000
    learning_rate: float = 3e-4  # 1.5e-3 works on this problem too
    num_envs: int = 2048
    num_steps: int = 16
    anneal_lr: bool = True
    min_learning_rate: float = 1e-5
    lr_anneal_frac: float = 0.8  # fraction of training over which LR anneals down to min_learning_rate; held at the floor afterwards (e.g. 0.5 = reach min at the halfway point)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 8
    update_epochs: int = 4
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
    lookahead_gates: int = 0  # number of upcoming gates (after the target) to include, in the drone body frame
    gravity_obs: bool = False  # append the gravity direction in the drone body frame (attitude/tilt cue)
    gate_orient_rotmat: bool = True  # gate orientation as a 9-D rotation matrix (True) or a 3-D facing normal (False)
    vel_body_obs: bool = False  # express velocity in the body frame (True) or the world frame (False)
    altitude_obs: bool = False  # append the drone's absolute world-z height
    checkpoint_every_iterations: int = 10
    episode_step_limit: int = 1500

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

    @staticmethod
    def create(**kwargs: Any) -> "Args":
        args = Args(**kwargs)
        if not 0.0 < args.lr_anneal_frac <= 1.0:
            raise ValueError(f"lr_anneal_frac must be in (0, 1], got {args.lr_anneal_frac}")
        args.batch_size = int(args.num_envs * args.num_steps)
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
        args.num_iterations = max(1, args.total_timesteps // args.batch_size)
        print(f"Created args with {args.batch_size=}, {args.minibatch_size=}, {args.num_iterations=}")
        return args


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


def _lookahead_gate_features(obs: dict[str, Array], n_lookahead: int, gate_orient_rotmat: bool) -> Array:
    """Body-frame relative position + relative orientation of the next ``n_lookahead`` gates.

    The features are egocentric (drone-local), like the target-gate terms, so they are rotation
    invariant and directly actionable. On a randomized track the policy cannot memorize the layout,
    so seeing the upcoming gate(s) is what lets it plan a line through the current gate that sets up
    the next one. Gates past the last (no successor) are zeroed; the policy reads all-zeros as "no
    further gate". The upcoming gate's nominal pose is available from t=0 even under a finite
    sensor_range, so this is real planning information, not a privileged leak. The orientation
    representation matches the target gate (``gate_orient_rotmat``): 9-D rotmat or 3-D facing normal.
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
        quat_rel = _quat_multiply(_quat_conjugate(obs["quat"]), gate_quat)
        orient = _quat_to_rotmat(quat_rel) if gate_orient_rotmat else _gate_facing_body(quat_rel)
        feats.append(delta_body * valid)
        feats.append(orient * valid)
    return jp.concatenate(feats, axis=-1)


@partial(jax.jit, static_argnames=("n_nearest_obstacles", "include_progress", "lookahead_gates", "gravity_obs", "gate_orient_rotmat", "vel_body_obs", "altitude_obs"))
def build_navigation_features(
    obs: dict[str, Array],
    n_nearest_obstacles: int,
    include_progress: bool = False,
    lookahead_gates: int = 0,
    gravity_obs: bool = False,
    gate_orient_rotmat: bool = True,
    vel_body_obs: bool = False,
    altitude_obs: bool = False,
) -> Array:
    """Build a compact navigation observation from the privileged environment state.

    Every structural choice is a toggle so a checkpoint's saved obs layout can be reconstructed
    exactly. The defaults (rotmat gate orientation, world-frame velocity, no gravity/altitude/
    lookahead/progress) reproduce the obs=34 known-good baseline.
    """
    obs = _normalize_obs(obs)
    gate_delta_world, gate_delta_body, gate_quat_rel = _target_gate_metrics(obs)
    # Gate orientation: 9-D relative rotation matrix (default) or its 3-D pass-through facing normal.
    gate_orient = _quat_to_rotmat(gate_quat_rel) if gate_orient_rotmat else _gate_facing_body(gate_quat_rel)
    nearest_obstacles = _nearest_obstacle_features(obs, n_nearest_obstacles)
    # gates_visited is a *sensed* fraction (within sensor_range), not a passed fraction.
    visited_ratio = jp.mean(obs["gates_visited"].astype(jp.float32), axis=-1, keepdims=True)
    feats = [gate_delta_world, gate_delta_body, gate_orient]
    if lookahead_gates > 0:
        feats.append(_lookahead_gate_features(obs, lookahead_gates, gate_orient_rotmat))
    # Velocity in the world frame (default) or the body frame (egocentric, ang_vel already is).
    vel = _rotate_world_to_body(obs["vel"], obs["quat"]) if vel_body_obs else obs["vel"]
    feats.extend([vel, obs["ang_vel"]])
    if gravity_obs:
        # Gravity direction expressed in the body frame: a compact attitude cue telling the policy
        # which way is down and how tilted it is (the drone's absolute attitude is otherwise only
        # implicit in the gate-relative terms). World down is (0, 0, -1).
        gravity_world = jp.broadcast_to(jp.asarray([0.0, 0.0, -1.0], dtype=obs["pos"].dtype), obs["pos"].shape)
        feats.append(_rotate_world_to_body(gravity_world, obs["quat"]))
    if altitude_obs:
        # Absolute altitude (world z): the one genuinely global cue the egocentric encoding drops. The
        # symmetry the body frame exploits (yaw + horizontal translation) does not include the vertical
        # axis (gravity breaks it), so height above the ground - relevant for the z<0 ground crash and
        # the absolute gate heights - is not recoverable from the gate-relative features alone.
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


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


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
