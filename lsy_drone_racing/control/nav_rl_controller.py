"""Controller that runs the standalone navigation PPO checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import os

from drone_models.core import load_params

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.train_nav_rl import (
    Agent,
    Args,
    attitude_setpoint_from_action,
    build_navigation_features,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray



class NavRLController(Controller):
    """Navigation controller backed by the PPO policy for either state or attitude control."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self._control_mode = str(config.env.control_mode)
        assert self._control_mode in {"attitude", "state"}, (
            f"Unsupported control_mode: {self._control_mode}"
        )
        self._device = torch.device("cpu")
        self._finished = False
        # Attitude action scaling must match training (train_nav_rl.AttitudeAction / Args).
        if self._control_mode == "attitude":
            params = load_params("so_rpy", config.sim.drone_model)
            self._thrust_min = float(params["thrust_min"]) * 4.0
            self._thrust_max = float(params["thrust_max"]) * 4.0
            self._max_angle = float(getattr(config.controller, "max_angle", Args.max_angle))
            self._max_yaw = float(getattr(config.controller, "max_yaw", Args.max_yaw))
        act_dim = 4 if self._control_mode == "attitude" else 13
        self._act_dim = act_dim
        self._n_nearest_obstacles = int(getattr(config.controller, "n_nearest_obstacles", 2))
        self._include_progress = bool(getattr(config.controller, "progress_obs", False))
        self._lookahead_gates = int(getattr(config.controller, "lookahead_gates", Args.lookahead_gates))
        self._gravity_obs = bool(getattr(config.controller, "gravity_obs", Args.gravity_obs))
        # Single previous action a_{t-1} appended to the observation (matches train_nav_rl.PrevAction).
        self._prev_action = np.zeros(act_dim, dtype=np.float32)

        # Resolve a checkpoint first (matched on act_dim and the obs layout implied by *its own*
        # training args), then adopt those feature flags so the observation builder reproduces the
        # checkpoint's obs layout. This ordering matters: building obs_dim from the config defaults
        # before adopting the checkpoint's args would reject a checkpoint that is actually loadable.
        model_path, checkpoint = self._resolve_model(config, obs=obs, act_dim=act_dim)
        state = self._extract_state_dict(checkpoint)
        self._apply_ckpt_args(self._extract_args(checkpoint))
        obs_dim = int(self._obs_tensor(obs).shape[-1])
        arch = checkpoint.get("arch")
        if isinstance(arch, dict) and arch.get("obs_shape") is not None:
            expected_obs_dim = int(arch["obs_shape"][0])
            if obs_dim != expected_obs_dim:
                raise RuntimeError(
                    f"Checkpoint '{model_path}' expects obs_dim={expected_obs_dim}, but controller built "
                    f"obs_dim={obs_dim}. Check the saved args versus the current observation builder."
                )
        actor_hdim, critic_hdim, init_logstd, init_logstd_last = self._resolve_architecture(checkpoint, state)
        self._agent = Agent(
            (obs_dim,),
            (act_dim,),
            actor_hdim=actor_hdim,
            critic_hdim=critic_hdim,
            init_logstd=init_logstd,
            init_logstd_last=init_logstd_last,
        ).to(self._device)
        try:
            self._agent.load_state_dict(state)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to load checkpoint '{model_path}'. "
                f"Expected nav policy with obs_dim={obs_dim} and act_dim={act_dim}."
            ) from exc
        self._agent.eval()

        # Optional debug UI: publish each control step over ZMQ when DEBUG_UI_ENABLE is set.
        # Lazy + guarded so deployments without the env var (or without pyzmq) are unaffected.
        self._t = 0
        self._dbg = None
        try:
            from lsy_drone_racing.debug_ui.publisher import get_publisher

            self._dbg = get_publisher()
        except Exception:  # noqa: BLE001 - the debug UI must never break the controller.
            self._dbg = None

    def _extract_state_dict(self, checkpoint: dict) -> dict:
        """Return the model state dict from either a checkpoint bundle or a raw state dict."""
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            return checkpoint["state_dict"]
        return checkpoint

    def _extract_args(self, checkpoint: dict) -> dict | None:
        """Return serialized training args when present."""
        args = checkpoint.get("args")
        return args if isinstance(args, dict) else None

    def _feature_flags(self, ckpt_args: dict | None) -> tuple[int, bool, int, bool]:
        """Resolve the observation feature flags, preferring the checkpoint's training args.

        Missing keys or explicit ``None`` values fall back to the controller's current settings, so
        a checkpoint only overrides the flags it actually recorded.
        """
        n, p, lk, g = (
            self._n_nearest_obstacles,
            self._include_progress,
            self._lookahead_gates,
            self._gravity_obs,
        )
        if ckpt_args:
            if ckpt_args.get("n_nearest_obstacles") is not None:
                n = int(ckpt_args["n_nearest_obstacles"])
            if ckpt_args.get("progress_obs") is not None:
                p = bool(ckpt_args["progress_obs"])
            if ckpt_args.get("lookahead_gates") is not None:
                lk = int(ckpt_args["lookahead_gates"])
            if ckpt_args.get("gravity_obs") is not None:
                g = bool(ckpt_args["gravity_obs"])
        return n, p, lk, g

    def _apply_ckpt_args(self, ckpt_args: dict | None) -> None:
        """Adopt the feature flags the checkpoint was trained with."""
        self._n_nearest_obstacles, self._include_progress, self._lookahead_gates, self._gravity_obs = (
            self._feature_flags(ckpt_args)
        )

    def _obs_dim_for_ckpt(
        self, obs: dict[str, NDArray[np.floating]], ckpt_args: dict | None, act_dim: int
    ) -> int:
        """Compute the obs dimension implied by a checkpoint's feature flags and the current builder."""
        n, p, lk, g = self._feature_flags(ckpt_args)
        obs_jax = {k: np.asarray(v)[None, ...] for k, v in obs.items()}
        base = build_navigation_features(
            obs_jax, n_nearest_obstacles=n, include_progress=p, lookahead_gates=lk, gravity_obs=g
        )
        return int(base.shape[-1]) + act_dim

    def _resolve_architecture(self, checkpoint: dict, state: dict) -> tuple[int, int, float, float | None]:
        """Recover architecture settings from checkpoint metadata, falling back to old inference."""
        arch = checkpoint.get("arch")
        if isinstance(arch, dict):
            actor_hdim = int(arch.get("actor_hdim", state["actor_mean.0.weight"].shape[0]))
            critic_hdim = int(arch.get("critic_hdim", state["critic.0.weight"].shape[0]))
            init_logstd = float(arch.get("init_logstd", -1.0))
            init_logstd_last = arch.get("init_logstd_last", 1.0)
            if init_logstd_last is not None:
                init_logstd_last = float(init_logstd_last)
            return actor_hdim, critic_hdim, init_logstd, init_logstd_last
        actor_hdim = int(state["actor_mean.0.weight"].shape[0])
        critic_hdim = int(state["critic.0.weight"].shape[0])
        init_logstd = float(state["actor_logstd"][0, 0].item())
        init_logstd_last = float(state["actor_logstd"][0, -1].item())
        return actor_hdim, critic_hdim, init_logstd, init_logstd_last

    def _checkpoint_matches(self, checkpoint: dict, obs_dim: int, act_dim: int) -> bool:
        """Check whether a checkpoint matches the current input/output dimensions."""
        arch = checkpoint.get("arch")
        if isinstance(arch, dict):
            action_shape = arch.get("action_shape")
            if action_shape is not None and int(action_shape[0]) != act_dim:
                return False
        state = self._extract_state_dict(checkpoint)
        required = {"actor_logstd", "critic.0.weight", "actor_mean.4.weight"}
        if not required.issubset(state.keys()):
            return False
        if tuple(state["actor_logstd"].shape) != (1, act_dim):
            return False
        if tuple(state["critic.0.weight"].shape)[1] != obs_dim:
            return False
        if tuple(state["actor_mean.4.weight"].shape)[0] != act_dim:
            return False
        return True

    def _load_checkpoint(self, path: Path) -> dict | None:
        """Load a checkpoint state dict, returning None on failure."""
        try:
            return torch.load(path, map_location="cpu")
        except Exception:
            return None

    def _resolve_model(
        self, config: dict, obs: dict[str, NDArray[np.floating]], act_dim: int
    ) -> tuple[Path, dict]:
        """Resolve and load a compatible checkpoint from config, env var, or local runs.

        Each candidate is validated against the obs dimension implied by *its own* training args, so
        a checkpoint trained with different feature flags than the current config defaults still
        loads (the controller then adopts those flags via :meth:`_apply_ckpt_args`).
        """
        # Explicit choices (model_path / NAV_RL_CKPT) are loaded and validated against their own args.
        for source, raw, base_dir in (
            ("model_path", getattr(config.controller, "model_path", None), Path(__file__).parent),
            ("NAV_RL_CKPT", os.environ.get("NAV_RL_CKPT"), Path.cwd()),
        ):
            if not raw:
                continue
            path = Path(raw)
            if not path.is_absolute():
                path = base_dir / path
            if not path.exists():
                raise FileNotFoundError(f"{source} does not exist: {path}")
            checkpoint = self._load_checkpoint(path)
            if checkpoint is None:
                raise RuntimeError(f"{source} '{path}' could not be loaded as a checkpoint.")
            obs_dim = self._obs_dim_for_ckpt(obs, self._extract_args(checkpoint), act_dim)
            if not self._checkpoint_matches(checkpoint, obs_dim=obs_dim, act_dim=act_dim):
                raise RuntimeError(
                    f"{source} '{path}' does not match the current controller architecture "
                    f"(obs_dim={obs_dim} for its feature flags, act_dim={act_dim})."
                )
            return path, checkpoint

        control_dir = Path(__file__).parent
        candidates = sorted(
            control_dir.rglob("ppo_nav_drone_racing.ckpt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for path in candidates:
            checkpoint = self._load_checkpoint(path)
            if checkpoint is None:
                continue
            obs_dim = self._obs_dim_for_ckpt(obs, self._extract_args(checkpoint), act_dim)
            if self._checkpoint_matches(checkpoint, obs_dim=obs_dim, act_dim=act_dim):
                return path, checkpoint
        if not candidates:
            raise FileNotFoundError(
                "No ppo_nav_drone_racing.ckpt found under lsy_drone_racing/control. "
                "Train the policy first or set controller.model_path."
            )
        raise FileNotFoundError(
            "No compatible ppo_nav_drone_racing.ckpt found under lsy_drone_racing/control. "
            f"Expected act_dim={act_dim} with an obs layout the current feature builder can "
            "reproduce. Train the policy first or set controller.model_path to a matching checkpoint."
        )

    def _base_obs(self, obs: dict[str, NDArray[np.floating]]) -> np.ndarray:
        """Compute the compact navigation feature vector for the current observation."""
        obs_jax = {k: np.asarray(v)[None, ...] for k, v in obs.items()}
        features = build_navigation_features(
            obs_jax,
            n_nearest_obstacles=self._n_nearest_obstacles,
            include_progress=self._include_progress,
            lookahead_gates=self._lookahead_gates,
            gravity_obs=self._gravity_obs,
        )
        return np.array(features[0], copy=True, dtype=np.float32)

    def _obs_tensor(self, obs: dict[str, NDArray[np.floating]]) -> torch.Tensor:
        """Convert env observations into the feature vector used during training (nav features + a_{t-1})."""
        base_obs = self._base_obs(obs)
        features = np.concatenate([base_obs, self._prev_action], axis=-1)
        return torch.as_tensor(features, dtype=torch.float32, device=self._device)

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Return either a 4-D attitude command or 13-D state command from the PPO policy."""
        if int(np.asarray(obs["target_gate"])) < 0:
            self._finished = True

        # Build the observation with the current previous action (a_{t-1}) before stepping the policy.
        obs_tensor = self._obs_tensor(obs).unsqueeze(0)
        with torch.no_grad():
            action, _, _, _ = self._agent.get_action_and_value(obs_tensor, deterministic=True)
        # The policy emits the compact navigation action; store it as a_{t-1} for the next step
        # (matching train_nav_rl.PrevAction, which appends the compact action, not the mapped command).
        action_np = action.squeeze(0).cpu().numpy().astype(np.float32)
        self._prev_action = action_np
        if self._control_mode == "attitude":
            # Map the policy action to the attitude command the env expects (same transform as the
            # AttitudeAction wrapper used in training).
            command = attitude_setpoint_from_action(
                action_np, self._thrust_min, self._thrust_max, self._max_angle, self._max_yaw, xp=np
            )
            command = command.astype(np.float32)
        else:
            command = action_np.astype(np.float32)
        if self._dbg is not None:
            self._dbg.publish(self._t, obs, command, self._prev_action)
        self._t += 1
        return command

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Signal completion once the episode is done."""
        self._finished = self._finished or terminated or truncated
        return self._finished

    def episode_callback(self):
        """Reset the finished flag and previous-action buffer between episodes."""
        self._finished = False
        self._prev_action = np.zeros(self._act_dim, dtype=np.float32)
        self._t = 0
