"""Butterfly Optimisation Algorithm (BOA) for tuning Q and Rw matrices.

Optimises 7 independent weight parameters that map to the full Q (12×12) and
Rw (4×4) diagonal matrices used in attitude_mpc_lin_boa.py.

The cost function minimised is:

    cost = w1 * IAE  +  w2 * ITAE  +  w3 * U2

where
    IAE  = integral of absolute position error  [m·s]
    ITAE = integral of time-weighted absolute position error  [m·s²]
    U2   = integral of squared control effort  [–]

Search space (7 parameters):
    x = [q_pos_xy, q_pos_z, q_rpy, q_vel, q_drpy, r_att, r_thrust]

    q_pos_xy  → Q[0], Q[1]  (horizontal position weights)
    q_pos_z   → Q[2]         (vertical position weight)
    q_rpy     → Q[3:6]       (orientation state weights)
    q_vel     → Q[6:9]       (velocity weights)
    q_drpy    → Q[9:12]      (angular-rate weights)
    r_att     → Rw[0:3]      (roll/pitch/yaw command weights)
    r_thrust  → Rw[3]        (thrust command weight)

Usage
-----
    python scripts/boa_tune.py                      # uses level0_attitude.toml
    python scripts/boa_tune.py --config level2_attitude.toml
    python scripts/boa_tune.py --n_iter 30 --n_butterflies 8
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import gymnasium
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

sys.path.insert(0, str(Path(__file__).parents[1]))

from lsy_drone_racing.utils import load_config, load_controller

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BOA hyper-parameters
# ---------------------------------------------------------------------------
_N_BUTTERFLIES_DEFAULT = 5
_N_ITERATIONS_DEFAULT = 20
_P_SWITCH = 0.8   # probability of global-search move  (vs local)
_C_SENSORY = 0.2  # step-size scale in normalised [0,1] space
_A_POWER = 0.1    # power exponent on normalised stimulus intensity
_MIN_FRAGRANCE = 0.02  # floor so the current-best butterfly still explores
_SEED = 42

# ---------------------------------------------------------------------------
# Cost-function weights   cost = completion_time + w1*IAE + w2*ITAE + w3*U2
#
# completion_time dominates (~10 s for a good run); the tracking terms provide
# a gradient signal when the drone doesn't finish.
# ---------------------------------------------------------------------------
W1 = 0.1    # IAE weight   (m·s)
W2 = 0.05   # ITAE weight  (m·s²)
W3 = 0.001  # U² weight

# Time (s) assigned as completion_time when the drone does not finish the track.
# Should exceed any achievable real completion time so DNF is always penalised.
DNF_TIME = 30.0

# Extra penalty on top of DNF_TIME for crashing (vs simply running out of time).
CRASH_PENALTY = 20.0

# Hard wall-clock timeout per episode.
MAX_EVAL_TIME = 30.0

# ---------------------------------------------------------------------------
# Search-space bounds
# ---------------------------------------------------------------------------
PARAM_NAMES = ["q_pos_xy", "q_pos_z", "q_rpy", "q_vel", "q_drpy", "r_att", "r_thrust", "t_total"]
BOUNDS_LOW  = np.array([10.0,  20.0,  0.1,   1.0,  0.1,  0.01,   1.0,  3.0])
BOUNDS_HIGH = np.array([500.0, 1000.0, 20.0, 200.0, 50.0, 10.0, 200.0, 15.0])

# Known-good prior from a previous BOA run.  Butterfly 0 is seeded here instead of
# randomly so the search refines an already-decent region of parameter space.
# Set to None to revert to a fully random initialisation.
PRIOR: list[float] | None = [388.88, 449.68, 17.17, 139.65, 4.79, 9.75, 152.32, 9.5]


# ---------------------------------------------------------------------------
# Helper: map 7-vector → Q diag (len 12) and Rw diag (len 4)
# ---------------------------------------------------------------------------
def params_to_diags(x: np.ndarray) -> tuple[list[float], list[float], float]:
    """Expand the 8-parameter BOA vector into Q diagonals, Rw diagonals, and t_total."""
    q_pos_xy, q_pos_z, q_rpy, q_vel, q_drpy, r_att, r_thrust, t_total = x.tolist()
    q_diag = [
        q_pos_xy, q_pos_xy, q_pos_z,
        q_rpy, q_rpy, q_rpy,
        q_vel, q_vel, q_vel,
        q_drpy, q_drpy, q_drpy,
    ]
    rw_diag = [r_att, r_att, r_att, r_thrust]
    return q_diag, rw_diag, t_total


# ---------------------------------------------------------------------------
# Single-episode evaluator
# ---------------------------------------------------------------------------
def evaluate(
    x: np.ndarray,
    env,
    config,
    controller_cls,
) -> float:
    """Run one episode with weight vector *x*, return scalar cost.

    cost = completion_time + W1*IAE + W2*ITAE + W3*U2

    completion_time is the actual flight time when all gates are passed.
    DNF_TIME is used if the drone does not finish within MAX_EVAL_TIME seconds.
    An extra CRASH_PENALTY is added on top when the drone crashes.
    """
    q_diag, rw_diag, t_total = params_to_diags(x)
    dt = 1.0 / config.env.freq
    max_steps = int(config.env.freq * MAX_EVAL_TIME)

    obs, info = env.reset()
    ctrl = controller_cls(obs, info, config, q_diag=q_diag, rw_diag=rw_diag, t_total=t_total)

    iae = itae = u2 = 0.0
    completed = False
    crashed = False
    completion_step = max_steps

    for step in range(max_steps):
        pos = np.asarray(obs["pos"]).squeeze()
        i_ref = min(ctrl._tick, ctrl._tick_max)
        pos_ref = ctrl._waypoints_pos[i_ref]
        err = float(np.linalg.norm(pos - pos_ref))

        t = step * dt
        iae  += err * dt
        itae += t * err * dt

        action = ctrl.compute_control(obs, info)
        u2 += float(np.sum(action ** 2)) * dt

        obs, reward, terminated, truncated, info = env.step(action)
        ctrl.step_callback(action, obs, reward, terminated, truncated, info)

        if int(np.asarray(obs["target_gate"]).squeeze()) == -1:
            completed = True
            completion_step = step + 1
            break
        if terminated:
            crashed = True
            break
        if truncated:
            break

    flight_time = completion_step * dt if completed else DNF_TIME
    cost = flight_time + W1 * iae + W2 * itae + W3 * u2
    if crashed:
        cost += CRASH_PENALTY
    return cost


# ---------------------------------------------------------------------------
# BOA optimiser
# ---------------------------------------------------------------------------
def run_boa(
    config,
    controller_cls,
    env,
    n_butterflies: int = _N_BUTTERFLIES_DEFAULT,
    n_iterations: int = _N_ITERATIONS_DEFAULT,
    prior: list[float] | None = None,
) -> tuple[np.ndarray, float]:
    """Run the Butterfly Optimisation Algorithm.

    Returns (best_params, best_cost).
    """
    rng = np.random.default_rng(_SEED)
    n_params = len(BOUNDS_LOW)

    # ---- Initialisation -----------------------------------------------
    # Butterflies are stored in normalised [0,1] space to decouple step sizes
    # from the very different parameter scales (e.g. q_pos_xy:10-500 vs r_att:0.01-10).
    X_n = rng.random((n_butterflies, n_params))  # normalised positions
    if prior is not None:
        prior_arr = np.asarray(prior, dtype=float)
        X_n[0] = _norm(prior_arr)
        logger.info("Butterfly 0 seeded from prior: %s", _fmt_params(prior_arr))
    logger.info("=== BOA: evaluating %d initial butterflies ===", n_butterflies)
    F = np.zeros(n_butterflies)
    for i in range(n_butterflies):
        F[i] = evaluate(_denorm(X_n[i]), env, config, controller_cls)
        logger.info("  butterfly %d: cost=%.4f  %s", i, F[i], _fmt_params(_denorm(X_n[i])))

    best_i = int(np.argmin(F))
    g_best_n = X_n[best_i].copy()  # best position in normalised space
    g_best_fit = float(F[best_i])
    logger.info("Initial best: butterfly %d  cost=%.4f", best_i, g_best_fit)

    # ---- Main loop ----------------------------------------------------
    for it in range(n_iterations):
        logger.info("--- Iteration %d / %d ---", it + 1, n_iterations)

        # Normalise costs within this iteration so fragrance reflects *relative*
        # quality, not absolute magnitude.  This keeps fi well-scaled regardless
        # of what the raw cost values happen to be.
        f_min, f_max = F.min(), F.max()
        I = (F - f_min) / (f_max - f_min + 1e-10)  # 0=best, 1=worst

        for i in range(n_butterflies):
            fi = _C_SENSORY * max(float(I[i]), _MIN_FRAGRANCE) ** _A_POWER
            r2 = float(rng.random()) ** 2

            if rng.random() < _P_SWITCH:
                # Global search: move toward the swarm's best position
                x_new_n = X_n[i] + (r2 * g_best_n - X_n[i]) * fi
            else:
                # Local search: move relative to two random peers
                others = [j for j in range(n_butterflies) if j != i]
                j, k = rng.choice(others, size=2, replace=False)
                x_new_n = X_n[i] + (r2 * X_n[j] - X_n[k]) * fi

            x_new_n = np.clip(x_new_n, 0.0, 1.0)
            f_new = evaluate(_denorm(x_new_n), env, config, controller_cls)

            if f_new < F[i]:
                X_n[i] = x_new_n
                F[i] = f_new
                if f_new < g_best_fit:
                    g_best_n = x_new_n.copy()
                    g_best_fit = f_new
                    logger.info(
                        "  [butterfly %d] new global best: cost=%.4f  %s",
                        i, g_best_fit, _fmt_params(_denorm(g_best_n)),
                    )

        logger.info("End of iteration %d: best cost=%.4f", it + 1, g_best_fit)

    return _denorm(g_best_n), g_best_fit


def _denorm(x_n: np.ndarray) -> np.ndarray:
    """Map normalised [0,1] vector back to actual parameter bounds."""
    return BOUNDS_LOW + x_n * (BOUNDS_HIGH - BOUNDS_LOW)


def _norm(x: np.ndarray) -> np.ndarray:
    """Map actual parameter values into normalised [0,1] space."""
    return (x - BOUNDS_LOW) / (BOUNDS_HIGH - BOUNDS_LOW)


def _fmt_params(x: np.ndarray) -> str:
    pairs = [f"{n}={v:.1f}" for n, v in zip(PARAM_NAMES, x)]
    return "  ".join(pairs)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(
    config: str = "level0_attitude.toml",
    n_butterflies: int = _N_BUTTERFLIES_DEFAULT,
    n_iter: int = _N_ITERATIONS_DEFAULT,
) -> None:
    """Run BOA tuning.

    Args:
        config: Config filename inside the ``config/`` directory.
        n_butterflies: Number of butterflies in the swarm.
        n_iter: Number of BOA iterations.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    config_path = Path(__file__).parents[1] / "config" / config
    cfg = load_config(config_path)
    cfg.sim.render = False  # headless — rendering would block the loop

    control_path = Path(__file__).parents[1] / "lsy_drone_racing" / "control"
    controller_cls = load_controller(control_path / "attitude_mpc_lin_boa.py")

    env = gymnasium.make(
        cfg.env.id,
        freq=cfg.env.freq,
        sim_config=cfg.sim,
        sensor_range=cfg.env.sensor_range,
        control_mode=cfg.env.control_mode,
        track=cfg.env.track,
        disturbances=cfg.env.get("disturbances"),
        randomizations=cfg.env.get("randomizations"),
        seed=cfg.env.seed,
    )
    env = JaxToNumpy(env)

    try:
        best_params, best_cost = run_boa(cfg, controller_cls, env, n_butterflies, n_iter, prior=PRIOR)
    finally:
        env.close()

    q_diag, rw_diag, t_total_best = params_to_diags(best_params)

    print("\n" + "=" * 60)
    print("BOA Optimisation Complete")
    print("=" * 60)
    print(f"Best cost : {best_cost:.6f}  (= flight_time + {W1}*IAE + {W2}*ITAE + {W3}*U²)")
    print()
    print("Optimal parameters:")
    for name, val in zip(PARAM_NAMES, best_params):
        print(f"  {name:<12} = {val:.4f}")
    print()
    print("Paste into attitude_mpc_lin_boa.py (or attitude_mpc_lin.py):")
    q_str = [f"{v:.2f}" for v in q_diag]
    rw_str = [f"{v:.2f}" for v in rw_diag]
    print(f"  self._t_total = {t_total_best:.2f}")
    print(f"  Q  = np.diag([{', '.join(q_str)}])")
    print(f"  Rw = np.diag([{', '.join(rw_str)}])")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
