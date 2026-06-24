# The Challenge

Build an autonomous **controller** that flies a Crazyflie through a gate course. You develop in
**simulation** (unlimited), and at the end you present your solution to the **judges** with a
**live flight on the real drone**. Speed matters — but how you solve it, how well it transfers to
the real world, and how robust it is matter more.

## What you build

One Python file: a subclass of [`Controller`](lsy_drone_racing/control/controller.py) implementing

```python
def compute_control(self, obs, info) -> np.ndarray: ...
```

Start from [`control/my_controller.py`](lsy_drone_racing/control/my_controller.py). Anything goes
inside: geometric/PID/MPC control, a trained RL policy, trajectory optimization, runtime
adaptation. **One controller class per file.** The same file runs in sim and on the real drone.

### The interface
- **Observation** (`obs` dict): `pos`, `quat` (xyzw), `vel`, `ang_vel`, `target_gate` (`-1` when
  finished), `gates_pos`, `gates_quat`, `gates_visited`, `obstacles_pos`, `obstacles_visited`.
  Gate/obstacle poses are exact only within the sensor range; otherwise the nominal (config) pose
  is reported.
- **Action**: a 13-D state setpoint `[x,y,z, vx,vy,vz, ax,ay,az, yaw, r_rate,p_rate,y_rate]`
  (`control_mode="state"`, default) or a 4-D `[collective_thrust, roll, pitch, yaw]`
  (`control_mode="attitude"`).

## The two things that make this hard (and decide the winner)

1. **Generalization — the demo track is held out.** You develop on *randomized* layouts; the real
   demo track is set by the organizers and revealed only at the end, within the same bounds. A
   controller tuned to one fixed layout will fail. **Read the gates from `obs` at runtime — never
   hardcode coordinates.**
2. **Sim-to-real transfer.** The same controller must fly the *real* drone, which behaves
   differently (real dynamics, onboard Lighthouse positioning, latency). You get limited real
   flight time, so you must engineer for the gap, not brute-force it in sim.

## Develop & self-check (simulation)

- Develop and test on **`level2.toml`** — randomized gates/obstacles. This is your dev target.
- Score yourself with the official evaluation (20 runs on `level2`, reports success rate +
  average lap time):
  ```bash
  python scripts/evaluate.py --controller <team>.py
  ```
  This is your **performance signal** and evidence for the judges — it does **not** rank you.
- Difficulty ladder: `level0` (static, perfect knowledge) → `level1` (randomized inertia) →
  `level2` (randomized track, the dev target) → `level3` (random tracks, stretch).

## How you're judged

Winners are decided at the **live judged demo**, not by a leaderboard. Each team explains their
solution and flies the real drone. Judges score a rubric weighted so **real-world flight +
sim-to-real transfer dominate**:

| Criterion | Weight |
|---|---:|
| Real-world flight performance (live demo) | 40 |
| Sim-to-real transfer | 25 |
| Technical approach & sophistication | 20 |
| Creativity / beyond the base task | 10 |
| Presentation & explanation | 5 |

A sim-only or memorized-track solution caps low. There may also be a **"Best Sim Performance"**
shout-out for the fastest robust controller by the evaluation above.

## Rules

- **Read the track from `obs`/`info` at runtime — do not hardcode gate coordinates.**
- Stay within the control loop budget (default 50 Hz); don't block the loop.
- Extra pip packages are allowed but **must be declared** (`pyproject.toml` / `pixi add`) so your
  run reproduces.
- One controller file per team. Don't edit the environment or the evaluation script.

## Real-drone time & submission

- **Booked, supervised flight slots** throughout the hack let you test sim-to-real and your own
  practice layouts. A safety officer is always present; complete the one-time **safety briefing**
  before your first slot. Real flight is a scarce resource (battery-limited) — iterate in sim,
  use slots to check transfer.
- **Submission:** bring your final controller file to your demo slot; the organizers run it on the
  real drone for the judges. No upload needed.

## Credit

Hardware, simulator, and the racing course are provided by the **Learning Systems Lab (LSY), TU
München**. Please credit LSY in your presentation and materials.
