# 🚁 LSY Drone Racing — Hackathon Edition

Build an autonomous controller that flies a **Crazyflie** through a gate course. You develop in
**simulation** (unlimited), then present your solution live to the judges flying the **real drone**
— on a track layout you don't see during development. Built on TUM LSY's
[`lsy_drone_racing`](https://github.com/learnsyslab/lsy_drone_racing) and
[`crazyflow`](https://github.com/learnsyslab/crazyflow).

👉 **Read [CHALLENGE.md](CHALLENGE.md) for the full challenge, rules, and how you're judged.**

---

## You write one file

A single controller — a subclass of
[`Controller`](lsy_drone_racing/control/controller.py) that maps the current observation to the
next command. Start from [`control/my_controller.py`](lsy_drone_racing/control/my_controller.py).
That's it: the same file runs in sim and on the real drone.

## Quickstart (≈ flying in sim in 15 min)

```bash
# 1. Fork this repo, then clone YOUR fork
git clone https://github.com/<you>/lsy-drone-hackathon.git
cd lsy-drone-hackathon

# 2a. Linux (recommended): pixi installs everything
pixi shell

# 2b. macOS / Windows (simulation only): conda + pip
conda create -n drones python=3.11 -y && conda activate drones
pip install -e ".[sim]"

# 3. Fly the example in simulation
python scripts/sim.py --config level0.toml

# 4. Make it yours
cp lsy_drone_racing/control/my_controller.py lsy_drone_racing/control/<team>.py
python scripts/sim.py      --config level2.toml --controller <team>.py   # randomized: the dev target
python scripts/evaluate.py --controller <team>.py                        # your performance signal
```

## Develop against the randomized track

- Develop and self-check on **`level2.toml`** (randomized gates/obstacles) — this is what
  `evaluate.py` scores (20 runs, success rate + average lap time). The shipped example controllers
  complete the *fixed* `level0` but **not** randomized `level2` — making a controller that
  generalizes is the actual challenge.
- The real demo track is **held out**: read gate poses from `obs` at runtime, never hardcode them.

## Real flight & submit

You deploy from **one central machine** in a booked window (you never connect to the drone
directly), and **two drones are shared** across teams. You can also put up gates to test your own
layouts. See **[REAL_FLIGHT.md](REAL_FLIGHT.md)** for how deployment works, how to place gates, and
how to put their exact coordinates into a config. Bring your final controller file to your demo
slot — the organizers run it for the judges. Judging + rules are in [CHALLENGE.md](CHALLENGE.md).

## Learn the interface

- The base class and every hook: [`control/controller.py`](lsy_drone_racing/control/controller.py)
- Example controllers: `state_controller.py` (trajectory), `attitude_mpc.py` (MPC),
  `attitude_rl.py` + `train_rl.py` (RL).
- Course docs: <https://lsy-drone-racing.readthedocs.io>

## Credit

Hardware, simulator, and the racing course are provided by the **Learning Systems Lab (LSY), TU
München**. Please credit LSY in your presentation.
