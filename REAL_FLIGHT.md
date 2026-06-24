# Testing on the Real Drone

Everything about getting your controller onto a real Crazyflie: how deployment works, how to set
up and test **your own** practice track, and the safety basics. (Full safety briefing happens at
the workshop — attendance is required before you fly.)

> **Develop in simulation first.** Real flight time is scarce (two shared drones, battery-limited).
> Only bring a controller that reliably completes the track in sim.

## How deployment works

- **You never connect to the drone directly.** There is **one central deployment computer** (Linux)
  wired to the drone via the Crazyradio. In your **booked window**, you run *your* controller on
  that machine; it talks to the drone for you.
- **Two drones are shared by all teams.** You rotate on deployment windows; while one flies, the
  other charges. Respect the schedule and battery rotation.
- A **safety officer is always present** and everyone keeps a **hand on the emergency stop**
  (`Ctrl-C` stops the drone). See [Safety](#safety-summary).

## Positioning: Lighthouse (no motion capture)

The drone localizes **itself** from the Lighthouse base stations. The world origin is a **marked
point + x-axis on the floor** (the "track origin"). **All coordinates in your config are in this
frame.**

There is **no tracking of the gates or obstacles** — the drone trusts that they are exactly where
your config says. So **the physical placement must match your config coordinates**, measured from
the origin. Any mismatch is error the drone cannot see.

## Test your own track (the demo track is secret)

You can put gates up and test your controller on a layout you design. Four steps:

### 1. Choose a layout (inside the flight volume)

- **Safety box** (origin at the marked floor point): `x ∈ [-2.5, 2.5]`, `y ∈ [-1.5, 1.5]`,
  `z ∈ [0, 2.0]` metres. Keep gates well inside.
- **Gates:** opening is **0.4 m**; the config `z` is the gate **center** height — tall gate center
  **1.195 m**, short **0.695 m**. The gate is crossed along its **forward normal** (its local
  x-axis), set by the `yaw`.
- **Obstacles ("posts"):** thin cylinders ~**1.52 m** tall (use config `z ≈ 1.55`).

### 2. Place the hardware

- For each **gate**: measure `x, y` from the origin along the marked axes, position the gate so its
  **center** is at `(x, y)` and at height `z`, and rotate it so the opening faces your chosen
  `yaw`. **Tape the floor** so a bumped gate can be put back exactly.
- Place each **obstacle** at its `(x, y)`. Put the drone on its **start marker**.
- Measure to a few cm — see [Accuracy](#5-accuracy-matters).

### 3. Put the locations into a config

Copy a config and edit the track to match what you placed (keep the `[env.randomizations]` and
`[env.track.safety_limits]` blocks from the file you copied):

```toml
# my_track.toml  (copied from config/level0.toml, then edited)
[controller]
file = "my_controller.py"

[deploy]
real_track_objects = false   # nothing measures the gates -> the config IS ground truth
lighthouse = true            # use the drone's onboard Lighthouse estimate (no mocap)
check_race_track = true
check_drone_start_pos = true
[[deploy.drones]]
id = 10
channel = 100
drone_model = "cf21B_500"

[env.track]
randomize = false
[[env.track.gates]]
pos = [0.5, 0.25, 0.7]      # gate CENTER (x, y, z) in metres, track frame
rpy = [0.0, 0.0, -0.78]     # yaw in radians; opening faces the gate's x-axis
# ... one [[env.track.gates]] block per gate ...

[[env.track.obstacles]]
pos = [0.0, 0.75, 1.55]
# ... one [[env.track.obstacles]] block per obstacle ...

[[env.track.drones]]
pos = [-1.5, 0.75, 0.05]    # the start marker
rpy = [0.0, 0.0, 0.0]
vel = [0, 0, 0]
ang_vel = [0, 0, 0]
```

You can also iterate on this layout **in simulation first** (`python scripts/sim.py --config
my_track.toml --controller <team>.py`) before using a real window.

### 4. Run it (in your window, on the central machine)

```bash
python scripts/deploy.py --config my_track.toml --controller <team>.py
```

It checks the start pose and track bounds, takes off, flies, prints the lap time, and lands.

### 5. Accuracy matters

Because nothing tracks the gates, the gap between where you **placed** a gate and the coordinates
in your **config** is error the drone can't perceive. Measure carefully (a few cm), and
**re-measure after any collision** that moves a gate or bumps a base station.

## Safety summary

The full briefing is at the workshop; the essentials:

- **Sim first** — only deploy a controller that completes in simulation.
- **Hand on the emergency stop at all times.** `Ctrl-C` stops the drone; use the hardware kill for
  emergencies. **Abort immediately** if the drone is unstable or heads for the net or people.
- **Stay behind the net.** Nothing in the net while the drone is armed. One safety officer per
  flight.
- **Don't bump the base stations** — it invalidates the positioning. If the estimate looks wrong,
  stop and tell an organizer.
- Handle batteries/LiPos per the organizers' instructions; return them to charge after your window.
