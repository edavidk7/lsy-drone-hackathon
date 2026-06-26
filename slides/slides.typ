#import "@preview/touying:0.7.3": *
#import themes.metropolis: *
#import "@preview/cetz:0.5.2": canvas, draw
#import "@preview/algo:0.3.4": algo, i, d, comment
#import "@preview/cetz-plot:0.1.4": plot


#let CTU_color = rgb("#0065BD")

#let title = [LSY Drone Racing: geometric planning + RL vs. end-to-end RL]
#let author = "David Korčák"

#show link: underline


#set document(
    title: title,
    author: author,
  )

#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  align: horizon,
  //config-common(handout: true),
  //config-common(show-notes-on-second-screen: right),
  config-info(
    title: title,
    author: author,
    date: datetime(year: 2026, month: 6, day: 26),
    institution: [TUM AI Hackathon · LSY Drone Racing],
    logo: none
  ),
  // rgb("0065BD"),
  config-colors(
    primary: CTU_color.darken(0%),
    secondary: CTU_color.lighten(20%),
    primary-light: CTU_color.lighten(60%),
    // primary-light: rgb("#ff0000")
    // tertiary: CTU_color.lighten(35%),
    neutral-lightest: rgb("#ffffff"),
    neutral-dark: CTU_color.darken(40%),
    // neutral-darkest: rgb("#000000"),
  ),
  header-right: none,
)


#show heading.where(level: 1): it => text(weight: "light", tracking: 2pt, smallcaps[#it])
#set text(size: 15pt)
#set par(leading: 0.55em)
#show heading.where(level: 3): set block(above: 0.7em, below: 0.5em)

// Horizontal obs -> ... -> ctrl pipeline of rounded boxes joined by arrows.
// steps: array of (title, sub, fill).
#let pipeline(steps, bw: 3.3, bh: 2.5, gap: 0.62) = align(center, canvas(length: 1cm, {
  import draw: *
  let arrow_color = CTU_color.darken(15%)
  for (i, s) in steps.enumerate() {
    let x = i * (bw + gap)
    rect((x, 0), (x + bw, bh), radius: 0.13, fill: s.fill, stroke: 1pt + CTU_color.darken(28%))
    content(
      (x + bw / 2, bh / 2),
      box(width: (bw - 0.3) * 1cm, align(center,
        text(fill: white, weight: "bold", size: 13pt, s.title)
          + linebreak()
          + text(fill: white.transparentize(12%), size: 10pt, s.sub),
      )),
    )
    if i < steps.len() - 1 {
      line(
        (x + bw, bh / 2), (x + bw + gap, bh / 2),
        stroke: 1.4pt + arrow_color, mark: (end: ">", fill: arrow_color),
      )
    }
  }
}))

#let c_obs = CTU_color.lighten(12%)
#let c_proc = CTU_color
#let c_pol = CTU_color.darken(16%)
#let c_ctrl = CTU_color.darken(30%)


== #smallcaps[Two paths to a superhuman lap time]

*Two complementary controllers* that share one learned dynamics model but split the
navigation question differently.

#grid(
  columns: (1fr, 1fr),
  gutter: 18pt,
  [
    === ① Min-Snap + RL (plan $arrow$ track)
    - *Collision-aware min-snap planner* lays a feasible path through the gates
    - A PPO policy *tracks* it as look-ahead points
    - Geometry = _where_ to fly, RL = _how_; re-plans in \~1 ms as gates appear
  ],
  [
    === ② End-to-End Nav RL (sense $arrow$ act)
    - One PPO policy maps *raw race state* straight to *commands*
    - *No explicit trajectory*, nothing to re-plan
    - Fully reactive; removes the planner as a failure point
  ],
)

*Why both?* Hybrid = robust + interpretable (a visible reference path); end-to-end = maximally
reactive, theoretically highest-performance, but more brittle and sensitive. A shared *debug UI* inspects, pauses, and compares either policy live for easier deployment.


== #smallcaps[Min-Snap + RL]

#text(size: 12pt, fill: gray)[Collision-aware geometric planner + learned path-tracker — `train_min_snap_rl.py` / `min_snap_rl_controller.py`]
#grid(
  columns: (1fr, 1fr),
  gutter: 18pt,
  [
    === Planning (classical, snap-optimal)
    - *Min-snap* polynomial spline through gate centers, waypoints forced *perpendicular* to each gate plane
    - Pole + gate-frame *keep-out corridors*; collision-aware seeding + clearance repair
    - *Acceleration-limited* time scaling (`A_LIMIT=6.0`, cruise `SPEED=0.6`) $arrow$ dynamically feasible
    - 1#super[st] plan: full L-BFGS snap minimization; *re-plan* on sensed gate motion (`>4 cm`): fast seed + repair, \~1 ms in-loop
  ],
  [
    === Tracking policy (RL)
    - *Obs (73-D):* drone state (13) + 10 look-ahead path deltas @ `0.1 s` (30) + 2-step history (26) + last action (4)
    - *Action:* 4-D `[r, p, y, thrust]`, yaw zeroed, scaled to $plus.minus pi/2$ and the drone's thrust bounds
    - Learned dynamics $arrow$ tracks tight turns far better than the onboard state controller
    - *PPO:* `2048` envs, `5.5 M` steps, $gamma=0.94$, 2$times$64 tanh MLP
  ],
)
#v(0.15em)
#align(center, text(size: 13pt, $
  r_t = exp(-2 d_t) - 0.06 norm("rpy") - 0.02 a_"thr"^2 - 0.4 (Delta a_"thr")^2 - 1.0 norm(Delta a_"xy")^2
  quad ("on crash:" r_t = -1)
$))
#v(0.1em)
#align(center, text(size: 10.5pt, fill: gray)[
  $d_t$ distance to the current look-ahead point $arrow$ $exp(dot)$ rewards *hugging the path* (=1 on it, decays away) ·
  out-of-bounds *crash* replaces it with $-1$ · $norm("rpy")$ keeps the drone *level* ·
  last three penalize *thrust energy* + *thrust/xy jerk* for smooth flight
])
#v(0.3em)
#pipeline((
  (title: "Obs", sub: "gates · obstacles · pose", fill: c_obs),
  (title: "Min-Snap Planner", sub: "collision-aware · replan ~1 ms", fill: c_proc),
  (title: "Look-ahead path", sub: "10 samples ahead", fill: c_pol),
  (title: "PPO Policy", sub: "tracks the path", fill: c_pol.darken(8%)),
  (title: "Ctrl", sub: "thrust + r / p / y", fill: c_ctrl),
), bw: 3.0)


== #smallcaps[End-to-End Navigation RL]

#text(size: 12pt, fill: gray)[One policy, raw geometry $arrow$ command — `train_nav_rl.py` / `nav_rl_common.py`]
#grid(
  columns: (1fr, 1fr),
  gutter: 18pt,
  [
    === Observation (egocentric)
    - Target-gate $Delta$pos (world *and* body frame)
    - Gate orientation: 9-D rotation matrix
    - Linear + angular velocity
    - 2 nearest obstacles, sensed-gate ratio
    - Progress `target/N`, prev action `a_(t-1)`
  ],
  [
    === Action & Reward
    - Two control modes:
      - `attitude` — 4-D thrust + r/p/y
      - `state` — 13-D full setpoint
    - *Shaped reward:* gate progress, gate-pass / success bonuses, crash + energy/jerk penalties:
  ],
)
#v(0.2em)
#align(center, text(size: 13pt, $
  r_t = r_"base" + 5(d_(t-1) - d_t) + 10 dot bb(1)_"pass" + 15 dot bb(1)_"success" - 5 dot bb(1)_"crash"
  - 0.01 a_"thr"^2 - 0.25 norm(Delta a_"rpy")^2 - 0.1 (Delta a_"thr")^2
$))
#v(0.15em)
#align(center, text(size: 10.5pt, fill: gray)[
  $r_"base"$ sparse env reward · $d$ distance to target gate $arrow$ reward *closing in* on it ·
  $bb(1)_"pass"$ bonus for crossing a gate · $bb(1)_"success"$ bonus for finishing the track ·
  $bb(1)_"crash"$ penalty for crashing · last three terms penalize *thrust energy* and *jerk* (action change) for smooth flight
])
#v(0.4em)
#pipeline((
  (title: "Obs", sub: "raw race state", fill: c_obs),
  (title: "Egocentric features", sub: "+ prev action", fill: c_proc),
  (title: "PPO Policy", sub: "actor MLP", fill: c_pol),
  (title: "Ctrl", sub: "attitude 4-D / state 13-D", fill: c_ctrl),
), bw: 3.0)
#v(0.25em)
#align(center, text(size: 11pt, fill: gray)[Standalone PPO · GAE · `2048` JAX envs · checkpoints store args + arch to rebuild the matching net])


== #smallcaps[Live Debug UI]

#text(size: 12pt, fill: gray)[Stop, resume, and inspect the policy mid-flight — `lsy_drone_racing/debug_ui/`]
#grid(
  columns: (1fr, 1fr),
  gutter: 2pt,
  [
    === What it shows
    - *Live 3D trajectory:* history + gates/obstacles
    - Position & action time-series updated in real-time
    - Sidebar: target gate, speed, action, rate of last step
  ],
  [
    === Stop / start the policy
    - Dashboard sends *stop / resume* over TCP to the `Controller`
    - *Stop* $arrow$ controller latches a *PID hover hold* at the current pose, safe to inspect
    - *Resume* $arrow$ control handed straight back to the policy
  ],
)

#figure(
  image("ui.png", width:70%),
     caption: [UI for debugging and human supervision of the drone policy on deployment]
   )

