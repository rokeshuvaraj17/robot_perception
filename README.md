# robot_perception

Twin Wings simulator, Monte Carlo localization and the Shared-Sector Occlusion
Likelihood (SSOL) for the AOE 4984 project "One Occluder, Many Beams".
The Python core runs on any laptop (no GPU, no ROS needed); `ros/` runs the same
filter in ROS 2 Jazzy with Gazebo and RViz.

## ROS 2 / Gazebo demo
Inside a ROS 2 Jazzy environment with `turtlebot3_gazebo` installed:
```bash
python3 ros/make_world.py --asym 0.3          # world + robot model -> ros/generated/
ros/restart.sh likelihood:=ssol               # or likelihood:=beam_mixture, add drive:=false for teleop
```
RViz shows the particles in both wings, each wing's posterior mass, the commit
decision, and (for SSOL) the beams it attributes to the shared occluder in orange.

## Setup and run
```bash
cd ssol_sim
/opt/homebrew/bin/python3.12 -m venv .venv          # any Python >= 3.10
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests                  # 14 tests
.venv/bin/python scripts/run_demo.py --asym 0.3      # plots -> out/
```

## Layout
| File | What it does |
|---|---|
| `twinwings/world.py` | Twin Wings map (two identical 7 x 5 m wings; doorway D recess of depth `asym` only in Wing A), free-space check, aliasing, person/chair/box occluders that stand in D then walk away |
| `twinwings/geometry.py` | Vectorized ray casting (segments + circles) |
| `twinwings/sensor.py` | RPLidar A1-like scan: 60 beams, 0.15–12 m, 5.5 Hz, range noise, no-return = `nan` |
| `twinwings/likelihoods.py` | Swappable scan likelihood. `BeamMixture` = comparator 1. SSOL and comparators 2–4 go here |
| `twinwings/mcl.py` | MCL: odometry+gyro motion model, likelihood (exponent one), resampling, mode masses |
| `twinwings/episode.py` | 60 s episode: robot path, occluder, scans, filter; commit rule (`tau_c`, abstain at 60 s) |
| `scripts/run_demo.py` | One episode per occluder, prints outcomes, saves figures |

To add a comparator, write a class with `name` and
`log_likelihood(z, r_exp) -> (N,)` and pass it to `run_episode`. Nothing else changes.

## Design choices to confirm with the team
1. **Asymmetry = depth of doorway D's recess** (0 / 0.15 / 0.30 m).
2. **Mixture particle filter with paired particles.** Each wing has a fixed block of
   N/K particles; blocks start as aliased copies, share motion-noise draws, and are
   resampled within the block. Without this, particle-sampling luck (~100 nats)
   outweighs the occluder effect (~20 nats/scan) and the zero-asymmetry control
   commits at random. Side effect: a wing's particles never die, so the filter
   can recover after a wrong commit (the commit itself is still recorded).
3. **Commit rule applied after the run** (first time a wing's mass reaches `tau_c`),
   so one run serves every threshold of a risk–coverage sweep.
4. The two wings are not connected by a corridor (pure relocalization within one wing).

## First results (baseline only, asym 0.30 m, seed 1)
| Occluder | Outcome | Beams hidden while standing |
|---|---|---|
| none | correct | 0 |
| box | **wrong** (commits to Wing B at t = 0; recovers at ~36 s) | ~3 |
| person | correct | ~0.5 |
| chair | correct | ~0.4 |

At asym 0 every case abstains. p95 update time is about 7 ms (limit 100 ms).

## SSOL (B5) first result: same outcomes as the baseline
`SSOL` in `twinwings/likelihoods.py` implements Eqs. (3)-(6): range-bin alphabet + no-return,
normalized f_b / g_b, 1-degree sector grid (<= 30 deg) pushed forward to 301 masks at 60 beams,
log-sum-exp with prefix sums. Tests confirm it sums to exactly 1 over all scans. p95 update ~10 ms.

Over seeds 0-4, asym 0.15/0.30, 60 and 120 beams, SSOL and the beam mixture give identical
commit outcomes (box: wrong in 19/20 runs for both). Per-scan gap with the box, 60 beams,
asym 0.30: beam mixture -19.05 nats, SSOL -19.12 (negative = favours the wrong wing).

Why, for one hidden beam (z = 3.24 m; true wing expects 3.50, wrong wing expects 3.20):
| term | log value |
|---|---|
| true wing, beam scored clean (f_A) | -8.47 |
| true wing, beam scored occluded (g_A) | -6.00 |
| wrong wing, wall hit (f_B) | -1.14 |
| prior: empty mask / one 3-beam mask | -0.61 / -6.40 |

Sharing the occluder saves ~2.5 nats per hidden beam but costs ~5.8 nats of prior once, while
the wrong wing explains the same reading as a sharp wall hit, ~4.9 nats per beam better than
the best occluded explanation. The dominant cost is emission sharpness, not the number of
per-beam occlusion events. Discuss with the team: H1 as stated is unlikely to hold when the
occluder stands on the wrong wing's wall line.
