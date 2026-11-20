"""One aliased-recovery episode: true motion, occluder, scans, MCL, commit rule.

The filter never sees ground truth: it gets noisy odometry/gyro deltas and
scans only. Ground truth is kept for labels (true wing) and diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import wrap_angle
from .mcl import MCL, MotionNoise
from .sensor import LidarConfig, simulate_scan
from .world import TwinWings, doorway_occluder

# Default trajectory: (v [m/s], omega [rad/s], duration [s]) in the local wing
# frame, starting at DEFAULT_START facing doorway D. Robot then idles.
DEFAULT_START = np.array([3.6, 1.8, np.pi / 2])
DEFAULT_COMMANDS = [
    (0.0, 0.0, 4.0),     # look at the (occluded) doorway
    (0.0, 0.4, 4.0),     # turn left
    (0.15, 0.0, 5.0),    # drive 0.75 m towards -x
    (0.0, -0.4, 4.0),    # face the doorway again
    (0.0, -0.4, 4.0),    # face +x
    (0.15, 0.0, 8.0),    # drive 1.2 m towards +x
    (0.0, 0.4, 4.0),     # face the doorway
]


@dataclass
class EpisodeSpec:
    asym: float = 0.30
    occluder: str | None = "box"     # "person" | "chair" | "box" | None
    true_wing: int = 0
    dwell: float = 20.0              # seconds the occluder stands in the doorway
    duration: float = 60.0           # T_max
    seed: int = 0
    start_local: np.ndarray = field(default_factory=lambda: DEFAULT_START.copy())
    commands: list = field(default_factory=lambda: list(DEFAULT_COMMANDS))
    # true wheel/gyro noise: trans proportional, trans floor [m], rot proportional, rot floor [rad]
    odom_noise: tuple = (0.03, 0.002, 0.02, 0.004)


@dataclass
class EpisodeResult:
    spec: EpisodeSpec
    likelihood_name: str
    times: np.ndarray            # (T,)
    masses: np.ndarray           # (T, K) posterior mass per wing after each update
    true_poses: np.ndarray       # (T, 3)
    est_poses: np.ndarray        # (T, 3)
    n_occluded: np.ndarray       # (T,) beams hit by the occluder
    update_ms: np.ndarray        # (T,) wall time of each complete update
    occluder_xy: np.ndarray      # (T, 2) occluder centre (nan if none)
    snapshots: dict              # time -> (particles, weights)

    def commit(self, tau_c: float = 0.95):
        return commit_decision(self.masses, self.times, tau_c, self.spec.duration)

    def outcome(self, tau_c: float = 0.95) -> str:
        k, _ = self.commit(tau_c)
        if k is None:
            return "abstain"
        return "correct" if k == self.spec.true_wing else "wrong"


def commit_decision(masses, times, tau_c: float, t_max: float = 60.0):
    """Commit to mode k at the first time its mass reaches tau_c.

    Returns (k, t_commit), or (None, t_max) to abstain. The filter does not
    depend on tau_c, so any threshold can be applied after the run (this is
    how risk-coverage curves are swept)."""
    hit = np.flatnonzero(masses.max(axis=1) >= tau_c)
    if len(hit) == 0:
        return None, t_max
    i = hit[0]
    return int(np.argmax(masses[i])), float(min(times[i], t_max))


def _integrate(pose, v, w, dt):
    x, y, th = pose
    if abs(w) < 1e-9:
        x += v * dt * np.cos(th)
        y += v * dt * np.sin(th)
    else:
        x += v / w * (np.sin(th + w * dt) - np.sin(th))
        y += -v / w * (np.cos(th + w * dt) - np.cos(th))
    return np.array([x, y, float(wrap_angle(th + w * dt))])


def _command_at(commands, t):
    for v, w, d in commands:
        if t < d:
            return v, w
        t -= d
    return 0.0, 0.0


def run_episode(spec: EpisodeSpec, likelihood, n_particles: int = 500,
                lidar: LidarConfig | None = None, motion: MotionNoise | None = None,
                snapshot_times=(0.0, 5.0, 15.0, 30.0, 60.0)) -> EpisodeResult:
    lidar = lidar or LidarConfig()
    world = TwinWings(asym=spec.asym)
    rng_world = np.random.default_rng(spec.seed)            # sensor + odometry noise
    rng_filter = np.random.default_rng(spec.seed + 10_000)  # particle noise
    occ = doorway_occluder(world, spec.occluder, spec.true_wing, dwell=spec.dwell) \
        if spec.occluder else None

    dt = 1.0 / lidar.rate_hz
    n_steps = int(np.floor(spec.duration / dt)) + 1
    pose = world.alias(spec.start_local, spec.true_wing)

    mcl = MCL(world, likelihood, lidar, n_particles, motion, rng=rng_filter)
    mcl.init_modes(pose)

    a_t, a_t0, a_r, a_r0 = spec.odom_noise
    K = world.K
    times = np.zeros(n_steps)
    masses = np.zeros((n_steps, K))
    true_poses = np.zeros((n_steps, 3))
    est_poses = np.zeros((n_steps, 3))
    n_occ = np.zeros(n_steps, dtype=int)
    upd_ms = np.zeros(n_steps)
    occ_xy = np.full((n_steps, 2), np.nan)
    snaps, pending = {}, sorted(snapshot_times)

    for i in range(n_steps):
        t = i * dt
        if i > 0:
            v, w = _command_at(spec.commands, t - dt)
            new_pose = _integrate(pose, v, w, dt)
            # true motion in the robot frame, corrupted -> what wheels/gyro report
            c, s = np.cos(pose[2]), np.sin(pose[2])
            ddx, ddy = new_pose[0] - pose[0], new_pose[1] - pose[1]
            dx, dy = c * ddx + s * ddy, -s * ddx + c * ddy
            dth = float(wrap_angle(new_pose[2] - pose[2]))
            st = a_t * np.hypot(dx, dy) + a_t0
            sr = a_r * abs(dth) + a_r0
            odom = (dx + rng_world.normal(0, st), dy + rng_world.normal(0, st),
                    dth + rng_world.normal(0, sr))
            pose = new_pose
            mcl.predict(odom)

        shapes = occ.shapes(t) if occ else None
        z, occluded = simulate_scan(world, pose, shapes, lidar, rng_world)
        mcl.update(z)

        times[i] = t
        masses[i] = mcl.mode_masses()
        true_poses[i] = pose
        est_poses[i] = mcl.estimate()
        n_occ[i] = int(occluded.sum())
        upd_ms[i] = mcl.last_update_ms
        if occ:
            occ_xy[i] = occ.position(t)
        while pending and t >= pending[0] - 1e-9:
            snaps[pending.pop(0)] = (mcl.particles.copy(), np.exp(mcl.logw))

    for ts in pending:  # snapshots requested at/after the last step
        snaps[ts] = (mcl.particles.copy(), np.exp(mcl.logw))

    return EpisodeResult(spec, getattr(likelihood, "name", type(likelihood).__name__),
                         times, masses, true_poses, est_poses, n_occ, upd_ms, occ_xy, snaps)
