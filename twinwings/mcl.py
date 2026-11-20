"""Monte Carlo localization with a swappable scan likelihood.

One update (Fig. 3 of the draft):
  noisy odometry + gyro -> propagate N particles -> same |B| beams
  -> scan likelihood (exponent one) -> normalize -> resample if ESS is low
  -> mode masses (commit test is applied afterwards, see episode.py)

Design choice: mixture particle filter with paired particles.
  The N particles are split into K fixed blocks, one per aliased mode (wing).
  Block k starts as the exact aliased copy of block 0, every block receives
  the same motion-noise draws, and resampling is done inside each block with
  a shared random offset. A mode's mass therefore moves only when the scan
  evidence differs between the wings (the doorway cue or an occluder), not
  because one wing happened to draw a luckier particle. Without this, a
  60-beam likelihood is so peaked that particle-sampling luck (~100 nats)
  swamps the occluder effect (~20 nats per scan) and the zero-asymmetry
  control commits at random instead of abstaining.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .geometry import wrap_angle
from .sensor import LidarConfig, expected_ranges

# Log-likelihood added for particles inside walls or obstacles.
_OCCUPIED_PENALTY = -1e3


@dataclass
class MotionNoise:
    """Per-step 1-sigma noise injected into each particle's motion."""
    a_trans: float = 0.08    # proportional to distance travelled
    a_trans0: float = 0.004  # m, floor
    a_rot: float = 0.05      # proportional to rotation
    a_rot0: float = 0.008    # rad, floor (gyro-aided heading)


class MCL:
    def __init__(self, world, likelihood, lidar: LidarConfig | None = None,
                 n_particles: int = 500, motion: MotionNoise | None = None,
                 resample_ess: float = 0.5, rng=None):
        self.world = world
        self.likelihood = likelihood
        self.lidar = lidar or LidarConfig()
        self.K = world.K
        if n_particles % self.K:
            raise ValueError(f"n_particles ({n_particles}) must be divisible by K ({self.K})")
        self.n = n_particles
        self.m = n_particles // self.K          # particles per mode block
        self.mode_of = np.repeat(np.arange(self.K), self.m)
        self.motion = motion or MotionNoise()
        self.resample_ess = resample_ess
        self.rng = rng if rng is not None else np.random.default_rng()
        self.particles = np.zeros((self.n, 3))
        self.logw = np.full(self.n, -np.log(self.n))
        self.last_update_ms = 0.0
        self.n_resamples = 0

    def _shared(self, draw):
        """Draw noise once for one block and repeat it for every mode block."""
        return np.tile(draw, self.K) if draw.ndim == 1 else np.tile(draw, (self.K, 1))

    # ------------------------------------------------------------ belief init
    def init_modes(self, pose, sigma_xy: float = 0.3, sigma_th: float = 0.2):
        """Multi-modal start: block k is a cluster around the aliased copy of
        `pose` in wing k; all blocks share the same offsets."""
        off_xy = self._shared(self.rng.normal(0, sigma_xy, (self.m, 2)))
        off_th = self._shared(self.rng.normal(0, sigma_th, self.m))
        centres = np.vstack([self.world.alias(pose, k) for k in range(self.K)])
        c = centres[self.mode_of]
        self.particles = np.column_stack([c[:, :2] + off_xy, wrap_angle(c[:, 2] + off_th)])
        self.logw = np.full(self.n, -np.log(self.n))

    # -------------------------------------------------------------- predict
    def predict(self, odom):
        """odom = (dx, dy, dtheta) measured in the robot frame since last scan."""
        dx, dy, dth = odom
        mo = self.motion
        trans = np.hypot(dx, dy)
        s_t = mo.a_trans * trans + mo.a_trans0
        s_r = mo.a_rot * abs(dth) + mo.a_rot0
        ndx = dx + self._shared(self.rng.normal(0, s_t, self.m))
        ndy = dy + self._shared(self.rng.normal(0, s_t, self.m))
        ndth = dth + self._shared(self.rng.normal(0, s_r, self.m))
        th = self.particles[:, 2]
        c, s = np.cos(th), np.sin(th)
        self.particles[:, 0] += c * ndx - s * ndy
        self.particles[:, 1] += s * ndx + c * ndy
        self.particles[:, 2] = wrap_angle(th + ndth)

    # --------------------------------------------------------------- update
    def update(self, z):
        t0 = time.perf_counter()
        r_exp = expected_ranges(self.world, self.particles, self.lidar)
        ll = self.likelihood.log_likelihood(z, r_exp)
        ll = np.where(self.world.is_free(self.particles[:, :2]), ll, ll + _OCCUPIED_PENALTY)
        logw = self.logw + ll
        logw -= _logsumexp(logw)
        self.logw = logw
        if self.min_block_ess() < self.resample_ess * self.m:
            self._resample()
        self.last_update_ms = (time.perf_counter() - t0) * 1e3

    def _block_logw(self):
        """(K, m) within-block normalized log weights and (K,) log mode masses."""
        lw = self.logw.reshape(self.K, self.m)
        log_mass = np.array([_logsumexp(row) for row in lw])
        return lw - log_mass[:, None], log_mass

    def min_block_ess(self) -> float:
        lw, _ = self._block_logw()
        return float(np.min(1.0 / np.sum(np.exp(2 * lw), axis=1)))

    def _resample(self):
        """Systematic resampling inside each mode block, shared random offset.
        Mode masses are preserved; only the particles within a mode change."""
        lw, log_mass = self._block_logw()
        u = (self.rng.random() + np.arange(self.m)) / self.m
        new = np.empty_like(self.particles)
        for k in range(self.K):
            cum = np.cumsum(np.exp(lw[k]))
            cum[-1] = 1.0
            idx = np.minimum(np.searchsorted(cum, u), self.m - 1)
            new[k * self.m:(k + 1) * self.m] = self.particles[k * self.m + idx]
        self.particles = new
        self.logw = np.repeat(log_mass - np.log(self.m), self.m)
        self.n_resamples += 1

    # ---------------------------------------------------------------- output
    def mode_masses(self) -> np.ndarray:
        return np.exp(self._block_logw()[1])

    def estimate(self) -> np.ndarray:
        """Weighted mean pose of the heaviest mode."""
        k = int(np.argmax(self.mode_masses()))
        sel = self.mode_of == k
        w = np.exp(self.logw[sel] - _logsumexp(self.logw[sel]))
        p = self.particles[sel]
        th = np.arctan2((w * np.sin(p[:, 2])).sum(), (w * np.cos(p[:, 2])).sum())
        return np.array([(w * p[:, 0]).sum(), (w * p[:, 1]).sum(), th])


def _logsumexp(a):
    m = np.max(a)
    return m + np.log(np.sum(np.exp(a - m)))
