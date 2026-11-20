"""Simulated low-cost 2D LiDAR (RPLidar A1-like), subsampled to |B| beams."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import raycast


@dataclass
class LidarConfig:
    n_beams: int = 60          # |B|, evenly spaced over 360 deg (Nav2 default max_beams)
    min_range: float = 0.15
    max_range: float = 12.0
    rate_hz: float = 5.5
    offset: float = 0.0        # angular offset of beam 0 (robot frame), fixed on dev data
    sigma0: float = 0.01       # range noise: sigma = sigma0 + sigma_rel * r
    sigma_rel: float = 0.01
    dropout: float = 0.005     # probability a valid return is lost (-> no-return)

    def bearings(self) -> np.ndarray:
        return self.offset + 2 * np.pi * np.arange(self.n_beams) / self.n_beams


def expected_ranges(world, poses, cfg: LidarConfig) -> np.ndarray:
    """Map-only ray cast for each pose: (N, 3) -> (N, B). np.inf = nothing in map."""
    poses = np.asarray(poses, dtype=float).reshape(-1, 3)
    b = cfg.bearings()
    n, nb = len(poses), len(b)
    origins = np.repeat(poses[:, :2], nb, axis=0)
    angles = (poses[:, 2:3] + b[None, :]).reshape(-1)
    return raycast(origins, angles, world.segments).reshape(n, nb)


def true_ranges(world, pose, occluder_shapes, cfg: LidarConfig):
    """Noise-free ranges with occluders, plus a mask of beams the occluder hit."""
    segs, circles = occluder_shapes if occluder_shapes is not None else (None, None)
    b = cfg.bearings()
    origins = np.repeat(np.asarray(pose[:2], float)[None], len(b), axis=0)
    angles = pose[2] + b
    r_map = raycast(origins, angles, world.segments)
    all_segs = world.segments if segs is None or not len(segs) else np.vstack([world.segments, segs])
    r = raycast(origins, angles, all_segs, circles)
    occluded = r < r_map - 1e-6
    return r, occluded


def simulate_scan(world, pose, occluder_shapes, cfg: LidarConfig, rng):
    """One noisy scan. Returns (z, occluded_mask); z uses np.nan for no-return."""
    r, occluded = true_ranges(world, pose, occluder_shapes, cfg)
    sigma = cfg.sigma0 + cfg.sigma_rel * np.minimum(r, cfg.max_range)
    z = r + rng.normal(0.0, 1.0, size=r.shape) * sigma
    lost = (r > cfg.max_range) | (z < cfg.min_range) | (z > cfg.max_range)
    lost |= rng.random(r.shape) < cfg.dropout
    z = np.where(lost, np.nan, z)
    return z, occluded
