"""Vectorized 2D ray casting against line segments and circles."""
from __future__ import annotations

import numpy as np

_EPS = 1e-9


def raycast(origins, angles, segments=None, circles=None):
    """Return the distance from each ray origin to its first hit.

    origins:  (R, 2) ray start points
    angles:   (R,)   ray directions in radians (world frame)
    segments: (S, 4) rows of x0, y0, x1, y1
    circles:  (C, 3) rows of cx, cy, radius
    Returns (R,) ranges; np.inf where the ray hits nothing.
    """
    origins = np.asarray(origins, dtype=float).reshape(-1, 2)
    angles = np.asarray(angles, dtype=float).reshape(-1)
    d = np.stack([np.cos(angles), np.sin(angles)], axis=-1)  # (R, 2)
    best = np.full(len(angles), np.inf)

    if segments is not None and len(segments):
        seg = np.asarray(segments, dtype=float)
        a = seg[:, :2]
        e = seg[:, 2:] - a  # (S, 2)
        ap0 = a[None, :, 0] - origins[:, None, 0]  # (R, S)
        ap1 = a[None, :, 1] - origins[:, None, 1]
        denom = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (ap0 * e[None, :, 1] - ap1 * e[None, :, 0]) / denom
            u = (ap0 * d[:, None, 1] - ap1 * d[:, None, 0]) / denom
        ok = (np.abs(denom) > 1e-12) & (t > _EPS) & (u >= 0.0) & (u <= 1.0)
        best = np.minimum(best, np.where(ok, t, np.inf).min(axis=1))

    if circles is not None and len(circles):
        cir = np.asarray(circles, dtype=float)
        oc0 = origins[:, None, 0] - cir[None, :, 0]  # (R, C)
        oc1 = origins[:, None, 1] - cir[None, :, 1]
        b = oc0 * d[:, None, 0] + oc1 * d[:, None, 1]
        c = oc0**2 + oc1**2 - cir[None, :, 2] ** 2
        disc = b**2 - c
        sq = np.sqrt(np.maximum(disc, 0.0))
        t1 = -b - sq
        t2 = -b + sq
        t = np.where(t1 > _EPS, t1, t2)  # t2 when the origin is inside the circle
        ok = (disc >= 0.0) & (t > _EPS)
        best = np.minimum(best, np.where(ok, t, np.inf).min(axis=1))

    return best


def point_segment_distance(points, segments):
    """(N, 2) points x (S, 4) segments -> (N, S) Euclidean distances."""
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    seg = np.asarray(segments, dtype=float)
    a = seg[:, :2]
    e = seg[:, 2:] - a
    len2 = np.maximum((e**2).sum(axis=1), 1e-12)
    ap0 = p[:, None, 0] - a[None, :, 0]
    ap1 = p[:, None, 1] - a[None, :, 1]
    s = np.clip((ap0 * e[None, :, 0] + ap1 * e[None, :, 1]) / len2, 0.0, 1.0)
    dx = ap0 - s * e[None, :, 0]
    dy = ap1 - s * e[None, :, 1]
    return np.hypot(dx, dy)


def box_segments(x0, y0, x1, y1):
    """Four segments of an axis-aligned rectangle."""
    return np.array(
        [[x0, y0, x1, y0], [x1, y0, x1, y1], [x1, y1, x0, y1], [x0, y1, x0, y0]],
        dtype=float,
    )


def wrap_angle(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi
