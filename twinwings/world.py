"""Twin Wings layout and moving occluders.

Two identical rooms ("wings") side by side. Wing B is Wing A shifted by
`wing_w + gap` in x, so a pose in one wing has an exact aliased twin in the
other. The only difference is doorway D in Wing A's top wall: a recess of depth
`asym` metres (0, 0.15 or 0.30). At asym = 0 the wings are identical.

Interpretation note: the proposal says the wings "differ only in one doorway"
with asymmetry {0, 0.15, 0.30} m. Here that asymmetry is the depth of the
doorway recess. Confirm this with the team.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import box_segments, point_segment_distance

ROBOT_RADIUS = 0.12


@dataclass
class TwinWings:
    asym: float = 0.30
    wing_w: float = 7.0
    wing_h: float = 5.0
    gap: float = 2.0
    door_x: tuple = (3.0, 4.2)  # doorway D extent along Wing A's top wall (local x)
    door_wing: int = 0          # wing that has the doorway
    K: int = 2                  # number of aliased modes
    segments: np.ndarray = field(init=False, repr=False)
    obstacle_boxes: list = field(init=False, repr=False)

    # Interior features (local coordinates) that make each pose unique within
    # a wing, so the only ambiguity left is *which* wing.
    PILLAR = (1.8, 3.0, 2.3, 3.5)
    CABINET = (5.8, 3.4, 6.6, 4.2)
    PARTITION = (4.8, 0.0, 4.8, 1.3)

    def __post_init__(self):
        segs, boxes = [], []
        for k in range(self.K):
            ox = self.offset(k)
            segs.append(self._wing_segments(k) + np.array([ox, 0, ox, 0]))
            for (x0, y0, x1, y1) in (self.PILLAR, self.CABINET):
                boxes.append((x0 + ox, y0, x1 + ox, y1))
        self.segments = np.vstack(segs)
        self.obstacle_boxes = boxes

    # ------------------------------------------------------------------ layout
    def offset(self, k: int) -> float:
        return k * (self.wing_w + self.gap)

    def _wing_segments(self, k: int) -> np.ndarray:
        w, h = self.wing_w, self.wing_h
        s = [[0, 0, w, 0], [w, 0, w, h], [0, h, 0, 0]]  # bottom, right, left
        dx0, dx1 = self.door_x
        if k == self.door_wing and self.asym > 0:
            a = self.asym
            s += [[0, h, dx0, h], [dx0, h, dx0, h + a], [dx0, h + a, dx1, h + a],
                  [dx1, h + a, dx1, h], [dx1, h, w, h]]
        else:
            s += [[0, h, w, h]]
        s = np.array(s, dtype=float)
        feats = [box_segments(*self.PILLAR), box_segments(*self.CABINET),
                 np.array([self.PARTITION], dtype=float)]
        return np.vstack([s] + feats)

    def door_center(self, k: int) -> np.ndarray:
        """Centre of the doorway opening (on the wall line) in wing k.

        In the wing without the doorway this is the same spot on its plain wall,
        so occluder placement does not depend on where the cue is (assumption A5).
        """
        return np.array([self.offset(k) + 0.5 * sum(self.door_x), self.wing_h])

    # ------------------------------------------------------------------ modes
    def wing_of(self, xy) -> np.ndarray:
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        return (xy[:, 0] >= self.wing_w + 0.5 * self.gap).astype(int)

    def alias(self, pose, k: int) -> np.ndarray:
        """Move a pose to the same local spot in wing k."""
        p = np.array(pose, dtype=float)
        src = self.wing_of(p[:2])[0]
        p[0] += self.offset(k) - self.offset(src)
        return p

    def is_free(self, xy, margin: float = ROBOT_RADIUS) -> np.ndarray:
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        x, y = xy[:, 0], xy[:, 1]
        inside = np.zeros(len(xy), dtype=bool)
        for k in range(self.K):
            ox = self.offset(k)
            inside |= (x > ox + margin) & (x < ox + self.wing_w - margin) & \
                       (y > margin) & (y < self.wing_h - margin)
        for (x0, y0, x1, y1) in self.obstacle_boxes:
            inside &= ~((x > x0) & (x < x1) & (y > y0) & (y < y1))
        clear = point_segment_distance(xy, self.segments).min(axis=1) > margin
        return inside & clear


# ---------------------------------------------------------------- occluders
OCCLUDER_KINDS = ("person", "chair", "box")


def occluder_shapes(kind: str, center):
    """Return (segments, circles) for an occluder centred at `center`.

    person: two legs, r = 6 cm, 25 cm apart (two separated arcs)
    chair:  four legs, r = 2 cm, on a 45 cm square (several small returns)
    box:    1.0 m x 0.25 m rectangle (one contiguous run)
    """
    cx, cy = float(center[0]), float(center[1])
    empty_s, empty_c = np.zeros((0, 4)), np.zeros((0, 3))
    if kind == "person":
        return empty_s, np.array([[cx - 0.125, cy, 0.06], [cx + 0.125, cy, 0.06]])
    if kind == "chair":
        h = 0.225
        return empty_s, np.array([[cx + sx * h, cy + sy * h, 0.02]
                                  for sx in (-1, 1) for sy in (-1, 1)])
    if kind == "box":
        return box_segments(cx - 0.5, cy - 0.125, cx + 0.5, cy + 0.125), empty_c
    raise ValueError(f"unknown occluder kind {kind!r}")


# Distance from an occluder's centre to its far side (towards the wall).
OCCLUDER_HALF_DEPTH = {"person": 0.06, "chair": 0.245, "box": 0.125}


@dataclass
class OccluderTrack:
    """An occluder that stands in the doorway, then walks away along waypoints."""
    kind: str
    waypoints: np.ndarray        # (W, 2); waypoints[0] is the standing spot
    dwell: float = 20.0          # seconds standing at waypoints[0]
    speed: float = 0.5           # m/s once it starts moving

    def position(self, t: float) -> np.ndarray:
        wp = np.asarray(self.waypoints, dtype=float)
        if t <= self.dwell or len(wp) == 1:
            return wp[0]
        dist = (t - self.dwell) * self.speed
        for a, b in zip(wp[:-1], wp[1:]):
            seg = np.linalg.norm(b - a)
            if dist <= seg:
                return a + (b - a) * (dist / max(seg, 1e-9))
            dist -= seg
        return wp[-1]

    def shapes(self, t: float):
        return occluder_shapes(self.kind, self.position(t))


def doorway_occluder(world: TwinWings, kind: str, wing: int, dwell: float = 20.0,
                     speed: float = 0.5) -> OccluderTrack:
    """Occluder standing in doorway D of `wing` with its far side against the
    recess end (the wall line when asym = 0), then walking into the room and
    off to the right. Its near face sits close to the plain wall line, which is
    what lets the wrong wing explain it as a wall hit."""
    door = world.door_center(wing)
    recess = world.asym if wing == world.door_wing else 0.0
    half = OCCLUDER_HALF_DEPTH[kind]
    stand = np.array([door[0], world.wing_h + recess - half - 0.01])
    ox = world.offset(wing)
    waypoints = np.array([stand, [door[0], 4.0], [ox + 6.0, 2.6]])
    return OccluderTrack(kind, waypoints, dwell=dwell, speed=speed)
