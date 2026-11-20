#!/usr/bin/env python3
"""ROS 2 wrapper around twinwings.MCL.

Subscribes:  /scan (sensor_msgs/LaserScan), /odom (nav_msgs/Odometry)
Publishes:   /map (OccupancyGrid, latched), /mcl/particles (PoseArray),
             /mcl/markers (MarkerArray: per-wing mass + commit status),
             /mcl/mode_mass (Float32MultiArray)

The same node is meant to run on the real robot: only the topics and the
`start_*` pose of the initial multi-modal belief change. The scan is
subsampled to `n_beams` evenly spaced bearings, as in the experiments.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import ColorRGBA, Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from twinwings import MCL, SSOL, BeamMixture, LidarConfig, TwinWings, expected_ranges  # noqa: E402
from twinwings.geometry import wrap_angle  # noqa: E402

LIKELIHOODS = {"beam_mixture": BeamMixture, "ssol": SSOL}
WING_NAMES = ["Wing A", "Wing B"]


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class SsolMcl(Node):
    def __init__(self):
        super().__init__("ssol_mcl")
        p = self.declare_parameter
        self.asym = p("asym", 0.30).value
        self.n_beams = p("n_beams", 60).value
        self.n_particles = p("n_particles", 500).value
        self.tau = p("tau_c", 0.95).value
        self.start = np.array([p("start_x", 3.6).value, p("start_y", 1.8).value, p("start_yaw", math.pi / 2).value])
        self.true_wing = p("true_wing", 0).value  # sim only, for labelling; -1 on the real robot
        lik_name = p("likelihood", "beam_mixture").value

        self.world = TwinWings(asym=self.asym)
        self.lidar = LidarConfig(n_beams=self.n_beams)
        self.mcl = MCL(self.world, LIKELIHOODS[lik_name](), self.lidar, self.n_particles,
                       rng=np.random.default_rng(0))
        self.lik_name = lik_name
        self.odom = None
        self.last_odom = None
        self.initialized = False
        self.t0 = None
        self.commit = None  # (wing, t)

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.pub_map = self.create_publisher(OccupancyGrid, "/map", latched)
        self.pub_parts = self.create_publisher(PoseArray, "/mcl/particles", 10)
        self.pub_mark = self.create_publisher(MarkerArray, "/mcl/markers", 10)
        self.pub_mass = self.create_publisher(Float32MultiArray, "/mcl/mode_mass", 10)
        self.create_subscription(Odometry, "/odom", self.on_odom, 20)
        self.create_subscription(LaserScan, "/scan", self.on_scan, qos_profile_sensor_data)
        self.publish_map()
        self.get_logger().info(f"SSOL MCL ready: {lik_name}, asym {self.asym} m, {self.n_beams} beams, "
                               f"{self.n_particles} particles, tau_c {self.tau}")

    # ------------------------------------------------------------------ inputs
    def on_odom(self, msg: Odometry):
        pp = msg.pose.pose
        self.odom = np.array([pp.position.x, pp.position.y, yaw_of(pp.orientation)])

    def subsample(self, msg: LaserScan) -> np.ndarray:
        ranges = np.asarray(msg.ranges, dtype=float)
        n = len(ranges)
        idx = np.round(((self.lidar.bearings() - msg.angle_min) % (2 * math.pi)) / msg.angle_increment).astype(int) % n
        z = ranges[idx]
        bad = ~np.isfinite(z) | (z < max(msg.range_min, self.lidar.min_range)) | \
            (z > min(msg.range_max, self.lidar.max_range) - 1e-3)
        return np.where(bad, np.nan, z)

    def on_scan(self, msg: LaserScan):
        if self.odom is None:
            return
        t = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        if not self.initialized:
            # seed at the robot's current pose: odom starts at the spawn/start pose,
            # so pose in map = start (+) odom. Correct even if the node starts late.
            sx, sy, sth = self.start
            ox, oy, oth = self.odom
            seed = np.array([sx + math.cos(sth) * ox - math.sin(sth) * oy,
                             sy + math.sin(sth) * ox + math.cos(sth) * oy,
                             float(wrap_angle(sth + oth))])
            self.mcl.init_modes(seed)
            self.last_odom = self.odom.copy()
            self.initialized = True
            self.t0 = t
        else:
            x0, y0, th0 = self.last_odom
            x1, y1, th1 = self.odom
            c, s = math.cos(th0), math.sin(th0)
            dx = c * (x1 - x0) + s * (y1 - y0)
            dy = -s * (x1 - x0) + c * (y1 - y0)
            self.mcl.predict((dx, dy, float(wrap_angle(th1 - th0))))
            self.last_odom = self.odom.copy()
        self.last_z = self.subsample(msg)
        self.mcl.update(self.last_z)
        masses = self.mcl.mode_masses()
        if self.commit is None and masses.max() >= self.tau:
            k = int(np.argmax(masses))
            self.commit = (k, t - self.t0)
            verdict = "" if self.true_wing < 0 else (" (CORRECT)" if k == self.true_wing else " (WRONG)")
            self.get_logger().warn(f"COMMIT to {WING_NAMES[k]} at t = {t - self.t0:.1f} s{verdict}")
        self.publish(msg.header.stamp, masses, t - self.t0)

    # ----------------------------------------------------------------- outputs
    def publish(self, stamp, masses, t_rel):
        pa = PoseArray()
        pa.header.frame_id = "map"
        pa.header.stamp = stamp
        for x, y, th in self.mcl.particles:
            ps = Pose()
            ps.position.x, ps.position.y = float(x), float(y)
            ps.orientation.z, ps.orientation.w = math.sin(th / 2), math.cos(th / 2)
            pa.poses.append(ps)
        self.pub_parts.publish(pa)
        self.pub_mass.publish(Float32MultiArray(data=[float(m) for m in masses]))

        ma = MarkerArray()
        for k in range(self.world.K):
            m = self._text(k, stamp, self.world.offset(k) + self.world.wing_w / 2, self.world.wing_h + 0.9,
                           f"{WING_NAMES[k]}: {masses[k]:.3f}", 0.45)
            g = float(masses[k])
            m.color.r, m.color.g, m.color.b = 1.0 - g, 0.2 + 0.8 * g, 0.2
            ma.markers.append(m)
        if self.commit is None:
            status, col = f"t = {t_rel:4.1f} s   undecided (tau_c = {self.tau})", (0.9, 0.9, 0.9)
        else:
            k, tc = self.commit
            wrong = self.true_wing >= 0 and k != self.true_wing
            status = f"COMMITTED to {WING_NAMES[k]} at {tc:.1f} s" + (
                "" if self.true_wing < 0 else ("  -- WRONG wing!" if wrong else "  -- correct"))
            col = (1.0, 0.25, 0.25) if wrong else (0.3, 1.0, 0.3)
        m = self._text(99, stamp, (self.world.offset(1) + self.world.wing_w) / 2, -1.2,
                       f"[{self.lik_name}]  {status}", 0.4)
        m.color.r, m.color.g, m.color.b = col
        ma.markers.append(m)
        ma.markers.append(self._walls(stamp))
        ma.markers.extend(self._ssol_view(stamp))
        self.pub_mark.publish(ma)

    def _ssol_view(self, stamp):
        """Show what SSOL believes: for the best particle of each wing, draw the
        beams it thinks are inside the occluded sector (orange) and label the
        posterior probability that the scan contains an occluder."""
        lik = self.mcl.likelihood
        if not hasattr(lik, "occlusion_posterior") or getattr(self, "last_z", None) is None:
            return []
        out = []
        z = self.last_z
        bearings = self.lidar.bearings()
        for k in range(self.world.K):
            blk = slice(k * self.mcl.m, (k + 1) * self.mcl.m)
            pose = self.mcl.particles[blk][int(np.argmax(self.mcl.logw[blk]))]
            r = expected_ranges(self.world, pose, self.lidar)
            p_beam, p_any = lik.occlusion_posterior(z, r)
            p_beam, p_any = p_beam[0], float(p_any[0])

            rays = Marker()
            rays.header.frame_id, rays.header.stamp = "map", stamp
            rays.ns, rays.id, rays.type, rays.action = "ssol_sector", 200 + k, Marker.LINE_LIST, Marker.ADD
            rays.pose.orientation.w = 1.0
            rays.scale.x = 0.06
            for b in np.flatnonzero(p_beam > 0.2):
                rng = z[b] if np.isfinite(z[b]) else min(r[0, b], self.lidar.max_range)
                a = pose[2] + bearings[b]
                for (x, y) in ((pose[0], pose[1]), (pose[0] + rng * math.cos(a), pose[1] + rng * math.sin(a))):
                    pt = Pose().position
                    pt.x, pt.y, pt.z = float(x), float(y), 0.05
                    rays.points.append(pt)
                col = ColorRGBA(r=1.0, g=0.55, b=0.0, a=float(min(1.0, 0.3 + p_beam[b])))
                rays.colors.extend([col, col])
            rays.color.a = 1.0
            if not rays.points:
                rays.action = Marker.DELETE
            out.append(rays)

            label = self._text(300 + k, stamp, self.world.offset(k) + self.world.wing_w / 2, -0.55,
                               f"SSOL: occluder here? {100 * p_any:.0f}%  "
                               f"({int((p_beam > 0.5).sum())} beams)", 0.32)
            label.color.r, label.color.g, label.color.b = 1.0, 0.6, 0.1
            out.append(label)
        return out

    def _walls(self, stamp):
        """Map walls as a line marker (RViz's Map display fails under the
        container's software OpenGL, so draw the walls this way too)."""
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = stamp
        m.ns, m.id, m.type, m.action = "walls", 50, Marker.LINE_LIST, Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.05
        m.color.r = m.color.g = m.color.b = 0.85
        m.color.a = 1.0
        for x0, y0, x1, y1 in self.world.segments:
            for x, y in ((x0, y0), (x1, y1)):
                pt = Pose().position
                pt.x, pt.y = float(x), float(y)
                m.points.append(pt)
        return m

    def _text(self, mid, stamp, x, y, text, size):
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = stamp
        m.ns, m.id, m.type, m.action = "ssol", mid, Marker.TEXT_VIEW_FACING, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x), float(y), 0.5
        m.pose.orientation.w = 1.0
        m.scale.z = size
        m.color.a = 1.0
        m.text = text
        return m

    def publish_map(self, res=0.05):
        x_min, y_min = -1.0, -2.0
        x_max = self.world.offset(1) + self.world.wing_w + 1.0
        y_max = self.world.wing_h + 1.5
        w, h = int(round((x_max - x_min) / res)), int(round((y_max - y_min) / res))
        grid = np.zeros((h, w), dtype=np.int8)
        for x0, y0, x1, y1 in self.world.segments:
            n = int(max(abs(x1 - x0), abs(y1 - y0)) / (res / 3)) + 2
            xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
            ix = np.clip(((xs - x_min) / res).astype(int), 0, w - 1)
            iy = np.clip(((ys - y_min) / res).astype(int), 0, h - 1)
            grid[iy, ix] = 100
        msg = OccupancyGrid()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = res
        msg.info.width, msg.info.height = w, h
        msg.info.origin.position.x, msg.info.origin.position.y = x_min, y_min
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.reshape(-1).tolist()
        self.pub_map.publish(msg)


def main():
    rclpy.init()
    node = SsolMcl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
