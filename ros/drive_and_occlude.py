#!/usr/bin/env python3
"""Scripted episode for the Gazebo demo (simulation only).

Drives the robot with the same velocity commands as twinwings.episode
(DEFAULT_COMMANDS) and, after `dwell` seconds, moves the occluder box out of
doorway D (teleport via the Gazebo set_pose service). On the real robot you
would drive by hand / teleop and move a real box or person instead.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from twinwings.episode import DEFAULT_COMMANDS, _command_at  # noqa: E402


class Driver(Node):
    def __init__(self):
        super().__init__("ssol_driver")
        self.settle = self.declare_parameter("settle", 3.0).value
        self.dwell = self.declare_parameter("dwell", 20.0).value
        self.world = self.declare_parameter("world", "twin_wings").value
        self.occ_to = (self.declare_parameter("occluder_to_x", 6.0).value,
                       self.declare_parameter("occluder_to_y", 2.6).value)
        self.pub = self.create_publisher(TwistStamped, "/cmd_vel", 10)
        self.t_start = None
        self.moved = False
        self.done = False
        self.create_timer(0.05, self.tick)

    def tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if now <= 0.0:
            return  # waiting for /clock
        if self.t_start is None:
            self.t_start = now + self.settle
            self.get_logger().info(f"episode starts in {self.settle:.0f} s; occluder leaves at t = {self.dwell:.0f} s")
        t = now - self.t_start
        v, w = (0.0, 0.0) if t < 0 else _command_at(DEFAULT_COMMANDS, t)
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_footprint"
        msg.twist.linear.x, msg.twist.angular.z = float(v), float(w)
        self.pub.publish(msg)
        if not self.done and t > sum(d for _, _, d in DEFAULT_COMMANDS):
            self.done = True
            self.get_logger().info("path finished; robot idles")
        if not self.moved and t >= self.dwell:
            self.moved = True
            x, y = self.occ_to
            req = f'name: "occluder_box", position: {{x: {x}, y: {y}, z: 0.3}}'
            cmd = ["gz", "service", "-s", f"/world/{self.world}/set_pose", "--reqtype", "gz.msgs.Pose",
                   "--reptype", "gz.msgs.Boolean", "--timeout", "3000", "--req", req]
            out = subprocess.run(cmd, capture_output=True, text=True)
            self.get_logger().warn(f"t = {t:.1f} s: occluder box moved out of doorway D -> "
                                   f"{out.stdout.strip() or out.stderr.strip()}")


def main():
    rclpy.init()
    node = Driver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
