"""Generate the Gazebo world and the robot model for the Twin Wings demo.

Writes into ros/generated/:
  twin_wings.sdf        - walls from twinwings.TwinWings + the occluder box in doorway D
  burger_a1/model.sdf   - TurtleBot3 Burger with its LiDAR set to RPLidar A1 limits
                          (0.15-12 m, 5.5 Hz, 360 samples). Stand-in for the MBot.

Run inside the ROS 2 container (needs the turtlebot3_gazebo package):
  python3 ros/make_world.py --asym 0.3
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from twinwings import TwinWings, doorway_occluder  # noqa: E402

WALL_H = 0.6
WALL_T = 0.02


def wall_links(segments):
    links = []
    for i, (x0, y0, x1, y1) in enumerate(segments):
        length = math.hypot(x1 - x0, y1 - y0)
        if length < 1e-6:
            continue
        cx, cy, yaw = (x0 + x1) / 2, (y0 + y1) / 2, math.atan2(y1 - y0, x1 - x0)
        geom = f"<geometry><box><size>{length + WALL_T:.4f} {WALL_T} {WALL_H}</size></box></geometry>"
        links.append(f"""
      <link name="wall_{i}">
        <pose>{cx:.4f} {cy:.4f} {WALL_H / 2} 0 0 {yaw:.6f}</pose>
        <collision name="c">{geom}</collision>
        <visual name="v">{geom}
          <material><ambient>0.55 0.55 0.6 1</ambient><diffuse>0.55 0.55 0.6 1</diffuse></material>
        </visual>
      </link>""")
    return "".join(links)


def world_sdf(world: TwinWings, occ_xy) -> str:
    cx = (world.offset(1) + world.wing_w) / 2
    box = "<geometry><box><size>1.0 0.25 0.6</size></box></geometry>"
    return f"""<?xml version="1.0"?>
<sdf version="1.8">
  <world name="twin_wings">
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
    <scene><shadows>false</shadows><ambient>0.6 0.6 0.6 1</ambient></scene>
    <gui fullscreen="0">
      <camera name="user_camera">
        <pose>{cx:.2f} -4.5 11 0 0.95 1.5708</pose>
      </camera>
    </gui>
    <light type="directional" name="sun">
      <cast_shadows>false</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.3 0.2 -0.9</direction>
    </light>
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="c"><geometry><plane><normal>0 0 1</normal><size>40 40</size></plane></geometry></collision>
        <visual name="v"><geometry><plane><normal>0 0 1</normal><size>40 40</size></plane></geometry>
          <material><ambient>0.85 0.85 0.85 1</ambient><diffuse>0.85 0.85 0.85 1</diffuse></material>
        </visual>
      </link>
    </model>
    <model name="twin_wings_walls">
      <static>true</static>{wall_links(world.segments)}
    </model>
    <model name="occluder_box">
      <static>true</static>
      <pose>{occ_xy[0]:.4f} {occ_xy[1]:.4f} 0.3 0 0 0</pose>
      <link name="link">
        <collision name="c">{box}</collision>
        <visual name="v">{box}
          <material><ambient>0.85 0.15 0.15 1</ambient><diffuse>0.85 0.15 0.15 1</diffuse></material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
"""


def robot_sdf(tb3_models: Path) -> str:
    src = (tb3_models / "turtlebot3_burger" / "model.sdf").read_text()
    src = src.replace('<model name="turtlebot3_burger">', '<model name="burger_a1">')
    lidar = src[src.index('name="hls_lfcd_lds"'):]
    block = lidar[:lidar.index("</sensor>")]
    new = re.sub(r"<min>[0-9.]+</min>", "<min>0.15</min>", block, count=1)
    new = re.sub(r"<max>[0-9.]+</max>", "<max>12.0</max>", new, count=1)
    new = re.sub(r"<update_rate>[0-9.]+</update_rate>", "<update_rate>5.5</update_rate>", new, count=1)
    return src.replace(block, new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asym", type=float, default=0.30)
    ap.add_argument("--tb3-models", default="/opt/ros/jazzy/share/turtlebot3_gazebo/models")
    args = ap.parse_args()

    world = TwinWings(asym=args.asym)
    occ = doorway_occluder(world, "box", 0)
    out = HERE / "generated"
    (out / "burger_a1").mkdir(parents=True, exist_ok=True)
    (out / "twin_wings.sdf").write_text(world_sdf(world, occ.position(0.0)))
    tb3 = Path(args.tb3_models)
    if tb3.exists():
        (out / "burger_a1" / "model.sdf").write_text(robot_sdf(tb3))
    else:
        print(f"warning: {tb3} not found, robot model not written", file=sys.stderr)
    print(f"wrote {out}/twin_wings.sdf (asym {args.asym} m, occluder at {occ.position(0.0).round(3)})")


if __name__ == "__main__":
    main()
