"""Twin Wings demo: Gazebo + TurtleBot3 (RPLidar A1 limits) + SSOL MCL + RViz.

Inside the container:
  cd ~/ssol_sim && python3 ros/make_world.py --asym 0.3
  ros2 launch ros/twin_wings.launch.py
Arguments: drive:=false to skip the scripted path/occluder (teleop instead),
           gui:=false for headless Gazebo.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
                            SetEnvironmentVariable, TimerAction)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "generated")
START = {"x": "3.6", "y": "1.8", "yaw": "1.5708"}


def generate_launch_description():
    tb3 = get_package_share_directory("turtlebot3_gazebo")
    sim = {"use_sim_time": True}
    drive, gui = LaunchConfiguration("drive"), LaunchConfiguration("gui")

    gz_gui = ExecuteProcess(cmd=["gz", "sim", "-r", os.path.join(GEN, "twin_wings.sdf")],
                            output="screen", condition=IfCondition(gui))
    gz_headless = ExecuteProcess(cmd=["gz", "sim", "-r", "-s", os.path.join(GEN, "twin_wings.sdf")],
                                 output="screen", condition=UnlessCondition(gui))
    rsp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(tb3, "launch", "robot_state_publisher.launch.py")),
        launch_arguments={"use_sim_time": "true"}.items())
    spawn = Node(package="ros_gz_sim", executable="create", output="screen",
                 arguments=["-name", "burger_a1", "-file", os.path.join(GEN, "burger_a1", "model.sdf"),
                            "-x", START["x"], "-y", START["y"], "-z", "0.01", "-Y", START["yaw"]])
    bridge = Node(package="ros_gz_bridge", executable="parameter_bridge", output="screen",
                  parameters=[{"config_file": os.path.join(tb3, "params", "turtlebot3_burger_bridge.yaml")}, sim])
    # map -> odom: odom starts at the spawn pose (sim only; on the robot the MCL would publish this)
    map_odom = Node(package="tf2_ros", executable="static_transform_publisher",
                    arguments=["--x", START["x"], "--y", START["y"], "--yaw", START["yaw"],
                               "--frame-id", "map", "--child-frame-id", "odom"], parameters=[sim])
    mcl = ExecuteProcess(cmd=["python3", os.path.join(HERE, "mcl_node.py"), "--ros-args",
                              "-p", "use_sim_time:=true", "-p", ["likelihood:=", LaunchConfiguration("likelihood")],
                              "-p", f"start_x:={START['x']}",
                              "-p", f"start_y:={START['y']}", "-p", f"start_yaw:={START['yaw']}"],
                         output="screen")
    driver = ExecuteProcess(cmd=["python3", os.path.join(HERE, "drive_and_occlude.py"), "--ros-args",
                                 "-p", "use_sim_time:=true"], output="screen", condition=IfCondition(drive))
    rviz = Node(package="rviz2", executable="rviz2", arguments=["-d", os.path.join(HERE, "twin_wings.rviz")],
                parameters=[sim], output="log")
    # top-down Gazebo camera over both wings (the world's <gui><camera> is ignored by gz-sim 8)
    camera = ExecuteProcess(cmd=["gz", "service", "-s", "/gui/move_to/pose", "--reqtype", "gz.msgs.GUICamera",
                                 "--reptype", "gz.msgs.Boolean", "--timeout", "5000", "--req",
                                 "pose: {position: {x: 8.0, y: 2.6, z: 13.0}, "
                                 "orientation: {x: -0.5, y: 0.5, z: 0.5, w: 0.5}}"],
                            output="log", condition=IfCondition(gui))

    return LaunchDescription([
        DeclareLaunchArgument("drive", default_value="true"),
        DeclareLaunchArgument("likelihood", default_value="ssol", description="ssol | beam_mixture"),
        DeclareLaunchArgument("gui", default_value="true"),
        SetEnvironmentVariable("TURTLEBOT3_MODEL", "burger"),
        SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", os.path.join(tb3, "models") + ":" + GEN),
        gz_gui, gz_headless, rsp, map_odom, rviz,
        TimerAction(period=4.0, actions=[spawn, bridge]),
        TimerAction(period=7.0, actions=[camera]),
        TimerAction(period=8.0, actions=[mcl]),
        TimerAction(period=10.0, actions=[driver]),
    ])
