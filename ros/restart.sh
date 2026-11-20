#!/bin/bash
# Restart the Twin Wings demo. Usage: restart.sh [drive:=true|false]
pkill -f '[r]os2 launch' ; pkill -f '[g]z sim' ; pkill -f '[r]viz2' ; pkill -f '[m]cl_node.py' ; pkill -f '[d]rive_and_occlude.py'
sleep 3
source /opt/ros/jazzy/setup.bash
export DISPLAY=:1
cd ~/ssol_sim && nohup ros2 launch ros/twin_wings.launch.py "$@" > /tmp/ssol_launch.log 2>&1 &
echo "launched: $*  (log: /tmp/ssol_launch.log)"
