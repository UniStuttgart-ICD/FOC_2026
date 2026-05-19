#!/usr/bin/env bash
set -euo pipefail

export DISPLAY="${DISPLAY:-:1}"
export NOVNC_PORT="${NOVNC_PORT:-6080}"
export VNC_PORT="${VNC_PORT:-5901}"
export VIZOR_ENABLE_MTC_PROOF="${VIZOR_ENABLE_MTC_PROOF:-0}"
export MOVEIT_PLANNING_RUN_ID="${MOVEIT_PLANNING_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
export MOVEIT_PLANNING_RUN_DIR="${MOVEIT_PLANNING_RUN_DIR:-/root/catkin_ws/logs/moveit_planning/runs/${MOVEIT_PLANNING_RUN_ID}}"
export MOVEIT_PLANNING_LOG_PATH="${MOVEIT_PLANNING_LOG_PATH:-${MOVEIT_PLANNING_RUN_DIR}/moveit_planning.jsonl}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-${MOVEIT_PLANNING_RUN_DIR}/ros}"

mkdir -p "$MOVEIT_PLANNING_RUN_DIR" "$(dirname "$MOVEIT_PLANNING_LOG_PATH")" "$ROS_LOG_DIR"

Xvfb "$DISPLAY" -screen 0 1600x1000x24 -ac +extension GLX +render -noreset &
XVFB_PID=$!

sleep 1
fluxbox >/tmp/fluxbox.log 2>&1 &
x11vnc -display "$DISPLAY" -forever -shared -nopw -rfbport "$VNC_PORT" >/tmp/x11vnc.log 2>&1 &
websockify --web=/usr/share/novnc/ "$NOVNC_PORT" "localhost:$VNC_PORT" >/tmp/novnc.log 2>&1 &

set +u
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
set -u

# Expose ROS parameter APIs through rosbridge for MCP safety checks.
rosrun rosapi rosapi_node >/tmp/rosapi.log 2>&1 || true &

# Expose read-only MoveIt current pose for the MCP agent.
/usr/local/bin/vizor_current_pose_service.py >/tmp/vizor_current_pose_service.log 2>&1 || true &

if [ "$VIZOR_ENABLE_MTC_PROOF" = "1" ]; then
  if [ -x /usr/local/bin/vizor_mtc_pick_server.py ]; then
    /usr/local/bin/vizor_mtc_pick_server.py >/tmp/vizor_mtc_pick_server.log 2>&1 || true &
  elif [ -f /usr/local/bin/vizor_mtc_pick_server.py ]; then
    python3 /usr/local/bin/vizor_mtc_pick_server.py >/tmp/vizor_mtc_pick_server.log 2>&1 || true &
  else
    echo "VIZOR_ENABLE_MTC_PROOF=1 but /usr/local/bin/vizor_mtc_pick_server.py is missing." >/tmp/vizor_mtc_pick_server.log
  fi
fi

# Launch RViz after ROS nodes have had time to start. Failure must not kill Vizor.
(
  sleep 12
  python3 - <<'PY'
import time

import rospy

pairs = (
    ("/UR10/robot_description", "/robot_description"),
    ("/UR10/robot_description_semantic", "/robot_description_semantic"),
    ("/UR10/robot_description_kinematics", "/robot_description_kinematics"),
    ("/UR10/robot_description_planning", "/robot_description_planning"),
)
deadline = time.time() + 20
while time.time() < deadline and not rospy.has_param("/UR10/robot_description"):
    time.sleep(0.5)
for src, dst in pairs:
    if rospy.has_param(src):
        rospy.set_param(dst, rospy.get_param(src))
PY
  roslaunch vizor_package rviz_ur10.launch >/tmp/rviz.log 2>&1 || true
) &

if [ "$#" -gt 0 ]; then
  exec "$@"
fi

exec roslaunch --wait vizor_package vizor2ros.launch \
  production:=false \
  task_log_path:=/root/catkin_ws/data/task_logs \
  stored_motion_path:=/root/catkin_ws/data/stored_plans \
  mongo_db_path:=/root/catkin_ws/data/mongodb \
  active:=HOLO1 \
  physical:=false
