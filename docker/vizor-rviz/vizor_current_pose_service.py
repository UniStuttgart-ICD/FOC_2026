#!/usr/bin/env python3
"""Expose the current UR10 MoveIt end-effector pose as a read-only ROS service."""

import json
import sys
import time
import traceback

import moveit_commander
import rospy
from std_srvs.srv import Trigger, TriggerResponse

ROBOT_NAME = "UR10"
GROUP_NAME = "arm"
ROBOT_DESCRIPTION = f"{ROBOT_NAME}/robot_description"
SERVICE_NAME = f"/{ROBOT_NAME}/get_current_pose"


def _pose_payload(group):
    pose = group.get_current_pose().pose
    planning_frame = group.get_planning_frame() or "base_link"
    return {
        "ok": True,
        "robot": ROBOT_NAME,
        "planning_frame": planning_frame,
        "pose": {
            "position": {
                "x": float(pose.position.x),
                "y": float(pose.position.y),
                "z": float(pose.position.z),
            },
            "orientation": {
                "x": float(pose.orientation.x),
                "y": float(pose.orientation.y),
                "z": float(pose.orientation.z),
                "w": float(pose.orientation.w),
            },
        },
    }


def _connect_group():
    while not rospy.is_shutdown():
        try:
            return moveit_commander.MoveGroupCommander(
                GROUP_NAME,
                ns=ROBOT_NAME,
                robot_description=ROBOT_DESCRIPTION,
            )
        except Exception as exc:  # pragma: no cover - runs inside ROS container
            rospy.logwarn("Waiting for MoveIt group %s/%s: %s", ROBOT_NAME, GROUP_NAME, exc)
            time.sleep(2.0)
    raise RuntimeError("ROS shutdown before MoveIt group was available")


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("vizor_current_pose_service", anonymous=False)
    group = _connect_group()

    def handle(_request):
        try:
            return TriggerResponse(success=True, message=json.dumps(_pose_payload(group), sort_keys=True))
        except Exception:  # pragma: no cover - runs inside ROS container
            rospy.logerr("Failed to read current MoveIt pose:\n%s", traceback.format_exc())
            return TriggerResponse(success=False, message="current pose unavailable")

    rospy.Service(SERVICE_NAME, Trigger, handle)
    rospy.loginfo("Current pose service ready at %s", SERVICE_NAME)
    rospy.spin()


if __name__ == "__main__":
    main()
