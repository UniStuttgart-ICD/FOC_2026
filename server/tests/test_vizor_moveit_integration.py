from __future__ import annotations

from uuid import uuid4

import pytest

CURRENT_POSE_RECOVERY_COMMAND = """docker exec vizor-demo bash -lc 'source /opt/ros/noetic/setup.bash; source /root/catkin_ws/devel/setup.bash; python3 - <<"PY"
import sys, rospy, moveit_commander
moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node("query_pose", anonymous=True)
group = moveit_commander.MoveGroupCommander("arm", ns="UR10", robot_description="UR10/robot_description")
pose = group.get_current_pose().pose
print(pose)
PY'"""


def _require_vizor_integration(vizor_integration: bool) -> None:
    if not vizor_integration:
        pytest.skip("Pass --vizor-integration with vizor-demo running")


def _target_pose() -> dict[str, dict[str, float]]:
    return {
        "position": {"x": 0.5723589519983855, "y": 0.3941410000780623, "z": 0.6235999970798317},
        "orientation": {
            "x": -2.0030704870235343e-16,
            "y": -0.7071067812590626,
            "z": -0.7071067811140325,
            "w": 4.329780280011331e-17,
        },
    }


@pytest.mark.integration
def test_plan_free_motion_against_running_vizor(vizor_integration):
    _require_vizor_integration(vizor_integration)

    from moveit_mcp.server import build_tools

    tools = build_tools(host='localhost', port=9090)
    plan_name = f"pi_integration_free_{uuid4().hex[:8]}"

    result = tools.plan_free_motion(
        "UR10",
        plan_name,
        _target_pose(),
        timeout_s=20.0,
    )

    assert result["ok"] is True
    assert result["feedback"]["can_execute"] is True
    assert result["raw"]["plan_name"] == plan_name
    assert result["raw"]["trajectory_points"] > 0


@pytest.mark.integration
def test_execute_verified_plan_against_running_vizor(vizor_integration):
    _require_vizor_integration(vizor_integration)

    from moveit_mcp.server import build_tools

    tools = build_tools(host='localhost', port=9090)
    plan_name = f"pi_integration_execute_{uuid4().hex[:8]}"

    planned = tools.plan_free_motion(
        "UR10",
        plan_name,
        _target_pose(),
        timeout_s=20.0,
    )
    assert planned["ok"] is True

    executed = tools.execute_plan("UR10", plan_name, timeout_s=20.0)

    assert executed["ok"] is True
    assert executed["verification"]["result"] == "pass"
