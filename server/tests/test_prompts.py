from prompts import SYSTEM_PROMPT

CANONICAL_TOOLS = {
    "moveit_get_current_pose",
    "moveit_plan_free_motion",
    "moveit_plan_cartesian_motion",
    "moveit_plan_and_execute_free_motion",
    "moveit_plan_and_execute_cartesian_motion",
    "moveit_execute_plan",
    "moveit_open_gripper",
    "moveit_close_gripper",
    "moveit_attach_object",
}

STALE_TOOLS = {
    "connect_robot",
    "disconnect_robot",
    "get_joints",
    "get_tcp_pose",
    "move_to_position",
    "move_to_pose",
    "move_linear",
    "move_joints",
    "control_gripper",
    "control_gripper_position",
    "moveit_get_robot_status",
    "moveit_plan_linear_motion",
    "moveit_plan_relative_motion",
    "moveit_list_named_poses",
    "moveit_plan_named_pose",
}


def test_prompt_lists_only_canonical_moveit_tools() -> None:
    for tool_name in CANONICAL_TOOLS:
        assert tool_name in SYSTEM_PROMPT

    for tool_name in STALE_TOOLS:
        assert tool_name not in SYSTEM_PROMPT


def test_prompt_requires_observe_plan_execute_verify_for_robot_actions() -> None:
    assert "observe" in SYSTEM_PROMPT.lower()
    assert "plan before" in SYSTEM_PROMPT.lower()
    assert "execute only" in SYSTEM_PROMPT.lower()
    assert "verify" in SYSTEM_PROMPT.lower()


def test_prompt_requires_fresh_pose_for_state_dependent_actions() -> None:
    prompt = SYSTEM_PROMPT.lower()
    assert "moveit_get_current_pose" in prompt
    assert "relative" in prompt
    assert "fresh" in prompt
    assert "last-known context is advisory" in prompt
