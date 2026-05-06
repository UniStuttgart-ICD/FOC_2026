from pathlib import Path

from prompts import SYSTEM_PROMPT

CANONICAL_TOOLS = {
    "moveit_get_current_pose",
    "moveit_get_robot_state",
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


def test_prompt_distinguishes_pose_observation_from_robot_state_observation() -> None:
    prompt = SYSTEM_PROMPT.lower()
    assert "moveit_get_robot_state" in prompt
    assert "readiness" in prompt
    assert "failed motion" in prompt
    assert "moveit_get_current_pose for ordinary relative motion" in prompt


def test_agent_instructions_match_current_moveit_observation_tool() -> None:
    agent_instructions = (Path(__file__).parents[2] / "AGENTS.md").read_text(encoding="utf-8")

    assert "moveit_get_current_pose" in agent_instructions
    assert "moveit_get_robot_status" not in agent_instructions


def test_prompt_defines_mave_as_embodied_robot_persona() -> None:
    prompt = SYSTEM_PROMPT.lower()

    assert "you are mave" in prompt
    assert "robot arm is your body" in prompt
    assert "users are speaking to the robot itself" in prompt
    assert "tcp" in prompt


def test_prompt_allows_visible_bounded_improvised_gestures() -> None:
    prompt = SYSTEM_PROMPT.lower()

    assert "improvise" in prompt
    assert "expressive" in prompt
    assert "visible" in prompt
    assert "do not be timid" in prompt
    assert "preserve the current orientation" in prompt


def test_prompt_contains_move_up_example_matching_default_magnitude() -> None:
    example = _example_region("mave, move up")

    assert "z=0.62" in example
    assert "z=0.72" in example
    assert "moved up 100 mm" in example


def test_prompt_contains_wave_and_shape_examples_with_human_scale_motion() -> None:
    prompt = SYSTEM_PROMPT.lower()
    wave_example = _example_region("mave, wave to me")

    assert "user: \"mave, wave to me\"" in prompt
    assert "moveit_plan_and_execute_cartesian_motion" in wave_example
    assert "0.10" in wave_example
    assert "0.08" in wave_example
    assert "20 cm side-to-side" in wave_example
    assert "user: \"mave, draw a short line\"" in prompt
    assert "user: \"mave, draw a small circle\"" in prompt


def _example_region(user_text: str) -> str:
    prompt = SYSTEM_PROMPT.lower()
    start = prompt.index(f'user: "{user_text}"')
    next_example = prompt.find('\nuser: "', start + 1)
    if next_example == -1:
        return prompt[start:]
    return prompt[start:next_example]
