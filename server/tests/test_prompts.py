from pathlib import Path

from agent_control.prompts import SPEECH_DELIVERY_STYLE, SYSTEM_PROMPT

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


def test_prompt_allows_optional_user_sensing_context() -> None:
    prompt = SYSTEM_PROMPT.lower()

    assert "there is no hololens" not in prompt
    assert "user sensing" in prompt
    assert "gaze" in prompt
    assert "stale" in prompt


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


def test_prompt_defines_kibbitz_as_separate_robot_controller() -> None:
    prompt = SYSTEM_PROMPT.lower()

    assert "you are kibbitz" in prompt
    assert "digital agent" in prompt
    assert "ar hologram" in prompt
    assert "plane between the digital and physical" in prompt
    assert "entity of his own" in prompt
    assert "not the robot" in prompt
    assert "control the ur10" in prompt
    assert "robot arm is your body" not in prompt
    assert "users are speaking to the robot itself" not in prompt
    assert "tcp" in prompt


def test_prompt_includes_reasoning_agent_persona_without_tts_delivery_rules() -> None:
    prompt = SYSTEM_PROMPT.lower()

    assert "reasoning agent persona" in prompt
    assert "ancient" in prompt
    assert "dryly erudite" in prompt
    assert "unsolicited advice" in prompt
    assert "japanese elder-scholar cadence" in prompt
    assert "fictional goblin rasp" in prompt
    assert "do not imitate a real accent" in prompt
    assert "do not use broken english" in prompt
    assert "hmmmmmm" in prompt
    assert "robot contract wins" in prompt
    assert "speak the transcript exactly" not in prompt


def test_prompt_allows_sparse_creative_speech_tags_in_final_responses() -> None:
    prompt = SYSTEM_PROMPT.lower()

    assert "creative speech tags" in prompt
    assert "final assistant speech only" in prompt
    assert "[short pause]" in prompt
    assert "[sigh]" in prompt
    assert "[laughing]" in prompt
    assert "[sarcasm]" in prompt
    assert "[robotic]" in prompt
    assert "do not put speech tags in tool arguments" in prompt
    assert "avoid adjective emotion tags" in prompt


def test_prompt_contains_speech_tag_few_shot_examples() -> None:
    prompt = SYSTEM_PROMPT.lower()

    assert "# speech tag examples" in prompt
    assert 'user: "kibbitz, are you ready?"' in prompt
    assert 'say `[short pause] hmmmmmm. i am ready.`' in prompt
    assert 'user: "kibbitz, that did not work?"' in prompt
    assert 'say `[sigh] hmmmmmm. i could not confirm the robot motion.`' in prompt


def test_speech_delivery_style_is_separate_from_reasoning_prompt() -> None:
    delivery = SPEECH_DELIVERY_STYLE.lower()

    assert "speech delivery style" in delivery
    assert "speak the transcript exactly" in delivery
    assert "clear, articulate, and steady" in delivery
    assert "brief, purposeful pauses" in delivery
    assert "dryly erudite" in delivery
    assert "japanese elder-scholar cadence" in delivery
    assert "fictional goblin rasp" in delivery
    assert "do not imitate a real accent" in delivery
    assert "separate digital agent" in delivery
    assert "do not add, remove, summarize, or rephrase words" in delivery


def test_prompt_source_comments_are_not_in_runtime_prompts() -> None:
    prompt_parts_dir = Path(__file__).parents[1] / "agent_control" / "prompt_parts"
    raw_delivery = (prompt_parts_dir / "speech_delivery_style.md").read_text(encoding="utf-8")
    guide_url = "https://aistudio.google.com/learn/gemini-tts-prompt-guide-with-tags"

    assert f"<!-- Reference: {guide_url} -->" in raw_delivery
    assert guide_url not in SYSTEM_PROMPT
    assert guide_url not in SPEECH_DELIVERY_STYLE
    assert "<!--" not in SYSTEM_PROMPT
    assert "<!--" not in SPEECH_DELIVERY_STYLE


def test_prompt_allows_visible_bounded_improvised_gestures() -> None:
    prompt = SYSTEM_PROMPT.lower()

    assert "improvise" in prompt
    assert "expressive" in prompt
    assert "visible" in prompt
    assert "do not be timid" in prompt
    assert "preserve the current orientation" in prompt
    assert "operate the robot" in prompt


def test_prompt_contains_move_up_example_matching_default_magnitude() -> None:
    example = _example_region("kibbitz, move up")

    assert "z=0.62" in example
    assert "z=0.82" in example
    assert "moved up 200 mm" in example


def test_prompt_contains_wave_and_shape_examples_with_human_scale_motion() -> None:
    prompt = SYSTEM_PROMPT.lower()
    wave_example = _example_region("kibbitz, wave to me")

    assert "user: \"kibbitz, wave to me\"" in prompt
    assert "moveit_plan_and_execute_cartesian_motion" in wave_example
    assert "0.20" in wave_example
    assert "0.15" in wave_example
    assert "40 cm side-to-side" in wave_example
    assert "user: \"kibbitz, draw a short line\"" in prompt
    assert "user: \"kibbitz, draw a small circle\"" in prompt
    assert "0.35-0.55 m total span" in prompt
    assert "1.3 m reach" in prompt


def _example_region(user_text: str) -> str:
    prompt = SYSTEM_PROMPT.lower()
    start = prompt.index(f'user: "{user_text}"')
    next_example = prompt.find('\nuser: "', start + 1)
    if next_example == -1:
        return prompt[start:]
    return prompt[start:next_example]
