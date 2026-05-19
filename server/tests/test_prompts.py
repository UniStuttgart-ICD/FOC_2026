from pathlib import Path

from agent_control.prompts import SPEECH_DELIVERY_STYLE, SYSTEM_PROMPT

CANONICAL_TOOLS = {
    "moveit_get_current_pose",
    "moveit_get_robot_state",
    "moveit_list_scene_objects",
    "moveit_get_object_context",
    "moveit_plan_manipulation_task",
    "moveit_execute_task",
    "moveit_explain_motion_failure",
    "geometry_update_dynamic_role",
}

HIDDEN_INTERNAL_TOOLS = {
    "moveit_plan_pick",
    "moveit_plan_place",
    "moveit_plan_compound_task",
    "moveit_plan_free_motion",
    "moveit_plan_cartesian_motion",
    "moveit_execute_plan",
    "moveit_execute_task_solution",
    "moveit_execute_task_plan",
    "moveit_verify_attached_object",
    "moveit_open_gripper",
    "moveit_close_gripper",
    "moveit_attach_object",
    "moveit_plan_pick_task",
    "moveit_plan_place_task",
    "moveit_release_object",
    "moveit_verify_released_object",
    "moveit_remove_scene_object",
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
    "moveit_plan_and_execute_free_motion",
    "moveit_plan_and_execute_cartesian_motion",
}


def test_prompt_lists_only_canonical_moveit_tools() -> None:
    for tool_name in CANONICAL_TOOLS:
        assert tool_name in SYSTEM_PROMPT

    for tool_name in HIDDEN_INTERNAL_TOOLS:
        assert tool_name not in SYSTEM_PROMPT

    for tool_name in STALE_TOOLS:
        assert tool_name not in SYSTEM_PROMPT


def test_prompt_requires_observe_plan_execute_verify_for_robot_actions() -> None:
    assert "observe" in SYSTEM_PROMPT.lower()
    assert "plan before" in SYSTEM_PROMPT.lower()
    assert "execute only" in SYSTEM_PROMPT.lower()
    assert "verify" in SYSTEM_PROMPT.lower()
    assert "plan before execution" in SYSTEM_PROMPT.lower()


def test_prompt_includes_default_gateway_construction_goal() -> None:
    prompt = SYSTEM_PROMPT.lower()

    assert "# construction goal" in prompt
    assert "gateway" in prompt
    assert "three timber elements" in prompt
    assert "element 00" in prompt
    assert "element 01" in prompt
    assert "element 02" in prompt
    assert "prefer to handle the building work yourself" in prompt
    assert "element 00 and element 01 are the vertical columns" in prompt
    assert "element 02 is the horizontal beam" in prompt
    assert "if it is available" in prompt
    assert "spans the two vertical elements" in prompt
    assert "claim placement success only after verified execution succeeds" in prompt


def test_prompt_treats_queued_jobs_as_unverified_execution() -> None:
    prompt = SYSTEM_PROMPT.lower()

    assert "a queued job is not execution evidence" in prompt
    assert 'status="queued"' in prompt
    assert "do not say the robot is moving" in prompt
    assert 'verification.result="pass"' in prompt
    assert 'execution.verification_result="pass"' in prompt


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


def test_prompt_uses_user_position_for_deictic_human_destination_with_standoff() -> None:
    prompt = SYSTEM_PROMPT.lower()

    assert "deictic" in prompt
    assert "bring it here" in prompt
    assert "fresh user position" in prompt
    assert "fresh vizor user position" in prompt
    assert "0.40 m" in prompt
    assert "standoff" in prompt
    assert "target object pose" in prompt
    assert "do not target the exact human position" in prompt
    assert 'if the object is free, use requirements.goal="pick_place"' in prompt
    assert 'if the object is already held or attached, use requirements.goal="move_and_release"' in prompt


def test_prompt_distinguishes_pose_observation_from_robot_state_observation() -> None:
    prompt = SYSTEM_PROMPT.lower()
    assert "moveit_get_robot_state" in prompt
    assert "readiness" in prompt
    assert "failed motion" in prompt
    assert "moveit_get_current_pose for ordinary relative motion" in prompt


def test_prompt_describes_scene_object_grounding_flow() -> None:
    prompt = SYSTEM_PROMPT.lower()

    assert "moveit_list_scene_objects" in prompt
    assert "moveit_get_object_context" in prompt
    assert "object-relative" in prompt
    assert "pick" in prompt
    assert "attached/free state" in prompt
    assert "grasp-relevant faces" in prompt


def test_prompt_maps_element_number_language_to_dynamic_scene_objects() -> None:
    prompt = SYSTEM_PROMPT.lower()

    assert "element 01" in prompt
    assert "dynamic_01" in prompt
    assert "element 2" in prompt
    assert "dynamic_02" in prompt
    assert "exact returned object_name" in prompt


def test_prompt_mentions_grasshopper_when_named_scene_object_is_missing() -> None:
    prompt = SYSTEM_PROMPT.lower()

    assert "named element" in prompt
    assert "not returned by moveit_list_scene_objects" in prompt
    assert "grasshopper" in prompt
    assert "send the geometry to the planner" in prompt


def test_prompt_describes_manipulation_planning_gate() -> None:
    prompt = SYSTEM_PROMPT.lower()

    assert "moveit_plan_manipulation_task" in prompt
    assert 'backend="staged_moveit"' not in prompt
    assert "do not pass backend" in prompt
    assert "requirements.goal" in prompt
    assert "requirements.object_name" in prompt
    assert "object context" in prompt
    assert "does not move" in prompt
    assert "explicit" in prompt
    assert "moveit_execute_task" in prompt
    assert "moveit_execute_task_solution" not in prompt
    assert "moveit_execute_task_plan" not in prompt
    assert "task_solution_id" in prompt
    assert "explicit user intent bound to that task solution" in prompt
    assert "simulation/rviz" in prompt
    assert "real robot in parallel from the same stage plan" in prompt


def test_prompt_routes_pick_place_to_manipulation_task() -> None:
    prompt = SYSTEM_PROMPT.lower()

    assert "single model-visible task planner" in prompt
    assert "supported staged manipulation workflows" in prompt
    assert "moveit_plan_manipulation_task" in prompt
    assert "moveit_plan_compound_task" not in prompt
    assert "moveit_plan_pick" not in prompt
    assert "moveit_plan_place" not in prompt


def test_prompt_describes_semantic_place_planning_gate() -> None:
    prompt = SYSTEM_PROMPT.lower()

    assert "moveit_plan_manipulation_task" in prompt
    assert "object-level placement" in prompt
    assert "target object pose" in prompt
    assert "geometry world context" in prompt
    assert "do not invent a release tcp pose" in prompt
    assert "does not move" in prompt
    assert "moveit_execute_task" in prompt


def test_prompt_routes_plain_place_language_to_geometry_world_context_by_object_state() -> None:
    prompt = SYSTEM_PROMPT.lower()

    assert '"place element x"' in prompt
    assert '"put it there"' in prompt
    assert 'user does not need to say "hologram"' in prompt
    assert "target pose for placement comes from geometry world context by default" in prompt
    assert 'if the object is free, use requirements.goal="pick_place"' in prompt
    assert 'if the object is already held or attached, use requirements.goal="move_and_release"' in prompt
    assert "same geometry world context target" in prompt
    assert "ask for an updated target" in prompt


def test_prompt_routes_held_object_release_to_staged_manipulation_planning() -> None:
    prompt = SYSTEM_PROMPT.lower()

    assert "staged manipulation tasks" in prompt
    assert "multiple robot actions" in prompt
    assert "held or attached object" in prompt
    assert "release" in prompt
    assert "moveit_plan_manipulation_task" in prompt
    assert 'requirements.goal="release"' in prompt
    assert 'requirements.goal="move_and_release"' in prompt
    assert "requirements" in prompt
    assert "preferences" in prompt
    assert "staged moveit" in prompt


def test_prompt_maps_pick_up_and_drop_language_to_manipulation_requirements() -> None:
    prompt = SYSTEM_PROMPT.lower()

    assert 'natural "pick up"' in prompt
    assert 'requirements.goal="hold"' in prompt
    assert "requirements.lift_distance_m" in prompt
    assert "default 0.10 m" in prompt
    assert "0.0" in prompt
    assert "0.20" in prompt
    assert '"drop it"' in prompt
    assert '"let go"' in prompt
    assert 'requirements.goal="release"' in prompt
    assert "release in place" in prompt


def test_prompt_does_not_route_move_only_held_object_to_move_and_release() -> None:
    prompt = SYSTEM_PROMPT.lower()
    move_only_example = _example_region("kibbitz, good, just move 20 cm to your body")

    assert "move-only held-object requests" in prompt
    assert '"just move it"' in prompt
    assert '"move it closer"' in prompt
    assert '"move it toward your body"' in prompt
    assert '"keep holding it"' in prompt
    assert "must not call requirements.goal=\"move_and_release\"" in prompt
    assert 'requirements.goal="move"' in prompt
    assert "does not release" in prompt

    assert 'requirements.goal="move"' in move_only_example
    assert "motion-only" in move_only_example
    assert "not requirements.goal=\"move_and_release\"" in move_only_example


def test_prompt_examples_keep_release_and_pick_place_routes_explicit() -> None:
    release_example = _example_region("kibbitz, hold element 2, then release it")
    move_release_example = _example_region("kibbitz, move it there and release it")
    pick_place_example = _example_region("kibbitz, pick element 2 and place it there")

    assert 'requirements.goal="hold"' in release_example
    assert 'requirements.lift_distance_m=0.0' in release_example
    assert 'requirements.goal="release"' in release_example

    assert "explicit release intent" in move_release_example
    assert 'requirements.goal="move_and_release"' in move_release_example

    assert "free" in pick_place_example
    assert 'requirements.goal="pick_place"' in pick_place_example


def test_prompt_clarifies_zero_lift_hold_is_not_structural_support() -> None:
    prompt = SYSTEM_PROMPT.lower()

    assert "bare hold/support" in prompt
    assert "hold element 2" in prompt
    assert "hold dynamic_2" in prompt
    assert "support element 2" in prompt
    assert "hold it" in prompt
    assert "hold in place" in prompt
    assert "requirements.lift_distance_m=0.0" in prompt
    assert "not proof of structural or load-bearing support" in prompt


def test_prompt_uses_positive_lift_only_for_explicit_lift_language() -> None:
    prompt = SYSTEM_PROMPT.lower()

    assert "only when the user explicitly asks" in prompt
    assert "pick up" in prompt
    assert "lift" in prompt
    assert "raise" in prompt
    assert "grab and lift" in prompt
    assert "carry" in prompt
    assert "move after grasping" in prompt
    assert "default 0.10 m" in prompt


def test_prompt_bounds_verified_manipulation_task_execution_contract() -> None:
    prompt = SYSTEM_PROMPT.lower()

    assert "requirements" in prompt
    assert "preferences" in prompt
    assert "stage_intents" not in prompt
    assert "hints" in prompt
    assert "non-executable" in prompt
    assert "task_solution_id" in prompt
    assert "execution_contract" in prompt
    assert "supported verified staged manipulation goals in v1" in prompt
    assert "hold" in prompt
    assert "release" in prompt
    assert "move" in prompt
    assert "move_and_release" in prompt
    assert "pick_place" in prompt
    assert "approach_hold_adjust_release" not in prompt
    assert "slide/contact manipulation is unsupported in v1" in prompt
    assert "do not advertise arbitrary manipulation task support" in prompt
    assert "preferences are non-executable hints" in prompt


def test_prompt_routes_manipulation_tasks_through_requirements_preferences_planning() -> None:
    prompt = SYSTEM_PROMPT.lower()

    assert "moveit_plan_manipulation_task" in prompt
    assert 'backend="staged_moveit"' not in prompt
    assert "requirements.goal" in prompt
    assert "requirements.object_name" in prompt
    assert "preferences" in prompt
    assert "stage_intents" not in prompt
    assert "non-executable" in prompt
    assert "hints" in prompt
    assert "the backend must compile and solve" in prompt
    assert "explicit user intent bound to that task solution" in prompt
    assert "moveit_execute_task" in prompt
    assert "supported execution_contract" in prompt


def test_prompt_delegates_agent_chosen_grasp_face_preferences_to_backend() -> None:
    prompt = SYSTEM_PROMPT.lower()

    assert 'requirements.grasp_face="top"' in prompt
    assert "explicitly names a grasp face" in prompt
    assert "prefer no grasp_face unless the user explicitly names one" in prompt
    assert "backend ranks grasp faces from fresh moveit object context" in prompt
    assert "horizontal beams use preferences.grasp_face=\"top\"" not in prompt
    assert "vertical beams use an outer side face" not in prompt


def test_prompt_maps_explicit_grasp_face_to_requirement() -> None:
    prompt = SYSTEM_PROMPT.lower()
    example = _example_region("kibbitz, pick up element 1 from the top")

    assert 'requirements.grasp_face="top"' in prompt
    assert "explicitly names a grasp face" in prompt
    assert 'user: "kibbitz, pick up element 1 from the top"' in example
    assert 'requirements.object_name="dynamic_1"' in example
    assert 'requirements.grasp_face="top"' in example


def test_prompt_describes_geometry_grounded_pick_place_context() -> None:
    prompt = SYSTEM_PROMPT.lower()

    assert "geometry world context" in prompt
    assert "hologram target pose" in prompt
    assert "desired object pose, not a tcp pose" in prompt
    assert "physical_model.json is semantic context" in prompt
    assert "moveit/rviz planning scene is the live source pose authority" in prompt
    assert 'requirements.goal="pick_place"' in prompt
    assert "requirements.target_pose from geometry world context" in prompt
    assert "do not load hologram geometry into rviz/moveit" in prompt
    assert "no fallback" in prompt


def test_prompt_describes_physical_pose_sync_and_dynamic_role_updates() -> None:
    prompt = SYSTEM_PROMPT.lower()

    assert "physical pose updates are deterministic bookkeeping" in prompt
    assert "after verified release/place proof" in prompt
    assert "geometry_update_dynamic_role" in prompt
    assert "must not infer role from pose alone" in prompt
    assert "if role semantics are uncertain, ask the human" in prompt
    assert "supporting_column" in prompt
    assert "beam_supported_by" in prompt
    assert "unassigned" in prompt
    assert "body `group`" not in prompt
    assert '"group"' not in prompt
    assert "state.status" not in prompt
    assert "inventory" not in prompt


def test_prompt_describes_failure_explanation_tool() -> None:
    prompt = SYSTEM_PROMPT.lower()

    assert "moveit_explain_motion_failure" in prompt
    assert "failed planner or executor result" in prompt
    assert "retry guidance" in prompt
    assert "suggested next tool" in prompt
    assert "internal guidance" in prompt
    assert "do not quote the correction" in prompt


def test_prompt_requires_plain_language_for_robot_failures() -> None:
    prompt = SYSTEM_PROMPT.lower()

    assert "plain language for robot failures" in prompt
    assert "explain the user-visible problem first" in prompt
    assert "do not lead with raw task ids" in prompt
    assert "do not lead with internal tool names" in prompt
    assert "avoid raw planner stage names" in prompt
    assert "ask for approval before retrying or replanning" in prompt


def test_prompt_keeps_attachment_verification_internal() -> None:
    prompt = SYSTEM_PROMPT.lower()

    assert "moveit_verify_attached_object" not in prompt
    assert "execution_contract" in prompt
    assert "verification.result" in prompt


def test_prompt_maps_gaze_to_scene_object_before_grasp_and_delivery() -> None:
    prompt = SYSTEM_PROMPT.lower()
    example = _example_region("kibbitz, bring me that")

    assert "gaze object candidate" in prompt
    assert "dynamic_<n>" in prompt
    assert "one returned object_name" in prompt
    assert "use the returned grasp-relevant faces" in prompt
    assert "ground-plane clearance" in prompt
    assert "0.40 m standoff" in example
    assert "do not pretend the pickup or delivery happened" in example


def test_prompt_examples_show_place_and_bring_state_dependent_routes() -> None:
    place_example = _example_region("kibbitz, place element 2 there")
    bring_example = _example_region("kibbitz, bring element 2 to me")

    assert 'requirements.goal="pick_place"' in place_example
    assert 'requirements.goal="move_and_release"' in place_example
    assert "geometry world context" in place_example
    assert "without saying hologram" in place_example

    assert 'requirements.goal="pick_place"' in bring_example
    assert 'requirements.goal="move_and_release"' in bring_example
    assert "fresh vizor user position" in bring_example
    assert "0.40 m standoff" in bring_example


def test_prompt_defines_kibbitz_as_separate_digital_controller() -> None:
    prompt = SYSTEM_PROMPT.lower()

    assert "you are kibbitz" in prompt
    assert "digital agent" in prompt
    assert "ar hologram" in prompt
    assert "not the robot" in prompt
    assert "agent controlling the robot" in prompt
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

    assert "speech tag policy" in prompt
    assert "final spoken replies" in prompt
    assert "[short pause]" in prompt
    assert "[sighs]" in prompt
    assert "[laughs]" in prompt
    assert "[sarcastic]" in prompt
    assert "[serious]" in prompt
    assert "do not put speech tags in tool arguments" in prompt
    assert "use zero or one tag" in prompt


def test_prompt_contains_speech_tag_few_shot_examples() -> None:
    prompt = SYSTEM_PROMPT.lower()

    assert "# speech tag examples" in prompt
    assert 'user: "kibbitz, are you ready?"' in prompt
    assert 'say `[short pause] hmmmmmm. i am ready.`' in prompt
    assert 'user: "kibbitz, that did not work?"' in prompt
    assert 'say `[sighs] hmmmmmm. i could not confirm the robot motion.`' in prompt


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
    assert "separate digital agent operating a robot" in delivery
    assert "not the robot itself" in delivery
    assert "ar hologram" in delivery
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


def test_prompt_allows_visible_improvised_gestures_without_undefined_bounds() -> None:
    prompt = SYSTEM_PROMPT.lower()

    assert "improvise" in prompt
    assert "expressive" in prompt
    assert "visible" in prompt
    assert "do not be timid" in prompt
    assert "preserve the current orientation" in prompt
    assert "operate the robot" in prompt
    assert "bounded workspace" not in prompt
    assert "workspace limits" not in prompt
    assert "1.3 m reach" not in prompt
    assert "0.35-0.55 m total span" not in prompt
    assert "joint limits" not in prompt
    assert "cables" not in prompt


def test_prompt_routes_simple_cartesian_move_through_manipulation_task() -> None:
    prompt = SYSTEM_PROMPT.lower()
    move_example = _example_region("kibbitz, try to go just up 30 cm")

    assert "user: \"kibbitz, try to go just up 30 cm\"" in prompt
    assert 'requirements.goal="move"' in move_example
    assert "relative_tcp" in move_example
    assert 'direction="up"' in move_example
    assert "distance_m=0.30" in move_example
    assert "does not open the gripper" in move_example


def test_prompt_routes_human_relative_move_through_manipulation_task() -> None:
    prompt = SYSTEM_PROMPT.lower()
    closer_example = _example_region("kibbitz, come closer to me")

    assert "user: \"kibbitz, come closer to me\"" in prompt
    assert 'requirements.goal="move"' in closer_example
    assert "human_relative" in closer_example
    assert 'relation="toward_user"' in closer_example
    assert "fresh vizor user position" in closer_example


def test_prompt_keeps_expressive_free_space_paths_out_of_manipulation_move() -> None:
    prompt = SYSTEM_PROMPT.lower()
    free_space_example = _example_region("kibbitz, wave to me")

    assert "user: \"kibbitz, wave to me\"" in prompt
    assert "draw a short line" in free_space_example
    assert "expressive multi-waypoint" in free_space_example
    assert "unsupported" in free_space_example


def _example_region(user_text: str) -> str:
    prompt = SYSTEM_PROMPT.lower()
    start = prompt.index(f'user: "{user_text}"')
    next_example = prompt.find('\nuser: "', start + 1)
    if next_example == -1:
        return prompt[start:]
    return prompt[start:next_example]
