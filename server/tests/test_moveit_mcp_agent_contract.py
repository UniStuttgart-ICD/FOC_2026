from pathlib import Path

import pytest

from moveit_mcp.server import build_mcp, build_tools
from moveit_mcp.vizor_client import FakeRosbridgeTransport

FINAL_POSITIONS = [0.0, -1.57, 1.57, 0.0, 0.0, 0.0]

CANONICAL_AGENT_TOOLS = {
    "moveit_get_current_pose",
    "moveit_get_robot_state",
    "moveit_list_scene_objects",
    "moveit_get_object_context",
    "moveit_plan_pick",
    "moveit_plan_place",
    "moveit_plan_free_motion",
    "moveit_plan_cartesian_motion",
    "moveit_execute_plan",
    "moveit_explain_motion_failure",
    "moveit_verify_attached_object",
    "moveit_remove_scene_object",
    "moveit_open_gripper",
    "moveit_close_gripper",
    "moveit_attach_object",
}
TASK_SOLUTION_TOOLS = {
    "moveit_plan_pick_task",
    "moveit_plan_place_task",
    "moveit_plan_compound_task",
    "moveit_plan_manipulation_task",
    "moveit_execute_task_solution",
}

PLANNING_TOOLS = ("moveit_plan_free_motion", "moveit_plan_cartesian_motion")
FORBIDDEN_AGENT_TOOLS = (
    "moveit_plan_and_execute_free_motion",
    "moveit_plan_and_execute_cartesian_motion",
    "moveit_plan_hold",
)


async def _registered_tools():
    tools = build_tools(transport=FakeRosbridgeTransport())
    mcp = build_mcp(tools=tools)
    return {tool.name: tool for tool in await mcp.list_tools()}


@pytest.mark.asyncio
async def test_agent_contract_registers_canonical_moveit_namespaced_tools():
    registered = await _registered_tools()

    assert CANONICAL_AGENT_TOOLS.issubset(registered)
    assert TASK_SOLUTION_TOOLS.issubset(registered)
    assert all(name.startswith("moveit_") for name in CANONICAL_AGENT_TOOLS)
    assert all(name.startswith("moveit_") for name in TASK_SOLUTION_TOOLS)
    assert set(FORBIDDEN_AGENT_TOOLS).isdisjoint(registered)


@pytest.mark.asyncio
async def test_agent_contract_task_solution_tool_descriptions_explain_plan_execute_boundary():
    registered = await _registered_tools()

    pick_description = registered["moveit_plan_pick_task"].description.lower()
    place_description = registered["moveit_plan_place_task"].description.lower()
    compound_description = registered["moveit_plan_compound_task"].description.lower()
    manipulation_description = registered["moveit_plan_manipulation_task"].description.lower()
    execute_description = registered["moveit_execute_task_solution"].description.lower()

    assert "task solution" in pick_description
    assert "does not execute" in pick_description
    assert "stage evidence" in pick_description
    assert "scene snapshot" in pick_description
    assert "mtc" in pick_description
    assert "raw mtc stage authoring" in pick_description
    assert "task solution" in place_description
    assert "does not execute" in place_description
    assert "stage evidence" in place_description
    assert "scene snapshot" in place_description
    assert "compound task" in compound_description
    assert "backend=\"mtc\"" in compound_description
    assert "requirements" in compound_description
    assert "preferences" in compound_description
    assert "stage-intent hints" in compound_description
    assert "execution_contract" in compound_description
    assert "does not execute" in compound_description
    assert "staged moveit" in manipulation_description
    assert "backend=\"staged_moveit\"" in manipulation_description
    assert "no mtc fallback" in manipulation_description
    assert "motion-only" in manipulation_description
    assert "does not release" in manipulation_description
    assert "agentpath" in manipulation_description
    assert "does not execute" in manipulation_description
    assert "execute_task_solution" in execute_description
    assert "task solution" in execute_description
    assert "stage evidence" in execute_description


@pytest.mark.asyncio
async def test_agent_contract_task_solution_execution_default_timeout_is_one_hundred_twenty_seconds():
    registered = await _registered_tools()

    timeout_schema = registered["moveit_execute_task_solution"].inputSchema["properties"]["timeout_s"]

    assert timeout_schema["default"] == 120.0


@pytest.mark.asyncio
async def test_agent_contract_descriptions_teach_safe_planning_workflow():
    registered = await _registered_tools()

    for name in PLANNING_TOOLS:
        description = registered[name].description
        assert "base_link" in description
        assert "raw.plan_name" in description
        assert "feedback.can_execute" in description
        assert "moveit_get_current_pose" in description
        assert "relative" in description
        assert "vague" in description


@pytest.mark.asyncio
async def test_agent_contract_robot_state_description_is_read_only_observation():
    registered = await _registered_tools()

    description = registered["moveit_get_robot_state"].description

    assert "Read UR10 pose" in description
    assert "physical-mode" in description
    assert "fake-controller joint state" in description
    assert "diagnosing motion failures" in description


@pytest.mark.asyncio
async def test_agent_contract_scene_object_descriptions_support_pick_grounding():
    registered = await _registered_tools()

    list_description = registered["moveit_list_scene_objects"].description
    context_description = registered["moveit_get_object_context"].description
    pick_description = registered["moveit_plan_pick"].description

    assert "planning-scene object discovery" in list_description
    assert "object names" in list_description
    assert "attached/free state" in list_description
    assert "read-only" in list_description.lower()
    assert "bounds" in context_description
    assert "grasp-relevant faces" in context_description
    assert "clearance" in context_description
    assert "moveit_list_scene_objects" in context_description
    assert "moveit_get_object_context" in pick_description
    assert "grasp" in pick_description
    assert "lift" in pick_description
    assert "raw.plan_name" in pick_description
    assert "raw.preposition" in pick_description
    assert "raw.workflow_segments" in pick_description
    assert "does not move" in pick_description.lower()


@pytest.mark.asyncio
async def test_agent_contract_execute_description_restricts_execution_to_verified_same_process_plans():
    registered = await _registered_tools()

    description = registered["moveit_execute_plan"].description

    assert "verified plan" in description
    assert "same MCP process" in description
    assert "raw.plan_name" in description
    assert "feedback.can_execute" in description


@pytest.mark.asyncio
async def test_agent_contract_diagnostic_tool_descriptions_teach_retry_and_proof_workflow():
    registered = await _registered_tools()

    failure_description = registered["moveit_explain_motion_failure"].description.lower()
    assert "failed planner or executor result" in failure_description
    assert "retry guidance" in failure_description
    assert "does not plan or execute" in failure_description

    attached_description = registered["moveit_verify_attached_object"].description.lower()
    assert "attached" in attached_description
    assert "moved with the gripper" in attached_description
    assert "does not execute" in attached_description


@pytest.mark.asyncio
async def test_agent_contract_cartesian_descriptions_enable_visible_improvised_gestures():
    registered = await _registered_tools()

    description = registered["moveit_plan_cartesian_motion"].description.lower()
    assert "expressive tcp paths" in description
    assert "waving" in description
    assert "drawing simple shapes" in description
    assert "ordered waypoints" in description
    assert "preserve orientation" in description
    assert "copy raw.pose.orientation" in description
    assert "visible" in description


@pytest.mark.asyncio
async def test_agent_contract_schemas_expose_agent_facing_inputs_not_legacy_ros_names():
    registered = await _registered_tools()

    free_schema = registered["moveit_plan_free_motion"].inputSchema
    cartesian_schema = registered["moveit_plan_cartesian_motion"].inputSchema

    assert set(free_schema["required"]) == {"target_pose"}
    assert "target_pose" in free_schema["properties"]
    assert "position" not in free_schema["properties"]
    assert "allow_existing_name" not in free_schema["properties"]
    assert "raw.plan_name" in free_schema["properties"]["plan_name"]["description"]

    assert set(cartesian_schema["required"]) == {"waypoints"}
    assert "waypoints" in cartesian_schema["properties"]
    assert "positions" not in cartesian_schema["properties"]
    assert "allow_existing_name" not in cartesian_schema["properties"]
    assert "raw.plan_name" in cartesian_schema["properties"]["plan_name"]["description"]

    object_context_schema = registered["moveit_get_object_context"].inputSchema
    assert set(object_context_schema["required"]) == {"object_name"}
    assert "object_name" in object_context_schema["properties"]
    assert "raw ID" in object_context_schema["properties"]["object_name"]["description"]

    pick_schema = registered["moveit_plan_pick"].inputSchema
    assert set(pick_schema["required"]) == {"object_name"}
    assert "object_name" in pick_schema["properties"]
    assert "grasp_face" in pick_schema["properties"]
    strategy = pick_schema["properties"]["planning_strategy"]
    assert set(strategy["enum"]) == {"auto", "cartesian", "sampled_approach"}
    assert strategy["default"] == "auto"
    assert "waypoints" not in pick_schema["properties"]

    failure_schema = registered["moveit_explain_motion_failure"].inputSchema
    assert {"failed_tool_name", "failed_tool_result"}.issubset(failure_schema["required"])
    assert "failed_tool_arguments" in failure_schema["properties"]

    attached_schema = registered["moveit_verify_attached_object"].inputSchema
    assert set(attached_schema["required"]) == {"object_name"}
    assert "object_name" in attached_schema["properties"]

    pick_task_schema = registered["moveit_plan_pick_task"].inputSchema
    assert set(pick_task_schema["required"]) == {"object_name"}
    assert "backend" not in pick_task_schema["properties"]

    compound_schema = registered["moveit_plan_compound_task"].inputSchema
    assert set(compound_schema["required"]) == {"requirements", "backend"}
    assert "object_name" not in compound_schema["properties"]
    assert "task_goal" not in compound_schema["properties"]
    assert compound_schema["properties"]["backend"]["const"] == "mtc"
    assert "goal" in compound_schema["properties"]["requirements"]["description"]
    assert "object_name" in compound_schema["properties"]["requirements"]["description"]
    assert "preferences" in compound_schema["properties"]
    assert "stage_intents" in compound_schema["properties"]

    manipulation_schema = registered["moveit_plan_manipulation_task"].inputSchema
    assert set(manipulation_schema["required"]) == {"requirements", "backend"}
    assert manipulation_schema["properties"]["backend"]["const"] == "staged_moveit"
    assert "goal" in manipulation_schema["properties"]["requirements"]["description"]
    assert "move" in manipulation_schema["properties"]["requirements"]["description"]
    assert "motion" in manipulation_schema["properties"]["requirements"]["description"]
    assert "object_name" in manipulation_schema["properties"]["requirements"]["description"]
    assert "preferences" in manipulation_schema["properties"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "call_args"),
    [
        (
            "moveit_plan_free_motion",
            {"target_pose": {"x": 0.5, "y": 0.0, "z": 0.3}, "plan_name": "contract_free"},
        ),
        (
            "moveit_plan_cartesian_motion",
            {
                "waypoints": [{"x": 0.4, "y": 0.0, "z": 0.3}, {"x": 0.5, "y": 0.0, "z": 0.3}],
                "plan_name": "contract_cartesian",
            },
        ),
    ],
)
async def test_agent_contract_planning_results_expose_execution_gate_fields(tool_name, call_args):
    plan_name = call_args["plan_name"]
    transport = FakeRosbridgeTransport()
    transport.queue_status_after_publish("/UR10/request/status", "success! ")
    transport.queue_planned_path_after_publish(
        "/UR10/request/planned_path",
        name=plan_name,
        points=3,
        final_positions=FINAL_POSITIONS,
    )
    tools = build_tools(transport=transport)
    mcp = build_mcp(tools=tools)

    _, payload = await mcp.call_tool(tool_name, call_args)

    assert payload["ok"] is True
    assert payload["feedback"]["can_execute"] is True
    assert payload["verification"]["result"] == "pass"
    assert payload["raw"]["plan_name"] == plan_name


@pytest.mark.asyncio
async def test_agent_contract_pick_planning_result_exposes_object_and_execution_gate_fields():
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene(
        "UR10",
        {
            "scene": {
                "world": {
                    "collision_objects": [
                        {
                            "id": "beam_001",
                            "header": {"frame_id": "base_link"},
                            "primitives": [{"type": 1, "dimensions": [0.3, 0.04, 0.04]}],
                            "primitive_poses": [
                                {
                                    "position": {"x": 0.4, "y": 0.2, "z": 0.12},
                                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                                }
                            ],
                        }
                    ]
                },
                "robot_state": {"attached_collision_objects": []},
            }
        },
        planning_frame="base_link",
    )
    transport.queue_status_after_publish("/UR10/request/status", "success! ")
    transport.queue_planned_path_after_publish(
        "/UR10/request/planned_path",
        name="contract_pick",
        points=3,
        final_positions=FINAL_POSITIONS,
    )
    tools = build_tools(transport=transport)
    mcp = build_mcp(tools=tools)

    _, payload = await mcp.call_tool(
        "moveit_plan_pick",
        {"object_name": "beam_001", "plan_name": "contract_pick", "planning_strategy": "cartesian"},
    )

    assert payload["ok"] is True
    assert payload["feedback"]["can_execute"] is True
    assert payload["raw"]["plan_name"] == "contract_pick"
    assert payload["raw"]["object_name"] == "beam_001"
    assert payload["raw"]["selected_grasp_face"]["name"] == "top"
    assert payload["raw"]["workflow_steps"][2]["tool"] == "moveit_close_gripper"


@pytest.mark.asyncio
async def test_agent_contract_auto_pick_returns_partial_diagnostic_before_local_pick():
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene(
        "UR10",
        {
            "scene": {
                "world": {
                    "collision_objects": [
                        {
                            "id": "beam_001",
                            "header": {"frame_id": "base_link"},
                            "primitives": [{"type": 1, "dimensions": [0.3, 0.04, 0.04]}],
                            "primitive_poses": [
                                {
                                    "position": {"x": 0.4, "y": 0.2, "z": 0.12},
                                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                                }
                            ],
                        }
                    ]
                },
                "robot_state": {"attached_collision_objects": []},
            }
        },
        planning_frame="base_link",
    )
    transport.queue_status_after_publish("/UR10/request/status", "success! ")
    transport.queue_planned_path_after_publish(
        "/UR10/request/planned_path",
        name="contract_pick__preposition",
        points=3,
        final_positions=FINAL_POSITIONS,
    )
    tools = build_tools(transport=transport)
    mcp = build_mcp(tools=tools)

    _, payload = await mcp.call_tool(
        "moveit_plan_pick",
        {"object_name": "beam_001", "plan_name": "contract_pick", "planning_strategy": "auto"},
    )

    assert payload["ok"] is False
    assert payload["error"] == "pick_segment_planning_failed"
    assert payload["failed_segment"] == "local_cartesian_pick"
    assert payload["feedback"]["can_execute"] is False
    assert payload["suggested_next_tool"] != "moveit_execute_plan"
    assert "plan_name" not in payload["raw"]
    assert payload["raw"]["partial_plan"] == {
        "kind": "preposition",
        "plan_name": "contract_pick__preposition",
    }
    assert payload["raw"]["stage_report"][-1]["name"] == "local_cartesian_pick"
    assert payload["raw"]["stage_report"][-1]["status"] == "failed"


@pytest.mark.asyncio
async def test_agent_contract_execution_failures_include_actionable_correction():
    tools = build_tools(transport=FakeRosbridgeTransport())
    mcp = build_mcp(tools=tools)

    _, payload = await mcp.call_tool("moveit_execute_plan", {"plan_name": "never_planned"})

    assert payload["ok"] is False
    assert "correction" in payload["feedback"]
    assert "Call" in payload["feedback"]["correction"]
    assert "raw.plan_name" in payload["feedback"]["correction"]


@pytest.mark.asyncio
async def test_agent_contract_failure_explainer_returns_retry_guidance():
    tools = build_tools(transport=FakeRosbridgeTransport())
    mcp = build_mcp(tools=tools)

    _, payload = await mcp.call_tool(
        "moveit_explain_motion_failure",
        {
            "failed_tool_name": "moveit_plan_cartesian_motion",
            "failed_tool_result": {"ok": False, "feedback": {"status": "incomplete path"}},
            "failed_tool_arguments": {"waypoints": [{"x": 0.4, "y": 0.0, "z": 0.3}]},
            "user_intent": "wave to me",
        },
    )

    assert payload["ok"] is True
    assert payload["retryable"] is True
    assert payload["suggested_next_tool"] == "moveit_plan_cartesian_motion"
    assert "smaller or safer target" in payload["correction"]


@pytest.mark.asyncio
async def test_agent_contract_attached_object_verifier_proves_object_moves_with_gripper():
    scene = {
        "scene": {
            "robot_state": {
                "attached_collision_objects": [
                    {
                        "link_name": "tool0",
                        "touch_links": ["tool0"],
                        "object": {
                            "id": "beam_001",
                            "header": {"frame_id": "tool0"},
                            "primitives": [{"type": 1, "dimensions": [0.3, 0.04, 0.04]}],
                            "primitive_poses": [
                                {
                                    "position": {"x": 0.0, "y": 0.0, "z": 0.04},
                                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                                }
                            ],
                        },
                    }
                ]
            }
        }
    }
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", scene, planning_frame="base_link")
    tools = build_tools(transport=transport)
    tools.gripper.attach("UR10", "beam_001")
    mcp = build_mcp(tools=tools)

    _, payload = await mcp.call_tool("moveit_verify_attached_object", {"object_name": "beam_001"})

    assert payload["ok"] is True
    assert payload["raw"]["attached_to"] == "tool0"
    assert payload["raw"]["moves_with_gripper"] is True


def test_docs_describe_safe_moveit_agent_workflow_with_canonical_tool_names():
    repo_root = Path(__file__).resolve().parents[2]
    documentation = (repo_root / "docs" / "VIZOR_MOVEIT_MCP.md").read_text(
        encoding="utf-8"
    )

    for tool_name in CANONICAL_AGENT_TOOLS:
        assert tool_name in documentation

    assert "moveit_get_current_pose" in documentation
    assert "moveit_plan_free_motion" in documentation
    assert "moveit_plan_cartesian_motion" in documentation
    assert "moveit_plan_pick" in documentation
    assert "moveit_list_scene_objects" in documentation
    assert "moveit_get_object_context" in documentation
    assert "moveit_execute_plan" in documentation
    assert "moveit_explain_motion_failure" in documentation
    assert "moveit_verify_attached_object" in documentation
    assert "Do not expose or use combined `moveit_plan_and_execute_*` tools" in documentation
    assert "raw.plan_name" in documentation
    assert "feedback.can_execute" in documentation
    assert "same MCP process" in documentation
    assert (repo_root / ".cursor" / "mcp-vizor-moveit.example.json").exists()
