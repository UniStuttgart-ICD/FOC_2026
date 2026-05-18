from __future__ import annotations

from typing import Annotated, Any, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from moveit_mcp.tools import MoveItMcpTools
from moveit_mcp.vizor_client import RosbridgeTransport, RoslibpyTransport

RobotName = Annotated[
    Literal["UR10"],
    Field(description="Robot namespace to control. Only UR10 is supported by this MoveIt MCP server."),
]
TargetPose = Annotated[
    dict[str, Any],
    Field(
        description=(
            "Target end-effector pose in the base_link planning frame. Use either "
            "{x, y, z} for position-only targets or {position: {x, y, z}, "
            "orientation: {x, y, z, w}} with a normalized quaternion."
        )
    ),
]
TargetPosition = Annotated[
    dict[str, Any],
    Field(description="Target object center position in base_link as {x, y, z}."),
]
Waypoints = Annotated[
    list[dict[str, Any]],
    Field(
        description=(
            "Ordered end-effector waypoints in the base_link planning frame. Each waypoint "
            "uses the same pose format as target_pose."
        )
    ),
]
PlanName = Annotated[
    str | None,
    Field(
        description=(
            "Optional caller label for the plan. Agents should normally omit this and use "
            "the returned raw.plan_name when executing."
        )
    ),
]
TimeoutSeconds = Annotated[
    float,
    Field(description="Seconds to wait for Vizor/MoveIt feedback before returning failure."),
]
ObjectName = Annotated[
    str,
    Field(description="Name of the free planning-scene object to attach to the gripper."),
]
VerifiedGripperClosed = Annotated[
    bool,
    Field(
        description=(
            "True only when an external verified real-robot gripper close already succeeded "
            "and this call should only synchronize MCP/MoveIt attachment state."
        )
    ),
]
VerifiedGripperOpen = Annotated[
    bool,
    Field(
        description=(
            "True only when an external verified real-robot gripper open already succeeded "
            "and this call should only synchronize MCP/MoveIt release state."
        )
    ),
]
SceneObjectName = Annotated[
    str,
    Field(description="Planning-scene object raw ID from moveit_list_scene_objects raw.objects[].name."),
]
GraspFace = Annotated[
    str,
    Field(description="Optional grasp face name from moveit_get_object_context raw.object.grasp_faces[].name."),
]
PickDistanceMeters = Annotated[
    float,
    Field(description="Positive distance in meters used to derive the pick approach, grasp standoff, or lift waypoint."),
]
PlaceDistanceMeters = Annotated[
    float,
    Field(description="Positive distance in meters used to derive the place approach, release standoff, or retreat waypoint."),
]
PlanningStrategy = Annotated[
    Literal["auto", "cartesian", "sampled_approach"],
    Field(
        description=(
            "Pick planner selection. Use auto by default, cartesian for one-shot waypoint planning, "
            "or sampled_approach when a sampled pick backend is available."
        )
    ),
]
PlaceOrientationMode = Annotated[
    Literal["keep", "horizontal", "vertical", "explicit"],
    Field(
        description=(
            "Place TCP orientation policy. keep preserves the current gripper orientation, "
            "horizontal uses a downward grasp orientation, vertical uses identity, and "
            "explicit requires target_pose.orientation."
        )
    ),
]
PlaceFace = Annotated[
    str | None,
    Field(description="Optional semantic place/support face name for agent traceability."),
]
CompoundTaskRequirements = Annotated[
    dict[str, Any],
    Field(
        description=(
            "Hard compound task requirements. Must include goal and object_name. "
            "Allowed goals are hold, release, move_and_release, and pick_place. "
            "For move_and_release and pick_place, "
            "include target_pose or target_position inside requirements."
        )
    ),
]
CompoundTaskPreferences = Annotated[
    dict[str, Any] | None,
    Field(
        description=(
            "Optional non-executable planner preferences such as grasp_face, orientation_mode, "
            "approach_distance_m, retreat_distance_m, or release behavior."
        )
    ),
]
CompoundStageIntents = Annotated[
    list[
        Literal[
            "observe_current_state",
            "approach_object",
            "close_gripper",
            "verify_attached",
            "lift",
            "move_to_pose",
            "adjust_pose",
            "open_gripper",
            "release_object",
            "verify_released",
        ]
    ],
    Field(description="Optional stage-intent hints. Slide, push, raw code, script, and raw waypoint hints are rejected."),
]
CompoundBackend = Annotated[
    Literal["mtc"],
    Field(description='Required backend for compound task planning. Must be "mtc"; no fallback backend exists.'),
]
ManipulationTaskRequirements = Annotated[
    dict[str, Any],
    Field(
        description=(
            "Hard staged manipulation requirements. Must include goal and object_name. "
            "Supported goals are hold, place, release, move_and_release, and pick_place. "
            "Target pose or position is required for place, move_and_release, and pick_place."
        )
    ),
]
ManipulationTaskPreferences = Annotated[
    dict[str, Any] | None,
    Field(description="Optional staged MoveIt preferences such as grasp_face and pick distances."),
]
ManipulationBackend = Annotated[
    Literal["staged_moveit"],
    Field(description='Required backend for staged manipulation planning. Must be "staged_moveit"; no MTC fallback exists.'),
]
TaskSolutionId = Annotated[
    str,
    Field(description="Exact raw.task_solution_id returned by a task-solution planning tool."),
]
FailedToolName = Annotated[
    str,
    Field(description="Exact MoveIt tool name that returned the failed planner, executor, or verification result."),
]
FailedToolArguments = Annotated[
    dict[str, Any] | None,
    Field(description="Original arguments sent to the failed tool, when available."),
]
FailedToolResult = Annotated[
    dict[str, Any] | str,
    Field(description="Failed tool output as the returned structured result object or compact text."),
]
UserIntent = Annotated[
    str | None,
    Field(description="User request that motivated the failed motion, when available."),
]


def build_tools(
    *,
    transport: RosbridgeTransport | None = None,
    host: str = "localhost",
    port: int = 9090,
    pick_task_backend: Literal["emulated", "mtc"] = "emulated",
) -> MoveItMcpTools:
    if transport is not None:
        return MoveItMcpTools.with_transport(transport, pick_task_backend=pick_task_backend)

    real_transport = RoslibpyTransport(host=host, port=port)
    real_transport.connect()
    return MoveItMcpTools.with_transport(real_transport, pick_task_backend=pick_task_backend)


def build_mcp(*, tools: MoveItMcpTools, host: str = "127.0.0.1", port: int = 8000) -> FastMCP:
    mcp = FastMCP("VizorMoveItServer", host=host, port=port)

    @mcp.tool()
    def moveit_get_current_pose(robot_name: RobotName = "UR10", timeout_s: TimeoutSeconds = 2.0) -> dict[str, Any]:
        """Read UR10's current end-effector pose in the base_link planning frame.

        Use this before relative or gesture-based moves so target poses can be grounded in
        the current robot state. Returns a verification envelope with raw.pose and
        raw.planning_frame.
        """
        return tools.get_current_pose(robot_name, timeout_s=timeout_s)

    @mcp.tool()
    def moveit_get_robot_state(robot_name: RobotName = "UR10", timeout_s: TimeoutSeconds = 2.0) -> dict[str, Any]:
        """Read UR10 pose, planning frame, physical-mode flag, and latest fake-controller joint state.

        Use this when the agent needs broader observation than the current TCP pose,
        especially before diagnosing motion failures or explaining whether the
        simulation is ready. This is read-only and does not plan, execute, or publish
        motion commands.
        """
        return tools.get_robot_state(robot_name, timeout_s=timeout_s)

    @mcp.tool()
    def moveit_list_scene_objects(robot_name: RobotName = "UR10", timeout_s: TimeoutSeconds = 2.0) -> dict[str, Any]:
        """Read-only planning-scene object discovery for UR10.

        Use this before pick or object-relative tasks to discover object names,
        frames, poses, primitive or mesh summaries, bounds, colors when available,
        and attached/free state. This tool only observes the MoveIt planning scene;
        it does not plan, execute, attach, or publish motion commands.
        """
        return tools.list_scene_objects(robot_name, timeout_s=timeout_s)

    @mcp.tool()
    def moveit_get_object_context(
        object_name: SceneObjectName,
        robot_name: RobotName = "UR10",
        timeout_s: TimeoutSeconds = 2.0,
    ) -> dict[str, Any]:
        """Read one planning-scene object's context for grounded pick reasoning.

        Call moveit_list_scene_objects first, then pass one returned raw ID here.
        Returns the object's pose, bounds, shape summaries, grasp-relevant faces,
        clearance above the ground plane when available, planning frame, and
        attached/free state. This is read-only and does not plan or execute.
        """
        return tools.get_object_context(robot_name, object_name, timeout_s=timeout_s)

    @mcp.tool()
    def moveit_plan_pick(
        object_name: SceneObjectName,
        robot_name: RobotName = "UR10",
        plan_name: PlanName = None,
        grasp_face: GraspFace = "top",
        approach_distance_m: PickDistanceMeters = 0.08,
        grasp_standoff_m: PickDistanceMeters = 0.01,
        lift_distance_m: PickDistanceMeters = 0.1,
        planning_strategy: PlanningStrategy = "auto",
        timeout_s: TimeoutSeconds = 10.0,
    ) -> dict[str, Any]:
        """Plan a grounded pick workflow for one planning-scene object; does not move the robot.

        Call moveit_list_scene_objects, then moveit_get_object_context, then pass one
        returned object_name here. The server derives grasp approach, pre-grasp,
        face-aware gripper orientation, close-gripper, attach, and lift steps from
        the selected grasp face. The default planning_strategy="auto" first plans a
        free-motion raw.preposition and reports raw.workflow_segments plus the next
        local Cartesian pick action. Execute only the returned raw.plan_name, and
        only when ok=true and feedback.can_execute=true.
        """
        return tools.plan_pick(
            robot_name,
            object_name,
            plan_name=plan_name,
            grasp_face=grasp_face,
            approach_distance_m=approach_distance_m,
            grasp_standoff_m=grasp_standoff_m,
            lift_distance_m=lift_distance_m,
            planning_strategy=planning_strategy,
            timeout_s=timeout_s,
            allow_existing_name=False,
        )

    @mcp.tool()
    def moveit_plan_pick_task(
        object_name: SceneObjectName,
        robot_name: RobotName = "UR10",
        grasp_face: GraspFace | None = None,
        timeout_s: TimeoutSeconds = 10.0,
    ) -> dict[str, Any]:
        """Plan a backend-selected MTC-shaped pick task solution; does not execute or move the robot.

        Returns raw.task_solution_id, scene snapshot, ordered stage evidence, and
        approval metadata for execute_task_solution. The default backend is emulated;
        a configured MTC backend returns sanitized evidence without raw MTC stage authoring.
        Use this task solution only after explicit execution approval.
        """
        return tools.plan_pick_task(
            robot_name,
            object_name,
            grasp_face=grasp_face,
            timeout_s=timeout_s,
        )

    @mcp.tool()
    def moveit_plan_place_task(
        object_name: SceneObjectName,
        robot_name: RobotName = "UR10",
        target_pose: TargetPose | None = None,
        target_position: TargetPosition | None = None,
        orientation_mode: PlaceOrientationMode = "keep",
        timeout_s: TimeoutSeconds = 10.0,
    ) -> dict[str, Any]:
        """Plan an emulated MTC-shaped place task solution; does not execute or move the robot.

        Returns raw.task_solution_id, scene snapshot, ordered stage evidence, and
        approval metadata for execute_task_solution. Use this task solution only
        after explicit execution approval.
        """
        return tools.plan_place_task(
            robot_name,
            object_name,
            target_pose=target_pose,
            target_position=target_position,
            orientation_mode=orientation_mode,
            timeout_s=timeout_s,
        )

    @mcp.tool()
    def moveit_plan_compound_task(
        requirements: CompoundTaskRequirements,
        backend: CompoundBackend,
        robot_name: RobotName = "UR10",
        preferences: CompoundTaskPreferences = None,
        stage_intents: CompoundStageIntents | None = None,
        timeout_s: TimeoutSeconds = 10.0,
    ) -> dict[str, Any]:
        """Plan an MTC compound task solution with raw.execution_contract; does not execute.

        The LLM supplies hard requirements and optional preferences or stage-intent hints;
        MTC compiles and solves the executable task graph. Requires backend="mtc"
        and fails closed when MTC is unavailable or incomplete. Returns a task
        solution with ordered typed execution_contract steps, stage evidence,
        candidate attempts, selected cost, and scene snapshot metadata.
        """
        return tools.plan_compound_task(
            robot_name,
            requirements=requirements,
            preferences=preferences,
            stage_intents=stage_intents,
            backend=backend,
            timeout_s=timeout_s,
        )

    @mcp.tool()
    def moveit_plan_manipulation_task(
        requirements: ManipulationTaskRequirements,
        backend: ManipulationBackend,
        robot_name: RobotName = "UR10",
        preferences: ManipulationTaskPreferences = None,
        timeout_s: TimeoutSeconds = 10.0,
    ) -> dict[str, Any]:
        """Plan a staged MoveIt manipulation task solution with raw.execution_contract; does not execute.

        Requires backend="staged_moveit"; no MTC fallback exists. For hold tasks,
        searches grasp candidates, proves required motion stages with non-empty
        trajectory preview evidence, and returns an approval payload plus AgentPath
        preview only when every required stage is planned.
        """
        return tools.plan_manipulation_task(
            robot_name,
            requirements=requirements,
            preferences=preferences,
            backend=backend,
            timeout_s=timeout_s,
        )

    @mcp.tool()
    def moveit_plan_place(
        object_name: SceneObjectName,
        robot_name: RobotName = "UR10",
        plan_name: PlanName = None,
        target_pose: TargetPose | None = None,
        target_position: TargetPosition | None = None,
        orientation_mode: PlaceOrientationMode = "keep",
        place_face: PlaceFace = None,
        support_face: PlaceFace = None,
        approach_distance_m: PlaceDistanceMeters = 0.08,
        place_standoff_m: PlaceDistanceMeters = 0.01,
        retreat_distance_m: PlaceDistanceMeters = 0.1,
        timeout_s: TimeoutSeconds = 10.0,
    ) -> dict[str, Any]:
        """Plan a grounded place workflow for one attached planning-scene object; does not move the robot.

        Call after a verified pick/attach. Pass either target_position for the target
        object pose center or target_pose for the target object pose when an explicit
        object orientation is needed.
        The server derives a release TCP pose, approach, open-gripper, detach, and
        retreat steps. Execute only the returned raw.plan_name, and only when ok=true
        and feedback.can_execute=true.
        """
        return tools.plan_place(
            robot_name,
            object_name,
            plan_name=plan_name,
            target_pose=target_pose,
            target_position=target_position,
            orientation_mode=orientation_mode,
            place_face=place_face,
            support_face=support_face,
            approach_distance_m=approach_distance_m,
            place_standoff_m=place_standoff_m,
            retreat_distance_m=retreat_distance_m,
            timeout_s=timeout_s,
            allow_existing_name=False,
        )

    @mcp.tool()
    def moveit_plan_free_motion(
        target_pose: TargetPose,
        robot_name: RobotName = "UR10",
        plan_name: PlanName = None,
        timeout_s: TimeoutSeconds = 10.0,
    ) -> dict[str, Any]:
        """Plan a collision-aware free-space motion to one target pose in base_link.

        Plan first; do not move the robot with this tool. For relative or vague
        motion, call moveit_get_current_pose first to ground the target. Execute only
        the returned raw.plan_name, and only when ok=true and feedback.can_execute=true.
        Omit plan_name unless the user explicitly needs a label.
        """
        return tools.plan_free_motion(
            robot_name,
            plan_name,
            target_pose,
            timeout_s=timeout_s,
            allow_existing_name=False,
        )

    @mcp.tool()
    def moveit_plan_cartesian_motion(
        waypoints: Waypoints,
        robot_name: RobotName = "UR10",
        plan_name: PlanName = None,
        timeout_s: TimeoutSeconds = 10.0,
    ) -> dict[str, Any]:
        """Plan a Cartesian path through ordered waypoints in base_link.

        Use this for straight-line motion, waypoint-following motion, expressive TCP paths,
        visible waving, tracing, drawing simple shapes, sweeping, or other
        multi-point gestures. Plan first; do not move the robot with this tool. For
        relative, vague, or gesture-based motion, call moveit_get_current_pose first
        to ground waypoints from the current TCP pose.
        To preserve orientation, copy raw.pose.orientation from moveit_get_current_pose into every waypoint.
        Execute only the returned raw.plan_name, and only when ok=true and
        feedback.can_execute=true. Omit plan_name unless needed.
        """
        return tools.plan_cartesian_motion(
            robot_name,
            plan_name,
            waypoints,
            timeout_s=timeout_s,
            allow_existing_name=False,
        )

    @mcp.tool()
    def moveit_execute_plan(
        plan_name: Annotated[str, Field(description="Exact raw.plan_name returned by a successful planning tool call.")],
        robot_name: RobotName = "UR10",
        timeout_s: TimeoutSeconds = 10.0,
    ) -> dict[str, Any]:
        """Execute only a verified plan from the same MCP process.

        Only execute a verified plan from the same MCP process. Call this only after
        moveit_plan_free_motion or moveit_plan_cartesian_motion returned ok=true and
        feedback.can_execute=true, using that response's raw.plan_name exactly.
        """
        return tools.execute_plan(robot_name, plan_name, timeout_s=timeout_s)

    @mcp.tool()
    def moveit_execute_task_solution(
        task_solution_id: TaskSolutionId,
        robot_name: RobotName = "UR10",
        timeout_s: TimeoutSeconds = 60.0,
    ) -> dict[str, Any]:
        """Run execute_task_solution for a stored task solution and return stage evidence.

        Executes only an in-memory task solution from the same MCP process. Stop on
        the first failed stage and use raw.stage_report plus raw.stages as proof.
        """
        return tools.execute_task_solution(robot_name, task_solution_id, timeout_s=timeout_s)

    @mcp.tool()
    def moveit_explain_motion_failure(
        failed_tool_name: FailedToolName,
        failed_tool_result: FailedToolResult,
        failed_tool_arguments: FailedToolArguments = None,
        user_intent: UserIntent = None,
        robot_name: RobotName = "UR10",
        timeout_s: TimeoutSeconds = 2.0,
    ) -> dict[str, Any]:
        """Explain one failed planner or executor result and return retry guidance.

        Use this after a MoveIt planner, executor, or verification tool returns ok=false
        or verification fails. Pass the failed tool name, its original arguments when
        available, and the failed result. The tool classifies the failure, returns a
        retryable flag, correction, and suggested next tool. It does not plan or execute.
        """
        return tools.explain_motion_failure(
            robot_name,
            failed_tool_name,
            failed_tool_result,
            failed_tool_arguments=failed_tool_arguments,
            user_intent=user_intent,
            timeout_s=timeout_s,
        )

    @mcp.tool()
    def moveit_verify_attached_object(
        object_name: SceneObjectName,
        robot_name: RobotName = "UR10",
        timeout_s: TimeoutSeconds = 2.0,
    ) -> dict[str, Any]:
        """Verify that one planning-scene object is attached and moved with the gripper.

        Use after executing a pick/place plan or attach workflow before claiming the
        object was picked up, placed, or moved with the gripper. Checks MCP gripper
        state and MoveIt planning-scene attachment state. This tool does not execute,
        attach, detach, or move the robot.
        """
        return tools.verify_attached_object(robot_name, object_name, timeout_s=timeout_s)

    @mcp.tool()
    def moveit_release_object(
        object_name: SceneObjectName,
        object_pose: TargetPose,
        robot_name: RobotName = "UR10",
        verified_gripper_open: VerifiedGripperOpen = False,
        timeout_s: TimeoutSeconds = 2.0,
    ) -> dict[str, Any]:
        """Detach a held planning-scene object only after verified gripper open evidence.

        Use from a backend-issued task execution contract after Verified Real Robot
        Execution has opened the physical gripper. This does not command the gripper.
        It only synchronizes the MoveIt planning scene and returns release evidence.
        """
        return tools.release_object(
            robot_name,
            object_name,
            object_pose=object_pose,
            verified_gripper_open=verified_gripper_open,
            timeout_s=timeout_s,
        )

    @mcp.tool()
    def moveit_verify_released_object(
        object_name: SceneObjectName,
        robot_name: RobotName = "UR10",
        timeout_s: TimeoutSeconds = 2.0,
    ) -> dict[str, Any]:
        """Verify that one planning-scene object is free and no longer held by the gripper."""
        return tools.verify_released_object(robot_name, object_name, timeout_s=timeout_s)

    @mcp.tool()
    def moveit_remove_scene_object(
        object_name: SceneObjectName,
        robot_name: RobotName = "UR10",
        timeout_s: TimeoutSeconds = 2.0,
    ) -> dict[str, Any]:
        """Remove one free planning-scene object after explicit operator cleanup intent.

        Refuses attached objects; release and verify them first. This tool only mutates
        the MoveIt planning scene and does not move the robot or command the gripper.
        """
        return tools.remove_scene_object(robot_name, object_name, timeout_s=timeout_s)

    @mcp.tool()
    def moveit_open_gripper(robot_name: RobotName = "UR10", timeout_s: TimeoutSeconds = 5.0) -> dict[str, Any]:
        """Open UR10's gripper through Vizor and verify /Robot/gripper plus /Robot/status feedback."""
        return tools.open_gripper(robot_name, timeout_s=timeout_s)

    @mcp.tool()
    def moveit_close_gripper(robot_name: RobotName = "UR10", timeout_s: TimeoutSeconds = 5.0) -> dict[str, Any]:
        """Close UR10's gripper through Vizor and verify /Robot/gripper plus /Robot/status feedback."""
        return tools.close_gripper(robot_name, timeout_s=timeout_s)

    @mcp.tool()
    def moveit_attach_object(
        object_name: ObjectName,
        robot_name: RobotName = "UR10",
        verified_gripper_closed: VerifiedGripperClosed = False,
    ) -> dict[str, Any]:
        """Attach a free planning-scene object to the gripper after the gripper has been closed."""
        return tools.attach_object(
            robot_name,
            object_name,
            verified_gripper_closed=verified_gripper_closed,
        )

    return mcp


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Vizor ROS 1 MoveIt MCP server")
    parser.add_argument("--rosbridge-host", default="localhost")
    parser.add_argument("--rosbridge-port", type=int, default=9090)
    parser.add_argument("--transport", choices=("stdio", "sse", "streamable-http"), default="stdio")
    parser.add_argument("--http-host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=8000)
    args = parser.parse_args()

    tools = build_tools(host=args.rosbridge_host, port=args.rosbridge_port)
    mcp = build_mcp(tools=tools, host=args.http_host, port=args.http_port)
    mcp.run(transport=args.transport)
