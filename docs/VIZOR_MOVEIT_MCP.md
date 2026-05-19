# Vizor MoveIt MCP

This MCP wraps the existing Vizor ROS 1 Noetic MoveIt topic contract over rosbridge. It is simulation-first: planning, execution, and gripper tools return structured verification feedback instead of fire-and-forget ROS publishes.

## What it provides

- Browser RViz/noVNC inside the existing `vizor-demo` container.
- Rosbridge access at `ws://localhost:9090`.
- Canonical FastMCP tools for Vizor MoveIt:
  - `moveit_get_current_pose`
  - `moveit_get_robot_state`
  - `moveit_list_scene_objects`
  - `moveit_get_object_context`
  - `moveit_plan_pick`
  - `moveit_plan_place`
  - `moveit_plan_pick_task`
  - `moveit_plan_place_task`
  - `moveit_plan_free_motion`
  - `moveit_plan_cartesian_motion`
  - `moveit_execute_plan`
  - `moveit_execute_task_solution`
  - `moveit_explain_motion_failure`
  - `moveit_verify_attached_object`
  - `moveit_remove_scene_object`
  - `moveit_open_gripper`
  - `moveit_close_gripper`
  - `moveit_attach_object`
- Verification envelopes with `ok`, `feedback`, `verification`, `evidence`, and `raw`.

Pipecat verified real-robot task execution exposes the semantic path: plan a task solution, get explicit approval, then call `moveit_execute_task_plan`. Direct MCP task execution with `moveit_execute_task_solution` remains the sim/emulated path.

## Operator dashboard shortcut

To start and monitor Vizor + RViz, MoveIt MCP, and the Pipecat voice agent from one browser page, run:

```powershell
.\Start-MAVE-Workshop.cmd
```

Open the printed `http://127.0.0.1:8787/?token=...` URL. Use the manual commands below only when you need to run services separately.

## Start Vizor + browser RViz

From the repo root:

```powershell
docker compose -f docker/compose/workshop.yml up --build
```

Wait for logs like:

```text
Rosbridge WebSocket server started at ws://0.0.0.0:9090
You can start planning now!
Ready to take commands for planning group arm.
```

Open the browser UI:

```text
http://127.0.0.1:6080/vnc_auto.html?host=127.0.0.1&port=6080&path=websockify&autoconnect=true&resize=remote
```

Use `vnc_auto.html`; the older `vnc.html` page can fail with a noVNC JavaScript error (`can't access property "type", ctrl is null`).

If the ROS Noetic EOL popup appears, click **OK**. RViz should show:

- `Global Status: Ok`
- `Grid`
- `UR10 RobotModel`
- UR10 rendered in the scene

## Start the MCP server

Run from this repository, not from the external Docker directory.

For Cursor/editor MCP clients, keep the default stdio transport:

```powershell
cd server
uv run python -m moveit_mcp --rosbridge-host localhost --rosbridge-port 9090
```

For the Pipecat voice agent, use Streamable HTTP on the existing MCP URL:

```powershell
cd server
uv run python -m moveit_mcp --rosbridge-host localhost --rosbridge-port 9090 --transport streamable-http --http-host 127.0.0.1 --http-port 8765
```

Pipecat should point to:

```text
http://127.0.0.1:8765/mcp
```

Path-stable HTTP form:

```powershell
uv --directory server run python -m moveit_mcp --rosbridge-host localhost --rosbridge-port 9090 --transport streamable-http --http-host 127.0.0.1 --http-port 8765
```

Cursor/MCP example config:

```text
.cursor/mcp-vizor-moveit.example.json
```

## Safe MCP workflow

Use `moveit_get_robot_state` when diagnosing readiness, failed execution, physical-mode state, or missing fake-controller feedback. Use `moveit_get_current_pose` for ordinary relative-motion grounding.

Use `moveit_list_scene_objects` before object-relative or pick tasks. It returns planning-scene object IDs, frames, poses, shape summaries, bounds, colors when available, and attached/free state. Use `moveit_get_object_context` with one returned object name to inspect bounds, grasp-relevant faces, clearance above the ground plane when available, and the planning frame.

Use `moveit_plan_pick_task` or `moveit_plan_place_task` for ordinary task-level pick/place workflows when available. Use `moveit_plan_compound_task` only for `hold`, `release`, `move_and_release`, and `pick_place`. It requires `backend="mtc"` and hard `requirements` with `requirements.goal` and `requirements.object_name`; transfer goals also require `requirements.target_pose` or `requirements.target_position`. `preferences` are non-executable planner hints. `stage_intents` are optional hints only. Unsupported goals or hints such as `slide`, `push`, raw code, scripts, or raw waypoints fail at planning. Solved task tools return a `raw.task_solution_id`, stage evidence, scene snapshot evidence, and an approval payload. They do not move the robot. Pipecat verified execution requires explicit `moveit_execute_task_plan` with that exact task solution after approval; direct MCP sim/emulated execution uses `moveit_execute_task_solution`.

Use legacy `moveit_plan_pick` and `moveit_plan_place` only when task-level tools are unavailable or a narrower ordinary plan is intended. A partial legacy pick result is diagnostic evidence and must not be executed as a pick.

Required motion workflow:

1. For relative motion or gestures, call `moveit_get_current_pose` first.
2. Call `moveit_plan_free_motion`, `moveit_plan_cartesian_motion`, `moveit_plan_pick_task`, `moveit_plan_place_task`, `moveit_plan_compound_task`, or a legacy planning tool.
3. Require `ok == true` and `feedback.can_execute == true`.
4. For Pipecat verified task execution, call `moveit_execute_task_plan` with the returned `raw.task_solution_id` after explicit approval. For direct MCP sim/emulated task execution, call `moveit_execute_task_solution`. For ordinary plans, call `moveit_execute_plan` with the returned `raw.plan_name`.
5. Require `verification.result == "pass"`.
6. If planning, execution, or verification fails, call `moveit_explain_motion_failure` with the failed tool name, arguments, result, and user intent when available.
7. After pick/place/compound execution, require attachment or release proof before claiming the object was picked, held, moved, placed, or released.

Do not expose or use combined `moveit_plan_and_execute_*` tools. Planning and execution are separate agent-visible verbs.

Do not execute a plan that was not verified in the same MCP process. Physical execution is blocked unless `/vizor_robot_control/physical` is confirmed false through rosapi.

Planning route note: active `free`, `cartesian`, and `sampled` planning routes use OMPL/RRTConnect. `moveit_plan_cartesian_motion` and `/UR10/request/cartesian` keep legacy names only; they do not guarantee straight TCP motion. Pilz LIN and `compute_cartesian_path` notes are historical diagnostics unless a future explicit route reintroduces them.

## Pick task backends

`moveit_plan_pick_task` defaults to the emulated backend. Emulated dynamic and vertical picks report multiple `raw.candidate_attempts` across object-appropriate grasp faces and distance variants, not a single `dynamic_1` side/back attempt.

Beam grasp policy is orientation-aware. Horizontal beams default to top-face candidates only unless the caller explicitly passes a side `grasp_face` preference. Vertical beams use side faces only; when scene relations are available, side faces marked as inner toward a neighbor or assembly center are excluded from automatic candidates.

A configured MTC backend is explicit opt-in. A solved MTC pick returns `raw.backend == "mtc"` with a returned `raw.task_solution_id`. An unsolved MTC call returns `ok=false` with `failed_stage` and `blocker`, does not silently fall back to emulation, and does not return a task solution.

The Vizor MTC proof service exposes ROS-side service boundaries at:

```text
/vizor_mtc/plan_pick_task
/vizor_mtc/plan_compound_task
```

It starts only when the Vizor image is launched with `VIZOR_ENABLE_MTC_PROOF=1`. Requests are passed through ROS params under each service name, for example `/vizor_mtc/plan_pick_task/request` and `/vizor_mtc/plan_compound_task/request`. The service returns sanitized stage summaries using current MTC stage terms: `CurrentState`, `Connect`, `GenerateGraspPose`, `ComputeIK`, `MoveRelative`, and `ModifyPlanningScene`.

These services currently fail closed until real typed MTC service plumbing and UR10/Robotiq semantic config are available. Failure evidence is diagnostic only.

## Compound task contract

`moveit_plan_compound_task` is MTC-only. The accepted goals are `hold`, `release`, `move_and_release`, and `pick_place`. The LLM provides hard task `requirements`, optional non-executable `preferences`, and optional `stage_intents` hints. The MCP/MTC backend compiles and solves the task. A solved response must include:

- `raw.task_solution_id`
- `raw.requirements` and `raw.preferences`
- `raw.execution_contract` with ordered typed steps
- source-stage and required-proof metadata for each step
- `/solution` preview evidence and AR preview evidence from `/vizor_robot_control`
- candidate attempts, candidate count, selected cost, scene snapshot, object context, and stage evidence
- no `raw.task_solution_id` when MTC is unavailable, unsupported, incomplete, unsolved, or non-previewable

Agent-facing compound calls use this shape:

```json
{
  "robot_name": "UR10",
  "backend": "mtc",
  "requirements": {
    "goal": "move_and_release",
    "object_name": "beam_001",
    "target_position": {"x": 0.55, "y": 0.2, "z": 0.12}
  },
  "preferences": {"grasp_face": "top"},
  "stage_intents": ["observe_current_state", "move_to_pose", "verify_released"]
}
```

Top-level `task_goal` and `object_name` are legacy/internal names after normalization, not the public compound task definition.

AR preview should expose the stable public path name `AgentPath`. Stage-level debug/cache names use `AgentPath:<two-digit-index>_<stage>`, for example `AgentPath:01_approach`. Publishing `AgentPath` to `/UR10/command/execute` is the AR approval/execution signal for the whole cached manipulation task. The local Vizor robot patch must not execute `AgentPath:*` debug trajectories directly; execution belongs to the verified task bridge. Publishing `AgentPath` to `/UR10/command/stop` cancels that active cached task; the next manipulation requires fresh observe and replan before another `AgentPath` execution.

Pipecat integration point: when Pipecat receives an approved task solution from MCP, its verified execution bridge should register/cache the task under `AgentPath`, pass staged debug names through for preview/evidence, send `/UR10/command/execute` with payload `AgentPath` only after explicit approval, and send `/UR10/command/stop` with payload `AgentPath` on cancellation. This repo only represents the local Vizor/MCP side of that bridge.

The verified executor supports these handlers: `motion`, `close_gripper`, `open_gripper`, `attach_object`, `release_object`, `verify_attached_object`, and `verify_released_object`.

Release is explicit. `moveit_release_object` requires `verified_gripper_open=true` and an `object_pose` from the execution contract; `moveit_verify_released_object` must pass before the agent claims a place or release completed. Plain compound `release` may report `raw.preview.ar_preview_mode="none_no_motion"` only for a legitimate solved release with held-object proof. The proof service must not invent solved release responses.

The current `/vizor_mtc/plan_compound_task` service is a boundary, not a real compound solver. It reports `ok=false` with stable `error`, `failed_stage`, `blocker`, `correction`, no `task_solution_id`, and `execution_contract.can_execute=false` until typed MTC compound construction, `/solution` preview export, and AR preview evidence are implemented. Pipecat must treat that as a planning failure, not as permission to use legacy planners or raw waypoints. The approved real-solver route is a typed C++ catkin package in the Vizor/RViz image, adapted from the official MoveIt Task Constructor pick/place demo patterns; the Python proof-service boundary remains diagnostic only because the required Noetic MTC Python modules are not importable in the current image.

## Gripper command path

`moveit_open_gripper` and `moveit_close_gripper` publish a `vizor_package/GeneralTask` to:

```text
/UR10/task/execute
```

The task name ends with `Gripper Open` or `Gripper Close`, matching the running `vizor_robot_control.py` branches. The legacy debug signal is:

- `/Robot/gripper` as `std_msgs/Bool` (`false` for open, `true` for close)

In the current Docker stack, `/Robot/gripper` is published by `/vizor_robot_control` and has no subscribers. Treat it as debug-only state, not attach/release proof. Attachment and release claims require explicit MoveIt planning-scene proof or verified real-gripper execution evidence from the Pipecat/verified-execution bridge.

## Agent-facing tool contract

The MCP tool metadata is part of the agent contract. Descriptions and schemas must keep teaching agents to:

- use canonical `moveit_*` tools;
- ground relative or vague motion with `moveit_get_current_pose` before planning;
- plan in `base_link` and gate execution on `ok == true`, `feedback.can_execute == true`, and the returned `raw.task_solution_id` or `raw.plan_name`;
- prefer `moveit_plan_pick_task` and `moveit_plan_place_task` for ordinary grounded object workflows;
- use `moveit_plan_compound_task` with `backend="mtc"` for supported compound workflows defined by `requirements` and optional `preferences`;
- in Pipecat verified mode, execute returned supported task solutions only with `moveit_execute_task_plan` after explicit approval;
- use `moveit_execute_task_solution` only for direct MCP sim/emulated task execution with the returned `raw.task_solution_id`;
- use legacy `moveit_plan_pick` and `moveit_plan_place` only when task-level tools are unavailable or a narrower ordinary plan is intended;
- call `moveit_execute_plan` only for a verified ordinary plan returned by the same MCP process;
- call `moveit_explain_motion_failure` after failed planner or executor evidence before retrying complex motion;
- call attachment or release verification after task execution before claiming the object moved, was held, placed, or released;
- keep planning and execution separate; do not expose combined plan-and-execute tools.

Run the contract evals with:

```bash
uv run pytest tests/test_moveit_mcp_agent_contract.py -v
```

## Manual motion proof command

With `vizor-demo` running and RViz open, this sends a visible out/back/out/back UR10 motion through the MCP tool layer:

PowerShell:

```powershell
cd server
@'
import time
from uuid import uuid4
from moveit_mcp.server import build_tools

base = {
    "position": {"x": 0.5723589519983855, "y": 0.3941410000780623, "z": 0.6235999970798317},
    "orientation": {
        "x": -2.0030704870235343e-16,
        "y": -0.7071067812590626,
        "z": -0.7071067811140325,
        "w": 4.329780280011331e-17,
    },
}

target = {
    "position": {"x": 0.5723589519983855, "y": 0.4441410000780623, "z": 0.6235999970798317},
    "orientation": base["orientation"],
}

tools = build_tools(host="localhost", port=9090)
for label, pose in [("out", target), ("back", base), ("out_again", target), ("home_again", base)]:
    plan_name = f"pi_live_{label}_{uuid4().hex[:8]}"
    plan = tools.plan_free_motion("UR10", plan_name, pose, timeout_s=25.0)
    print(label, "plan", plan["ok"], plan["feedback"]["status"], plan["raw"].get("trajectory_points"))
    if not plan["ok"]:
        raise SystemExit(plan)

    time.sleep(0.75)
    executed = tools.execute_plan("UR10", plan_name, timeout_s=25.0)
    print(label, "execute", executed["ok"], executed["verification"]["result"], executed["feedback"]["status"])
    if not executed["ok"]:
        raise SystemExit(executed)
    time.sleep(2.0)
'@ | Set-Content .tmp_vizor_motion.py
uv run python .tmp_vizor_motion.py
Remove-Item .tmp_vizor_motion.py
```

Git Bash equivalent:

```bash
uv run python - <<'PY'
# Paste the same Python body from the PowerShell block above here.
PY
```

Expected output shape:

```text
out plan True success! 20
out execute True pass final joint state matched
back plan True success! 20
back execute True pass final joint state matched
```

## Tests

Local unit and integration-skip path:

```bash
uv run pytest tests/test_moveit_mcp_models.py tests/test_vizor_client.py tests/test_moveit_mcp_planning_tools.py tests/test_moveit_mcp_execution_tools.py tests/test_moveit_mcp_gripper_tools.py tests/test_moveit_mcp_server.py tests/test_vizor_moveit_integration.py -v
```

Live Vizor integration, with `vizor-demo` running:

```bash
uv run pytest tests/test_vizor_moveit_integration.py -v --vizor-integration
```

## Current pose and coordinate rule

Poses are sent in the robot-local ROS planning frame. Current `UR10` planning frame is `base_link`.

Use the current MoveIt pose as the reachable baseline when a target is uncertain or planning fails; do not infer coordinates from the RViz camera view.

The `vizor-demo` image starts a read-only ROS service:

```text
/UR10/get_current_pose  std_srvs/Trigger
```

The MoveIt MCP exposes that service as:

```text
moveit_get_current_pose(robot_name="UR10")
```

Expected MCP payload:

```json
{
  "ok": true,
  "robot": "UR10",
  "tool": "get_current_pose",
  "raw": {
    "planning_frame": "base_link",
    "pose": {
      "position": {"x": 0.0, "y": 0.0, "z": 0.0},
      "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
    },
    "source": "/UR10/get_current_pose"
  }
}
```

Manual pose query inside `vizor-demo`:

```bash
docker exec vizor-demo bash -lc 'source /opt/ros/noetic/setup.bash; source /root/catkin_ws/devel/setup.bash; python3 - <<"PY"
import sys, rospy, moveit_commander
moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node("query_pose", anonymous=True)
group = moveit_commander.MoveGroupCommander("arm", ns="UR10", robot_description="UR10/robot_description")
pose = group.get_current_pose().pose
print(pose)
PY'
```

## Troubleshooting

### noVNC red error page

If you see:

```text
noVNC encountered an error:
can't access property "type", ctrl is null
```

Use:

```text
http://127.0.0.1:6080/vnc_auto.html?host=127.0.0.1&port=6080&path=websockify&autoconnect=true&resize=remote
```

Do not use plain `vnc.html` for this image.

### RViz opens but the robot is not visible

Rebuild the image so the custom UR10 RViz config is copied in:

```powershell
docker compose -f docker/compose/workshop.yml up --build --force-recreate
```

Then reopen `vnc_auto.html` and click **OK** on the ROS EOL popup.

### `moveit_get_current_pose` returns unavailable

Check that the current-pose service is running in `vizor-demo`:

```bash
docker exec vizor-demo bash -lc 'source /opt/ros/noetic/setup.bash; rosservice list | grep /UR10/get_current_pose'
```

If missing, rebuild/restart `vizor-demo` because the service is launched by `/usr/local/bin/start-vizor-desktop.sh`.

Inspect service logs:

```bash
docker exec vizor-demo bash -lc 'tail -100 /tmp/vizor_current_pose_service.log'
```

### MCP refuses execution with `physical mode unknown`

Check that rosapi is running in `vizor-demo`:

```bash
docker exec vizor-demo bash -lc 'source /opt/ros/noetic/setup.bash; rosservice list | grep /rosapi/get_param'
```

If missing, rebuild/restart `vizor-demo`.

### Planning fails with `planning result invalid`

The target pose is likely unreachable or in collision. Query the current pose with `moveit_get_current_pose` and use it as a baseline, then move by a small offset such as `+0.05` in `y`.
