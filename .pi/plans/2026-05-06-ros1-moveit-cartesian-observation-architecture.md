# ROS1 MoveIt Cartesian Fix And Observation Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the ROS1 Vizor Cartesian MoveIt signature mismatch, expose one higher-signal observation tool to the voice agent, and document the system locations plus reference inspirations in `ARCHITECTURE.md`.

**Architecture:** Keep Pipecat as the Voice Runtime and agent orchestration host. Keep `Multi-Actor-Interface-Library` as the host-side ROS1 MoveIt MCP bridge. Fix the actual MoveIt call in the ROS1 Vizor container/source image, because the MCP only publishes ROS requests and does not call `compute_cartesian_path`.

**Tech Stack:** Python 3.12, pytest, LangChain/LangGraph, FastMCP, ROS1 Noetic, rosbridge, MoveIt `moveit_commander`, Docker `vizor-demo`.

---

## Fresh Context Brief

Read these first:

- `C:\Users\Samuel\Documents\github\pipecat\AGENTS.md`
- `C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\AGENTS.md`
- `C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\CONTEXT.md`
- `C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\ARCHITECTURE.md`
- `C:\Users\Samuel\Documents\github\Multi-Actor-Interface-Library\docs\VIZOR_MOVEIT_MCP.md`

Important facts:

- We are running ROS1 Noetic, not ROS2.
- The relevant live container is `vizor-demo`.
- The MCP process runs on the host from `C:\Users\Samuel\Documents\github\Multi-Actor-Interface-Library`.
- The MCP publishes ROS1 messages through rosbridge at `ws://localhost:9090`.
- The broken call is inside the ROS1 Vizor robot node, not in the MCP wrapper.
- Confirmed live container call site:

```text
vizor-demo:/root/catkin_ws/src/vizor_lib/src/vizor_lib/robot.py:227
result = self.move_group.compute_cartesian_path(target_poses, eef_step, jump_threshold)
```

Confirmed live binding signature:

```text
MoveGroupCommander.compute_cartesian_path(self, waypoints, eef_step, avoid_collisions=True, path_constraints=None)
```

The fix is:

```python
result = self.move_group.compute_cartesian_path(
    target_poses,
    eef_step,
    avoid_collisions=True,
)
```

Do not treat a container writable-layer edit as persistent. The current `vizor-demo` has no bind mounts.

## File Map

Pipecat repo:

- Modify: `C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\ARCHITECTURE.md`
- Modify: `C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server\robot_control\call_validation.py`
- Modify: `C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server\robot_control\context.py`
- Modify: `C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server\prompts.py`
- Modify: `C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server\tests\test_robot_call_validation.py`
- Modify: `C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server\tests\test_robot_context.py`
- Modify: `C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server\tests\test_robot_mcp_bridge.py`
- Modify: `C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server\tests\test_prompts.py`

MoveIt MCP repo:

- Modify: `C:\Users\Samuel\Documents\github\Multi-Actor-Interface-Library\moveit_mcp\server.py`
- Modify: `C:\Users\Samuel\Documents\github\Multi-Actor-Interface-Library\moveit_mcp\tools.py`
- Modify: `C:\Users\Samuel\Documents\github\Multi-Actor-Interface-Library\moveit_mcp\vizor_client.py`
- Modify: `C:\Users\Samuel\Documents\github\Multi-Actor-Interface-Library\tests\test_moveit_mcp_planning_tools.py`
- Modify: `C:\Users\Samuel\Documents\github\Multi-Actor-Interface-Library\tests\test_moveit_mcp_server.py`
- Modify: `C:\Users\Samuel\Documents\github\Multi-Actor-Interface-Library\tests\test_vizor_client.py`
- Modify: `C:\Users\Samuel\Documents\github\Multi-Actor-Interface-Library\tests\test_moveit_mcp_agent_contract.py`
- Modify: `C:\Users\Samuel\Documents\github\Multi-Actor-Interface-Library\docs\VIZOR_MOVEIT_MCP.md`

Vizor ROS1 source/image:

- Persistent source location is not yet confirmed in the host repos.
- Live path: `vizor-demo:/root/catkin_ws/src/vizor_lib/src/vizor_lib/robot.py`
- Search likely host roots:
  - `C:\Users\Samuel\Documents\github\BehFab2025_VizorHRC`
  - `C:\Users\Samuel\Documents\github\foc_workshop`

## Task 1: Reconfirm The ROS1 MoveIt Failure Boundary

**Files:**
- Read: Docker container `vizor-demo`
- Read: `C:\Users\Samuel\Documents\github\Multi-Actor-Interface-Library\moveit_mcp\vizor_client.py`

- [ ] **Step 1: Confirm running containers**

Run:

```powershell
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"
```

Expected: `vizor-demo` is running from `local/noetic-vizor-rviz:latest`.

- [ ] **Step 2: Confirm the live binding signature**

Run:

```powershell
docker exec vizor-demo bash -lc 'source /opt/ros/noetic/setup.bash; source /root/catkin_ws/devel/setup.bash; python3 - <<'"'"'PY'"'"'
import inspect
import moveit_commander
from moveit_commander.move_group import MoveGroupCommander
print(inspect.signature(MoveGroupCommander.compute_cartesian_path))
PY'
```

Expected:

```text
(self, waypoints, eef_step, avoid_collisions=True, path_constraints=None)
```

- [ ] **Step 3: Confirm the live call site**

Run:

```powershell
docker exec vizor-demo bash -lc 'nl -ba /root/catkin_ws/src/vizor_lib/src/vizor_lib/robot.py | sed -n "215,235p"'
```

Expected: line near `227` calls:

```python
result = self.move_group.compute_cartesian_path(target_poses, eef_step, jump_threshold)
```

- [ ] **Step 4: Confirm the MCP is only a ROS publisher**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\Multi-Actor-Interface-Library
rg -n "compute_cartesian_path|request/cartesian|PlanningCartesian" moveit_mcp tests docs
```

Expected: `moveit_mcp/vizor_client.py` publishes `vizor_package/PlanningCartesian`; no MCP code calls `compute_cartesian_path`.

## Task 2: Fix The Persistent ROS1 Vizor Source Or Prepare A Minimal Container Patch

**Files:**
- Modify: persistent `vizor_lib/robot.py` when located.
- Fallback live-only modify: `vizor-demo:/root/catkin_ws/src/vizor_lib/src/vizor_lib/robot.py`

- [ ] **Step 1: Locate persistent source**

Run:

```powershell
rg -n "compute_cartesian_path|jump_threshold|class Robot" C:\Users\Samuel\Documents\github\BehFab2025_VizorHRC C:\Users\Samuel\Documents\github\foc_workshop --glob "!nul"
```

Expected: find a real `robot.py` source file or only docs. If only docs appear, use the live-container patch path below and document that it is temporary.

- [ ] **Step 2: If persistent source exists, write the failing/static regression check**

Create or update the closest available test for the Vizor source. If there is no test harness, create a small script in that repo under `.pi/checks/check_moveit_cartesian_signature.py`:

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
matches = list(ROOT.rglob("robot.py"))
targets = [path for path in matches if "vizor_lib" in str(path)]
if not targets:
    raise SystemExit("No vizor_lib robot.py found")

bad = []
for path in targets:
    text = path.read_text(encoding="utf-8")
    if "compute_cartesian_path(target_poses, eef_step, jump_threshold)" in text:
        bad.append(str(path))

if bad:
    raise SystemExit("Old MoveIt Cartesian signature still present: " + ", ".join(bad))
```

Run it from the repo where it was created:

```powershell
python .pi\checks\check_moveit_cartesian_signature.py
```

Expected before fix: FAIL with `Old MoveIt Cartesian signature still present`.

- [ ] **Step 3: Patch persistent source**

Replace:

```python
result = self.move_group.compute_cartesian_path(target_poses, eef_step, jump_threshold)
```

with:

```python
result = self.move_group.compute_cartesian_path(
    target_poses,
    eef_step,
    avoid_collisions=True,
)
```

Keep `eef_step = 0.05`. Remove or stop using `jump_threshold = 0.0` in the multi-waypoint branch if it becomes unused.

- [ ] **Step 4: If no persistent source exists, apply a live-container patch with backup**

Run:

```powershell
docker exec vizor-demo bash -lc 'cp /root/catkin_ws/src/vizor_lib/src/vizor_lib/robot.py /root/catkin_ws/src/vizor_lib/src/vizor_lib/robot.py.bak-20260506'
docker exec vizor-demo bash -lc 'python3 - <<'"'"'PY'"'"'
from pathlib import Path
path = Path("/root/catkin_ws/src/vizor_lib/src/vizor_lib/robot.py")
text = path.read_text()
old = "result = self.move_group.compute_cartesian_path(target_poses, eef_step, jump_threshold)"
new = """result = self.move_group.compute_cartesian_path(
                    target_poses,
                    eef_step,
                    avoid_collisions=True,
                )"""
if old not in text:
    raise SystemExit("old call not found")
path.write_text(text.replace(old, new))
PY'
docker exec vizor-demo bash -lc 'nl -ba /root/catkin_ws/src/vizor_lib/src/vizor_lib/robot.py | sed -n "220,235p"'
```

Expected: the live file shows the new keyword call. This does not affect the running Python process yet.

- [ ] **Step 5: Restart only the ROS1 control node or recreate the container through the normal operator path**

Preferred persistent path: rebuild/restart from the Docker compose flow documented in `C:\Users\Samuel\Documents\github\Multi-Actor-Interface-Library\docs\VIZOR_MOVEIT_MCP.md`.

If doing a live-only patch, explicitly tell the user before restart that it is temporary. Then restart the normal Vizor stack from:

```powershell
cd C:\Users\Samuel\Documents\github\BehFab2025_VizorHRC\BehFab2025_VizorHRC\01_Docker
docker compose -f vizor_config.yml up --build
```

Expected logs:

```text
Rosbridge WebSocket server started at ws://0.0.0.0:9090
You can start planning now!
Ready to take commands for planning group arm.
```

## Task 3: Add `moveit_get_robot_state` To The ROS1 MoveIt MCP

**Files:**
- Modify: `C:\Users\Samuel\Documents\github\Multi-Actor-Interface-Library\moveit_mcp\vizor_client.py`
- Modify: `C:\Users\Samuel\Documents\github\Multi-Actor-Interface-Library\moveit_mcp\tools.py`
- Modify: `C:\Users\Samuel\Documents\github\Multi-Actor-Interface-Library\moveit_mcp\server.py`
- Test: `C:\Users\Samuel\Documents\github\Multi-Actor-Interface-Library\tests\test_vizor_client.py`
- Test: `C:\Users\Samuel\Documents\github\Multi-Actor-Interface-Library\tests\test_moveit_mcp_planning_tools.py`
- Test: `C:\Users\Samuel\Documents\github\Multi-Actor-Interface-Library\tests\test_moveit_mcp_server.py`

This is the first additional observation tool. Do not add a broad ROS introspection surface yet.

- [ ] **Step 1: Add failing client test**

In `tests/test_vizor_client.py`, add:

```python
def test_get_robot_state_combines_pose_physical_mode_and_joint_state():
    transport = FakeRosbridgeTransport(physical_mode=False)
    transport.set_current_pose(
        "UR10",
        {
            "position": {"x": 0.57, "y": 0.39, "z": 0.62},
            "orientation": {"x": 0.0, "y": -0.70710678, "z": -0.70710678, "w": 0.0},
        },
        planning_frame="base_link",
    )
    transport.queue_joint_state("/UR10/move_group/fake_controller_joint_states", [0, -1.57, 1.57, 0, 0, 0])
    client = VizorClient(transport=transport)

    result = client.get_robot_state(robot="UR10", timeout_s=0.1)

    assert result.robot == "UR10"
    assert result.pose is not None
    assert result.planning_frame == "base_link"
    assert result.physical_mode is False
    assert result.joint_state == [0, -1.57, 1.57, 0, 0, 0]
```

Run:

```powershell
cd C:\Users\Samuel\Documents\github\Multi-Actor-Interface-Library
uv run pytest tests/test_vizor_client.py::test_get_robot_state_combines_pose_physical_mode_and_joint_state -q
```

Expected: FAIL because `get_robot_state` does not exist yet.

- [ ] **Step 2: Add transport read for current joint state**

In `moveit_mcp/vizor_client.py`, extend `RosbridgeTransport`:

```python
def read_joint_state(self, topic: str, timeout_s: float) -> list[float] | None: ...
```

Add to `FakeRosbridgeTransport`:

```python
def read_joint_state(self, topic: str, timeout_s: float) -> list[float] | None:
    self.events.append(("read_joint_state", topic, timeout_s))
    return self.wait_for_joint_state(topic, timeout_s)
```

Add to `RoslibpyTransport`:

```python
def read_joint_state(self, topic: str, timeout_s: float) -> list[float] | None:
    return self.wait_for_joint_state(topic, timeout_s)
```

- [ ] **Step 3: Add robot state feedback model**

In `moveit_mcp/vizor_client.py`, add near `CurrentPoseFeedback`:

```python
@dataclass(frozen=True)
class RobotStateFeedback:
    robot: str
    ok: bool
    status: str
    planning_frame: str | None
    pose: Pose | None
    physical_mode: bool | None
    joint_state: list[float] | None
    source: str
    message: str
```

- [ ] **Step 4: Add `VizorClient.get_robot_state`**

In `VizorClient`, add:

```python
def get_robot_state(self, *, robot: str, timeout_s: float = 2.0) -> RobotStateFeedback:
    with self._lock_for(robot):
        pose_feedback = self.get_current_pose(robot=robot, timeout_s=timeout_s)
        physical_mode = self.transport.read_physical_mode(PHYSICAL_PARAM)
        joint_topic = f"/{robot}/move_group/fake_controller_joint_states"
        joint_state = self.transport.read_joint_state(joint_topic, timeout_s)
        ok = pose_feedback.ok and physical_mode is not None
        status = "robot state observed" if ok else "robot state incomplete"
        return RobotStateFeedback(
            robot=robot,
            ok=ok,
            status=status,
            planning_frame=pose_feedback.planning_frame,
            pose=pose_feedback.pose,
            physical_mode=physical_mode,
            joint_state=joint_state,
            source=f"{pose_feedback.source}; {PHYSICAL_PARAM}; {joint_topic}",
            message="Robot state observed" if ok else "Robot state is missing pose or physical-mode feedback",
        )
```

If the nested lock deadlocks, replace the inner `get_current_pose` call with the body of `get_current_pose`; the existing lock is `RLock`, so it should be safe.

- [ ] **Step 5: Add tool result wrapper**

In `moveit_mcp/tools.py`, add:

```python
def get_robot_state(self, robot: str, timeout_s: float = 2.0) -> dict[str, Any]:
    feedback = self.client.get_robot_state(robot=robot, timeout_s=timeout_s)
    checks = [
        VerificationCheck("current_pose_observed", feedback.pose is not None, str(feedback.pose)),
        VerificationCheck("physical_mode_observed", feedback.physical_mode is not None, str(feedback.physical_mode)),
        VerificationCheck("joint_state_observed", feedback.joint_state is not None, str(feedback.joint_state)),
    ]
    evidence = [
        Evidence("ros_observation", feedback.source),
    ]
    raw = {
        "planning_frame": feedback.planning_frame,
        "pose": feedback.pose.to_msg() if feedback.pose is not None else None,
        "physical_mode": feedback.physical_mode,
        "joint_state": feedback.joint_state,
        "source": feedback.source,
    }
    if feedback.ok:
        return ToolResult.pass_result(
            robot=robot,
            tool="get_robot_state",
            phase="observed",
            status=feedback.status,
            message=feedback.message,
            checks=checks,
            evidence=evidence,
            raw=raw,
        ).to_dict()
    return ToolResult.fail_result(
        robot=robot,
        tool="get_robot_state",
        phase="observed",
        status=feedback.status,
        message=feedback.message,
        correction="Check rosbridge, /UR10/get_current_pose, and /vizor_robot_control/physical before retrying.",
        checks=checks,
        evidence=evidence,
        raw=raw,
    ).to_dict()
```

- [ ] **Step 6: Register the canonical MCP tool**

In `moveit_mcp/server.py`, add after `moveit_get_current_pose`:

```python
@mcp.tool()
def moveit_get_robot_state(robot_name: RobotName = "UR10", timeout_s: TimeoutSeconds = 2.0) -> dict[str, Any]:
    """Read UR10 pose, planning frame, physical-mode flag, and latest fake-controller joint state.

    Use this when the agent needs broader observation than the current TCP pose, especially
    before diagnosing motion failures or explaining whether the simulation is ready.
    This is a read-only ROS1 observation tool.
    """
    return tools.get_robot_state(robot_name, timeout_s=timeout_s)
```

Do not add a legacy alias unless a current client needs it.

- [ ] **Step 7: Add tool tests**

In `tests/test_moveit_mcp_planning_tools.py`, add:

```python
def test_get_robot_state_returns_read_only_state_feedback():
    transport = FakeRosbridgeTransport(physical_mode=False)
    transport.set_current_pose("UR10", CURRENT_POSE, planning_frame="base_link")
    transport.queue_joint_state("/UR10/move_group/fake_controller_joint_states", FINAL_POSITIONS)
    tools = MoveItMcpTools.with_fake_transport(transport)

    result = tools.get_robot_state("UR10", timeout_s=0.1)

    assert result["ok"] is True
    assert result["tool"] == "get_robot_state"
    assert result["feedback"]["can_execute"] is False
    assert result["verification"]["result"] == "pass"
    assert result["raw"]["planning_frame"] == "base_link"
    assert result["raw"]["pose"] == CURRENT_POSE
    assert result["raw"]["physical_mode"] is False
    assert result["raw"]["joint_state"] == FINAL_POSITIONS
```

In `tests/test_moveit_mcp_server.py`, extend canonical tool registration assertions to include:

```python
"moveit_get_robot_state",
```

Add a call test:

```python
async def test_get_robot_state_tool_returns_state_payload():
    transport = FakeRosbridgeTransport(physical_mode=False)
    transport.set_current_pose("UR10", CURRENT_POSE, planning_frame="base_link")
    transport.queue_joint_state("/UR10/move_group/fake_controller_joint_states", FINAL_POSITIONS)
    tools = MoveItMcpTools.with_fake_transport(transport)
    mcp = build_mcp(tools=tools)

    _, payload = await mcp.call_tool("moveit_get_robot_state", {})

    assert payload["tool"] == "get_robot_state"
    assert payload["raw"]["physical_mode"] is False
```

- [ ] **Step 8: Run MCP tests**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\Multi-Actor-Interface-Library
uv run pytest tests/test_vizor_client.py tests/test_moveit_mcp_planning_tools.py tests/test_moveit_mcp_server.py tests/test_moveit_mcp_agent_contract.py -q
```

Expected: PASS.

## Task 4: Expose `moveit_get_robot_state` To The Pipecat Agent

**Files:**
- Modify: `server/robot_control/call_validation.py`
- Modify: `server/robot_control/context.py`
- Modify: `server/prompts.py`
- Test: `server/tests/test_robot_call_validation.py`
- Test: `server/tests/test_robot_context.py`
- Test: `server/tests/test_robot_mcp_bridge.py`
- Test: `server/tests/test_prompts.py`

- [ ] **Step 1: Add failing call-validation test**

In `server/tests/test_robot_call_validation.py`, add:

```python
def test_accepts_robot_state_observation_arguments():
    validate_robot_tool_call("moveit_get_robot_state", {"robot_name": "UR10", "timeout_s": 2.0})
    assert canonical_mcp_tool_name("moveit_get_robot_state") == "moveit_get_robot_state"
    assert "pose" in agent_tool_description("moveit_get_robot_state").lower()
    assert "physical" in agent_tool_description("moveit_get_robot_state").lower()
```

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run pytest tests/test_robot_call_validation.py::test_accepts_robot_state_observation_arguments -q
```

Expected: FAIL.

- [ ] **Step 2: Update allowed tools**

In `server/robot_control/call_validation.py`, change:

```python
CANONICAL_ONLY_MCP_TOOL_NAMES: frozenset[str] = frozenset()
```

to:

```python
CANONICAL_ONLY_MCP_TOOL_NAMES: frozenset[str] = frozenset({"moveit_get_robot_state"})
```

Add description:

```python
"moveit_get_robot_state": (
    "Observe the UR10 current pose, planning frame, physical-mode flag, and latest "
    "fake-controller joint state. Use it to diagnose readiness or motion failures; "
    "use moveit_get_current_pose for ordinary relative motion grounding."
),
```

Add allowed args:

```python
"moveit_get_robot_state": {"robot_name", "timeout_s"},
```

Change validation branch:

```python
if name in {"moveit_get_current_pose", "moveit_get_robot_state"}:
    _validate_timeout(arguments.get("timeout_s"))
    return
```

- [ ] **Step 3: Update robot context**

In `server/robot_control/context.py`, change:

```python
if tool_name not in {"moveit_get_current_pose", "moveit_get_robot_status"}:
    return
```

to:

```python
if tool_name not in {"moveit_get_current_pose", "moveit_get_robot_state"}:
    return
```

Add or adjust tests in `server/tests/test_robot_context.py`:

```python
def test_robot_context_updates_from_robot_state_observation():
    store = RobotContextStore(time_fn=lambda: 10.0)
    output = json.dumps(
        {
            "structured_content": {
                "ok": True,
                "robot": "UR10",
                "raw": {
                    "pose": {
                        "position": {"x": 0.57, "y": 0.39, "z": 0.62},
                        "orientation": {"x": 0.0, "y": -0.70710678, "z": -0.70710678, "w": 0.0},
                    },
                    "physical_mode": False,
                    "joint_state": [0, -1.57, 1.57, 0, 0, 0],
                },
            }
        }
    )

    store.update_from_tool_result("moveit_get_robot_state", output)

    assert store.has_recent_robot_observation(max_age_s=1.0)
    assert store.latest_tcp_pose()["position"]["z"] == 0.62
```

- [ ] **Step 4: Update prompt**

In `server/prompts.py`, add `moveit_get_robot_state` to the available tools section:

```text
- moveit_get_robot_state: observe current pose, planning frame, physical-mode flag, and latest fake-controller joint state.
```

Add a tool-use rule:

```text
- Use moveit_get_robot_state when diagnosing readiness, a failed motion, or whether simulation feedback is available; use moveit_get_current_pose for ordinary relative motion grounding.
```

Do not remove the fresh-pose rule.

- [ ] **Step 5: Update prompt tests**

In `server/tests/test_prompts.py`, add `moveit_get_robot_state` to `CANONICAL_TOOLS`. Remove it from `STALE_TOOLS` if present.

Add:

```python
def test_prompt_distinguishes_pose_observation_from_robot_state_observation() -> None:
    prompt = SYSTEM_PROMPT.lower()
    assert "moveit_get_robot_state" in prompt
    assert "readiness" in prompt
    assert "failed motion" in prompt
    assert "moveit_get_current_pose for ordinary relative motion" in prompt
```

- [ ] **Step 6: Update MCP bridge tests**

In `server/tests/test_robot_mcp_bridge.py`, add `moveit_get_robot_state` to a fake canonical server:

```python
Tool(name="moveit_get_robot_state", description="State", inputSchema={"type": "object"})
```

Assert it is advertised with agent-friendly description and called by canonical name:

```python
assert "moveit_get_robot_state" in [tool["name"] for tool in bridge.function_tools()]
await bridge.call_tool("moveit_get_robot_state", {"robot_name": "UR10"})
assert ("moveit_get_robot_state", {"robot_name": "UR10"}) in server.called
```

- [ ] **Step 7: Run Pipecat tests**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run pytest tests/test_robot_call_validation.py tests/test_robot_context.py tests/test_robot_mcp_bridge.py tests/test_prompts.py -q
```

Expected: PASS.

## Task 5: Document System Locations And Reference Inspirations In `ARCHITECTURE.md`

**Files:**
- Modify: `C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\ARCHITECTURE.md`

Keep the current style: stable, short, target architecture, no incident timeline.

- [ ] **Step 1: Add a system locator note under `### MoveIt MCP Boundary`**

Replace that section with this concise version:

```markdown
### MoveIt MCP Boundary

MoveIt MCP is the execution seam into the ROS 1 robot simulation stack. The voice agent routes movement through MoveIt planning/execution workflows. MoveIt and the robot simulation stack are the movement-safety boundary.

The host-side ROS 1 MoveIt MCP lives in `C:\Users\Samuel\Documents\github\Multi-Actor-Interface-Library\moveit_mcp`. It exposes FastMCP tools and talks to ROS 1 through rosbridge. The main entrypoint is `moveit_mcp.server`, the agent-facing tool wrappers live in `moveit_mcp.tools`, and the ROS 1 topic/service adapter lives in `moveit_mcp.vizor_client`.

The Vizor ROS 1 container owns the downstream MoveIt node and robot control code. In the running `vizor-demo` container, the MoveIt server is `/UR10/move_group`, the app-facing control node is `/vizor_robot_control`, and the robot logic is under `/root/catkin_ws/src/vizor_lib/src/vizor_lib/`. Treat container paths as runtime locators; persistent fixes belong in the Docker image source.

Agent-facing robot tools should stay semantic and narrow: observation tools, planning tools, verified execution tools, gripper tools, and future failure-explanation tools. Do not expose broad ROS control or raw topic mutation tools to Agent Orchestration by default.
```

- [ ] **Step 2: Add a short reference section under `## Cross-Cutting Concerns`**

Add after `### Observability` or before `### Documentation hygiene`:

```markdown
### Reference inspirations

Use these as inspiration for agent-first robotics patterns, not as dependencies or sources of truth:

- [NASA JPL ROSA](https://github.com/nasa-jpl/rosa): ROS agent pattern for introspection-first operation and diagnosis.
- [RobotMCP ROS MCP Server](https://github.com/robotmcp/ros-mcp-server): MCP boundary pattern for ROS topic/service/action observation and control.
- [RAI](https://robotecai.github.io/rai/faq/ROS_2_Overview/): connector pattern for agent tools, robot status, and readiness-gated interaction.
- [ROS-LLM](https://arxiv.org/abs/2406.19741): structured behavior execution and reflection pattern for ROS actions/services.
- [APYROBO](https://github.com/apyrobo/apyrobo): semantic capability, safety policy, observability, and replay ideas.
- [Pipecat function calling](https://docs.pipecat.ai/pipecat/learn/function-calling): voice runtime pattern for tool calls inside a conversational pipeline.
- [OpenAI Realtime MCP](https://developers.openai.com/api/docs/guides/realtime-mcp): MCP lifecycle, tool narrowing, and approval patterns for realtime agents.

Common lesson: the agent owns sequencing and tool choice, while the robot layer owns typed capability boundaries, readiness checks, planning, execution verification, and hard safety constraints.
```

- [ ] **Step 3: Add/update documentation test if one exists**

Search:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
rg -n "ARCHITECTURE|Reference inspirations|MoveIt MCP Boundary" tests ..
```

If no architecture-doc test exists, do not invent a new one unless the local pattern clearly supports it. Keep the architecture doc short.

## Task 6: Update MoveIt MCP Documentation

**Files:**
- Modify: `C:\Users\Samuel\Documents\github\Multi-Actor-Interface-Library\docs\VIZOR_MOVEIT_MCP.md`

- [ ] **Step 1: Add the new observation tool to the tool list**

Add:

```markdown
- `moveit_get_robot_state`
```

- [ ] **Step 2: Update safe workflow**

Add:

```markdown
Use `moveit_get_robot_state` when diagnosing readiness, failed execution, physical-mode state, or missing fake-controller feedback. Use `moveit_get_current_pose` for ordinary relative-motion grounding.
```

- [ ] **Step 3: Document the ROS1 Cartesian signature constraint**

Add a short note:

```markdown
This stack runs ROS 1 Noetic. In the current Vizor image, `MoveGroupCommander.compute_cartesian_path` exposes `(waypoints, eef_step, avoid_collisions=True, path_constraints=None)`. Multi-waypoint Cartesian planning must pass `avoid_collisions=True`; do not pass the old `jump_threshold` argument in this environment.
```

## Task 7: Verify The Live Wave Path

**Files:**
- Read: running services
- Optional evidence: save under `C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server\evidence\`

- [ ] **Step 1: Start services**

Use the operator dashboard if desired:

```powershell
cd C:\Users\Samuel\Documents\github\Multi-Actor-Interface-Library
uv run python scripts/run_operator_dashboard.py
```

Or start separately:

```powershell
cd C:\Users\Samuel\Documents\github\BehFab2025_VizorHRC\BehFab2025_VizorHRC\01_Docker
docker compose -f vizor_config.yml up --build
```

```powershell
cd C:\Users\Samuel\Documents\github\Multi-Actor-Interface-Library
uv run python -m moveit_mcp --rosbridge-host localhost --rosbridge-port 9090 --transport streamable-http --http-host 127.0.0.1 --http-port 8765
```

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run bot.py --profile hybrid_low_latency
```

- [ ] **Step 2: Smoke the new observation tool directly**

Run a direct MCP call using the existing project helper style or a short script:

```powershell
cd C:\Users\Samuel\Documents\github\Multi-Actor-Interface-Library
@'
from moveit_mcp.server import build_tools
tools = build_tools(host="localhost", port=9090)
print(tools.get_robot_state("UR10", timeout_s=2.0))
'@ | Set-Content .tmp_robot_state.py
uv run python .tmp_robot_state.py
Remove-Item .tmp_robot_state.py
```

Expected: `ok` is true, `raw.pose` exists, `raw.physical_mode` is false, and `raw.joint_state` is either a list or explicitly absent with a failed `joint_state_observed` check.

- [ ] **Step 3: Smoke Cartesian planning directly**

Use `moveit_get_current_pose`, then call `moveit_plan_cartesian_motion` with small bounded waypoints around the returned pose.

Expected: no `compute_cartesian_path` signature error appears in container logs. The MCP returns structured `ok=false` for unreachable/incomplete paths or `ok=true` for a successful plan; it must not hang until the client read timeout.

- [ ] **Step 4: Run the voice command**

Open:

```text
http://localhost:7860/client
```

Say:

```text
Mave, wave to me.
```

Expected:

- Pipecat transcript captures the command.
- Agent calls `moveit_get_current_pose` or `moveit_get_robot_state`.
- Agent calls `moveit_plan_and_execute_cartesian_motion`.
- Container logs do not show `compute_cartesian_path(... list, float, float) did not match C++ signature`.
- Assistant final text is brief and not `I encountered an error. Please try again.`

## Task 8: Final Verification Commands

- [ ] **Step 1: Pipecat focused tests**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run pytest tests/test_robot_call_validation.py tests/test_robot_context.py tests/test_robot_mcp_bridge.py tests/test_prompts.py -q
```

Expected: PASS.

- [ ] **Step 2: MoveIt MCP focused tests**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\Multi-Actor-Interface-Library
uv run pytest tests/test_vizor_client.py tests/test_moveit_mcp_planning_tools.py tests/test_moveit_mcp_server.py tests/test_moveit_mcp_agent_contract.py -q
```

Expected: PASS.

- [ ] **Step 3: Lint if configured and quick enough**

Run in each repo only if local config supports it:

```powershell
uv run ruff check .
```

Expected: PASS or report unrelated pre-existing failures clearly.

## Self-Review Checklist

- [ ] `ARCHITECTURE.md` says ROS 1 where the MoveIt MCP boundary is documented.
- [ ] `ARCHITECTURE.md` includes inspiration references as inspiration, not dependencies.
- [ ] `ARCHITECTURE.md` remains stable and short; no log timestamps or incident timeline.
- [ ] The MoveIt MCP docs mention the ROS1 Noetic Cartesian signature constraint.
- [ ] `moveit_get_robot_state` is read-only and does not execute or publish motion commands.
- [ ] Pipecat bridge only exposes the new observation tool after local validation.
- [ ] The live wave path no longer hits the old `jump_threshold` signature error.
