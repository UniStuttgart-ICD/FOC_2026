# Vizor MCP Design

Date: 2026-05-11

## Decision

Build a separate `vizor_mcp` package next to `moveit_mcp` in `C:\Users\Samuel\Documents\github\Multi-Actor-Interface-Library`.

`vizor_mcp` should read directly from ROSBridge, normalize the latest HoloLens/Vizor observations into one compact user-sensing context, and expose that context through a read-only MCP tool:

- `vizor_get_sensor_context`

Pipecat should not put this tool in the model's normal tool list. The host graph should call it before each LLM request and inject the result as a separate "User sensing context" block, alongside the existing robot context.

## Why Direct ROS

The old `VizorWebControl/backend` service used ROSBridge as the real source of truth, then persisted topic messages to MongoDB. For agent context, Mongo adds latency, schema drift, and an extra failure point. Since ROS is already running, the MCP should subscribe to ROS directly and keep a small latest-message cache in memory.

Mongo can remain useful for dashboards, replay, or study logs, but it should not be the runtime dependency for grounding the agent.

## MCP Shape

Use the same SDK style as `moveit_mcp`:

- `mcp.server.fastmcp.FastMCP`
- typed `Annotated[..., Field(...)]` parameters
- `@mcp.tool(...)`
- `streamable-http` transport for Pipecat
- `stdio` transport for local/manual debugging
- structured dict returns so clients receive `structuredContent`

Use MCP tool annotations for the primary sensor tool:

- `readOnlyHint=True`
- `destructiveHint=False`
- `idempotentHint=True`
- `openWorldHint=True`

The installed `mcp` package exposes these on `ToolAnnotations`, and `FastMCP.tool(...)` accepts `annotations` plus `structured_output`.

## Proposed Package Layout

```text
Multi-Actor-Interface-Library/
  vizor_mcp/
    __init__.py
    __main__.py
    server.py
    tools.py
    ros_client.py
    models.py
    transforms.py
  tests/
    test_vizor_mcp_context.py
    test_vizor_mcp_ros_cache.py
```

### `server.py`

Responsibilities:

- build `FastMCP("VizorSensingServer", ...)`
- register read-only tools
- start and stop the ROS client through a FastMCP lifespan
- parse CLI args:
  - `--rosbridge-host`
  - `--rosbridge-port`
  - `--transport stdio|sse|streamable-http`
  - `--http-host`
  - `--http-port`
  - `--max-age-s`
  - `--enable-holo1-tracking-on-startup`

### `ros_client.py`

Responsibilities:

- connect to ROSBridge with `roslibpy`
- subscribe to known semantic topics
- keep a thread-safe latest-message cache
- record `received_monotonic_s`, source topic, message type, and raw payload
- optionally resolve topic types through `/rosapi/topic_type`
- publish `/WorkerPool/control` with `{"data": "HOLO1_position_on"}` only when explicitly configured
- reconnect or report degraded status without crashing the MCP process

Default topic map:

```python
{
    "gaze": { "topic": "/HOLO1_GazePoint", "message_type": "std_msgs/String" },
    "user_transform": { "topic": "/HOLO1_Transform", "message_type": "geometry_msgs/Pose" },
    "manual_target": { "topic": "/Robot/target_manual", "message_type": "geometry_msgs/Pose" },
}
```

The message type for `/HOLO1_Transform` should be configurable because the old backend stored it dynamically through topic configs.

### `transforms.py`

Centralize coordinate conversion and test it. Start with the DF2025 transform:

```text
robot.x = -unity.y - 0.173
robot.y = -unity.z + 0.051
robot.z =  unity.x + 0.103
```

Keep offsets configurable. The old implementation had more than one transform variant, so this should be one named calibration with tests.

### `models.py`

Use Pydantic models or dataclasses for:

- `Vector3`
- `Quaternion`
- `Pose`
- `SensorReading`
- `VizorSensorContext`

Each semantic field should carry freshness:

- `available`
- `age_s`
- `stale`
- `source_topic`
- `message_type`

### `tools.py`

Primary tool:

```python
def get_sensor_context(max_age_s: float = 2.0, include_raw: bool = False) -> dict:
    ...
```

MCP registration name:

- `vizor_get_sensor_context`

Optional diagnostic tool:

- `vizor_get_status`

Do not expose raw arbitrary ROS publish/subscribe tools to the agent. That would be too much capability and too little semantics.

## `vizor_get_sensor_context` Output

Return one stable envelope:

```json
{
  "ok": true,
  "tool": "vizor_get_sensor_context",
  "source": "rosbridge",
  "rosbridge": {
    "connected": true,
    "host": "localhost",
    "port": 9090
  },
  "freshness": {
    "max_age_s": 2.0,
    "stale": false
  },
  "gaze": {
    "available": true,
    "target": "box_01",
    "raw_target": "dynamic_box_01",
    "age_s": 0.2,
    "stale": false,
    "source_topic": "/HOLO1_GazePoint"
  },
  "user": {
    "available": true,
    "position": {"x": 0.34, "y": -0.72, "z": 1.25},
    "orientation": {"x": 0, "y": 0, "z": 0, "w": 1},
    "frame": "robot_base",
    "raw_unity_pose": {
      "position": {"x": 1.147, "y": -0.513, "z": 0.771}
    },
    "age_s": 0.2,
    "stale": false,
    "source_topic": "/HOLO1_Transform"
  },
  "manual_target": {
    "available": false,
    "position": null,
    "orientation": null,
    "frame": "robot_base",
    "age_s": null,
    "stale": true,
    "source_topic": "/Robot/target_manual"
  },
  "calibration": {
    "name": "df2025_unity_to_robot",
    "offset_m": {"x": -0.173, "y": 0.051, "z": 0.103}
  }
}
```

If ROSBridge is disconnected, return `ok=false` with `retryable=true`, but still include any last-known cache with stale flags if available.

## Pipecat Integration

Create a new folder in the Pipecat server:

```text
pipecat-agent/server/user_sensing/
  __init__.py
  context.py
  mcp_bridge.py
```

### `user_sensing/context.py`

Owns `UserSensingContextStore`, separate from `RobotContextStore`.

It should render a small instruction block:

```text
User sensing context:
- This context is advisory only.
- Use fresh gaze/user/target data to resolve "this", "that", "there", or "near me".
- If relevant sensing data is missing or stale, ask a clarifying question instead of guessing.
- gaze target: ...
- user position: ...
- manual target: ...
```

### `user_sensing/mcp_bridge.py`

Create a read-only MCP bridge that connects to the Vizor MCP server and calls only `vizor_get_sensor_context`.

This bridge should not expose `function_tools()` to the model. The graph calls it as host-side context loading.

If we want to avoid duplicating the Streamable HTTP session helper from `robot_control.mcp_bridge`, extract that helper to a neutral module such as:

```text
pipecat-agent/server/mcp_client/streamable_http.py
```

### `LangGraphRobotAgent`

Rename the observe node from `observe_current_pose` to something like `observe_runtime_context`.

Before every `call_model`, run:

1. `moveit_get_current_pose` when robot state needs refresh, preserving current behavior.
2. `vizor_get_sensor_context` whenever user sensing is configured.

Important edge change: `repair_missing_action` should route back through `observe_runtime_context`, not directly to `call_model`, so repaired LLM calls also receive fresh user sensing.

`_instructions()` should become:

```python
return "\n\n".join(
    [
        SYSTEM_PROMPT,
        self._robot_context.render_instruction_block(),
        self._user_sensing_context.render_instruction_block(),
    ]
)
```

### Prompt Contract

Remove the current line that says there is no HoloLens, gaze target, world model, or user-position data.

Replace it with a user-sensing rule:

- User sensing may include HoloLens gaze, user pose, and manual target data.
- Treat it as advisory and time-sensitive.
- Use it only when fresh enough for the requested action.
- Ask for clarification when the user reference cannot be grounded safely.

## Configuration

Add an optional Vizor MCP URL, independent from MoveIt MCP:

```text
MCP_ROBOT_URL=http://127.0.0.1:8000/mcp
MCP_VIZOR_URL=http://127.0.0.1:8001/mcp
USER_SENSING_ENABLED=true
USER_SENSING_MAX_AGE_S=2.0
```

If `USER_SENSING_ENABLED=false` or no URL is configured, Pipecat should run exactly as it does today.

## Tests

MCP package tests:

- `dynamic_` prefix stripping for gaze targets
- Unity-to-robot transform
- missing-topic and stale-topic envelopes
- ROS cache thread safety with a fake transport
- `vizor_get_sensor_context` returns stable structured output

Pipecat tests:

- context store renders no-data, fresh-data, and stale-data blocks
- graph loads user sensing before initial model call
- graph loads user sensing after a robot tool call before the next model call
- repair path also refreshes user sensing before the next model call
- user sensing bridge failure degrades to stale/unavailable context, not a failed turn

Manual smoke:

1. Start ROSBridge and Vizor stack.
2. Start `vizor_mcp` on `http://127.0.0.1:8001/mcp`.
3. Call `vizor_get_sensor_context` from an MCP client.
4. Start Pipecat with `MCP_VIZOR_URL`.
5. Confirm process trace contains a `user_sensing.mcp.call_tool` span before each `agent.model_call`.

## Build Order

1. Implement `vizor_mcp` package with fake transport tests.
2. Add live ROS smoke script/manual command.
3. Add Pipecat `user_sensing` context and bridge.
4. Wire `LangChainAgentProcessor`, pipeline config, and `LangGraphRobotAgent`.
5. Update prompt contract.
6. Run unit tests and one live MCP smoke.
