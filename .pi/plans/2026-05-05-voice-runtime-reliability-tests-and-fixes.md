# Voice Runtime Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make voice turns reliable by reducing false wake carryover, preventing empty metrics rows, making uncertain completions honest, and supporting visible multi-waypoint Cartesian gestures.

**Architecture:** Keep fixes behind existing narrow modules: wake gating stays in `voice_runtime/wake_command.py` and runtime profile config, metrics lifecycle stays in `metrics.py`, and robot tool argument repair/normalization stays in `langgraph_robot_agent.py` plus `robot_control/mcp_bridge.py`. Work is split into mostly independent batches so parallel agents can implement without touching the same files.

**Tech Stack:** Python, pytest, Pipecat frames/observers, LangGraph robot agent, MoveIt MCP bridge.

---

## Current tracer tests already added in this session

These tests now exist and pass:

- `server/tests/test_metrics.py::test_observer_does_not_record_empty_turn_from_delayed_tts_audio_after_stop`
- `server/tests/test_metrics.py::test_observer_discards_empty_turn_closed_by_stale_bot_stop`
- `server/tests/test_langgraph_robot_agent.py::test_graph_stops_after_max_tool_turns`
- `server/tests/test_voice_runtime_agent_turn.py::test_agent_turn_emits_fallback_when_backend_yields_no_text`
- `server/tests/test_voice_runtime_profiles.py::test_bundled_streaming_profiles_keep_wake_prebuffer_short`
- `server/tests/test_langgraph_robot_agent.py::test_graph_repairs_missing_wave_waypoints_from_current_pose`
- `server/tests/test_robot_call_validation.py::test_accepts_high_level_cartesian_plan_and_execute_with_multiple_waypoints`
- `server/tests/test_robot_mcp_bridge.py::test_normalizes_cartesian_points_alias_before_mcp_call`

Validation run:

```bash
cd pipecat-agent/server
uv run pytest tests/test_metrics.py tests/test_langgraph_robot_agent.py tests/test_voice_runtime_agent_turn.py tests/test_robot_call_validation.py tests/test_robot_mcp_bridge.py tests/test_voice_runtime_profiles.py -q
# Expected: 82 passed
```

---

## Parallel execution layout

Run these as in-session subagents, not detached async:

### Batch A: parallel-safe

- Task 1: Metrics lifecycle hardening (`server/metrics.py`, `server/tests/test_metrics.py`)
- Task 2: Wake gate short-buffer and transcript cleanup (`server/voice_runtime/wake_command.py`, `server/runtime_profiles.toml`, wake tests)
- Task 3: Tool alias normalization (`server/robot_control/mcp_bridge.py`, bridge/validation tests)

### Batch B: run after Batch A

- Task 4: Gesture waypoint repair (`server/langgraph_robot_agent.py`, robot agent tests)
- Task 5: Pose observation loop latency (`server/langgraph_robot_agent.py`, robot agent tests)

Task 4 and Task 5 both touch `server/langgraph_robot_agent.py`; do not run them in the same worktree unless using isolated git worktrees and manual merge review.

---

### Task 1: Metrics lifecycle hardening

**Files:**
- Modify: `server/metrics.py`
- Modify: `server/tests/test_metrics.py`

- [ ] **Step 1: Add the delayed-TTS regression test**

Add to `server/tests/test_metrics.py`:

```python
@pytest.mark.asyncio
async def test_observer_does_not_record_empty_turn_from_delayed_tts_audio_after_stop(tmp_path: Path):
    path = tmp_path / "metrics.jsonl"
    recorder = VoiceMetricsRecorder(
        profile="hybrid_low_latency",
        category="benchmark_streaming",
        path=path,
        include_text=True,
    )
    observer = VoiceMetricsObserver(recorder)

    await observer.on_push_frame(_pushed(UserStartedSpeakingFrame()))
    await observer.on_push_frame(_pushed(UserStoppedSpeakingFrame()))
    await observer.on_push_frame(
        _pushed(TranscriptionFrame(text="move up", user_id="u", timestamp="t", finalized=True))
    )
    await observer.on_push_frame(_pushed(LLMTextFrame(text="Moved up.")))
    await observer.on_push_frame(_pushed(LLMFullResponseEndFrame()))
    await observer.on_push_frame(
        _pushed(TTSAudioRawFrame(audio=b"\0\0", sample_rate=16000, num_channels=1))
    )
    await observer.on_push_frame(_pushed(TTSStoppedFrame()))
    await observer.on_push_frame(
        _pushed(TTSAudioRawFrame(audio=b"\0\0", sample_rate=16000, num_channels=1))
    )
    await observer.on_push_frame(_pushed(BotStoppedSpeakingFrame()))

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["transcript"] == "move up"
    assert records[0]["response"] == "Moved up."
```

- [ ] **Step 2: Run it and verify RED**

```bash
cd pipecat-agent/server
uv run pytest tests/test_metrics.py::test_observer_does_not_record_empty_turn_from_delayed_tts_audio_after_stop -q
# Expected before fix: FAIL with two records, second transcript=''
```

- [ ] **Step 3: Prevent TTS audio from creating orphan turns**

In `server/metrics.py`, change `_mark_tts_first_audio()` to:

```python
    def _mark_tts_first_audio(self) -> None:
        if self._tts_first_audio_marked:
            return
        if self._current_turn() is None:
            return
        self._mark("tts_first_audio")
        self._tts_first_audio_marked = True
```

- [ ] **Step 4: Add the stale bot-stop empty-turn test**

Add to `server/tests/test_metrics.py`:

```python
@pytest.mark.asyncio
async def test_observer_discards_empty_turn_closed_by_stale_bot_stop(tmp_path: Path):
    path = tmp_path / "metrics.jsonl"
    recorder = VoiceMetricsRecorder(
        profile="hybrid_low_latency",
        category="benchmark_streaming",
        path=path,
        include_text=True,
    )
    observer = VoiceMetricsObserver(recorder)

    await observer.on_push_frame(_pushed(UserStartedSpeakingFrame()))
    await observer.on_push_frame(_pushed(BotStoppedSpeakingFrame()))

    assert not path.exists()
```

- [ ] **Step 5: Run it and verify RED**

```bash
cd pipecat-agent/server
uv run pytest tests/test_metrics.py::test_observer_discards_empty_turn_closed_by_stale_bot_stop -q
# Expected before fix: FAIL because an empty turn is written
```

- [ ] **Step 6: Add discard support and skip empty turns**

In `VoiceMetricsRecorder`, add:

```python
    def discard_turn(self, turn_id: str) -> None:
        self._turns.pop(turn_id, None)
```

In `VoiceMetricsObserver._finish_turn()`, use:

```python
    def _finish_turn(self) -> None:
        if self._current_turn_id is None:
            return
        turn = self._current_turn()
        if turn is not None and not turn.transcript and not turn.response:
            self._recorder.discard_turn(self._current_turn_id)
        else:
            self._mark("tts_done")
            self._recorder.finish_turn(self._current_turn_id)
        self._current_turn_id = None
        self._wake_marked = False
        self._stt_marked = False
        self._tts_first_audio_marked = False
```

- [ ] **Step 7: Verify Task 1**

```bash
cd pipecat-agent/server
uv run pytest tests/test_metrics.py -q
# Expected: PASS
```

---

### Task 2: Wake gate short-buffer and transcript cleanup

**Files:**
- Modify: `server/runtime_profiles.toml`
- Modify: `server/voice_runtime/wake_command.py`
- Modify: `server/tests/test_voice_runtime_profiles.py`
- Modify: `server/tests/test_voice_runtime_wake_command.py`

- [ ] **Step 1: Add bundled prebuffer test**

Add to `server/tests/test_voice_runtime_profiles.py`:

```python
def test_bundled_streaming_profiles_keep_wake_prebuffer_short() -> None:
    server_dir = Path(__file__).resolve().parents[1]

    profile = load_runtime_profile(
        profiles_path=default_profiles_path(server_dir),
        server_dir=server_dir,
        profile_name="hybrid_low_latency",
    )

    assert profile.wake.pre_buffer_s <= 0.5
```

Also import `default_profiles_path` from `voice_runtime.profiles`.

- [ ] **Step 2: Run it and verify RED**

```bash
cd pipecat-agent/server
uv run pytest tests/test_voice_runtime_profiles.py::test_bundled_streaming_profiles_keep_wake_prebuffer_short -q
# Expected before config change: FAIL because pre_buffer_s is 1.5
```

- [ ] **Step 3: Shorten bundled wake prebuffers**

In `server/runtime_profiles.toml`, set every wake-enabled profile to:

```toml
pre_buffer_s = 0.5
```

- [ ] **Step 4: Verify Task 2**

```bash
cd pipecat-agent/server
uv run pytest tests/test_voice_runtime_profiles.py::test_bundled_streaming_profiles_keep_wake_prebuffer_short -q
# Expected: PASS
```

Do not broaden wake-transcript cleanup in this task. The intended fix is to reduce wake audio reaching STT by keeping the prebuffer short.

---

### Task 3: Tool alias normalization

**Files:**
- Modify: `server/robot_control/mcp_bridge.py`
- Modify: `server/tests/test_robot_mcp_bridge.py`
- Modify: `server/tests/test_robot_call_validation.py`

- [ ] **Step 1: Add high-level multi-waypoint validation test**

Add to `server/tests/test_robot_call_validation.py`:

```python
def test_accepts_high_level_cartesian_plan_and_execute_with_multiple_waypoints() -> None:
    validate_robot_tool_call(
        "moveit_plan_and_execute_cartesian_motion",
        {
            "robot_name": "UR10",
            "waypoints": [
                VALID_POSE,
                {**VALID_POSE, "position": {"x": 0.57, "y": 0.49, "z": 0.67}},
                {**VALID_POSE, "position": {"x": 0.57, "y": 0.29, "z": 0.67}},
                {**VALID_POSE, "position": {"x": 0.57, "y": 0.39, "z": 0.67}},
            ],
            "timeout_s": 10.0,
        },
    )
```

- [ ] **Step 2: Run validation test**

```bash
cd pipecat-agent/server
uv run pytest tests/test_robot_call_validation.py::test_accepts_high_level_cartesian_plan_and_execute_with_multiple_waypoints -q
# Expected: PASS if existing validation already supports list waypoints
```

- [ ] **Step 3: Add points alias bridge test**

Add to `server/tests/test_robot_mcp_bridge.py`:

```python
@pytest.mark.asyncio
async def test_normalizes_cartesian_points_alias_before_mcp_call():
    server = FakeLegacyWorkflowServer()
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=server)
    await bridge.connect()
    points = [
        {
            "position": {"x": 0.1, "y": 0.2, "z": 0.3},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
        {
            "position": {"x": 0.1, "y": 0.3, "z": 0.38},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
    ]

    await bridge.call_tool(
        "moveit_plan_and_execute_cartesian_motion",
        {"robot_name": "UR10", "points": points, "timeout_s": 10.0},
    )

    assert server.called == [
        (
            "plan_and_execute_cartesian_motion",
            {"robot_name": "UR10", "waypoints": points, "timeout_s": 10.0},
        )
    ]
```

- [ ] **Step 4: Run it and verify RED**

```bash
cd pipecat-agent/server
uv run pytest tests/test_robot_mcp_bridge.py::test_normalizes_cartesian_points_alias_before_mcp_call -q
# Expected before fix: FAIL because validation rejects points or no MCP call occurs
```

- [ ] **Step 5: Normalize alias before validation and MCP call**

In `server/robot_control/mcp_bridge.py`, update `call_tool()` to validate/call with normalized arguments:

```python
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        normalized_arguments = _normalize_agent_arguments(name, arguments)
        try:
            validate_robot_tool_call(name, normalized_arguments)
        except RobotCallValidationError as exc:
            return _serialize_validation_failure(exc)

        backing_tool_name = self._backing_tool_names.get(name)
        if backing_tool_name is None:
            raise RobotMCPError(f"Tool is not allowed: {name}")
        result = await self._server.call_tool(backing_tool_name, normalized_arguments)
        return _serialize_tool_result(result)
```

Add helper:

```python
def _normalize_agent_arguments(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name not in {
        "moveit_plan_cartesian_motion",
        "moveit_plan_and_execute_cartesian_motion",
    }:
        return arguments
    if "waypoints" in arguments:
        return arguments
    points = arguments.get("points", arguments.get("positions"))
    if points is None:
        return arguments
    normalized = {key: value for key, value in arguments.items() if key not in {"points", "positions"}}
    normalized["waypoints"] = points
    return normalized
```

- [ ] **Step 6: Verify Task 3**

```bash
cd pipecat-agent/server
uv run pytest tests/test_robot_mcp_bridge.py tests/test_robot_call_validation.py -q
# Expected: PASS
```

---

### Task 4: Gesture waypoint repair

**Files:**
- Modify: `server/langgraph_robot_agent.py`
- Modify: `server/tests/test_langgraph_robot_agent.py`

- [ ] **Step 1: Add wave repair test**

Add to `server/tests/test_langgraph_robot_agent.py`:

```python
@pytest.mark.asyncio
async def test_graph_repairs_missing_wave_waypoints_from_current_pose() -> None:
    tool = tool_call(
        "moveit_plan_and_execute_cartesian_motion",
        arguments={"robot_name": "UR10", "plan_name": "wave", "timeout_s": 10},
    )
    fixture = make_graph(
        [
            CodexResponseResult(
                tool_calls=[tool],
                output_items=[
                    output_item(
                        "moveit_plan_and_execute_cartesian_motion",
                        arguments=tool.arguments,
                    )
                ],
            ),
            CodexResponseResult(text="Waved."),
        ]
    )

    await fixture.graph.run_turn(turn("wave to me"))

    waypoints = fixture.bridge.calls[1][1]["waypoints"]
    assert len(waypoints) >= 4
    assert {waypoint["position"]["y"] for waypoint in waypoints} >= {0.1, 0.3}
    assert all(waypoint["position"]["z"] >= 0.38 for waypoint in waypoints)
    assert all(
        waypoint["orientation"] == {"x": 0.0, "y": -0.7071, "z": -0.7071, "w": 0.0}
        for waypoint in waypoints
    )
```

- [ ] **Step 2: Run it and verify RED**

```bash
cd pipecat-agent/server
uv run pytest tests/test_langgraph_robot_agent.py::test_graph_repairs_missing_wave_waypoints_from_current_pose -q
# Expected before fix: FAIL with KeyError: 'waypoints'
```

- [ ] **Step 3: Add pose helper and wave waypoint repair**

In `server/langgraph_robot_agent.py`, add helper:

```python
def _pose_at(
    x: float, y: float, z: float, orientation: dict[str, Any] | None
) -> dict[str, Any]:
    pose: dict[str, Any] = {
        "position": {"x": round(x, 4), "y": round(y, 4), "z": round(z, 4)}
    }
    if orientation is not None:
        pose["orientation"] = dict(orientation)
    return pose
```

Refactor `_relative_target_pose()` to use `_latest_pose_components()` and `_pose_at()`, then add:

```python
    def _cartesian_gesture_waypoints(self, user_text: str) -> list[dict[str, Any]] | None:
        words = set(re.findall(r"[a-zA-Z']+", user_text.lower()))
        if "wave" not in words and "waving" not in words:
            return None
        pose_components = self._latest_pose_components()
        if pose_components is None:
            return None
        x, y, z, orientation = pose_components
        lifted_z = z + 0.08
        return [
            _pose_at(x, y, lifted_z, orientation),
            _pose_at(x, y + 0.10, lifted_z, orientation),
            _pose_at(x, y - 0.10, lifted_z, orientation),
            _pose_at(x, y, lifted_z, orientation),
        ]
```

Call it after relative target repair fails:

```python
            gesture_waypoints = self._cartesian_gesture_waypoints(user_text)
            if gesture_waypoints is not None:
                return {**arguments, "waypoints": gesture_waypoints}
```

- [ ] **Step 4: Verify Task 4**

```bash
cd pipecat-agent/server
uv run pytest tests/test_langgraph_robot_agent.py::test_graph_repairs_missing_wave_waypoints_from_current_pose tests/test_langgraph_robot_agent.py::test_graph_repairs_cartesian_waypoints_from_current_pose -q
# Expected: PASS
```

---

### Task 5: Pose observation loop latency

**Files:**
- Modify: `server/langgraph_robot_agent.py`
- Modify: `server/tests/test_langgraph_robot_agent.py`

- [ ] **Step 1: Add a test that a pure observation tool loop does not refresh pose before every Codex retry**

Add to `server/tests/test_langgraph_robot_agent.py`:

```python
@pytest.mark.asyncio
async def test_graph_does_not_refresh_pose_before_every_codex_retry_for_observation_loop() -> None:
    pose = tool_call("moveit_get_current_pose")
    fixture = make_graph(
        [
            CodexResponseResult(
                tool_calls=[pose],
                output_items=[output_item("moveit_get_current_pose")],
            ),
            CodexResponseResult(text="Robot pose is ready."),
        ]
    )

    text = await fixture.graph.run_turn(turn("where is the pose?"))

    assert text == "Robot pose is ready."
    assert fixture.bridge.calls == [
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
    ]
```

- [ ] **Step 2: Run it and verify RED**

```bash
cd pipecat-agent/server
uv run pytest tests/test_langgraph_robot_agent.py::test_graph_does_not_refresh_pose_before_every_codex_retry_for_observation_loop -q
# Expected before fix: FAIL because current behavior calls pose three times
```

- [ ] **Step 3: Add per-turn observation freshness flag**

In `RobotAgentState`, add:

```python
    observed_this_turn: bool
```

In `run_turn()` initial state, set:

```python
            "observed_this_turn": False,
```

In `_observe_current_pose()`, skip if already observed in the current graph turn:

```python
    async def _observe_current_pose(self, state: RobotAgentState) -> dict[str, Any]:
        tools = self._tool_bridge.function_tools()
        if state.get("observed_this_turn"):
            return {"tools": tools}
        observe_tool_name = _first_available_tool(tools, OBSERVE_TOOL_NAMES)
        if observe_tool_name is None:
            return {"tools": tools}
        logger.info(f"Refreshing robot observation before Codex request with {observe_tool_name}")
        await self._execute_tool(observe_tool_name, {"robot_name": VIZOR_ROBOT_NAME})
        return {"tools": tools, "observed_this_turn": True}
```

- [ ] **Step 4: Update existing expectation**

Change `test_graph_sends_tool_output_back_to_codex` or replace it with the new test so it expects two pose calls, not three.

- [ ] **Step 5: Verify Task 5**

```bash
cd pipecat-agent/server
uv run pytest tests/test_langgraph_robot_agent.py -q
# Expected: PASS
```

---

## Final verification

Run:

```bash
cd pipecat-agent/server
uv run pytest -q
```

Expected: all tests pass.

Then run the voice scenario manually and check logs:

- No repeated empty metrics turns after assistant speech.
- No `I completed the action but have nothing to report.` response.
- Wake-gated commands use a shorter prebuffer.
- `wave to me` produces a visible multi-waypoint Cartesian action or a clear verified failure.
