# Process Trace Design

## Goal

Add an always-on, lightweight `process_trace` module for reusable voice robot projects.

The module records what happened across the full spoken-command path:

```text
wake -> speech capture -> STT -> Agent Turn -> LangGraph -> model call
  -> Task Policy -> Robot Call Validation -> MCP tool call -> TTS
```

The first version focuses on tracing only. Visualization and replay are follow-up work that reads the trace JSONL.

## Decisions

- Module name: `process_trace`.
- V1 tracing is always enabled by default.
- V1 records semantic spans and events, not every Pipecat frame.
- V1 writes local append-only JSONL.
- V1 stores full transcript, assistant text, tool arguments, and tool outputs by default.
- V1 never records raw audio, API keys, auth material, secrets, or environment dumps.
- V1 uses a custom local core, not LangSmith, Langfuse, OpenTelemetry, or a collector as the source of truth.
- V1 record shape is OpenTelemetry-shaped so later exporters can mirror traces to OpenTelemetry-compatible tools.

## Architecture

`process_trace` is a cross-cutting Module. It observes and persists runtime activity, but it does not own runtime behavior.

It must not reorder the Pipecat pipeline, control LangGraph, choose robot tools, validate robot calls, or call MCP. Existing Modules keep their ownership:

```text
Voice Runtime -> emits wake, speech, STT, Agent Turn, and TTS spans
Agent Control -> emits graph turn, LangGraph node, and model-call spans
Robot Control -> emits policy, validation, Robot Context, and MCP spans
process_trace -> correlates and persists spans/events
```

The deep Module interface is small:

```python
tracer.start_session(profile=..., category=...)
tracer.start_turn(input_text=...)
with tracer.span("agent.model_call", attributes={...}):
    ...
tracer.event("robot.context_update", attributes={...})
```

The Implementation owns IDs, parentage, timestamps, JSON safety, writer failure behavior, and schema versioning.

## Components

Target package:

```text
server/process_trace/
  __init__.py
  trace.py      # ProcessTracer interface, span context manager, no-op tracer
  records.py    # schema helpers
  jsonl.py      # append-only writer
  context.py    # contextvars for current session, turn, trace, and parent span
```

Core modules should stay pure. They should not import Pipecat, LangGraph, MCP, provider factories, Agent Control, Voice Runtime, or Robot Control.

Adapters live at call sites or in thin adapter modules. `pipeline_builder.py` and `agent_processor_factory.py` inject a concrete tracer into runtime adapters. Tests can use a memory writer or `NoopProcessTracer`.

## Trace Model

One bot run creates a session. One spoken command creates a turn. Spans nest under the active turn:

```text
session
  turn
    voice.wake
    voice.speech_capture
    voice.stt
    voice.agent_turn
      agent.backend_turn
      agent.graph_turn
        agent.langgraph_node.observe_current_pose
          robot.task_policy
          robot.call_validation
          robot.mcp.call_tool
        agent.langgraph_node.call_model
          agent.model_call
        agent.langgraph_node.execute_robot_tool
          robot.task_policy
          robot.call_validation
          robot.mcp.call_tool
    voice.tts
```

Records are completed span records plus instant event records. V1 does not need separate span-start and span-end rows.

Required fields:

```text
schema_version
record_type              # span or event
trace_id
span_id
parent_span_id
session_id
turn_id
name
module
started_at_unix_ns
ended_at_unix_ns
duration_ms
status                   # ok, error, cancelled
attributes
events
```

Example:

```json
{
  "schema_version": 1,
  "record_type": "span",
  "trace_id": "tr-20260507-abc123",
  "span_id": "sp-12",
  "parent_span_id": "sp-9",
  "session_id": "session-20260507-abc123",
  "turn_id": "turn-4",
  "name": "robot.mcp.call_tool",
  "module": "robot_control",
  "started_at_unix_ns": 1778170000123000000,
  "ended_at_unix_ns": 1778170000965100000,
  "duration_ms": 842.1,
  "status": "ok",
  "attributes": {
    "tool.name": "moveit_get_current_pose",
    "tool.arguments": {"robot_name": "UR10"},
    "tool.result": {"ok": true}
  },
  "events": []
}
```

Use OpenTelemetry-style vocabulary where it fits cheaply, such as `gen_ai.*` for model calls and `tool.*` or `mcp.*` for tool execution. Do not depend on unstable OpenTelemetry GenAI or MCP semantic conventions in v1.

## Library Decision

Research checked OpenTelemetry, Pipecat tracing, LangSmith, Langfuse, LangGraph streaming/debug events, Chrome Trace, Perfetto, speedscope, Eliot, Phoenix, Logfire, structlog, and JSON logging libraries.

Decision: use a custom local JSONL core for v1, with OpenTelemetry-shaped records.

Reasoning:

- Pipecat has built-in OpenTelemetry tracing for conversation, turn, STT, LLM, and TTS spans, but it expects tracing extras and an exporter/collector path.
- LangSmith and Langfuse are useful LLM observability products, but they are service-backed and not the right always-on local source of truth.
- LangGraph streaming/debug events can supplement traces, but explicit spans in node/tool/policy code are more reliable.
- Chrome Trace and Perfetto are good visualization export targets, not the canonical runtime log.
- Structlog, python-json-logger, and Loguru are logging plumbing, not span schemas.

Keep optional future adapters possible:

```text
process_trace JSONL -> OpenTelemetry exporter
process_trace JSONL -> Langfuse/LangSmith/Phoenix/Logfire
process_trace JSONL -> Chrome Trace or Perfetto visualization
```

Primary references:

- [Pipecat OpenTelemetry tracing](https://docs.pipecat.ai/api-reference/server/utilities/opentelemetry)
- [Pipecat observers](https://docs.pipecat.ai/api-reference/server/utilities/observers/observer-pattern)
- [OpenTelemetry Python instrumentation](https://opentelemetry.io/docs/languages/python/instrumentation/)
- [OpenTelemetry Python exporters](https://opentelemetry.io/docs/languages/python/exporters/)
- [LangGraph streaming](https://docs.langchain.com/oss/python/langgraph/streaming)
- [LangSmith LangGraph tracing](https://docs.langchain.com/langsmith/trace-with-langgraph)
- [Langfuse OpenTelemetry](https://langfuse.com/integrations/native/opentelemetry)

## Runtime Configuration

Tracing is enabled by default in runtime profiles:

```toml
[profiles.hybrid_low_latency.process_trace]
enabled = true
path = "logs/process_trace.jsonl"
include_text = true
include_tool_payloads = true
```

The app composition root creates the concrete tracer from profile config. Profile parsing belongs with Runtime Profile parsing, but concrete JSONL writer construction is app configuration.

## Integration Points

Voice Runtime:

- `WakeDetectedFrame`: `voice.wake`
- `UserStartedSpeakingFrame` and `UserStoppedSpeakingFrame`: `voice.speech_capture`
- finalized `TranscriptionFrame`: `voice.stt`
- `AgentTurnProcessor`: `voice.agent_turn`
- `TTSAudioRawFrame` and `TTSStoppedFrame`: `voice.tts`

Agent Control:

- `LangChainAgentProcessor.run_turn`: `agent.backend_turn`
- `LangGraphRobotAgent.run_turn`: `agent.graph_turn`
- each LangGraph node method: `agent.langgraph_node`
- `model.ainvoke`: `agent.model_call`

Robot Control:

- `validate_task_step` call site: `robot.task_policy`
- `validate_robot_tool_call` call site: `robot.call_validation`
- `RobotMCPBridge.connect`: `robot.mcp.connect`
- `RobotMCPBridge.list_tools` path inside connect: `robot.mcp.list_tools`
- `RobotMCPBridge.call_tool`: `robot.mcp.call_tool`
- `RobotContextStore.update_from_tool_result`: `robot.context_update`

Because the project supports Python 3.10, do not rely only on implicit async context propagation through LangChain/LangGraph internals. Pass trace metadata through `RunnableConfig` or explicit arguments where needed.

## Error Handling

Tracing must never break a robot run.

- JSONL write failure logs one warning and disables the writer for the current session.
- A span body exception records `status="error"` and re-raises the original exception.
- Async cancellation records `status="cancelled"` and re-raises cancellation.
- Non-JSON-safe payloads are converted to compact strings.
- Payload recording must reject obvious secret/auth keys.
- No raw audio is written.

## Testing

Core tests:

- create session, turn, trace, span, and parent IDs
- nest spans with `contextvars`
- write valid JSONL
- record errors and cancellations
- disable writer after write failure
- serialize non-JSON-safe payloads safely

Integration tests:

- `AgentTurnProcessor` emits an Agent Turn span around a fake backend call
- `LangGraphRobotAgent` emits graph, node, model, and tool spans with fake model and bridge
- `RobotMCPBridge` emits validation and MCP spans with a fake server
- `pipeline_builder.py` wires a tracer when profile tracing is enabled

Structural tests:

- pure `process_trace` core modules do not import Pipecat, LangGraph, MCP, providers, Voice Runtime, Agent Control, or Robot Control
- Voice Runtime still does not import Agent Control or Robot Control
- Robot Control still does not import Voice Runtime or Agent Control

## Reuse

Future talking robot projects should be able to reuse the core unchanged:

```text
process_trace core stays
Pipecat adapter may change
Agent orchestration adapter may change
Robot/tool adapter may change
visualization/exporter may change
```

The deletion test should pass: if `process_trace` is removed, correlation, span timing, JSON safety, writer failure handling, and reusable trace schema would reappear across many callers.

## Out Of Scope

- HTML visualization
- Perfetto or Chrome Trace export
- OpenTelemetry exporter
- LangSmith or Langfuse integration
- raw audio logging
- durable database storage
- cloud upload
- automatic performance diagnosis
- changing robot safety behavior

