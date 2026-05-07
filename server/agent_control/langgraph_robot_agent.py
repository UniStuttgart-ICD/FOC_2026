"""LangGraph orchestration for robot agent turns."""

from __future__ import annotations

import asyncio
import inspect
import json
import operator
import uuid
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from loguru import logger

from agent_control.prompts import SYSTEM_PROMPT
from process_trace import NoopProcessTracer, ProcessTracer
from robot_control.call_validation import (
    RobotCallValidationError,
    executable_plan_name,
    structured_robot_call_error,
)
from robot_control.context import RobotContextStore
from robot_control.mcp_bridge import RobotMCPError
from robot_control.task_policy import structured_task_policy_error, validate_task_step
from voice_runtime.agent_turn import AgentTurnInput
from voice_runtime.timing import elapsed_ms_since, monotonic_s

MAX_AGENT_TOOL_TURNS = 6
VIZOR_ROBOT_NAME = "UR10"
PLAN_TOOL_NAMES = {"moveit_plan_free_motion", "moveit_plan_cartesian_motion"}
ACTION_TOOL_NAMES = {
    "moveit_plan_free_motion",
    "moveit_plan_cartesian_motion",
    "moveit_plan_and_execute_free_motion",
    "moveit_plan_and_execute_cartesian_motion",
    "moveit_execute_plan",
    "moveit_open_gripper",
    "moveit_close_gripper",
    "moveit_attach_object",
}
OBSERVE_TOOL_NAMES = ("moveit_get_current_pose",)
NO_TEXT_RESPONSE = "I could not confirm that the action completed."
MAX_MISSING_ACTION_REPAIRS = 1
ROBOT_ACTION_TERMS = (
    "move",
    "go",
    "raise",
    "lower",
    "lift",
    "drop",
    "wave",
    "draw",
    "point",
    "gesture",
    "open",
    "close",
    "grab",
    "release",
)
FUTURE_PROMISE_TERMS = (
    "i'll",
    "i’ll",
    "i will",
    "i’m",
    "i’m going to",
    "i am going to",
    "let me",
    "first, then",
    "then i'll",
    "then i’ll",
    "then make",
    "then do",
)
ProcessTracerLike = ProcessTracer | NoopProcessTracer


class RobotAgentState(TypedDict):
    user_text: str
    messages: Annotated[list[BaseMessage], operator.add]
    tools: list[dict[str, Any]]
    tool_turns: int
    observed_this_turn: bool
    needs_action_tool: bool
    action_tool_ran: bool
    missing_action_repairs: int
    final_text: str
    error_text: str | None


@dataclass(frozen=True)
class SupportedToolStep:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class SupportedAction:
    steps: list[SupportedToolStep]
    success_text: str


class LangGraphRobotAgent:
    """Runs a robot dialogue turn through a LangGraph state machine."""

    def __init__(
        self,
        *,
        model: Any,
        tool_bridge: Any,
        robot_context: RobotContextStore,
        thread_id: str | None = None,
        tracer: ProcessTracerLike | None = None,
    ) -> None:
        self._model = model
        self._tool_bridge = tool_bridge
        self._robot_context = robot_context
        self._thread_id = thread_id or f"robot-agent-{uuid.uuid4()}"
        self._tracer = tracer or NoopProcessTracer()
        self._latest_state: dict[str, Any] | None = None
        self._graph = self._compile_graph()

    async def run_turn(self, turn: AgentTurnInput) -> str:
        state: RobotAgentState = {
            "user_text": turn.user_text,
            "messages": _messages_from_turn(turn),
            "tools": [],
            "tool_turns": 0,
            "observed_this_turn": False,
            "needs_action_tool": _looks_like_robot_action_request(turn.user_text),
            "action_tool_ran": False,
            "missing_action_repairs": 0,
            "final_text": "",
            "error_text": None,
        }
        attributes: dict[str, Any] = {
            "thread_id": self._thread_id,
            "message_count": len(state["messages"]),
        }
        if self._tracer.options.include_text:
            attributes["user_text"] = turn.user_text
        async with self._tracer.span(
            "agent.graph_turn",
            "agent_control",
            attributes=attributes,
        ):
            result = await self._graph.ainvoke(
                state,
                {"configurable": {"thread_id": self._thread_id}},
            )
        self._latest_state = result
        return str(result.get("final_text") or NO_TEXT_RESPONSE)

    def latest_state(self) -> dict[str, Any]:
        return dict(self._latest_state or {})

    def _compile_graph(self):
        builder = StateGraph(RobotAgentState)
        builder.add_node(
            "observe_current_pose",
            self._traced_node("observe_current_pose", self._observe_current_pose),
        )
        builder.add_node("call_model", self._traced_node("call_model", self._call_model))
        builder.add_node(
            "execute_robot_tool",
            self._traced_node("execute_robot_tool", self._execute_robot_tool),
        )
        builder.add_node(
            "repair_missing_action",
            self._traced_node("repair_missing_action", self._repair_missing_action),
        )
        builder.add_node(
            "execute_supported_action",
            self._traced_node("execute_supported_action", self._execute_supported_action),
        )
        builder.add_node(
            "stop_after_tool_limit",
            self._traced_node("stop_after_tool_limit", self._stop_after_tool_limit),
        )
        builder.add_node(
            "final_response",
            self._traced_node("final_response", self._final_response),
        )
        builder.add_edge(START, "observe_current_pose")
        builder.add_edge("observe_current_pose", "call_model")
        builder.add_conditional_edges("call_model", self._route_after_model)
        builder.add_edge("execute_robot_tool", "observe_current_pose")
        builder.add_edge("repair_missing_action", "call_model")
        builder.add_edge("execute_supported_action", END)
        builder.add_edge("stop_after_tool_limit", END)
        builder.add_edge("final_response", END)
        return builder.compile(checkpointer=InMemorySaver())

    def _traced_node(self, node_name: str, node_fn: Any) -> Any:
        async def wrapped(state: RobotAgentState) -> dict[str, Any]:
            async with self._tracer.span(
                f"agent.langgraph_node.{node_name}",
                "agent_control",
                attributes={
                    "node.name": node_name,
                    "tool_turns": state.get("tool_turns", 0),
                    "message_count": len(state.get("messages", [])),
                },
            ):
                result = node_fn(state)
                if inspect.isawaitable(result):
                    result = await result
                return result

        return wrapped

    async def _observe_current_pose(self, state: RobotAgentState) -> dict[str, Any]:
        tools = self._tool_bridge.function_tools()
        if state.get("observed_this_turn"):
            return {"tools": tools}
        observe_tool_name = _first_available_tool(tools, OBSERVE_TOOL_NAMES)
        if observe_tool_name is None:
            return {"tools": tools}
        logger.info("Refreshing robot observation before Codex request with {}", observe_tool_name)
        _, observed = await self._execute_observation_tool(
            observe_tool_name, {"robot_name": VIZOR_ROBOT_NAME}
        )
        return {"tools": tools, "observed_this_turn": observed}

    async def _call_model(self, state: RobotAgentState) -> dict[str, Any]:
        tools = state["tools"] or self._tool_bridge.function_tools()
        model = self._model.bind_tools(
            _tools_for_model_binding(tools),
            tool_choice=_tool_choice_for_state(state),
        )
        system = SystemMessage(content=self._instructions())
        messages = [system, *state["messages"]]
        trace_attributes: dict[str, Any] = {
            "tool_turns": state["tool_turns"],
            "message_count": len(messages),
            "tool_count": len(tools),
        }
        started = monotonic_s()
        logger.info(
            "LangChain request start tool_turns={} messages={} tools={}",
            state["tool_turns"],
            len(messages),
            len(tools),
        )
        async with self._tracer.span(
            "agent.model_call",
            "agent_control",
            attributes=trace_attributes,
        ):
            try:
                message = await model.ainvoke(messages)
            except asyncio.CancelledError:
                logger.warning(
                    "LangChain request cancelled elapsed_ms={} tool_turns={} messages={} tools={}",
                    elapsed_ms_since(started),
                    state["tool_turns"],
                    len(messages),
                    len(tools),
                )
                raise
            except Exception as exc:
                logger.exception(
                    "LangChain request failed elapsed_ms={} tool_turns={} messages={} tools={} error={}",
                    elapsed_ms_since(started),
                    state["tool_turns"],
                    len(messages),
                    len(tools),
                    exc,
                )
                raise
            else:
                tool_calls = getattr(message, "tool_calls", []) or []
                tool_call_names = [str(call.get("name") or "") for call in tool_calls]
                trace_attributes["tool_call_count"] = len(tool_calls)
                trace_attributes["tool_call_names"] = tool_call_names
                trace_attributes["text_length"] = len(_message_text(message))
                logger.info(
                    "LangChain request end elapsed_ms={} tool_calls={} text_len={}",
                    elapsed_ms_since(started),
                    tool_call_names,
                    trace_attributes["text_length"],
                )
        return {"messages": [message], "tools": tools}

    def _route_after_model(
        self, state: RobotAgentState
    ) -> Literal[
        "execute_robot_tool",
        "repair_missing_action",
        "execute_supported_action",
        "stop_after_tool_limit",
        "final_response",
    ]:
        if state["error_text"]:
            return "final_response"
        last = _last_ai_message(state["messages"])
        if last is None:
            return "final_response"
        if last.tool_calls:
            if state["tool_turns"] >= MAX_AGENT_TOOL_TURNS:
                return "stop_after_tool_limit"
            return "execute_robot_tool"
        if _should_repair_missing_action(state, last):
            return "repair_missing_action"
        if _should_execute_supported_action_fallback(state, last, self._robot_context):
            return "execute_supported_action"
        return "final_response"

    def _repair_missing_action(self, state: RobotAgentState) -> dict[str, Any]:
        return {
            "messages": [
                HumanMessage(
                    content=(
                        "The previous response described a future robot action but did not call "
                        "a MoveIt action tool. For this movement request, call exactly one "
                        "available MoveIt action tool now, or explain a concrete blocker if no "
                        "safe tool call is possible. Do not say you will do it later."
                    )
                )
            ],
            "missing_action_repairs": state["missing_action_repairs"] + 1,
        }

    async def _execute_supported_action(self, state: RobotAgentState) -> dict[str, Any]:
        action = _supported_action_from_text(
            state["user_text"], self._robot_context.latest_tcp_pose()
        )
        if action is None:
            return {"final_text": NO_TEXT_RESPONSE}

        self._tracer.event(
            "agent.supported_action_fallback",
            "agent_control",
            attributes={
                "step_count": len(action.steps),
                "tool.names": [step.name for step in action.steps],
            },
        )
        output = ""
        execution_verified = True
        for step in action.steps:
            started = monotonic_s()
            logger.info("Robot supported-action fallback start name={}", step.name)
            output = await self._execute_tool(step.name, step.arguments)
            logger.info(
                "Robot supported-action fallback end name={} elapsed_ms={}",
                step.name,
                elapsed_ms_since(started),
            )
            if not _output_has_verified_execution(output):
                execution_verified = False
                break

        observed_this_turn = False
        observe_tool_name = _first_available_tool(
            state["tools"] or self._tool_bridge.function_tools(), OBSERVE_TOOL_NAMES
        )
        if observe_tool_name is not None:
            _, observed_this_turn = await self._execute_observation_tool(
                observe_tool_name, {"robot_name": VIZOR_ROBOT_NAME}
            )

        final_text = (
            action.success_text if execution_verified else _execution_failure_text(output)
        )
        return {
            "final_text": final_text,
            "action_tool_ran": True,
            "observed_this_turn": observed_this_turn,
        }

    async def _execute_robot_tool(self, state: RobotAgentState) -> dict[str, Any]:
        last = _last_ai_message(state["messages"])
        if last is None:
            return {"messages": [], "tool_turns": state["tool_turns"]}

        tool_messages: list[ToolMessage] = []
        observed_this_turn = state["observed_this_turn"]
        action_tool_ran = state["action_tool_ran"]
        for index, tool_call in enumerate(last.tool_calls):
            name = str(tool_call.get("name") or "")
            call_args = tool_call.get("args")
            args = call_args if isinstance(call_args, dict) else {}
            call_id = str(tool_call.get("id") or "")
            if index > 0:
                self._tracer.event(
                    "agent.extra_tool_call_rejected",
                    "agent_control",
                    attributes={
                        "tool.name": name,
                        "tool_call_id": call_id,
                        "tool_call_index": index,
                    },
                )
                tool_messages.append(
                    ToolMessage(content=_one_tool_at_a_time_error(name), tool_call_id=call_id)
                )
                continue

            started = monotonic_s()
            logger.info("Robot tool start name={} call_id={}", name, call_id)
            if name in OBSERVE_TOOL_NAMES:
                output, observed_this_turn = await self._execute_observation_tool(name, dict(args))
            else:
                output = await self._execute_tool(name, dict(args))
                action_tool_ran = action_tool_ran or name in ACTION_TOOL_NAMES
                observed_this_turn = False
            logger.info(
                "Robot tool end name={} call_id={} elapsed_ms={}",
                name,
                call_id,
                elapsed_ms_since(started),
            )
            tool_messages.append(ToolMessage(content=output, tool_call_id=call_id))

        return {
            "messages": tool_messages,
            "tool_turns": state["tool_turns"] + 1,
            "observed_this_turn": observed_this_turn,
            "action_tool_ran": action_tool_ran,
        }

    def _stop_after_tool_limit(self, state: RobotAgentState) -> dict[str, Any]:
        last = _last_ai_message(state["messages"])
        if last is None:
            return {"final_text": NO_TEXT_RESPONSE}

        return {
            "messages": [
                ToolMessage(
                    content=_tool_limit_error(str(tool_call.get("name") or "")),
                    tool_call_id=str(tool_call.get("id") or ""),
                )
                for tool_call in last.tool_calls
            ],
            "final_text": NO_TEXT_RESPONSE,
        }

    def _final_response(self, state: RobotAgentState) -> dict[str, Any]:
        if state["error_text"]:
            return {"final_text": state["error_text"]}
        last = _last_ai_message(state["messages"])
        if last is None:
            return {"final_text": NO_TEXT_RESPONSE}
        text = _message_text(last)
        if (
            not text
            and state["needs_action_tool"]
            and not state["action_tool_ran"]
            and state["missing_action_repairs"] >= MAX_MISSING_ACTION_REPAIRS
        ):
            return {"final_text": "Where would you like me to move?"}
        return {"final_text": text or NO_TEXT_RESPONSE}

    def _instructions(self) -> str:
        return f"{SYSTEM_PROMPT}\n\n{self._robot_context.render_instruction_block()}"

    async def _execute_observation_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> tuple[str, bool]:
        output = await self._execute_tool(name, arguments)
        observed = (
            _output_has_current_pose(output) and self._robot_context.latest_tcp_pose() is not None
        )
        return output, observed

    async def _call_policy_checked_tool(self, name: str, arguments: dict[str, Any]) -> str:
        policy_attributes: dict[str, Any] = {"tool.name": name}
        async with self._tracer.span(
            "robot.task_policy",
            "robot_control",
            attributes=policy_attributes,
        ):
            decision = validate_task_step(name, arguments, self._robot_context)
            policy_attributes["decision_ok"] = decision.ok
            if decision.error is not None:
                policy_attributes["error"] = decision.error
            if decision.suggested_next_tool is not None:
                policy_attributes["suggested_next_tool"] = decision.suggested_next_tool
        if not decision.ok:
            return json.dumps(structured_task_policy_error(decision), ensure_ascii=False)
        output = await self._tool_bridge.call_tool(name, arguments)
        self._robot_context.update_from_tool_result(name, output)
        self._tracer.event(
            "robot.context_update",
            "robot_control",
            attributes={"tool.name": name},
        )
        return output

    async def _execute_tool(self, name: str, arguments: dict[str, Any]) -> str:
        try:
            output = await self._call_policy_checked_tool(name, arguments)
            plan_name = executable_plan_name(output)
            if name in PLAN_TOOL_NAMES and plan_name:
                self._robot_context.remember_executable_plan(plan_name)
            return output
        except RobotMCPError as exc:
            validation_error = RobotCallValidationError(
                str(exc),
                correction="Check the robot control server, then retry the robot action.",
            )
            return json.dumps(structured_robot_call_error(validation_error), ensure_ascii=False)



def _messages_from_turn(turn: AgentTurnInput) -> list[BaseMessage]:
    return [HumanMessage(content=turn.user_text)]


def _last_ai_message(messages: list[BaseMessage]) -> AIMessage | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str) and part.strip():
                parts.append(part.strip())
            elif (
                isinstance(part, dict)
                and part.get("type") == "text"
                and isinstance(part.get("text"), str)
                and part["text"].strip()
            ):
                parts.append(part["text"].strip())
        return "\n".join(parts)
    return ""


def _first_available_tool(tools: list[dict[str, Any]], names: tuple[str, ...]) -> str | None:
    tool_names = {tool.get("name") for tool in tools}
    for name in names:
        if name in tool_names:
            return name
    return None


def _tools_for_model_binding(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_tool_for_model_binding(tool) for tool in tools]


def _tool_for_model_binding(tool: dict[str, Any]) -> dict[str, Any]:
    if tool.get("type") != "function" or isinstance(tool.get("function"), dict):
        return tool
    name = tool.get("name")
    if not isinstance(name, str) or not name:
        return tool

    function: dict[str, Any] = {"name": name}
    description = tool.get("description")
    if isinstance(description, str):
        function["description"] = description
    parameters = tool.get("parameters")
    if isinstance(parameters, dict):
        function["parameters"] = parameters
    return {"type": "function", "function": function}


def _looks_like_robot_action_request(text: str) -> bool:
    normalized = text.casefold()
    return any(term in normalized for term in ROBOT_ACTION_TERMS)


def _should_repair_missing_action(state: RobotAgentState, last: AIMessage) -> bool:
    if not state["needs_action_tool"]:
        return False
    if state["action_tool_ran"]:
        return False
    if state["missing_action_repairs"] >= MAX_MISSING_ACTION_REPAIRS:
        return False
    if state["tool_turns"] >= MAX_AGENT_TOOL_TURNS:
        return False
    text = _message_text(last).casefold()
    return not text or any(term in text for term in FUTURE_PROMISE_TERMS)


def _should_execute_supported_action_fallback(
    state: RobotAgentState,
    last: AIMessage,
    robot_context: RobotContextStore,
) -> bool:
    if not state["needs_action_tool"]:
        return False
    if state["action_tool_ran"]:
        return False
    if state["missing_action_repairs"] < MAX_MISSING_ACTION_REPAIRS:
        return False
    text = _message_text(last).casefold()
    if text and not any(term in text for term in FUTURE_PROMISE_TERMS):
        return False
    return _supported_action_from_text(state["user_text"], robot_context.latest_tcp_pose()) is not None


def _tool_choice_for_state(state: RobotAgentState) -> str:
    if (
        state["needs_action_tool"]
        and not state["action_tool_ran"]
        and state["missing_action_repairs"] > 0
    ):
        return "required"
    return "auto"


def _supported_action_from_text(text: str, pose: dict[str, Any] | None) -> SupportedAction | None:
    components = _pose_components(pose)
    if components is None:
        return None
    position, orientation = components
    normalized = text.casefold()

    if "wave" in normalized:
        return _cartesian_wave_action(position, orientation)

    if "up" in normalized and "down" in normalized:
        distance_m = _distance_m(normalized)
        return _free_motion_sequence(
            position,
            orientation,
            [
                {"dy": 0.0, "dz": distance_m},
                {"dy": 0.0, "dz": -distance_m},
            ],
            "Moved up and down.",
        )

    if "up" in normalized or "raise" in normalized or "lift" in normalized:
        distance_m = _distance_m(normalized)
        return _free_motion_action(
            position,
            orientation,
            distance_m,
            f"Moved up {distance_m * 1000:.0f} mm.",
        )

    if "down" in normalized or "lower" in normalized or "drop" in normalized:
        distance_m = _distance_m(normalized)
        return _free_motion_action(
            position,
            orientation,
            -distance_m,
            f"Moved down {distance_m * 1000:.0f} mm.",
        )

    return None


def _free_motion_action(
    position: dict[str, float],
    orientation: dict[str, float],
    delta_z_m: float,
    success_text: str,
) -> SupportedAction:
    return SupportedAction(
        steps=[
            _free_motion_step(
                position,
                orientation,
                {"dy": 0.0, "dz": delta_z_m},
            )
        ],
        success_text=success_text,
    )


def _free_motion_sequence(
    position: dict[str, float],
    orientation: dict[str, float],
    offsets: list[dict[str, float]],
    success_text: str,
) -> SupportedAction:
    return SupportedAction(
        steps=[_free_motion_step(position, orientation, offset) for offset in offsets],
        success_text=success_text,
    )


def _cartesian_wave_action(
    position: dict[str, float],
    orientation: dict[str, float],
) -> SupportedAction:
    return SupportedAction(
        steps=[
            SupportedToolStep(
                name="moveit_plan_and_execute_cartesian_motion",
                arguments={
                    "robot_name": VIZOR_ROBOT_NAME,
                    "waypoints": [
                        _cartesian_waypoint(position, orientation, 0.0, 0.15),
                        _cartesian_waypoint(position, orientation, 0.20, 0.15),
                        _cartesian_waypoint(position, orientation, -0.20, 0.15),
                        _cartesian_waypoint(position, orientation, 0.0, 0.15),
                    ],
                    "timeout_s": 10,
                },
            )
        ],
        success_text="Waved.",
    )


def _cartesian_waypoint(
    position: dict[str, float],
    orientation: dict[str, float],
    delta_y_m: float,
    delta_z_m: float,
) -> dict[str, dict[str, float]]:
    return {
        "position": {
            "x": position["x"],
            "y": position["y"] + delta_y_m,
            "z": position["z"] + delta_z_m,
        },
        "orientation": orientation,
    }


def _free_motion_step(
    position: dict[str, float],
    orientation: dict[str, float],
    offset: dict[str, float],
) -> SupportedToolStep:
    return SupportedToolStep(
        name="moveit_plan_and_execute_free_motion",
        arguments={
            "robot_name": VIZOR_ROBOT_NAME,
            "target_pose": {
                "position": {
                    "x": position["x"],
                    "y": position["y"] + offset["dy"],
                    "z": position["z"] + offset["dz"],
                },
                "orientation": orientation,
            },
            "timeout_s": 10,
        },
    )


def _distance_m(text: str, *, default: float = 0.20) -> float:
    if "bit" in text or "slightly" in text:
        return 0.05
    if "lot" in text or "far" in text or "large" in text:
        return 0.45
    return default


def _pose_components(
    pose: dict[str, Any] | None,
) -> tuple[dict[str, float], dict[str, float]] | None:
    if not isinstance(pose, dict):
        return None
    raw_position = pose.get("position")
    raw_orientation = pose.get("orientation")
    if not isinstance(raw_position, dict) or not isinstance(raw_orientation, dict):
        return None

    try:
        position = {
            "x": float(raw_position["x"]),
            "y": float(raw_position["y"]),
            "z": float(raw_position["z"]),
        }
        orientation = {
            "x": float(raw_orientation["x"]),
            "y": float(raw_orientation["y"]),
            "z": float(raw_orientation["z"]),
            "w": float(raw_orientation["w"]),
        }
    except (KeyError, TypeError, ValueError):
        return None
    return position, orientation


def _output_has_verified_execution(output: str) -> bool:
    structured_content = _structured_content_from_output(output)
    if not isinstance(structured_content, dict) or structured_content.get("ok") is not True:
        return False
    verification = structured_content.get("verification")
    if isinstance(verification, dict) and verification.get("result") == "pass":
        return True
    execution = structured_content.get("execution")
    return isinstance(execution, dict) and execution.get("verification_result") == "pass"


def _execution_failure_text(output: str) -> str:
    structured_content = _structured_content_from_output(output)
    if isinstance(structured_content, dict):
        error = structured_content.get("error")
        if isinstance(error, str) and error:
            return error
        feedback = structured_content.get("feedback")
        if isinstance(feedback, dict) and isinstance(feedback.get("message"), str):
            return feedback["message"]
    return NO_TEXT_RESPONSE


def _structured_content_from_output(output: str) -> Any:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload.get("structured_content")


def _output_has_current_pose(output: str) -> bool:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    structured_content = payload.get("structured_content")
    if not isinstance(structured_content, dict) or structured_content.get("ok") is not True:
        return False
    if isinstance(structured_content.get("tcp_pose"), dict):
        return True
    raw = structured_content.get("raw")
    return isinstance(raw, dict) and isinstance(raw.get("pose"), dict)


def _one_tool_at_a_time_error(next_tool_name: str) -> str:
    return json.dumps(
        {
            "ok": False,
            "error": "Only one robot tool call may be executed per model turn.",
            "correction": "Wait for the previous tool result, then retry this tool call if still needed.",
            "retryable": True,
            "suggested_next_tool": next_tool_name or None,
        },
        ensure_ascii=False,
    )


def _tool_limit_error(next_tool_name: str) -> str:
    return json.dumps(
        {
            "ok": False,
            "error": "Robot tool turn limit reached.",
            "correction": "Stop tool use and explain the blocker to the user.",
            "retryable": False,
            "suggested_next_tool": next_tool_name or None,
        },
        ensure_ascii=False,
    )
