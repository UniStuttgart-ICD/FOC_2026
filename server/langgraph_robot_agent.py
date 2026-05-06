"""LangGraph orchestration for Codex robot agent turns."""

from __future__ import annotations

import asyncio
import json
import operator
import uuid
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from loguru import logger

from prompts import SYSTEM_PROMPT
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

MAX_CODEX_TOOL_TURNS = 3
VIZOR_ROBOT_NAME = "UR10"
PLAN_TOOL_NAMES = {"moveit_plan_free_motion", "moveit_plan_cartesian_motion"}
OBSERVE_TOOL_NAMES = ("moveit_get_current_pose",)
NO_TEXT_RESPONSE = "I could not confirm that the action completed."


class RobotAgentState(TypedDict):
    user_text: str
    messages: Annotated[list[BaseMessage], operator.add]
    tools: list[dict[str, Any]]
    tool_turns: int
    observed_this_turn: bool
    final_text: str
    error_text: str | None


class LangGraphRobotAgent:
    """Runs a Codex robot dialogue turn through a LangGraph state machine."""

    def __init__(
        self,
        *,
        model: Any,
        tool_bridge: Any,
        robot_context: RobotContextStore,
        thread_id: str | None = None,
    ) -> None:
        self._model = model
        self._tool_bridge = tool_bridge
        self._robot_context = robot_context
        self._thread_id = thread_id or f"codex-robot-agent-{uuid.uuid4()}"
        self._latest_state: dict[str, Any] | None = None
        self._graph = self._compile_graph()

    async def run_turn(self, turn: AgentTurnInput) -> str:
        state: RobotAgentState = {
            "user_text": turn.user_text,
            "messages": _messages_from_turn(turn),
            "tools": [],
            "tool_turns": 0,
            "observed_this_turn": False,
            "final_text": "",
            "error_text": None,
        }
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
        builder.add_node("observe_current_pose", self._observe_current_pose)
        builder.add_node("call_model", self._call_model)
        builder.add_node("execute_robot_tool", self._execute_robot_tool)
        builder.add_node("final_response", self._final_response)
        builder.add_edge(START, "observe_current_pose")
        builder.add_edge("observe_current_pose", "call_model")
        builder.add_conditional_edges("call_model", self._route_after_model)
        builder.add_edge("execute_robot_tool", "observe_current_pose")
        builder.add_edge("final_response", END)
        return builder.compile(checkpointer=InMemorySaver())

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
        model = self._model.bind_tools(tools)
        system = SystemMessage(content=self._instructions())
        started = monotonic_s()
        logger.info(
            "Codex LangChain request start tool_turns={} messages={} tools={}",
            state["tool_turns"],
            len(state["messages"]),
            len(tools),
        )
        try:
            message = await model.ainvoke([system, *state["messages"]])
        except asyncio.CancelledError:
            logger.warning(
                "Codex LangChain request cancelled elapsed_ms={} tool_turns={} messages={} tools={}",
                elapsed_ms_since(started),
                state["tool_turns"],
                len(state["messages"]),
                len(tools),
            )
            raise
        except Exception as exc:
            logger.exception(
                "Codex LangChain request failed elapsed_ms={} tool_turns={} messages={} tools={} error={}",
                elapsed_ms_since(started),
                state["tool_turns"],
                len(state["messages"]),
                len(tools),
                exc,
            )
            raise
        else:
            logger.info(
                "Codex LangChain request end elapsed_ms={} tool_calls={} text_len={}",
                elapsed_ms_since(started),
                [call.get("name") for call in getattr(message, "tool_calls", [])],
                len(str(message.content or "")),
            )
        return {"messages": [message], "tools": tools}

    def _route_after_model(self, state: RobotAgentState) -> Literal["execute_robot_tool", "final_response"]:
        if state["error_text"]:
            return "final_response"
        last = _last_ai_message(state["messages"])
        if last is None or not last.tool_calls:
            return "final_response"
        if state["tool_turns"] >= MAX_CODEX_TOOL_TURNS:
            return "final_response"
        return "execute_robot_tool"

    async def _execute_robot_tool(self, state: RobotAgentState) -> dict[str, Any]:
        last = _last_ai_message(state["messages"])
        if last is None:
            return {"messages": [], "tool_turns": state["tool_turns"]}

        tool_messages: list[ToolMessage] = []
        observed_this_turn = state["observed_this_turn"]
        for index, tool_call in enumerate(last.tool_calls):
            name = str(tool_call.get("name") or "")
            call_args = tool_call.get("args")
            args = call_args if isinstance(call_args, dict) else {}
            call_id = str(tool_call.get("id") or "")
            if index > 0:
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
        }

    def _final_response(self, state: RobotAgentState) -> dict[str, Any]:
        if state["error_text"]:
            return {"final_text": state["error_text"]}
        last = _last_ai_message(state["messages"])
        if last is None:
            return {"final_text": NO_TEXT_RESPONSE}
        text = str(last.content or "").strip()
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
        decision = validate_task_step(name, arguments, self._robot_context)
        if not decision.ok:
            return json.dumps(structured_task_policy_error(decision), ensure_ascii=False)
        output = await self._tool_bridge.call_tool(name, arguments)
        self._robot_context.update_from_tool_result(name, output)
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


def _first_available_tool(tools: list[dict[str, Any]], names: tuple[str, ...]) -> str | None:
    tool_names = {tool.get("name") for tool in tools}
    for name in names:
        if name in tool_names:
            return name
    return None


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
