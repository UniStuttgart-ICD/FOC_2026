"""LangGraph orchestration for robot agent turns."""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import operator
import uuid
from collections.abc import Awaitable, Callable, Iterable
from typing import Annotated, Any, Literal, TypedDict, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from loguru import logger

from agent_control.prompts import SYSTEM_PROMPT
from agent_control.robot_job_submission import (
    QUEUEABLE_ROBOT_ACTION_TOOLS,
    RobotJobSubmitter,
)
from process_trace import NoopProcessTracer, ProcessTracer
from robot_control.call_validation import (
    SUPPORTED_TASK_PLAN_REQUIRED_PROOFS,
    RobotCallValidationError,
    agent_tool_description,
    ensure_task_solution_execution_allowed,
    executable_plan_name,
    structured_robot_call_error,
    validate_robot_tool_call,
)
from robot_control.context import RecentTaskSolution, RobotContextStore
from robot_control.execution_intent import (
    explicit_execute_requested as _explicit_execute_requested,
)
from robot_control.execution_intent import (
    looks_like_robot_action_request as _looks_like_robot_action_request,
)
from robot_control.execution_intent import (
    should_auto_execute_successful_plan,
)
from robot_control.manipulation_plans import parse_task_solution_result
from robot_control.mcp_bridge import RobotMCPError
from robot_control.shared_geometry.pose_update import update_physical_model_pose
from robot_control.shared_geometry.role_update import update_dynamic_role
from robot_control.task_policy import (
    DEFAULT_EXECUTABLE_PLAN_MAX_AGE_S,
    TaskPolicyDecision,
    structured_task_policy_error,
    validate_task_step,
)
from robot_control.verified_execution_client import (
    VERIFIED_EXECUTION_DEFAULT_TIMEOUT_S,
    VerifiedExecutionClient,
    verified_execution_output_to_json,
)
from user_sensing.context import UserSensingContextStore
from voice_runtime.agent_turn import AgentTurnInput
from voice_runtime.timing import elapsed_ms_since, monotonic_s

MAX_AGENT_TOOL_TURNS = 6
TASK_PLAN_STAGE_MAX_ATTEMPTS = 2
TASK_PLAN_OBSERVATION_MAX_ATTEMPTS = 3
TASK_PLAN_OBSERVATION_RETRY_DELAY_S = 0.2
TASK_PLAN_POSE_OBSERVATION_TIMEOUT_S = 2.0
VIZOR_ROBOT_NAME = "UR10"
GEOMETRY_UPDATE_DYNAMIC_ROLE_TOOL_NAME = "geometry_update_dynamic_role"
MODEL_VISIBLE_TASK_PLANNER_TOOL_NAME = "moveit_plan_manipulation_task"
MODEL_HIDDEN_TASK_PLANNER_TOOL_NAMES = {
    "moveit_plan_compound_task",
    "moveit_plan_pick_task",
    "moveit_plan_place_task",
}
MODEL_HIDDEN_MOTION_PLANNER_TOOL_NAMES = {
    "moveit_plan_free_motion",
    "moveit_plan_cartesian_motion",
    "moveit_plan_pick",
    "moveit_plan_place",
}
MODEL_HIDDEN_CONTRACT_INTERNAL_TOOL_NAMES = {
    "moveit_release_object",
    "moveit_verify_released_object",
    "moveit_remove_scene_object",
    "moveit_open_gripper",
    "moveit_close_gripper",
    "moveit_attach_object",
    "moveit_verify_attached_object",
}
MODEL_HIDDEN_TOOL_NAMES = (
    MODEL_HIDDEN_TASK_PLANNER_TOOL_NAMES
    | MODEL_HIDDEN_MOTION_PLANNER_TOOL_NAMES
    | MODEL_HIDDEN_CONTRACT_INTERNAL_TOOL_NAMES
)
TASK_LEVEL_REPLAN_TOOL_NAME = MODEL_VISIBLE_TASK_PLANNER_TOOL_NAME
SUPPORTED_TASK_SOLUTION_KINDS = {
    "pick",
    "place",
    "hold",
    "move_and_release",
    "pick_place",
}
SUPPORTED_TASK_PLAN_HANDLERS = {
    "motion",
    "close_gripper",
    "open_gripper",
    "attach_object",
    "release_object",
    "verify_attached_object",
    "verify_released_object",
}
TASK_PLAN_STAGE_REQUIRED_PROOF_BY_HANDLER = {
    "motion": "emulated_motion_plan",
    "close_gripper": "verified_gripper_closed",
    "open_gripper": "verified_gripper_open",
    "attach_object": "planning_scene_attached",
    "release_object": "planning_scene_update",
    "verify_attached_object": "attachment_check",
    "verify_released_object": "release_check",
}
TASK_EXECUTION_INSTRUCTION = (
    "Use moveit_execute_task for returned task_solution_id values. It executes in RViz/simulation "
    "first and also attempts real robot execution when connected; simulation success counts as "
    "execution success with real robot status reported separately."
)
PLAN_TOOL_NAMES = {
    "moveit_plan_free_motion",
    "moveit_plan_cartesian_motion",
    "moveit_plan_pick",
    "moveit_plan_place",
    "moveit_plan_compound_task",
    "moveit_plan_manipulation_task",
}
ACTION_TOOL_NAMES = {
    "moveit_plan_free_motion",
    "moveit_plan_cartesian_motion",
    "moveit_plan_pick",
    "moveit_plan_place",
    "moveit_plan_compound_task",
    "moveit_plan_manipulation_task",
    "moveit_execute_task",
    "moveit_execute_plan",
    "moveit_execute_task_solution",
    "moveit_execute_task_plan",
    "moveit_go_home",
    "moveit_sync_real_robot_state",
    "moveit_verify_attached_object",
    "moveit_release_object",
    "moveit_verify_released_object",
    "moveit_remove_scene_object",
    "moveit_open_gripper",
    "moveit_close_gripper",
    "moveit_attach_object",
}
AFTER_SUCCESS_ACTION_TOOL_NAMES = PLAN_TOOL_NAMES | {"moveit_open_gripper"}
OBSERVE_TOOL_NAMES = ("moveit_get_current_pose",)
NO_TEXT_RESPONSE = "I could not confirm that the action completed."
MAX_MISSING_ACTION_REPAIRS = 1
FUTURE_PROMISE_TERMS = (
    "i'll",
    "i’ll",
    "i will",
    "i’m",
    "i’m going to",
    "i am going to",
    "i can try",
    "i can retry",
    "let me",
    "first, then",
    "then i'll",
    "then i’ll",
    "then make",
    "then do",
    "will retry",
    "will try",
)
ProcessTracerLike = ProcessTracer | NoopProcessTracer


class RobotAgentState(TypedDict):
    user_text: str
    messages: Annotated[list[BaseMessage], operator.add]
    tools: list[dict[str, Any]]
    tool_turns: int
    observed_this_turn: bool
    allow_pending_plan_execution: bool
    needs_action_tool: bool
    action_tool_ran: bool
    queued_robot_job: bool
    missing_action_repairs: int
    manipulation_planner_repairs: int
    final_text: str
    error_text: str | None


class LangGraphRobotAgent:
    """Runs a robot dialogue turn through a LangGraph state machine."""

    def __init__(
        self,
        *,
        model: Any,
        tool_bridge: Any,
        robot_context: RobotContextStore,
        geometry_world_context: Any | None = None,
        user_sensing_bridge: Any | None = None,
        user_sensing_context: UserSensingContextStore | None = None,
        user_sensing_max_age_s: float = 2.0,
        thread_id: str | None = None,
        job_submitter: RobotJobSubmitter | None = None,
        robot_job_blackboard_summary: Callable[[], str | None] | None = None,
        verified_execution_client: VerifiedExecutionClient | None = None,
        tracer: ProcessTracerLike | None = None,
    ) -> None:
        self._model = model
        self._tool_bridge = tool_bridge
        self._robot_context = robot_context
        self._geometry_world_context = geometry_world_context
        self._user_sensing_bridge = user_sensing_bridge
        self._user_sensing_context = user_sensing_context or UserSensingContextStore()
        self._user_sensing_max_age_s = user_sensing_max_age_s
        self._thread_id = thread_id or f"robot-agent-{uuid.uuid4()}"
        self._job_submitter = job_submitter
        self._robot_job_blackboard_summary = robot_job_blackboard_summary
        self._verified_execution_client = verified_execution_client
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
            "allow_pending_plan_execution": turn.allow_pending_plan_execution,
            "needs_action_tool": _looks_like_robot_action_request(turn.user_text),
            "action_tool_ran": False,
            "queued_robot_job": False,
            "missing_action_repairs": 0,
            "manipulation_planner_repairs": 0,
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
        builder.add_node(
            "execute_pending_plan",
            self._traced_node("execute_pending_plan", self._execute_pending_plan),
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
            "stop_after_tool_limit",
            self._traced_node("stop_after_tool_limit", self._stop_after_tool_limit),
        )
        builder.add_node(
            "final_response",
            self._traced_node("final_response", self._final_response),
        )
        builder.add_edge(START, "observe_current_pose")
        builder.add_conditional_edges("observe_current_pose", self._route_after_observation)
        builder.add_conditional_edges("call_model", self._route_after_model)
        builder.add_edge("execute_pending_plan", END)
        builder.add_conditional_edges("execute_robot_tool", self._route_after_robot_tool)
        builder.add_edge("repair_missing_action", "observe_current_pose")
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
        available_tools = self._tool_bridge.function_tools()
        tools = self._model_visible_tools(available_tools)
        await self._refresh_user_sensing_context()
        if state.get("observed_this_turn"):
            return {"tools": tools}
        if _is_direct_verified_recovery_request(state["user_text"]):
            return {"tools": tools}
        observe_tool_name = _first_available_tool(available_tools, OBSERVE_TOOL_NAMES)
        if observe_tool_name is None:
            return {"tools": tools}
        logger.info("Refreshing robot observation before Codex request with {}", observe_tool_name)
        _, observed = await self._execute_observation_tool(
            observe_tool_name, {"robot_name": VIZOR_ROBOT_NAME}
        )
        return {"tools": tools, "observed_this_turn": observed}

    def _route_after_observation(
        self, state: RobotAgentState
    ) -> Literal["execute_pending_plan", "call_model"]:
        if (
            state["allow_pending_plan_execution"]
            and self._job_submitter is not None
            and _should_execute_latest_pending_plan(state["user_text"], self._robot_context)
        ):
            return "execute_pending_plan"
        return "call_model"

    async def _execute_pending_plan(self, state: RobotAgentState) -> dict[str, Any]:
        pending = self._robot_context.latest_pending_executable_plan(
            max_age_s=DEFAULT_EXECUTABLE_PLAN_MAX_AGE_S
        )
        if pending is None or _is_task_stage_attempt_plan_name(pending.plan_name):
            return {}

        arguments = {
            "robot_name": pending.robot_name or VIZOR_ROBOT_NAME,
            "plan_name": pending.plan_name,
            "timeout_s": VERIFIED_EXECUTION_DEFAULT_TIMEOUT_S,
        }
        if self._job_submitter is not None:
            output = await self._submit_policy_checked_robot_job(
                "moveit_execute_plan",
                arguments,
                requested_by_turn_id=self._tracer.current_context().turn_id,
                user_text=state["user_text"],
                after_success_tool=pending.after_success_tool,
                after_success_arguments=pending.after_success_arguments,
                execute_via_mcp=pending.execute_via_mcp,
            )
        elif self._verified_execution_client is not None:
            output = await self._execute_verified_plan_tool(
                arguments,
                user_text=state["user_text"],
            )
        else:
            output = await self._execute_tool(
                "moveit_execute_plan",
                arguments,
                user_text=state["user_text"],
            )
        return {
            "final_text": _execution_result_text(output, pending.plan_name),
            "action_tool_ran": True,
        }

    async def _call_model(self, state: RobotAgentState) -> dict[str, Any]:
        tools = state["tools"] or self._model_visible_tools(self._tool_bridge.function_tools())
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
        return "final_response"

    def _route_after_robot_tool(self, state: RobotAgentState) -> str:
        if state["queued_robot_job"] or state["final_text"]:
            return END
        return "observe_current_pose"

    def _repair_missing_action(self, state: RobotAgentState) -> dict[str, Any]:
        return {
            "messages": [
                HumanMessage(
                    content=(
                        "The previous response described a future robot action but did not call "
                        "a MoveIt action tool. For this movement request, call exactly one "
                        "available MoveIt action tool now, or explain a concrete blocker if no "
                        "valid MoveIt tool call is possible. Do not say you will do it later."
                    )
                )
            ],
            "missing_action_repairs": state["missing_action_repairs"] + 1,
        }

    async def _execute_robot_tool(self, state: RobotAgentState) -> dict[str, Any]:
        last = _last_ai_message(state["messages"])
        if last is None:
            return {"messages": [], "tool_turns": state["tool_turns"]}

        tool_messages: list[ToolMessage] = []
        observed_this_turn = state["observed_this_turn"]
        action_tool_ran = state["action_tool_ran"]
        queued_robot_job = state["queued_robot_job"]
        manipulation_planner_repairs = state["manipulation_planner_repairs"]
        final_text = state["final_text"]
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
            if name == GEOMETRY_UPDATE_DYNAMIC_ROLE_TOOL_NAME:
                output = _execute_geometry_update_dynamic_role(dict(args))
            elif name in OBSERVE_TOOL_NAMES:
                output, observed_this_turn = await self._execute_observation_tool(name, dict(args))
            elif name == MODEL_VISIBLE_TASK_PLANNER_TOOL_NAME:
                output = await self._execute_tool(
                    name,
                    dict(args),
                    user_text=state["user_text"],
                    allow_execution=state["allow_pending_plan_execution"],
                )
                action_tool_ran = True
                (
                    final_text,
                    manipulation_planner_repairs,
                ) = _task_planning_result_text(
                    output,
                    repair_attempts=manipulation_planner_repairs,
                )
                observed_this_turn = False
            elif name == "moveit_execute_task":
                output = await self._execute_task_tool(
                    dict(args),
                    user_text=state["user_text"],
                    allow_execution=state["allow_pending_plan_execution"],
                )
                action_tool_ran = True
                final_text = _task_execution_result_text(output)
                observed_this_turn = False
            elif name == "moveit_execute_task_plan":
                output = await self._execute_verified_task_plan_tool(
                    dict(args),
                    user_text=state["user_text"],
                    allow_execution=state["allow_pending_plan_execution"],
                )
                action_tool_ran = True
                final_text = _task_plan_execution_result_text(output)
                observed_this_turn = False
            elif name == "moveit_execute_plan" and self._verified_execution_client is not None:
                output = await self._execute_verified_plan_tool(
                    dict(args),
                    user_text=state["user_text"],
                    allow_execution=state["allow_pending_plan_execution"],
                )
                action_tool_ran = True
                if _execution_succeeded(output):
                    final_text = _execution_result_text(
                        output,
                        str(args.get("plan_name") or ""),
                    )
                observed_this_turn = False
            elif name in {"moveit_go_home", "moveit_sync_real_robot_state"}:
                output = await self._execute_verified_recovery_tool(
                    name,
                    dict(args),
                    user_text=state["user_text"],
                    allow_execution=state["allow_pending_plan_execution"],
                )
                action_tool_ran = True
                final_text = _execution_result_text(output, name)
                observed_this_turn = False
            elif self._job_submitter is not None and name in QUEUEABLE_ROBOT_ACTION_TOOLS:
                job_user_text = state["user_text"] if state["allow_pending_plan_execution"] else None
                output = await self._submit_policy_checked_robot_job(
                    name,
                    dict(args),
                    requested_by_turn_id=self._tracer.current_context().turn_id,
                    user_text=job_user_text,
                    allow_execution=state["allow_pending_plan_execution"],
                )
                action_tool_ran = action_tool_ran or name in ACTION_TOOL_NAMES
                queued_robot_job = _queued_job_result_ok(output)
                if queued_robot_job:
                    final_text = _queued_job_result_text(name, dict(args), job_user_text)
                observed_this_turn = False
            else:
                output = await self._execute_tool(
                    name,
                    dict(args),
                    user_text=state["user_text"],
                    allow_execution=state["allow_pending_plan_execution"],
                )
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
            "queued_robot_job": queued_robot_job,
            "manipulation_planner_repairs": manipulation_planner_repairs,
            "final_text": final_text,
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
        parts = [
            SYSTEM_PROMPT,
            self._task_execution_mode_instruction(),
            self._robot_context.render_instruction_block(),
        ]
        if self._geometry_world_context is not None:
            parts.append(self._geometry_world_context.render_instruction_block())
        if self._robot_job_blackboard_summary is not None:
            job_blackboard_summary = self._robot_job_blackboard_summary()
            if job_blackboard_summary:
                parts.append(job_blackboard_summary)
        if self._user_sensing_bridge is not None:
            parts.append(self._user_sensing_context.render_instruction_block())
        return "\n\n".join(parts)

    def _model_visible_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        visible_names = {
            "moveit_get_current_pose",
            "moveit_get_robot_state",
            "moveit_list_scene_objects",
            "moveit_get_object_context",
            "moveit_explain_motion_failure",
            "moveit_execute_task",
        }
        visible = [
            tool
            for tool in tools
            if tool.get("name") in visible_names
        ]
        task_planner_tool = _model_visible_task_planner_tool(tools)
        if task_planner_tool is not None:
            visible.append(task_planner_tool)
        visible.append(_geometry_update_dynamic_role_tool())
        return visible

    def _task_execution_mode_instruction(self) -> str:
        return TASK_EXECUTION_INSTRUCTION

    async def _refresh_user_sensing_context(self) -> None:
        if self._user_sensing_bridge is None:
            return
        attributes = {"tool.name": "vizor_get_sensor_context"}
        async with self._tracer.span(
            "user_sensing.mcp.call_tool",
            "user_sensing",
            attributes=attributes,
        ):
            try:
                output = await self._user_sensing_bridge.read_context(
                    max_age_s=self._user_sensing_max_age_s
                )
            except Exception as exc:
                attributes["error"] = str(exc)
                logger.warning("User sensing context refresh failed: {}", exc)
                return
        self._user_sensing_context.update_from_tool_result(output)
        summary_attributes = self._user_sensing_context.summary_attributes()
        logger.info("User sensing context updated: {}", self._user_sensing_context.summary_text())
        if self._tracer.options.include_tool_payloads:
            self._tracer.event(
                "user_sensing.mcp.tool_result",
                "user_sensing",
                attributes={
                    "tool.name": "vizor_get_sensor_context",
                    "tool.result": output,
                },
            )
        self._tracer.event(
            "user_sensing.context_update",
            "user_sensing",
            attributes={
                "tool.name": "vizor_get_sensor_context",
                **summary_attributes,
            },
        )

    async def _execute_observation_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> tuple[str, bool]:
        output = await self._execute_tool(name, arguments)
        observed = (
            _output_has_current_pose(output) and self._robot_context.latest_tcp_pose() is not None
        )
        return output, observed

    async def _validate_robot_task_step(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        user_text: str | None = None,
        allow_execution: bool = True,
    ) -> TaskPolicyDecision:
        policy_attributes: dict[str, Any] = {"tool.name": name}
        async with self._tracer.span(
            "robot.task_policy",
            "robot_control",
            attributes=policy_attributes,
        ):
            decision = validate_task_step(
                name,
                arguments,
                self._robot_context,
                user_text=user_text,
                explicit_execute_requested=allow_execution
                and _explicit_execute_requested(user_text),
            )
            policy_attributes["decision_ok"] = decision.ok
            if decision.error is not None:
                policy_attributes["error"] = decision.error
            if decision.suggested_next_tool is not None:
                policy_attributes["suggested_next_tool"] = decision.suggested_next_tool
        return decision

    async def _submit_policy_checked_robot_job(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        requested_by_turn_id: str | None,
        user_text: str | None,
        allow_execution: bool = True,
        after_success_tool: str | None = None,
        after_success_arguments: dict[str, Any] | None = None,
        execute_via_mcp: bool = False,
    ) -> str:
        decision = await self._validate_robot_task_step(
            name,
            arguments,
            user_text=user_text,
            allow_execution=allow_execution,
        )
        if not decision.ok:
            return json.dumps(_structured_task_policy_error(decision), ensure_ascii=False)
        assert self._job_submitter is not None
        return await self._job_submitter.submit_tool(
            name,
            arguments,
            requested_by_turn_id=requested_by_turn_id,
            user_text=user_text,
            after_success_tool=after_success_tool,
            after_success_arguments=after_success_arguments,
            execute_via_mcp=execute_via_mcp,
        )

    async def _call_policy_checked_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        user_text: str | None = None,
        allow_execution: bool = True,
    ) -> str:
        decision = await self._validate_robot_task_step(
            name,
            arguments,
            user_text=user_text,
            allow_execution=allow_execution,
        )
        if not decision.ok:
            return json.dumps(_structured_task_policy_error(decision), ensure_ascii=False)
        if name == "moveit_execute_task_solution":
            verified_error = self._verified_real_robot_task_solution_error(arguments)
            if verified_error is not None:
                return json.dumps(verified_error, ensure_ascii=False)
            self._record_task_solution_approval_if_explicit(
                arguments,
                user_text=user_text,
                allow_execution=allow_execution,
            )
            try:
                ensure_task_solution_execution_allowed(self._robot_context, arguments)
            except RobotCallValidationError as exc:
                return json.dumps(structured_robot_call_error(exc), ensure_ascii=False)
        output = await self._tool_bridge.call_tool(name, arguments)
        self._robot_context.update_from_tool_result(name, output)
        self._tracer.event(
            "robot.context_update",
            "robot_control",
            attributes={"tool.name": name},
        )
        return output

    async def _execute_task_tool(
        self,
        arguments: dict[str, Any],
        *,
        user_text: str | None,
        allow_execution: bool = True,
    ) -> str:
        try:
            validate_robot_tool_call("moveit_execute_task", arguments)
        except RobotCallValidationError as exc:
            return json.dumps(structured_robot_call_error(exc), ensure_ascii=False)

        self._record_task_solution_approval_if_explicit(
            arguments,
            user_text=user_text,
            allow_execution=allow_execution,
        )
        _normalize_recent_task_solution_execution_contract(self._robot_context, arguments)
        try:
            ensure_task_solution_execution_allowed(self._robot_context, arguments)
        except RobotCallValidationError as exc:
            return json.dumps(structured_robot_call_error(exc), ensure_ascii=False)

        output = await self._execute_task_contract_stages(
            arguments,
            user_text=user_text,
            allow_execution=allow_execution,
            validation_tool_name="moveit_execute_task",
            public_tool_name="moveit_execute_task",
            run_ar_rviz=True,
            require_verified_client=False,
            approval_checked=True,
        )
        self._robot_context.update_from_tool_result("moveit_execute_task", output)
        return output

    async def _execute_simulation_task_solution_tool(
        self,
        arguments: dict[str, Any],
        *,
        user_text: str | None,
        allow_execution: bool,
    ) -> str:
        try:
            validate_robot_tool_call("moveit_execute_task_solution", arguments)
        except RobotCallValidationError as exc:
            return json.dumps(structured_robot_call_error(exc), ensure_ascii=False)
        self._record_task_solution_approval_if_explicit(
            arguments,
            user_text=user_text,
            allow_execution=allow_execution,
        )
        try:
            ensure_task_solution_execution_allowed(self._robot_context, arguments)
        except RobotCallValidationError as exc:
            return json.dumps(structured_robot_call_error(exc), ensure_ascii=False)
        try:
            output = await self._tool_bridge.call_tool("moveit_execute_task_solution", arguments)
        except RobotMCPError as exc:
            validation_error = RobotCallValidationError(
                str(exc),
                correction="Check the robot control server, then retry the robot action.",
            )
            return json.dumps(structured_robot_call_error(validation_error), ensure_ascii=False)
        self._robot_context.update_from_tool_result("moveit_execute_task_solution", output)
        self._tracer.event(
            "robot.context_update",
            "robot_control",
            attributes={"tool.name": "moveit_execute_task_solution"},
        )
        return output

    def _record_task_solution_approval_if_explicit(
        self,
        arguments: dict[str, Any],
        *,
        user_text: str | None,
        allow_execution: bool,
    ) -> None:
        if not allow_execution or not _explicit_execute_requested(user_text):
            return
        task_solution_id = arguments.get("task_solution_id")
        if not isinstance(task_solution_id, str) or not task_solution_id.strip():
            return
        trace_context = self._tracer.current_context()
        approval_turn_id = trace_context.turn_id or f"approval-{uuid.uuid4().hex}"
        recorded = self._robot_context.record_task_solution_approval(
            task_solution_id,
            approval_turn_id=approval_turn_id,
        )
        self._tracer.event(
            "robot.task_solution_approval",
            "robot_control",
            attributes={
                "approval_recorded": recorded,
                "task_solution_id": task_solution_id,
            },
        )

    def _verified_real_robot_task_solution_error(
        self, _arguments: dict[str, Any]
    ) -> dict[str, Any] | None:
        if self._verified_execution_client is None:
            return None
        exc = RobotCallValidationError(
            "Wrong task execution tool for real-robot mode",
            correction="Use moveit_execute_task_plan with the same task_solution_id.",
        )
        return structured_robot_call_error(
            exc,
            retryable=True,
            suggested_next_tool="moveit_execute_task_plan",
        )

    async def _execute_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        user_text: str | None = None,
        allow_execution: bool = True,
    ) -> str:
        try:
            pending = None
            if name == "moveit_execute_plan":
                plan_name = arguments.get("plan_name")
                if isinstance(plan_name, str):
                    pending = self._robot_context.pending_executable_plan(
                        plan_name,
                        max_age_s=DEFAULT_EXECUTABLE_PLAN_MAX_AGE_S,
                    )
            output = await self._call_policy_checked_tool(
                name,
                arguments,
                user_text=user_text,
                allow_execution=allow_execution,
            )
            if name == "moveit_execute_plan" and _execution_succeeded(output):
                await self._execute_after_success_action(
                    pending.after_success_tool if pending is not None else None,
                    pending.after_success_arguments if pending is not None else None,
                    user_text=user_text,
                    allow_execution=allow_execution,
                )
            return output
        except RobotMCPError as exc:
            validation_error = RobotCallValidationError(
                str(exc),
                correction="Check the robot control server, then retry the robot action.",
            )
            return json.dumps(structured_robot_call_error(validation_error), ensure_ascii=False)

    async def _execute_task_contract_stages(
        self,
        arguments: dict[str, Any],
        *,
        user_text: str | None,
        allow_execution: bool,
        validation_tool_name: str,
        public_tool_name: str,
        run_ar_rviz: bool,
        require_verified_client: bool,
        approval_checked: bool,
    ) -> str:
        try:
            validate_robot_tool_call(validation_tool_name, arguments)
        except RobotCallValidationError as exc:
            return json.dumps(structured_robot_call_error(exc), ensure_ascii=False)
        if require_verified_client and self._verified_execution_client is None:
            exc = RobotCallValidationError(
                "Verified Real Robot Execution client is unavailable.",
                correction="Start or configure the verified execution server, then retry.",
            )
            return json.dumps(
                structured_robot_call_error(exc, retryable=True, suggested_next_tool=None),
                ensure_ascii=False,
            )

        task_solution_id = str(arguments["task_solution_id"]).strip()
        robot_name = str(arguments.get("robot_name") or VIZOR_ROBOT_NAME)
        timeout_s = float(arguments.get("timeout_s") or VERIFIED_EXECUTION_DEFAULT_TIMEOUT_S)
        recent = self._robot_context.recent_task_solution
        if recent is None or recent.task_solution_id != task_solution_id:
            return _task_plan_error(
                "Task plan execution requires the exact recent task_solution_id.",
                "Plan the compound task again, then retry moveit_execute_task_plan with that task_solution_id.",
                suggested_next_tool=TASK_LEVEL_REPLAN_TOOL_NAME,
            )
        if recent.task_kind not in SUPPORTED_TASK_SOLUTION_KINDS:
            return _task_plan_error(
                f"Task plan execution does not support task kind: {recent.task_kind}.",
                "Plan a supported pick/place task, then retry moveit_execute_task_plan.",
                retryable=False,
                suggested_next_tool=None,
            )
        raw = recent.raw
        if raw is None:
            return _task_plan_error(
                "Task plan execution requires the recent raw task solution.",
                "Plan the compound task again, then retry moveit_execute_task_plan with that task_solution_id.",
                suggested_next_tool=TASK_LEVEL_REPLAN_TOOL_NAME,
            )
        execution_steps, execution_steps_error = _task_plan_execution_steps(
            raw,
            task_kind=recent.task_kind,
        )
        if execution_steps_error is not None:
            return execution_steps_error
        assert execution_steps is not None
        _remember_task_solution_execution_contract_steps(
            self._robot_context,
            recent=recent,
            execution_steps=execution_steps,
        )
        if not approval_checked:
            self._record_task_solution_approval_if_explicit(
                arguments,
                user_text=user_text,
                allow_execution=allow_execution,
            )
            try:
                ensure_task_solution_execution_allowed(self._robot_context, arguments)
            except RobotCallValidationError as exc:
                return json.dumps(structured_robot_call_error(exc), ensure_ascii=False)

        execution_id = uuid.uuid4().hex[:8]
        ar_rviz_plan_names: list[str] = []
        physical_plan_names: list[str] = []
        completed_steps: list[dict[str, Any]] = []
        verified_gripper_closed = False
        verified_gripper_open = False
        task_verified = False
        attached_object_verified = False
        released_object_verified = False
        release_verification: dict[str, str] | None = None
        release_proof_output: str | None = None
        released_object_name: str | None = None
        available_tool_names = _function_tool_names(self._tool_bridge.function_tools())
        contract_tool_names = _contract_tool_names(self._tool_bridge)
        has_gripper_contract = any(
            str(step.get("handler")) in {"close_gripper", "open_gripper"}
            for step in execution_steps
        )
        physical_available, real_robot_result = await self._physical_task_readiness_result(
            timeout_s=timeout_s,
            requires_gripper=has_gripper_contract,
        )

        for step in execution_steps:
            step_name = str(step.get("name") or step.get("tool") or "")
            step_handler = str(step["handler"])
            if step_handler == "motion":
                waypoint = _task_plan_step_waypoint(step, raw)
                if waypoint is None:
                    return _task_plan_error(
                        "Task plan motion step references a missing waypoint.",
                        "Plan a supported pick/place task again, then retry moveit_execute_task_plan.",
                        suggested_next_tool=None,
                    )
                pose_output = await self._execute_task_plan_pose_observation(
                    robot_name=robot_name,
                    user_text=user_text,
                    allow_execution=allow_execution,
                )
                if not _tool_ok(pose_output):
                    return await self._task_plan_stage_failure(
                        stage="observe_current_pose",
                        step_name=step_name,
                        output=pose_output,
                        failed_tool_name="moveit_get_current_pose",
                        failed_tool_arguments=_task_plan_pose_observation_arguments(robot_name),
                        task_solution_id=task_solution_id,
                        recent=recent,
                        completed_steps=completed_steps,
                        verified_plan_names=ar_rviz_plan_names,
                        attached_object_verified=attached_object_verified,
                        released_object_verified=released_object_verified,
                        available_tool_names=available_tool_names,
                        user_text=user_text,
                    )
                base_plan_name = f"{task_solution_id}_{_task_plan_step_label(step_name)}"
                stage_succeeded = False
                last_failed_stage = "planning"
                last_failed_output = ""
                last_failed_tool_name: str | None = None
                last_failed_tool_arguments: dict[str, Any] | None = None
                for attempt_index in range(1, TASK_PLAN_STAGE_MAX_ATTEMPTS + 1):
                    plan_name = _task_plan_attempt_name(
                        base_plan_name,
                        execution_id=execution_id,
                        attempt_index=attempt_index,
                    )
                    planning_tool, planning_args = _task_plan_motion_call(
                        step_name,
                        attempt_index=attempt_index,
                        robot_name=robot_name,
                        plan_name=plan_name,
                        waypoint=waypoint,
                        timeout_s=timeout_s,
                    )
                    plan_output = await self._execute_tool(
                        planning_tool,
                        planning_args,
                        user_text=None,
                        allow_execution=allow_execution,
                    )
                    if not _tool_ok(plan_output):
                        last_failed_stage = "planning"
                        last_failed_output = plan_output
                        last_failed_tool_name = planning_tool
                        last_failed_tool_arguments = planning_args
                        if attempt_index < TASK_PLAN_STAGE_MAX_ATTEMPTS:
                            pose_output = await self._execute_task_plan_pose_observation(
                                robot_name=robot_name,
                                user_text=user_text,
                                allow_execution=allow_execution,
                            )
                            if not _tool_ok(pose_output):
                                return await self._task_plan_stage_failure(
                                    stage="observe_current_pose",
                                    step_name=step_name,
                                    output=pose_output,
                                    failed_tool_name="moveit_get_current_pose",
                                    failed_tool_arguments=_task_plan_pose_observation_arguments(robot_name),
                                    task_solution_id=task_solution_id,
                                    recent=recent,
                                    completed_steps=completed_steps,
                                    verified_plan_names=ar_rviz_plan_names,
                                    attached_object_verified=attached_object_verified,
                                    released_object_verified=released_object_verified,
                                    available_tool_names=available_tool_names,
                                    user_text=user_text,
                                )
                        continue
                    executable_name = executable_plan_name(plan_output) or plan_name
                    execute_arguments = {
                        "robot_name": robot_name,
                        "plan_name": executable_name,
                        "timeout_s": timeout_s,
                    }
                    if run_ar_rviz:
                        execute_output = await self._execute_tool(
                            "moveit_execute_plan",
                            execute_arguments,
                            user_text=user_text,
                            allow_execution=allow_execution,
                        )
                    else:
                        execute_output = await self._execute_verified_plan_tool(
                            execute_arguments,
                            user_text=user_text,
                            allow_execution=allow_execution,
                        )
                    self._robot_context.consume_executable_plan(plan_name)
                    if executable_name != plan_name:
                        self._robot_context.consume_executable_plan(executable_name)
                    if not _execution_succeeded(execute_output):
                        last_failed_stage = "verified_execution"
                        last_failed_output = execute_output
                        last_failed_tool_name = "moveit_execute_plan"
                        last_failed_tool_arguments = execute_arguments
                        continue
                    ar_rviz_plan_names.append(executable_name)
                    if run_ar_rviz and physical_available:
                        physical_output = await self._execute_verified_plan_direct(
                            robot_name=robot_name,
                            plan_name=executable_name,
                            timeout_s=timeout_s,
                        )
                        if _execution_succeeded(physical_output):
                            physical_plan_names.append(executable_name)
                        else:
                            physical_available = False
                            real_robot_result = _physical_task_failed_result(
                                failed_stage="verified_execution",
                                failed_tool_name="moveit_execute_plan",
                                failed_tool_arguments=execute_arguments,
                                failed_tool_result=_json_payload(physical_output),
                                verified_plan_names=physical_plan_names,
                            )
                    completed_steps.append({"name": step_name, "handler": step_handler})
                    stage_succeeded = True
                    break
                if not stage_succeeded:
                    return await self._task_plan_stage_failure(
                        stage=last_failed_stage,
                        step_name=step_name,
                        output=last_failed_output,
                        failed_tool_name=last_failed_tool_name,
                        failed_tool_arguments=last_failed_tool_arguments,
                        task_solution_id=task_solution_id,
                        recent=recent,
                        completed_steps=completed_steps,
                        verified_plan_names=ar_rviz_plan_names,
                        attached_object_verified=attached_object_verified,
                        released_object_verified=released_object_verified,
                        available_tool_names=available_tool_names,
                        user_text=user_text,
                    )
                continue
            if step_handler == "close_gripper":
                close_arguments = {"robot_name": robot_name, "timeout_s": timeout_s}
                if run_ar_rviz:
                    close_output = await self._execute_contract_mcp_tool(
                        "moveit_close_gripper",
                        close_arguments,
                    )
                else:
                    assert self._verified_execution_client is not None
                    close_output = verified_execution_output_to_json(
                        await self._verified_execution_client.close_gripper(**close_arguments)
                    )
                    self._robot_context.update_from_tool_result(
                        "moveit_close_gripper",
                        close_output,
                    )
                if not _tool_ok(close_output):
                    return await self._task_plan_stage_failure(
                        stage="close_gripper",
                        step_name=step_name,
                        output=close_output,
                        failed_tool_name="moveit_close_gripper",
                        failed_tool_arguments=close_arguments,
                        task_solution_id=task_solution_id,
                        recent=recent,
                        completed_steps=completed_steps,
                        verified_plan_names=ar_rviz_plan_names,
                        attached_object_verified=attached_object_verified,
                        released_object_verified=released_object_verified,
                        available_tool_names=available_tool_names,
                        user_text=user_text,
                    )
                verified_gripper_closed = True
                verified_gripper_open = False
                if run_ar_rviz and physical_available:
                    physical_output = await self._execute_verified_gripper_direct(
                        "moveit_close_gripper",
                        robot_name=robot_name,
                        timeout_s=timeout_s,
                    )
                    if not _tool_ok(physical_output):
                        physical_available = False
                        real_robot_result = _physical_task_failed_result(
                            failed_stage="close_gripper",
                            failed_tool_name="moveit_close_gripper",
                            failed_tool_arguments=close_arguments,
                            failed_tool_result=_json_payload(physical_output),
                            verified_plan_names=physical_plan_names,
                        )
                completed_steps.append({"name": step_name, "handler": step_handler})
                continue
            if step_handler == "open_gripper":
                open_arguments = {"robot_name": robot_name, "timeout_s": timeout_s}
                if run_ar_rviz:
                    open_output = await self._execute_contract_mcp_tool(
                        "moveit_open_gripper",
                        open_arguments,
                    )
                else:
                    assert self._verified_execution_client is not None
                    open_output = verified_execution_output_to_json(
                        await self._verified_execution_client.open_gripper(**open_arguments)
                    )
                    self._robot_context.update_from_tool_result(
                        "moveit_open_gripper",
                        open_output,
                    )
                if not _tool_ok(open_output):
                    return await self._task_plan_stage_failure(
                        stage="open_gripper",
                        step_name=step_name,
                        output=open_output,
                        failed_tool_name="moveit_open_gripper",
                        failed_tool_arguments=open_arguments,
                        task_solution_id=task_solution_id,
                        recent=recent,
                        completed_steps=completed_steps,
                        verified_plan_names=ar_rviz_plan_names,
                        attached_object_verified=attached_object_verified,
                        released_object_verified=released_object_verified,
                        available_tool_names=available_tool_names,
                        user_text=user_text,
                    )
                verified_gripper_open = True
                verified_gripper_closed = False
                if run_ar_rviz and physical_available:
                    physical_output = await self._execute_verified_gripper_direct(
                        "moveit_open_gripper",
                        robot_name=robot_name,
                        timeout_s=timeout_s,
                    )
                    if not _tool_ok(physical_output):
                        physical_available = False
                        real_robot_result = _physical_task_failed_result(
                            failed_stage="open_gripper",
                            failed_tool_name="moveit_open_gripper",
                            failed_tool_arguments=open_arguments,
                            failed_tool_result=_json_payload(physical_output),
                            verified_plan_names=physical_plan_names,
                        )
                completed_steps.append({"name": step_name, "handler": step_handler})
                continue
            if step_handler == "attach_object":
                if not verified_gripper_closed:
                    return _task_plan_error(
                        "Task plan attach_object requires verified gripper close evidence.",
                        "Execute a backend contract with close_gripper before attach_object.",
                        retryable=False,
                        suggested_next_tool=None,
                    )
                object_name = _task_plan_step_object_name(step, recent.object_name)
                if object_name is None:
                    return _task_plan_error(
                        "Task plan attach_object requires an object_name.",
                        "Plan a supported pick/place task again with object fields.",
                        retryable=False,
                        suggested_next_tool=None,
                    )
                attach_arguments = {
                    "robot_name": robot_name,
                    "object_name": object_name,
                    "verified_gripper_closed": True,
                }
                attach_output = await self._execute_contract_mcp_tool(
                    "moveit_attach_object",
                    attach_arguments,
                )
                if not _tool_ok(attach_output):
                    return await self._task_plan_stage_failure(
                        stage="attach_object",
                        step_name=step_name,
                        output=attach_output,
                        failed_tool_name="moveit_attach_object",
                        failed_tool_arguments=attach_arguments,
                        task_solution_id=task_solution_id,
                        recent=recent,
                        completed_steps=completed_steps,
                        verified_plan_names=ar_rviz_plan_names,
                        attached_object_verified=attached_object_verified,
                        released_object_verified=released_object_verified,
                        available_tool_names=available_tool_names,
                        user_text=user_text,
                    )
                completed_steps.append({"name": step_name, "handler": step_handler})
                continue
            if step_handler == "release_object":
                if not verified_gripper_open:
                    return _task_plan_error(
                        "Task plan release_object requires verified gripper open evidence.",
                        "Execute a backend contract with open_gripper before release_object.",
                        retryable=False,
                        suggested_next_tool=None,
                    )
                release_tool = _task_plan_step_tool(step, default="moveit_release_object")
                if release_tool not in contract_tool_names:
                    return _task_plan_error(
                        f"Task plan release_object requires unavailable tool: {release_tool}.",
                        "Expose the release/detach MCP tool in the bridge, then replan.",
                        retryable=False,
                        suggested_next_tool=None,
                    )
                object_name = _task_plan_step_object_name(step, recent.object_name)
                if object_name is None:
                    return _task_plan_error(
                        "Task plan release_object requires an object_name.",
                        "Plan a supported pick/place task again with object fields.",
                        retryable=False,
                        suggested_next_tool=None,
                    )
                release_args = _task_plan_step_arguments(step)
                release_args.update(
                    {
                        "robot_name": robot_name,
                        "object_name": object_name,
                        "verified_gripper_open": True,
                    }
                )
                release_output = await self._execute_contract_mcp_tool(
                    release_tool,
                    release_args,
                )
                if not _tool_ok(release_output):
                    return await self._task_plan_stage_failure(
                        stage="release_object",
                        step_name=step_name,
                        output=release_output,
                        failed_tool_name=release_tool,
                        failed_tool_arguments=release_args,
                        task_solution_id=task_solution_id,
                        recent=recent,
                        completed_steps=completed_steps,
                        verified_plan_names=ar_rviz_plan_names,
                        attached_object_verified=attached_object_verified,
                        released_object_verified=released_object_verified,
                        available_tool_names=available_tool_names,
                        user_text=user_text,
                    )
                completed_steps.append({"name": step_name, "handler": step_handler})
                continue
            if step_handler == "verify_attached_object":
                object_name = _task_plan_step_object_name(step, recent.object_name)
                if object_name is None:
                    return _task_plan_error(
                        "Task plan verify_attached_object requires an object_name.",
                        "Plan a supported pick/place task again with object fields.",
                        retryable=False,
                        suggested_next_tool=None,
                    )
                verify_arguments = {
                    "robot_name": robot_name,
                    "object_name": object_name,
                    "timeout_s": timeout_s,
                }
                verify_output = await self._execute_contract_mcp_tool(
                    "moveit_verify_attached_object",
                    verify_arguments,
                )
                if not _tool_ok(verify_output):
                    return await self._task_plan_stage_failure(
                        stage="verify_attached_object",
                        step_name=step_name,
                        output=verify_output,
                        failed_tool_name="moveit_verify_attached_object",
                        failed_tool_arguments=verify_arguments,
                        task_solution_id=task_solution_id,
                        recent=recent,
                        completed_steps=completed_steps,
                        verified_plan_names=ar_rviz_plan_names,
                        attached_object_verified=attached_object_verified,
                        released_object_verified=released_object_verified,
                        available_tool_names=available_tool_names,
                        user_text=user_text,
                    )
                task_verified = True
                attached_object_verified = True
                completed_steps.append({"name": step_name, "handler": step_handler})
                continue
            if step_handler == "verify_released_object":
                object_name = _task_plan_step_object_name(step, recent.object_name)
                if object_name is None:
                    return _task_plan_error(
                        "Task plan verify_released_object requires an object_name.",
                        "Plan a supported pick/place task again with object fields.",
                        retryable=False,
                        suggested_next_tool=None,
                    )
                verify_tool = _task_plan_step_tool(step, default="moveit_verify_released_object")
                verify_args = _task_plan_step_arguments(step)
                verify_args.update(
                    {"robot_name": robot_name, "object_name": object_name, "timeout_s": timeout_s}
                )
                if verify_tool not in contract_tool_names:
                    return _task_plan_error(
                        f"Task plan verify_released_object requires unavailable tool: {verify_tool}.",
                        "Expose the release verification MCP tool in the bridge, then replan.",
                        retryable=False,
                        suggested_next_tool=None,
                    )
                verify_output = await self._execute_contract_mcp_tool(
                    verify_tool,
                    verify_args,
                )
                if not _tool_ok(verify_output):
                    return await self._task_plan_stage_failure(
                        stage="verify_released_object",
                        step_name=step_name,
                        output=verify_output,
                        failed_tool_name=verify_tool,
                        failed_tool_arguments=verify_args,
                        task_solution_id=task_solution_id,
                        recent=recent,
                        completed_steps=completed_steps,
                        verified_plan_names=ar_rviz_plan_names,
                        attached_object_verified=attached_object_verified,
                        released_object_verified=released_object_verified,
                        available_tool_names=available_tool_names,
                        user_text=user_text,
                    )
                if not _release_verification_succeeded(verify_output, object_name):
                    return await self._task_plan_stage_failure(
                        stage="verify_released_object",
                        step_name=step_name,
                        output=verify_output,
                        failed_tool_name=verify_tool,
                        failed_tool_arguments=verify_args,
                        task_solution_id=task_solution_id,
                        recent=recent,
                        completed_steps=completed_steps,
                        verified_plan_names=ar_rviz_plan_names,
                        attached_object_verified=attached_object_verified,
                        released_object_verified=released_object_verified,
                        available_tool_names=available_tool_names,
                        user_text=user_text,
                    )
                task_verified = True
                released_object_verified = True
                release_verification = {"result": "pass"}
                release_proof_output = verify_output
                released_object_name = object_name
                completed_steps.append({"name": step_name, "handler": step_handler})
                continue
            return _task_plan_error(
                f"Task plan workflow contains an unsupported step handler: {step_handler}.",
                "Plan a supported pick/place task again, then retry moveit_execute_task_plan.",
                retryable=False,
                suggested_next_tool=None,
            )

        if not task_verified:
            return _task_plan_error(
                "Task plan execution requires a backend verification step.",
                "Plan a supported pick/place task again with attachment or release verification.",
                retryable=False,
                suggested_next_tool=None,
            )
        simulation_result: dict[str, Any] = {
            "ok": True,
            "tool": public_tool_name,
            "status": "executed",
            "completed_steps": [dict(step) for step in completed_steps],
            "verified_plan_names": list(ar_rviz_plan_names),
            "verification": {"result": "pass"},
        }
        if release_verification is not None:
            simulation_result["release_verification"] = release_verification
        if real_robot_result.get("status") == "executed":
            real_robot_result["verified_plan_names"] = list(physical_plan_names)
        structured_content: dict[str, Any] = {
            "ok": True,
            "tool": public_tool_name,
            "robot_name": robot_name,
            "task_solution_id": task_solution_id,
            "object_name": recent.object_name,
            "simulation": simulation_result,
            "real_robot": real_robot_result,
            "verification": {"result": "pass"},
        }
        if released_object_verified and release_proof_output is not None:
            update_reason = (
                "verified_pick_place_release"
                if recent.task_kind == "pick_place"
                else "verified_place_release"
            )
            physical_model_update = await self._update_physical_model_pose_after_release(
                object_name=released_object_name or recent.object_name,
                reason=update_reason,
                release_proof_output=release_proof_output,
            )
            structured_content["physical_model_update"] = physical_model_update
        return json.dumps(
            {
                "content": [_task_execution_content_text(structured_content)],
                "structured_content": structured_content,
                "is_error": False,
            },
            ensure_ascii=False,
        )

    async def _physical_task_readiness_result(
        self,
        *,
        timeout_s: float,
        requires_gripper: bool,
    ) -> tuple[bool, dict[str, Any]]:
        if self._verified_execution_client is None:
            return False, _physical_task_unavailable_result(
                "Verified real robot execution is not connected."
            )
        get_readiness = getattr(self._verified_execution_client, "get_readiness", None)
        if not callable(get_readiness):
            return False, _physical_task_unavailable_result(
                "Verified real robot readiness is unavailable."
            )
        try:
            readiness = await cast(Callable[..., Awaitable[Any]], get_readiness)(
                timeout_s=timeout_s,
            )
        except Exception as exc:
            return False, _physical_task_unavailable_result(
                "Verified real robot execution is not responsive.",
                error=str(exc),
            )
        if not isinstance(readiness, dict):
            return False, _physical_task_unavailable_result(
                "Verified real robot readiness returned an unreadable result.",
                readiness=readiness,
            )
        if readiness.get("server_available") is False:
            return False, _physical_task_unavailable_result(
                "Verified real robot execution is unavailable.",
                readiness=readiness,
            )
        if readiness.get("robot_connected") is False:
            return False, _physical_task_unavailable_result(
                "Verified real robot is not connected.",
                readiness=readiness,
            )
        if requires_gripper and readiness.get("gripper_connected") is False:
            return False, _physical_task_unavailable_result(
                "Verified real robot gripper is not connected.",
                readiness=readiness,
            )
        return True, {
            "ok": True,
            "status": "executed",
            "message": "Verified physical execution completed.",
            "readiness": readiness,
        }

    async def _execute_verified_gripper_direct(
        self,
        name: str,
        *,
        robot_name: str,
        timeout_s: float,
    ) -> str:
        assert self._verified_execution_client is not None
        try:
            if name == "moveit_close_gripper":
                output = await self._verified_execution_client.close_gripper(
                    robot_name=robot_name,
                    timeout_s=timeout_s,
                )
            else:
                output = await self._verified_execution_client.open_gripper(
                    robot_name=robot_name,
                    timeout_s=timeout_s,
                )
        except Exception as exc:
            return json.dumps(
                {
                    "structured_content": {
                        "ok": False,
                        "robot": robot_name,
                        "tool": name,
                        "status": "failed",
                        "error": str(exc),
                        "verification": {"result": "fail"},
                    },
                    "is_error": True,
                },
                ensure_ascii=False,
            )
        output_json = verified_execution_output_to_json(output)
        self._robot_context.update_from_tool_result(name, output_json)
        return output_json

    async def _execute_verified_task_plan_tool(
        self,
        arguments: dict[str, Any],
        *,
        user_text: str | None,
        allow_execution: bool = True,
    ) -> str:
        try:
            validate_robot_tool_call("moveit_execute_task_plan", arguments)
        except RobotCallValidationError as exc:
            return json.dumps(structured_robot_call_error(exc), ensure_ascii=False)
        if self._verified_execution_client is None:
            exc = RobotCallValidationError(
                "Verified Real Robot Execution client is unavailable.",
                correction="Start or configure the verified execution server, then retry.",
            )
            return json.dumps(
                structured_robot_call_error(exc, retryable=True, suggested_next_tool=None),
                ensure_ascii=False,
            )
        task_solution_id = str(arguments["task_solution_id"]).strip()
        robot_name = str(arguments.get("robot_name") or VIZOR_ROBOT_NAME)
        timeout_s = float(arguments.get("timeout_s") or VERIFIED_EXECUTION_DEFAULT_TIMEOUT_S)
        recent = self._robot_context.recent_task_solution
        if recent is None or recent.task_solution_id != task_solution_id:
            return _task_plan_error(
                "Task plan execution requires the exact recent task_solution_id.",
                "Plan the compound task again, then retry moveit_execute_task_plan with that task_solution_id.",
                suggested_next_tool=TASK_LEVEL_REPLAN_TOOL_NAME,
            )
        if recent.task_kind not in SUPPORTED_TASK_SOLUTION_KINDS:
            return _task_plan_error(
                f"Task plan execution does not support task kind: {recent.task_kind}.",
                "Plan a supported pick/place task, then retry moveit_execute_task_plan.",
                retryable=False,
                suggested_next_tool=None,
            )
        raw = recent.raw
        if raw is None:
            return _task_plan_error(
                "Task plan execution requires the recent raw task solution.",
                "Plan the compound task again, then retry moveit_execute_task_plan with that task_solution_id.",
                suggested_next_tool=TASK_LEVEL_REPLAN_TOOL_NAME,
            )
        execution_steps, execution_steps_error = _task_plan_execution_steps(
            raw,
            task_kind=recent.task_kind,
        )
        if execution_steps_error is not None:
            return execution_steps_error
        assert execution_steps is not None
        _remember_task_solution_execution_contract_steps(
            self._robot_context,
            recent=recent,
            execution_steps=execution_steps,
        )
        self._record_task_solution_approval_if_explicit(
            arguments,
            user_text=user_text,
            allow_execution=allow_execution,
        )
        try:
            ensure_task_solution_execution_allowed(self._robot_context, arguments)
        except RobotCallValidationError as exc:
            return json.dumps(structured_robot_call_error(exc), ensure_ascii=False)

        execution_id = uuid.uuid4().hex[:8]
        verified_plan_names: list[str] = []
        completed_steps: list[dict[str, Any]] = []
        verified_gripper_closed = False
        verified_gripper_open = False
        task_verified = False
        attached_object_verified = False
        released_object_verified = False
        release_verification: dict[str, str] | None = None
        release_proof_output: str | None = None
        released_object_name: str | None = None
        available_tool_names = _function_tool_names(self._tool_bridge.function_tools())
        contract_tool_names = _contract_tool_names(self._tool_bridge)
        for step in execution_steps:
            step_name = str(step.get("name") or step.get("tool") or "")
            step_handler = str(step["handler"])
            if step_handler == "motion":
                waypoint = _task_plan_step_waypoint(step, raw)
                if waypoint is None:
                    return _task_plan_error(
                        "Task plan motion step references a missing waypoint.",
                        "Plan a supported pick/place task again, then retry moveit_execute_task_plan.",
                        suggested_next_tool=None,
                    )
                pose_output = await self._execute_task_plan_pose_observation(
                    robot_name=robot_name,
                    user_text=user_text,
                    allow_execution=allow_execution,
                )
                if not _tool_ok(pose_output):
                    return await self._task_plan_stage_failure(
                        stage="observe_current_pose",
                        step_name=step_name,
                        output=pose_output,
                        failed_tool_name="moveit_get_current_pose",
                        failed_tool_arguments=_task_plan_pose_observation_arguments(robot_name),
                        task_solution_id=task_solution_id,
                        recent=recent,
                        completed_steps=completed_steps,
                        verified_plan_names=verified_plan_names,
                        attached_object_verified=attached_object_verified,
                        released_object_verified=released_object_verified,
                        available_tool_names=available_tool_names,
                        user_text=user_text,
                    )
                base_plan_name = f"{task_solution_id}_{_task_plan_step_label(step_name)}"
                stage_succeeded = False
                last_failed_stage = "planning"
                last_failed_output = ""
                last_failed_tool_name: str | None = None
                last_failed_tool_arguments: dict[str, Any] | None = None
                for attempt_index in range(1, TASK_PLAN_STAGE_MAX_ATTEMPTS + 1):
                    plan_name = _task_plan_attempt_name(
                        base_plan_name,
                        execution_id=execution_id,
                        attempt_index=attempt_index,
                    )
                    planning_tool, planning_args = _task_plan_motion_call(
                        step_name,
                        attempt_index=attempt_index,
                        robot_name=robot_name,
                        plan_name=plan_name,
                        waypoint=waypoint,
                        timeout_s=timeout_s,
                    )
                    plan_output = await self._execute_tool(
                        planning_tool,
                        planning_args,
                        user_text=None,
                        allow_execution=allow_execution,
                    )
                    if not _tool_ok(plan_output):
                        last_failed_stage = "planning"
                        last_failed_output = plan_output
                        last_failed_tool_name = planning_tool
                        last_failed_tool_arguments = planning_args
                        if attempt_index < TASK_PLAN_STAGE_MAX_ATTEMPTS:
                            pose_output = await self._execute_task_plan_pose_observation(
                                robot_name=robot_name,
                                user_text=user_text,
                                allow_execution=allow_execution,
                            )
                            if not _tool_ok(pose_output):
                                return await self._task_plan_stage_failure(
                                    stage="observe_current_pose",
                                    step_name=step_name,
                                    output=pose_output,
                                    failed_tool_name="moveit_get_current_pose",
                                    failed_tool_arguments=_task_plan_pose_observation_arguments(robot_name),
                                    task_solution_id=task_solution_id,
                                    recent=recent,
                                    completed_steps=completed_steps,
                                    verified_plan_names=verified_plan_names,
                                    attached_object_verified=attached_object_verified,
                                    released_object_verified=released_object_verified,
                                    available_tool_names=available_tool_names,
                                    user_text=user_text,
                                )
                        continue
                    executable_name = executable_plan_name(plan_output) or plan_name
                    execute_output = await self._execute_verified_plan_tool(
                        {
                            "robot_name": robot_name,
                            "plan_name": executable_name,
                            "timeout_s": timeout_s,
                        },
                        user_text=user_text,
                        allow_execution=allow_execution,
                    )
                    self._robot_context.consume_executable_plan(plan_name)
                    if executable_name != plan_name:
                        self._robot_context.consume_executable_plan(executable_name)
                    if not _execution_succeeded(execute_output):
                        last_failed_stage = "verified_execution"
                        last_failed_output = execute_output
                        last_failed_tool_name = "moveit_execute_plan"
                        last_failed_tool_arguments = {
                            "robot_name": robot_name,
                            "plan_name": executable_name,
                            "timeout_s": timeout_s,
                        }
                        continue
                    verified_plan_names.append(executable_name)
                    completed_steps.append({"name": step_name, "handler": step_handler})
                    stage_succeeded = True
                    break
                if not stage_succeeded:
                    return await self._task_plan_stage_failure(
                        stage=last_failed_stage,
                        step_name=step_name,
                        output=last_failed_output,
                        failed_tool_name=last_failed_tool_name,
                        failed_tool_arguments=last_failed_tool_arguments,
                        task_solution_id=task_solution_id,
                        recent=recent,
                        completed_steps=completed_steps,
                        verified_plan_names=verified_plan_names,
                        attached_object_verified=attached_object_verified,
                        released_object_verified=released_object_verified,
                        available_tool_names=available_tool_names,
                        user_text=user_text,
                    )
                continue
            if step_handler == "close_gripper":
                close_output = verified_execution_output_to_json(
                    await self._verified_execution_client.close_gripper(
                        robot_name=robot_name,
                        timeout_s=timeout_s,
                    )
                )
                if not _tool_ok(close_output):
                    return await self._task_plan_stage_failure(
                        stage="close_gripper",
                        step_name=step_name,
                        output=close_output,
                        failed_tool_name="moveit_close_gripper",
                        failed_tool_arguments={"robot_name": robot_name, "timeout_s": timeout_s},
                        task_solution_id=task_solution_id,
                        recent=recent,
                        completed_steps=completed_steps,
                        verified_plan_names=verified_plan_names,
                        attached_object_verified=attached_object_verified,
                        released_object_verified=released_object_verified,
                        available_tool_names=available_tool_names,
                        user_text=user_text,
                    )
                self._robot_context.update_from_tool_result(
                    "moveit_close_gripper",
                    close_output,
                )
                verified_gripper_closed = True
                verified_gripper_open = False
                completed_steps.append({"name": step_name, "handler": step_handler})
                continue
            if step_handler == "open_gripper":
                open_output = verified_execution_output_to_json(
                    await self._verified_execution_client.open_gripper(
                        robot_name=robot_name,
                        timeout_s=timeout_s,
                    )
                )
                if not _tool_ok(open_output):
                    return await self._task_plan_stage_failure(
                        stage="open_gripper",
                        step_name=step_name,
                        output=open_output,
                        failed_tool_name="moveit_open_gripper",
                        failed_tool_arguments={"robot_name": robot_name, "timeout_s": timeout_s},
                        task_solution_id=task_solution_id,
                        recent=recent,
                        completed_steps=completed_steps,
                        verified_plan_names=verified_plan_names,
                        attached_object_verified=attached_object_verified,
                        released_object_verified=released_object_verified,
                        available_tool_names=available_tool_names,
                        user_text=user_text,
                    )
                self._robot_context.update_from_tool_result(
                    "moveit_open_gripper",
                    open_output,
                )
                verified_gripper_open = True
                verified_gripper_closed = False
                completed_steps.append({"name": step_name, "handler": step_handler})
                continue
            if step_handler == "attach_object":
                if not verified_gripper_closed:
                    return _task_plan_error(
                        "Task plan attach_object requires verified gripper close evidence.",
                        "Execute a backend contract with close_gripper before attach_object.",
                        retryable=False,
                        suggested_next_tool=None,
                    )
                object_name = _task_plan_step_object_name(step, recent.object_name)
                if object_name is None:
                    return _task_plan_error(
                        "Task plan attach_object requires an object_name.",
                        "Plan a supported pick/place task again with object fields.",
                        retryable=False,
                        suggested_next_tool=None,
                    )
                attach_output = await self._execute_tool(
                    "moveit_attach_object",
                    {
                        "robot_name": robot_name,
                        "object_name": object_name,
                        "verified_gripper_closed": True,
                    },
                    user_text=user_text,
                    allow_execution=allow_execution,
                )
                if not _tool_ok(attach_output):
                    return await self._task_plan_stage_failure(
                        stage="attach_object",
                        step_name=step_name,
                        output=attach_output,
                        failed_tool_name="moveit_attach_object",
                        failed_tool_arguments={
                            "robot_name": robot_name,
                            "object_name": object_name,
                            "verified_gripper_closed": True,
                        },
                        task_solution_id=task_solution_id,
                        recent=recent,
                        completed_steps=completed_steps,
                        verified_plan_names=verified_plan_names,
                        attached_object_verified=attached_object_verified,
                        released_object_verified=released_object_verified,
                        available_tool_names=available_tool_names,
                        user_text=user_text,
                    )
                completed_steps.append({"name": step_name, "handler": step_handler})
                continue
            if step_handler == "release_object":
                if not verified_gripper_open:
                    return _task_plan_error(
                        "Task plan release_object requires verified gripper open evidence.",
                        "Execute a backend contract with open_gripper before release_object.",
                        retryable=False,
                        suggested_next_tool=None,
                    )
                release_tool = _task_plan_step_tool(step, default="moveit_release_object")
                if release_tool not in contract_tool_names:
                    return _task_plan_error(
                        f"Task plan release_object requires unavailable tool: {release_tool}.",
                        "Expose the release/detach MCP tool in the bridge, then replan.",
                        retryable=False,
                        suggested_next_tool=None,
                    )
                object_name = _task_plan_step_object_name(step, recent.object_name)
                if object_name is None:
                    return _task_plan_error(
                        "Task plan release_object requires an object_name.",
                        "Plan a supported pick/place task again with object fields.",
                        retryable=False,
                        suggested_next_tool=None,
                    )
                release_args = _task_plan_step_arguments(step)
                release_args.update(
                    {
                        "robot_name": robot_name,
                        "object_name": object_name,
                        "verified_gripper_open": True,
                    }
                )
                release_output = await self._execute_contract_mcp_tool(
                    release_tool,
                    release_args,
                )
                if not _tool_ok(release_output):
                    return await self._task_plan_stage_failure(
                        stage="release_object",
                        step_name=step_name,
                        output=release_output,
                        failed_tool_name=release_tool,
                        failed_tool_arguments=release_args,
                        task_solution_id=task_solution_id,
                        recent=recent,
                        completed_steps=completed_steps,
                        verified_plan_names=verified_plan_names,
                        attached_object_verified=attached_object_verified,
                        released_object_verified=released_object_verified,
                        available_tool_names=available_tool_names,
                        user_text=user_text,
                    )
                completed_steps.append({"name": step_name, "handler": step_handler})
                continue
            if step_handler == "verify_attached_object":
                object_name = _task_plan_step_object_name(step, recent.object_name)
                if object_name is None:
                    return _task_plan_error(
                        "Task plan verify_attached_object requires an object_name.",
                        "Plan a supported pick/place task again with object fields.",
                        retryable=False,
                        suggested_next_tool=None,
                    )
                verify_output = await self._execute_tool(
                    "moveit_verify_attached_object",
                    {"robot_name": robot_name, "object_name": object_name, "timeout_s": timeout_s},
                    user_text=user_text,
                    allow_execution=allow_execution,
                )
                if not _tool_ok(verify_output):
                    return await self._task_plan_stage_failure(
                        stage="verify_attached_object",
                        step_name=step_name,
                        output=verify_output,
                        failed_tool_name="moveit_verify_attached_object",
                        failed_tool_arguments={
                            "robot_name": robot_name,
                            "object_name": object_name,
                            "timeout_s": timeout_s,
                        },
                        task_solution_id=task_solution_id,
                        recent=recent,
                        completed_steps=completed_steps,
                        verified_plan_names=verified_plan_names,
                        attached_object_verified=attached_object_verified,
                        released_object_verified=released_object_verified,
                        available_tool_names=available_tool_names,
                        user_text=user_text,
                    )
                task_verified = True
                attached_object_verified = True
                completed_steps.append({"name": step_name, "handler": step_handler})
                continue
            if step_handler == "verify_released_object":
                object_name = _task_plan_step_object_name(step, recent.object_name)
                if object_name is None:
                    return _task_plan_error(
                        "Task plan verify_released_object requires an object_name.",
                        "Plan a supported pick/place task again with object fields.",
                        retryable=False,
                        suggested_next_tool=None,
                    )
                verify_tool = _task_plan_step_tool(step, default="moveit_verify_released_object")
                verify_args = _task_plan_step_arguments(step)
                verify_args.update(
                    {"robot_name": robot_name, "object_name": object_name, "timeout_s": timeout_s}
                )
                if verify_tool not in contract_tool_names:
                    return _task_plan_error(
                        f"Task plan verify_released_object requires unavailable tool: {verify_tool}.",
                        "Expose the release verification MCP tool in the bridge, then replan.",
                        retryable=False,
                        suggested_next_tool=None,
                    )
                verify_output = await self._execute_contract_mcp_tool(
                    verify_tool,
                    verify_args,
                )
                if not _tool_ok(verify_output):
                    return await self._task_plan_stage_failure(
                        stage="verify_released_object",
                        step_name=step_name,
                        output=verify_output,
                        failed_tool_name=verify_tool,
                        failed_tool_arguments=verify_args,
                        task_solution_id=task_solution_id,
                        recent=recent,
                        completed_steps=completed_steps,
                        verified_plan_names=verified_plan_names,
                        attached_object_verified=attached_object_verified,
                        released_object_verified=released_object_verified,
                        available_tool_names=available_tool_names,
                        user_text=user_text,
                    )
                if not _release_verification_succeeded(verify_output, object_name):
                    return await self._task_plan_stage_failure(
                        stage="verify_released_object",
                        step_name=step_name,
                        output=verify_output,
                        failed_tool_name=verify_tool,
                        failed_tool_arguments=verify_args,
                        task_solution_id=task_solution_id,
                        recent=recent,
                        completed_steps=completed_steps,
                        verified_plan_names=verified_plan_names,
                        attached_object_verified=attached_object_verified,
                        released_object_verified=released_object_verified,
                        available_tool_names=available_tool_names,
                        user_text=user_text,
                    )
                task_verified = True
                released_object_verified = True
                release_verification = {"result": "pass"}
                release_proof_output = verify_output
                released_object_name = object_name
                completed_steps.append({"name": step_name, "handler": step_handler})
                continue
            return _task_plan_error(
                f"Task plan workflow contains an unsupported step handler: {step_handler}.",
                "Plan a supported pick/place task again, then retry moveit_execute_task_plan.",
                retryable=False,
                suggested_next_tool=None,
            )

        if not task_verified:
            return _task_plan_error(
                "Task plan execution requires a backend verification step.",
                "Plan a supported pick/place task again with attachment or release verification.",
                retryable=False,
                suggested_next_tool=None,
            )
        structured_content: dict[str, Any] = {
            "ok": True,
            "tool": "moveit_execute_task_plan",
            "task_solution_id": task_solution_id,
            "object_name": recent.object_name,
            "verified_plan_names": verified_plan_names,
            "verification": {"result": "pass"},
        }
        if release_verification is not None:
            structured_content["release_verification"] = release_verification
        if released_object_verified and release_proof_output is not None:
            update_reason = (
                "verified_pick_place_release"
                if recent.task_kind == "pick_place"
                else "verified_place_release"
            )
            physical_model_update = await self._update_physical_model_pose_after_release(
                object_name=released_object_name or recent.object_name,
                reason=update_reason,
                release_proof_output=release_proof_output,
            )
            structured_content["physical_model_update"] = physical_model_update
        return json.dumps(
            {
                "content": ["Verified task plan execution completed."],
                "structured_content": structured_content,
                "is_error": False,
            },
            ensure_ascii=False,
        )

    async def _execute_contract_mcp_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> str:
        try:
            contract_call = getattr(self._tool_bridge, "call_contract_tool", None)
            if callable(contract_call):
                output = await cast(
                    Callable[[str, dict[str, Any]], Awaitable[str]],
                    contract_call,
                )(name, arguments)
            else:
                output = await self._tool_bridge.call_tool(name, arguments)
        except RobotMCPError as exc:
            validation_error = RobotCallValidationError(
                str(exc),
                correction="Check the robot control server, then retry the robot action.",
            )
            return json.dumps(structured_robot_call_error(validation_error), ensure_ascii=False)
        self._robot_context.update_from_tool_result(name, output)
        self._tracer.event(
            "robot.context_update",
            "robot_control",
            attributes={"tool.name": name},
        )
        return output

    async def _execute_task_plan_pose_observation(
        self,
        *,
        robot_name: str,
        user_text: str | None,
        allow_execution: bool,
    ) -> str:
        output = ""
        for attempt_index in range(1, TASK_PLAN_OBSERVATION_MAX_ATTEMPTS + 1):
            output = await self._execute_tool(
                "moveit_get_current_pose",
                _task_plan_pose_observation_arguments(robot_name),
                user_text=user_text,
                allow_execution=allow_execution,
            )
            if _tool_ok(output):
                return output
            if attempt_index < TASK_PLAN_OBSERVATION_MAX_ATTEMPTS:
                await asyncio.sleep(TASK_PLAN_OBSERVATION_RETRY_DELAY_S)
        return output

    async def _update_physical_model_pose_after_release(
        self,
        *,
        object_name: str,
        reason: str,
        release_proof_output: str,
    ) -> dict[str, Any]:
        pose_evidence = _pose_evidence_from_output(
            release_proof_output,
            object_name=object_name,
            source="moveit_verify_released_object",
        )
        if pose_evidence is None:
            return _physical_model_update_failure(
                "Full object pose evidence was not found in verified release proof.",
                "Use release proof with object position and orientation, or run an explicit operator sync.",
                retryable=True,
            )
        try:
            result = update_physical_model_pose(object_name, reason, pose_evidence)
        except Exception as exc:
            return _physical_model_update_failure(
                f"physical model update failed: {exc}",
                "Check the physical model file and retry the sync.",
                retryable=True,
            )
        if isinstance(result, dict):
            return result
        return _physical_model_update_failure(
            "physical model update returned a non-object result",
            "Check the physical model update helper.",
            retryable=False,
        )

    async def _task_plan_stage_failure(
        self,
        *,
        stage: str,
        step_name: str,
        output: str,
        failed_tool_name: str | None,
        failed_tool_arguments: dict[str, Any] | None,
        task_solution_id: str,
        recent: Any,
        completed_steps: list[dict[str, Any]],
        verified_plan_names: list[str],
        attached_object_verified: bool,
        released_object_verified: bool,
        available_tool_names: set[str],
        user_text: str | None,
    ) -> str:
        failed_tool_result = _json_payload(output)
        recovery = {
            "task_solution_id": task_solution_id,
            "task_kind": recent.task_kind,
            "object_name": recent.object_name,
            "scene_snapshot_id": recent.scene_snapshot_id,
            "failed_step": step_name,
            "failed_stage": stage,
            "failed_tool_name": failed_tool_name,
            "failed_tool_arguments": dict(failed_tool_arguments or {}),
            "failed_tool_result": failed_tool_result,
            "completed_steps": [dict(step) for step in completed_steps],
            "verified_plan_names": list(verified_plan_names),
            "gripper_state": self._robot_context.gripper_state(),
            "attached_object_verified": attached_object_verified,
            "released_object_verified": released_object_verified,
        }
        self._robot_context.remember_task_failure(
            task_solution_id=task_solution_id,
            task_kind=recent.task_kind,
            object_name=recent.object_name,
            scene_snapshot_id=recent.scene_snapshot_id,
            failed_step=step_name,
            failed_stage=stage,
            failed_tool_name=failed_tool_name,
            failed_tool_arguments=dict(failed_tool_arguments or {}),
            failed_tool_result=failed_tool_result,
            completed_steps=[dict(step) for step in completed_steps],
            verified_plan_names=list(verified_plan_names),
            gripper_state=self._robot_context.gripper_state(),
            attached_object_verified=attached_object_verified,
            released_object_verified=released_object_verified,
        )
        diagnostic: Any | None = None
        if (
            failed_tool_name is not None
            and failed_tool_name != "moveit_explain_motion_failure"
            and "moveit_explain_motion_failure" in available_tool_names
        ):
            explain_args: dict[str, Any] = {
                "failed_tool_name": failed_tool_name,
                "failed_tool_result": failed_tool_result,
            }
            if failed_tool_arguments is not None:
                explain_args["failed_tool_arguments"] = dict(failed_tool_arguments)
                if "timeout_s" in failed_tool_arguments:
                    explain_args["timeout_s"] = failed_tool_arguments["timeout_s"]
            if isinstance(user_text, str) and user_text.strip():
                explain_args["user_intent"] = user_text
            diagnostic_output = await self._execute_contract_mcp_tool(
                "moveit_explain_motion_failure",
                explain_args,
            )
            diagnostic = _json_payload(diagnostic_output)
            recovery["diagnostic"] = diagnostic
        return _task_plan_stage_error(
            stage,
            step_name,
            output,
            task_solution_id=task_solution_id,
            failed_tool_name=failed_tool_name,
            failed_tool_arguments=failed_tool_arguments,
            recovery=recovery,
            diagnostic=diagnostic,
        )

    async def _execute_verified_plan_direct(
        self,
        *,
        robot_name: str,
        plan_name: str,
        timeout_s: float,
    ) -> str:
        assert self._verified_execution_client is not None
        trace_attributes: dict[str, Any] = {
            "plan_name": plan_name,
            "robot_name": robot_name,
            "timeout_s": timeout_s,
        }
        async with self._tracer.span(
            "robot.verified_execution.execute_plan",
            "robot_control",
            attributes=trace_attributes,
        ):
            try:
                output = await self._verified_execution_client.execute_plan(
                    robot_name=robot_name,
                    plan_name=plan_name,
                    timeout_s=timeout_s,
                )
                output_json = verified_execution_output_to_json(output)
            except Exception as exc:
                output_json = json.dumps(
                    {
                        "structured_content": {
                            "ok": False,
                            "robot": robot_name,
                            "tool": "moveit_execute_plan",
                            "phase": "pre_execute",
                            "status": "failed",
                            "error": str(exc),
                            "feedback": {"plan_name": plan_name},
                            "verification": {"result": "fail"},
                            "raw": {"plan_name": plan_name},
                        },
                        "is_error": True,
                    },
                    ensure_ascii=False,
                )
            try:
                output_payload = json.loads(output_json)
            except json.JSONDecodeError:
                output_payload = None
            structured_content = (
                output_payload.get("structured_content")
                if isinstance(output_payload, dict)
                else None
            )
            if isinstance(structured_content, dict):
                trace_attributes["execute.status"] = structured_content.get("status")
                trace_attributes["execute.ok"] = structured_content.get("ok")
                feedback = structured_content.get("feedback")
                if isinstance(feedback, dict) and isinstance(
                    feedback.get("state_sync_published"), bool
                ):
                    trace_attributes["state_sync_published"] = feedback[
                        "state_sync_published"
                    ]
        self._robot_context.update_from_tool_result("moveit_execute_plan", output_json)
        return output_json

    async def _execute_verified_plan_tool(
        self,
        arguments: dict[str, Any],
        *,
        user_text: str | None,
        allow_execution: bool = True,
    ) -> str:
        try:
            validate_robot_tool_call("moveit_execute_plan", arguments)
        except RobotCallValidationError as exc:
            return json.dumps(structured_robot_call_error(exc), ensure_ascii=False)
        decision = await self._validate_robot_task_step(
            "moveit_execute_plan",
            arguments,
            user_text=user_text,
            allow_execution=allow_execution,
        )
        if not decision.ok:
            return json.dumps(structured_task_policy_error(decision), ensure_ascii=False)
        assert self._verified_execution_client is not None
        robot_name = str(arguments.get("robot_name") or VIZOR_ROBOT_NAME)
        plan_name = str(arguments.get("plan_name") or "")
        timeout_s = float(arguments.get("timeout_s") or VERIFIED_EXECUTION_DEFAULT_TIMEOUT_S)
        pending = self._robot_context.pending_executable_plan(
            plan_name,
            max_age_s=DEFAULT_EXECUTABLE_PLAN_MAX_AGE_S,
        )
        output_json = await self._execute_verified_plan_direct(
            robot_name=robot_name,
            plan_name=plan_name,
            timeout_s=timeout_s,
        )
        if _execution_succeeded(output_json):
            await self._execute_after_success_action(
                pending.after_success_tool if pending is not None else None,
                pending.after_success_arguments if pending is not None else None,
                user_text=user_text,
                allow_execution=allow_execution,
            )
        return output_json

    async def _execute_verified_recovery_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        user_text: str | None,
        allow_execution: bool = True,
    ) -> str:
        try:
            validate_robot_tool_call(name, arguments)
        except RobotCallValidationError as exc:
            return json.dumps(structured_robot_call_error(exc), ensure_ascii=False)
        if self._verified_execution_client is None:
            exc = RobotCallValidationError(
                "Verified Real Robot Execution client is unavailable.",
                correction="Start or configure the verified execution server, then retry.",
            )
            return json.dumps(
                structured_robot_call_error(exc, retryable=True, suggested_next_tool=None),
                ensure_ascii=False,
            )
        if name == "moveit_go_home" and (
            not allow_execution or not _explicit_go_home_requested(user_text)
        ):
            exc = RobotCallValidationError(
                "Go home requires explicit user/operator intent.",
                correction="Ask the operator for approval before sending the robot home.",
            )
            return json.dumps(
                structured_robot_call_error(exc, retryable=True, suggested_next_tool=None),
                ensure_ascii=False,
            )
        robot_name = str(arguments.get("robot_name") or VIZOR_ROBOT_NAME)
        timeout_s = float(arguments.get("timeout_s") or VERIFIED_EXECUTION_DEFAULT_TIMEOUT_S)
        trace_attributes: dict[str, Any] = {
            "tool.name": name,
            "robot_name": robot_name,
            "timeout_s": timeout_s,
        }
        span_name = (
            "robot.verified_execution.go_home"
            if name == "moveit_go_home"
            else "robot.verified_execution.sync_real_robot_state"
        )
        async with self._tracer.span(
            span_name,
            "robot_control",
            attributes=trace_attributes,
        ):
            if name == "moveit_go_home":
                output = await self._verified_execution_client.go_home(
                    robot_name=robot_name,
                    timeout_s=timeout_s,
                )
            else:
                output = await self._verified_execution_client.sync_real_robot_state(
                    robot_name=robot_name,
                    timeout_s=timeout_s,
                )
            output_json = verified_execution_output_to_json(output)
            result = _structured_result_payload(output_json)
            if result is not None:
                trace_attributes["execute.ok"] = result.get("ok")
                trace_attributes["execute.status"] = result.get("status")
                feedback = result.get("feedback")
                if isinstance(feedback, dict) and isinstance(
                    feedback.get("state_sync_published"), bool
                ):
                    trace_attributes["state_sync_published"] = feedback[
                        "state_sync_published"
                    ]
        return output_json

    async def _execute_after_success_action(
        self,
        name: str | None,
        arguments: dict[str, Any] | None,
        *,
        user_text: str | None,
        allow_execution: bool = True,
    ) -> None:
        if not allow_execution:
            return
        if not isinstance(name, str) or not isinstance(arguments, dict):
            return
        if name not in AFTER_SUCCESS_ACTION_TOOL_NAMES:
            return
        try:
            validate_robot_tool_call(name, arguments)
        except RobotCallValidationError:
            return
        await self._execute_tool(
            name,
            dict(arguments),
            user_text=user_text,
            allow_execution=allow_execution,
        )



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


def _function_tool_names(tools: list[dict[str, Any]]) -> set[str]:
    return {
        str(tool.get("name"))
        for tool in tools
        if isinstance(tool.get("name"), str)
    }


def _contract_tool_names(tool_bridge: Any) -> set[str]:
    bridge_contract_tool_names = getattr(tool_bridge, "contract_tool_names", None)
    if callable(bridge_contract_tool_names):
        names = cast(Callable[[], Iterable[str]], bridge_contract_tool_names)()
        return {
            str(name)
            for name in names
            if isinstance(name, str)
        }
    return set()


def _tools_for_model_binding(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_tool_for_model_binding(tool) for tool in tools]


def _model_visible_task_planner_tool(tools: list[dict[str, Any]]) -> dict[str, Any] | None:
    return _function_tool_by_name(tools, MODEL_VISIBLE_TASK_PLANNER_TOOL_NAME)


def _function_tool_by_name(
    tools: list[dict[str, Any]],
    name: str,
) -> dict[str, Any] | None:
    for tool in tools:
        if tool.get("name") == name:
            return tool
    return None


def _geometry_update_dynamic_role_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "name": GEOMETRY_UPDATE_DYNAMIC_ROLE_TOOL_NAME,
        "description": (
            "Update the local physical model role for a dynamic object after the human "
            "confirms its structural role. Ask the human when structural role is uncertain; "
            "do not infer support relationships from view-dependent wording."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "object_name": {
                    "type": "string",
                    "description": "Canonical dynamic object name such as dynamic_1.",
                },
                "role": {
                    "type": "object",
                    "description": (
                        "Structured role payload, for example {'type': 'unassigned'}, "
                        "{'type': 'supporting_column', 'supports': ['dynamic_2']}, or "
                        "{'type': 'beam_supported_by', 'supported_by': ['dynamic_1']}."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": "Why the role update is justified by the current turn.",
                },
            },
            "required": ["object_name", "role", "reason"],
            "additionalProperties": False,
        },
        "strict": None,
    }


def _verified_recovery_tools() -> list[dict[str, Any]]:
    parameters = {
        "type": "object",
        "properties": {
            "robot_name": {"type": "string"},
            "timeout_s": {"type": "number"},
        },
        "required": ["robot_name"],
        "additionalProperties": False,
    }
    return [
        {
            "type": "function",
            "name": "moveit_go_home",
            "description": agent_tool_description("moveit_go_home"),
            "parameters": dict(parameters),
            "strict": None,
        },
        {
            "type": "function",
            "name": "moveit_sync_real_robot_state",
            "description": agent_tool_description("moveit_sync_real_robot_state"),
            "parameters": dict(parameters),
            "strict": None,
        },
    ]


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


def _should_execute_latest_pending_plan(text: str, robot_context: RobotContextStore) -> bool:
    if not _explicit_execute_requested(text):
        return False
    pending = robot_context.latest_pending_executable_plan(
        max_age_s=DEFAULT_EXECUTABLE_PLAN_MAX_AGE_S
    )
    return pending is not None and not _is_task_stage_attempt_plan_name(pending.plan_name)


def _is_direct_verified_recovery_request(text: str) -> bool:
    normalized = text.casefold()
    if "go home" in normalized:
        return True
    if "sync" in normalized and "robot" in normalized and "state" in normalized:
        return True
    if "align" in normalized and "rviz" in normalized:
        return True
    return False


def _explicit_go_home_requested(text: str | None) -> bool:
    if not text:
        return False
    normalized = text.casefold()
    return (
        "go home" in normalized
        or "send the robot home" in normalized
        or "send robot home" in normalized
        or "return home" in normalized
    )


def _is_task_stage_attempt_plan_name(plan_name: str) -> bool:
    while plan_name.startswith("cached__"):
        plan_name = plan_name.removeprefix("cached__")
    if not (plan_name.startswith("pick_task_") or plan_name.startswith("place_task_")):
        return False
    _, separator, suffix = plan_name.rpartition("_try")
    return bool(separator) and suffix.isdigit()


def _execution_result_text(output: str, plan_name: str) -> str:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return NO_TEXT_RESPONSE
    if not isinstance(payload, dict):
        return NO_TEXT_RESPONSE
    structured = payload.get("structured_content")
    result = structured if isinstance(structured, dict) else payload
    if result.get("ok") is False:
        for key in ("correction", "error"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return NO_TEXT_RESPONSE
    if result.get("status") == "queued":
        return "Execution queued."
    verification = result.get("verification")
    if isinstance(verification, dict) and verification.get("result") == "pass":
        return "Execution complete."
    return NO_TEXT_RESPONSE


def _task_planning_result_text(output: str, *, repair_attempts: int) -> tuple[str, int]:
    result = parse_task_solution_result(MODEL_VISIBLE_TASK_PLANNER_TOOL_NAME, output)
    if result is not None:
        return "Plan ready.", repair_attempts

    structured = _structured_result_payload(output)
    if structured is None:
        return "Planning finished without a readable task solution result.", repair_attempts

    if _manipulation_planner_timed_out(structured):
        return "Planning timed out before a complete task solution was returned.", repair_attempts

    if structured.get("ok") is False or structured.get("is_error") is True:
        if _repairable_manipulation_planner_schema_error(structured) and repair_attempts < 1:
            return "", repair_attempts + 1
        if _manipulation_planner_failure_needs_model_feedback(structured):
            return "", repair_attempts
        return (
            _structured_feedback_text(structured)
            or "Planning failed before a complete task solution was returned.",
            repair_attempts,
        )

    return "Planning finished without a complete task solution.", repair_attempts


def _task_plan_execution_result_text(output: str) -> str:
    result = _structured_result_payload(output)
    if result is not None and _task_plan_failure_needs_model_feedback(result):
        return _task_plan_failure_result_text(result)
    return _execution_result_text(output, "moveit_execute_task_plan")


def _task_execution_result_text(output: str) -> str:
    result = _structured_result_payload(output)
    if result is not None and _task_plan_failure_needs_model_feedback(result):
        return _task_plan_failure_result_text(result)
    if result is not None:
        simulation = result.get("simulation")
        real_robot = result.get("real_robot")
        if (
            result.get("ok") is True
            and isinstance(simulation, dict)
            and simulation.get("ok") is True
            and isinstance(real_robot, dict)
        ):
            return _task_execution_content_text(result)
    return _execution_result_text(output, "moveit_execute_task")


def _task_plan_failure_result_text(result: dict[str, Any]) -> str:
    task_solution_id = _string_value(result.get("task_solution_id")) or "the approved task"
    failed_step = _string_value(result.get("failed_step")) or "workflow step"
    failed_stage = _string_value(result.get("failed_stage")) or "execution"
    text = f"Execution of {task_solution_id} failed at {failed_step} during {failed_stage}."
    evidence = _task_plan_failure_evidence_text(result)
    if evidence:
        text = f"{text} MoveIt/tool failure: {evidence.rstrip('.')}."
    return f"{text} No new plan was executed. Please approve the next action before I retry or replan."


def _task_plan_failure_evidence_text(result: dict[str, Any]) -> str:
    failed_tool_result = _structured_dict_payload(result.get("failed_tool_result"))
    if failed_tool_result is not None:
        feedback = failed_tool_result.get("feedback")
        if isinstance(feedback, dict):
            for key in ("message", "status", "error", "correction"):
                value = feedback.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        for key in ("error", "message", "correction"):
            value = failed_tool_result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    diagnostic = _structured_dict_payload(result.get("diagnostic"))
    if diagnostic is not None:
        return _structured_feedback_text(diagnostic)
    return _structured_feedback_text(result)


def _structured_dict_payload(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    structured = value.get("structured_content")
    return structured if isinstance(structured, dict) else value


def _string_value(value: Any) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else ""


def _task_execution_content_text(result: dict[str, Any]) -> str:
    real_robot = result.get("real_robot")
    if isinstance(real_robot, dict):
        status = real_robot.get("status")
        if status == "failed":
            return "Execution completed in AR/RViz, but physical execution failed."
        if status == "unavailable":
            return "Execution completed in AR/RViz; physical status unavailable."
    return "Execution complete."


def _physical_task_unavailable_result(
    message: str,
    *,
    error: str | None = None,
    readiness: Any | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ok": False,
        "status": "unavailable",
        "message": message,
    }
    if error:
        result["error"] = error
    if readiness is not None:
        result["readiness"] = readiness
    return result


def _physical_task_failed_result(
    *,
    failed_stage: str | dict[str, Any],
    failed_tool_name: str,
    failed_tool_arguments: dict[str, Any],
    failed_tool_result: Any,
    verified_plan_names: Iterable[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ok": False,
        "status": "failed",
        "message": "Execution completed in AR/RViz, but physical execution failed.",
        "failed_stage": failed_stage,
        "failed_tool_name": failed_tool_name,
        "failed_tool_arguments": dict(failed_tool_arguments),
        "failed_tool_result": failed_tool_result,
        "verified_plan_names": list(verified_plan_names),
    }
    payloads: list[dict[str, Any]] = []
    if isinstance(failed_tool_result, dict):
        structured = failed_tool_result.get("structured_content")
        if isinstance(structured, dict):
            payloads.append(structured)
        payloads.append(failed_tool_result)
    for key in ("error", "correction", "verification"):
        for payload in payloads:
            value = payload.get(key)
            if value:
                result[key] = value
                break
    return result


def _manipulation_planner_timed_out(result: dict[str, Any]) -> bool:
    return any("timed out" in value.lower() for value in _structured_text_values(result))


def _repairable_manipulation_planner_schema_error(result: dict[str, Any]) -> bool:
    text = " ".join(_structured_text_values(result)).lower()
    return (
        "unexpected argument for moveit_plan_manipulation_task: backend" in text
        or "remove backend" in text
    )


def _manipulation_planner_failure_needs_model_feedback(result: dict[str, Any]) -> bool:
    if _repairable_manipulation_planner_schema_error(result):
        return False
    if result.get("retryable") is True:
        return True
    for key in ("suggested_next_tool", "suggested_next_action", "failed_stage", "failure_code"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return True
    return False


def _structured_feedback_text(result: dict[str, Any]) -> str:
    for key in ("correction", "error"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    feedback = result.get("feedback")
    if isinstance(feedback, dict):
        for key in ("correction", "message", "error"):
            value = feedback.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _structured_text_values(result: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("error", "correction", "message", "code", "suggested_next_tool"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    feedback = result.get("feedback")
    if isinstance(feedback, dict):
        for key in ("error", "correction", "message"):
            value = feedback.get(key)
            if isinstance(value, str) and value.strip():
                values.append(value.strip())
    return values


def _task_execution_target_result(output: str, *, tool_name: str) -> dict[str, Any]:
    result = _structured_result_payload(output)
    if result is None:
        return {
            "ok": False,
            "tool": tool_name,
            "status": "failed",
            "error": "Tool returned an unreadable result.",
        }
    ok = result.get("ok") is True
    status = result.get("status")
    if not isinstance(status, str) or not status.strip():
        status = "executed" if ok else "failed"
    summary: dict[str, Any] = {
        "ok": ok,
        "tool": tool_name,
        "status": status,
    }
    for key in (
        "error",
        "correction",
        "retryable",
        "suggested_next_tool",
        "verification",
        "verified_plan_names",
    ):
        value = result.get(key)
        if value is not None:
            summary[key] = value
    return summary


def _execute_geometry_update_dynamic_role(arguments: dict[str, Any]) -> str:
    object_name = arguments.get("object_name")
    role = arguments.get("role")
    reason = arguments.get("reason")
    if not isinstance(object_name, str) or not object_name.strip():
        result = _geometry_update_dynamic_role_failure("object_name is required")
    elif not isinstance(role, dict):
        result = _geometry_update_dynamic_role_failure(
            "role must be one of the structured dynamic role payloads"
        )
    elif not isinstance(reason, str) or not reason.strip():
        result = _geometry_update_dynamic_role_failure("reason is required")
    else:
        result = update_dynamic_role(object_name.strip(), role, reason.strip())
    return json.dumps(result, ensure_ascii=False)


def _geometry_update_dynamic_role_failure(error: str) -> dict[str, object]:
    return {
        "ok": False,
        "error": error,
        "correction": "Use object_name, a structured dynamic role payload, and a human-grounded reason.",
        "retryable": True,
    }


def _structured_result_payload(output: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    structured = payload.get("structured_content")
    if isinstance(structured, dict):
        return structured
    return payload


def _structured_task_policy_error(decision: TaskPolicyDecision) -> dict[str, Any]:
    payload = structured_task_policy_error(decision)
    correction = payload.get("correction")
    if isinstance(correction, str) and any(
        tool_name in correction for tool_name in MODEL_HIDDEN_TASK_PLANNER_TOOL_NAMES
    ):
        payload["correction"] = (
            "Use moveit_plan_manipulation_task for task-level manipulation planning."
        )
    if payload.get("suggested_next_tool") in MODEL_HIDDEN_TASK_PLANNER_TOOL_NAMES:
        payload["suggested_next_tool"] = TASK_LEVEL_REPLAN_TOOL_NAME
    return payload


def _pose_evidence_from_output(
    output: str,
    *,
    object_name: str,
    source: str,
) -> dict[str, object] | None:
    payload = _json_payload(output)
    if not isinstance(payload, dict):
        return None
    containers = _pose_candidate_containers(payload)
    for container in containers:
        pose = _full_pose_from_container(container)
        if pose is not None:
            return {"object_name": object_name, "source": source, "pose": pose}
    return None


def _pose_candidate_containers(payload: dict[str, Any]) -> list[dict[str, Any]]:
    containers: list[dict[str, Any]] = []
    structured = payload.get("structured_content")
    if isinstance(structured, dict):
        containers.append(structured)
        raw = structured.get("raw")
        if isinstance(raw, dict):
            containers.append(raw)
    raw = payload.get("raw")
    if isinstance(raw, dict):
        containers.append(raw)
    containers.append(payload)
    return containers


def _full_pose_from_container(container: dict[str, Any]) -> dict[str, dict[str, float]] | None:
    for key in ("object_pose", "pose"):
        pose = container.get(key)
        full_pose = _full_pose(pose)
        if full_pose is not None:
            return full_pose
    return _full_pose(container)


def _full_pose(value: Any) -> dict[str, dict[str, float]] | None:
    if not isinstance(value, dict):
        return None
    position = value.get("position")
    orientation = value.get("orientation")
    if not isinstance(position, dict) or not isinstance(orientation, dict):
        return None
    xyz = _finite_pose_fields(position, ("x", "y", "z"))
    quat = _finite_pose_fields(orientation, ("x", "y", "z", "w"))
    if xyz is None or quat is None:
        return None
    return {
        "position": {"x": xyz[0], "y": xyz[1], "z": xyz[2]},
        "orientation": {"x": quat[0], "y": quat[1], "z": quat[2], "w": quat[3]},
    }


def _finite_pose_fields(values: dict[Any, Any], keys: tuple[str, ...]) -> list[float] | None:
    result: list[float] = []
    for key in keys:
        value = values.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        number = float(value)
        if not math.isfinite(number):
            return None
        result.append(number)
    return result


def _physical_model_update_failure(
    error: str,
    correction: str,
    *,
    retryable: bool,
) -> dict[str, object]:
    return {
        "ok": False,
        "error": error,
        "correction": correction,
        "retryable": retryable,
    }


def _task_plan_failure_needs_model_feedback(result: dict[str, Any]) -> bool:
    if result.get("ok") is not False:
        return False
    if "failed_tool_result" not in result:
        return False
    suggested_next_tool = result.get("suggested_next_tool")
    return isinstance(suggested_next_tool, str) and bool(suggested_next_tool.strip())


def _queued_job_result_ok(output: str) -> bool:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict) or payload.get("is_error") is True:
        return False
    structured = payload.get("structured_content")
    return isinstance(structured, dict) and structured.get("status") == "queued"


def _queued_job_result_text(name: str, arguments: dict[str, Any], user_text: str | None) -> str:
    if name in PLAN_TOOL_NAMES:
        if should_auto_execute_successful_plan(user_text):
            return "Planning now. I will execute the first successful plan."
        return "Planning now. I will report when a plan is ready."
    if name == "moveit_execute_plan":
        return "Execution queued."
    return "Queued robot action."


def _execution_succeeded(output: str) -> bool:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    structured = payload.get("structured_content")
    result = structured if isinstance(structured, dict) else payload
    verification = result.get("verification")
    return isinstance(verification, dict) and verification.get("result") == "pass"


def _tool_ok(output: str) -> bool:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict) or payload.get("is_error") is True:
        return False
    structured = payload.get("structured_content")
    return isinstance(structured, dict) and structured.get("ok") is True


def _task_plan_execution_steps(
    raw: dict[str, Any],
    *,
    task_kind: str,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    execution_contract = raw.get("execution_contract")
    if execution_contract is not None:
        contract_steps = _task_plan_contract_steps(execution_contract)
        stage_shaped_contract = _task_plan_contract_uses_stages(execution_contract)
        if contract_steps is None:
            return None, _task_plan_error(
                "Task plan execution_contract must contain ordered steps.",
                "Plan a supported pick/place task again, then retry moveit_execute_task_plan.",
                retryable=False,
                suggested_next_tool=None,
            )
        steps: list[dict[str, Any]] = []
        for step in contract_steps:
            if not isinstance(step, dict):
                return None, _task_plan_error(
                    "Task plan workflow contains an unsupported step.",
                    "Plan a supported pick/place task again, then retry moveit_execute_task_plan.",
                    retryable=False,
                    suggested_next_tool=None,
                )
            handler = _task_plan_step_handler(step)
            if handler not in SUPPORTED_TASK_PLAN_HANDLERS:
                return None, _task_plan_error(
                    f"Task plan workflow contains an unsupported step handler: {handler}.",
                    "Plan a supported pick/place task again, then retry moveit_execute_task_plan.",
                    retryable=False,
                    suggested_next_tool=None,
                )
            normalized = dict(step)
            normalized["handler"] = handler
            if stage_shaped_contract:
                _normalize_task_plan_stage_contract_step(normalized, handler=handler)
            missing_contract_field = _task_plan_missing_contract_field(normalized)
            if missing_contract_field is not None:
                return None, _task_plan_error(
                    f"Task plan execution_contract step is missing {missing_contract_field}.",
                    "Replan with a backend task solution that includes source stage and proof metadata.",
                    retryable=False,
                    suggested_next_tool=None,
                )
            if (
                normalized.get("handler") == "motion"
                and normalized.get("required_proof") == "verified_motion_plan"
                and (
                    not isinstance(normalized.get("plan_handle"), str)
                    or not str(normalized.get("plan_handle")).strip()
                )
            ):
                return None, _task_plan_error(
                    "Task plan execution_contract motion step is missing plan_handle.",
                    "Plan the manipulation task again with MoveIt preview evidence.",
                    retryable=False,
                    suggested_next_tool=None,
                )
            steps.append(normalized)
        return steps, None

    if task_kind != "pick":
        return None, _task_plan_error(
            "Task plan execution requires a backend execution_contract.",
            "Plan a supported compound task again, then retry moveit_execute_task_plan.",
            retryable=False,
            suggested_next_tool=None,
        )
    waypoints = raw.get("waypoints")
    workflow_steps = raw.get("workflow_steps")
    if not isinstance(waypoints, list) or not isinstance(workflow_steps, list):
        return None, _task_plan_error(
            "Task plan execution requires task waypoints and workflow steps.",
            "Plan the compound task again, then retry moveit_execute_task_plan with that task_solution_id.",
            suggested_next_tool=TASK_LEVEL_REPLAN_TOOL_NAME,
        )
    steps = []
    for step in workflow_steps:
        if not isinstance(step, dict):
            return None, _task_plan_error(
                "Task plan workflow contains an unsupported step.",
                "Plan the compound task again, then retry moveit_execute_task_plan.",
                suggested_next_tool=TASK_LEVEL_REPLAN_TOOL_NAME,
            )
        handler = _task_plan_legacy_pick_step_handler(step)
        if handler is None:
            return None, _task_plan_error(
                "Task plan workflow contains an unsupported step.",
                "Plan the compound task again, then retry moveit_execute_task_plan.",
                suggested_next_tool=TASK_LEVEL_REPLAN_TOOL_NAME,
            )
        normalized = dict(step)
        normalized["handler"] = handler
        steps.append(normalized)
    steps.append({"handler": "verify_attached_object", "name": "verify_attached_object"})
    return steps, None


def _remember_task_solution_execution_contract_steps(
    context: RobotContextStore,
    *,
    recent: RecentTaskSolution,
    execution_steps: list[dict[str, Any]],
) -> None:
    raw = recent.raw
    if not isinstance(raw, dict):
        return
    normalized_raw = dict(raw)
    execution_contract = normalized_raw.get("execution_contract")
    normalized_steps = [dict(step) for step in execution_steps]
    if isinstance(execution_contract, dict):
        normalized_contract = dict(execution_contract)
        normalized_contract["steps"] = normalized_steps
        normalized_raw["execution_contract"] = normalized_contract
    else:
        normalized_raw["execution_contract"] = normalized_steps
    context.remember_task_solution(
        task_solution_id=recent.task_solution_id,
        task_kind=recent.task_kind,
        object_name=recent.object_name,
        backend=recent.backend,
        scene_snapshot_id=recent.scene_snapshot_id,
        approval_required=recent.approval_required,
        raw=normalized_raw,
    )


def _normalize_recent_task_solution_execution_contract(
    context: RobotContextStore,
    arguments: dict[str, Any],
) -> None:
    task_solution_id = arguments.get("task_solution_id")
    if not isinstance(task_solution_id, str) or not task_solution_id.strip():
        return
    recent = context.recent_task_solution
    if recent is None or recent.task_solution_id != task_solution_id:
        return
    raw = recent.raw
    if not isinstance(raw, dict) or "execution_contract" not in raw:
        return
    execution_steps, execution_steps_error = _task_plan_execution_steps(
        raw,
        task_kind=recent.task_kind,
    )
    if execution_steps_error is not None or execution_steps is None:
        return
    _remember_task_solution_execution_contract_steps(
        context,
        recent=recent,
        execution_steps=execution_steps,
    )


def _task_plan_contract_steps(execution_contract: Any) -> list[Any] | None:
    if isinstance(execution_contract, list):
        return execution_contract
    if not isinstance(execution_contract, dict):
        return None
    steps = execution_contract.get("steps")
    if not isinstance(steps, list):
        steps = execution_contract.get("stages")
    return steps if isinstance(steps, list) else None


def _task_plan_contract_uses_stages(execution_contract: Any) -> bool:
    if not isinstance(execution_contract, dict):
        return False
    if isinstance(execution_contract.get("steps"), list):
        return False
    return isinstance(execution_contract.get("stages"), list)


def _normalize_task_plan_stage_contract_step(step: dict[str, Any], *, handler: str) -> None:
    source_stage = _task_plan_stage_source_stage(step)
    if source_stage is not None:
        step["source_stage"] = source_stage
    required_proof = step.get("required_proof")
    if (
        isinstance(required_proof, str)
        and required_proof.strip() in SUPPORTED_TASK_PLAN_REQUIRED_PROOFS
    ):
        step["required_proof"] = required_proof.strip()
        return
    step["required_proof"] = TASK_PLAN_STAGE_REQUIRED_PROOF_BY_HANDLER[handler]


def _task_plan_stage_source_stage(step: dict[str, Any]) -> str | None:
    source_stage = step.get("source_stage")
    if isinstance(source_stage, str) and source_stage.strip():
        return source_stage.strip()
    for key in ("name", "intent", "tool"):
        value = step.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _task_plan_missing_contract_field(step: dict[str, Any]) -> str | None:
    source_stage = step.get("source_stage")
    if not isinstance(source_stage, str) or not source_stage.strip():
        return "source_stage"
    required_proof = step.get("required_proof")
    if not isinstance(required_proof, str) or not required_proof.strip():
        return "required_proof"
    return None


def _task_plan_legacy_pick_step_handler(step: dict[str, Any]) -> str | None:
    step_name = str(step.get("name") or step.get("tool") or "")
    step_kind = str(step.get("kind") or step.get("type") or "")
    if step_kind == "motion" or isinstance(step.get("waypoint_index"), int):
        return "motion"
    if step_name in {"close", "close_gripper"} or step.get("tool") == "moveit_close_gripper":
        return "close_gripper"
    if step_name in {"attach", "attach_object"} or step.get("tool") == "moveit_attach_object":
        return "attach_object"
    return None


def _task_plan_step_handler(step: dict[str, Any]) -> str:
    raw_handler = step.get("handler")
    if not isinstance(raw_handler, str) or not raw_handler.strip():
        raw_handler = step.get("tool")
    if not isinstance(raw_handler, str) or not raw_handler.strip():
        raw_handler = step.get("kind")
    if not isinstance(raw_handler, str) or not raw_handler.strip():
        raw_handler = step.get("type")
    if not isinstance(raw_handler, str) or not raw_handler.strip():
        raw_handler = step.get("intent")
    if not isinstance(raw_handler, str) or not raw_handler.strip():
        raw_handler = step.get("name")
    handler = str(raw_handler or "").strip().lower()
    handler = handler.removeprefix("moveit_")
    aliases = {
        "close": "close_gripper",
        "open": "open_gripper",
        "verified_close": "close_gripper",
        "verified_open": "open_gripper",
        "attach": "attach_object",
        "release": "release_object",
        "detach": "release_object",
        "detach_object": "release_object",
        "verify_attached": "verify_attached_object",
        "verify_attachment": "verify_attached_object",
        "attachment_proof": "verify_attached_object",
        "verify_released": "verify_released_object",
        "verify_release": "verify_released_object",
        "release_proof": "verify_released_object",
    }
    if handler == "motion":
        return "motion"
    if handler == "gripper":
        step_name = str(step.get("name") or step.get("intent") or "").strip().lower()
        if step_name in {"close", "close_gripper"}:
            return "close_gripper"
        if step_name in {"open", "open_gripper"}:
            return "open_gripper"
        return aliases.get(step_name, step_name)
    if handler == "scene":
        step_name = str(
            step.get("name") or step.get("intent") or step.get("tool") or ""
        ).strip().lower()
        step_name = step_name.removeprefix("moveit_")
        return aliases.get(step_name, step_name)
    if handler == "verify":
        step_name = str(step.get("name") or step.get("intent") or "").strip().lower()
        return aliases.get(step_name, step_name)
    return aliases.get(handler, handler)


def _task_plan_step_waypoint(step: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any] | None:
    waypoint = step.get("waypoint")
    if isinstance(waypoint, dict):
        return dict(waypoint)
    target_pose = step.get("target_pose")
    if isinstance(target_pose, dict):
        return dict(target_pose)
    waypoints = raw.get("waypoints")
    if not isinstance(waypoints, list):
        return None
    return _task_plan_waypoint(step, waypoints)


def _task_plan_step_arguments(step: dict[str, Any]) -> dict[str, Any]:
    arguments = step.get("arguments")
    if not isinstance(arguments, dict):
        arguments = step.get("args")
    return dict(arguments) if isinstance(arguments, dict) else {}


def _task_plan_step_object_name(step: dict[str, Any], fallback: str | None) -> str | None:
    arguments = _task_plan_step_arguments(step)
    object_name = arguments.get("object_name")
    if not isinstance(object_name, str) or not object_name.strip():
        object_name = step.get("object_name")
    if not isinstance(object_name, str) or not object_name.strip():
        object_name = fallback
    return object_name.strip() if isinstance(object_name, str) and object_name.strip() else None


def _task_plan_step_tool(step: dict[str, Any], *, default: str) -> str:
    tool = step.get("tool")
    if not isinstance(tool, str) or not tool.strip():
        tool = step.get("tool_name")
    return tool.strip() if isinstance(tool, str) and tool.strip() else default


def _release_verification_succeeded(output: str, object_name: str) -> bool:
    payload = _json_payload(output)
    if not isinstance(payload, dict):
        return False
    structured = payload.get("structured_content")
    if not isinstance(structured, dict):
        structured = payload
    raw = structured.get("raw")
    raw = raw if isinstance(raw, dict) else {}
    attached_object = raw.get("mcp_attached_object", structured.get("attached_object"))
    if isinstance(attached_object, str) and attached_object.strip() == object_name:
        return False
    scene_state = str(raw.get("planning_scene_state") or structured.get("status") or "")
    scene_state = scene_state.strip().lower()
    if scene_state == "attached":
        return False
    attached_values = [
        structured.get("attached"),
        structured.get("is_attached"),
        raw.get("attached"),
        raw.get("is_attached"),
    ]
    released_scene_states = {
        "released",
        "detached",
        "free",
        "world",
        "not_attached",
        "not attached",
    }
    if attached_object is None or attached_object == "" or attached_object is False:
        if scene_state in released_scene_states:
            return True
        if raw.get("mcp_gripper_holds_object") is False and any(
            value is False for value in attached_values
        ):
            return True
    return False


def _task_plan_waypoint(step: dict[str, Any], waypoints: list[Any]) -> dict[str, Any] | None:
    waypoint_index = step.get("waypoint_index")
    if not isinstance(waypoint_index, int):
        return None
    if waypoint_index < 0 or waypoint_index >= len(waypoints):
        return None
    waypoint = waypoints[waypoint_index]
    return dict(waypoint) if isinstance(waypoint, dict) else None


def _task_plan_step_label(step_name: str) -> str:
    label = "".join(char if char.isalnum() or char == "_" else "_" for char in step_name.strip())
    return label or "motion"


def _task_plan_attempt_name(
    base_plan_name: str,
    *,
    execution_id: str,
    attempt_index: int,
) -> str:
    return f"{base_plan_name}_{execution_id}_try{attempt_index}"


def _task_plan_motion_call(
    step_name: str,
    *,
    attempt_index: int,
    robot_name: str,
    plan_name: str,
    waypoint: dict[str, Any],
    timeout_s: float,
) -> tuple[str, dict[str, Any]]:
    normalized_step = _task_plan_step_label(step_name).lower()
    use_free_motion = normalized_step in {"approach", "connect_to_pre_grasp", "connect_to_place"} or (
        normalized_step in {"pre_grasp", "approach_grasp"} and attempt_index > 1
    )
    if use_free_motion:
        return (
            "moveit_plan_free_motion",
            {
                "robot_name": robot_name,
                "plan_name": plan_name,
                "target_pose": waypoint,
                "timeout_s": timeout_s,
            },
        )
    return (
        "moveit_plan_cartesian_motion",
        {
            "robot_name": robot_name,
            "plan_name": plan_name,
            "waypoints": [waypoint],
            "timeout_s": timeout_s,
        },
    )


def _task_plan_pose_observation_arguments(robot_name: str) -> dict[str, Any]:
    return {"robot_name": robot_name, "timeout_s": TASK_PLAN_POSE_OBSERVATION_TIMEOUT_S}


def _task_plan_error(
    error: str,
    correction: str,
    *,
    retryable: bool = True,
    suggested_next_tool: str | None = "moveit_get_current_pose",
) -> str:
    payload: dict[str, Any] = {
        "ok": False,
        "error": error,
        "correction": correction,
        "retryable": retryable,
    }
    if suggested_next_tool is not None:
        payload["suggested_next_tool"] = suggested_next_tool
    return json.dumps(payload, ensure_ascii=False)


def _task_plan_stage_error(
    stage: str,
    step_name: str,
    output: str,
    *,
    task_solution_id: str,
    failed_tool_name: str | None = None,
    failed_tool_arguments: dict[str, Any] | None = None,
    recovery: dict[str, Any] | None = None,
    diagnostic: Any | None = None,
) -> str:
    payload: dict[str, Any] = {
        "ok": False,
        "error": f"Task plan {stage} failed at {step_name or 'workflow step'}.",
        "correction": "Inspect the failed tool result, then replan before retrying task execution.",
        "retryable": True,
        "task_solution_id": task_solution_id,
        "failed_stage": stage,
        "failed_step": step_name,
        "failed_tool_result": _json_payload(output),
        "suggested_next_tool": "moveit_explain_motion_failure",
    }
    if failed_tool_name is not None:
        payload["failed_tool_name"] = failed_tool_name
    if failed_tool_arguments is not None:
        payload["failed_tool_arguments"] = failed_tool_arguments
    if recovery is not None:
        payload["recovery"] = recovery
    if diagnostic is not None:
        payload["diagnostic"] = diagnostic
    return json.dumps(payload, ensure_ascii=False)


def _json_payload(output: str) -> Any:
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return output


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


def _tool_choice_for_state(state: RobotAgentState) -> str:
    if (
        state["needs_action_tool"]
        and not state["action_tool_ran"]
        and state["missing_action_repairs"] > 0
    ):
        return "required"
    return "auto"


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
