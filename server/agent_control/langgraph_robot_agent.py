"""LangGraph orchestration for robot agent turns."""

from __future__ import annotations

import asyncio
import inspect
import json
import operator
import uuid
from typing import Annotated, Any, Literal, TypedDict

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
    RobotCallValidationError,
    ensure_task_solution_execution_allowed,
    executable_plan_name,
    structured_robot_call_error,
    validate_robot_tool_call,
)
from robot_control.context import RobotContextStore
from robot_control.execution_intent import (
    explicit_execute_requested as _explicit_execute_requested,
)
from robot_control.execution_intent import (
    looks_like_robot_action_request as _looks_like_robot_action_request,
)
from robot_control.execution_intent import (
    should_auto_execute_successful_plan,
)
from robot_control.mcp_bridge import RobotMCPError
from robot_control.task_policy import (
    DEFAULT_EXECUTABLE_PLAN_MAX_AGE_S,
    TaskPolicyDecision,
    structured_task_policy_error,
    validate_task_step,
)
from robot_control.verified_execution_client import VerifiedExecutionClient
from user_sensing.context import UserSensingContextStore
from voice_runtime.agent_turn import AgentTurnInput
from voice_runtime.timing import elapsed_ms_since, monotonic_s

MAX_AGENT_TOOL_TURNS = 6
TASK_PLAN_STAGE_MAX_ATTEMPTS = 2
TASK_PLAN_OBSERVATION_MAX_ATTEMPTS = 3
TASK_PLAN_OBSERVATION_RETRY_DELAY_S = 0.2
VIZOR_ROBOT_NAME = "UR10"
PLAN_TOOL_NAMES = {
    "moveit_plan_free_motion",
    "moveit_plan_cartesian_motion",
    "moveit_plan_pick",
    "moveit_plan_place",
}
ACTION_TOOL_NAMES = {
    "moveit_plan_free_motion",
    "moveit_plan_cartesian_motion",
    "moveit_plan_pick",
    "moveit_plan_place",
    "moveit_execute_plan",
    "moveit_execute_task_solution",
    "moveit_execute_task_plan",
    "moveit_verify_attached_object",
    "moveit_open_gripper",
    "moveit_close_gripper",
    "moveit_attach_object",
}
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
        user_sensing_bridge: Any | None = None,
        user_sensing_context: UserSensingContextStore | None = None,
        user_sensing_max_age_s: float = 2.0,
        thread_id: str | None = None,
        job_submitter: RobotJobSubmitter | None = None,
        verified_execution_client: VerifiedExecutionClient | None = None,
        tracer: ProcessTracerLike | None = None,
    ) -> None:
        self._model = model
        self._tool_bridge = tool_bridge
        self._robot_context = robot_context
        self._user_sensing_bridge = user_sensing_bridge
        self._user_sensing_context = user_sensing_context or UserSensingContextStore()
        self._user_sensing_max_age_s = user_sensing_max_age_s
        self._thread_id = thread_id or f"robot-agent-{uuid.uuid4()}"
        self._job_submitter = job_submitter
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
        tools = self._tool_bridge.function_tools()
        await self._refresh_user_sensing_context()
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
            "timeout_s": 10.0,
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
            if name in OBSERVE_TOOL_NAMES:
                output, observed_this_turn = await self._execute_observation_tool(name, dict(args))
            elif name == "moveit_execute_task_plan":
                output = await self._execute_verified_task_plan_tool(
                    dict(args),
                    user_text=state["user_text"],
                    allow_execution=state["allow_pending_plan_execution"],
                )
                action_tool_ran = True
                observed_this_turn = False
            elif name == "moveit_execute_plan" and self._verified_execution_client is not None:
                output = await self._execute_verified_plan_tool(
                    dict(args),
                    user_text=state["user_text"],
                    allow_execution=state["allow_pending_plan_execution"],
                )
                action_tool_ran = True
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
        parts = [SYSTEM_PROMPT, self._robot_context.render_instruction_block()]
        if self._user_sensing_bridge is not None:
            parts.append(self._user_sensing_context.render_instruction_block())
        return "\n\n".join(parts)

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
            return json.dumps(structured_task_policy_error(decision), ensure_ascii=False)
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
            return json.dumps(structured_task_policy_error(decision), ensure_ascii=False)
        if name == "moveit_execute_task_solution":
            self._record_task_solution_approval_if_explicit(
                arguments,
                user_text=user_text,
                allow_execution=allow_execution,
            )
            try:
                ensure_task_solution_execution_allowed(self._robot_context, arguments)
            except RobotCallValidationError as exc:
                return json.dumps(structured_robot_call_error(exc), ensure_ascii=False)
            verified_error = self._verified_real_robot_task_solution_error(arguments)
            if verified_error is not None:
                return json.dumps(verified_error, ensure_ascii=False)
        output = await self._tool_bridge.call_tool(name, arguments)
        self._robot_context.update_from_tool_result(name, output)
        self._tracer.event(
            "robot.context_update",
            "robot_control",
            attributes={"tool.name": name},
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
        self, arguments: dict[str, Any]
    ) -> dict[str, Any] | None:
        if self._verified_execution_client is None:
            return None
        task_solution_id = arguments.get("task_solution_id")
        recent = self._robot_context.recent_task_solution
        if (
            not isinstance(task_solution_id, str)
            or recent is None
            or recent.task_solution_id != task_solution_id
            or recent.backend != "emulated"
        ):
            return None
        exc = RobotCallValidationError(
            "Task solution execution is not wired to Verified Real Robot Execution",
            correction=(
                "Do not claim physical task execution. The current task solution has "
                "emulated arm-motion stages; use a verified cached trajectory plan or "
                "add verified task-solution execution before retrying this task."
            ),
        )
        return structured_robot_call_error(
            exc,
            retryable=False,
            suggested_next_tool=None,
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
        self._record_task_solution_approval_if_explicit(
            arguments,
            user_text=user_text,
            allow_execution=allow_execution,
        )
        try:
            ensure_task_solution_execution_allowed(self._robot_context, arguments)
        except RobotCallValidationError as exc:
            return json.dumps(structured_robot_call_error(exc), ensure_ascii=False)

        task_solution_id = str(arguments["task_solution_id"]).strip()
        robot_name = str(arguments.get("robot_name") or VIZOR_ROBOT_NAME)
        timeout_s = float(arguments.get("timeout_s") or 10.0)
        recent = self._robot_context.recent_task_solution
        if recent is None or recent.task_solution_id != task_solution_id:
            return _task_plan_error(
                "Task plan execution requires the exact recent task_solution_id.",
                "Plan the pick task again, then retry moveit_execute_task_plan with that task_solution_id.",
                suggested_next_tool="moveit_plan_pick_task",
            )
        if recent.task_kind != "pick":
            return _task_plan_error(
                "Task plan execution currently supports pick task solutions only.",
                "Use a pick task solution, or execute place workflows through supported verified plan steps.",
                retryable=False,
                suggested_next_tool=None,
            )
        raw = recent.raw
        if raw is None:
            return _task_plan_error(
                "Task plan execution requires the recent raw task solution.",
                "Plan the pick task again, then retry moveit_execute_task_plan with that task_solution_id.",
                suggested_next_tool="moveit_plan_pick_task",
            )
        waypoints = raw.get("waypoints")
        workflow_steps = raw.get("workflow_steps")
        if not isinstance(waypoints, list) or not isinstance(workflow_steps, list):
            return _task_plan_error(
                "Task plan execution requires task waypoints and workflow steps.",
                "Plan the pick task again, then retry moveit_execute_task_plan with that task_solution_id.",
                suggested_next_tool="moveit_plan_pick_task",
            )

        execution_id = uuid.uuid4().hex[:8]
        verified_plan_names: list[str] = []
        for step in workflow_steps:
            if not isinstance(step, dict):
                return _task_plan_error(
                    "Task plan workflow contains an unsupported step.",
                    "Plan the pick task again, then retry moveit_execute_task_plan.",
                    suggested_next_tool="moveit_plan_pick_task",
                )
            step_name = str(step.get("name") or step.get("tool") or "")
            step_kind = str(step.get("kind") or step.get("type") or "")
            if step_kind == "motion" or isinstance(step.get("waypoint_index"), int):
                waypoint = _task_plan_waypoint(step, waypoints)
                if waypoint is None:
                    return _task_plan_error(
                        "Task plan motion step references a missing waypoint.",
                        "Plan the pick task again, then retry moveit_execute_task_plan.",
                        suggested_next_tool="moveit_plan_pick_task",
                    )
                pose_output = await self._execute_task_plan_pose_observation(
                    robot_name=robot_name,
                    user_text=user_text,
                    allow_execution=allow_execution,
                )
                if not _tool_ok(pose_output):
                    return _task_plan_stage_error("observe_current_pose", step_name, pose_output)
                base_plan_name = f"{task_solution_id}_{_task_plan_step_label(step_name)}"
                stage_succeeded = False
                last_error = ""
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
                        user_text=user_text,
                        allow_execution=allow_execution,
                    )
                    if not _tool_ok(plan_output):
                        last_error = _task_plan_stage_error("planning", step_name, plan_output)
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
                        last_error = _task_plan_stage_error(
                            "verified_execution", step_name, execute_output
                        )
                        continue
                    verified_plan_names.append(executable_name)
                    stage_succeeded = True
                    break
                if not stage_succeeded:
                    return last_error
                continue
            if step_name in {"close", "close_gripper"} or step.get("tool") == "moveit_close_gripper":
                close_output = await self._execute_tool(
                    "moveit_close_gripper",
                    {"robot_name": robot_name, "timeout_s": timeout_s},
                    user_text=user_text,
                    allow_execution=allow_execution,
                )
                if not _tool_ok(close_output):
                    return _task_plan_stage_error("close_gripper", step_name, close_output)
                continue
            if step_name in {"attach", "attach_object"} or step.get("tool") == "moveit_attach_object":
                attach_output = await self._execute_tool(
                    "moveit_attach_object",
                    {"robot_name": robot_name, "object_name": recent.object_name},
                    user_text=user_text,
                    allow_execution=allow_execution,
                )
                if not _tool_ok(attach_output):
                    return _task_plan_stage_error("attach_object", step_name, attach_output)
                continue
            return _task_plan_error(
                "Task plan workflow contains an unsupported step.",
                "Plan the pick task again, then retry moveit_execute_task_plan.",
                suggested_next_tool="moveit_plan_pick_task",
            )

        verify_output = await self._execute_tool(
            "moveit_verify_attached_object",
            {"robot_name": robot_name, "object_name": recent.object_name, "timeout_s": timeout_s},
            user_text=user_text,
            allow_execution=allow_execution,
        )
        if not _tool_ok(verify_output):
            return _task_plan_stage_error("verify_attached_object", "verify", verify_output)
        return json.dumps(
            {
                "content": ["Verified task plan execution completed."],
                "structured_content": {
                    "ok": True,
                    "tool": "moveit_execute_task_plan",
                    "task_solution_id": task_solution_id,
                    "object_name": recent.object_name,
                    "verified_plan_names": verified_plan_names,
                    "verification": {"result": "pass"},
                },
                "is_error": False,
            },
            ensure_ascii=False,
        )

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
                {"robot_name": robot_name},
                user_text=user_text,
                allow_execution=allow_execution,
            )
            if _tool_ok(output):
                return output
            if attempt_index < TASK_PLAN_OBSERVATION_MAX_ATTEMPTS:
                await asyncio.sleep(TASK_PLAN_OBSERVATION_RETRY_DELAY_S)
        return output

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
        timeout_s = float(arguments.get("timeout_s") or 10.0)
        output = await self._verified_execution_client.execute_plan(
            robot_name=robot_name,
            plan_name=plan_name,
            timeout_s=timeout_s,
        )
        pending = self._robot_context.pending_executable_plan(
            plan_name,
            max_age_s=DEFAULT_EXECUTABLE_PLAN_MAX_AGE_S,
        )
        self._robot_context.update_from_tool_result("moveit_execute_plan", output)
        if _execution_succeeded(output):
            await self._execute_after_success_action(
                pending.after_success_tool if pending is not None else None,
                pending.after_success_arguments if pending is not None else None,
                user_text=user_text,
                allow_execution=allow_execution,
            )
        return output

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
        if not name.startswith("moveit_plan_"):
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


def _should_execute_latest_pending_plan(text: str, robot_context: RobotContextStore) -> bool:
    if not _explicit_execute_requested(text):
        return False
    pending = robot_context.latest_pending_executable_plan(
        max_age_s=DEFAULT_EXECUTABLE_PLAN_MAX_AGE_S
    )
    return pending is not None and not _is_task_stage_attempt_plan_name(pending.plan_name)


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
        return f"Queued execution for plan {plan_name}."
    verification = result.get("verification")
    if isinstance(verification, dict) and verification.get("result") == "pass":
        return f"Executed plan {plan_name}."
    return NO_TEXT_RESPONSE


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
        plan_name = arguments.get("plan_name")
        if isinstance(plan_name, str) and plan_name:
            return f"Queued execution for plan {plan_name}."
        return "Queued execution."
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
    use_free_motion = normalized_step in {"approach", "connect_to_pre_grasp"} or (
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


def _task_plan_stage_error(stage: str, step_name: str, output: str) -> str:
    return json.dumps(
        {
            "ok": False,
            "error": f"Task plan {stage} failed at {step_name or 'workflow step'}.",
            "correction": "Inspect the failed tool result, then replan before retrying task execution.",
            "retryable": True,
            "failed_tool_result": _json_payload(output),
            "suggested_next_tool": "moveit_explain_motion_failure",
        },
        ensure_ascii=False,
    )


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
