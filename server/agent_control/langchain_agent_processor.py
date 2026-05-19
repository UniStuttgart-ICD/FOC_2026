from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any

from loguru import logger

from agent_control.langgraph_robot_agent import LangGraphRobotAgent
from agent_control.robot_job_submission import RobotJobSubmitter
from agent_control.status_replies import (
    action_complete_reply,
    execution_complete_reply,
    plan_ready_reply,
)
from embodiment.animations import EmbodimentAnimationController
from process_trace import NoopProcessTracer, ProcessTracer
from robot_control.context import RobotContextStore
from robot_control.job_board import RobotJob, RobotJobBoard, RobotJobEvent, RobotJobEventType
from robot_control.job_worker import RobotJobWorker
from robot_control.mcp_bridge import RobotMCPBridge
from robot_control.shared_geometry import GeometryWorldContextStore
from robot_control.verified_execution_client import (
    HttpVerifiedExecutionClient,
    VerifiedExecutionClient,
)
from user_sensing.context import UserSensingContextStore
from user_sensing.mcp_bridge import UserSensingMCPBridge
from voice_runtime.agent_turn import AgentTurnInput

ProcessTracerLike = ProcessTracer | NoopProcessTracer
PLAN_JOB_TOOLS = frozenset({"moveit_plan_free_motion", "moveit_plan_cartesian_motion"})


class LangChainAgentProcessor:
    """Runs agent turns through an API-key-backed LangChain chat model."""

    def __init__(
        self,
        mcp_server_url: str,
        *,
        chat_model: Any,
        model_label: str,
        tool_bridge: Any | None = None,
        robot_job_board: RobotJobBoard | None = None,
        robot_job_worker: Any | None = None,
        mcp_vizor_url: str | None = None,
        user_sensing_bridge: Any | None = None,
        user_sensing_max_age_s: float = 2.0,
        verified_execution_url: str | None = None,
        verified_execution_client: VerifiedExecutionClient | None = None,
        embodiment_controller: EmbodimentAnimationController | None = None,
        tracer: ProcessTracerLike | None = None,
    ):
        self._mcp_server_url = mcp_server_url
        self._mcp_vizor_url = mcp_vizor_url
        self._chat_model = chat_model
        self._model_label = model_label
        self._tracer = tracer or NoopProcessTracer()
        self._tool_bridge = tool_bridge
        self._robot_job_board = robot_job_board or RobotJobBoard(tracer=self._tracer)
        self._robot_job_submitter = RobotJobSubmitter(self._robot_job_board)
        self._robot_job_worker = robot_job_worker
        self._user_sensing_bridge = user_sensing_bridge
        self._user_sensing_context = UserSensingContextStore()
        self._geometry_world_context = GeometryWorldContextStore()
        self._user_sensing_max_age_s = user_sensing_max_age_s
        self._verified_execution_client = (
            verified_execution_client
            if verified_execution_client is not None
            else _verified_execution_client(verified_execution_url)
        )
        self._embodiment_controller = embodiment_controller
        self._owns_tool_bridge = tool_bridge is None
        self._owns_user_sensing_bridge = user_sensing_bridge is None
        self._connected = False
        self._model_logged = False
        self._robot_context = RobotContextStore()
        self._recorded_job_context_sequences: set[int] = set()
        self._thread_id = f"langchain-agent-{id(self)}"
        self._graph_agent: LangGraphRobotAgent | None = None
        self._graph_chat_model: Any | None = None
        self._graph_tool_bridge: Any | None = None
        self._graph_user_sensing_bridge: Any | None = None

    @property
    def robot_job_board(self) -> RobotJobBoard:
        return self._robot_job_board

    async def connect(self) -> None:
        await self._ensure_connected()

    async def disconnect(self) -> None:
        if self._robot_job_worker is not None:
            stop = getattr(self._robot_job_worker, "stop", None)
            if stop is not None:
                await stop()
        if self._tool_bridge is not None and (self._connected or not self._owns_tool_bridge):
            await self._tool_bridge.disconnect()
        if self._user_sensing_bridge is not None and (
            self._connected or not self._owns_user_sensing_bridge
        ):
            disconnect = getattr(self._user_sensing_bridge, "disconnect", None)
            if disconnect is not None:
                await disconnect()
        self._tool_bridge = None
        self._user_sensing_bridge = None
        self._graph_agent = None
        self._graph_chat_model = None
        self._graph_tool_bridge = None
        self._robot_job_worker = None
        self._graph_user_sensing_bridge = None
        self._connected = False
        logger.info("LangChain API-key agent disconnected")

    async def notifications(self):
        sequence = 0
        while True:
            events = self._robot_job_board.events_since(sequence)
            for event in events:
                sequence = max(sequence, event.sequence)
                if event.event_type is RobotJobEventType.COMPLETED:
                    self._record_completed_job_in_context(event)
                    yield _completed_job_notification(event)
                elif event.event_type is RobotJobEventType.FAILED:
                    yield await self._failed_job_notification(event)
            await asyncio.sleep(0.05)

    async def run_turn(self, turn: AgentTurnInput):
        async with self._tracer.span(
            "agent.backend_turn",
            "agent_control",
            attributes={
                "model_label": self._model_label,
                "message_count": len(turn.messages),
            },
        ):
            logger.info("User said: {}", turn.user_text)
            try:
                await self._ensure_connected()
            except Exception as exc:
                logger.error("LangChain agent connection error: {}", exc)
                response = "I can't reach the robot control server right now."
            else:
                tool_bridge = self._tool_bridge
                if tool_bridge is None:
                    response = "I can't reach the robot control server right now."
                else:
                    if not self._model_logged:
                        logger.info("LangChain model: {}", self._model_label)
                        self._model_logged = True

                    self._record_completed_job_results_in_context()
                    graph = self._graph_agent_for(self._chat_model, tool_bridge)
                    try:
                        response = await graph.run_turn(turn)
                    except Exception as exc:
                        logger.error("LangChain agent error: {}", exc)
                        response = "I encountered an error. Please try again."
        yield response

    async def _ensure_connected(self) -> None:
        if self._connected:
            return
        if self._tool_bridge is None:
            self._tool_bridge = _robot_mcp_bridge(self._mcp_server_url, tracer=self._tracer)
        await self._tool_bridge.connect()
        if self._robot_job_worker is None:
            self._robot_job_worker = RobotJobWorker(
                board=self._robot_job_board,
                tool_bridge=self._tool_bridge,
                verified_execution_client=self._verified_execution_client,
            )
        start = getattr(self._robot_job_worker, "start", None)
        if start is not None:
            await start()
        if self._user_sensing_bridge is None and self._mcp_vizor_url:
            self._user_sensing_bridge = _user_sensing_mcp_bridge(
                self._mcp_vizor_url,
                tracer=self._tracer,
            )
        if self._user_sensing_bridge is not None:
            connect = getattr(self._user_sensing_bridge, "connect", None)
            if connect is not None:
                try:
                    await connect()
                    logger.info(
                        "Vizor user sensing MCP connected url={} max_age_s={}",
                        self._mcp_vizor_url or "injected",
                        self._user_sensing_max_age_s,
                    )
                except Exception as exc:
                    logger.warning("Vizor user sensing MCP connection failed: {}", exc)
                    if self._owns_user_sensing_bridge:
                        self._user_sensing_bridge = None
        self._connected = True
        logger.info("LangChain API-key agent connected")

    def _graph_agent_for(self, chat_model: Any, tool_bridge: Any) -> LangGraphRobotAgent:
        if (
            self._graph_agent is None
            or self._graph_chat_model is not chat_model
            or self._graph_tool_bridge is not tool_bridge
            or self._graph_user_sensing_bridge is not self._user_sensing_bridge
        ):
            kwargs: dict[str, Any] = {
                "model": chat_model,
                "tool_bridge": tool_bridge,
                "robot_context": self._robot_context,
                "geometry_world_context": self._geometry_world_context,
                "robot_job_blackboard_summary": self._robot_job_blackboard_summary,
                "thread_id": self._thread_id,
                "job_submitter": self._robot_job_submitter,
                "verified_execution_client": self._verified_execution_client,
                "embodiment_controller": self._embodiment_controller,
                "tracer": self._tracer,
            }
            if self._user_sensing_bridge is not None:
                kwargs.update(
                    {
                        "user_sensing_bridge": self._user_sensing_bridge,
                        "user_sensing_context": self._user_sensing_context,
                        "user_sensing_max_age_s": self._user_sensing_max_age_s,
                    }
                )
            self._graph_agent = LangGraphRobotAgent(**kwargs)
            self._graph_chat_model = chat_model
            self._graph_tool_bridge = tool_bridge
            self._graph_user_sensing_bridge = self._user_sensing_bridge
        return self._graph_agent

    async def _failed_job_notification(self, event: RobotJobEvent) -> str:
        error = str(event.payload.get("error") or "unknown error")
        job = self._robot_job_board.get(event.job_id)
        if job is None or self._tool_bridge is None:
            return f"The robot action hit a snag: {error}"

        prompt = _failed_job_recovery_prompt(job, error)
        turn = AgentTurnInput(
            user_text=prompt,
            messages=[{"role": "user", "content": prompt}],
            allow_pending_plan_execution=False,
        )
        graph = self._graph_agent_for(self._chat_model, self._tool_bridge)
        try:
            return await graph.run_turn(turn)
        except Exception as exc:
            logger.error("Robot job recovery turn failed: {}", exc)
            return f"The robot action hit a snag: {error}"

    def _record_completed_job_in_context(self, event: RobotJobEvent) -> None:
        if event.sequence in self._recorded_job_context_sequences:
            return
        job = self._robot_job_board.get(event.job_id)
        if job is None or job.result is None:
            return
        self._robot_context.update_from_tool_result(job.tool_name, job.result)
        self._recorded_job_context_sequences.add(event.sequence)

    def _record_completed_job_results_in_context(self) -> None:
        for event in self._robot_job_board.events_since(0):
            if event.event_type is RobotJobEventType.COMPLETED:
                self._record_completed_job_in_context(event)

    def _robot_job_blackboard_summary(self) -> str | None:
        return self._robot_job_board.render_instruction_block(
            context_recorded_sequences=self._recorded_job_context_sequences,
        )


def _robot_mcp_bridge(mcp_server_url: str, *, tracer: ProcessTracerLike) -> Any:
    if _accepts_tracer_keyword(RobotMCPBridge):
        return RobotMCPBridge(mcp_server_url, tracer=tracer)
    return RobotMCPBridge(mcp_server_url)


def _user_sensing_mcp_bridge(mcp_server_url: str, *, tracer: ProcessTracerLike) -> Any:
    if _accepts_tracer_keyword(UserSensingMCPBridge):
        return UserSensingMCPBridge(mcp_server_url, tracer=tracer)
    return UserSensingMCPBridge(mcp_server_url)


def _verified_execution_client(base_url: str | None) -> VerifiedExecutionClient | None:
    if base_url is None or not base_url.strip():
        return None
    return HttpVerifiedExecutionClient(base_url)


def _completed_job_notification(event: RobotJobEvent) -> str:
    seed = event.payload.get("result") or event.job_id or event.tool_name
    if event.tool_name in PLAN_JOB_TOOLS:
        return plan_ready_reply(seed)
    if event.tool_name == "moveit_execute_plan":
        return execution_complete_reply(seed)
    return action_complete_reply(seed)


def _accepts_tracer_keyword(callable_obj: Any) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == "tracer" or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _failed_job_recovery_prompt(job: RobotJob, error: str) -> str:
    arguments = _format_jsonish(job.arguments)
    tool_result = _format_jsonish(job.result)
    original_user_request = job.user_text or "unknown"
    return (
        "A queued MoveIt robot action failed after the robot worker handled it.\n"
        "Any structured tool result below is the authority on motion feasibility and execution proof.\n"
        f"Original user request: {original_user_request}\n"
        f"Tool: {job.tool_name}\n"
        f"Tool arguments:\n{arguments}\n"
        f"Error summary: {error}\n"
        f"Tool result:\n{tool_result}\n\n"
        "Use this failure data to respond as the robot agent. If a nearby alternate motion "
        "can still satisfy the user's intent by improvising, call an available MoveIt planning "
        "or diagnostic tool now. Ask the human/operator for guidance when recovery needs "
        "judgment, physical inspection, approval, or a changed task. "
        "Do not claim completion until a future tool result verifies execution."
    )


def _format_jsonish(value: Any) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return value
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
