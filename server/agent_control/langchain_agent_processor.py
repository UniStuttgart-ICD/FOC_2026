from __future__ import annotations

import asyncio
import inspect
from typing import Any

from loguru import logger

from agent_control.langgraph_robot_agent import LangGraphRobotAgent
from agent_control.robot_job_submission import RobotJobSubmitter
from process_trace import NoopProcessTracer, ProcessTracer
from robot_control.context import RobotContextStore
from robot_control.job_board import RobotJobBoard, RobotJobEventType
from robot_control.job_worker import RobotJobWorker
from robot_control.mcp_bridge import RobotMCPBridge
from voice_runtime.agent_turn import AgentTurnInput

ProcessTracerLike = ProcessTracer | NoopProcessTracer


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
        tracer: ProcessTracerLike | None = None,
    ):
        self._mcp_server_url = mcp_server_url
        self._chat_model = chat_model
        self._model_label = model_label
        self._tool_bridge = tool_bridge
        self._robot_job_board = robot_job_board or RobotJobBoard()
        self._robot_job_submitter = RobotJobSubmitter(self._robot_job_board)
        self._robot_job_worker = robot_job_worker
        self._tracer = tracer or NoopProcessTracer()
        self._owns_tool_bridge = tool_bridge is None
        self._connected = False
        self._model_logged = False
        self._robot_context = RobotContextStore()
        self._thread_id = f"langchain-agent-{id(self)}"
        self._graph_agent: LangGraphRobotAgent | None = None
        self._graph_chat_model: Any | None = None
        self._graph_tool_bridge: Any | None = None

    async def connect(self) -> None:
        await self._ensure_connected()

    async def disconnect(self) -> None:
        if self._robot_job_worker is not None:
            stop = getattr(self._robot_job_worker, "stop", None)
            if stop is not None:
                await stop()
        if self._tool_bridge is not None and (self._connected or not self._owns_tool_bridge):
            await self._tool_bridge.disconnect()
        self._tool_bridge = None
        self._graph_agent = None
        self._graph_chat_model = None
        self._graph_tool_bridge = None
        self._robot_job_worker = None
        self._connected = False
        logger.info("LangChain API-key agent disconnected")

    async def notifications(self):
        sequence = 0
        while True:
            events = self._robot_job_board.events_since(sequence)
            for event in events:
                sequence = max(sequence, event.sequence)
                if event.event_type is RobotJobEventType.COMPLETED:
                    yield "Job complete."
                elif event.event_type is RobotJobEventType.FAILED:
                    error = event.payload.get("error", "unknown error")
                    yield f"The robot action hit a snag: {error}"
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
            )
        start = getattr(self._robot_job_worker, "start", None)
        if start is not None:
            await start()
        self._connected = True
        logger.info("LangChain API-key agent connected")

    def _graph_agent_for(self, chat_model: Any, tool_bridge: Any) -> LangGraphRobotAgent:
        if (
            self._graph_agent is None
            or self._graph_chat_model is not chat_model
            or self._graph_tool_bridge is not tool_bridge
        ):
            self._graph_agent = LangGraphRobotAgent(
                model=chat_model,
                tool_bridge=tool_bridge,
                robot_context=self._robot_context,
                thread_id=self._thread_id,
                job_submitter=self._robot_job_submitter,
                tracer=self._tracer,
            )
            self._graph_chat_model = chat_model
            self._graph_tool_bridge = tool_bridge
        return self._graph_agent


def _robot_mcp_bridge(mcp_server_url: str, *, tracer: ProcessTracerLike) -> Any:
    if _accepts_tracer_keyword(RobotMCPBridge):
        return RobotMCPBridge(mcp_server_url, tracer=tracer)
    return RobotMCPBridge(mcp_server_url)


def _accepts_tracer_keyword(callable_obj: Any) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == "tracer" or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
