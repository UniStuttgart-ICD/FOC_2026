from __future__ import annotations

from typing import Any

from loguru import logger

from langgraph_robot_agent import LangGraphRobotAgent
from robot_control.context import RobotContextStore
from robot_control.mcp_bridge import RobotMCPBridge
from voice_runtime.agent_turn import AgentTurnInput


class LangChainAgentProcessor:
    """Runs agent turns through an API-key-backed LangChain chat model."""

    def __init__(
        self,
        mcp_server_url: str,
        *,
        chat_model: Any,
        model_label: str,
        tool_bridge: Any | None = None,
    ):
        self._mcp_server_url = mcp_server_url
        self._chat_model = chat_model
        self._model_label = model_label
        self._tool_bridge = tool_bridge
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
        if self._tool_bridge is not None and (self._connected or not self._owns_tool_bridge):
            await self._tool_bridge.disconnect()
        self._tool_bridge = None
        self._graph_agent = None
        self._graph_chat_model = None
        self._graph_tool_bridge = None
        self._connected = False
        logger.info("LangChain API-key agent disconnected")

    async def run_turn(self, turn: AgentTurnInput):
        logger.info("User said: {}", turn.user_text)
        try:
            await self._ensure_connected()
        except Exception as exc:
            logger.error("LangChain agent connection error: {}", exc)
            yield "I can't reach the robot control server right now."
            return

        tool_bridge = self._tool_bridge
        if tool_bridge is None:
            yield "I can't reach the robot control server right now."
            return

        if not self._model_logged:
            logger.info("LangChain model: {}", self._model_label)
            self._model_logged = True

        graph = self._graph_agent_for(self._chat_model, tool_bridge)
        try:
            yield await graph.run_turn(turn)
        except Exception as exc:
            logger.error("LangChain agent error: {}", exc)
            yield "I encountered an error. Please try again."

    async def _ensure_connected(self) -> None:
        if self._connected:
            return
        if self._tool_bridge is None:
            self._tool_bridge = RobotMCPBridge(self._mcp_server_url)
        await self._tool_bridge.connect()
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
            )
            self._graph_chat_model = chat_model
            self._graph_tool_bridge = tool_bridge
        return self._graph_agent
